import importlib
import inspect
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import MethodType

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub
UATVR = importlib.import_module("modules.modeling_mulit").UATVR
MultiPositiveCrossEn = importlib.import_module(
    "modules.until_module"
).MultiPositiveCrossEn
PIENet = importlib.import_module("prob_models.pie_model").PIENet
UncertaintyModuleText = importlib.import_module("prob_models.uncertainty_module").UncertaintyModuleText


def test_forward_accepts_explicit_hard_negative_kwargs():
    signature = inspect.signature(UATVR.forward)

    for name in [
        "video_group_id",
        "sample_index",
        "hard_video",
        "hard_video_mask",
        "hard_valid",
    ]:
        assert name in signature.parameters


def test_msrvtt_training_requires_video_group_id():
    with pytest.raises(ValueError, match="MSRVTT training requires video_group_id"):
        UATVR.resolve_video_group_ids(
            None,
            local_batch=2,
            device=torch.device("cpu"),
            task_config=Namespace(datatype="msrvtt"),
        )


def test_non_msrvtt_fallback_group_ids_are_disjoint_across_ranks():
    rank0 = UATVR.resolve_video_group_ids(
        None,
        local_batch=2,
        device=torch.device("cpu"),
        task_config=Namespace(datatype="msvd", rank=0),
    )
    rank1 = UATVR.resolve_video_group_ids(
        None,
        local_batch=2,
        device=torch.device("cpu"),
        task_config=Namespace(datatype="msvd", rank=1),
    )

    assert set(rank0.tolist()).isdisjoint(rank1.tolist())


def test_provided_video_group_ids_are_long_and_match_local_batch():
    resolved = UATVR.resolve_video_group_ids(
        torch.tensor([4, 5], dtype=torch.int32),
        local_batch=2,
        device=torch.device("cpu"),
        task_config=Namespace(datatype="msrvtt"),
    )

    assert resolved.dtype == torch.long
    with pytest.raises(ValueError, match="video_group_id must use an integer dtype"):
        UATVR.resolve_video_group_ids(
            torch.tensor([4.5, 5.5]),
            local_batch=2,
            device=torch.device("cpu"),
            task_config=Namespace(datatype="msrvtt"),
        )
    with pytest.raises(ValueError, match="video_group_id length=1.*local batch=2"):
        UATVR.resolve_video_group_ids(
            torch.tensor([4]),
            local_batch=2,
            device=torch.device("cpu"),
            task_config=Namespace(datatype="msrvtt"),
        )


def _make_forward_only_model(retrieval_logits, group_ids, hard_negative_loss=0.0):
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.loss_fct = MultiPositiveCrossEn()
    model.loss_activations = {"hard_negative": False}
    model.task_config = Namespace(datatype="msrvtt", world_size=1, rank=0)
    model.loose_type = True

    def flatten_video_input(_self, video):
        return video, 1

    def get_sequence_output(_self, input_ids, token_type_ids, attention_mask, shaped):
        batch = input_ids.size(0)
        return torch.zeros(batch, 1, 2), torch.zeros(batch, 2, 2)

    def get_visual_output(_self, video, video_mask, shaped, video_frame):
        batch = video_mask.size(0)
        return torch.zeros(batch, 1, 2), torch.zeros(batch, 1, 1, 2)

    def get_similarity_logits(_self, *args, video_group_id=None, **kwargs):
        assert torch.equal(video_group_id, group_ids)
        zero = retrieval_logits.new_zeros(())
        return {
            "retrieve_logits": retrieval_logits,
            "video_group_id": group_ids,
            "MIL_loss": zero,
            "evidential_loss": zero,
            "neg_reg_loss": zero,
            "orth_loss": zero,
            "hard_negative_loss": retrieval_logits.new_tensor(hard_negative_loss),
            "uacl_intra_loss": zero,
            "uacl_kl_loss": zero,
        }

    model._flatten_video_input = MethodType(flatten_video_input, model)
    model.get_sequence_output = MethodType(get_sequence_output, model)
    model.get_visual_output = MethodType(get_visual_output, model)
    model.get_similarity_logits = MethodType(get_similarity_logits, model)
    model.train()
    return model


def test_forward_uses_bidirectional_multi_positive_loss_and_reports_telemetry():
    retrieval_logits = torch.tensor(
        [[4.0, 1.0, 0.0], [3.0, 2.0, -1.0], [0.0, 1.0, 5.0]],
        requires_grad=True,
    )
    group_ids = torch.tensor([7, 7, 9])
    model = _make_forward_only_model(retrieval_logits, group_ids)

    loss_dict = model(
        torch.zeros(3, 1, 2, dtype=torch.long),
        torch.zeros(3, 1, 2, dtype=torch.long),
        torch.ones(3, 1, 2, dtype=torch.long),
        torch.zeros(3, 1, 1, 1, 1, 1, 1),
        torch.ones(3, 1, 1, dtype=torch.long),
        video_group_id=group_ids,
    )
    expected = (
        model.loss_fct(retrieval_logits, group_ids, group_ids)
        + model.loss_fct(retrieval_logits.T, group_ids, group_ids)
    ) / 2

    torch.testing.assert_close(loss_dict["sim_loss"], expected)
    assert loss_dict["unique_video_count"].item() == 2
    assert loss_dict["duplicate_sample_count"].item() == 1
    assert loss_dict["mean_positive_count"].item() == pytest.approx(5 / 3)


def test_hard_negative_loss_is_separate_from_multi_positive_candidates():
    retrieval_logits = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    group_ids = torch.tensor([1, 2])
    model = _make_forward_only_model(
        retrieval_logits, group_ids, hard_negative_loss=7.0
    )

    loss_dict = model(
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.ones(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 1, 1, 1, 1, 1),
        torch.ones(2, 1, 1, dtype=torch.long),
        video_group_id=group_ids,
    )

    expected_main = torch.nn.functional.cross_entropy(
        retrieval_logits, torch.arange(2)
    )
    torch.testing.assert_close(loss_dict["sim_loss"], expected_main)
    torch.testing.assert_close(
        loss_dict["total"], expected_main + retrieval_logits.new_tensor(7.0)
    )


@pytest.mark.parametrize(
    ("backbone_type", "checkpoint_key", "detected_type"),
    [
        ("openai_clip", "clip.visual.patch_embed.proj.weight", "eva_clip"),
        ("eva_clip", "clip.visual.conv1.weight", "openai_clip"),
    ],
)
def test_checkpoint_backbone_contract_rejects_opposite_backbone(
    backbone_type,
    checkpoint_key,
    detected_type,
):
    with pytest.raises(ValueError, match=rf"backbone_type={backbone_type}.*{detected_type}"):
        UATVR._validate_checkpoint_backbone({checkpoint_key: torch.ones(1)}, backbone_type)


@pytest.mark.parametrize(
    ("backbone_type", "checkpoint_key"),
    [
        ("openai_clip", "clip.visual.conv1.weight"),
        ("eva_clip", "clip.visual.patch_embed.proj.weight"),
    ],
)
def test_checkpoint_backbone_contract_accepts_matching_resume(backbone_type, checkpoint_key):
    UATVR._validate_checkpoint_backbone({checkpoint_key: torch.ones(1)}, backbone_type)


def test_checkpoint_backbone_contract_allows_upper_layers_without_backbone_identity():
    state_dict = {
        "transformerClip.resblocks.0.attn.in_proj_weight": torch.ones(1),
        "sap.anchor_queries": torch.ones(1),
    }

    UATVR._validate_checkpoint_backbone(state_dict, "openai_clip")
    UATVR._validate_checkpoint_backbone(state_dict, "eva_clip")


def test_checkpoint_backbone_contract_rejects_mixed_backbone_keys():
    state_dict = {
        "clip.visual.conv1.weight": torch.ones(1),
        "clip.visual.patch_embed.proj.weight": torch.ones(1),
    }

    with pytest.raises(ValueError, match="mixed.*openai_clip.*eva_clip"):
        UATVR._validate_checkpoint_backbone(state_dict, "openai_clip")


def test_uatvr_backbone_contract_rejects_output_and_model_dimension_mismatch():
    adapter = types.SimpleNamespace(
        output_dim=768,
        supports_text_hidden=True,
        supports_visual_hidden=True,
    )

    with pytest.raises(ValueError, match=r"output_dim=768.*embed_dim=512.*d_model=512"):
        UATVR._validate_backbone_contract(adapter, embed_dim=512, d_model=512)


@pytest.mark.parametrize("capability", ["supports_text_hidden", "supports_visual_hidden"])
def test_uatvr_backbone_contract_rejects_missing_required_hidden_capability(capability):
    adapter = types.SimpleNamespace(
        output_dim=512,
        supports_text_hidden=True,
        supports_visual_hidden=True,
    )
    setattr(adapter, capability, False)

    with pytest.raises(ValueError, match=capability):
        UATVR._validate_backbone_contract(adapter, embed_dim=512, d_model=512)


def test_evidential_matrix_loss_prefers_confident_diagonal():
    confident = torch.tensor(
        [
            [8.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, 8.0],
        ]
    )
    flat = torch.zeros_like(confident)

    confident_loss, _ = UATVR.evidential_matrix_loss(confident)
    flat_loss, _ = UATVR.evidential_matrix_loss(flat)

    assert confident_loss < flat_loss


def test_model_weight_defaults_match_uncertainty_only_setting():
    task_config = Namespace()

    assert getattr(task_config, "w_uncertainty_reg", 1e-3) == 1e-3


def test_loss_activation_resolver_disables_aux_losses_for_hygiene_profile():
    task_config = Namespace(
        experiment_profile="hygiene",
        uncertainty_mode="evidential",
        w_mil=0.01,
        w_evidential=0.01,
        w_neg_reg=0.05,
        w_orth=0.1,
        w_hard_negative=0.05,
        use_explicit_hard_negative_loss=True,
        use_uacl_intra_alignment=True,
        w_uacl_intra=0.01,
        w_uacl_kl=1e-4,
    )

    active = UATVR.resolve_loss_activations(task_config)

    assert active["mil"] is False
    assert active["evidential"] is False
    assert active["neg_reg"] is False
    assert active["orth"] is False
    assert active["hard_negative"] is False
    assert active["uacl_intra"] is False
    assert active["uacl_kl"] is False


def test_hygiene_profile_marks_auxiliary_parameter_names_as_frozen():
    task_config = Namespace(experiment_profile="hygiene")

    frozen_names = [
        "sap.anchor_queries",
        "pie_net_text.fc.weight",
        "uncertain_net_text.fc_logsigma.weight",
        "ada_norm_text.scale.weight",
        "spatial_enhancer.proj.weight",
    ]
    trainable_names = [
        "clip.visual.proj",
        "transformerClip.resblocks.0.attn.in_proj_weight",
        "text_weight_fc.0.weight",
        "video_weight_fc.0.weight",
        "frame_position_embeddings.weight",
        "word_position_embeddings.weight",
    ]

    for name in frozen_names:
        assert UATVR.should_freeze_parameter_for_profile(name, task_config)
    for name in trainable_names:
        assert not UATVR.should_freeze_parameter_for_profile(name, task_config)


def test_hygiene_prob_mu_final_score_keeps_required_paths_trainable():
    task_config = Namespace(
        experiment_profile="hygiene",
        final_score_mode="wti_prob_mu",
        use_ada_norm=True,
    )

    trainable_names = [
        "sap.anchor_queries",
        "pie_net_text.fc.weight",
        "uncertain_net_text.fc_logsigma.weight",
        "ada_norm_text.scale.weight",
    ]
    frozen_names = [
        "spatial_enhancer.proj.weight",
    ]

    for name in trainable_names:
        assert not UATVR.should_freeze_parameter_for_profile(name, task_config)
    for name in frozen_names:
        assert UATVR.should_freeze_parameter_for_profile(name, task_config)


def test_hygiene_anchor_wti_final_score_keeps_only_sap_trainable():
    task_config = Namespace(
        experiment_profile="hygiene",
        final_score_mode="wti_anchor_wti",
        use_ada_norm=True,
    )

    assert not UATVR.should_freeze_parameter_for_profile("sap.anchor_queries", task_config)
    assert not UATVR.should_freeze_parameter_for_profile("sap.decoder.layers.0.self_attn.in_proj_weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("sap.anchor_proj.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("sap.evidential_head.dirichlet_layer.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("pie_net_text.fc.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("uncertain_net_text.fc_logsigma.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("ada_norm_text.scale.weight", task_config)


def test_hygiene_qc_sap_final_score_keeps_sap_and_gate_trainable():
    task_config = Namespace(
        experiment_profile="hygiene",
        final_score_mode="wti_qc_sap",
        use_ada_norm=True,
    )

    assert not UATVR.should_freeze_parameter_for_profile("sap.anchor_queries", task_config)
    assert not UATVR.should_freeze_parameter_for_profile("sap.decoder.layers.0.self_attn.in_proj_weight", task_config)
    assert not UATVR.should_freeze_parameter_for_profile("qc_sap_text_proj.weight", task_config)
    assert not UATVR.should_freeze_parameter_for_profile("qc_sap_anchor_proj.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("sap.anchor_proj.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("sap.evidential_head.dirichlet_layer.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("pie_net_text.fc.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("uncertain_net_text.fc_logsigma.weight", task_config)


def test_qc_sap_projection_layers_freeze_when_score_mode_is_inactive():
    task_config = Namespace(
        experiment_profile="default",
        final_score_mode="wti",
    )

    assert UATVR.should_freeze_parameter_for_profile("qc_sap_text_proj.weight", task_config)
    assert UATVR.should_freeze_parameter_for_profile("qc_sap_anchor_proj.weight", task_config)


def test_compose_final_retrieval_logits_adds_prob_mu_component():
    wti_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prob_mu_logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]])

    final_logits, score_source = UATVR.compose_final_retrieval_logits(
        wti_logits,
        final_score_mode="wti_prob_mu",
        lambda_prob=0.25,
        lambda_anchor=0.0,
        prob_mu_logits=prob_mu_logits,
        anchor_wti_logits=None,
    )

    assert score_source == "wti_prob_mu"
    assert torch.allclose(final_logits, wti_logits + 0.25 * prob_mu_logits)


def test_compose_final_retrieval_logits_adds_qc_sap_component():
    wti_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    qc_sap_logits = torch.tensor([[0.25, -0.25], [0.5, -0.5]])

    final_logits, score_source = UATVR.compose_final_retrieval_logits(
        wti_logits,
        final_score_mode="wti_qc_sap",
        lambda_prob=0.0,
        lambda_anchor=0.0,
        lambda_qc_sap=0.2,
        prob_mu_logits=None,
        anchor_wti_logits=None,
        qc_sap_logits=qc_sap_logits,
    )

    assert score_source == "wti_qc_sap"
    assert torch.allclose(final_logits, wti_logits + 0.2 * qc_sap_logits)


def test_query_conditioned_sap_logits_return_pairwise_gate_diagnostics():
    text_pooled = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )

    logits, stats = UATVR.compute_query_conditioned_sap_logits(
        text_pooled,
        anchors,
        logit_scale=torch.tensor(1.0),
        temperature=0.1,
    )

    assert logits.shape == (2, 2)
    assert stats["gate_entropy_pos"] < stats["gate_entropy_neg"]
    assert stats["gate_top1_mass_pos"] > stats["gate_top1_mass_neg"]
    assert stats["diag"] > stats["off"]
    assert stats["gap"] > 0
    assert stats["std"] > 0


def test_matrix_gap_stats_supports_rectangular_eval_chunks():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.5],
            [0.2, 4.0, 0.1],
        ]
    )

    stats = UATVR._matrix_gap_stats(logits)

    assert stats["diag"] == 3.5
    assert stats["off"] == pytest.approx((1.0 + 0.5 + 0.2 + 0.1) / 4)
    assert stats["gap"] == pytest.approx(3.5 - 0.45)
    assert stats["std"] > 0


def test_matrix_gap_stats_treats_same_video_off_diagonal_as_positive():
    logits = torch.tensor(
        [[4.0, 3.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 6.0]]
    )
    groups = torch.tensor([7, 7, 9])
    mask = groups[:, None].eq(groups[None, :])

    stats = UATVR._matrix_gap_stats(logits, positive_mask=mask)

    assert stats["diag"] == pytest.approx((4.0 + 3.0 + 2.0 + 5.0 + 6.0) / 5)
    assert stats["off"] == pytest.approx(0.5)


def test_matrix_gap_stats_remain_finite_when_batch_has_no_negatives():
    logits = torch.tensor([[2.0, 1.0], [3.0, 4.0]])
    positive_mask = torch.ones_like(logits, dtype=torch.bool)

    stats = UATVR._matrix_gap_stats(logits, positive_mask=positive_mask)

    assert stats["diag"] == pytest.approx(2.5)
    assert stats["off"] == 0.0
    assert stats["gap"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_query_conditioned_sap_diagnostics_use_supplied_positive_mask():
    text_pooled = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )
    groups = torch.tensor([7, 7, 9])
    positive_mask = groups[:, None].eq(groups[None, :])

    logits, stats = UATVR.compute_query_conditioned_sap_logits(
        text_pooled,
        anchors,
        logit_scale=torch.tensor(1.0),
        temperature=0.1,
        positive_mask=positive_mask,
    )
    expected = UATVR._matrix_gap_stats(logits, positive_mask=positive_mask)

    assert stats["diag"] == pytest.approx(expected["diag"])
    assert stats["off"] == pytest.approx(expected["off"])
    assert stats["gate_entropy_pos"] < stats["gate_entropy_neg"]
    assert stats["gate_top1_mass_pos"] > stats["gate_top1_mass_neg"]


def test_loss_activation_resolver_matches_uncertainty_mode_semantics():
    active = UATVR.resolve_loss_activations(
        Namespace(
            experiment_profile="default",
            uncertainty_mode="evidential",
            w_mil=0.01,
            w_evidential=0.01,
            w_neg_reg=0.05,
            w_orth=0.1,
            use_explicit_hard_negative_loss=False,
            use_uacl_intra_alignment=False,
            w_uacl_intra=0.0,
            w_uacl_kl=0.0,
        )
    )
    inactive = UATVR.resolve_loss_activations(
        Namespace(
            experiment_profile="default",
            uncertainty_mode="none",
            w_mil=0.01,
            w_evidential=0.01,
            w_neg_reg=0.05,
            w_orth=0.1,
            use_explicit_hard_negative_loss=False,
            use_uacl_intra_alignment=False,
            w_uacl_intra=0.0,
            w_uacl_kl=0.0,
        )
    )

    assert active["evidential"] is True
    assert active["neg_reg"] is True
    assert inactive["evidential"] is False
    assert inactive["neg_reg"] is False


def test_pienet_padding_mask_removes_padding_from_attention():
    torch.manual_seed(0)
    pie = PIENet(1, 4, 4, 2)
    out = torch.zeros(1, 4)
    x = torch.randn(1, 4, 4)
    pad_mask = torch.tensor([[False, False, True, True]])

    _out, attn, _residual = pie(out, x, pad_mask=pad_mask)

    assert torch.allclose(attn[0, 2:, 0], torch.zeros(2), atol=1e-6)
    assert torch.isclose(attn[0, :2, 0].sum(), torch.tensor(1.0), atol=1e-6)


def test_uncertainty_module_text_uses_true_padding_mask_for_lengths():
    torch.manual_seed(0)
    module = UncertaintyModuleText(4, 4, 2)
    out = torch.randn(2, 4)
    x = torch.randn(2, 5, 4)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, True, True],
        ]
    )

    result = module(out, x, pad_mask=pad_mask)

    assert result["logsigma"].shape == (2, 4)
    assert torch.allclose(result["attention"][0, 3:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(result["attention"][1, 1:, 0], torch.zeros(4), atol=1e-6)


def test_evidential_similarity_penalizes_high_uncertainty():
    """认知不确定性越高，evidential 相似度越低。"""
    mu_video = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # [2, 2], 已 L2 norm
    text_pooled = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # [2, 2]
    # 低不确定性
    epistemic_low = torch.full((2, 4, 2), 0.01)
    sim_low = UATVR._evidential_similarity(mu_video, text_pooled, epistemic_low)
    # 高不确定性
    epistemic_high = torch.full((2, 4, 2), 5.0)
    sim_high = UATVR._evidential_similarity(mu_video, text_pooled, epistemic_high)
    # 低不确定性的对角值应大于高不确定性的
    assert sim_low.diagonal().mean() > sim_high.diagonal().mean()


def test_evidential_nll_loss_penalizes_weak_positives():
    """正对分数越低，NLL loss 越高。"""
    alpha_dir = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    # 强正对
    sim_strong = torch.tensor([[2.0, -0.5], [-0.5, 2.0]])
    loss_strong = UATVR._evidential_nll_loss(sim_strong, alpha_dir)
    # 弱正对
    sim_weak = torch.tensor([[0.1, -0.5], [-0.5, 0.1]])
    loss_weak = UATVR._evidential_nll_loss(sim_weak, alpha_dir)
    assert loss_weak > loss_strong


def test_evidential_neg_reg_loss_penalizes_high_negative_evidence():
    """负对证据量越高，neg_reg loss 越高。"""
    # 高负对分数
    sim_high_neg = torch.tensor([[1.0, 3.0, 3.0], [3.0, 1.0, 3.0], [3.0, 3.0, 1.0]])
    loss_high = UATVR._evidential_neg_reg_loss(sim_high_neg)
    # 低负对分数
    sim_low_neg = torch.tensor([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    loss_low = UATVR._evidential_neg_reg_loss(sim_low_neg)
    assert loss_high > loss_low


def test_explicit_hard_negative_infonce_matches_concatenated_denominator():
    retrieval_logits = torch.tensor(
        [
            [3.0, 0.1, -0.2],
            [0.0, 3.2, 0.4],
            [0.1, -0.3, 2.8],
        ]
    )
    hard_logits = torch.tensor(
        [
            [2.0, 4.0, 8.0],
            [2.4, 0.2, 8.0],
            [1.8, 0.3, 8.0],
        ]
    )
    valid = torch.tensor([1, 1, 0])

    loss = UATVR._hard_negative_infonce_loss(retrieval_logits, hard_logits, valid)

    masked_hard = hard_logits.masked_fill(~valid.to(dtype=torch.bool).unsqueeze(0), torch.finfo(hard_logits.dtype).min)
    expected = torch.nn.functional.cross_entropy(
        torch.cat([retrieval_logits, masked_hard], dim=1),
        torch.arange(retrieval_logits.size(0)),
    )
    assert torch.allclose(loss, expected)


def test_explicit_hard_negative_infonce_ignores_all_invalid_hard_negatives():
    retrieval_logits = torch.eye(3)
    hard_logits = torch.full((3, 3), 100.0)
    valid = torch.zeros(3, dtype=torch.long)

    loss = UATVR._hard_negative_infonce_loss(retrieval_logits, hard_logits, valid)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_select_closest_gaussian_sample_picks_highest_cosine_sample():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_closest_gaussian_sample(mean, samples)

    assert torch.allclose(selected, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_select_uacl_gaussian_sample_keeps_closest_strategy_behavior():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_uacl_gaussian_sample(mean, samples, strategy="closest")

    assert torch.allclose(selected, UATVR._select_closest_gaussian_sample(mean, samples))


def test_select_uacl_gaussian_sample_random_returns_one_sample_per_row():
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_uacl_gaussian_sample(mean, samples, strategy="random")

    assert selected.shape == mean.shape
    for row_idx in range(samples.size(0)):
        assert any(torch.allclose(selected[row_idx], sample) for sample in samples[row_idx])


def test_select_uacl_gaussian_sample_rejects_unknown_strategy():
    mean = torch.tensor([[1.0, 0.0]])
    samples = torch.tensor([[[1.0, 0.0]]])

    with pytest.raises(ValueError, match="Unknown UACL sample strategy"):
        UATVR._select_uacl_gaussian_sample(mean, samples, strategy="bad")


def test_uacl_intra_contrastive_loss_prefers_aligned_pairs():
    anchor = torch.eye(3)
    aligned = anchor.clone()
    shuffled = anchor[[1, 2, 0]]

    aligned_loss = UATVR._uacl_intra_contrastive_loss(anchor, aligned, temperature=0.1)
    shuffled_loss = UATVR._uacl_intra_contrastive_loss(anchor, shuffled, temperature=0.1)

    assert aligned_loss < shuffled_loss


def test_logvar_kl_is_zero_at_unit_variance_and_positive_otherwise():
    unit_logvar = torch.zeros(2, 3)
    shifted_logvar = torch.full((2, 3), 0.5)

    assert torch.isclose(UATVR._logvar_kl(unit_logvar), torch.tensor(0.0))
    assert UATVR._logvar_kl(shifted_logvar) > 0


def _tiny_wti_model(dtype=torch.float32):
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.text_weight_fc = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    model.video_weight_fc = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    torch.nn.init.zeros_(model.text_weight_fc.weight)
    torch.nn.init.zeros_(model.video_weight_fc.weight)
    return model


def _reference_uniform_wti(text, video, text_mask, video_mask):
    expected = text.new_empty(text.size(0), video.size(0))
    for text_index in range(text.size(0)):
        valid_text = text[text_index, text_mask[text_index].bool()]
        for video_index in range(video.size(0)):
            valid_video = video[video_index, video_mask[video_index].bool()]
            similarity = valid_text @ valid_video.T
            expected[text_index, video_index] = (
                similarity.max(dim=1).values.mean()
                + similarity.max(dim=0).values.mean()
            ) / 2
    return expected


def test_wti_padding_zero_cannot_beat_negative_valid_similarity():
    model = _tiny_wti_model()
    text = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    video = torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]])

    logits = model.weighted_token_wise_intersection(
        text,
        video,
        torch.tensor([[1, 0]]),
        torch.tensor([[1, 0]]),
    )

    torch.testing.assert_close(logits, torch.tensor([[-1.0]]))


def test_wti_supports_rectangular_batches_and_different_sequence_lengths():
    model = _tiny_wti_model()
    text = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]],
            [[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]],
        ]
    )
    video = torch.tensor(
        [
            [[-1.0, 0.0], [88.0, 88.0]],
            [[0.0, -1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
        ]
    )
    text_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    video_mask = torch.tensor([[1, 0], [1, 1], [1, 1]])

    logits = model.weighted_token_wise_intersection(
        text, video, text_mask, video_mask
    )

    assert logits.shape == (2, 3)
    torch.testing.assert_close(
        logits,
        _reference_uniform_wti(text, video, text_mask, video_mask),
    )


def test_wti_without_padding_matches_existing_formula_and_hand_calculation():
    model = _tiny_wti_model()
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    video = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]])

    logits = model.weighted_token_wise_intersection(
        text,
        video,
        torch.ones(1, 2),
        torch.ones(1, 3),
    )

    # T2V=(1+0)/2, V2T=(1+0+0)/3, final=(T2V+V2T)/2.
    torch.testing.assert_close(logits, torch.tensor([[5.0 / 12.0]]))


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64, torch.float32])
def test_wti_accepts_binary_bool_integer_and_float_masks(mask_dtype):
    model = _tiny_wti_model()
    text = torch.tensor([[[1.0, 0.0], [9.0, 9.0]]])
    video = torch.tensor([[[-1.0, 0.0], [8.0, 8.0]]])
    text_mask = torch.tensor([[1, 0]], dtype=mask_dtype)
    video_mask = torch.tensor([[1, 0]], dtype=mask_dtype)

    logits = model.weighted_token_wise_intersection(
        text, video, text_mask, video_mask
    )

    torch.testing.assert_close(logits, torch.tensor([[-1.0]]))


@pytest.mark.parametrize(
    ("dtype", "device"),
    [
        (torch.float32, "cpu"),
        (torch.bfloat16, "cpu"),
        (torch.float16, "cuda"),
    ],
)
def test_wti_is_finite_for_supported_floating_dtypes(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable for fp16 WTI coverage")
    model = _tiny_wti_model(dtype=dtype).to(device)
    text = torch.tensor(
        [[[1.0, 0.0], [7.0, 7.0]]], dtype=dtype, device=device
    )
    video = torch.tensor(
        [[[-1.0, 0.0], [6.0, 6.0]]], dtype=dtype, device=device
    )
    mask = torch.tensor([[1, 0]], device=device)

    logits = model.weighted_token_wise_intersection(text, video, mask, mask)

    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.float().cpu(), torch.tensor([[-1.0]]))


def test_wti_padding_path_has_finite_gradients():
    model = _tiny_wti_model()
    text = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [4.0, 4.0]],
            [[-1.0, 0.0], [3.0, 3.0], [2.0, 2.0]],
        ],
        requires_grad=True,
    )
    video = torch.tensor(
        [
            [[-1.0, 0.0], [0.0, -1.0]],
            [[1.0, 0.0], [5.0, 5.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    text_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    video_mask = torch.tensor([[1, 1], [1, 0], [1, 1]])

    logits = model.weighted_token_wise_intersection(
        text, video, text_mask, video_mask
    )
    logits.sum().backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(video.grad).all()
    assert torch.isfinite(model.text_weight_fc.weight.grad).all()
    assert torch.isfinite(model.video_weight_fc.weight.grad).all()


@pytest.mark.parametrize(
    ("text_mask", "video_mask", "message"),
    [
        (
            torch.tensor([[1, 0], [0, 0], [1, 1]]),
            torch.tensor([[1, 0], [1, 1], [1, 0], [1, 1]]),
            r"no valid text token at batch indices=\[1\]",
        ),
        (
            torch.tensor([[1, 0], [1, 1], [1, 0]]),
            torch.tensor([[0, 0], [1, 1], [1, 0], [0, 0]]),
            r"no valid video frame at batch indices=\[0, 3\]",
        ),
    ],
)
def test_wti_rejects_partially_empty_samples_with_exact_indices(
    text_mask, video_mask, message
):
    model = _tiny_wti_model()
    text = torch.zeros(3, 2, 2)
    video = torch.zeros(4, 2, 2)

    with pytest.raises(ValueError, match=message):
        model.weighted_token_wise_intersection(
            text, video, text_mask, video_mask
        )


@pytest.mark.parametrize(
    ("text_shape", "video_shape", "text_mask_shape", "video_mask_shape", "message"),
    [
        ((2, 2), (3, 2, 2), (2, 2), (3, 2), "text_token must be 3D"),
        ((2, 2, 2), (3, 2, 2, 1), (2, 2), (3, 2), "frame_token must be 3D"),
        ((2, 2, 2), (3, 2, 3), (2, 2), (3, 2), "feature dimensions must match"),
        (
            (2, 2, 2),
            (3, 2, 2),
            (2, 1),
            (3, 2),
            r"attention_mask shape=\(2, 1\).*expected=\(2, 2\)",
        ),
        (
            (2, 2, 2),
            (3, 2, 2),
            (2, 2),
            (1, 2),
            r"video_mask shape=\(1, 2\).*expected=\(3, 2\)",
        ),
    ],
)
def test_wti_rejects_invalid_shapes_before_broadcasting(
    text_shape,
    video_shape,
    text_mask_shape,
    video_mask_shape,
    message,
):
    model = _tiny_wti_model()
    text = torch.zeros(text_shape)
    video = torch.zeros(video_shape)
    text_mask = torch.ones(text_mask_shape)
    video_mask = torch.ones(video_mask_shape)

    with pytest.raises(ValueError, match=message):
        model.weighted_token_wise_intersection(
            text, video, text_mask, video_mask
        )


def test_wti_rejects_mask_device_mismatch_with_clear_error():
    model = _tiny_wti_model()
    text = torch.zeros(1, 2, 2)
    video = torch.zeros(1, 2, 2)
    attention_mask = torch.ones(1, 2, device="meta")
    video_mask = torch.ones(1, 2)

    with pytest.raises(
        ValueError,
        match=r"attention_mask device=meta.*text_token device=cpu",
    ):
        model.weighted_token_wise_intersection(
            text, video, attention_mask, video_mask
        )


def test_wti_rejects_non_binary_masks():
    model = _tiny_wti_model()
    tokens = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="attention_mask must be binary"):
        model.weighted_token_wise_intersection(
            tokens,
            tokens,
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[1, 0]]),
        )
