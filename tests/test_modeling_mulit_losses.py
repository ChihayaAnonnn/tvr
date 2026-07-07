import importlib
import inspect
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub
UATVR = importlib.import_module("modules.modeling_mulit").UATVR
PIENet = importlib.import_module("prob_models.pie_model").PIENet
UncertaintyModuleText = importlib.import_module("prob_models.uncertainty_module").UncertaintyModuleText


def test_forward_accepts_explicit_hard_negative_kwargs():
    signature = inspect.signature(UATVR.forward)

    for name in ["sample_index", "hard_video", "hard_video_mask", "hard_valid"]:
        assert name in signature.parameters


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
