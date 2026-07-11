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

UATVR = importlib.import_module("modules.modeling_mulit").UATVR
MultiPositiveCrossEn = importlib.import_module(
    "modules.until_module"
).MultiPositiveCrossEn
LayerNorm = importlib.import_module("modules.module_clip").LayerNorm

def test_openai_clip_layer_norm_precision_is_propagated():
    clip = torch.nn.Sequential(LayerNorm(4), LayerNorm(4))
    count = UATVR.configure_clip_layer_norm_precision(
        clip, backbone_type="openai_clip", precision="fp32"
    )
    assert count == 2
    assert all(module.precision == "fp32" for module in clip)


def test_eva_backbone_is_not_modified_by_clip_layer_norm_precision():
    backbone = torch.nn.Sequential(torch.nn.LayerNorm(4))
    assert UATVR.configure_clip_layer_norm_precision(
        backbone, backbone_type="eva_clip", precision="fp16"
    ) == 0


def test_openai_clip_gradient_checkpointing_configures_selected_visual_layers():
    text_transformer = types.SimpleNamespace(
        grad_checkpointing=False,
        grad_checkpointing_layers=12,
    )
    visual_transformer = types.SimpleNamespace(
        grad_checkpointing=False,
        grad_checkpointing_layers=12,
        layers=12,
    )
    backbone = types.SimpleNamespace(
        transformer=text_transformer,
        visual=types.SimpleNamespace(transformer=visual_transformer),
    )

    count = UATVR.configure_clip_gradient_checkpointing(
        backbone,
        backbone_type="openai_clip",
        enabled=True,
        visual_layers=4,
    )

    assert count == 1
    assert text_transformer.grad_checkpointing is False
    assert text_transformer.grad_checkpointing_layers == 0
    assert visual_transformer.grad_checkpointing is True
    assert visual_transformer.grad_checkpointing_layers == 4


def test_eva_backbone_is_not_modified_by_clip_gradient_checkpointing():
    transformer = types.SimpleNamespace(grad_checkpointing=False)
    backbone = types.SimpleNamespace(transformer=transformer)

    assert UATVR.configure_clip_gradient_checkpointing(
        backbone, backbone_type="eva_clip", enabled=True
    ) == 0
    assert transformer.grad_checkpointing is False


class _FakeClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.return_hidden_values = []

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        self.return_hidden_values.append(return_hidden)
        pooled = torch.ones(image.size(0), 2)
        if return_hidden:
            return pooled, pooled[:, None, :]
        return pooled


def test_visual_encoder_does_not_request_patch_hidden():
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.clip = _FakeClip()
    video = torch.zeros(2, 3, 2, 2)
    cls = model.get_visual_output(
        video,
        torch.tensor([[1, 1]]),
        shaped=True,
        video_frame=2,
    )
    assert cls.shape == (1, 2, 2)
    assert model.clip.return_hidden_values == [False]


def test_eval_scoring_accepts_direct_visual_tensor_cache():
    from main_task_retrieval import _run_on_single_gpu, eval_epoch

    class _EvalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.loose_type = True
            self.seen_visual = None

        def get_similarity_logits(
            self,
            sequence_output,
            text_token,
            visual_output,
            attention_mask,
            video_mask,
            loose_type,
        ):
            self.seen_visual = visual_output
            logits = torch.einsum(
                "atd,bvd->ab",
                text_token,
                visual_output,
            )
            return logits, {}

    model = _EvalModel()
    visual_cache = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
    ]
    result = _run_on_single_gpu(
        Namespace(eval_vid_chunk_size=8, output_dir=""),
        model,
        [(torch.ones(1, 1), torch.zeros(1, 1))],
        [(torch.ones(1, 1),), (torch.ones(1, 1),)],
        [(torch.zeros(1, 1, 2), torch.tensor([[[1.0, 0.0]]]))],
        visual_cache,
    )

    assert model.seen_visual.shape == (2, 1, 2)
    assert result[0].shape == (1, 2)
    source = inspect.getsource(eval_epoch)
    assert (
        "[tensor.to(devc) for tensor in batch_visual_output_list]"
        in source
    )


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
    model.use_explicit_hard_negative_loss = False
    model.hard_negative_enabled = False
    model.task_config = Namespace(datatype="msrvtt", world_size=1, rank=0)
    model.loose_type = True

    def flatten_video_input(_self, video):
        return video, 1

    def get_sequence_output(_self, input_ids, token_type_ids, attention_mask, shaped):
        batch = input_ids.size(0)
        return torch.zeros(batch, 1, 2), torch.zeros(batch, 2, 2)

    def get_visual_output(_self, video, video_mask, shaped, video_frame):
        batch = video_mask.size(0)
        return torch.zeros(batch, 1, 2)

    def get_similarity_logits(_self, *args, video_group_id=None, **kwargs):
        assert torch.equal(video_group_id, group_ids)
        return {
            "retrieve_logits": retrieval_logits,
            "video_group_id": group_ids,
            "hard_negative_loss": retrieval_logits.new_tensor(hard_negative_loss),
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
    assert set(loss_dict) == {
        "total",
        "sim_loss",
        "hard_negative_loss",
        "unique_video_count",
        "duplicate_sample_count",
        "mean_positive_count",
    }
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
        "text_weight_fc.0.weight": torch.ones(1),
    }

    UATVR._validate_checkpoint_backbone(state_dict, "openai_clip")
    UATVR._validate_checkpoint_backbone(state_dict, "eva_clip")


@pytest.mark.parametrize(
    "key",
    [
        "sap.anchor_tokens",
        "module.sap.anchor_tokens",
        "qc_sap_text_proj.weight",
        "qc_sap_anchor_proj.weight",
        "pie_net_text.query_proj.weight",
        "uncertain_net_text.fc_logsigma.weight",
        "ada_norm_text.scale_net.weight",
        "expansion_tokens",
        "module.expansion_tokens",
        "spatial_enhancer.attention.qkvo.weight",
    ],
)
def test_retired_auxiliary_checkpoint_keys_are_rejected(key):
    with pytest.raises(ValueError, match=r"retired auxiliary checkpoint"):
        UATVR._validate_retired_checkpoint_keys({key: torch.ones(1)})


@pytest.mark.parametrize(
    "key",
    [
        "text_weight_fc.0.weight",
        "video_weight_fc.0.weight",
        "transformerClip.resblocks.0.attn.in_proj_weight",
        "clip.visual.conv1.weight",
        "clip.visual.patch_embed.proj.weight",
    ],
)
def test_active_wti_checkpoint_keys_are_not_rejected(key):
    UATVR._validate_retired_checkpoint_keys({key: torch.ones(1)})


def test_from_pretrained_rejects_retired_keys_before_backbone_loading(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("backbone validation must not run first")

    monkeypatch.setattr(UATVR, "_validate_checkpoint_backbone", fail_if_called)
    task_config = types.SimpleNamespace(backbone_type="openai_clip", local_rank=0)
    with pytest.raises(ValueError, match=r"retired auxiliary checkpoint"):
        UATVR.from_pretrained(
            "cross-base",
            state_dict={"sap.anchor_tokens": torch.ones(1)},
            task_config=task_config,
        )


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


def test_wti_eva_capability_still_requires_text_hidden():
    spec = types.SimpleNamespace(
        supports_text_hidden=False,
        supports_visual_hidden=True,
    )

    with pytest.raises(ValueError, match="text hidden"):
        UATVR._validate_eva_spec_capabilities(spec)


def test_wti_eva_capability_never_requires_visual_patch_hidden():
    spec = types.SimpleNamespace(
        supports_text_hidden=True,
        supports_visual_hidden=False,
    )

    assert UATVR._validate_eva_spec_capabilities(spec) == (True, False)



@pytest.mark.parametrize(
    "name",
    [
        "probabilistic_text",
        "compose_final_retrieval_logits",
        "compute_query_conditioned_sap_logits",
        "resolve_loss_activations",
        "evidential_matrix_loss",
        "_select_uacl_gaussian_sample",
        "_uacl_intra_contrastive_loss",
        "_logvar_kl",
        "_evidential_similarity",
        "_evidential_nll_loss",
        "_evidential_neg_reg_loss",
    ],
)
def test_retired_auxiliary_model_api_is_absent(name):
    assert not hasattr(UATVR, name)


def test_visual_hidden_is_absent_from_active_similarity_api():
    assert "visual_output_hidden" not in inspect.signature(
        UATVR.get_similarity_logits
    ).parameters
    assert "visual_output_hidden" not in inspect.signature(
        UATVR._loose_similarity
    ).parameters


def test_constructor_source_has_no_retired_auxiliary_fields():
    source = inspect.getsource(UATVR.__init__)
    for retired in (
        "self.sap",
        "self.qc_sap_",
        "self.pie_net_text",
        "self.uncertain_net_text",
        "self.ada_norm_text",
        "self.expansion_tokens",
        "self.spatial_enhancer",
    ):
        assert retired not in source


def test_hard_negative_activation_requires_positive_weight():
    assert UATVR._resolve_hard_negative_enabled(True, 0.0) is False
    assert UATVR._resolve_hard_negative_enabled(False, 0.1) is False
    assert UATVR._resolve_hard_negative_enabled(True, 0.1) is True
    with pytest.raises(ValueError, match="non-negative"):
        UATVR._resolve_hard_negative_enabled(True, -0.1)


def _make_wti_unit_model():
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.task_config = Namespace(world_size=1, rank=0)
    model.clip = _FakeClip()
    model.text_weight_fc = torch.nn.Linear(2, 1, bias=False)
    model.video_weight_fc = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.text_weight_fc.weight)
    torch.nn.init.zeros_(model.video_weight_fc.weight)
    model.use_explicit_hard_negative_loss = False
    model.hard_negative_enabled = False
    model.w_hard_negative = 0.25
    model._diag_step = 0
    model._diag_interval = 1000
    model._diag_chain = {}
    model._hard_negative_chain = {}
    return model


def _wti_inputs():
    text = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    video = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    mask = torch.ones(2, 1, dtype=torch.long)
    groups = torch.tensor([10, 20])
    return text, video, mask, groups


def test_train_and_eval_use_identical_wti_score_source():
    model = _make_wti_unit_model()
    text, video, mask, groups = _wti_inputs()
    model.eval()
    eval_logits = model._wti_similarity(text, video, mask, mask)
    model.train()
    train_result = model._wti_similarity(
        text, video, mask, mask, video_group_id=groups
    )
    torch.testing.assert_close(train_result["retrieve_logits"], eval_logits)
    assert train_result["video_group_id"].tolist() == [10, 20]


def test_hard_negative_is_separate_and_weighted_once():
    model = _make_wti_unit_model()
    text, video, mask, groups = _wti_inputs()
    hard_video = video.flip(0)
    hard_valid = torch.tensor([True, True])
    model.eval()
    expected_retrieve = model._wti_similarity(text, video, mask, mask)
    model.train()
    model.use_explicit_hard_negative_loss = True
    model.hard_negative_enabled = True
    result = model._wti_similarity(
        text,
        video,
        mask,
        mask,
        video_group_id=groups,
        hard_visual_output=hard_video,
        hard_video_mask=mask,
        hard_valid=hard_valid,
    )
    expected_hard_logits = model.weighted_token_wise_intersection(
        torch.nn.functional.normalize(text, dim=-1),
        torch.nn.functional.normalize(hard_video, dim=-1),
        mask,
        mask,
    ) * model.clip.logit_scale.exp()
    expected_raw = model._hard_negative_infonce_loss(
        result["retrieve_logits"], expected_hard_logits, hard_valid
    )
    torch.testing.assert_close(result["retrieve_logits"], expected_retrieve)
    torch.testing.assert_close(
        result["hard_negative_loss"], model.w_hard_negative * expected_raw
    )
    assert result["video_group_id"].tolist() == [10, 20]


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
