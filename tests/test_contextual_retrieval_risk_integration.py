import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import main_task_retrieval as retrieval
from main_task_retrieval import get_args
import modules.modeling_mulit as modeling
from modules.modeling_mulit import UATVR
from prob_models.contextual_retrieval_risk import ContextualRetrievalRiskSidecar


class _TinyA3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = nn.Linear(4, 4)
        self.retrieval_risk_sidecar = nn.Linear(4, 2)
        self.contextual_risk_sidecar = ContextualRetrievalRiskSidecar(
            embed_dim=4,
            projection_dim=3,
            hidden_dim=5,
            topk=2,
            dropout=0.0,
        )
        self.contextual_risk_sidecar_only = True


def _parse(monkeypatch, extra, *, train=True):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_task_retrieval.py",
            "--do_train" if train else "--do_eval",
            "--expand_msrvtt_sentences",
            "--output_dir",
            "/tmp/contextual-risk-config-test",
            *extra,
        ],
    )
    return get_args()


def test_legacy_cli_keeps_contextual_sidecar_disabled(monkeypatch):
    args = _parse(monkeypatch, [])

    assert args.use_contextual_risk_sidecar is False
    assert args.contextual_risk_sidecar_only is False
    assert args.contextual_risk_topk == 16
    assert args.contextual_risk_projection_dim == 128
    assert args.contextual_risk_hidden_dim == 128
    assert args.contextual_risk_temperature == 1.0
    assert args.w_contextual_risk_fit == 0.25
    assert args.w_contextual_risk_rank == 0.25


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--use_contextual_risk_sidecar"], "use_uatvr_full"),
        (
            ["--use_uatvr_full", "--contextual_risk_sidecar_only"],
            "use_contextual_risk_sidecar",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
            ],
            "use_retrieval_risk_sidecar",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
            ],
            "contextual_risk_sidecar_only",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--retrieval_risk_sidecar_only",
            ],
            "sidecar_only",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--contextual_risk_topk",
                "0",
            ],
            "positive",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--contextual_risk_projection_dim",
                "0",
            ],
            "positive",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--contextual_risk_hidden_dim",
                "0",
            ],
            "positive",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--contextual_risk_temperature",
                "0",
            ],
            "positive",
        ),
        (
            [
                "--use_uatvr_full",
                "--use_retrieval_risk_sidecar",
                "--use_contextual_risk_sidecar",
                "--contextual_risk_sidecar_only",
                "--w_contextual_risk_fit",
                "-0.1",
            ],
            "non-negative",
        ),
    ],
)
def test_invalid_contextual_cli_contracts_fail(monkeypatch, extra, match):
    with pytest.raises(ValueError, match=match):
        _parse(monkeypatch, extra)


def test_evaluation_use_flag_does_not_require_sidecar_only(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--init_model",
            "/tmp/contextual-risk-eval.pt",
            "--use_uatvr_full",
            "--use_contextual_risk_sidecar",
        ],
        train=False,
    )

    assert args.use_contextual_risk_sidecar is True
    assert args.contextual_risk_sidecar_only is False


def test_a3_only_freezes_exact_prefix_and_controls_training_mode():
    model = _TinyA3Model()
    count = retrieval.configure_contextual_risk_sidecar_training(
        model,
        SimpleNamespace(
            use_contextual_risk_sidecar=True,
            contextual_risk_sidecar_only=True,
        ),
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert count > 0
    assert sorted(trainable_names) == sorted(
        name
        for name, _ in model.named_parameters()
        if name.startswith("contextual_risk_sidecar.")
    )

    model.train()
    retrieval._set_contextual_risk_training_mode(model)

    assert model.training
    assert model.contextual_risk_sidecar.training
    assert not model.clip.training
    assert not model.retrieval_risk_sidecar.training
    assert model.contextual_risk_sidecar_only


def test_a3_only_rejects_future_i1_only_mode():
    with pytest.raises(ValueError, match="sidecar_only"):
        retrieval.configure_contextual_risk_sidecar_training(
            _TinyA3Model(),
            SimpleNamespace(
                use_contextual_risk_sidecar=True,
                contextual_risk_sidecar_only=True,
                intrinsic_concentration_only=True,
            ),
        )


def _tiny_uatvr_for_a3():
    model = object.__new__(UATVR)
    nn.Module.__init__(model)
    model.use_contextual_risk_sidecar = True
    model.contextual_risk_sidecar_only = True
    model.w_contextual_risk_fit = 0.25
    model.w_contextual_risk_rank = 0.25
    model.contextual_risk_temperature = 1.0
    model.contextual_risk_sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=4,
        projection_dim=3,
        hidden_dim=5,
        topk=2,
        dropout=0.0,
    )
    return model


def test_gathered_rectangular_shapes_reach_detached_a3_objective():
    model = _tiny_uatvr_for_a3()
    logits = torch.randn(5, 3, requires_grad=True)
    text_means = torch.nn.functional.normalize(
        torch.randn(5, 4), dim=-1
    ).requires_grad_()
    video_means = torch.nn.functional.normalize(
        torch.randn(3, 4), dim=-1
    ).requires_grad_()

    terms = model._contextual_risk_terms(
        logits,
        text_means,
        video_means,
        torch.tensor([7, 7, 8, 9, 9]),
        torch.tensor([7, 8, 9]),
    )
    terms["contextual_risk_loss"].backward()

    assert terms["contextual_risk_loss"].requires_grad
    assert logits.grad is None
    assert text_means.grad is None
    assert video_means.grad is None
    assert terms["contextual_risk_bce"].ndim == 0
    assert terms["contextual_risk_fit"].ndim == 0
    assert terms["contextual_risk_rank"].ndim == 0


def test_pooled_video_embedding_is_masked_mean_then_l2_normalised():
    visual = torch.tensor(
        [[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    pooled = UATVR._pooled_video_embedding(visual, mask)

    expected = torch.nn.functional.normalize(
        torch.tensor([[1.5, 2.0]]), dim=-1
    )
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(pooled.norm(dim=-1), torch.ones(1))


def test_pooled_video_embedding_survives_an_all_padded_row():
    pooled = UATVR._pooled_video_embedding(
        torch.randn(2, 3, 4), torch.zeros(2, 3)
    )

    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(pooled, torch.zeros(2, 4))


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(0.0))


def test_wti_path_gathers_global_axes_before_a3_objective(monkeypatch):
    gathered_shapes = []
    gathered_group_shapes = []

    def gather(tensor, _task_config):
        gathered_shapes.append(tuple(tensor.shape))
        return torch.cat((tensor, tensor), dim=0)

    def gather_no_grad(tensor, _task_config):
        gathered_group_shapes.append(tuple(tensor.shape))
        return torch.cat((tensor, tensor), dim=0)

    monkeypatch.setattr(modeling, "allgather", gather)
    monkeypatch.setattr(modeling, "allgather_no_grad", gather_no_grad)
    model = _tiny_uatvr_for_a3()
    model.task_config = SimpleNamespace()
    model.clip = _FakeClip()
    model.text_weight_fc = nn.Linear(4, 1)
    model.video_weight_fc = nn.Linear(4, 1)
    model.log_sigma_min = None
    model.log_sigma_max = None

    output = model._loose_similarity(
        sequence_output=torch.randn(2, 1, 4),
        text_token=torch.randn(2, 3, 4),
        visual_output=torch.randn(2, 2, 4),
        visual_output_hidden=torch.randn(2, 2, 1, 4),
        attention_mask=torch.ones(2, 3),
        video_mask=torch.ones(2, 2),
        sim_header="meanP",
        group_ids=torch.tensor([7, 9]),
    )

    assert output["retrieve_logits"].shape == (4, 4)
    assert output["contextual_risk_loss"].requires_grad
    assert (2, 1, 4) in gathered_shapes
    assert (2, 3, 4) in gathered_shapes
    assert (2, 2, 4) in gathered_shapes
    assert gathered_group_shapes == [(2,)]


def test_contextual_scoring_detaches_all_retrieval_inputs():
    model = _tiny_uatvr_for_a3()
    logits = torch.randn(3, 3, requires_grad=True)
    text = torch.randn(3, 4, requires_grad=True)
    video = torch.randn(3, 4, requires_grad=True)

    output = model.score_contextual_retrieval_risk(
        logits, text, video, direction="t2v"
    )
    output["failure_risk"].sum().backward()

    assert logits.grad is None and text.grad is None and video.grad is None


def test_legacy_state_dict_has_only_new_a3_keys_missing(caplog):
    source = _TinyA3Model()
    legacy = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("contextual_risk_sidecar.")
    }
    restored = _TinyA3Model()

    incompatible = restored.load_state_dict(legacy, strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        name.startswith("contextual_risk_sidecar.")
        for name in incompatible.missing_keys
    )

    caplog.set_level("INFO")
    loaded = _TinyA3Model()
    UATVR.init_preweight(
        loaded,
        legacy,
        task_config=SimpleNamespace(local_rank=0),
    )
    missing_summaries = [
        record.getMessage()
        for record in caplog.records
        if "新增模块" in record.getMessage()
    ]
    assert len(missing_summaries) == 1
    assert "(contextual_risk_sidecar×8)" in missing_summaries[0]
