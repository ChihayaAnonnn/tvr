import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main_task_retrieval import (
    configure_retrieval_risk_sidecar_training,
    get_args,
    set_retrieval_risk_training_mode,
)
from modules.modeling_mulit import UATVR
from prob_models.retrieval_risk import RetrievalRiskSidecar


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.retrieval_risk_sidecar = nn.Linear(3, 2)


def test_old_command_line_keeps_sidecar_disabled(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_task_retrieval.py",
            "--do_train",
            "--expand_msrvtt_sentences",
            "--output_dir",
            "/tmp/retrieval-risk-args-test",
        ],
    )

    args = get_args()

    assert args.use_retrieval_risk_sidecar is False
    assert args.retrieval_risk_sidecar_only is False
    assert args.w_retrieval_risk == 0.0
    assert args.retrieval_risk_temperature == 1.0
    assert args.retrieval_risk_hidden_dim == 32


def test_sidecar_cli_requires_june_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_task_retrieval.py",
            "--do_train",
            "--expand_msrvtt_sentences",
            "--output_dir",
            "/tmp/retrieval-risk-args-test",
            "--use_retrieval_risk_sidecar",
        ],
    )

    with pytest.raises(ValueError, match="use_uatvr_full"):
        get_args()


def test_sidecar_only_freezes_every_other_parameter():
    model = _TinyModel()
    args = SimpleNamespace(
        use_retrieval_risk_sidecar=True,
        retrieval_risk_sidecar_only=True,
    )

    configure_retrieval_risk_sidecar_training(model, args)

    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "retrieval_risk_sidecar.weight",
        "retrieval_risk_sidecar.bias",
    }


def test_sidecar_only_requires_sidecar_mode():
    with pytest.raises(ValueError, match="requires"):
        configure_retrieval_risk_sidecar_training(
            _TinyModel(),
            SimpleNamespace(
                use_retrieval_risk_sidecar=False,
                retrieval_risk_sidecar_only=True,
            ),
        )


def test_standard_training_does_not_change_trainable_parameters():
    model = _TinyModel()
    before = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }

    configure_retrieval_risk_sidecar_training(
        model,
        SimpleNamespace(
            use_retrieval_risk_sidecar=False,
            retrieval_risk_sidecar_only=False,
        ),
    )

    after = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    assert after == before


def _tiny_uatvr_for_risk():
    model = object.__new__(UATVR)
    nn.Module.__init__(model)
    model.use_retrieval_risk_sidecar = True
    model.retrieval_risk_sidecar_only = True
    model.w_retrieval_risk = 0.05
    model.retrieval_risk_temperature = 1.0
    model.retrieval_risk_sidecar = RetrievalRiskSidecar(hidden_dim=8)
    return model


def test_sidecar_mode_selects_raw_wti_logits_bit_for_bit():
    model = _tiny_uatvr_for_risk()
    raw = torch.randn(4, 4)
    confidence_weighted = raw * torch.sigmoid(torch.randn(4, 4))

    selected = model._select_training_retrieval_logits(
        raw, confidence_weighted
    )

    torch.testing.assert_close(selected, raw, rtol=0, atol=0)


def test_model_risk_terms_use_group_ids_and_isolate_logits():
    model = _tiny_uatvr_for_risk()
    logits = torch.tensor(
        [[3.0, 4.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]],
        requires_grad=True,
    )

    terms = model._retrieval_risk_terms(logits, torch.tensor([7, 7, 9]))
    terms["retrieval_risk_loss"].backward()

    assert logits.grad is None
    assert terms["retrieval_risk_loss"].requires_grad
    assert terms["retrieval_risk_edl"].ndim == 0
    assert terms["retrieval_risk_fit"].ndim == 0
    assert terms["retrieval_risk_rank"].ndim == 0


def test_model_risk_terms_require_group_ids_during_training():
    with pytest.raises(ValueError, match="group_ids"):
        _tiny_uatvr_for_risk()._retrieval_risk_terms(
            torch.randn(3, 3), None
        )


def test_sidecar_only_training_keeps_frozen_modules_in_eval_mode():
    model = _TinyModel()
    model.retrieval_risk_sidecar_only = True
    model.train()

    set_retrieval_risk_training_mode(model)

    assert model.training is False
    assert model.backbone.training is False
    assert model.retrieval_risk_sidecar.training is True
