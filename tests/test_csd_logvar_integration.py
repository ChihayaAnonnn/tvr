from types import SimpleNamespace

import torch
from torch import nn

import main_task_retrieval as retrieval
from main_task_retrieval import (
    configure_csd_logvar_sidecar_training,
    set_csd_logvar_training_mode,
)
from modules.modeling_mulit import UATVR
from prob_models.csd_logvar import CSDLogVarSidecar


class _TinyA2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.retrieval_risk_sidecar = nn.Linear(4, 2)
        self.csd_logvar_sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
        self.csd_logvar_sidecar_only = True


def test_a2_only_freezes_every_parameter_outside_exact_prefix():
    model = _TinyA2Model()
    trainable_count = configure_csd_logvar_sidecar_training(
        model,
        SimpleNamespace(
            use_csd_logvar_sidecar=True,
            csd_logvar_sidecar_only=True,
        ),
    )
    assert trainable_count == 64
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable and all(
        name.startswith("csd_logvar_sidecar.") for name in trainable
    )
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 64


def test_one_step_smoke_flushes_diagnostics_before_display_interval():
    assert retrieval._should_log_training_step(1, log_step=20, smoke_cap=1)
    assert not retrieval._should_log_training_step(1, log_step=20, smoke_cap=None)
    assert retrieval._should_log_training_step(20, log_step=20, smoke_cap=None)


def test_a2_mode_keeps_only_new_sidecar_in_train_mode():
    model = _TinyA2Model().train()
    set_csd_logvar_training_mode(model)
    assert model.training is True
    assert model.backbone.training is False
    assert model.retrieval_risk_sidecar.training is False
    assert model.csd_logvar_sidecar.training is True


def _tiny_uatvr_for_a2():
    model = object.__new__(UATVR)
    nn.Module.__init__(model)
    model.use_csd_logvar_sidecar = True
    model.csd_logvar_sidecar_only = True
    model.w_csd, model.w_csd_grounding, model.w_csd_vib = 1.0, 0.05, 1e-4
    model.csd_hardness_temperature = 1.0
    model.csd_logvar_sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    return model


def test_uatvr_a2_terms_update_only_new_sidecar():
    model = _tiny_uatvr_for_a2()
    wti = torch.randn(3, 3, requires_grad=True)
    text_mu = torch.nn.functional.normalize(
        torch.randn(3, 4), dim=-1
    ).requires_grad_()
    video_mu = torch.nn.functional.normalize(
        torch.randn(3, 4), dim=-1
    ).requires_grad_()
    terms = model._csd_logvar_terms(
        wti, text_mu, video_mu, torch.tensor([7, 7, 9])
    )
    terms["csd_logvar_loss"].backward()
    assert wti.grad is None and text_mu.grad is None and video_mu.grad is None


def test_sidecar_state_dict_round_trip_is_exact(tmp_path):
    torch.manual_seed(19)
    source = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    text_mu = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    video_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    expected = source(text_mu, video_mu)
    checkpoint = tmp_path / "csd-logvar.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(text_mu, video_mu)
    for name in ("text_logvar", "video_logvar", "match_logits"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
