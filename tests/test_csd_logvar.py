import math

import pytest
import torch

from prob_models.csd_logvar import CSDLogVarSidecar, closed_form_sampled_distance


def test_sidecar_has_registered_shape_and_initialization():
    sidecar = CSDLogVarSidecar()
    assert sum(parameter.numel() for parameter in sidecar.parameters()) == 263_426
    assert torch.count_nonzero(sidecar.text_head[-1].weight) == 0
    assert torch.count_nonzero(sidecar.video_head[-1].weight) == 0
    torch.testing.assert_close(sidecar.text_head[-1].bias, torch.full((512,), -8.0))
    torch.testing.assert_close(torch.nn.functional.softplus(sidecar.raw_scale), torch.tensor(5.0))
    assert float(sidecar.shift) == 5.0


def test_closed_form_distance_uses_log_variance_not_log_standard_deviation():
    text_mu = torch.tensor([[0.0, 1.0]])
    video_mu = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    text_logvar = torch.log(torch.tensor([[0.25, 1.0]]))
    video_logvar = torch.log(torch.tensor([[4.0, 0.25], [1.0, 1.0]]))
    actual = closed_form_sampled_distance(text_mu, video_mu, text_logvar, video_logvar)
    torch.testing.assert_close(actual, torch.tensor([[9.5, 7.25]]))


def test_forward_clamps_logvar_and_reports_components():
    sidecar = CSDLogVarSidecar(embed_dim=2, hidden_dim=3)
    with torch.no_grad():
        sidecar.text_head[-1].weight.zero_()
        sidecar.text_head[-1].bias.fill_(-20.0)
        sidecar.video_head[-1].weight.zero_()
        sidecar.video_head[-1].bias.fill_(20.0)
    text_mu = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
    video_mu = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
    output = sidecar(text_mu, video_mu)
    assert output["text_logvar"].shape == (2, 2)
    assert output["video_logvar"].shape == (3, 2)
    torch.testing.assert_close(
        output["text_logvar"], torch.full((2, 2), sidecar.logvar_min)
    )
    torch.testing.assert_close(
        output["video_logvar"], torch.full((3, 2), sidecar.logvar_max)
    )
    torch.testing.assert_close(output["distance"], output["mean_distance"] + output["variance_distance"])
    torch.testing.assert_close(output["u_text"], output["text_logvar"].exp().mean(dim=-1))
    torch.testing.assert_close(output["u_video"], output["video_logvar"].exp().mean(dim=-1))
    torch.testing.assert_close(
        output["match_logits"],
        -torch.nn.functional.softplus(sidecar.raw_scale) * output["distance"]
        + sidecar.shift,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kwargs", [
    {"embed_dim": 0}, {"hidden_dim": 0},
    {"logvar_min": 4.0, "logvar_max": -12.0},
    {"init_logvar": -13.0}, {"initial_scale": 0.0},
])
def test_sidecar_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CSDLogVarSidecar(**kwargs)
