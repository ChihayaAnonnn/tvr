import pytest
import torch

from prob_models.csd_logvar import CSDLogVarSidecar, csd_logvar_objective
from prob_models.retrieval_risk import positive_mask_from_group_ids


def _fixture():
    torch.manual_seed(3)
    text_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    video_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    logits = torch.tensor([[4., 3., 0.], [3., 4., 0.], [0., 0., 5.]], requires_grad=True)
    mask = positive_mask_from_group_ids(torch.tensor([7, 7, 9]))
    return text_mu, video_mu, logits, mask


def test_objective_accepts_multi_positives_and_detaches_inputs():
    text_mu, video_mu, logits, mask = _fixture()
    sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    loss, output = csd_logvar_objective(sidecar, text_mu, video_mu, logits, mask)
    assert all(
        not value.requires_grad
        for value in output.values()
        if isinstance(value, torch.Tensor)
    )
    loss.backward()
    assert text_mu.grad is None and video_mu.grad is None and logits.grad is None
    assert sum(parameter.grad.abs().sum() for parameter in sidecar.parameters()) > 0
    assert output["text_hardness"].shape == (3,)
    assert output["video_hardness"].shape == (3,)


def test_g_only_removes_scale_and_shift_gradients():
    text_mu, video_mu, logits, mask = _fixture()
    sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    loss, _ = csd_logvar_objective(sidecar, text_mu, video_mu, logits, mask,
                                   w_csd=0.0, w_grounding=0.05, w_vib=1e-4)
    loss.backward()
    assert sidecar.raw_scale.grad is None or torch.count_nonzero(sidecar.raw_scale.grad) == 0
    assert sidecar.shift.grad is None or torch.count_nonzero(sidecar.shift.grad) == 0


def test_objective_rejects_missing_positives_and_negative_weights():
    means = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
    sidecar = CSDLogVarSidecar(embed_dim=2, hidden_dim=2)
    with pytest.raises(ValueError, match="positive"):
        csd_logvar_objective(sidecar, means, means, torch.eye(3), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="weights"):
        csd_logvar_objective(sidecar, means, means, torch.eye(3), torch.eye(3, dtype=torch.bool), w_csd=-1)
