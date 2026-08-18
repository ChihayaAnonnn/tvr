"""Unit tests for Fix 1: SAP u_mode failure-supervision BCE loss."""
import torch

from modules.modeling_mulit import UATVR


def test_sap_failure_bce_loss_perfect_calibration_is_zero():
    """u_mode == failure_label → BCE = 0 (in the limit)."""
    B = 8
    wti_logits = torch.eye(B) * 10.0 - 5.0
    wti_logits[:, 0] = -5.0
    wti_logits[3, 0] = 5.0
    u_mode = torch.zeros(B)
    u_mode[0] = 1.0
    loss = UATVR._sap_failure_bce_loss(u_mode, wti_logits)
    assert loss.item() < 1e-3


def test_sap_failure_bce_loss_anticorrelated_is_high():
    """u_mode inverted from failure → BCE >> 0."""
    B = 4
    wti_logits = torch.eye(B) * 10.0
    u_mode = torch.ones(B) * 0.99
    loss = UATVR._sap_failure_bce_loss(u_mode, wti_logits)
    assert loss.item() > 3.0


def test_sap_failure_bce_loss_uniform_uncertain_baseline():
    """u_mode = 0.5 for all → BCE ≈ log(2) regardless of failure rate."""
    B = 6
    wti_logits = torch.eye(B) * 10.0
    wti_logits[2, 2] = -10.0
    wti_logits[2, 5] = 10.0
    u_mode = torch.full((B,), 0.5)
    loss = UATVR._sap_failure_bce_loss(u_mode, wti_logits)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-4


def test_sap_failure_bce_loss_gradient_flows_to_u_mode():
    """Gradient must flow back to u_mode (via BCE), but NOT to wti_logits."""
    B = 4
    wti_logits = torch.eye(B, requires_grad=True) * 5.0
    u_mode = torch.full((B,), 0.5, requires_grad=True)
    loss = UATVR._sap_failure_bce_loss(u_mode, wti_logits)
    loss.backward()
    assert u_mode.grad is not None and u_mode.grad.abs().sum() > 0
    assert wti_logits.grad is None or wti_logits.grad.abs().sum() == 0


def test_sap_failure_bce_loss_clamps_extreme_u_mode():
    """u_mode = 0 or 1 (would blow up log) must be clamped, loss finite."""
    B = 4
    wti_logits = torch.eye(B) * 10.0
    u_mode = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = UATVR._sap_failure_bce_loss(u_mode, wti_logits)
    assert torch.isfinite(loss)
