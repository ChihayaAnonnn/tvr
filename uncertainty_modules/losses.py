import torch
from torch.nn import functional as F


def _require_same_shape(
    left: torch.Tensor,
    right: torch.Tensor,
    names: str,
) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{names} must have the same shape")


def uncertainty_ranking_loss(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    _require_same_shape(clean, degraded, "clean and degraded")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    return F.relu(float(margin) + clean - degraded).mean()


def semantic_consistency_loss(
    clean_mean: torch.Tensor,
    degraded_mean: torch.Tensor,
) -> torch.Tensor:
    _require_same_shape(
        clean_mean,
        degraded_mean,
        "clean_mean and degraded_mean",
    )
    return F.mse_loss(clean_mean, degraded_mean)


def variance_prior_loss(
    variance: torch.Tensor,
    target_log_variance: float | torch.Tensor,
) -> torch.Tensor:
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError("variance must be finite and positive")
    target = torch.as_tensor(
        target_log_variance,
        dtype=variance.dtype,
        device=variance.device,
    )
    return torch.mean((torch.log(variance) - target).square())
