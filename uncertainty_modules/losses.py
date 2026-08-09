import math

import torch
from torch.nn import functional as F

from uncertainty_modules.types import CRFRankingLoss


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
    margin = float(margin)
    if not math.isfinite(margin):
        raise ValueError("margin must be finite")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    return F.relu(margin + clean - degraded).mean()


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
    if not torch.isfinite(target).all():
        raise ValueError("target_log_variance must be finite")
    return torch.mean((torch.log(variance) - target).square())


def _masked_mean_over_T(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean along the temporal axis with padded positions zero-weighted.

    Padded positions carry arbitrary junk, so we always multiply by the mask
    before summing; the divisor is ``clamp_min(1)`` to keep the result finite
    when a batch element has no valid frames (shouldn't happen in practice).
    """
    mask_float = mask.to(values.dtype)
    numer = (values * mask_float).sum(dim=1)
    denom = mask_float.sum(dim=1).clamp_min(1)
    return numer / denom


def crf_ranking_loss(
    u_clean_frame: torch.Tensor,
    u_primary_frame: torch.Tensor,
    primary_rank: torch.Tensor,
    video_mask: torch.Tensor,
    clean_weights: torch.Tensor,
    order_low_frame: torch.Tensor | None = None,
    order_high_frame: torch.Tensor | None = None,
    order_low_rank: torch.Tensor | None = None,
    order_high_rank: torch.Tensor | None = None,
    order_mask: torch.Tensor | None = None,
    order_weights: torch.Tensor | None = None,
    base_margin: float = 0.05,
    lambda_cf: float = 1.0,
    lambda_cv: float = 1.0,
    lambda_of: float = 1.0,
    lambda_ov: float = 1.0,
) -> CRFRankingLoss:
    """Four frozen CRF ranking losses (Task 5 of the CRF plan).

    Frozen formulas (see plan Appendix D):

    * ``primary_margin = base_margin * primary_rank`` broadcast over T
    * ``clean_frame = mean_over_batch( masked_mean_over_T(
        relu(primary_margin + u_clean - u_primary), video_mask ) )``
    * ``clean_video = mean( relu(primary_margin + sum_t w * u_clean
        - sum_t w * u_primary) )`` with ``w = clean_weights``
    * ``order_frame`` mirrors ``clean_frame`` with
      ``order_margin = base_margin * (order_high_rank - order_low_rank)`` and
      the mean is taken over the actual order pair count (not B_local)
    * ``order_video`` mirrors ``clean_video`` under the same order margin
    * ``total = λ_cf·clean_frame + λ_cv·clean_video + λ_of·order_frame + λ_ov·order_video``

    When no order pairs are supplied (any of ``order_low_frame``,
    ``order_high_frame``, ``order_low_rank``, ``order_high_rank`` is None),
    both order terms are set to ``u_clean_frame.sum() * 0.0`` — a
    dtype/device-preserving zero that never NaNs.
    """
    _require_same_shape(u_clean_frame, u_primary_frame, "u_clean_frame and u_primary_frame")
    if u_clean_frame.shape != video_mask.shape:
        raise ValueError("video_mask must match u_clean_frame shape")
    if u_clean_frame.shape != clean_weights.shape:
        raise ValueError("clean_weights must match u_clean_frame shape")
    if primary_rank.shape != (u_clean_frame.shape[0],):
        raise ValueError("primary_rank must have shape (B,)")

    dtype = u_clean_frame.dtype
    device = u_clean_frame.device
    mask_bool = video_mask.to(torch.bool)

    primary_rank_t = primary_rank.to(dtype).view(-1, 1)
    primary_margin = float(base_margin) * primary_rank_t

    clean_frame_pointwise = torch.relu(primary_margin + u_clean_frame - u_primary_frame)
    clean_frame_per_sample = _masked_mean_over_T(clean_frame_pointwise, mask_bool)
    clean_frame = clean_frame_per_sample.mean()

    clean_weights_masked = clean_weights * mask_bool.to(dtype)
    pooled_clean = (clean_weights_masked * u_clean_frame).sum(dim=1)
    pooled_primary = (clean_weights_masked * u_primary_frame).sum(dim=1)
    clean_video_per_sample = torch.relu(
        primary_margin.squeeze(1) + pooled_clean - pooled_primary
    )
    clean_video = clean_video_per_sample.mean()

    have_order = (
        order_low_frame is not None
        and order_high_frame is not None
        and order_low_rank is not None
        and order_high_rank is not None
    )
    if have_order and order_low_frame.shape[0] > 0:
        _require_same_shape(order_low_frame, order_high_frame, "order_low_frame and order_high_frame")
        n_pairs = order_low_frame.shape[0]
        if order_low_rank.shape != (n_pairs,) or order_high_rank.shape != (n_pairs,):
            raise ValueError("order_low_rank/order_high_rank must have shape (n_pairs,)")
        if order_mask is None:
            order_mask_bool = torch.ones_like(order_low_frame, dtype=torch.bool)
        else:
            order_mask_bool = order_mask.to(torch.bool)
        if order_weights is None:
            order_weights_t = torch.full_like(order_low_frame, 1.0 / order_low_frame.shape[1])
        else:
            order_weights_t = order_weights.to(dtype)

        rank_gap = (order_high_rank - order_low_rank).to(dtype).view(-1, 1)
        order_margin = float(base_margin) * rank_gap

        order_frame_pointwise = torch.relu(order_margin + order_low_frame - order_high_frame)
        order_frame_per_pair = _masked_mean_over_T(order_frame_pointwise, order_mask_bool)
        order_frame = order_frame_per_pair.mean()

        order_weights_masked = order_weights_t * order_mask_bool.to(dtype)
        pooled_low = (order_weights_masked * order_low_frame).sum(dim=1)
        pooled_high = (order_weights_masked * order_high_frame).sum(dim=1)
        order_video_per_pair = torch.relu(
            order_margin.squeeze(1) + pooled_low - pooled_high
        )
        order_video = order_video_per_pair.mean()
    else:
        zero = u_clean_frame.sum() * 0.0
        order_frame = zero
        order_video = zero

    total = (
        float(lambda_cf) * clean_frame
        + float(lambda_cv) * clean_video
        + float(lambda_of) * order_frame
        + float(lambda_ov) * order_video
    )
    return CRFRankingLoss(
        total=total,
        clean_frame=clean_frame,
        clean_video=clean_video,
        order_frame=order_frame,
        order_video=order_video,
    )
