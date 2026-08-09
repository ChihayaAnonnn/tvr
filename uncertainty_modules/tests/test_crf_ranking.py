"""Task 5 — Four frozen CRF ranking losses."""

from __future__ import annotations

import torch

from uncertainty_modules.losses import crf_ranking_loss
from uncertainty_modules.types import CRFRankingLoss


def test_crf_ranking_result_dataclass_fields_are_scalars():
    result = crf_ranking_loss(
        u_clean_frame=torch.zeros(2, 2),
        u_primary_frame=torch.ones(2, 2),
        primary_rank=torch.tensor([1, 2]),
        video_mask=torch.ones(2, 2, dtype=torch.bool),
        clean_weights=torch.full((2, 2), 0.5),
    )
    assert isinstance(result, CRFRankingLoss)
    for name in ("total", "clean_frame", "clean_video", "order_frame", "order_video"):
        assert getattr(result, name).ndim == 0


def test_crf_ranking_components_match_frozen_formula():
    result = crf_ranking_loss(
        u_clean_frame=torch.tensor([[0.10, 0.20], [0.20, 0.30]]),
        u_primary_frame=torch.tensor([[0.30, 0.40], [0.25, 0.35]]),
        primary_rank=torch.tensor([1, 3]),
        video_mask=torch.ones(2, 2, dtype=torch.bool),
        clean_weights=torch.full((2, 2), 0.5),
        order_low_frame=torch.tensor([[0.30, 0.40]]),
        order_high_frame=torch.tensor([[0.50, 0.60]]),
        order_low_rank=torch.tensor([1]),
        order_high_rank=torch.tensor([3]),
        order_mask=torch.ones(1, 2, dtype=torch.bool),
        order_weights=torch.full((1, 2), 0.5),
        base_margin=0.05,
    )
    expected_sum = result.clean_frame + result.clean_video + result.order_frame + result.order_video
    assert torch.allclose(result.total, expected_sum)


def test_lambda_weights_scale_each_component_independently():
    """total = λ_cf·cf + λ_cv·cv + λ_of·of + λ_ov·ov"""
    kw = dict(
        u_clean_frame=torch.tensor([[0.10, 0.20]]),
        u_primary_frame=torch.tensor([[0.30, 0.35]]),
        primary_rank=torch.tensor([2]),
        video_mask=torch.ones(1, 2, dtype=torch.bool),
        clean_weights=torch.full((1, 2), 0.5),
        order_low_frame=torch.tensor([[0.30, 0.35]]),
        order_high_frame=torch.tensor([[0.50, 0.55]]),
        order_low_rank=torch.tensor([2]),
        order_high_rank=torch.tensor([3]),
        order_mask=torch.ones(1, 2, dtype=torch.bool),
        order_weights=torch.full((1, 2), 0.5),
        base_margin=0.05,
    )
    base = crf_ranking_loss(**kw)
    scaled = crf_ranking_loss(
        **kw, lambda_cf=2.0, lambda_cv=3.0, lambda_of=4.0, lambda_ov=5.0
    )
    expected = (
        2.0 * base.clean_frame
        + 3.0 * base.clean_video
        + 4.0 * base.order_frame
        + 5.0 * base.order_video
    )
    assert torch.allclose(scaled.total, expected)


def test_masked_padding_never_changes_frame_loss():
    mask = torch.tensor([[True, False]])
    kw = dict(
        primary_rank=torch.tensor([1]),
        video_mask=mask,
        clean_weights=torch.tensor([[1.0, 0.0]]),
    )
    left = crf_ranking_loss(
        u_clean_frame=torch.tensor([[0.1, 999.0]]),
        u_primary_frame=torch.tensor([[0.2, -999.0]]),
        **kw,
    )
    right = crf_ranking_loss(
        u_clean_frame=torch.tensor([[0.1, 0.0]]),
        u_primary_frame=torch.tensor([[0.2, 0.0]]),
        **kw,
    )
    assert torch.allclose(left.total, right.total)


def test_order_loss_is_normalized_by_pair_count():
    """Constant per-pair inputs -> order loss identical whether n_pairs=1 or 4."""

    def compute(n):
        low = torch.full((n, 2), 0.20)
        high = torch.full((n, 2), 0.21)
        return crf_ranking_loss(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.tensor([1]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.full((1, 2), 0.5),
            order_low_frame=low,
            order_high_frame=high,
            order_low_rank=torch.ones(n, dtype=torch.long),
            order_high_rank=torch.full((n,), 2, dtype=torch.long),
            order_mask=torch.ones_like(low, dtype=torch.bool),
            order_weights=torch.full_like(low, 0.5),
        )

    one, four = compute(1), compute(4)
    assert torch.allclose(one.order_frame, four.order_frame)
    assert torch.allclose(one.order_video, four.order_video)


def test_no_order_slots_produces_zero_order_terms_preserving_dtype_device():
    result = crf_ranking_loss(
        u_clean_frame=torch.zeros(2, 3, dtype=torch.float32),
        u_primary_frame=torch.ones(2, 3, dtype=torch.float32),
        primary_rank=torch.tensor([2, 3]),
        video_mask=torch.ones(2, 3, dtype=torch.bool),
        clean_weights=torch.full((2, 3), 1.0 / 3.0),
    )
    assert result.order_frame.item() == 0.0
    assert result.order_video.item() == 0.0
    assert result.order_frame.dtype == torch.float32
    assert result.order_video.dtype == torch.float32


def test_primary_margin_scales_with_ordinal_rank():
    """primary_margin = base_margin * primary_rank; larger rank -> larger loss."""

    def compute(rank_value):
        return crf_ranking_loss(
            u_clean_frame=torch.tensor([[0.10, 0.10]]),
            u_primary_frame=torch.tensor([[0.10, 0.10]]),  # zero gap -> loss = margin
            primary_rank=torch.tensor([rank_value]),
            video_mask=torch.ones(1, 2, dtype=torch.bool),
            clean_weights=torch.full((1, 2), 0.5),
            base_margin=0.05,
        )

    r1 = compute(1)
    r3 = compute(3)
    # clean_frame with zero gap = relu(margin) = 0.05 * rank; r3 must be 3x r1.
    assert torch.allclose(r3.clean_frame, 3 * r1.clean_frame, atol=1e-6)
    assert torch.allclose(r3.clean_video, 3 * r1.clean_video, atol=1e-6)


def test_clean_video_uses_relevance_weighted_pooling():
    """clean_video = relu(margin + sum_t w * u_clean - sum_t w * u_primary)."""
    u_clean = torch.tensor([[0.10, 0.20]])
    u_primary = torch.tensor([[0.30, 0.40]])
    weights = torch.tensor([[0.5, 0.5]])
    primary_rank = torch.tensor([2])
    result = crf_ranking_loss(
        u_clean_frame=u_clean,
        u_primary_frame=u_primary,
        primary_rank=primary_rank,
        video_mask=torch.ones(1, 2, dtype=torch.bool),
        clean_weights=weights,
        base_margin=0.05,
    )
    margin = 0.05 * 2
    pooled_clean = (weights * u_clean).sum(dim=1)
    pooled_primary = (weights * u_primary).sum(dim=1)
    expected = torch.relu(margin + pooled_clean - pooled_primary).mean()
    assert torch.allclose(result.clean_video, expected)


def test_backward_gradient_flows_through_all_uncertainty_paths():
    u_clean = torch.zeros(2, 2, requires_grad=True)
    u_primary = torch.zeros(2, 2, requires_grad=True)
    order_low = torch.zeros(1, 2, requires_grad=True)
    order_high = torch.zeros(1, 2, requires_grad=True)
    result = crf_ranking_loss(
        u_clean_frame=u_clean,
        u_primary_frame=u_primary,
        primary_rank=torch.tensor([1, 2]),
        video_mask=torch.ones(2, 2, dtype=torch.bool),
        clean_weights=torch.full((2, 2), 0.5),
        order_low_frame=order_low,
        order_high_frame=order_high,
        order_low_rank=torch.tensor([1]),
        order_high_rank=torch.tensor([2]),
        order_mask=torch.ones(1, 2, dtype=torch.bool),
        order_weights=torch.full((1, 2), 0.5),
    )
    result.total.backward()
    for tensor in (u_clean, u_primary, order_low, order_high):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) > 0
