"""Task 6 — CRF training head integration tests.

These tests exercise ``modules.crf_head.CRFTrainingHead`` and
``paired_visual_forward`` without loading CLIP weights. A tiny linear
encoder stands in for ``UATVR.get_visual_output``; every gradient /
freeze / identity invariant the plan requires is checked at the head
level so a full-model smoke can rely on them.
"""

from __future__ import annotations

import torch
from torch import nn

from modules.crf_head import CRFTrainingHead, paired_visual_forward
from uncertainty_modules.types import CRFRankingLoss


def _make_head(input_dim=8, hidden_dim=16) -> CRFTrainingHead:
    torch.manual_seed(0)
    return CRFTrainingHead(input_dim=input_dim, hidden_dim=hidden_dim, pooling_temperature=0.0)


def test_global_head_is_frozen_by_default():
    head = _make_head()
    for p in head.video_aleatoric.global_uncertainty_head.parameters():
        assert p.requires_grad is False
    # Other aleatoric parameters must remain trainable.
    for p in head.video_aleatoric.uncertainty_head.parameters():
        assert p.requires_grad is True
    for p in head.video_aleatoric.relevance_head.parameters():
        assert p.requires_grad is True


def test_optimizer_over_requires_grad_filter_excludes_global_head():
    head = _make_head()
    optimizer = torch.optim.SGD(
        [p for p in head.parameters() if p.requires_grad], lr=0.01
    )
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for p in head.video_aleatoric.global_uncertainty_head.parameters():
        assert id(p) not in optimized_ids


def test_score_features_returns_expected_shapes():
    head = _make_head(input_dim=4, hidden_dim=8)
    features = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    u_frame, u_video, relevance = head.score_features(features, mask)
    assert u_frame.shape == (2, 5)
    assert u_video.shape == (2,)
    assert relevance.shape == (2, 5)
    assert torch.allclose(relevance.sum(dim=1), torch.ones(2), atol=1e-4)


def test_identity_pair_delta_is_below_registered_bound():
    """Same input encoded twice must give the same risk — plan requires |Δ| < 5e-4."""
    head = _make_head()
    head.eval()
    features = torch.randn(3, 7, 8)
    mask = torch.ones(3, 7, dtype=torch.bool)
    with torch.no_grad():
        _, u1, _ = head.score_features(features, mask)
        _, u2, _ = head.score_features(features.clone(), mask.clone())
    assert torch.all(torch.abs(u2 - u1) < 5e-4)


def test_score_features_double_forward_deterministic_in_training_dtype():
    head = _make_head()
    features = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        first = head.score_features(features, mask)[1]
        second = head.score_features(features, mask)[1]
    assert torch.allclose(first, second, rtol=1e-4, atol=1e-5)


def test_compute_ranking_loss_returns_named_result_and_finite_total():
    head = _make_head()
    B, T, D = 4, 5, 8
    n_order = 2
    clean = torch.randn(B, T, D)
    primary = torch.randn(B, T, D)
    secondary = torch.randn(n_order, T, D)
    mask = torch.ones(B, T)
    result = head.compute_ranking_loss(
        clean_features=clean,
        primary_features=primary,
        secondary_features=secondary,
        primary_rank=torch.tensor([1, 2, 3, 1]),
        order_rank=torch.tensor([[1, 3], [2, 1]]),
        order_batch_indices=torch.tensor([0, 2]),
        video_mask=mask,
    )
    assert isinstance(result, CRFRankingLoss)
    assert torch.isfinite(result.total)
    assert torch.isfinite(result.clean_frame)
    assert torch.isfinite(result.clean_video)
    assert torch.isfinite(result.order_frame)
    assert torch.isfinite(result.order_video)


def test_ranking_loss_gradient_isolation_only_touches_aleatoric_heads():
    """Backward from ranking loss must not touch a mock retrieval backbone.

    We simulate the isolation contract: features fed to the head are already
    detached from the encoder; therefore grads flow ONLY through the
    aleatoric heads, never back into the encoder.
    """
    head = _make_head()
    encoder = nn.Linear(6, 8)  # stand-in for CLIP visual encoder
    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in encoder.parameters():
        if p.grad is not None:
            p.grad.zero_()

    video_input = torch.randn(4, 3, 6)  # [B, T, in_dim]
    B, T = video_input.shape[:2]
    with torch.no_grad():
        raw = encoder(video_input)  # simulate visual encoder run without grad
    features = raw.detach()  # what compute_ranking_loss consumes

    result = head.compute_ranking_loss(
        clean_features=features,
        primary_features=features + 0.1,
        secondary_features=(features + 0.2)[:1],
        primary_rank=torch.tensor([1, 2, 3, 1]),
        order_rank=torch.tensor([[1, 3]]),
        order_batch_indices=torch.tensor([0]),
        video_mask=torch.ones(B, T),
    )
    result.total.backward()

    # Encoder must have zero gradient — features were detached.
    for p in encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0

    # Aleatoric heads (except global) must receive grad; global head must not.
    trainable_grads = [
        p for name, p in head.video_aleatoric.named_parameters()
        if not name.startswith("global_uncertainty_head")
    ]
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in trainable_grads)
    for p in head.video_aleatoric.global_uncertainty_head.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_paired_visual_forward_isolates_gradient_correctly():
    """Only clean_features carries grad back to the encoder; degraded/order do not."""

    encoder = nn.Linear(6, 8)

    def encoder_fn(video):
        # video: [B, T, in_dim] -> features [B, T, out_dim]
        return encoder(video)

    clean = torch.randn(2, 3, 6, requires_grad=False)
    degraded = torch.randn(2, 3, 6, requires_grad=False)
    order = torch.randn(1, 3, 6, requires_grad=False)

    clean_feat, degraded_feat, order_feat = paired_visual_forward(
        encoder_fn, clean, degraded, order, training=True
    )
    # Clean features must have grad; degraded / order must not.
    assert clean_feat.requires_grad is True
    assert degraded_feat.requires_grad is False
    assert order_feat.requires_grad is False

    # Backprop through clean must reach encoder; running backward on
    # degraded_feat would just be a no-op since it has no grad.
    loss = clean_feat.sum()
    for p in encoder.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    for p in encoder.parameters():
        assert p.grad is not None
        assert torch.count_nonzero(p.grad) > 0


def test_paired_visual_forward_empty_order_returns_correctly_shaped_tensor():
    encoder = nn.Linear(4, 6)

    def encoder_fn(video):
        return encoder(video)

    clean = torch.randn(2, 3, 4)
    degraded = torch.randn(2, 3, 4)
    empty_order = torch.empty((0, 3, 4))
    _, _, order_feat = paired_visual_forward(encoder_fn, clean, degraded, empty_order)
    assert order_feat.shape == (0, 3, 6)


def test_ranking_loss_swaps_primary_secondary_by_rank_into_low_high():
    """Order-slot: when primary_rank > secondary_rank, low = secondary, high = primary."""

    head = _make_head(input_dim=2, hidden_dim=4)
    # Craft features so that primary produces high u and secondary produces low u.
    # We rely on constant features → u is constant; the SWAP happens because the
    # loss consumes rank ordering; verify by testing that the loss value is the
    # same regardless of which side is "primary" as long as ranks are consistent.
    B, T, D = 1, 2, 2
    a = torch.full((B, T, D), 0.5)
    b = torch.full((B, T, D), 0.7)
    mask = torch.ones(B, T)

    # Case A: primary = a (rank 1), secondary = b (rank 3) → low=primary, high=secondary
    res_a = head.compute_ranking_loss(
        clean_features=a, primary_features=a, secondary_features=b,
        primary_rank=torch.tensor([1]),
        order_rank=torch.tensor([[1, 3]]),
        order_batch_indices=torch.tensor([0]),
        video_mask=mask,
    )
    # Case B: primary = b (rank 3), secondary = a (rank 1) → low=secondary, high=primary
    res_b = head.compute_ranking_loss(
        clean_features=a, primary_features=b, secondary_features=a,
        primary_rank=torch.tensor([3]),
        order_rank=torch.tensor([[3, 1]]),
        order_batch_indices=torch.tensor([0]),
        video_mask=mask,
    )
    # Order term should be identical because low/high resolve to the same
    # underlying tensors after the swap.
    assert torch.allclose(res_a.order_frame, res_b.order_frame, atol=1e-6)
    assert torch.allclose(res_a.order_video, res_b.order_video, atol=1e-6)
