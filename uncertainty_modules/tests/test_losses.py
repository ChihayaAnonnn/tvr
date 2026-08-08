import pytest
import torch

from uncertainty_modules.losses import (
    semantic_consistency_loss,
    uncertainty_ranking_loss,
    variance_prior_loss,
)


def test_ranking_loss_rewards_higher_degraded_uncertainty():
    clean = torch.tensor([0.2, 0.4])
    well_ranked = torch.tensor([0.5, 0.7])
    poorly_ranked = torch.tensor([0.1, 0.2])

    assert uncertainty_ranking_loss(
        clean,
        well_ranked,
        margin=0.1,
    ) < uncertainty_ranking_loss(
        clean,
        poorly_ranked,
        margin=0.1,
    )


def test_ranking_loss_rejects_invalid_shape_or_margin():
    with pytest.raises(ValueError, match="same shape"):
        uncertainty_ranking_loss(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        uncertainty_ranking_loss(torch.ones(2), torch.ones(2), margin=-0.1)


def test_semantic_consistency_is_zero_for_identical_embeddings():
    embedding = torch.randn(3, 4)

    assert semantic_consistency_loss(
        embedding,
        embedding,
    ).item() == pytest.approx(0.0)


def test_semantic_consistency_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        semantic_consistency_loss(torch.ones(2, 4), torch.ones(3, 4))


def test_variance_prior_is_zero_at_target():
    variance = torch.full((2, 4), 0.5)
    target = torch.log(torch.tensor(0.5))

    assert variance_prior_loss(
        variance,
        target,
    ).item() == pytest.approx(0.0)


def test_variance_prior_rejects_nonpositive_variance():
    with pytest.raises(ValueError, match="positive"):
        variance_prior_loss(torch.tensor([0.0]), 0.0)
