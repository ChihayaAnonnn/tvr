"""Retrieval-grounded uncertainty utilities.

The score path is deliberately detached from the retrieval model: this module
learns to describe an existing candidate ranking and never modifies it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _oriented_scores(logits: torch.Tensor, direction: str) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape={tuple(logits.shape)}")
    if direction not in {"t2v", "v2t"}:
        raise ValueError("direction must be 't2v' or 'v2t'")

    scores = logits if direction == "t2v" else logits.T
    if scores.size(1) < 2:
        raise ValueError("retrieval risk requires at least two candidates")
    if not torch.isfinite(scores).all():
        raise ValueError("retrieval risk requires finite logits")
    return scores


def score_distribution_features(
    logits: torch.Tensor,
    direction: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Summarize each query's complete candidate-score distribution."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = _oriented_scores(logits, direction).float()
    mean = scores.mean(dim=1)
    std = scores.std(dim=1, unbiased=False)
    standardized = (scores - mean[:, None]) / std.clamp_min(1e-6)[:, None]
    top2 = standardized.topk(k=2, dim=1).values

    probability = F.softmax(standardized / temperature, dim=1)
    entropy = -(
        probability * probability.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(scores.size(1))
    concentration = (
        torch.logsumexp(standardized / temperature, dim=1)
        - math.log(scores.size(1))
    )

    direction_code = scores.new_tensor(
        [1.0, 0.0] if direction == "t2v" else [0.0, 1.0]
    ).expand(scores.size(0), -1)
    scalar_features = torch.stack(
        (
            scores.max(dim=1).values,
            mean,
            std,
            top2[:, 0] - top2[:, 1],
            entropy,
            concentration,
        ),
        dim=1,
    )
    return torch.cat((scalar_features, direction_code), dim=1)


class RetrievalRiskSidecar(nn.Module):
    """Small binary evidential head over detached retrieval-score features."""

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        logits: torch.Tensor,
        direction: str,
    ) -> dict[str, torch.Tensor]:
        features = score_distribution_features(
            logits.detach(), direction, self.temperature
        )
        alpha = F.softplus(self.network(features)) + 1.0
        strength = alpha.sum(dim=1)
        return {
            "alpha": alpha,
            "failure_risk": alpha[:, 1] / strength,
            "vacuity": 2.0 / strength,
            "features": features,
        }


def positive_mask_from_group_ids(
    query_group_ids: torch.Tensor,
    candidate_group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a possibly multi-positive target from stable sample groups."""

    query = torch.as_tensor(query_group_ids).reshape(-1)
    candidate = (
        query
        if candidate_group_ids is None
        else torch.as_tensor(candidate_group_ids, device=query.device).reshape(-1)
    )
    if query.numel() == 0 or candidate.numel() == 0:
        raise ValueError("query and candidate group IDs must be non-empty")
    return query[:, None].eq(candidate[None, :])


def _validate_positive_mask(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.as_tensor(
        positive_mask, device=scores.device, dtype=torch.bool
    )
    if mask.shape != scores.shape:
        raise ValueError(
            "positive_mask must match logits: "
            f"mask={tuple(mask.shape)} logits={tuple(scores.shape)}"
        )
    if not mask.any(dim=1).all():
        raise ValueError("every query must have at least one positive")
    if mask.all(dim=1).any():
        raise ValueError("every query must have at least one negative")
    return mask


def retrieval_hardness(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Detached soft rank pressure from all negatives against all positives."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = _oriented_scores(logits, "t2v")
    mask = _validate_positive_mask(scores, positive_mask)
    positive = torch.logsumexp(
        scores.masked_fill(~mask, -torch.inf), dim=1
    )
    negative = torch.logsumexp(
        scores.masked_fill(mask, -torch.inf), dim=1
    )
    return F.softplus((negative - positive) / temperature).detach()


def _binary_evidential_loss(
    alpha: torch.Tensor,
    failure: torch.Tensor,
) -> torch.Tensor:
    target = torch.stack((1.0 - failure, failure), dim=1)
    strength = alpha.sum(dim=1, keepdim=True)
    probability = alpha / strength
    fit = ((target - probability) ** 2).sum(dim=1)
    variance = (
        alpha
        * (strength - alpha)
        / (strength.square() * (strength + 1.0))
    ).sum(dim=1)
    return (fit + variance).mean()


def _robust_rank_normalize(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(torch.argsort(values))
    return order.to(values.dtype) / max(values.numel() - 1, 1)


def _pairwise_uncertainty_ranking_loss(
    predicted: torch.Tensor,
    hardness: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    hardness_delta = hardness[:, None] - hardness[None, :]
    valid = hardness_delta.abs() > margin
    if not valid.any():
        return predicted.sum() * 0.0
    predicted_delta = predicted[:, None] - predicted[None, :]
    direction = hardness_delta.sign()
    return F.softplus(
        -direction[valid] * predicted_delta[valid]
    ).mean()


def _direction_risk_objective(
    sidecar: RetrievalRiskSidecar,
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    direction: str,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = logits if direction == "t2v" else logits.T
    mask = positive_mask if direction == "t2v" else positive_mask.T
    output = sidecar(logits, direction)
    hardness = retrieval_hardness(scores, mask, temperature)
    top_index = scores.detach().argmax(dim=1, keepdim=True)
    failure = (~mask.gather(1, top_index).squeeze(1)).float()

    evidential = _binary_evidential_loss(output["alpha"], failure)
    fit = F.smooth_l1_loss(
        output["failure_risk"], _robust_rank_normalize(hardness)
    )
    ranking = _pairwise_uncertainty_ranking_loss(
        output["failure_risk"], hardness
    )
    loss = evidential + fit + ranking
    return loss, {
        **output,
        "hardness": hardness,
        "failure": failure,
        "edl": evidential,
        "fit": fit,
        "rank": ranking,
    }


def retrieval_risk_objective(
    sidecar: RetrievalRiskSidecar,
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train row- and column-conditioned failure risk symmetrically."""

    scores = _oriented_scores(logits, "t2v")
    mask = _validate_positive_mask(scores, positive_mask)
    t2v_loss, t2v = _direction_risk_objective(
        sidecar, scores, mask, "t2v", temperature
    )
    v2t_loss, v2t = _direction_risk_objective(
        sidecar, scores, mask, "v2t", temperature
    )
    return 0.5 * (t2v_loss + v2t_loss), {
        "edl": 0.5 * (t2v["edl"] + v2t["edl"]),
        "fit": 0.5 * (t2v["fit"] + v2t["fit"]),
        "rank": 0.5 * (t2v["rank"] + v2t["rank"]),
        "t2v": t2v,
        "v2t": v2t,
    }
