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
