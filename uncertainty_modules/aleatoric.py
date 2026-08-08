import math

import torch
from torch import nn
from torch.nn import functional as F

from .types import AleatoricOutput, ProbabilisticEmbedding


class AleatoricUncertaintyModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        min_variance: float = 1e-6,
        max_variance: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        min_variance = float(min_variance)
        max_variance = float(max_variance)
        if not math.isfinite(min_variance) or not math.isfinite(max_variance):
            raise ValueError("variance bounds must be finite")
        if min_variance <= 0 or max_variance <= min_variance:
            raise ValueError(
                "variance bounds must satisfy 0 < min_variance < max_variance"
            )

        self.input_dim = input_dim
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.relevance_head = self._head(input_dim, hidden_dim, 1)
        self.uncertainty_head = self._head(input_dim, hidden_dim, input_dim)
        self.global_uncertainty_head = self._head(
            input_dim,
            hidden_dim,
            input_dim,
        )

    @staticmethod
    def _head(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _bounded_variance(self, raw: torch.Tensor) -> torch.Tensor:
        positive = F.softplus(raw)
        return self._bound_positive(positive)

    def _bound_positive(self, positive: torch.Tensor) -> torch.Tensor:
        unit_interval = 1.0 - torch.reciprocal(1.0 + positive)
        return self.min_variance + (
            self.max_variance - self.min_variance
        ) * unit_interval

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> AleatoricOutput:
        if mask is None:
            mask = torch.ones(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )

        relevance_logits = self.relevance_head(features).squeeze(-1)
        relevance_logits = relevance_logits.masked_fill(~mask, -torch.inf)
        relevance_weights = torch.softmax(relevance_logits, dim=1)

        token_variance = self._bounded_variance(
            self.uncertainty_head(features)
        )
        local_uncertainty = token_variance.mean(dim=-1)
        aggregation_logits = relevance_logits - local_uncertainty.detach()
        aggregation_weights = torch.softmax(
            aggregation_logits,
            dim=1,
        )

        mean = torch.sum(
            aggregation_weights.unsqueeze(-1) * features,
            dim=1,
        )
        global_variance = self._bounded_variance(
            self.global_uncertainty_head(mean)
        )
        propagated_variance = torch.sum(
            aggregation_weights.unsqueeze(-1).square() * token_variance,
            dim=1,
        )
        total_positive = propagated_variance + global_variance
        total_variance = self._bound_positive(total_positive)

        embedding = ProbabilisticEmbedding(
            mean=mean,
            variance=total_variance,
            uncertainty_score=total_variance.mean(dim=-1),
        )
        return AleatoricOutput(
            embedding=embedding,
            local_uncertainty=local_uncertainty,
            global_uncertainty=global_variance.mean(dim=-1),
            relevance_weights=relevance_weights,
            aggregation_weights=aggregation_weights,
        )
