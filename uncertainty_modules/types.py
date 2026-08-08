from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class ProbabilisticEmbedding:
    mean: torch.Tensor
    variance: torch.Tensor
    uncertainty_score: torch.Tensor


@dataclass(frozen=True)
class AleatoricOutput:
    embedding: ProbabilisticEmbedding
    local_uncertainty: torch.Tensor
    global_uncertainty: torch.Tensor
    relevance_weights: torch.Tensor
    aggregation_weights: torch.Tensor


@dataclass(frozen=True)
class EpistemicOutput:
    uncertainty_score: torch.Tensor
    component_scores: Mapping[str, torch.Tensor] = field(default_factory=dict)
    diagonal_variance: torch.Tensor | None = None
    nearest_prototype_distance: torch.Tensor | None = None
    nearest_prototype_index: torch.Tensor | None = None
    is_ood: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_scores",
            MappingProxyType(dict(self.component_scores)),
        )
