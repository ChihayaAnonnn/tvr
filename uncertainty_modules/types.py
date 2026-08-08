from dataclasses import dataclass

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
