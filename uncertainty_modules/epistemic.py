import torch
from torch import nn
from torch.nn import functional as F

from .types import EpistemicOutput


class EpistemicUncertaintyModule(nn.Module):
    @staticmethod
    def _validate_samples(samples: torch.Tensor) -> None:
        if (
            samples.ndim != 3
            or samples.shape[0] < 2
            or samples.shape[1] == 0
            or samples.shape[2] == 0
        ):
            raise ValueError(
                "samples must have shape [M, B, D] with M >= 2"
            )
        if (
            not samples.is_floating_point()
            or not torch.isfinite(samples).all()
        ):
            raise ValueError(
                "samples must contain finite floating-point values"
            )

    def _disagreement(
        self,
        samples: torch.Tensor,
        name: str,
    ) -> EpistemicOutput:
        self._validate_samples(samples)
        diagonal_variance = torch.var(samples, dim=0, unbiased=False)
        uncertainty_score = diagonal_variance.mean(dim=-1)
        return EpistemicOutput(
            uncertainty_score=uncertainty_score,
            component_scores={name: uncertainty_score},
            diagonal_variance=diagonal_variance,
        )

    def from_ensemble(self, samples: torch.Tensor) -> EpistemicOutput:
        return self._disagreement(samples, "ensemble")

    def from_mc_dropout(self, samples: torch.Tensor) -> EpistemicOutput:
        return self._disagreement(samples, "mc_dropout")

    def from_prototypes(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        metric: str = "cosine",
    ) -> EpistemicOutput:
        if (
            features.ndim != 2
            or prototypes.ndim != 2
            or features.shape[0] == 0
            or prototypes.shape[0] == 0
        ):
            raise ValueError(
                "features and prototypes must be non-empty rank-2 tensors"
            )
        if (
            features.shape[1] != prototypes.shape[1]
            or features.device != prototypes.device
            or features.dtype != prototypes.dtype
        ):
            raise ValueError(
                "features and prototypes must share feature size, device, and dtype"
            )
        if (
            not features.is_floating_point()
            or not torch.isfinite(features).all()
            or not torch.isfinite(prototypes).all()
        ):
            raise ValueError(
                "features and prototypes must contain finite floating-point values"
            )

        if metric == "cosine":
            eps = torch.finfo(features.dtype).eps
            normalized_features = F.normalize(features, dim=-1, eps=eps)
            normalized_prototypes = F.normalize(prototypes, dim=-1, eps=eps)
            similarities = normalized_features @ normalized_prototypes.transpose(
                0,
                1,
            )
            distances = 1.0 - similarities.clamp(min=-1.0, max=1.0)
        elif metric == "squared_euclidean":
            distances = torch.cdist(features, prototypes).square()
        else:
            raise ValueError(
                "metric must be 'cosine' or 'squared_euclidean'"
            )

        nearest_distance, nearest_index = distances.min(dim=1)
        return EpistemicOutput(
            uncertainty_score=nearest_distance,
            component_scores={"prototype": nearest_distance},
            nearest_prototype_distance=nearest_distance,
            nearest_prototype_index=nearest_index,
        )
