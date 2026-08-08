import math
from collections.abc import Mapping

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
            differences = features.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = differences.square().sum(dim=-1)
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

    def combine(
        self,
        scores: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
        calibration: Mapping[
            str,
            tuple[float | torch.Tensor, float | torch.Tensor],
        ],
        threshold: float | None = None,
    ) -> EpistemicOutput:
        if not scores:
            raise ValueError("scores must not be empty")
        if set(scores) != set(weights) or set(scores) != set(calibration):
            raise ValueError(
                "scores, weights, and calibration must have identical keys"
            )

        first_score = next(iter(scores.values()))
        if (
            first_score.ndim != 1
            or first_score.shape[0] == 0
            or not first_score.is_floating_point()
        ):
            raise ValueError(
                "every score must be a non-empty floating-point [B] tensor"
            )

        weight_values: dict[str, float] = {}
        total_weight = 0.0
        standardized_scores: dict[str, torch.Tensor] = {}
        for name, score in scores.items():
            if (
                score.shape != first_score.shape
                or score.device != first_score.device
                or score.dtype != first_score.dtype
                or not torch.isfinite(score).all()
            ):
                raise ValueError(
                    "all scores must share shape, device, dtype, and be finite"
                )

            weight = float(weights[name])
            if not math.isfinite(weight):
                raise ValueError("weights must be finite")
            if weight < 0:
                raise ValueError("weights must be nonnegative")
            weight_values[name] = weight
            total_weight += weight

            mean, standard_deviation = calibration[name]
            mean_tensor = torch.as_tensor(
                mean,
                dtype=score.dtype,
                device=score.device,
            )
            standard_deviation_tensor = torch.as_tensor(
                standard_deviation,
                dtype=score.dtype,
                device=score.device,
            )
            if mean_tensor.numel() != 1 or standard_deviation_tensor.numel() != 1:
                raise ValueError(
                    "each calibration mean and standard deviation must contain a single value"
                )
            mean_tensor = mean_tensor.reshape(())
            standard_deviation_tensor = standard_deviation_tensor.reshape(())
            if (
                not torch.isfinite(mean_tensor).all()
                or not torch.isfinite(standard_deviation_tensor).all()
            ):
                raise ValueError(
                    "calibration means and standard deviations must be finite"
                )
            if torch.any(standard_deviation_tensor <= 0):
                raise ValueError(
                    "calibration standard deviations must be positive"
                )
            standardized_scores[name] = (
                score - mean_tensor
            ) / standard_deviation_tensor

        if not math.isfinite(total_weight):
            raise ValueError("the sum of weights must be finite")
        if total_weight <= 0:
            raise ValueError("at least one weight must be positive")

        combined_score = torch.zeros_like(first_score)
        for name, standardized_score in standardized_scores.items():
            combined_score = combined_score + (
                weight_values[name] / total_weight
            ) * standardized_score

        if threshold is None:
            is_ood = None
        else:
            threshold = float(threshold)
            if not math.isfinite(threshold):
                raise ValueError("threshold must be finite")
            is_ood = combined_score > threshold

        return EpistemicOutput(
            uncertainty_score=combined_score,
            component_scores=dict(scores),
            is_ood=is_ood,
        )
