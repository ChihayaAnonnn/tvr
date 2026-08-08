import torch
from torch import nn

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
