"""Bidirectional matching between stochastic text and video prototypes."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PrototypeMatchOutput:
    """Scores and uncertainty for stochastic prototype matches."""

    logits: torch.Tensor
    pair_uncertainty: torch.Tensor
    text_prototype_scores: torch.Tensor
    video_prototype_scores: torch.Tensor
    stochastic_pair_scores: torch.Tensor


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _fp32_context(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class BidirectionalSoftPrototypeMatcher(nn.Module):
    """Compute stable, bidirectional log-mean-exp prototype similarities."""

    def __init__(self, temperature: float, *, hard_max: bool = False, eps: float = 1e-6):
        super().__init__()
        self.temperature = _positive_finite(temperature, "temperature")
        self.eps = _positive_finite(eps, "eps")
        self.hard_max = bool(hard_max)

    @staticmethod
    def _validate_samples(
        text_samples: torch.Tensor,
        video_samples: torch.Tensor,
        *,
        aligned: bool,
    ) -> None:
        for name, samples in (("text_samples", text_samples), ("video_samples", video_samples)):
            if samples.ndim != 3:
                raise ValueError(f"{name} must have shape [batch, prototypes, dimension]")
            if not samples.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")

        if text_samples.size(1) != video_samples.size(1):
            raise ValueError("text and video samples must have the same number of prototypes")
        if text_samples.size(2) != video_samples.size(2):
            raise ValueError("text and video samples must have the same embedding dimension")
        if aligned and text_samples.size(0) != video_samples.size(0):
            raise ValueError("aligned text and video samples must have the same batch size")

    def _scores_from_cosine(self, cosine: torch.Tensor, log_k: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hard_max:
            return cosine.max(dim=-1).values, cosine.max(dim=-2).values

        scaled_cosine = cosine / self.temperature
        text_scores = self.temperature * (torch.logsumexp(scaled_cosine, dim=-1) - log_k)
        video_scores = self.temperature * (torch.logsumexp(scaled_cosine, dim=-2) - log_k)
        return text_scores, video_scores

    def _output(self, cosine: torch.Tensor, log_k: float) -> PrototypeMatchOutput:
        text_scores, video_scores = self._scores_from_cosine(cosine, log_k)
        logits = 0.5 * (text_scores.mean(dim=-1) + video_scores.mean(dim=-1))
        pair_uncertainty = 0.5 * (text_scores.var(dim=-1, unbiased=False) + video_scores.var(dim=-1, unbiased=False))
        stochastic_pair_scores = 0.5 * (text_scores + video_scores)
        return PrototypeMatchOutput(
            logits=logits,
            pair_uncertainty=pair_uncertainty,
            text_prototype_scores=text_scores,
            video_prototype_scores=video_scores,
            stochastic_pair_scores=stochastic_pair_scores,
        )

    def forward(self, text_samples: torch.Tensor, video_samples: torch.Tensor) -> PrototypeMatchOutput:
        """Score every text-video combination, including rectangular batches."""

        self._validate_samples(text_samples, video_samples, aligned=False)
        with _fp32_context(text_samples):
            text = F.normalize(text_samples.float(), dim=-1, eps=self.eps)
            video = F.normalize(video_samples.float(), dim=-1, eps=self.eps)
            cosine = torch.einsum("tad,vbd->tvab", text, video)
            return self._output(cosine, math.log(text.size(1)))

    def score_pairs(self, text_samples: torch.Tensor, video_samples: torch.Tensor) -> PrototypeMatchOutput:
        """Score only aligned text-video pairs without constructing a full matrix."""

        self._validate_samples(text_samples, video_samples, aligned=True)
        with _fp32_context(text_samples):
            text = F.normalize(text_samples.float(), dim=-1, eps=self.eps)
            video = F.normalize(video_samples.float(), dim=-1, eps=self.eps)
            cosine = torch.einsum("pad,pbd->pab", text, video)
            return self._output(cosine, math.log(text.size(1)))
