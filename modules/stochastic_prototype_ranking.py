"""Bidirectional matching between stochastic text and video prototypes."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from prob_models.reparameterized_distribution import (
    DistributionOutput,
    ReparameterizedDistributionHead,
    antithetic_standard_normal,
)


@dataclass(frozen=True)
class PrototypeMatchOutput:
    """Scores and uncertainty for stochastic prototype matches."""

    logits: torch.Tensor
    pair_uncertainty: torch.Tensor
    text_prototype_scores: torch.Tensor
    video_prototype_scores: torch.Tensor
    stochastic_pair_scores: torch.Tensor


@dataclass(frozen=True)
class StochasticRankOutput:
    """Ranking loss and hard-negative diagnostics for one retrieval direction."""

    loss: torch.Tensor
    inversion_probability: torch.Tensor
    negative_indices: torch.Tensor


@dataclass(frozen=True)
class RSPROutput:
    """Probability distributions and their stochastic retrieval scores."""

    text_distribution: DistributionOutput
    video_distribution: DistributionOutput
    probabilistic_logits: torch.Tensor
    pair_uncertainty: torch.Tensor
    stochastic_pair_scores: torch.Tensor
    anchor_kl: torch.Tensor


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return value


def _positive_even_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
        raise ValueError(f"{name} must be a positive even integer")
    return value


def _validate_sampling_request(sample_count: int, mean_only: bool) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    if mean_only:
        if sample_count != 1:
            raise ValueError("mean-only sampling requires sample_count=1")
        return
    _positive_even_integer(sample_count, "sample_count")


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


class StochasticRankLoss(nn.Module):
    """Mine cross-group negatives for multi-positive stochastic ranking."""

    def __init__(
        self,
        temperature: float = 0.07,
        *,
        hard_negative_count: int = 8,
        margin: float = 0.0,
    ):
        super().__init__()
        self.temperature = _positive_finite(temperature, "temperature")
        if isinstance(hard_negative_count, bool) or not isinstance(hard_negative_count, int):
            raise ValueError("hard_negative_count must be a positive integer")
        if hard_negative_count <= 0:
            raise ValueError("hard_negative_count must be a positive integer")
        self.hard_negative_count = hard_negative_count
        self.margin = _nonnegative_finite(margin, "margin")

    @staticmethod
    def _validate_inputs(
        stochastic_scores: torch.Tensor,
        group_ids: torch.Tensor,
        mining_logits: torch.Tensor,
    ) -> None:
        if stochastic_scores.ndim != 3:
            raise ValueError("stochastic_scores must have shape [batch, batch, samples]")
        batch_size, candidate_count, sample_count = stochastic_scores.shape
        if batch_size != candidate_count:
            raise ValueError("stochastic_scores must be square with shape [batch, batch, samples]")
        if batch_size == 0:
            raise ValueError("stochastic_scores must have a nonempty batch")
        if sample_count == 0:
            raise ValueError("stochastic_scores must have a nonempty samples dimension")
        if not stochastic_scores.is_floating_point():
            raise ValueError("stochastic_scores must use a floating-point dtype")

        if group_ids.ndim != 1 or group_ids.size(0) != batch_size:
            raise ValueError("group_ids must have shape [batch]")
        if group_ids.dtype == torch.bool or group_ids.is_floating_point() or group_ids.is_complex():
            raise ValueError("group_ids must use an integer dtype")

        if mining_logits.ndim != 2 or mining_logits.shape != (batch_size, batch_size):
            raise ValueError("mining_logits must have shape [batch, batch]")
        if not mining_logits.is_floating_point():
            raise ValueError("mining_logits must use a floating-point dtype")
        if stochastic_scores.device != group_ids.device or stochastic_scores.device != mining_logits.device:
            raise ValueError("stochastic_scores, group_ids, and mining_logits must use the same device")

    def forward(
        self,
        stochastic_scores: torch.Tensor,
        group_ids: torch.Tensor,
        mining_logits: torch.Tensor,
    ) -> StochasticRankOutput:
        """Compute one directional loss with every same-group candidate as positive."""

        self._validate_inputs(stochastic_scores, group_ids, mining_logits)
        positive_mask = group_ids[:, None].eq(group_ids[None, :])
        negative_mask = ~positive_mask
        if not negative_mask.any(dim=1).all():
            raise ValueError("each query must have at least one valid negative candidate")

        positive_count = positive_mask.sum(dim=1, keepdim=True)
        positive_scores = (stochastic_scores * positive_mask.unsqueeze(-1)).sum(dim=1) / positive_count

        candidate_count = int(negative_mask.sum(dim=1).min().item())
        negative_count = min(self.hard_negative_count, candidate_count)
        mining_scores = mining_logits.detach()
        if mining_scores.dtype in {torch.float16, torch.bfloat16}:
            mining_scores = mining_scores.float()
        negative_indices = (
            mining_scores.masked_fill(~negative_mask, float("-inf"))
            .topk(
                negative_count,
                dim=1,
            )
            .indices
        )
        gather_index = negative_indices.unsqueeze(-1).expand(-1, -1, stochastic_scores.size(-1))
        negative_scores = stochastic_scores.gather(dim=1, index=gather_index)
        difference = negative_scores - positive_scores.unsqueeze(1)
        loss = F.softplus((difference + self.margin) / self.temperature).mean()
        inversion_probability = torch.sigmoid(difference / self.temperature).mean(dim=-1)
        return StochasticRankOutput(
            loss=loss,
            inversion_probability=inversion_probability,
            negative_indices=negative_indices,
        )

    def bidirectional(
        self,
        stochastic_scores: torch.Tensor,
        group_ids: torch.Tensor,
        mining_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, StochasticRankOutput, StochasticRankOutput]:
        """Average directional losses after independently mining each direction."""

        text_to_video = self(stochastic_scores, group_ids, mining_logits)
        video_to_text = self(
            stochastic_scores.transpose(0, 1),
            group_ids,
            mining_logits.transpose(0, 1),
        )
        return 0.5 * (text_to_video.loss + video_to_text.loss), text_to_video, video_to_text


class RSPRCore(nn.Module):
    """Compose probabilistic embedding heads, matching, and ranking support."""

    def __init__(
        self,
        dim: int,
        sample_count: int = 4,
        eval_sample_count: int = 8,
        match_temperature: float = 0.07,
        rank_temperature: float = 0.07,
        hard_negative_count: int = 8,
        prior_std: float = 0.1,
        hard_max: bool = False,
        eval_seed: int = 0,
        *,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if sample_count != 1:
            _positive_even_integer(sample_count, "sample_count")
        eval_sample_count = _positive_even_integer(eval_sample_count, "eval_sample_count")
        if isinstance(eval_seed, bool) or not isinstance(eval_seed, int):
            raise ValueError("eval_seed must be an integer")

        self.sample_count = sample_count
        self.eval_sample_count = eval_sample_count
        self.text_distribution = ReparameterizedDistributionHead(
            dim=dim,
            hidden_dim=hidden_dim,
            prior_std=prior_std,
            eps=eps,
        )
        self.video_distribution = ReparameterizedDistributionHead(
            dim=dim,
            hidden_dim=hidden_dim,
            prior_std=prior_std,
            eps=eps,
        )
        self.matcher = BidirectionalSoftPrototypeMatcher(
            temperature=match_temperature,
            hard_max=hard_max,
            eps=eps,
        )
        self.rank_loss = StochasticRankLoss(
            temperature=rank_temperature,
            hard_negative_count=hard_negative_count,
        )

        self.register_buffer(
            "fixed_text_noise",
            self._fixed_noise(dim, eval_sample_count, eval_seed),
        )
        self.register_buffer(
            "fixed_video_noise",
            self._fixed_noise(dim, eval_sample_count, eval_seed + 1),
        )

    @staticmethod
    def _fixed_noise(dim: int, sample_count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return antithetic_standard_normal(
            batch_size=1,
            sample_count=sample_count,
            dim=dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=generator,
        ).squeeze(0)

    def _evaluation_noise(
        self,
        fixed_noise: torch.Tensor,
        *,
        batch_size: int,
        sample_count: int,
    ) -> torch.Tensor:
        return fixed_noise[:sample_count].unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        *,
        sample_count: int,
        mean_only: bool = False,
        detach_samples: bool = False,
        text_noise: torch.Tensor | None = None,
        video_noise: torch.Tensor | None = None,
    ) -> RSPROutput:
        """Return stochastic text-video scores and their source distributions."""

        _validate_sampling_request(sample_count, mean_only)
        if not self.training and sample_count > self.eval_sample_count:
            raise ValueError("sample_count cannot exceed eval_sample_count during evaluation")

        if not self.training:
            if text_noise is None:
                text_noise = self._evaluation_noise(
                    self.fixed_text_noise,
                    batch_size=text_tokens.size(0),
                    sample_count=sample_count,
                )
            if video_noise is None:
                video_noise = self._evaluation_noise(
                    self.fixed_video_noise,
                    batch_size=video_tokens.size(0),
                    sample_count=sample_count,
                )

        text_distribution = self.text_distribution(
            text_tokens,
            text_mask,
            sample_count=sample_count,
            noise=text_noise,
            mean_only=mean_only,
            detach_samples=detach_samples,
        )
        video_distribution = self.video_distribution(
            video_tokens,
            video_mask,
            sample_count=sample_count,
            noise=video_noise,
            mean_only=mean_only,
            detach_samples=detach_samples,
        )
        match = self.matcher(text_distribution.samples, video_distribution.samples)
        return RSPROutput(
            text_distribution=text_distribution,
            video_distribution=video_distribution,
            probabilistic_logits=match.logits,
            pair_uncertainty=match.pair_uncertainty,
            stochastic_pair_scores=match.stochastic_pair_scores,
            anchor_kl=0.5 * (text_distribution.anchor_kl + video_distribution.anchor_kl),
        )
