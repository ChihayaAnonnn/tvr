"""Deterministic mean recall followed by bounded stochastic pair reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modules.stochastic_prototype_ranking import (
    BidirectionalSoftPrototypeMatcher,
)


@dataclass(frozen=True)
class TopRRetrievalOutput:
    """Directional retrieval scores and shared pair diagnostics."""

    text_to_video_logits: torch.Tensor
    video_to_text_logits: torch.Tensor
    mean_logits: torch.Tensor
    pair_uncertainty: torch.Tensor


def _score_selected_pairs(
    text_indices: torch.Tensor,
    video_indices: torch.Tensor,
    text_samples: torch.Tensor,
    video_samples: torch.Tensor,
    matcher: BidirectionalSoftPrototypeMatcher,
    pair_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    uncertainties = []
    for start in range(0, text_indices.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, text_indices.numel())
        match = matcher.score_pairs(
            text_samples[text_indices[start:end]],
            video_samples[video_indices[start:end]],
        )
        scores.append(match.logits)
        uncertainties.append(match.pair_uncertainty)
    return torch.cat(scores), torch.cat(uncertainties)


def rerank_top_r(
    deterministic_logits: torch.Tensor,
    text_mean: torch.Tensor,
    video_mean: torch.Tensor,
    text_samples: torch.Tensor,
    video_samples: torch.Tensor,
    matcher: BidirectionalSoftPrototypeMatcher,
    *,
    top_r: int,
    deterministic_temperature: float,
    probabilistic_temperature: float,
    probabilistic_weight: float,
    pair_chunk_size: int,
) -> TopRRetrievalOutput:
    """Recall candidates by normalized means and rerank aligned pairs only."""

    if isinstance(top_r, bool) or not isinstance(top_r, int) or top_r < 0:
        raise ValueError("top_r must be a nonnegative integer")
    if (
        isinstance(pair_chunk_size, bool)
        or not isinstance(pair_chunk_size, int)
        or pair_chunk_size <= 0
    ):
        raise ValueError("pair_chunk_size must be a positive integer")
    for name, value in (
        ("deterministic_temperature", deterministic_temperature),
        ("probabilistic_temperature", probabilistic_temperature),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    mean_logits = F.normalize(text_mean, dim=-1) @ F.normalize(
        video_mean, dim=-1
    ).T
    pair_uncertainty = torch.full_like(mean_logits, torch.nan)
    if top_r == 0:
        return TopRRetrievalOutput(
            text_to_video_logits=mean_logits,
            video_to_text_logits=mean_logits,
            mean_logits=mean_logits,
            pair_uncertainty=pair_uncertainty,
        )

    text_count, video_count = mean_logits.shape
    t2v_count = min(top_r, video_count)
    v2t_count = min(top_r, text_count)

    t2v_video_indices = mean_logits.topk(t2v_count, dim=1).indices
    t2v_text_indices = (
        torch.arange(text_count, device=mean_logits.device)
        .unsqueeze(1)
        .expand(-1, t2v_count)
    )
    v2t_text_indices = mean_logits.topk(v2t_count, dim=0).indices
    v2t_video_indices = (
        torch.arange(video_count, device=mean_logits.device)
        .unsqueeze(0)
        .expand(v2t_count, -1)
    )

    flat_t2v_text = t2v_text_indices.reshape(-1)
    flat_t2v_video = t2v_video_indices.reshape(-1)
    flat_v2t_text = v2t_text_indices.reshape(-1)
    flat_v2t_video = v2t_video_indices.reshape(-1)
    all_text_indices = torch.cat((flat_t2v_text, flat_v2t_text))
    all_video_indices = torch.cat((flat_t2v_video, flat_v2t_video))
    probability, uncertainty = _score_selected_pairs(
        all_text_indices,
        all_video_indices,
        text_samples,
        video_samples,
        matcher,
        pair_chunk_size,
    )

    deterministic = deterministic_logits[all_text_indices, all_video_indices]
    selected_final = (
        deterministic / deterministic_temperature
        + probabilistic_weight * probability / probabilistic_temperature
    )

    t2v_pair_count = flat_t2v_text.numel()
    text_to_video_logits = torch.full_like(selected_final.new_empty(mean_logits.shape), float("-inf"))
    video_to_text_logits = torch.full_like(text_to_video_logits, float("-inf"))
    text_to_video_logits[flat_t2v_text, flat_t2v_video] = selected_final[
        :t2v_pair_count
    ]
    video_to_text_logits[flat_v2t_text, flat_v2t_video] = selected_final[
        t2v_pair_count:
    ]
    pair_uncertainty[all_text_indices, all_video_indices] = uncertainty.to(
        pair_uncertainty.dtype
    )

    return TopRRetrievalOutput(
        text_to_video_logits=text_to_video_logits,
        video_to_text_logits=video_to_text_logits,
        mean_logits=mean_logits,
        pair_uncertainty=pair_uncertainty,
    )
