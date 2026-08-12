"""Candidate-conditioned, detached retrieval-failure risk sidecar."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from prob_models.retrieval_risk import (
    _oriented_scores,
    _pairwise_uncertainty_ranking_loss,
    _robust_rank_normalize,
    _validate_positive_mask,
    retrieval_hardness,
    score_distribution_features,
)


def _validate_embeddings(
    query_means: torch.Tensor,
    candidate_means: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.as_tensor(query_means).detach().float()
    candidate = torch.as_tensor(candidate_means).detach().float()
    if query.ndim != 2 or candidate.ndim != 2:
        raise ValueError("query_means and candidate_means must be 2-D")
    if query.size(0) != scores.size(0):
        raise ValueError("query_means rows must match the number of queries")
    if candidate.size(0) != scores.size(1):
        raise ValueError("candidate_means rows must match the number of candidates")
    if query.size(1) != candidate.size(1):
        raise ValueError("query_means and candidate_means must share embedding width")
    if not torch.isfinite(query).all() or not torch.isfinite(candidate).all():
        raise ValueError("query_means and candidate_means must be finite")
    return query, candidate


def contextual_topk_features(
    logits: torch.Tensor,
    query_means: torch.Tensor,
    candidate_means: torch.Tensor,
    direction: str,
    topk: int = 16,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Build detached, direction-oriented contextual retrieval features."""

    if topk <= 0:
        raise ValueError("topk must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = _oriented_scores(logits.detach(), direction).float()
    query, candidate = _validate_embeddings(query_means, candidate_means, scores)
    effective_k = min(topk, scores.size(1))
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = (scores - mean) / std
    top_values, top_indices = normalized.topk(effective_k, dim=1)
    top_valid = torch.ones_like(top_values, dtype=torch.bool)
    if effective_k < topk:
        width = topk - effective_k
        top_indices = torch.cat(
            (top_indices, top_indices[:, -1:].expand(-1, width)), dim=1
        )
        top_values = torch.cat(
            (top_values, top_values.new_zeros(top_values.size(0), width)), dim=1
        )
        top_valid = torch.cat(
            (top_valid, top_valid.new_zeros(top_valid.size(0), width)), dim=1
        )

    selected = candidate[top_indices]
    attention_logits = top_values.masked_fill(~top_valid, -torch.inf) / temperature
    attention = torch.softmax(attention_logits, dim=1)
    context = torch.sum(attention.unsqueeze(-1) * selected, dim=1)
    gaps = (top_values[:, :1] - top_values).masked_fill(~top_valid, 0.0)
    if effective_k == 1:
        entropy = attention.new_zeros(attention.size(0))
    else:
        entropy = -(
            attention * attention.clamp_min(1e-12).log()
        ).sum(dim=1) / math.log(effective_k)
    score_features = score_distribution_features(logits.detach(), direction, temperature)
    direction_code = score_features.new_tensor(
        [1.0, 0.0] if direction == "t2v" else [0.0, 1.0]
    ).expand(scores.size(0), -1)

    return {
        "query_mean": query,
        "top1_mean": selected[:, 0],
        "context_mean": context,
        "score_features": score_features,
        "normalized_scores": top_values,
        "score_gaps": gaps,
        "top_indices": top_indices,
        "top_valid": top_valid,
        "attention": attention,
        "normalized_entropy": score_features[:, 4],
        "topk_entropy": entropy,
        "direction_code": direction_code,
    }


class ContextualRetrievalRiskSidecar(nn.Module):
    """A shared failure logit conditioned on query and top-ranked candidates."""

    def __init__(
        self,
        embed_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 32,
        topk: int = 16,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or projection_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding and hidden dimensions must be positive")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embed_dim = int(embed_dim)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.text_projection = nn.Linear(embed_dim, projection_dim)
        self.video_projection = nn.Linear(embed_dim, projection_dim)
        self.network = nn.Sequential(
            nn.Linear(3 * projection_dim + 8 + 2 * topk + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        logits: torch.Tensor,
        query_means: torch.Tensor,
        candidate_means: torch.Tensor,
        direction: str,
    ) -> dict[str, torch.Tensor]:
        features = contextual_topk_features(
            logits,
            query_means,
            candidate_means,
            direction,
            self.topk,
            self.temperature,
        )
        if features["query_mean"].size(1) != self.embed_dim:
            raise ValueError("embedding width must match embed_dim")

        if direction == "t2v":
            query_projection = self.text_projection(features["query_mean"])
            candidate_projection = self.video_projection(features["top1_mean"])
            context_projection = self.video_projection(features["context_mean"])
        else:
            query_projection = self.video_projection(features["query_mean"])
            candidate_projection = self.text_projection(features["top1_mean"])
            context_projection = self.text_projection(features["context_mean"])

        network_features = torch.cat(
            (
                query_projection,
                candidate_projection,
                context_projection,
                features["score_features"],
                features["normalized_scores"],
                features["score_gaps"],
                features["direction_code"],
            ),
            dim=1,
        )
        failure_logit = self.network(network_features).squeeze(1)
        return {
            **features,
            "features": network_features,
            "failure_logit": failure_logit,
            "failure_risk": torch.sigmoid(failure_logit),
        }


def _direction_objective(
    sidecar: ContextualRetrievalRiskSidecar,
    logits: torch.Tensor,
    query_means: torch.Tensor,
    candidate_means: torch.Tensor,
    positive_mask: torch.Tensor,
    direction: str,
    hardness_temperature: float,
    fit_weight: float,
    rank_weight: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    scores = _oriented_scores(logits.detach(), direction)
    mask = _validate_positive_mask(
        scores, positive_mask if direction == "t2v" else positive_mask.T
    )
    output = sidecar(logits, query_means, candidate_means, direction)
    top_index = scores.argmax(dim=1, keepdim=True)
    failure = (~mask.gather(1, top_index).squeeze(1)).float()
    num_positive = failure.sum()
    num_negative = failure.numel() - num_positive
    positive_weight = (num_negative / num_positive.clamp_min(1.0)).clamp(0.25, 4.0)
    bce = F.binary_cross_entropy_with_logits(
        output["failure_logit"], failure, pos_weight=positive_weight.detach()
    )
    hardness = retrieval_hardness(scores, mask, hardness_temperature)
    fit = F.smooth_l1_loss(
        output["failure_risk"], _robust_rank_normalize(hardness)
    )
    ranking = _pairwise_uncertainty_ranking_loss(
        output["failure_risk"], hardness
    )
    direction_loss = bce + fit_weight * fit + rank_weight * ranking
    valid_ranking = (hardness[:, None] - hardness[None, :]).abs() > 0.05
    risk = output["failure_risk"]
    return direction_loss, {
        **output,
        "failure": failure,
        "hardness": hardness,
        "bce": bce,
        "fit": fit,
        "rank": ranking,
        "positive_weight": positive_weight.detach(),
        "failure_rate": failure.mean(),
        "prediction_mean": risk.detach().mean(),
        "prediction_std": risk.detach().std(unbiased=False),
        "target_mean": failure.mean(),
        "target_std": failure.std(unbiased=False),
        "valid_ranking_fraction": valid_ranking.float().mean(),
        "entropy": output["topk_entropy"].detach().mean(),
        "saturation_fraction": (
            (risk.detach() <= 0.05) | (risk.detach() >= 0.95)
        ).float().mean(),
    }


def contextual_retrieval_risk_objective(
    sidecar: ContextualRetrievalRiskSidecar,
    logits: torch.Tensor,
    text_means: torch.Tensor,
    video_means: torch.Tensor,
    positive_mask: torch.Tensor,
    hardness_temperature: float = 1.0,
    fit_weight: float = 0.25,
    rank_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Train symmetric candidate-conditioned failure risk without retriever gradients."""

    if hardness_temperature <= 0:
        raise ValueError("hardness_temperature must be positive")
    if fit_weight < 0 or rank_weight < 0:
        raise ValueError("fit_weight and rank_weight must be non-negative")
    scores = _oriented_scores(logits.detach(), "t2v")
    mask = _validate_positive_mask(scores, positive_mask)
    t2v_loss, t2v = _direction_objective(
        sidecar,
        logits,
        text_means,
        video_means,
        mask,
        "t2v",
        hardness_temperature,
        fit_weight,
        rank_weight,
    )
    v2t_loss, v2t = _direction_objective(
        sidecar,
        logits,
        video_means,
        text_means,
        mask,
        "v2t",
        hardness_temperature,
        fit_weight,
        rank_weight,
    )
    return 0.5 * (t2v_loss + v2t_loss), {
        "bce": 0.5 * (t2v["bce"] + v2t["bce"]),
        "fit": 0.5 * (t2v["fit"] + v2t["fit"]),
        "rank": 0.5 * (t2v["rank"] + v2t["rank"]),
        "t2v": t2v,
        "v2t": v2t,
    }
