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


_ALL_CONTEXTUAL_FEATURE_BLOCKS = frozenset(
    {
        "query_semantics", "top1_semantics", "context_semantics",
        "score_distribution", "topk_scores", "score_gaps", "direction",
    }
)
# Feature sets containing this block widen the network; every other set keeps
# the pre-SAP capacity so the ablations stay capacity-matched among themselves.
SAP_UNCERTAINTY_BLOCK = "sap_uncertainty"
CONTEXTUAL_FEATURE_SETS = {
    "full": _ALL_CONTEXTUAL_FEATURE_BLOCKS,
    "score_only": frozenset(
        {"score_distribution", "topk_scores", "score_gaps", "direction"}
    ),
    "semantic_only": frozenset(
        {"query_semantics", "top1_semantics", "direction"}
    ),
    "no_context": _ALL_CONTEXTUAL_FEATURE_BLOCKS - {"context_semantics"},
    "no_score_geometry": frozenset(
        {"query_semantics", "top1_semantics", "context_semantics", "direction"}
    ),
    # `full` plus SAP's own uncertainty heads: the treatment arm.
    "full_sap": _ALL_CONTEXTUAL_FEATURE_BLOCKS | {SAP_UNCERTAINTY_BLOCK},
    # SAP uncertainty in isolation: does the probe's own uncertainty predict
    # retrieval failure without any score-geometry or semantic evidence?
    "sap_only": frozenset({SAP_UNCERTAINTY_BLOCK, "direction"}),
}
SAP_CONTEXTUAL_FEATURE_SETS = frozenset(
    name
    for name, blocks in CONTEXTUAL_FEATURE_SETS.items()
    if SAP_UNCERTAINTY_BLOCK in blocks
)
# `sap_uncertainty` is appended LAST so that every pre-existing block keeps its
# offset. When `use_sap_uncertainty=False` the block is dropped from the layout
# entirely and the feature vector is bit-identical to the pre-SAP sidecar.
CONTEXTUAL_FEATURE_BLOCK_LAYOUT = (
    ("query_semantics", "projection_dim"),
    ("top1_semantics", "projection_dim"),
    ("context_semantics", "projection_dim"),
    ("score_distribution", "score_distribution"),
    ("topk_scores", "topk"),
    ("score_gaps", "topk"),
    ("direction", "direction"),
    ("sap_uncertainty", "sap_uncertainty"),
)

# Per-video SAP uncertainty descriptor width, and the three slots it occupies
# in the feature vector (query-side, top-1 candidate, top-k context).
SAP_UNCERTAINTY_DIM = 5
SAP_UNCERTAINTY_SLOTS = 3
SAP_UNCERTAINTY_FIELDS = (
    "u_mode",
    "log_evidence",
    "modal_entropy",
    "epistemic_scale",
    "aleatoric_scale",
)


def _contextual_feature_slices(
    projection_dim: int,
    topk: int,
    sap_uncertainty_dim: int = 0,
) -> dict[str, slice]:
    widths = {
        "projection_dim": projection_dim,
        "score_distribution": 8,
        "topk": topk,
        "direction": 2,
        "sap_uncertainty": SAP_UNCERTAINTY_SLOTS * sap_uncertainty_dim,
    }
    offset = 0
    slices = {}
    for block, width_name in CONTEXTUAL_FEATURE_BLOCK_LAYOUT:
        width = widths[width_name]
        if width == 0:
            continue
        slices[block] = slice(offset, offset + width)
        offset += width
    return slices


def sap_uncertainty_descriptor(
    u_mode: torch.Tensor,
    alpha_dir: torch.Tensor,
    epistemic_cont: torch.Tensor,
    logsigma: torch.Tensor,
) -> torch.Tensor:
    """Reduce raw SAP uncertainty heads to a per-video descriptor `[N, 5]`.

    Fields, in order, match `SAP_UNCERTAINTY_FIELDS`:
      u_mode          discrete modal uncertainty K/S, already in (0, 1]
      log_evidence    log of the Dirichlet total evidence S = sum(alpha)
      modal_entropy   entropy of alpha/S, normalized to [0, 1]
      epistemic_scale RMS of the continuous epistemic tensor
      aleatoric_scale mean mixture log-sigma
    """

    u = torch.as_tensor(u_mode).detach().float().reshape(-1)
    alpha = torch.as_tensor(alpha_dir).detach().float()
    epistemic = torch.as_tensor(epistemic_cont).detach().float()
    sigma = torch.as_tensor(logsigma).detach().float()
    if alpha.ndim != 2:
        raise ValueError("alpha_dir must be [N, K]")
    if epistemic.ndim != 3:
        raise ValueError("epistemic_cont must be [N, K, D]")
    if sigma.ndim != 2:
        raise ValueError("logsigma must be [N, D]")
    if not (u.size(0) == alpha.size(0) == epistemic.size(0) == sigma.size(0)):
        raise ValueError("SAP uncertainty heads must share the video axis")

    num_anchors = alpha.size(1)
    strength = alpha.sum(dim=1).clamp_min(1e-6)
    probs = alpha / strength.unsqueeze(1)
    if num_anchors > 1:
        modal_entropy = -(
            probs * probs.clamp_min(1e-12).log()
        ).sum(dim=1) / math.log(num_anchors)
    else:
        modal_entropy = probs.new_zeros(probs.size(0))
    descriptor = torch.stack(
        (
            u,
            strength.log(),
            modal_entropy,
            epistemic.pow(2).mean(dim=(1, 2)).clamp_min(1e-12).sqrt(),
            sigma.mean(dim=1),
        ),
        dim=1,
    )
    if descriptor.size(1) != SAP_UNCERTAINTY_DIM:
        raise ValueError("SAP uncertainty descriptor width drifted from the spec")
    if not torch.isfinite(descriptor).all():
        raise ValueError("SAP uncertainty descriptor must be finite")
    return descriptor


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
        feature_set: str = "full",
        use_sap_uncertainty: bool = False,
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
        if feature_set not in CONTEXTUAL_FEATURE_SETS:
            raise ValueError(
                f"feature_set must be one of {sorted(CONTEXTUAL_FEATURE_SETS)}"
            )

        self.embed_dim = int(embed_dim)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.feature_set = feature_set
        self.use_sap_uncertainty = bool(use_sap_uncertainty)
        if self.use_sap_uncertainty != (feature_set in SAP_CONTEXTUAL_FEATURE_SETS):
            raise ValueError(
                "use_sap_uncertainty must be enabled exactly for the feature "
                f"sets {sorted(SAP_CONTEXTUAL_FEATURE_SETS)}; got "
                f"feature_set='{feature_set}' with "
                f"use_sap_uncertainty={self.use_sap_uncertainty}"
            )
        self.sap_uncertainty_dim = (
            SAP_UNCERTAINTY_DIM if self.use_sap_uncertainty else 0
        )
        self.active_feature_blocks = tuple(
            sorted(CONTEXTUAL_FEATURE_SETS[feature_set])
        )
        self.feature_slices = _contextual_feature_slices(
            projection_dim, topk, self.sap_uncertainty_dim
        )
        feature_width = max(
            feature_slice.stop for feature_slice in self.feature_slices.values()
        )
        feature_mask = torch.ones(feature_width)
        for block, feature_slice in self.feature_slices.items():
            if block not in CONTEXTUAL_FEATURE_SETS[feature_set]:
                feature_mask[feature_slice] = 0.0
        self.register_buffer("feature_mask", feature_mask, persistent=False)
        self.text_projection = nn.Linear(embed_dim, projection_dim)
        self.video_projection = nn.Linear(embed_dim, projection_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_width, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _check_uncertainty(
        self, tensor: torch.Tensor, rows: int, role: str
    ) -> torch.Tensor:
        value = torch.as_tensor(tensor).detach().float()
        if value.ndim != 2 or value.size(1) != self.sap_uncertainty_dim:
            raise ValueError(
                f"{role}_uncertainty must be [N, {self.sap_uncertainty_dim}]"
            )
        if value.size(0) != rows:
            raise ValueError(
                f"{role}_uncertainty rows must match the number of {role}s"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{role}_uncertainty must be finite")
        return value

    def _sap_uncertainty_block(
        self,
        features: dict[str, torch.Tensor],
        num_candidates: int,
        query_uncertainty: torch.Tensor | None,
        candidate_uncertainty: torch.Tensor | None,
    ) -> torch.Tensor:
        """Assemble the [query | top-1 | context] SAP uncertainty slots.

        SAP uncertainty is a per-VIDEO quantity, so exactly one side carries it
        in each direction: the query side for v2t, the candidate side for t2v.
        The absent side is zero-filled and the direction code tells the network
        which slots are live.
        """

        if query_uncertainty is None and candidate_uncertainty is None:
            raise ValueError(
                "use_sap_uncertainty is enabled but no SAP uncertainty was "
                "supplied for either the query or the candidate axis"
            )
        top_indices = features["top_indices"]
        attention = features["attention"]
        num_queries = top_indices.size(0)
        reference = features["query_mean"]
        zeros = reference.new_zeros(num_queries, self.sap_uncertainty_dim)

        if query_uncertainty is None:
            query_slot = zeros
        else:
            query_slot = self._check_uncertainty(
                query_uncertainty, num_queries, "query"
            ).to(device=reference.device, dtype=reference.dtype)

        if candidate_uncertainty is None:
            top1_slot = zeros
            context_slot = zeros
        else:
            candidate = self._check_uncertainty(
                candidate_uncertainty, num_candidates, "candidate"
            ).to(device=reference.device, dtype=reference.dtype)
            selected = candidate[top_indices]
            top1_slot = selected[:, 0]
            context_slot = torch.sum(
                attention.unsqueeze(-1) * selected, dim=1
            )
        return torch.cat((query_slot, top1_slot, context_slot), dim=1)

    def forward(
        self,
        logits: torch.Tensor,
        query_means: torch.Tensor,
        candidate_means: torch.Tensor,
        direction: str,
        query_uncertainty: torch.Tensor | None = None,
        candidate_uncertainty: torch.Tensor | None = None,
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

        feature_blocks = [
            query_projection,
            candidate_projection,
            context_projection,
            features["score_features"],
            features["normalized_scores"],
            features["score_gaps"],
            features["direction_code"],
        ]
        if self.use_sap_uncertainty:
            sap_block = self._sap_uncertainty_block(
                features,
                int(torch.as_tensor(candidate_means).shape[0]),
                query_uncertainty,
                candidate_uncertainty,
            )
            feature_blocks.append(sap_block)
            features = {**features, "sap_uncertainty": sap_block}
        network_features = torch.cat(feature_blocks, dim=1)
        network_features = network_features * self.feature_mask.to(
            device=network_features.device, dtype=network_features.dtype
        )
        failure_logit = self.network(network_features).squeeze(1)
        return {
            **features,
            "features": network_features,
            "feature_set": self.feature_set,
            "use_sap_uncertainty": self.use_sap_uncertainty,
            "active_feature_blocks": self.active_feature_blocks,
            "feature_block_offsets": {
                block: (feature_slice.start, feature_slice.stop)
                for block, feature_slice in self.feature_slices.items()
            },
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
    query_uncertainty: torch.Tensor | None = None,
    candidate_uncertainty: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    scores = _oriented_scores(logits.detach(), direction)
    mask = _validate_positive_mask(
        scores, positive_mask if direction == "t2v" else positive_mask.T
    )
    output = sidecar(
        logits,
        query_means,
        candidate_means,
        direction,
        query_uncertainty=query_uncertainty,
        candidate_uncertainty=candidate_uncertainty,
    )
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
    video_uncertainty: torch.Tensor | None = None,
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
        # t2v: texts query, videos are the candidates.
        candidate_uncertainty=video_uncertainty,
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
        # v2t: videos query, texts are the candidates and carry no SAP head.
        query_uncertainty=video_uncertainty,
    )
    return 0.5 * (t2v_loss + v2t_loss), {
        "bce": 0.5 * (t2v["bce"] + v2t["bce"]),
        "fit": 0.5 * (t2v["fit"] + v2t["fit"]),
        "rank": 0.5 * (t2v["rank"] + v2t["rank"]),
        "t2v": t2v,
        "v2t": v2t,
    }
