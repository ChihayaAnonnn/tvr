"""Pair-level dual-source uncertainty-aware evidence refinement.

This module deliberately contains no learned parameters.  It derives two local
routing signals from deterministic, complementary feature subspaces and uses
them to decide where a small amount of cross-modal context may update the pair
representation before the original WTI aggregation is applied.

The caller must pass L2-normalized token/frame features and the already
masked-and-normalized WTI token/frame weights.  Returned scores do not include
CLIP's ``logit_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class PairEvidenceRefinerOutput:
    """Output of :class:`DualSourcePairEvidenceRefiner`.

    ``diagnostics`` only contains detached scalar tensors, so retaining this
    object for logging cannot retain the training autograd graph.
    """

    scores: Tensor
    diagnostics: dict[str, Tensor]


class DualSourcePairEvidenceRefiner(nn.Module):
    """Refine pair-conditioned evidence before deterministic WTI scoring.

    Four interleaved feature subsets (``k::4``) form deterministic alignment
    views.  For each text token and video frame, data ambiguity is the mean
    view entropy normalized by the number of valid opposite-modal elements.
    View disagreement is the Jensen--Shannon divergence between the four view
    distributions, normalized by ``log(min(num_views, valid_count))``.

    The detached routing gate is ``(1 - ambiguity) * disagreement``.  It only
    controls how much mean-view cross-modal context is mixed into the original
    full-dimensional representation.  Alignment context and refined WTI
    scores remain differentiable with respect to both input modalities and the
    caller-provided base WTI weights.
    """

    _UPDATE_EPS = 1e-6

    def __init__(
        self,
        *,
        num_views: int = 4,
        lambda_max: float = 0.1,
        query_block_size: int = 16,
        candidate_block_size: int = 32,
        alignment_temperature: float = 0.07,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if num_views != 4:
            raise ValueError(
                "DualSourcePairEvidenceRefiner requires exactly four "
                f"complementary views, got num_views={num_views}"
            )
        if not math.isfinite(lambda_max) or not 0.0 <= lambda_max <= 1.0:
            raise ValueError(
                "lambda_max must be finite and in [0, 1], "
                f"got {lambda_max}"
            )
        if query_block_size <= 0:
            raise ValueError(
                "query_block_size must be positive, "
                f"got {query_block_size}"
            )
        if candidate_block_size <= 0:
            raise ValueError(
                "candidate_block_size must be positive, "
                f"got {candidate_block_size}"
            )
        if (
            not math.isfinite(alignment_temperature)
            or alignment_temperature <= 0.0
        ):
            raise ValueError(
                "alignment_temperature must be finite and positive, "
                f"got {alignment_temperature}"
            )

        self.num_views = num_views
        self.lambda_max = float(lambda_max)
        self.query_block_size = int(query_block_size)
        self.candidate_block_size = int(candidate_block_size)
        self.alignment_temperature = float(alignment_temperature)
        self.use_checkpoint = bool(use_checkpoint)

    def extra_repr(self) -> str:
        return (
            f"num_views={self.num_views}, lambda_max={self.lambda_max}, "
            f"query_block_size={self.query_block_size}, "
            f"candidate_block_size={self.candidate_block_size}, "
            f"alignment_temperature={self.alignment_temperature}, "
            f"use_checkpoint={self.use_checkpoint}"
        )

    def forward(
        self,
        text_token: Tensor,
        frame_token: Tensor,
        attention_mask: Tensor,
        video_mask: Tensor,
        text_weight: Tensor,
        video_weight: Tensor,
    ) -> PairEvidenceRefinerOutput:
        """Return pair-refined WTI scores and detached aggregate diagnostics.

        Args:
            text_token: L2-normalized text features shaped ``[A, T, D]``.
            frame_token: L2-normalized frame features shaped ``[B, V, D]``.
            attention_mask: Binary valid-token mask shaped ``[A, T]``.
            video_mask: Binary valid-frame mask shaped ``[B, V]``.
            text_weight: Masked-softmax base WTI weights shaped ``[A, T]``.
            video_weight: Masked-softmax base WTI weights shaped ``[B, V]``.

        The score has shape ``[A, B]`` and has not been multiplied by CLIP's
        learned logit scale.
        """

        text_valid, video_valid = self._validate_inputs(
            text_token,
            frame_token,
            attention_mask,
            video_mask,
            text_weight,
            video_weight,
        )

        query_rows: list[Tensor] = []
        diagnostic_totals = text_token.new_zeros(6, dtype=torch.float32)
        should_checkpoint = self.use_checkpoint and torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (text_token, frame_token, text_weight, video_weight)
        )

        for query_start in range(0, text_token.size(0), self.query_block_size):
            query_end = min(
                query_start + self.query_block_size, text_token.size(0)
            )
            candidate_scores: list[Tensor] = []
            for candidate_start in range(
                0, frame_token.size(0), self.candidate_block_size
            ):
                candidate_end = min(
                    candidate_start + self.candidate_block_size,
                    frame_token.size(0),
                )
                block_inputs = (
                    text_token[query_start:query_end],
                    frame_token[candidate_start:candidate_end],
                    text_valid[query_start:query_end],
                    video_valid[candidate_start:candidate_end],
                    text_weight[query_start:query_end],
                    video_weight[candidate_start:candidate_end],
                )
                if should_checkpoint:
                    block_score, block_diagnostics = checkpoint(
                        self._compute_block,
                        *block_inputs,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    block_score, block_diagnostics = self._compute_block(
                        *block_inputs
                    )
                candidate_scores.append(block_score)
                diagnostic_totals = (
                    diagnostic_totals + block_diagnostics.float()
                )
            query_rows.append(torch.cat(candidate_scores, dim=1))

        scores = torch.cat(query_rows, dim=0)
        representation_count = diagnostic_totals[5].clamp_min(1.0)
        diagnostics = {
            "data_ambiguity_mean": (
                diagnostic_totals[0] / representation_count
            ).detach(),
            "view_disagreement_mean": (
                diagnostic_totals[1] / representation_count
            ).detach(),
            "gate_mean": (
                diagnostic_totals[2] / representation_count
            ).detach(),
            "representation_update_norm": (
                diagnostic_totals[3] / representation_count
            ).detach(),
            "representation_update_rate": (
                diagnostic_totals[4] / representation_count
            ).detach(),
        }
        return PairEvidenceRefinerOutput(scores=scores, diagnostics=diagnostics)

    def _compute_block(
        self,
        text_token: Tensor,
        frame_token: Tensor,
        text_valid: Tensor,
        video_valid: Tensor,
        text_weight: Tensor,
        video_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute one query/candidate block.

        Activation checkpointing wraps this whole function during training, so
        full-dimensional pair contexts are recomputed rather than retained for
        every block until backward.
        """

        view_similarities = []
        normalize_eps = max(
            1e-6, float(torch.finfo(text_token.dtype).eps)
        )
        for view_index in range(self.num_views):
            # Build alignment views in FP32.  In FP16 the default normalize
            # epsilon (1e-12) underflows to zero, so a valid globally
            # normalized feature whose current interleaved subspace is all
            # zeros would otherwise produce 0 / 0 and poison the full pair.
            # The input-dtype floor also keeps the derivative at a zero/tiny
            # subspace representable when it is cast back to low precision.
            text_view = F.normalize(
                text_token[..., view_index :: self.num_views].float(),
                dim=-1,
                eps=normalize_eps,
            )
            frame_view = F.normalize(
                frame_token[..., view_index :: self.num_views].float(),
                dim=-1,
                eps=normalize_eps,
            )
            view_similarities.append(
                torch.einsum("qtd,cvd->qctv", text_view, frame_view)
            )
        # [Q, C, K, T, V]; only a pair block is materialized.
        view_logits = torch.stack(view_similarities, dim=2)
        view_logits = view_logits / self.alignment_temperature

        text_to_video = torch.softmax(
            view_logits.masked_fill(
                ~video_valid[None, :, None, None, :],
                torch.finfo(view_logits.dtype).min,
            ),
            dim=-1,
        )
        video_to_text = torch.softmax(
            view_logits.masked_fill(
                ~text_valid[:, None, None, :, None],
                torch.finfo(view_logits.dtype).min,
            ),
            dim=-2,
        )

        text_view_entropy = self._entropy(text_to_video, dim=-1)
        frame_view_entropy = self._entropy(video_to_text, dim=-2)
        mean_text_alignment = text_to_video.mean(dim=2)
        mean_frame_alignment = video_to_text.mean(dim=2)

        valid_frame_count = video_valid.sum(dim=-1)
        valid_text_count = text_valid.sum(dim=-1)
        text_ambiguity = self._normalized_mean_entropy(
            text_view_entropy,
            valid_frame_count[None, :, None],
        )
        frame_ambiguity = self._normalized_mean_entropy(
            frame_view_entropy,
            valid_text_count[:, None, None],
        )
        text_disagreement = self._normalized_js_divergence(
            self._entropy(mean_text_alignment, dim=-1),
            text_view_entropy,
            valid_frame_count[None, :, None],
        )
        frame_disagreement = self._normalized_js_divergence(
            self._entropy(mean_frame_alignment, dim=-2),
            frame_view_entropy,
            valid_text_count[:, None, None],
        )

        text_pair_valid = text_valid[:, None, :]
        frame_pair_valid = video_valid[None, :, :]
        text_gate = (
            (1.0 - text_ambiguity) * text_disagreement
        ).clamp(0.0, 1.0)
        frame_gate = (
            (1.0 - frame_ambiguity) * frame_disagreement
        ).clamp(0.0, 1.0)
        text_gate = text_gate.masked_fill(~text_pair_valid, 0.0).detach()
        frame_gate = frame_gate.masked_fill(~frame_pair_valid, 0.0).detach()

        alignment_dtype = text_token.dtype
        mean_text_alignment = mean_text_alignment.to(alignment_dtype)
        mean_frame_alignment = mean_frame_alignment.to(alignment_dtype)
        text_context = torch.einsum(
            "qctv,cvd->qctd", mean_text_alignment, frame_token
        )
        frame_context = torch.einsum(
            "qctv,qtd->qcvd", mean_frame_alignment, text_token
        )

        base_text = text_token[:, None, :, :]
        base_frame = frame_token[None, :, :, :]
        refined_text = F.normalize(
            base_text
            + self.lambda_max
            * text_gate.to(alignment_dtype).unsqueeze(-1)
            * (text_context - base_text),
            dim=-1,
            eps=normalize_eps,
        )
        refined_frame = F.normalize(
            base_frame
            + self.lambda_max
            * frame_gate.to(alignment_dtype).unsqueeze(-1)
            * (frame_context - base_frame),
            dim=-1,
            eps=normalize_eps,
        )

        refined_similarity = torch.einsum(
            "qctd,qcvd->qctv", refined_text, refined_frame
        )
        pair_valid = (
            text_valid[:, None, :, None]
            & video_valid[None, :, None, :]
        )
        refined_similarity = refined_similarity.masked_fill(
            ~pair_valid, torch.finfo(refined_similarity.dtype).min
        )
        text_to_video_score = refined_similarity.max(dim=-1).values
        text_to_video_score = text_to_video_score.masked_fill(
            ~text_pair_valid, 0.0
        )
        text_to_video_score = torch.einsum(
            "qct,qt->qc", text_to_video_score, text_weight
        )
        video_to_text_score = refined_similarity.max(dim=-2).values
        video_to_text_score = video_to_text_score.masked_fill(
            ~frame_pair_valid, 0.0
        )
        video_to_text_score = torch.einsum(
            "qcv,cv->qc", video_to_text_score, video_weight
        )
        score = (text_to_video_score + video_to_text_score) / 2.0

        text_update_norm = torch.linalg.vector_norm(
            (refined_text - base_text).float(), dim=-1
        )
        frame_update_norm = torch.linalg.vector_norm(
            (refined_frame - base_frame).float(), dim=-1
        )
        text_valid_float = text_pair_valid.to(dtype=torch.float32)
        frame_valid_float = frame_pair_valid.to(dtype=torch.float32)
        representation_count = (
            text_valid_float.sum() * frame_token.size(0)
            + frame_valid_float.sum() * text_token.size(0)
        )
        diagnostic_sums = torch.stack(
            (
                (text_ambiguity.float() * text_valid_float).sum()
                + (frame_ambiguity.float() * frame_valid_float).sum(),
                (text_disagreement.float() * text_valid_float).sum()
                + (frame_disagreement.float() * frame_valid_float).sum(),
                (text_gate.float() * text_valid_float).sum()
                + (frame_gate.float() * frame_valid_float).sum(),
                (text_update_norm * text_valid_float).sum()
                + (frame_update_norm * frame_valid_float).sum(),
                (
                    (text_update_norm > self._UPDATE_EPS).float()
                    * text_valid_float
                ).sum()
                + (
                    (frame_update_norm > self._UPDATE_EPS).float()
                    * frame_valid_float
                ).sum(),
                representation_count,
            )
        ).detach()
        return score, diagnostic_sums

    @staticmethod
    def _entropy(probabilities: Tensor, *, dim: int) -> Tensor:
        tiny = torch.finfo(probabilities.dtype).tiny
        return -(
            probabilities * probabilities.clamp_min(tiny).log()
        ).sum(dim=dim)

    @staticmethod
    def _normalized_mean_entropy(
        view_entropy: Tensor, valid_count: Tensor
    ) -> Tensor:
        denominator = valid_count.clamp_min(2).to(view_entropy.dtype).log()
        normalized = view_entropy.mean(dim=2) / denominator
        normalized = torch.where(
            valid_count > 1, normalized, torch.zeros_like(normalized)
        )
        return normalized.clamp(0.0, 1.0)

    def _normalized_js_divergence(
        self,
        mixture_entropy: Tensor,
        view_entropy: Tensor,
        valid_count: Tensor,
    ) -> Tensor:
        js_divergence = mixture_entropy - view_entropy.mean(dim=2)
        maximum_components = valid_count.clamp_max(self.num_views)
        denominator = maximum_components.clamp_min(2).to(
            js_divergence.dtype
        ).log()
        normalized = js_divergence / denominator
        normalized = torch.where(
            maximum_components > 1,
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized.clamp(0.0, 1.0)

    def _validate_inputs(
        self,
        text_token: Tensor,
        frame_token: Tensor,
        attention_mask: Tensor,
        video_mask: Tensor,
        text_weight: Tensor,
        video_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if text_token.dim() != 3:
            raise ValueError(
                "text_token must be 3D [A,T,D], "
                f"got shape={tuple(text_token.shape)}"
            )
        if frame_token.dim() != 3:
            raise ValueError(
                "frame_token must be 3D [B,V,D], "
                f"got shape={tuple(frame_token.shape)}"
            )
        if text_token.size(0) == 0 or frame_token.size(0) == 0:
            raise ValueError("text and video batch dimensions must be non-empty")
        if text_token.size(-1) != frame_token.size(-1):
            raise ValueError(
                "feature dimensions must match: "
                f"text={text_token.size(-1)} "
                f"video={frame_token.size(-1)}"
            )
        if text_token.size(-1) < self.num_views:
            raise ValueError(
                "feature dimension must be at least num_views: "
                f"D={text_token.size(-1)} num_views={self.num_views}"
            )
        if text_token.device != frame_token.device:
            raise ValueError(
                "text and video features must share a device: "
                f"text={text_token.device} video={frame_token.device}"
            )
        if text_token.dtype != frame_token.dtype:
            raise ValueError(
                "text and video features must share a dtype: "
                f"text={text_token.dtype} video={frame_token.dtype}"
            )
        if not text_token.is_floating_point():
            raise ValueError(
                "text and video features must use a floating dtype, "
                f"got {text_token.dtype}"
            )
        if not bool(torch.isfinite(text_token).all().item()) or not bool(
            torch.isfinite(frame_token).all().item()
        ):
            raise ValueError("text and video features must be finite")

        expected_text_shape = text_token.shape[:2]
        expected_video_shape = frame_token.shape[:2]
        if attention_mask.shape != expected_text_shape:
            raise ValueError(
                f"attention_mask shape={tuple(attention_mask.shape)} "
                f"expected={tuple(expected_text_shape)}"
            )
        if video_mask.shape != expected_video_shape:
            raise ValueError(
                f"video_mask shape={tuple(video_mask.shape)} "
                f"expected={tuple(expected_video_shape)}"
            )
        if text_weight.shape != expected_text_shape:
            raise ValueError(
                f"text_weight shape={tuple(text_weight.shape)} "
                f"expected={tuple(expected_text_shape)}"
            )
        if video_weight.shape != expected_video_shape:
            raise ValueError(
                f"video_weight shape={tuple(video_weight.shape)} "
                f"expected={tuple(expected_video_shape)}"
            )
        for name, tensor in (
            ("attention_mask", attention_mask),
            ("video_mask", video_mask),
            ("text_weight", text_weight),
            ("video_weight", video_weight),
        ):
            if tensor.device != text_token.device:
                raise ValueError(
                    f"{name} device={tensor.device} does not match "
                    f"feature device={text_token.device}"
                )
        if text_weight.dtype != text_token.dtype:
            raise ValueError(
                f"text_weight dtype={text_weight.dtype} does not match "
                f"feature dtype={text_token.dtype}"
            )
        if video_weight.dtype != frame_token.dtype:
            raise ValueError(
                f"video_weight dtype={video_weight.dtype} does not match "
                f"feature dtype={frame_token.dtype}"
            )
        if not bool(
            ((attention_mask == 0) | (attention_mask == 1)).all().item()
        ):
            raise ValueError("attention_mask must be binary with values 0 or 1")
        if not bool(((video_mask == 0) | (video_mask == 1)).all().item()):
            raise ValueError("video_mask must be binary with values 0 or 1")

        text_valid = attention_mask.to(dtype=torch.bool)
        video_valid = video_mask.to(dtype=torch.bool)
        empty_text = (~text_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
        empty_video = (~video_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
        if empty_text.numel():
            raise ValueError(
                "no valid text token at batch "
                f"indices={empty_text.tolist()}"
            )
        if empty_video.numel():
            raise ValueError(
                "no valid video frame at batch "
                f"indices={empty_video.tolist()}"
            )
        for name, weight, valid in (
            ("text_weight", text_weight, text_valid),
            ("video_weight", video_weight, video_valid),
        ):
            if not bool(torch.isfinite(weight).all().item()):
                raise ValueError(f"{name} must be finite")
            if bool((weight < 0).any().item()):
                raise ValueError(f"{name} must be non-negative")
            if bool((weight.masked_select(~valid) != 0).any().item()):
                raise ValueError(f"{name} must be zero at masked positions")
            expected_sum = torch.ones_like(weight.sum(dim=-1))
            tolerance = 5e-3 if weight.dtype in (torch.float16, torch.bfloat16) else 1e-5
            if not torch.allclose(
                weight.sum(dim=-1),
                expected_sum,
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError(f"{name} must sum to one over valid positions")
        return text_valid, video_valid
