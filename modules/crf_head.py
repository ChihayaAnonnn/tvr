"""CRF training head — composable into UATVR without requiring CLIP weights.

This module holds the ``video_aleatoric`` attribute the plan references and
the paired-forward + ranking-loss orchestration. Split out so it can be
unit-tested independently of the full retrieval model (which needs CLIP
weights and a ~1B parameter compute graph to instantiate).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from uncertainty_modules.aleatoric import AleatoricUncertaintyModule
from uncertainty_modules.losses import crf_ranking_loss
from uncertainty_modules.types import CRFRankingLoss


class CRFTrainingHead(nn.Module):
    """Aleatoric risk head trained on CRF-degraded video pairs.

    The head:

    * Owns a :class:`AleatoricUncertaintyModule` under the attribute name
      ``video_aleatoric`` — the plan explicitly forbids renaming this so
      state-dict compatibility is preserved.
    * Freezes ``video_aleatoric.global_uncertainty_head`` in ``__init__``.
      The training loop's optimizer must consume only trainable parameters
      via ``filter(lambda p: p.requires_grad, ...)`` (or explicit iteration)
      so the frozen head never appears in any param group.
    * Uses ``pooling_temperature=0.0`` by default per the plan; the real
      short-circuit prevents ``0 * -inf → NaN`` on masked positions.
    * Provides :meth:`score_features` for single-pass CRF diagnostics and
      :meth:`compute_ranking_loss` for the training-time paired forward.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pooling_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.video_aleatoric = AleatoricUncertaintyModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pooling_temperature=pooling_temperature,
        )
        # Explicitly freeze the global head — it is unsupervised by this plan.
        self.video_aleatoric.global_uncertainty_head.requires_grad_(False)

    def score_features(
        self,
        features: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass risk scorer: ``(u_frame [B,T], u_video [B], relevance [B,T])``."""

        output = self.video_aleatoric(features, video_mask.to(torch.bool))
        u_frame = output.local_uncertainty
        u_video = torch.sum(output.relevance_weights * u_frame, dim=1)
        return u_frame, u_video, output.relevance_weights

    def compute_ranking_loss(
        self,
        *,
        clean_features: torch.Tensor,
        primary_features: torch.Tensor,
        secondary_features: Optional[torch.Tensor],
        primary_rank: torch.Tensor,
        order_rank: torch.Tensor,
        order_batch_indices: torch.Tensor,
        video_mask: torch.Tensor,
        base_margin: float = 0.05,
        lambda_cf: float = 1.0,
        lambda_cv: float = 1.0,
        lambda_of: float = 1.0,
        lambda_ov: float = 1.0,
    ) -> CRFRankingLoss:
        """Compute the four-term CRF ranking loss from ALREADY-DETACHED features.

        Callers (the model's paired forward) are responsible for the
        gradient-isolation dance: encode clean with grad for retrieval, then
        pass ``.detach()`` copies into this head so ranking gradients flow only
        through the aleatoric heads. Encode degraded / order under
        ``torch.no_grad()`` in the visual encoder to save memory and prevent
        double-counting backward through the shared backbone.

        Order pairs: ``primary`` for the order slot lives at
        ``primary_features[order_batch_indices]``; ``secondary`` lives at
        ``secondary_features`` (shape ``[n_order, T, D]``). This method swaps
        them into ``low_frame`` / ``high_frame`` based on ordinal rank, so the
        ranking loss always sees ``low`` = less-compressed = smaller CRF.
        """

        video_mask_bool = video_mask.to(torch.bool)

        clean_out = self.video_aleatoric(clean_features, video_mask_bool)
        primary_out = self.video_aleatoric(primary_features, video_mask_bool)
        u_clean = clean_out.local_uncertainty
        u_primary = primary_out.local_uncertainty
        clean_weights = clean_out.relevance_weights.detach()

        have_order = (
            secondary_features is not None
            and secondary_features.shape[0] > 0
            and order_batch_indices.numel() > 0
        )
        low_frame = high_frame = low_rank = high_rank = order_mask = order_weights = None
        if have_order:
            order_video_mask_bool = video_mask_bool[order_batch_indices]
            secondary_out = self.video_aleatoric(secondary_features, order_video_mask_bool)
            u_secondary = secondary_out.local_uncertainty
            u_primary_at_order = u_primary[order_batch_indices]

            p_rank_long = order_rank[:, 0].to(torch.long)
            s_rank_long = order_rank[:, 1].to(torch.long)
            primary_is_low = (p_rank_long < s_rank_long).view(-1, 1)

            low_frame = torch.where(primary_is_low, u_primary_at_order, u_secondary)
            high_frame = torch.where(primary_is_low, u_secondary, u_primary_at_order)
            primary_is_low_flat = primary_is_low.squeeze(1)
            low_rank = torch.where(primary_is_low_flat, p_rank_long, s_rank_long)
            high_rank = torch.where(primary_is_low_flat, s_rank_long, p_rank_long)

            order_mask = order_video_mask_bool
            order_weights = clean_weights[order_batch_indices]

        return crf_ranking_loss(
            u_clean_frame=u_clean,
            u_primary_frame=u_primary,
            primary_rank=primary_rank,
            video_mask=video_mask_bool,
            clean_weights=clean_weights,
            order_low_frame=low_frame,
            order_high_frame=high_frame,
            order_low_rank=low_rank,
            order_high_rank=high_rank,
            order_mask=order_mask,
            order_weights=order_weights,
            base_margin=base_margin,
            lambda_cf=lambda_cf,
            lambda_cv=lambda_cv,
            lambda_of=lambda_of,
            lambda_ov=lambda_ov,
        )


def paired_visual_forward(
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    clean_video: torch.Tensor,
    degraded_video: torch.Tensor,
    order_video: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode clean + degraded + order with strict gradient isolation.

    * Clean is encoded WITH grad — the returned tensor is the source of the
      retrieval-loss branch AND, after ``.detach()``, the source of the
      aleatoric-loss branch. Callers are responsible for the detach.
    * Degraded + order are encoded under ``torch.no_grad()``; the results are
      already computed without grad. The encoder call is done under
      ``self.training`` remaining true, so BN / dropout stay in training
      mode — the plan explicitly rules out a temporary ``.eval()`` swap.
    """

    clean_features = encoder_fn(clean_video)
    with torch.no_grad():
        degraded_features = encoder_fn(degraded_video)
        if order_video.numel() > 0:
            order_features = encoder_fn(order_video)
        else:
            order_features = order_video.new_zeros(
                (0,) + degraded_features.shape[1:]
            )
    return clean_features, degraded_features.detach(), order_features.detach()
