"""Semantic Anchor Probing (SAP) module.

Decomposes video frame features into N semantically meaningful anchors,
each with its own uncertainty estimate.  Produces:
  - anchor tokens  [B, N, D]   (for WTI matching)
  - gate scores    [B, N]      (semantic relevance)
  - alpha          [B, N]      (uncertainty-modulated weights)
  - mu_raw         [B, D]      (aggregated mean for probabilistic embedding)
  - logsigma       [B, D]      (composed video-level log-variance)
  - anchor_logsigma[B, N, D]   (per-anchor log-variance)
"""

import torch
import torch.nn as nn


class SemanticAnchorProbing(nn.Module):

    def __init__(self, d_model=512, num_anchors=16, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors

        # Learnable semantic anchors
        self.anchor_tokens = nn.Parameter(torch.randn(num_anchors, d_model))
        nn.init.trunc_normal_(self.anchor_tokens, std=0.02)

        # Self-Attn among anchors + Cross-Attn with video frames
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Semantic relevance gate  g_n = σ(MLP(q_n))
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Per-anchor uncertainty head  ℓ_n = W_u q_n + b_u
        # Bias initialized to 0.5 → initial logsigma ≈ 0.5, var ≈ exp(0.5) ≈ 1.65
        # Prevents immediate variance collapse to near-zero at training start.
        self.uncertainty_head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, 0.5)

        # Learnable scale controlling how much uncertainty affects gating.
        # Initialized to 0.1 so uncertainty modulation has a meaningful effect from step 1.
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, video_features, padding_mask=None):
        """
        Args:
            video_features: [B, T, D] frame-level features (L2-normed)
            padding_mask:   [B, T]    True = padded position
        Returns:
            dict with keys listed in module docstring
        """
        B = video_features.shape[0]
        anchors = self.anchor_tokens.unsqueeze(0).expand(B, -1, -1)

        anchors = self.decoder(
            tgt=anchors, memory=video_features,
            memory_key_padding_mask=padding_mask,
        )
        anchors = self.norm(anchors)  # [B, N, D]

        # --- semantic relevance ---
        gate_scores = self.gate_fc(anchors).squeeze(-1)  # [B, N]

        # --- per-anchor uncertainty ---
        anchor_logsigma = self.uncertainty_head(anchors)  # [B, N, D]

        # --- uncertainty-modulated gating ---
        # α_n = softmax( log g_n  −  β · norm(mean(ℓ_n)) )
        # Normalize anchor_unc_scalar per-sample so beta's effect is scale-invariant.
        anchor_unc_scalar = anchor_logsigma.mean(dim=-1)  # [B, N]
        unc_mean = anchor_unc_scalar.mean(dim=-1, keepdim=True)
        unc_std  = anchor_unc_scalar.std(dim=-1, keepdim=True) + 1e-6
        anchor_unc_norm = (anchor_unc_scalar - unc_mean) / unc_std   # [B, N] zero-mean, unit-std
        modulated_logits = (
            torch.log(gate_scores + 1e-9)
            - self.beta * anchor_unc_norm
        )
        alpha = torch.softmax(modulated_logits, dim=-1)   # [B, N]

        # --- compositional aggregation ---
        anchors_norm = anchors / (anchors.norm(dim=-1, keepdim=True) + 1e-9)
        mu_raw = torch.einsum("bn,bnd->bd", alpha, anchors_norm)
        mu_raw = mu_raw / (mu_raw.norm(dim=-1, keepdim=True) + 1e-9)

        logsigma = torch.einsum("bn,bnd->bd", alpha, anchor_logsigma)

        return {
            "anchors": anchors,
            "gate_scores": gate_scores,
            "alpha": alpha,
            "mu_raw": mu_raw,
            "logsigma": logsigma,
            "anchor_logsigma": anchor_logsigma,
        }
