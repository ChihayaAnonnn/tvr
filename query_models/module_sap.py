"""Semantic Anchor Probing (SAP) module.

以时空 token（T 帧 × S patches）为 memory，16 个可学习锚点通过
cross-attention 聚焦到特定帧的特定 patch 区域，保留时空局部性。
Sigmoid 独立门控：每个 anchor 独立计算 sigmoid 权重，L1 归一化聚合，
无竞争，结构性消除锚点坍缩。

输出:
  - anchors       [B, N, D]   锚点表征
  - mu_raw        [B, D]      加权聚合均值
  - logsigma      [B, D]      加权聚合 log 方差
  - gate_scores   [B, N]      每锚点独立门控分数（诊断用）
"""

import torch
import torch.nn as nn


class SemanticAnchorProbing(nn.Module):

    def __init__(self, d_model=512, num_anchors=16, nhead=8, num_layers=2,
                 log_sigma_min=None, log_sigma_max=None):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # 可学习语义锚点
        self.anchor_tokens = nn.Parameter(torch.randn(num_anchors, d_model))
        nn.init.trunc_normal_(self.anchor_tokens, std=0.02)

        # 锚点间 self-attention + 与时空 token cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 每锚点不确定性头：ℓ_n = W_u q_n + b_u
        # 偏置初始化 0.5 → 初始 logsigma ≈ 0.5, var ≈ 1.65
        self.uncertainty_head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, 0.5)

        # Sigmoid 独立门控：每个 anchor 独立输出标量权重 [0, 1]
        # 2 层 MLP + sigmoid，无竞争，结构性消除坍缩
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, video_features, padding_mask=None):
        """
        Args:
            video_features: [B, T*S, D] 时空 token（T 帧 × S patches）
            padding_mask:   [B, T*S]    True = padded position
        Returns:
            dict: anchors, mu_raw, logsigma, gate_scores
        """
        B = video_features.shape[0]
        anchors = self.anchor_tokens.unsqueeze(0).expand(B, -1, -1)

        anchors = self.decoder(
            tgt=anchors, memory=video_features,
            memory_key_padding_mask=padding_mask,
        )
        anchors = self.norm(anchors)  # [B, N, D]

        # 每锚点 log 方差
        anchor_logsigma = self.uncertainty_head(anchors)  # [B, N, D]
        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            anchor_logsigma = torch.clamp(anchor_logsigma, min=self.log_sigma_min, max=self.log_sigma_max)

        # Sigmoid 独立门控：每个 anchor 独立计算权重，L1 归一化
        gate_scores = torch.sigmoid(self.gate_fc(anchors)).squeeze(-1)  # [B, N]
        alpha = gate_scores / (gate_scores.sum(dim=1, keepdim=True) + 1e-9)  # [B, N]

        # 加权聚合
        mu_raw = (alpha.unsqueeze(-1) * anchors).sum(dim=1)  # [B, D]
        mu_raw = mu_raw / (mu_raw.norm(dim=-1, keepdim=True) + 1e-9)

        # mixture 方差：log(Σ α_i σ_i²)，而非 Σ α_i log(σ_i²)
        anchor_var = torch.exp(anchor_logsigma)  # [B, N, D]
        agg_var = (alpha.unsqueeze(-1) * anchor_var).sum(dim=1)  # [B, D]
        logsigma = torch.log(agg_var + 1e-8)  # [B, D]

        return {
            "anchors": anchors,
            "mu_raw": mu_raw,
            "logsigma": logsigma,
            "gate_scores": gate_scores,
        }
