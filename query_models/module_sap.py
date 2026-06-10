"""Semantic Anchor Probing (SAP) module — Evidential 版本。

以时空 token（T 帧 × S patches）为 memory，K 个可学习锚点通过
cross-attention 聚焦到特定帧的特定 patch 区域，保留时空局部性。

双层主观不确定性建模：
  - 离散层：Dirichlet 证据量 α_dir → 模态概率 p_k = α_k / S
  - 连续层：NIG 参数 (γ, ν, α_nig, β_nig) → 认知不确定性 U_epistemic

聚合方式：用模态概率 p_k 对 gamma 加权求和，替代原 Sigmoid 门控。

输出:
  - anchors           [B, K, D]   锚点表征 (decoder 输出)
  - gamma             [B, K, D]   NIG 均值参数 (替代原 anchors 用于聚合)
  - mu_raw            [B, D]      模态概率加权聚合均值 (L2 归一化)
  - logsigma          [B, D]      mixture log 方差 (供 MIL 采样)
  - alpha_dir         [B, K]      Dirichlet 证据量
  - modal_probs       [B, K]      模态概率 p_k = α_k / S
  - u_mode            [B]         离散模态不确定性 U = K / S
  - epistemic_cont    [B, K, D]   连续认知不确定性 β/(ν(α-1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialUncertaintyHead(nn.Module):
    """双层主观不确定性头：离散 Dirichlet + 连续 NIG。

    输入 anchor 特征 [B, K, D]，同时输出：
      - 离散层：Dirichlet 非负证据量 alpha_dir [B, K]
      - 连续层：NIG 分布参数 (gamma, v, alpha_nig, beta_nig) [B, K, D]
    """

    def __init__(self, d_model=512, n_anchors=16, beta_max=5.0):
        super().__init__()
        self.n_anchors = n_anchors
        self.d_model = d_model
        self.beta_max = beta_max  # beta_nig 上界，防止认知不确定性爆炸

        # 离散层：预测 K 个语义模态的 Dirichlet 证据量
        self.dirichlet_layer = nn.Linear(d_model, 1)

        # 连续层：为每个模态预测 NIG 分布的 4 个参数 (γ, ν, α_nig, β_nig)
        self.nig_layer = nn.Linear(d_model, 4 * d_model)

    def forward(self, anchor_features):
        """
        Args:
            anchor_features: [B, K, D] decoder 输出的锚点表征
        Returns:
            dict: alpha_dir [B,K], gamma [B,K,D], v/alpha_nig/beta_nig [B,K,D],
                  per_anchor_logsigma [B,K,D], u_mode [B]
        """
        B, K, D = anchor_features.shape

        # ---- 离散狄利克雷层 ----
        dir_logits = self.dirichlet_layer(anchor_features).squeeze(-1)  # [B, K]
        # 证据量必须 > 0，softplus(x) + 1 保证 α ≥ 1
        alpha_dir = F.softplus(dir_logits) + 1.0  # [B, K]
        S = torch.sum(alpha_dir, dim=-1, keepdim=True)  # 总证据量 [B, 1]
        u_mode = (K / S).squeeze(-1)  # 模态不确定性 [B]

        # ---- 连续 NIG 层 ----
        nig_params = self.nig_layer(anchor_features)  # [B, K, 4*D]
        nig_params = nig_params.view(B, K, 4, D)

        gamma = nig_params[:, :, 0, :]  # 核心表征（均值）
        v = F.softplus(nig_params[:, :, 1, :]) + 1e-6        # 证据尺度 > 0
        alpha_nig = F.softplus(nig_params[:, :, 2, :]) + 1.0 + 1e-6  # α > 1
        # beta_nig 使用 bounded sigmoid 限制上界，防止认知不确定性爆炸
        # beta_nig ∈ (0, beta_max)，训练中期梯度不会无限外扩
        beta_nig = self.beta_max * torch.sigmoid(nig_params[:, :, 3, :]) + 1e-6

        # 连续特征维度的认知不确定性 β / (ν(α-1))
        epistemic_cont = beta_nig / (v * (alpha_nig - 1.0))  # [B, K, D]

        # per-anchor log 方差（供 mixture 聚合和 MIL 采样）
        per_anchor_logsigma = torch.log(beta_nig / (alpha_nig - 1.0) + 1e-8)

        return {
            "alpha_dir": alpha_dir,           # [B, K]
            "u_mode": u_mode,                 # [B]
            "gamma": gamma,                   # [B, K, D]
            "v": v,                           # [B, K, D]
            "alpha_nig": alpha_nig,           # [B, K, D]
            "beta_nig": beta_nig,             # [B, K, D]
            "epistemic_cont": epistemic_cont, # [B, K, D]
            "per_anchor_logsigma": per_anchor_logsigma,  # [B, K, D]
        }


class SemanticAnchorProbing(nn.Module):
    """SAP — Evidential 版本：Dirichlet 模态概率 + NIG 连续不确定性。"""

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

        # 双层主观不确定性头（替代原 gate_fc + uncertainty_head）
        self.evidential_head = EvidentialUncertaintyHead(d_model, num_anchors)

    def forward(self, video_features, padding_mask=None):
        """
        Args:
            video_features: [B, T*S, D] 时空 token
            padding_mask:   [B, T*S]    True = padded position
        Returns:
            dict: anchors, gamma, mu_raw, logsigma, alpha_dir, modal_probs,
                  u_mode, epistemic_cont
        """
        B = video_features.shape[0]
        anchors = self.anchor_tokens.unsqueeze(0).expand(B, -1, -1)

        anchors = self.decoder(
            tgt=anchors, memory=video_features,
            memory_key_padding_mask=padding_mask,
        )
        anchors = self.norm(anchors)  # [B, K, D]

        # ---- 双层不确定性（detach 阻断不确定性梯度反传至 decoder，消除双头冲突）----
        ev = self.evidential_head(anchors.detach())

        gamma = ev["gamma"]               # [B, K, D] NIG 均值
        alpha_dir = ev["alpha_dir"]        # [B, K]   Dirichlet 证据量
        modal_probs = alpha_dir / (alpha_dir.sum(dim=1, keepdim=True) + 1e-9)  # [B, K]

        # clamp per-anchor logsigma（在 mixture 聚合前）
        per_anchor_logsigma = ev["per_anchor_logsigma"]  # [B, K, D]
        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            per_anchor_logsigma = torch.clamp(
                per_anchor_logsigma,
                min=self.log_sigma_min, max=self.log_sigma_max,
            )

        # ---- 模态概率加权聚合 ----
        mu_raw = (modal_probs.unsqueeze(-1) * gamma).sum(dim=1)  # [B, D]
        mu_raw = mu_raw / (mu_raw.norm(dim=-1, keepdim=True) + 1e-9)

        # mixture 方差：log(Σ p_k σ_k²)，供 MIL 采样使用
        anchor_var = torch.exp(per_anchor_logsigma)  # [B, K, D]
        agg_var = (modal_probs.unsqueeze(-1) * anchor_var).sum(dim=1)  # [B, D]
        logsigma = torch.log(agg_var + 1e-8)  # [B, D]

        return {
            "anchors": anchors,           # [B, K, D] decoder 输出（供 orth_loss）
            "gamma": gamma,               # [B, K, D] NIG 均值
            "mu_raw": mu_raw,             # [B, D]    模态概率聚合均值 (L2 norm)
            "logsigma": logsigma,         # [B, D]    mixture log 方差 (供 MIL)
            "alpha_dir": alpha_dir,       # [B, K]    Dirichlet 证据量
            "modal_probs": modal_probs,   # [B, K]    模态概率 p_k
            "u_mode": ev["u_mode"],       # [B]       离散模态不确定性
            "epistemic_cont": ev["epistemic_cont"],  # [B, K, D] 连续认知不确定性
            "per_anchor_var": anchor_var, # [B, K, D] 每 anchor NIG 方差 β/(α-1)
        }
