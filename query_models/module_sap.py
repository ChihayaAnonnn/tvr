"""Semantic Anchor Probing (SAP) module — 非学习不确定性版本（方向1）。

以时空 token（T 帧 × S patches）为 memory，K 个可学习锚点通过
cross-attention 聚焦到特定帧的特定 patch 区域，保留时空局部性。

不确定性从 anchor 表征的统计量直接计算，无需学习：
  - 锚点多样性：anchor 间 cosine 差异度 → 视频内容复杂度
  - 模态熵：Dirichlet 概率分布的熵 → 模态模糊度

聚合方式：用模态概率 p_k 对 anchor 表征加权求和。

输出:
  - anchors           [B, K, D]   锚点表征 (decoder 输出)
  - mu_raw            [B, D]      模态概率加权聚合均值 (L2 归一化)
  - logsigma          [B, D]      mixture log 方差 (供 MIL 采样)
  - alpha_dir         [B, K]      Dirichlet 证据量
  - modal_probs       [B, K]      模态概率 p_k = α_k / S
  - u_mode            [B]         离散模态不确定性 U = K / S
  - epistemic_cont    [B, 1, 1]   非学习不确定性 (anchor 多样性 × 模态熵)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialUncertaintyHead(nn.Module):
    """Dirichlet 模态概率头。

    输入 anchor 特征 [B, K, D]，输出 Dirichlet 证据量。
    不确定性（epistemic_cont / logsigma）改由 SAP.forward() 中
    从 anchor 统计量直接计算（方向1：非学习不确定性）。
    """

    def __init__(self, d_model=512, n_anchors=16):
        super().__init__()
        self.n_anchors = n_anchors
        self.d_model = d_model

        # 离散狄利克雷层：预测 K 个语义模态的证据量
        self.dirichlet_layer = nn.Linear(d_model, 1)

    def forward(self, anchor_features):
        """
        Args:
            anchor_features: [B, K, D] 投影后的锚点表征
        Returns:
            dict: alpha_dir [B,K], u_mode [B]
        """
        B, K, D = anchor_features.shape

        dir_logits = self.dirichlet_layer(anchor_features).squeeze(-1)  # [B, K]
        alpha_dir = F.softplus(dir_logits) + 1.0  # [B, K]
        S = torch.sum(alpha_dir, dim=-1, keepdim=True)
        u_mode = (K / S).squeeze(-1)  # [B]

        return {
            "alpha_dir": alpha_dir,  # [B, K]
            "u_mode": u_mode,        # [B]
        }


class SemanticAnchorProbing(nn.Module):
    """SAP — 非学习不确定性版本：Dirichlet 模态概率 + anchor 统计量不确定性。"""

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

        # Dirichlet 模态概率头（不确定性改由 anchor 统计量计算）
        self.evidential_head = EvidentialUncertaintyHead(d_model, num_anchors)

        # 锚点投影层：替代原 NIG gamma 的 Linear 变换，给每个锚点聚合前的表征自由度
        self.anchor_proj = nn.Linear(d_model, d_model)

    def forward(self, video_features, padding_mask=None):
        """
        Args:
            video_features: [B, T*S, D] 时空 token
            padding_mask:   [B, T*S]    True = padded position
        Returns:
            dict: anchors, mu_raw, logsigma, alpha_dir, modal_probs,
                  u_mode, epistemic_cont
        """
        B = video_features.shape[0]
        K = self.num_anchors
        anchors = self.anchor_tokens.unsqueeze(0).expand(B, -1, -1)

        anchors = self.decoder(
            tgt=anchors, memory=video_features,
            memory_key_padding_mask=padding_mask,
        )
        anchors = self.norm(anchors)  # [B, K, D]

        # ---- 锚点投影（给聚合前表征自由度）----
        projected = self.anchor_proj(anchors)            # [B, K, D]

        # ---- Dirichlet 模态概率（detach 阻断梯度反传至 decoder）----
        ev = self.evidential_head(projected.detach())

        alpha_dir = ev["alpha_dir"]                      # [B, K]
        modal_probs = alpha_dir / (alpha_dir.sum(dim=1, keepdim=True) + 1e-9)  # [B, K]

        # ---- 非学习不确定性：从 anchor 统计量直接计算 ----
        # 锚点多样性：anchor 间 cosine 差异度 → 视频内容复杂度
        anchors_norm = F.normalize(anchors, dim=-1)               # [B, K, D]
        anchor_sim = torch.bmm(anchors_norm, anchors_norm.transpose(1, 2))  # [B, K, K]
        off_mask = ~torch.eye(K, dtype=torch.bool, device=anchors.device)
        diversity = (1.0 - anchor_sim[:, off_mask].view(B, K, K - 1).mean(dim=-1)).mean(dim=-1)  # [B]

        # 模态熵：Dirichlet 分布的离散程度
        modal_entropy = -(modal_probs * torch.log(modal_probs + 1e-8)).sum(dim=-1)  # [B]
        modal_entropy_norm = modal_entropy / torch.tensor(K, dtype=modal_entropy.dtype).log()  # [B] ∈ [0, 1]

        # 复合不确定性
        uncertainty = diversity * modal_entropy_norm  # [B]
        epistemic_cont = uncertainty.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        # ---- 模态概率加权聚合 ----
        mu_raw = (modal_probs.unsqueeze(-1) * projected).sum(dim=1)  # [B, D]
        mu_raw = mu_raw / (mu_raw.norm(dim=-1, keepdim=True) + 1e-9)

        # logsigma：从 anchor 间方差计算（高 diversity → 高方差 → 高 logsigma）
        anchor_dim_var = torch.var(anchors, dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        logsigma = torch.log(anchor_dim_var + 1e-8)  # [B, 1]
        if self.log_sigma_min is not None and self.log_sigma_max is not None:
            logsigma = torch.clamp(logsigma, min=self.log_sigma_min, max=self.log_sigma_max)
        logsigma = logsigma.expand(-1, self.d_model)  # [B, D]

        return {
            "anchors": anchors,           # [B, K, D] decoder 输出（供 orth_loss）
            "mu_raw": mu_raw,             # [B, D]    模态概率聚合均值 (L2 norm)
            "logsigma": logsigma,         # [B, D]    mixture log 方差 (供 MIL)
            "alpha_dir": alpha_dir,       # [B, K]    Dirichlet 证据量
            "modal_probs": modal_probs,   # [B, K]    模态概率 p_k
            "u_mode": ev["u_mode"],       # [B]       离散模态不确定性
            "epistemic_cont": epistemic_cont,  # [B, 1, 1] 非学习不确定性
            "per_anchor_var": torch.var(anchors, dim=-1),  # [B, K] 每 anchor 维度方差（诊断用）
        }
