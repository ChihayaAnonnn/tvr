"""Uncertainty modules
Reference code:
    PIENet in
    https://github.com/yalesong/pvse/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from prob_models.pie_model import MultiHeadSelfAttention

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


class UncertaintyAdaNorm(nn.Module):
    """
    不确定性感知的自适应归一化层 (inspired by Helios adaLN)。
    用 logsigma 动态生成 scale/shift，让归一化行为因样本不确定性而异：
      output = LayerNorm(x) * (1 + scale) + shift
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, logsigma):
        """
        x: [B, D] 均值特征
        logsigma: [B, D] 对数方差
        """
        scale, shift = self.proj(logsigma).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class UncertaintyModuleImage(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return {
            "logsigma": out,
            "attention": attn,
        }


class UncertaintyModuleText(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()

        self.rnn = nn.GRU(d_in, d_out // 2, bidirectional=True, batch_first=True)
        self.embed_dim = d_out

        # 添加fc2用于处理全局池化特征
        self.fc2 = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        # 从pad_mask计算序列长度，如果没有pad_mask则使用序列的最大长度
        if pad_mask is not None:
            lengths = pad_mask.sum(dim=1).long()  # [B]
        else:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        # pack_padded_sequence 要求 length >= 1
        lengths = torch.clamp(lengths, min=1)

        # Forward propagate RNNs
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        gather_idx = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        gru_out = torch.gather(padded[0], 1, gather_idx).squeeze(1)

        # 结合注意力残差、GRU输出和全局池化特征
        fc_out = self.fc2(out)  # 全局特征 → FC
        out = self.fc(residual) + gru_out + fc_out  # 注意力残差 + GRU输出 + 全局特征

        return {
            "logsigma": out,
            "attention": attn,
        }


class UncertaintyModuleTextMamba(nn.Module):
    def __init__(self, d_in, d_out, d_h, d_state=16, d_conv=4, expand=4):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        # 添加fc2用于处理全局池化特征
        self.fc2 = nn.Linear(d_in, d_out)

        self.init_weights()

        # 使用 Mamba 替代 GRU
        if Mamba2 is None:
            raise ImportError(
                "mamba_ssm is required for UncertaintyModuleTextMamba. "
                "Please install it or use UncertaintyModuleText instead."
            )

        self.mamba = Mamba2(
            d_model=d_in,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=False,
        )
        self.norm_mamba = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

        self.embed_dim = d_out

        # 可学习的融合权重：
        # alpha: Attention vs Mamba 的权重（alpha越大，越信任Attention）
        # beta: 全局特征的权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值0.5，表示Attention和Mamba平等
        self.beta = nn.Parameter(torch.tensor(0.3))  # 初始值0.3，全局特征权重较小

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, out, x, pad_mask=None):
        """
        Args:
            out: [B, D] - 全局池化特征 (text_pooled)
            x: [B, T, D] - token序列 (text_token)
            pad_mask: [B, T] - padding mask，1表示有效token，0表示padding
        """
        residual, attn = self.attention(x, pad_mask)

        # 对 padding 位置进行 mask（设置为 0）
        if pad_mask is not None:
            # pad_mask: [B, T], 1=valid, 0=padding
            # 将 padding 位置设为 0
            mask_expanded = pad_mask.unsqueeze(-1).float()  # [B, T, 1]
            x_masked = x * mask_expanded
        else:
            x_masked = x

        # Mamba 处理（Pre-Norm + 残差连接）
        x_norm = self.norm_mamba(x_masked)
        x_seq = x_masked + self.dropout(self.mamba(x_norm))

        # 提取最后时刻的输出（类似 GRU 的 last hidden state）
        if pad_mask is not None:
            # 获取每个序列的实际长度
            lengths = pad_mask.sum(dim=1).long()  # [B]
            # 收集每个序列最后一个有效时刻的特征
            batch_indices = torch.arange(x_seq.size(0), device=x_seq.device)
            # 确保索引不越界
            lengths = torch.clamp(lengths - 1, min=0)
            seq_out = x_seq[batch_indices, lengths]  # [B, D]
        else:
            # 如果没有 mask，使用最后一个时刻
            seq_out = x_seq[:, -1, :]  # [B, D]

        # 结合注意力残差、Mamba输出和全局池化特征（使用可学习的加权融合）
        fc_out = self.fc2(out)  # 全局特征 → FC
        attention_out = self.fc(residual)  # Attention残差 → FC

        # 使用sigmoid确保alpha在[0,1]之间，平衡Attention和Mamba
        alpha_weight = torch.sigmoid(self.alpha)  # α ∈ [0, 1]
        seq_fusion = alpha_weight * attention_out + (1 - alpha_weight) * seq_out

        # 全局特征的权重（使用sigmoid确保在合理范围内）
        beta_weight = torch.sigmoid(self.beta)  # β ∈ [0, 1]

        # 最终的加权融合：融合后的序列特征 + 全局特征
        out = seq_fusion + beta_weight * fc_out

        return {
            "logsigma": out,
            "attention": attn,
        }


class EvidentialUncertaintyHead(nn.Module):
    def __init__(self, d_model=512, n_anchors=16):
        super().__init__()
        self.n_anchors = n_anchors
        self.d_model = d_model

        # 1. 离散层：预测 16 个语义主观模态的证据量
        self.dirichlet_layer = nn.Linear(d_model, 1)

        # 2. 连续层：为每个模态预测 NIG 分布的 4 个参数 (γ, v, α_nig, β_nig)
        # γ (mean), v (evidence scale), α (shape), β (scale)
        self.nig_layer = nn.Linear(d_model, 4 * d_model)

    def forward(self, anchor_features):
        # anchor_features 形状: [B, 16, 512]
        B, N, D = anchor_features.shape

        # ---- 离散狄利克雷层 ----
        dir_logits = self.dirichlet_layer(anchor_features).squeeze(-1)  # [B, 16]
        # 证据量必须大于 0，通常公式为 softplus(x) + 1
        alpha_dir = F.softplus(dir_logits) + 1.0
        S = torch.sum(alpha_dir, dim=-1, keepdim=True)  # 总证据量 [B, 1]
        u_mode = self.n_anchors / S  # 模态不确定性 [B, 1]

        # ---- 连续 NIG 层 ----
        nig_params = self.nig_layer(anchor_features)  # [B, 16, 4*512]
        nig_params = nig_params.view(B, N, 4, D)

        gamma = nig_params[:, :, 0, :]  # 核心表征向量（均值），无特定激活或用 tanh
        v = F.softplus(nig_params[:, :, 1, :]) + 1e-6
        alpha_nig = F.softplus(nig_params[:, :, 2, :]) + 1.0 + 1e-6
        beta_nig = F.softplus(nig_params[:, :, 3, :]) + 1e-6

        # 计算连续特征维度的认知不确定性 (Epistemic Uncertainty)
        epistemic_cont = beta_nig / (v * (alpha_nig - 1.0))

        return {
            "alpha_dir": alpha_dir,  # [B, 16] 离散证据
            "u_mode": u_mode,  # [B, 1] 离散不确定性
            "gamma": gamma,  # [B, 16, 512] 连续特征中心
            "epistemic_cont": epistemic_cont,  # [B, 16, 512] 连续不确定性
        }
