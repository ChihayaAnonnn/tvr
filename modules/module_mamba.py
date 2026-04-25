import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class LayerNormConv(nn.LayerNorm):
    """LayerNorm for Conv2D with channel-last format."""
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type).permute(0, 3, 1, 2)  # -> [B, C, H, W]

class VideoMambaEncoder(nn.Module):
    def __init__(self, dim, proj_dim=256, d_state=16, d_conv=4, expand=2, 
                 return_sequence=True, use_multiscale=False, scale_factors=None):
        super().__init__()
        self.return_sequence = return_sequence
        self.use_multiscale = use_multiscale
        self.dim = dim
        
        if scale_factors is None:
            scale_factors = [0.5, 1.0, 2.0]  # 默认三个尺度
        self.scale_factors = scale_factors

        # 1. 多尺度特征提取模块
        if self.use_multiscale:
            self.multiscale_stages = nn.ModuleList()
            for scale in scale_factors:
                stage = self._build_scale_stage(dim, scale)
                self.multiscale_stages.append(stage)
            # Scale-Aware Embedding: 区分不同尺度分支
            self.scale_emb = nn.Parameter(torch.randn(len(self.scale_factors), 1, 1, dim))
        
        # 2. Mamba 时序建模
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=False
        )

        # 2.1 正则化与残差（稳定训练）
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

        # 3. 投影层
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, proj_dim)
        )
        # 3.1 位置/类型编码
        self.T_max = 256
        self.temporal_emb = nn.Parameter(torch.zeros(1, self.T_max, dim))  # 时序位置编码
        self.token_type_emb = nn.Parameter(torch.randn(2, 1, dim))  # 0: cls, 1: spatial/multiscale
    
    def _build_scale_stage(self, dim, scale):
        """构建单个尺度的处理模块
        
        Args:
            dim: 特征维度
            scale: 缩放因子，可以是浮点数或整数
                - 如果是整数且>=10: 表示目标尺寸 (如14表示14x14)
                - 如果是浮点数: 表示缩放比例 (如2.0表示2倍上采样)
        """
        layers = []
        out_dim = dim
        
        # 如果scale是小整数(1-20)，视为目标尺寸而非比例
        if isinstance(scale, int) and 1 <= scale <= 20:
            target_size = scale
            # 使用自适应池化到目标尺寸
            layers = [nn.AdaptiveAvgPool2d(target_size)]
        elif scale == 4.0:
            # 上采样 4x
            layers = [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                LayerNormConv(dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            ]
            out_dim = dim // 4
        elif scale == 2.0:
            # 上采样 2x (7x7 → 14x14)
            layers = [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
            ]
            out_dim = dim // 2
        elif scale == 1.0:
            # 保持原始尺度 (7x7)
            layers = []
        elif scale == 0.5:
            # 下采样 0.5x (7x7 → 3x3)
            layers = [nn.AdaptiveAvgPool2d(3)]
        elif scale == 0.25:
            # 下采样 0.25x (7x7 → 2x2)
            layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
        elif scale == 0.14:
            # 全局池化到 1x1
            layers = [nn.AdaptiveAvgPool2d(1)]
        else:
            raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
        
        # 添加后续的卷积处理
        layers.extend([
            nn.Conv2d(out_dim, dim, kernel_size=1),
            LayerNormConv(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            LayerNormConv(dim)
        ])
        
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        # x: [B, T, L, D]
        B, T, L, D = x.shape
        
        if self.use_multiscale:
            # ========== 多尺度模式 ==========
            # (1) 提取空间特征图
            # 假设 L = 50 (1 CLS + 49 patches)
            H = W = int(math.sqrt(L - 1))  # H=W=7
            
            # 分离 CLS token 和 spatial tokens
            cls_tokens = x[:, :, 0, :]  # [B, T, D]
            spatial_tokens = x[:, :, 1:, :].reshape(B * T, H, W, D).permute(0, 3, 1, 2)
            # spatial_tokens: [B*T, D, H, W]
            
            # (2) 多尺度特征提取
            multiscale_features = []
            for i, stage in enumerate(self.multiscale_stages):
                feat = stage(spatial_tokens)  # [B*T, D, H', W']
                # 展平空间维度: [B*T, D, H', W'] -> [B, T, H'*W', D]
                feat = feat.flatten(2).permute(0, 2, 1)  # [B*T, H'*W', D]
                feat = feat.view(B, T, -1, D)  # [B, T, H'*W', D]
                # 注入尺度感知编码
                feat = feat + self.scale_emb[i]
                multiscale_features.append(feat)
            
            # (3) Scale-wise聚合: 展平时序维度并concat
            multiscale_flat = [feat.view(B, -1, D) for feat in multiscale_features]
            # multiscale_flat[i]: [B, T*spatial_tokens_i, D]
            
            # Concat所有尺度特征
            multiscale_concat = torch.cat(multiscale_flat, dim=1)
            # [B, sum(T*spatial_tokens_i), D]
            # Token-Type: spatial/multiscale token 类型偏置
            multiscale_concat = multiscale_concat + self.token_type_emb[1]
            # Temporal PE: 仅对帧级CLS序列注入时间信息
            cls_tokens = cls_tokens + self.temporal_emb[:, :T, :]
            # Token-Type: CLS token 类型偏置
            cls_tokens = cls_tokens + self.token_type_emb[0]
            
            # 加上原始CLS tokens
            x_multiscale = torch.cat([cls_tokens, multiscale_concat], dim=1)
            # [B, T + sum(T*spatial_tokens_i), D]
            
            # (4) Mamba时序融合（Pre-Norm + 残差 + Post-Norm）
            x_in = x_multiscale  # [B, seq_len, D]
            x_norm = self.norm_in(x_in)
            x_m = self.mamba(x_norm)
            x = x_in + self.dropout(x_m)
            x = self.norm_out(x)
            
            # (5) 提取前T个token作为时序表示 (对应原始CLS tokens位置)
            x = x[:, :T, :]  # [B, T, D]
        
        else:
            # ========== 简单模式（原始实现）==========
            # (1) 空间池化 -> [B,T,D]
            x_in = x.mean(dim=2)
            # Temporal PE: 注入帧序时间位置
            x_in = x_in + self.temporal_emb[:, :T, :]

            # (2) Mamba（Pre-Norm + 残差 + Post-Norm）-> [B, T, D]
            x_norm = self.norm_in(x_in)
            x_m = self.mamba(x_norm)
            x = x_in + self.dropout(x_m)
            x = self.norm_out(x)

        # (3) 投影和归一化
        if self.return_sequence:
            # 时序序列表示 -> [B, T, proj_dim]
            z_seq = self.proj(x)
            z_seq = nn.functional.normalize(z_seq, dim=-1)
            
            # 视频级表示 -> [B, proj_dim]
            vid_emb = x.mean(dim=1)
            z_vid = self.proj(vid_emb)
            z_vid = nn.functional.normalize(z_vid, dim=-1)
            
            # 同时返回两种表示
            return z_seq, z_vid
        else:
            # 只返回视频级表示 -> [B, proj_dim]
            vid_emb = x.mean(dim=1)
            z = self.proj(vid_emb)
            z = nn.functional.normalize(z, dim=-1)
            return z