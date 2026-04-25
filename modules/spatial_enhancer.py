import torch
import torch.nn as nn

from .rala import GateLinearAttention3D, GateLinearAttentionNoSilu, RoPE, RoPE3D


class SpatialEnhancer(nn.Module):
    """
    RALA (Rotary Linear Attention) 封装模块。
    rope_mode='2d': 逐帧空间注意力，输入 [B, C, H, W]
    rope_mode='3d': 跨帧时空注意力，输入 [B, C, T, H, W]
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, rope_mode: str = '2d'):
        super().__init__()
        self.rope_mode = rope_mode
        if rope_mode == '3d':
            self.rope = RoPE3D(embed_dim, num_heads)
            self.attention = GateLinearAttention3D(embed_dim, num_heads)
        else:
            self.rope = RoPE(embed_dim, num_heads)
            self.attention = GateLinearAttentionNoSilu(embed_dim, num_heads)
        self.cache_shape = None
        self.register_buffer('cached_sin', None, persistent=False)
        self.register_buffer('cached_cos', None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        2d mode: x [B, C, H, W] -> [B, C, H, W]
        3d mode: x [B, C, T, H, W] -> [B, C, T, H, W]
        """
        device = x.device

        if self.rope_mode == '3d':
            if x.dim() != 5:
                raise ValueError(f"3D mode expects [B, C, T, H, W], got {x.shape}")
            _, _, T, H, W = x.shape
            cache_key = (T, H, W)
        else:
            if x.dim() != 4:
                raise ValueError(f"2D mode expects [B, C, H, W], got {x.shape}")
            _, _, H, W = x.shape
            cache_key = (H, W)

        if self.cache_shape != cache_key or self.cached_sin is None:
            self.cache_shape = cache_key
            sin, cos = self.rope(cache_key)
            self.cached_sin = sin.to(device)
            self.cached_cos = cos.to(device)

        if self.cached_sin.device != device:
            self.cached_sin = self.cached_sin.to(device)
            self.cached_cos = self.cached_cos.to(device)

        return self.attention(x, self.cached_sin, self.cached_cos)

