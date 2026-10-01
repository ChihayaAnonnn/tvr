"""Masked temporal transformer with explicit capacity."""

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads, max_frames):
        super().__init__()
        if (
            any(type(x) is not int or x <= 0 for x in (dim, num_heads, max_frames))
            or type(num_layers) is not int
            or num_layers < 0
            or dim % num_heads
        ):
            raise ValueError("invalid temporal capacities")
        self.dim = dim
        self.max_frames = max_frames
        self.position = nn.Parameter(torch.randn(max_frames, dim) * 0.01) if num_layers else None
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False) if num_layers else None

    def forward(self, frames, mask):
        if (
            frames.ndim != 3
            or frames.shape[-1] != self.dim
            or not 0 < frames.shape[1] <= self.max_frames
            or mask.dtype != torch.bool
            or mask.shape != frames.shape[:2]
            or not torch.isfinite(frames).all()
        ):
            raise ValueError("invalid temporal input")
        if frames.shape[0] and not mask.any(-1).all():
            raise ValueError("empty temporal row")
        if not frames.shape[0]:
            return frames, frames.new_empty((0, self.dim))
        with torch.autocast(device_type=frames.device.type, enabled=False):
            x = frames.float().masked_fill(~mask[..., None], 0)
            if self.transformer is not None:
                x = self.transformer(x + self.position[: x.shape[1]], src_key_padding_mask=~mask)
            x = x.masked_fill(~mask[..., None], 0)
            return x, x.sum(1) / mask.sum(1, keepdim=True)
