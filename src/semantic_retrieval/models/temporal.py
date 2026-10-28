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

    def forward(self, frames, mask, *, position_ids=None):
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
        if position_ids is not None:
            if position_ids.dtype != torch.long or position_ids.shape != mask.shape or position_ids.device != frames.device:
                raise ValueError("invalid temporal position identities")
            valid = position_ids[mask]
            if ((valid < 0) | (valid >= self.max_frames)).any():
                raise ValueError("temporal position out of capacity")
            if any(len(torch.unique(row[m])) != int(m.sum()) for row, m in zip(position_ids, mask)):
                raise ValueError("duplicate valid temporal positions")
        if not frames.shape[0]:
            return frames, frames.new_empty((0, self.dim))
        with torch.autocast(device_type=frames.device.type, enabled=False):
            x = frames.float().masked_fill(~mask[..., None], 0)
            if self.transformer is not None:
                x = self.transformer(x + (self.position[: x.shape[1]] if position_ids is None else self.position[position_ids.masked_fill(~mask, 0)]), src_key_padding_mask=~mask)
            x = x.masked_fill(~mask[..., None], 0)
            return x, x.sum(1) / mask.sum(1, keepdim=True)
