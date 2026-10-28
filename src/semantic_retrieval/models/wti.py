"""Padding-safe weighted token/frame max matching."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedWTI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.text_weight = nn.Linear(dim, 1)
        self.frame_weight = nn.Linear(dim, 1)

    def forward(self, text, frames, text_mask, frame_mask):
        for name, values, mask in [("text", text, text_mask), ("frames", frames, frame_mask)]:
            if (
                values.ndim != 3
                or values.shape[-1] != self.dim
                or not values.is_floating_point()
                or not torch.isfinite(values).all()
            ):
                raise ValueError(f"{name} must be finite [batch,length,dim] floating features")
            if mask.dtype != torch.bool or mask.shape != values.shape[:2] or not mask.any(1).all():
                raise ValueError(f"{name} requires a nonempty boolean mask per row")
        if text.device != frames.device or text_mask.device != text.device or frame_mask.device != frames.device:
            raise ValueError("matching inputs must share a device")
        with torch.autocast(device_type=text.device.type, enabled=False):
            text, frames = text.float(), frames.float()
            tw = self.text_weight(text).squeeze(-1).masked_fill(~text_mask, -torch.inf).softmax(-1)
            fw = self.frame_weight(frames).squeeze(-1).masked_fill(~frame_mask, -torch.inf).softmax(-1)
            similarities = torch.einsum("qld,vtd->qvlt", F.normalize(text, dim=-1), F.normalize(frames, dim=-1))
            pair_mask = text_mask[:, None, :, None] & frame_mask[None, :, None, :]
            similarities = similarities.masked_fill(~pair_mask, -torch.inf)
            text_max = similarities.max(-1).values.masked_fill(~text_mask[:, None, :], 0)
            frame_max = similarities.max(-2).values.masked_fill(~frame_mask[None, :, :], 0)
            return ((text_max * tw[:, None, :]).sum(-1) + (frame_max * fw[None, :, :]).sum(-1)) / 2
