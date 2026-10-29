"""Spatial slots; neither object identities nor cross-frame tracks."""
import torch
from torch import nn
import torch.nn.functional as F


def window_indices(device=None):
    return torch.tensor([[7*r+c for r in range(a-1,a+2) for c in range(b-1,b+2)]
                         for a in (1,3,5) for b in (1,3,5)], dtype=torch.long, device=device)


def lexical_mask(text_mask):
    if (not isinstance(text_mask, torch.Tensor) or text_mask.dtype != torch.bool
            or text_mask.ndim != 2 or min(text_mask.shape) < 1):
        raise ValueError('invalid lexical mask')
    positions = torch.arange(text_mask.shape[1], device=text_mask.device)[None]
    lengths = text_mask.sum(-1)[:, None]
    if not torch.equal(text_mask, positions < lengths):
        raise ValueError('text mask must be contiguous')
    result = text_mask & (positions > 0) & (positions < lengths-1)
    if not result.any(-1).all(): raise ValueError('local query requires lexical tokens')
    return result


class LocalQueryTokens(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        if type(dim) is not int or dim <= 0 or dim % 8:
            raise ValueError('invalid local attention dimension')
        self.query = nn.Parameter(torch.randn(9, dim) * dim**-0.5)
        self.attention = nn.MultiheadAttention(dim, 8, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer('indices', window_indices(), persistent=False)

    def forward(self, patches, frame_mask):
        if (not isinstance(patches, torch.Tensor) or patches.ndim != 4
                or patches.shape[2:] != (49, self.query.shape[-1])
                or not patches.is_floating_point() or not torch.isfinite(patches).all()
                or not isinstance(frame_mask, torch.Tensor) or frame_mask.dtype != torch.bool
                or frame_mask.shape != patches.shape[:2] or frame_mask.device != patches.device
                or min(frame_mask.shape) < 1 or not frame_mask.any(-1).all()):
            raise ValueError('invalid local patch/frame tensors')
        b,f,_,d = patches.shape
        patches = patches.masked_fill(~frame_mask[..., None, None], 0)
        kv = patches[:, :, self.indices, :].reshape(b*f*9, 9, d)
        q = self.query[None, None].expand(b,f,-1,-1).reshape(b*f*9,1,d)
        u,_ = self.attention(q,kv,kv,need_weights=False)
        out = F.normalize(self.norm(q+u),dim=-1,eps=1e-6).reshape(b,f,9,d)
        return out.masked_fill(~frame_mask[...,None,None],0)
