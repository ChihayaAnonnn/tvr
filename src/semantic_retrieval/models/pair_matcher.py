"""Requested-pair matching; attention and ranking share projections."""

import torch
from torch import nn
from torch.nn import functional as F

from semantic_retrieval.models.wti import MaskedWTI
from semantic_retrieval.types import PairScores


def validate_pairs(pairs, q, v):
    if (
        pairs.dtype != torch.long
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or (pairs < 0).any()
        or (pairs[:, 0] >= q).any()
        or (pairs[:, 1] >= v).any()
    ):
        raise ValueError("invalid pair indices")
    if len(pairs) != len(torch.unique(pairs, dim=0)):
        raise ValueError("duplicate pairs require canonical deduplication")


def validate_offsets(offsets, n):
    if (
        offsets.dtype != torch.long
        or offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != n
        or (offsets[1:] < offsets[:-1]).any()
    ):
        raise ValueError("invalid packed offsets")


class PairMatcher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.feature_dim = dim * 4 + 2
        self.text_projection = nn.Linear(dim, dim, bias=False)
        self.frame_projection = nn.Linear(dim, dim, bias=False)
        self.dustbin = nn.Parameter(torch.zeros(()))
        self.a_raw = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))
        self.wti = MaskedWTI(dim)
        self.mlp = nn.Sequential(nn.Linear(self.feature_dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, f, pair_indices, *, return_alignment=False):
        validate_pairs(pair_indices, len(f.pooled_text), len(f.pooled_video))
        if not len(pair_indices):
            zero = self.a_raw * 0
            return PairScores(zero.expand(0), zero.expand(0, self.feature_dim), zero.expand(0), None, None)
        qi, vi = pair_indices.unbind(1)
        tm = f.text_mask[qi]
        fm = f.frame_mask[vi]
        if not tm.any(-1).all() or not fm.any(-1).all():
            raise ValueError("matching requires valid tokens and frames")
        with torch.autocast(device_type=f.text_tokens.device.type, enabled=False):
            text = self.text_projection(f.text_tokens[qi].float())
            frames = self.frame_projection(f.frame_tokens[vi].float())
            if not torch.isfinite(text).all() or not torch.isfinite(frames).all():
                raise ValueError("nonfinite pair features")
            sim = torch.einsum("nld,ntd->nlt", F.normalize(text, dim=-1), F.normalize(frames, dim=-1))
            masked = sim.masked_fill(~fm[:, None, :], -torch.inf)
            att = torch.cat([masked, self.dustbin.expand(len(qi), text.shape[1], 1)], -1).softmax(-1) * tm[..., None]
            aligned = torch.einsum("nlt,ntd->nld", att[..., :-1], frames)
            pool = aligned.sum(1) / tm.sum(1, keepdim=True)
            dust = att[..., -1].sum(1) / tm.sum(1)
            tw = self.wti.text_weight(text).squeeze(-1).masked_fill(~tm, -torch.inf).softmax(-1)
            fw = self.wti.frame_weight(frames).squeeze(-1).masked_fill(~fm, -torch.inf).softmax(-1)
            pair_sim = sim.masked_fill(~(tm[..., None] & fm[:, None, :]), -torch.inf)
            wti = (
                (pair_sim.max(-1).values.masked_fill(~tm, 0) * tw).sum(-1)
                + (pair_sim.max(-2).values.masked_fill(~fm, 0) * fw).sum(-1)
            ) / 2
            q = self.text_projection(f.pooled_text[qi].float())
            v = self.frame_projection(f.pooled_video[vi].float())
            h = torch.cat([q, v, q * v, pool, dust[:, None], wti[:, None]], -1)
            logit = F.softplus(self.a_raw) * wti + self.b + self.mlp(h).squeeze(-1)
            return PairScores(logit, h, dust, att if return_alignment else None, sim if return_alignment else None)
