"""Query-conditioned Gaussian scalar scores participate in ranking."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from semantic_retrieval.models.pair_matcher import PairMatcher, validate_offsets
from semantic_retrieval.evidence.types import PairPredictiveScores, positive


def validate_predictive_inputs(loc, scale, offsets, noise, temperature):
    positive(temperature)
    if loc.ndim != 1 or scale.shape != loc.shape or noise.ndim != 2 or noise.shape[1] != len(loc) or not noise.shape[0]:
        raise ValueError('invalid score/MC shapes')
    validate_offsets(offsets, len(loc))
    if any(not torch.isfinite(x).all() for x in (loc, scale, noise)) or not (scale > 0).all():
        raise ValueError('nonfinite score or nonpositive scale')


def mc_log_probabilities(loc, scale, offsets, noise, *, temperature):
    validate_predictive_inputs(loc, scale, offsets, noise, temperature)
    segments = []
    for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        if a == b:
            segments.append(loc[a:b]+scale[a:b]*0)
            continue
        scores = (loc[None, a:b].float()+scale[None, a:b].float()*noise[:, a:b].float())/temperature
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite sampled score')
        segments.append(torch.logsumexp(scores.log_softmax(-1), 0)-math.log(len(noise)))
    return torch.cat(segments)


class GaussianPairMatcher(nn.Module):
    def __init__(self, dim, *, scale_floor):
        super().__init__()
        positive(scale_floor)
        self.base = PairMatcher(dim)
        self.feature_dim = self.base.feature_dim
        self.scale_floor = scale_floor
        self.scale_head = nn.Sequential(nn.Linear(self.feature_dim+4, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, features, pairs, offsets, *, pair_ids, coverage, sampling):
        base = self.base(features, pairs)
        if coverage.shape != (len(pairs), 4) or not torch.isfinite(coverage).all():
            raise ValueError('four observable coverage scalars required per pair')
        scale = F.softplus(self.scale_head(torch.cat((base.features, coverage), -1)).squeeze(-1))+self.scale_floor
        noise = sampling.noise(pair_ids, device=scale.device, dtype=scale.dtype)
        logp = mc_log_probabilities(base.logits, scale, offsets, noise, temperature=sampling.config.temperature)
        return PairPredictiveScores(base.logits, scale, base.features, logp, pair_ids, sampling.identity)
