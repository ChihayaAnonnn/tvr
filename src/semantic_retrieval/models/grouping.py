"""Symmetric query-conditioned equivalence and deterministic complete-link grouping."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class EquivalencePredictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(nn.Linear(3 * feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, 1))

    def forward(self, a, b):
        if (
            a.shape != b.shape
            or a.ndim != 2
            or a.shape[-1] != self.feature_dim
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("invalid equivalence features")
        return self.network(torch.cat([a + b, (a - b).abs(), a * b], -1)).squeeze(-1)


def group_loss(logits, labels):
    if (
        logits.shape != labels.shape
        or logits.ndim != 1
        or not torch.isfinite(logits).all()
        or not ((labels == -1) | (labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("invalid group supervision")
    known = labels >= 0
    return F.binary_cross_entropy_with_logits(logits[known], labels[known].float()) if known.any() else logits.sum() * 0


def conservative_groups(ids, probabilities, threshold):
    p = torch.as_tensor(probabilities)
    if (
        len(set(ids)) != len(ids)
        or p.shape != (len(ids), len(ids))
        or not torch.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not torch.allclose(p, p.T)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid equivalence matrix/threshold")
    positions = {i: j for j, i in enumerate(ids)}
    groups = []
    for i in sorted(ids):
        group = next((g for g in groups if all(p[positions[i], positions[j]] >= threshold for j in g)), None)
        if group is None:
            groups.append([i])
        else:
            group.append(i)
    return tuple(tuple(g) for g in groups)


class ValidatedGrouper:
    """Configured validation evidence, never caller-supplied online labels."""

    def __init__(self, predictor, config):
        if hashlib.sha256(config.validation_report.read_bytes()).hexdigest() != config.validation_report_sha256:
            raise ValueError("grouping validation report hash mismatch")
        if predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError("group predictor must be frozen")
        self.predictor = predictor
        self.config = config
        self.validated = True

    @torch.no_grad()
    def groups(self, ids, pair_scores):
        n = len(ids)
        p = pair_scores.logits.new_ones(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                p[i, j] = p[j, i] = self.predictor(
                    pair_scores.features[i : i + 1], pair_scores.features[j : j + 1]
                ).sigmoid()[0]
        return conservative_groups(ids, p, self.config.threshold)
