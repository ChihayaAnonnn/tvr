"""Permutation-invariant candidate presence from ranking features."""

import torch
from torch import nn

from semantic_retrieval.models.pair_matcher import validate_offsets


class AnswerabilityHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(2 * (feature_dim + 1), feature_dim), nn.GELU(), nn.Linear(feature_dim, 1)
        )

    def forward(self, pair_scores, query_offsets):
        validate_offsets(query_offsets, len(pair_scores.logits))
        if pair_scores.features.shape != (len(pair_scores.logits), self.feature_dim):
            raise ValueError("incompatible pair features")
        values = torch.cat([pair_scores.features, pair_scores.logits[:, None]], -1)
        out = []
        for start, end in zip(query_offsets[:-1].tolist(), query_offsets[1:].tolist()):
            if start == end:
                out.append(pair_scores.logits.sum() * 0 - 30)
            else:
                row = values[start:end]
                out.append(self.network(torch.cat([row.mean(0), row.max(0).values])).squeeze(-1))
        return torch.stack(out)
