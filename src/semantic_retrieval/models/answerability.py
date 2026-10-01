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


def score_statistics(logits):
    """Eight preregistered statistics, temperature one, population dispersion."""
    if logits.ndim != 1 or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("score statistics require a nonempty finite row")
    top = logits.topk(min(2, len(logits))).values
    second = top[1] if len(top) > 1 else top[0]
    log_probs = logits.log_softmax(0)
    entropy = -(log_probs.exp() * log_probs).sum()
    if len(logits) > 1:
        entropy = entropy / logits.new_tensor(len(logits)).log()
    else:
        entropy = entropy * 0
    return torch.stack(
        [
            top[0],
            second,
            top[0] - second,
            logits.mean(),
            logits.std(unbiased=False),
            entropy,
            logits.logsumexp(0),
            logits.new_tensor(1 + len(logits)).log(),
        ]
    )


class ScoreOnlyAnswerabilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 2880), nn.GELU(), nn.Linear(2880, 2880), nn.GELU(), nn.Linear(2880, 1)
        )

    def forward(self, pair_scores, query_offsets):
        validate_offsets(query_offsets, len(pair_scores.logits))
        out = []
        for start, end in zip(query_offsets[:-1].tolist(), query_offsets[1:].tolist()):
            if start == end:
                out.append(pair_scores.logits.detach().sum() * 0 - 30)
            else:
                row = score_statistics(pair_scores.logits[start:end].detach())
                out.append(self.network(row).squeeze(-1))
        return torch.stack(out)
