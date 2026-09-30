"""Identity-stable ranking with explicit annotation eligibility."""

import numpy as np


def _validate(scores, candidate_ids):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 2-D matrix")
    ids = list(candidate_ids)
    if len(ids) != scores.shape[1] or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique strings matching the gallery")
    return scores, np.asarray(ids, dtype=str)


def positive_ranks(scores, positive_mask, candidate_ids):
    scores, ids = _validate(scores, candidate_ids)
    mask = np.asarray(positive_mask)
    if mask.dtype != bool or mask.shape != scores.shape or not mask.any(axis=1).all():
        raise ValueError("each query requires a boolean positive mask with at least one positive")
    ranks = []
    for row, positive in zip(scores, mask):
        order = np.lexsort((ids, -row))
        ranks.append(int(np.flatnonzero(positive[order])[0]))
    return np.asarray(ranks, dtype=np.int64)


def retrieval_metrics(scores, relevance, candidate_ids):
    scores, _ = _validate(scores, candidate_ids)
    relevance = np.asarray(relevance)
    if relevance.shape != scores.shape or not np.isin(relevance, [-1, 0, 1]).all():
        raise ValueError("relevance must match scores and contain -1, 0, or 1")
    unknown = (relevance == -1).any(axis=1)
    has_positive = (relevance == 1).any(axis=1)
    eligible = ~unknown & has_positive
    result = dict(
        num_eligible=int(eligible.sum()),
        num_no_answer=int((~unknown & ~has_positive).sum()),
        num_unknown=int(unknown.sum()),
    )
    ranks = positive_ranks(scores[eligible], relevance[eligible] == 1, candidate_ids)
    for k in (1, 5, 10):
        result[f"R{k}"] = float(100 * np.mean(ranks < k)) if ranks.size else None
    result["MedianR"] = float(np.median(ranks) + 1) if ranks.size else None
    result["MeanR"] = float(np.mean(ranks) + 1) if ranks.size else None
    return result
