"""Offline analysis of the score matrices an RSPR evaluation dumped.

Two questions about reranking cannot be answered from the headline metrics:

* Does the pair uncertainty predict whether a retrieval is right?
* Does reranking use the matcher's *information*, or would any perturbation of
  the same magnitude reorder the near-ties just as well?

Both are pure functions of the dumped matrices, so they belong here rather than
in another GPU pass per question.
"""

from __future__ import annotations

import numpy as np


def _scored_masks(dump: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.isfinite(dump["t2v"]), np.isfinite(dump["v2t"])


def recover_rerank_coefficient(dump: dict, tolerance: float = 1e-4) -> float:
    """Return the single factor the run applied to the matcher score.

    The evaluation writes ``S_det/T_d + lambda * c * S_prob/T_p`` but not the
    factor itself, and hand-typing it risks replaying a control at a scale the
    run never used. Recovering it from the scores makes that impossible.
    """

    differences, probabilities = [], []
    for final, mask in zip((dump["t2v"], dump["v2t"]), _scored_masks(dump)):
        differences.append((final[mask] - dump["recall"][mask]).astype(np.float64))
        probabilities.append(dump["probability"][mask].astype(np.float64))
    difference = np.concatenate(differences)
    probability = np.concatenate(probabilities)
    if difference.size == 0 or not np.any(probability):
        raise ValueError("no scored pairs carry a usable rerank coefficient")
    coefficient = float(probability @ difference / (probability @ probability))
    # Judge the fit in score units: a per-pair ratio explodes wherever the
    # matcher output is near zero, which says nothing about the coefficient.
    residual = float(np.max(np.abs(difference - coefficient * probability)))
    scale = max(1.0, float(np.max(np.abs(difference))))
    if residual > tolerance * scale:
        raise ValueError(
            "scores are not a single affine function of the matcher output; "
            f"coefficient leaves a residual of {residual:.3e}"
        )
    return coefficient


def replay_rerank(
    dump: dict,
    probability: np.ndarray,
    coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescore the recorded candidate pairs with a substitute matcher output."""

    t2v_mask, v2t_mask = _scored_masks(dump)
    outputs = []
    for mask in (t2v_mask, v2t_mask):
        replayed = np.full(mask.shape, -np.inf)
        replayed[mask] = dump["recall"][mask] + coefficient * probability[mask]
        outputs.append(replayed)
    return outputs[0], outputs[1]


def permute_probability_within_candidates(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    axis: int = 1,
) -> np.ndarray:
    """Shuffle each ranking list's matcher scores among its own candidates.

    Keeps the per-list score distribution — and therefore the magnitude of the
    reordering — while destroying which candidate each score belongs to. A gain
    that survives this came from tie-breaking, not from the matcher.
    """

    generator = np.random.default_rng(seed)
    working = probability if axis == 1 else probability.T
    working_mask = mask if axis == 1 else mask.T
    output = np.full_like(working, np.nan)
    for index in range(working.shape[0]):
        selected = np.flatnonzero(working_mask[index])
        if selected.size == 0:
            continue
        output[index, selected] = generator.permutation(working[index, selected])
    return output if axis == 1 else output.T


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a positive outranks a negative, ties counted as half."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes present")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def count_rerank_flips(
    baseline: np.ndarray,
    reranked: np.ndarray,
) -> tuple[int, int]:
    """Return how many top-1 hits reranking repaired and how many it broke."""

    truth = np.arange(baseline.shape[0])
    before = np.argmax(baseline, axis=1)
    after = np.argmax(reranked, axis=1)
    repaired = int(((before != truth) & (after == truth)).sum())
    broken = int(((before == truth) & (after != truth)).sum())
    return repaired, broken


def candidate_recall_ceiling(scored: np.ndarray) -> float:
    """Fraction of queries whose true video survived into the reranked pool."""

    truth = np.arange(scored.shape[0])
    return float(scored[truth, truth].mean())


def top1_uncertainty_and_correctness(
    scores: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the uncertainty of each query's own top-1 hit, plus whether it is right."""

    top1 = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    return uncertainty[rows, top1], top1 == rows
