"""Deterministic score-only failure-risk baselines for retrieval."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def _query_scores(scores: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if direction not in {"t2v", "v2t"}:
        raise ValueError("direction must be 't2v' or 'v2t'")
    if values.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    oriented = values if direction == "t2v" else values.T
    if oriented.shape[0] == 0:
        raise ValueError("scores must contain at least one query")
    if oriented.shape[1] < 2:
        raise ValueError("each query must have at least two candidates")
    if not np.isfinite(oriented).all():
        raise ValueError("scores must be finite")
    return oriented


def score_risk_baselines(
    scores: np.ndarray, direction: str
) -> dict[str, np.ndarray]:
    """Return query-level risks, with larger values meaning greater risk.

    All confidence-shaped quantities are mapped to ``[0, 1]`` so that their
    calibration metrics have a direct probability interpretation.  Energy is
    intentionally shift-sensitive; the other four baselines are invariant to
    adding a constant to every candidate score of a query.
    """

    queries = _query_scores(scores, direction)
    order = np.argsort(-queries, axis=1, kind="stable")
    top_two = np.take_along_axis(queries, order[:, :2], axis=1)
    margin = top_two[:, 0] - top_two[:, 1]

    maximum = queries.max(axis=1, keepdims=True)
    exponentiated = np.exp(queries - maximum)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    top_probability = np.take_along_axis(
        probabilities, order[:, :1], axis=1
    )[:, 0]
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    entropy_terms[positive] = -probabilities[positive] * np.log(
        probabilities[positive]
    )
    normalized_entropy = entropy_terms.sum(axis=1) / np.log(queries.shape[1])

    variance = np.var(queries, axis=1, dtype=np.float64)
    risks = {
        "one_minus_top1": 1.0 - top_probability,
        "negative_margin": expit(-margin),
        "entropy": normalized_entropy,
        "negative_energy": expit(-logsumexp(queries, axis=1)),
        "score_variance": 1.0 / (1.0 + variance),
    }
    if any(not np.isfinite(values).all() for values in risks.values()):
        raise ValueError("derived risk baselines must be finite")
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in risks.items()
    }
