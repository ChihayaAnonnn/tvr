"""Metrics for retrieval-conditioned failure risk."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def build_eval_positive_mask(
    num_text: int,
    num_video: int,
    cut_off_points: Sequence[int] | None = None,
) -> np.ndarray:
    """Map evaluation text/video rows to all valid positives."""

    if num_text <= 0 or num_video <= 0:
        raise ValueError("num_text and num_video must be positive")
    if cut_off_points is None:
        if num_text != num_video:
            raise ValueError(
                "one-to-one evaluation requires equal text and video counts"
            )
        return np.eye(num_text, dtype=bool)

    cutoffs = [int(value) for value in cut_off_points]
    if len(cutoffs) != num_video:
        raise ValueError("cut_off_points length must equal num_video")
    if not cutoffs or cutoffs[-1] != num_text:
        raise ValueError("final cut_off_point must equal num_text")
    if any(end <= start for start, end in zip([0] + cutoffs[:-1], cutoffs)):
        raise ValueError("cut_off_points must be strictly increasing")

    mask = np.zeros((num_text, num_video), dtype=bool)
    for video_index, (start, end) in enumerate(
        zip([0] + cutoffs[:-1], cutoffs)
    ):
        mask[start:end, video_index] = True
    return mask


def positive_ranks(scores: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    """Return zero-based rank of the best positive for every query."""

    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(positive_mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != mask.shape:
        raise ValueError("scores and positive_mask must be equal 2-D matrices")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not mask.any(axis=1).all():
        raise ValueError("each query requires at least one positive")

    best_positive = np.where(mask, scores, -np.inf).max(axis=1)
    return (scores > best_positive[:, None]).sum(axis=1).astype(np.int64)


def _binary_auroc(failure: np.ndarray, risk: np.ndarray) -> float:
    failure = np.asarray(failure, dtype=bool)
    risk = np.asarray(risk, dtype=np.float64)
    num_positive = int(failure.sum())
    num_negative = int((~failure).sum())
    if num_positive == 0 or num_negative == 0:
        raise ValueError("AUROC requires both success and failure samples")
    ranks = rankdata(risk, method="average")
    positive_rank_sum = float(ranks[failure].sum())
    mann_whitney = (
        positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    )
    return mann_whitney / (num_positive * num_negative)


def _bootstrap_auroc(
    failure: np.ndarray,
    risk: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    value = _binary_auroc(failure, risk)
    if samples == 0:
        return value, value

    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(failure)
    negative = np.flatnonzero(~failure)
    bootstrapped = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected_positive = rng.choice(
            positive, size=positive.size, replace=True
        )
        selected_negative = rng.choice(
            negative, size=negative.size, replace=True
        )
        selected = np.concatenate((selected_negative, selected_positive))
        bootstrapped[index] = _binary_auroc(
            failure[selected], risk[selected]
        )
    low, high = np.percentile(bootstrapped, [2.5, 97.5])
    return float(low), float(high)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _risk_coverage(ranks: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    order = np.argsort(risk, kind="stable")
    correct = (ranks[order] == 0).astype(np.float64)
    count = np.arange(1, correct.size + 1, dtype=np.float64)
    accuracy = np.cumsum(correct) / count
    coverage = count / correct.size
    error = 1.0 - accuracy
    integrate = getattr(np, "trapezoid", np.trapz)
    payload = {"aurc": float(integrate(error, coverage))}
    for point in (0.25, 0.50, 0.75, 1.00):
        kept = max(1, int(math.ceil(point * correct.size)))
        payload[f"accuracy_at_{point:.2f}"] = float(accuracy[kept - 1])
    return payload


def evaluate_retrieval_risk(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    risk: np.ndarray,
    vacuity: np.ndarray,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Aggregate risk quality without altering the retrieval scores."""

    ranks = positive_ranks(scores, positive_mask)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    vacuity = np.asarray(vacuity, dtype=np.float64).reshape(-1)
    if risk.shape != ranks.shape or vacuity.shape != ranks.shape:
        raise ValueError("risk and vacuity must have one value per query")
    if not np.isfinite(risk).all() or not np.isfinite(vacuity).all():
        raise ValueError("risk and vacuity must be finite")

    failure = ranks > 0
    auroc = _binary_auroc(failure, risk)
    ci_low, ci_high = _bootstrap_auroc(
        failure, risk, int(bootstrap_samples), int(seed)
    )
    spearman = spearmanr(risk, ranks).statistic
    if not np.isfinite(spearman):
        raise ValueError("Spearman correlation requires non-constant inputs")

    return {
        "num_queries": int(ranks.size),
        "num_failures": int(failure.sum()),
        "spearman_risk_vs_rank": float(spearman),
        "failure_auroc": {
            "value": float(auroc),
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "risk_coverage": _risk_coverage(ranks, risk),
        "risk_stats": _summary(risk),
        "vacuity_stats": _summary(vacuity),
    }
