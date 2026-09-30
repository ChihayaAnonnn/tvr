"""Validated calibration and paired bootstrap, migrated from retrieval_risk_metrics."""

import math

import numpy as np
from scipy.stats import rankdata


def _binary_auroc(failure: np.ndarray, risk: np.ndarray) -> float:
    failure = np.asarray(failure, dtype=bool)
    risk = np.asarray(risk, dtype=np.float64)
    num_positive = int(failure.sum())
    num_negative = int((~failure).sum())
    if num_positive == 0 or num_negative == 0:
        raise ValueError("AUROC requires both success and failure samples")
    ranks = rankdata(risk, method="average")
    positive_rank_sum = float(ranks[failure].sum())
    mann_whitney = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return mann_whitney / (num_positive * num_negative)


def _validate_binary_inputs(labels: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(labels).reshape(-1)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    if raw_labels.shape != risk.shape:
        raise ValueError("labels and risk must have equal shape")
    if raw_labels.size == 0:
        raise ValueError("labels and risk must not be empty")
    if not np.isin(raw_labels, (0, 1, False, True)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(risk).all():
        raise ValueError("risk must be finite")
    failure = raw_labels.astype(bool)
    if failure.all() or not failure.any():
        raise ValueError("metrics require both success and failure samples")
    return failure, risk


def binary_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    """Return fixed-bin ECE, Brier score, and binary NLL."""

    raw_labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if raw_labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal shape")
    if raw_labels.size == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not np.isin(raw_labels, (0, 1, False, True)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if isinstance(bins, bool) or int(bins) != bins or bins <= 0:
        raise ValueError("bins must be a positive integer")

    failure = raw_labels.astype(np.float64)
    bin_index = np.minimum((probabilities * int(bins)).astype(int), int(bins) - 1)
    ece = 0.0
    for index in range(int(bins)):
        selected = bin_index == index
        if selected.any():
            ece += float(selected.mean()) * abs(float(probabilities[selected].mean()) - float(failure[selected].mean()))
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return {
        "ece": float(ece),
        "brier": float(np.mean((probabilities - failure) ** 2)),
        "nll": float(-np.mean(failure * np.log(clipped) + (1.0 - failure) * np.log1p(-clipped))),
    }


def _paired_metric_value(labels: np.ndarray, risk: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return float(_binary_auroc(labels, risk))
    if metric == "aurc":
        order = np.argsort(risk, kind="stable")
        count = np.arange(1, labels.size + 1, dtype=np.float64)
        return float(np.mean(np.cumsum(labels[order]) / count))
    raise ValueError("metric must be 'auroc' or 'aurc'")


def paired_bootstrap_delta(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Return a deterministic stratified paired-bootstrap ``first-second`` delta."""

    failure, first = _validate_binary_inputs(labels, first)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if second.shape != first.shape:
        raise ValueError("labels, first, and second must have equal shape")
    if not np.isfinite(second).all():
        raise ValueError("first and second risks must be finite")
    if metric not in {"auroc", "aurc"}:
        raise ValueError("metric must be 'auroc' or 'aurc'")
    if isinstance(samples, bool) or int(samples) != samples or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    samples = int(samples)
    seed = int(seed)

    point_delta = _paired_metric_value(failure, first, metric) - _paired_metric_value(failure, second, metric)
    rng = np.random.default_rng(seed)
    positive_positions = np.flatnonzero(failure)
    negative_positions = np.flatnonzero(~failure)
    selected = np.empty(failure.size, dtype=np.int64)
    deltas = np.empty(samples, dtype=np.float64)
    valid = 0
    for _ in range(samples):
        selected[positive_positions] = rng.choice(positive_positions, size=positive_positions.size, replace=True)
        selected[negative_positions] = rng.choice(negative_positions, size=negative_positions.size, replace=True)
        delta = _paired_metric_value(failure, first[selected], metric) - (
            _paired_metric_value(failure, second[selected], metric)
        )
        if np.isfinite(delta):
            deltas[valid] = delta
            valid += 1
    minimum_valid = int(math.ceil(0.95 * samples))
    if valid < minimum_valid:
        raise ValueError("fewer than 95% of paired bootstrap replicates are valid")
    low, high = np.percentile(deltas[:valid], [2.5, 97.5])
    return {
        "metric": metric,
        "point_delta": float(point_delta),
        "ci_low": float(low),
        "ci_high": float(high),
        "samples": samples,
        "seed": seed,
        "valid_replicates": valid,
    }


def binary_metrics(labels, probabilities, *, bins=15):
    result = binary_calibration_metrics(labels, probabilities, bins=bins)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    result.update(num_positive=int(labels.sum()), num_negative=int((~labels).sum()))
    result["auroc"] = _binary_auroc(labels, probabilities) if labels.any() and not labels.all() else None
    return result
