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
    *,
    return_samples: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    if samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    value = _binary_auroc(failure, risk)
    if samples == 0:
        bootstrap = np.empty(0, dtype=np.float64)
        return (value, value, bootstrap) if return_samples else (value, value)

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
    if return_samples:
        return float(low), float(high), bootstrapped
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


def _prefix_mean_aurc(ranks: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    correct = (ranks[order] == 0).astype(np.float64)
    count = np.arange(1, correct.size + 1, dtype=np.float64)
    return float(np.mean(1.0 - np.cumsum(correct) / count))


def evaluate_retrieval_risk(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    risk: np.ndarray,
    vacuity: np.ndarray | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    return_array_evidence: bool = False,
) -> dict[str, object]:
    """Aggregate risk quality without altering the retrieval scores."""

    ranks = positive_ranks(scores, positive_mask)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    if risk.shape != ranks.shape:
        raise ValueError("risk must have one value per query")
    if not np.isfinite(risk).all():
        raise ValueError("risk must be finite")
    if vacuity is not None:
        vacuity = np.asarray(vacuity, dtype=np.float64).reshape(-1)
        if vacuity.shape != ranks.shape:
            raise ValueError("vacuity must have one value per query")
        if not np.isfinite(vacuity).all():
            raise ValueError("vacuity must be finite")

    failure = ranks > 0
    auroc = _binary_auroc(failure, risk)
    bootstrap_result = _bootstrap_auroc(
        failure,
        risk,
        int(bootstrap_samples),
        int(seed),
        return_samples=return_array_evidence,
    )
    if return_array_evidence:
        ci_low, ci_high, bootstrap_values = bootstrap_result
    else:
        ci_low, ci_high = bootstrap_result
    spearman = spearmanr(risk, ranks).statistic
    if not np.isfinite(spearman):
        raise ValueError("Spearman correlation requires non-constant inputs")

    result = {
        "num_queries": int(ranks.size),
        "num_failures": int(failure.sum()),
        "spearman_risk_vs_rank": float(spearman),
        "failure_auroc": {
            "value": float(auroc),
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "aurc": _prefix_mean_aurc(ranks, risk),
        "risk_coverage": _risk_coverage(ranks, risk),
        "risk_stats": _summary(risk),
        "saturation_fraction": float(
            np.mean((risk < 1e-4) | (risk > 1.0 - 1e-4))
        ),
    }
    if vacuity is not None:
        result["vacuity_stats"] = _summary(vacuity)
    if return_array_evidence:
        result["_array_evidence"] = {
            "query_ranks": ranks,
            "bootstrap_auroc": bootstrap_values,
        }
    return result


def evaluate_intrinsic_uncertainty(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    uncertainty: np.ndarray,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Aggregate query-level intrinsic uncertainty by retrieval direction."""

    ranks = positive_ranks(scores, positive_mask)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if uncertainty.shape != ranks.shape or not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty must be finite with one value per query")

    failure = ranks > 0
    value = _binary_auroc(failure, uncertainty)
    ci_low, ci_high = _bootstrap_auroc(
        failure,
        uncertainty,
        int(bootstrap_samples),
        int(seed),
    )
    rho = spearmanr(uncertainty, ranks).statistic
    if not np.isfinite(rho):
        raise ValueError("Spearman correlation requires non-constant inputs")
    return {
        "num_queries": int(ranks.size),
        "num_failures": int(failure.sum()),
        "spearman_uncertainty_vs_rank": float(rho),
        "failure_auroc": {
            "value": float(value),
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "risk_coverage": _risk_coverage(ranks, uncertainty),
        "uncertainty_stats": _summary(uncertainty),
    }


def summarize_logvar(
    logvar: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, object]:
    """Summarize log-variance, variance, and exact clamp-bound rates."""

    source_values = np.asarray(logvar)
    if not np.issubdtype(source_values.dtype, np.floating):
        raise ValueError("logvar must have a floating-point dtype")
    if (
        source_values.ndim != 2
        or source_values.size == 0
        or not np.isfinite(source_values).all()
    ):
        raise ValueError("logvar must be a non-empty finite 2-D array")
    source_type = source_values.dtype.type
    low_fraction = np.mean(source_values == source_type(lower))
    high_fraction = np.mean(source_values == source_type(upper))
    values = source_values.astype(np.float64, copy=False)
    return {
        "logvar": _summary(values.reshape(-1)),
        "variance": _summary(np.exp(values).reshape(-1)),
        "clamp_low_fraction": float(low_fraction),
        "clamp_high_fraction": float(high_fraction),
    }


def _stream_summary(chunks) -> dict[str, float]:
    count, total, square_total = 0, 0.0, 0.0
    minimum, maximum = np.inf, -np.inf
    for chunk in chunks:
        values = np.asarray(chunk, dtype=np.float64)
        count += values.size
        total += float(values.sum())
        square_total += float(np.square(values).sum())
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    mean = total / count
    variance = max(square_total / count - mean * mean, 0.0)
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": float(np.sqrt(variance)),
    }


def summarize_csd_components(
    text_mu: np.ndarray,
    video_mu: np.ndarray,
    text_logvar: np.ndarray,
    video_logvar: np.ndarray,
    chunk_size: int = 256,
) -> dict[str, dict[str, float]]:
    """Summarize CSD mean/variance distances without a full pair matrix."""

    text_mu = np.asarray(text_mu, dtype=np.float64)
    video_mu = np.asarray(video_mu, dtype=np.float64)
    text_logvar = np.asarray(text_logvar, dtype=np.float64)
    video_logvar = np.asarray(video_logvar, dtype=np.float64)
    arrays = (text_mu, video_mu, text_logvar, video_logvar)
    if any(
        values.ndim != 2
        or values.size == 0
        or not np.isfinite(values).all()
        for values in arrays
    ):
        raise ValueError("CSD inputs must be non-empty finite 2-D arrays")
    if text_mu.shape != text_logvar.shape:
        raise ValueError("text mean/logvar shapes must match")
    if video_mu.shape != video_logvar.shape:
        raise ValueError("video mean/logvar shapes must match")
    if chunk_size <= 0 or text_mu.shape[1] != video_mu.shape[1]:
        raise ValueError("chunk_size and embedding dimensions must be valid")

    text_variance = np.exp(text_logvar).sum(1)
    video_variance = np.exp(video_logvar).sum(1)

    def mean_chunks():
        video_norm = np.square(video_mu).sum(1)[None, :]
        for start in range(0, len(text_mu), chunk_size):
            text = text_mu[start : start + chunk_size]
            yield np.maximum(
                np.square(text).sum(1)[:, None]
                + video_norm
                - 2.0 * text @ video_mu.T,
                0.0,
            )

    def variance_chunks():
        for start in range(0, len(text_variance), chunk_size):
            yield (
                text_variance[start : start + chunk_size, None]
                + video_variance[None, :]
            )

    return {
        "mean_distance": _stream_summary(mean_chunks()),
        "variance_distance": _stream_summary(variance_chunks()),
    }
