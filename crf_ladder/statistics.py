"""Pre-registered statistics for the CRF ladder evaluation.

The functions here are the ONLY authorised statistics for the CRF-audit /
challenge / MSVD reports. Every numerical detail (tie handling, log base,
bootstrap resampling unit, saturation threshold) matches Task 8 of the CRF
degradation ladder plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float


def paired_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of positions where ``a < b``. Ties count as failures.

    "Pre-registered" means we do not silently score ties as 0.5 — a tied
    forecast is not a successful ranking; treating it as one would leak
    optimism into every gate.
    """

    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must share shape")
    if a.size == 0:
        raise ValueError("paired_accuracy requires at least one element")
    return float((a < b).mean())


def laplacian_video_score(frames_uint8_rgb: np.ndarray) -> float:
    """Baseline "sharpness" score used to falsify the CRF-audit gate.

    Formula (from the plan):
        Y     = 0.299 * R + 0.587 * G + 0.114 * B      # BT.601 luma
        var   = cv2.Laplacian(Y, CV_64F, ksize=3).var()   # per-frame
        score = -log( median_over_frames(var) + 1e-8 )

    Higher score => *smoother* video => *more* apparent compression damage
    (from this baseline's perspective). The audit compares u against this
    score; the alternative-hypothesis gate requires (u - Laplacian)
    bootstrap CI low > 0 on challenge pairs.
    """

    frames = np.asarray(frames_uint8_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"laplacian_video_score expects [T, H, W, 3] uint8; got {frames.shape}"
        )
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover — cv2 pinned in requirements
        raise RuntimeError("laplacian_video_score requires opencv-python") from exc
    frames_f = frames.astype(np.float64)
    y = (
        0.299 * frames_f[..., 0]
        + 0.587 * frames_f[..., 1]
        + 0.114 * frames_f[..., 2]
    )
    variances = np.array(
        [cv2.Laplacian(y_frame, cv2.CV_64F, ksize=3).var() for y_frame in y]
    )
    median_var = float(np.median(variances))
    return float(-np.log(median_var + 1e-8))


def bootstrap_video_accuracy(
    correct_by_video: np.ndarray,
    seed: int,
    samples: int,
    ci: float = 0.95,
) -> BootstrapResult:
    """Video-row bootstrap of the mean accuracy.

    ``correct_by_video`` has shape ``[V, K]`` where each row is one video's K
    boolean/{0,1} correctness values (for K challenge pairs, for instance).
    Resampling is at the VIDEO row level — never at the flattened pair level
    — so per-video correlation between pairs is preserved. For paired
    ``delta_acc(u, baseline)`` callers reuse the same seed on both bootstrap
    calls; identical resample indices then flow through both arrays,
    coupling the CI to the same replicate structure.
    """

    correct = np.asarray(correct_by_video, dtype=float)
    if correct.ndim != 2:
        raise ValueError(
            f"correct_by_video must be [V, K]; got shape {correct.shape}"
        )
    if correct.size == 0:
        raise ValueError("correct_by_video is empty")
    V = correct.shape[0]
    rng = np.random.default_rng(int(seed))
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, V, size=V)
        means[i] = correct[idx].mean()
    alpha = (1.0 - float(ci)) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    point = float(correct.mean())
    return BootstrapResult(point=point, low=low, high=high)


def saturation_rate(u: np.ndarray) -> float:
    """Fraction of samples within 1% of the min or max of ``u``."""

    u = np.asarray(u, dtype=float).ravel()
    if u.size == 0:
        return 0.0
    umin = float(u.min())
    umax = float(u.max())
    span = umax - umin
    if span == 0.0:
        return 1.0
    threshold = 0.01 * span
    near_low = np.abs(u - umin) < threshold
    near_high = np.abs(u - umax) < threshold
    return float((near_low | near_high).mean())


def extrapolation_readable(
    u34: np.ndarray,
    u42: np.ndarray,
    u50: np.ndarray,
    sat_rate: float,
) -> bool:
    """Pre-registered readability gate for CRF50 extrapolation.

        readable = median(u50 - u42) >= 0.2 * median(u42 - u34) and sat_rate <= 0.10
    """
    gap_low = float(np.median(np.asarray(u42) - np.asarray(u34)))
    gap_high = float(np.median(np.asarray(u50) - np.asarray(u42)))
    return gap_high >= 0.2 * gap_low and float(sat_rate) <= 0.10


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ between two 1-D arrays. Exploratory ONLY — never a gate.

    Kept in this module so plots / reports can use one implementation, but
    reports must tag it ``exploratory: true`` per the plan.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("spearman_correlation requires equal-shape arrays of size >= 2")
    ranks_a = _rankdata_average(a)
    ranks_b = _rankdata_average(b)
    ranks_a -= ranks_a.mean()
    ranks_b -= ranks_b.mean()
    denom = float(np.sqrt((ranks_a ** 2).sum() * (ranks_b ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((ranks_a * ranks_b).sum() / denom)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-rank helper (dependency-free equivalent of scipy.stats.rankdata)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Average ranks over ties.
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # 1-indexed
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks
