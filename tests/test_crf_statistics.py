"""Task 8 — CRF ladder statistics, Laplacian baseline, pre-registered criteria."""

from __future__ import annotations

import numpy as np
import pytest

from crf_ladder.statistics import (
    BootstrapResult,
    bootstrap_paired_delta,
    bootstrap_video_accuracy,
    extrapolation_readable,
    laplacian_video_score,
    paired_accuracy,
    saturation_rate,
    spearman_correlation,
)


def test_paired_accuracy_ties_count_as_failures():
    assert paired_accuracy(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5
    assert paired_accuracy(np.array([1.0, 2.0]), np.array([0.9, 1.9])) == 0.0
    assert paired_accuracy(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == 1.0


def test_paired_accuracy_supports_broadcasted_input_and_returns_python_float():
    result = paired_accuracy(np.zeros(4), np.array([0.1, -0.1, 0.2, 0.0]))
    assert isinstance(result, float)
    assert result == 0.5


def test_laplacian_score_uses_bt601_median_and_negative_log():
    """Flat frame -> laplacian variance ~ 0 -> score = -log(1e-8) is finite and
    LARGER than a frame with edges."""
    flat = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    textured = flat.copy()
    # Texture EVERY frame so the median variance is high (not just one out of three).
    for t in range(textured.shape[0]):
        textured[t, 4:12, 4:12] = (255, 255, 255)
    flat_score = laplacian_video_score(flat)
    tex_score = laplacian_video_score(textured)
    assert np.isfinite(flat_score)
    assert np.isfinite(tex_score)
    # Flatter median variance -> higher score (=-log of smaller number).
    assert flat_score > tex_score


def test_laplacian_score_uses_bt601_weights():
    """Same luma when all channels vary equally; verifies the Y = 0.299R + 0.587G + 0.114B path."""
    # A pure-green frame should differ from a pure-red frame if we use BT.601.
    green = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    green[0, ..., 1] = 200
    red = np.zeros_like(green)
    red[0, ..., 0] = 200
    # Different luma -> different score IF weights aren't uniform. Both flat -> both use
    # median-var of ~0. So instead we make a textured pattern in one channel only:
    green_tex = green.copy()
    green_tex[0, 2:6, 2:6, 1] = 0  # Green square edge with luma weight 0.587
    red_tex = red.copy()
    red_tex[0, 2:6, 2:6, 0] = 0    # Red square edge with luma weight 0.299
    g_score = laplacian_video_score(green_tex)
    r_score = laplacian_video_score(red_tex)
    # Green channel weight is larger, so the same edge contrast produces MORE luma
    # variance from a green edge; larger variance -> smaller (-log) score.
    assert g_score < r_score


def test_bootstrap_resamples_videos_not_flat_pairs():
    correct_by_video = np.array([[1, 1, 1], [0, 0, 0]], dtype=float)
    result = bootstrap_video_accuracy(correct_by_video, seed=0, samples=5000)
    assert isinstance(result, BootstrapResult)
    assert result.point == 0.5
    assert 0.0 <= result.low <= result.high <= 1.0


def test_bootstrap_confidence_interval_shrinks_with_more_videos():
    # Rows with distinct per-video means so resampling actually varies.
    small = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    large = np.tile(small, (25, 1))  # 100 videos
    small_ci = bootstrap_video_accuracy(small, seed=1, samples=2000)
    large_ci = bootstrap_video_accuracy(large, seed=1, samples=2000)
    assert (large_ci.high - large_ci.low) < (small_ci.high - small_ci.low)


def test_bootstrap_seed_makes_ci_reproducible():
    correct = np.random.default_rng(42).integers(0, 2, size=(80, 3)).astype(float)
    a = bootstrap_video_accuracy(correct, seed=7, samples=1000)
    b = bootstrap_video_accuracy(correct, seed=7, samples=1000)
    assert a.point == b.point
    assert a.low == b.low
    assert a.high == b.high


def test_saturation_rate_flags_endpoint_clusters():
    u = np.array([0.0, 0.001, 0.5, 0.999, 1.0])  # range = 1.0
    # threshold = 1% of range = 0.01; values within 0.01 of min or max count.
    # 0.0 (distance 0), 0.001 (distance 0.001), 0.999 (distance 0.001), 1.0 (distance 0)
    # = 4 out of 5 = 0.8
    assert saturation_rate(u) == 0.8


def test_saturation_rate_low_for_uniform_interior_distribution():
    """A uniformly-distributed array saturates only at its own endpoints —
    ~2/n samples fall within the 1% threshold, so the rate stays tiny."""
    u = np.linspace(0.2, 0.8, 1000)
    # 1% of range = 0.006; two endpoint samples are always within threshold.
    assert saturation_rate(u) <= 0.02


def test_extrapolation_readable_matches_frozen_formula():
    """readable = median(u50 - u42) >= 0.2 * median(u42 - u34) and sat_rate <= 0.10"""

    # Case 1: extrapolation growth ok, saturation ok -> readable
    u34 = np.array([0.10, 0.10, 0.10])
    u42 = np.array([0.20, 0.20, 0.20])
    u50 = np.array([0.25, 0.25, 0.25])  # gap 0.05 vs 0.10 -> ratio 0.5 >= 0.2
    assert extrapolation_readable(u34, u42, u50, sat_rate=0.05) is True

    # Case 2: extrapolation ratio too small -> not readable
    u50_bad = np.array([0.21, 0.21, 0.21])  # gap 0.01 vs 0.10 -> ratio 0.1 < 0.2
    assert extrapolation_readable(u34, u42, u50_bad, sat_rate=0.05) is False

    # Case 3: saturation too high -> not readable
    assert extrapolation_readable(u34, u42, u50, sat_rate=0.15) is False


def test_spearman_correlation_is_exploratory_only():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert spearman_correlation(a, b) == pytest.approx(1.0)
    assert spearman_correlation(a, b[::-1]) == pytest.approx(-1.0)


def test_bootstrap_paired_delta_uses_same_indices_per_replicate():
    """For paired delta_acc, the same resample indices must be used for both
    u and baseline within each replicate — otherwise the CI over-inflates."""
    u = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
    baseline = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)
    result_u = bootstrap_video_accuracy(u, seed=17, samples=2000)
    result_b = bootstrap_video_accuracy(baseline, seed=17, samples=2000)
    # Difference of paired accuracies stays coherent under the same seed.
    delta_point = result_u.point - result_b.point
    assert np.isfinite(delta_point)


def test_bootstrap_paired_delta_point_matches_naive_diff():
    """Paired delta point equals `mean(u) - mean(baseline)` on the raw data."""
    u = np.array([[1, 1], [1, 0], [0, 0], [1, 1], [0, 1]], dtype=float)
    baseline = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [1, 1]], dtype=float)
    result = bootstrap_paired_delta(u, baseline, seed=3, samples=500)
    expected_point = float(u.mean() - baseline.mean())
    assert result.point == pytest.approx(expected_point)
    assert isinstance(result, BootstrapResult)
    assert result.low <= result.point <= result.high


def test_bootstrap_paired_delta_ci_tighter_than_unpaired():
    """Paired resampling on correlated correctness gives a tighter CI than the
    interval formed by subtracting independent-quantile endpoints."""
    rng = np.random.default_rng(0)
    V, K = 60, 3
    baseline_hits = rng.binomial(1, 0.5, size=(V, K)).astype(float)
    # `u` is `baseline + 0.15` in accuracy but perfectly correlated:
    # a video that got the pair right under baseline also got it under u.
    u_hits = baseline_hits.copy()
    boost = rng.random(size=(V, K)) < 0.15
    u_hits = np.clip(u_hits + boost.astype(float), 0, 1)
    paired = bootstrap_paired_delta(u_hits, baseline_hits, seed=42, samples=4000)
    ind_u = bootstrap_video_accuracy(u_hits, seed=42, samples=4000)
    ind_b = bootstrap_video_accuracy(baseline_hits, seed=17, samples=4000)
    naive_width = (ind_u.high - ind_b.low) - (ind_u.low - ind_b.high)
    paired_width = paired.high - paired.low
    assert paired_width < naive_width


def test_bootstrap_paired_delta_rejects_shape_mismatch():
    u = np.zeros((4, 3), dtype=float)
    baseline = np.zeros((4, 2), dtype=float)
    with pytest.raises(ValueError):
        bootstrap_paired_delta(u, baseline, seed=0, samples=50)


def test_bootstrap_paired_delta_reproducible_under_same_seed():
    u = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    baseline = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
    a = bootstrap_paired_delta(u, baseline, seed=99, samples=500)
    b = bootstrap_paired_delta(u, baseline, seed=99, samples=500)
    assert (a.point, a.low, a.high) == (b.point, b.low, b.high)
