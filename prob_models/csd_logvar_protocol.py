"""Paired-invariance and checkpoint-selection helpers for A2 experiments."""

from __future__ import annotations

import hashlib
import math
from numbers import Number
from pathlib import Path

import numpy as np
import torch

A2_PREFIX = "csd_logvar_sidecar."


def array_sha256(array) -> str:
    """Hash array dtype, shape, and contiguous value bytes."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def non_csd_state_dict_mismatches(
    base_checkpoint,
    candidate_checkpoint,
) -> list[str]:
    """Return non-A2 keys that differ from the registered A1 checkpoint."""

    base = torch.load(
        Path(base_checkpoint), map_location="cpu", weights_only=True
    )
    candidate = torch.load(
        Path(candidate_checkpoint), map_location="cpu", weights_only=True
    )
    expected = {name for name in base if not name.startswith(A2_PREFIX)}
    actual = {name for name in candidate if not name.startswith(A2_PREFIX)}
    mismatches = sorted(expected.symmetric_difference(actual))
    for name in sorted(expected.intersection(actual)):
        left = base[name]
        right = candidate[name]
        if (
            left.shape != right.shape
            or not torch.equal(left.to(right.dtype), right)
        ):
            mismatches.append(name)
    return sorted(set(mismatches))


def compare_paired_invariance(baseline, candidate):
    """Compare all registered A1 outputs by exact equality."""

    checks = {
        "raw_wti": (
            baseline["raw_wti_sha256"] == candidate["raw_wti_sha256"]
        ),
        "a1_failure_risk": (
            baseline["a1_failure_risk_sha256"]
            == candidate["a1_failure_risk_sha256"]
        ),
        "a1_vacuity": (
            baseline["a1_vacuity_sha256"]
            == candidate["a1_vacuity_sha256"]
        ),
        "retrieval": baseline["retrieval"] == candidate["retrieval"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_csd_logvar_checkpoint(candidates):
    """Choose the valid bidirectional winner without consulting test data."""

    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError("no valid CSD logvar checkpoint")
    return max(
        valid,
        key=lambda item: (
            min(item["rho"]),
            sum(item["auroc"]) / 2.0,
            -int(item["epoch"]),
        ),
    )


def _monotone_selective_accuracy(direction_metrics):
    curve = direction_metrics["risk_coverage"]
    values = [
        curve[f"accuracy_at_{point:.2f}"]
        for point in (0.25, 0.50, 0.75, 1.00)
    ]
    return all(left >= right for left, right in zip(values, values[1:]))


def csd_logvar_acceptance(metrics):
    """Evaluate the registered bidirectional A2 acceptance thresholds."""

    t2v = metrics["t2v"]
    v2t = metrics["v2t"]
    distributions = (
        metrics["text_distribution"],
        metrics["video_distribution"],
    )
    checks = {
        "rho": min(
            t2v["spearman_uncertainty_vs_rank"],
            v2t["spearman_uncertainty_vs_rank"],
        )
        >= 0.20,
        "auroc": min(
            t2v["failure_auroc"]["value"],
            v2t["failure_auroc"]["value"],
        )
        > 0.5,
        "selective_accuracy": (
            _monotone_selective_accuracy(t2v)
            and _monotone_selective_accuracy(v2t)
        ),
        "nonzero_variance": all(
            item["variance"]["std"] > 0 for item in distributions
        ),
        "clamp_rate": all(
            item["clamp_low_fraction"] + item["clamp_high_fraction"]
            < 0.05
            for item in distributions
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def numeric_metrics_are_finite(value) -> bool:
    """Recursively reject any non-finite numeric metric value."""

    if isinstance(value, dict):
        return all(numeric_metrics_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_metrics_are_finite(item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.isfinite(value).all())
        return all(numeric_metrics_are_finite(item) for item in value.flat)
    if isinstance(value, Number) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True
