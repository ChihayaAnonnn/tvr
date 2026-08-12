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


def file_sha256(path) -> str:
    """Hash a checkpoint file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CANDIDATE_ARTIFACT_FIELDS = {
    "epoch",
    "path",
    "model_sha256",
    "optimizer_path",
    "optimizer_sha256",
}


def validate_csd_logvar_candidate_artifacts(candidates, *, output_dir):
    """Verify every persisted model/optimizer pair without changing evidence."""

    output_path = Path(output_dir).resolve()
    seen_epochs = set()
    seen_models = set()
    seen_optimizers = set()
    for candidate in candidates:
        missing = sorted(_CANDIDATE_ARTIFACT_FIELDS.difference(candidate))
        if missing:
            raise ValueError(
                "recorded A2 candidate missing lineage fields: "
                + ", ".join(missing)
            )
        epoch = int(candidate["epoch"])
        model_path = Path(candidate["path"]).resolve()
        optimizer_path = Path(candidate["optimizer_path"]).resolve()
        if epoch in seen_epochs:
            raise ValueError("recorded A2 candidate epochs must be unique")
        if model_path in seen_models or optimizer_path in seen_optimizers:
            raise ValueError("recorded A2 candidate artifact paths must be unique")
        seen_epochs.add(epoch)
        seen_models.add(model_path)
        seen_optimizers.add(optimizer_path)
        if model_path.parent != output_path:
            raise ValueError(
                "recorded model checkpoint is outside the output directory"
            )
        if optimizer_path.parent != output_path:
            raise ValueError(
                "recorded optimizer checkpoint is outside the output directory"
            )
        try:
            current_model_sha256 = file_sha256(model_path)
        except OSError as exc:
            raise ValueError(
                "persisted A2 candidate model checkpoint is unavailable"
            ) from exc
        if current_model_sha256 != candidate["model_sha256"]:
            raise ValueError(
                "persisted A2 candidate model checkpoint SHA-256 mismatch"
            )
        try:
            current_optimizer_sha256 = file_sha256(optimizer_path)
        except OSError as exc:
            raise ValueError(
                "persisted A2 candidate optimizer checkpoint is unavailable"
            ) from exc
        if current_optimizer_sha256 != candidate["optimizer_sha256"]:
            raise ValueError(
                "persisted A2 candidate optimizer checkpoint SHA-256 mismatch"
            )
    return list(candidates)


def validate_csd_logvar_resume_lineage(
    candidates,
    *,
    output_dir,
    model_checkpoint,
    optimizer_checkpoint,
    optimizer_payload,
    artifacts_verified=False,
):
    """Require an exact recorded same-run model/optimizer resume pair."""

    output_path = Path(output_dir).resolve()
    requested_model = Path(model_checkpoint).resolve()
    requested_optimizer = Path(optimizer_checkpoint).resolve()
    if not artifacts_verified:
        validate_csd_logvar_candidate_artifacts(
            candidates,
            output_dir=output_path,
        )
    matches = [
        candidate
        for candidate in candidates
        if Path(candidate["path"]).resolve() == requested_model
    ]
    if len(matches) != 1:
        raise ValueError(
            "A2 continuation model must match exactly one recorded candidate"
        )
    candidate = matches[0]
    highest_epoch = max(int(item["epoch"]) for item in candidates)
    if int(candidate["epoch"]) != highest_epoch:
        raise ValueError(
            "A2 continuation must resume the highest-epoch recorded candidate"
        )
    recorded_model = Path(candidate["path"]).resolve()
    if recorded_model.parent != output_path:
        raise ValueError("recorded model checkpoint is outside the output directory")
    if (
        not artifacts_verified
        and file_sha256(recorded_model) != candidate["model_sha256"]
    ):
        raise ValueError("A2 continuation model checkpoint SHA-256 mismatch")

    recorded_optimizer = Path(candidate["optimizer_path"]).resolve()
    if recorded_optimizer.parent != output_path:
        raise ValueError(
            "recorded optimizer checkpoint is outside the output directory"
        )
    if requested_optimizer != recorded_optimizer:
        raise ValueError("A2 continuation optimizer checkpoint path mismatch")
    if (
        not artifacts_verified
        and file_sha256(recorded_optimizer) != candidate["optimizer_sha256"]
    ):
        raise ValueError("A2 continuation optimizer checkpoint SHA-256 mismatch")
    if "epoch" not in optimizer_payload:
        raise ValueError("A2 continuation optimizer checkpoint epoch is missing")
    if int(optimizer_payload["epoch"]) != int(candidate["epoch"]):
        raise ValueError("A2 continuation optimizer checkpoint epoch mismatch")
    if candidate.get("non_a2_mismatches"):
        raise ValueError("A2 continuation candidate has non-A2 state mismatches")
    return candidate


def validate_csd_logvar_selected_checkpoint(selection_payload, *, output_dir):
    """Resolve and hash the recorded winner immediately before final test."""

    winner = selection_payload.get("winner")
    candidates = selection_payload.get("candidates")
    if not isinstance(winner, dict) or not isinstance(candidates, list):
        raise ValueError("A2 selection payload requires a winner and candidates")
    matches = [
        candidate
        for candidate in candidates
        if int(candidate.get("epoch", -1)) == int(winner.get("epoch", -2))
    ]
    if len(matches) != 1:
        raise ValueError("selected A2 winner must match one recorded candidate")
    recorded = matches[0]
    winner_path = Path(winner.get("path", "")).resolve()
    recorded_path = Path(recorded.get("path", "")).resolve()
    if winner_path != recorded_path:
        raise ValueError("selected A2 model path differs from recorded candidate")
    if winner.get("model_sha256") != recorded.get("model_sha256"):
        raise ValueError("selected A2 model hash differs from recorded candidate")
    if recorded_path.parent != Path(output_dir).resolve():
        raise ValueError("selected A2 model is outside the output directory")
    try:
        current_sha256 = file_sha256(recorded_path)
    except OSError as exc:
        raise ValueError("selected A2 model checkpoint is unavailable") from exc
    if current_sha256 != recorded["model_sha256"]:
        raise ValueError("selected A2 model SHA-256 mismatch")
    return recorded


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


def choose_csd_logvar_checkpoint(candidates, *, output_dir=None):
    """Choose the valid bidirectional winner without consulting test data."""

    if output_dir is not None:
        validate_csd_logvar_candidate_artifacts(
            candidates,
            output_dir=output_dir,
        )
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
