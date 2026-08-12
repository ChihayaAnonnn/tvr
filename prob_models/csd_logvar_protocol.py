"""Paired-invariance and checkpoint-selection helpers for A2 experiments."""

from __future__ import annotations

from prob_models import uncertainty_protocol as _generic


A2_PREFIX = "csd_logvar_sidecar."
_A2_PAIRED_DIGEST_KEYS = (
    "raw_wti_sha256",
    "a1_failure_risk_sha256",
    "a1_vacuity_sha256",
    "retrieval",
)


def array_sha256(array) -> str:
    """Hash array dtype, shape, and contiguous value bytes."""

    return _generic.array_sha256(array)


def file_sha256(path) -> str:
    """Hash a checkpoint file without loading it into memory."""

    return _generic.file_sha256(path)


def validate_csd_logvar_candidate_artifacts(candidates, *, output_dir):
    """Verify every persisted model/optimizer pair without changing evidence."""

    return _generic.validate_candidate_artifacts(
        candidates, output_dir=output_dir, method_name="A2"
    )


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

    return _generic._validate_resume_lineage(
        candidates,
        output_dir=output_dir,
        model_checkpoint=model_checkpoint,
        optimizer_checkpoint=optimizer_checkpoint,
        optimizer_payload=optimizer_payload,
        sidecar_prefix=A2_PREFIX,
        method_name="A2",
        artifacts_verified=artifacts_verified,
    )


def validate_csd_logvar_selected_checkpoint(selection_payload, *, output_dir):
    """Resolve and hash the recorded winner immediately before final test."""

    return _generic.validate_selected_checkpoint(
        selection_payload, output_dir=output_dir, method_name="A2"
    )


def non_csd_state_dict_mismatches(
    base_checkpoint,
    candidate_checkpoint,
) -> list[str]:
    """Return non-A2 keys that differ from the registered A1 checkpoint."""

    return _generic.non_sidecar_state_dict_mismatches(
        base_checkpoint, candidate_checkpoint, (A2_PREFIX,)
    )


def compare_paired_invariance(baseline, candidate):
    """Compare all registered A1 outputs by exact equality."""

    for key in _A2_PAIRED_DIGEST_KEYS:
        baseline[key]
        candidate[key]
    return _generic.compare_paired_invariance(
        baseline, candidate, _A2_PAIRED_DIGEST_KEYS
    )


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
    return _generic.monotone_selective_accuracy(direction_metrics)


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
            item["clamp_low_fraction"] + item["clamp_high_fraction"] < 0.05
            for item in distributions
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def numeric_metrics_are_finite(value) -> bool:
    """Recursively reject any non-finite numeric metric value."""

    return _generic.numeric_metrics_are_finite(value)
