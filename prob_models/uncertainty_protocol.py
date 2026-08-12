"""Method-neutral uncertainty experiment protocol helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CANDIDATE_ARTIFACT_FIELDS = {
    "epoch",
    "path",
    "model_sha256",
    "optimizer_path",
    "optimizer_sha256",
}

A3_REQUIRED_PAIRED_DIGEST_KEYS = (
    "raw_wti_sha256",
    "a1_failure_risk_sha256",
    "a1_vacuity_sha256",
    "retrieval",
)


def array_sha256(array) -> str:
    """Hash array dtype, shape, and contiguous value bytes."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path: PathLike) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_metrics_are_finite(value) -> bool:
    """Recursively reject any non-finite numeric metric value."""

    if isinstance(value, Mapping):
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


def validate_candidate_artifacts(
    candidates: Sequence[dict],
    *,
    output_dir: PathLike,
    method_name: str,
) -> list[dict]:
    """Verify all persisted model/optimizer pairs in one candidate lineage."""

    output_path = Path(output_dir).resolve()
    seen_epochs: set[int] = set()
    seen_artifact_paths: set[Path] = set()
    for candidate in candidates:
        missing = sorted(_CANDIDATE_ARTIFACT_FIELDS.difference(candidate))
        if missing:
            raise ValueError(
                f"recorded {method_name} candidate missing lineage fields: "
                + ", ".join(missing)
            )
        epoch = int(candidate["epoch"])
        model_path = Path(candidate["path"]).resolve()
        optimizer_path = Path(candidate["optimizer_path"]).resolve()
        if epoch in seen_epochs:
            raise ValueError(
                f"recorded {method_name} candidate epochs must be unique"
            )
        if (
            model_path in seen_artifact_paths
            or optimizer_path in seen_artifact_paths
            or model_path == optimizer_path
        ):
            raise ValueError(
                f"recorded {method_name} candidate artifact paths must be unique"
            )
        seen_epochs.add(epoch)
        seen_artifact_paths.update((model_path, optimizer_path))
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
                f"persisted {method_name} candidate model checkpoint is unavailable"
            ) from exc
        if current_model_sha256 != candidate["model_sha256"]:
            raise ValueError(
                f"persisted {method_name} candidate model checkpoint SHA-256 mismatch"
            )
        try:
            current_optimizer_sha256 = file_sha256(optimizer_path)
        except OSError as exc:
            raise ValueError(
                f"persisted {method_name} candidate optimizer checkpoint is unavailable"
            ) from exc
        if current_optimizer_sha256 != candidate["optimizer_sha256"]:
            raise ValueError(
                f"persisted {method_name} candidate optimizer checkpoint SHA-256 mismatch"
            )
    return list(candidates)


def _mismatch_field(method_name: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in method_name
    ).strip("_")
    return f"non_{normalized}_mismatches"


def _validate_resume_lineage(
    candidates: Sequence[dict],
    *,
    output_dir: PathLike,
    model_checkpoint: PathLike,
    optimizer_checkpoint: PathLike,
    optimizer_payload: Mapping,
    sidecar_prefix: str,
    method_name: str,
    artifacts_verified: bool,
) -> dict:
    output_path = Path(output_dir).resolve()
    requested_model = Path(model_checkpoint).resolve()
    requested_optimizer = Path(optimizer_checkpoint).resolve()
    if not artifacts_verified:
        validate_candidate_artifacts(
            candidates, output_dir=output_path, method_name=method_name
        )
    matches = [
        candidate
        for candidate in candidates
        if Path(candidate["path"]).resolve() == requested_model
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{method_name} continuation model must match exactly one recorded candidate"
        )
    candidate = matches[0]
    highest_epoch = max(int(item["epoch"]) for item in candidates)
    if int(candidate["epoch"]) != highest_epoch:
        raise ValueError(
            f"{method_name} continuation must resume the highest-epoch recorded candidate"
        )
    recorded_model = Path(candidate["path"]).resolve()
    if recorded_model.parent != output_path:
        raise ValueError("recorded model checkpoint is outside the output directory")
    if (
        not artifacts_verified
        and file_sha256(recorded_model) != candidate["model_sha256"]
    ):
        raise ValueError(
            f"{method_name} continuation model checkpoint SHA-256 mismatch"
        )
    recorded_optimizer = Path(candidate["optimizer_path"]).resolve()
    if recorded_optimizer.parent != output_path:
        raise ValueError(
            "recorded optimizer checkpoint is outside the output directory"
        )
    if requested_optimizer != recorded_optimizer:
        raise ValueError(
            f"{method_name} continuation optimizer checkpoint path mismatch"
        )
    if (
        not artifacts_verified
        and file_sha256(recorded_optimizer) != candidate["optimizer_sha256"]
    ):
        raise ValueError(
            f"{method_name} continuation optimizer checkpoint SHA-256 mismatch"
        )
    if "epoch" not in optimizer_payload:
        raise ValueError(
            f"{method_name} continuation optimizer checkpoint epoch is missing"
        )
    if int(optimizer_payload["epoch"]) != int(candidate["epoch"]):
        raise ValueError(
            f"{method_name} continuation optimizer checkpoint epoch mismatch"
        )
    mismatch_field = _mismatch_field(method_name)
    mismatches = candidate.get(
        mismatch_field, candidate.get("non_sidecar_mismatches", [])
    )
    if mismatches:
        raise ValueError(
            f"{method_name} continuation candidate has non-{method_name} state mismatches"
        )
    if not sidecar_prefix:
        raise ValueError("sidecar_prefix must be non-empty")
    return candidate


def validate_resume_lineage(
    candidates: Sequence[dict],
    *,
    output_dir: PathLike,
    model_checkpoint: PathLike,
    optimizer_checkpoint: PathLike,
    optimizer_payload: Mapping,
    sidecar_prefix: str,
    method_name: str,
) -> dict:
    """Require an exact highest-epoch model/optimizer resume pair."""

    return _validate_resume_lineage(
        candidates,
        output_dir=output_dir,
        model_checkpoint=model_checkpoint,
        optimizer_checkpoint=optimizer_checkpoint,
        optimizer_payload=optimizer_payload,
        sidecar_prefix=sidecar_prefix,
        method_name=method_name,
        artifacts_verified=False,
    )


def validate_selected_checkpoint(
    selection_payload: Mapping,
    *,
    output_dir: PathLike,
    method_name: str,
) -> dict:
    """Resolve and hash a recorded winner immediately before final test."""

    winner = selection_payload.get("winner")
    candidates = selection_payload.get("candidates")
    if not isinstance(winner, dict) or not isinstance(candidates, list):
        raise ValueError(
            f"{method_name} selection payload requires a winner and candidates"
        )
    matches = [
        candidate
        for candidate in candidates
        if int(candidate.get("epoch", -1)) == int(winner.get("epoch", -2))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected {method_name} winner must match one recorded candidate"
        )
    recorded = matches[0]
    winner_path = Path(winner.get("path", "")).resolve()
    recorded_path = Path(recorded.get("path", "")).resolve()
    if winner_path != recorded_path:
        raise ValueError(
            f"selected {method_name} model path differs from recorded candidate"
        )
    if winner.get("model_sha256") != recorded.get("model_sha256"):
        raise ValueError(
            f"selected {method_name} model hash differs from recorded candidate"
        )
    if recorded_path.parent != Path(output_dir).resolve():
        raise ValueError(
            f"selected {method_name} model is outside the output directory"
        )
    try:
        current_sha256 = file_sha256(recorded_path)
    except OSError as exc:
        raise ValueError(
            f"selected {method_name} model checkpoint is unavailable"
        ) from exc
    if current_sha256 != recorded["model_sha256"]:
        raise ValueError(f"selected {method_name} model SHA-256 mismatch")
    return recorded


def non_sidecar_state_dict_mismatches(
    base_checkpoint: PathLike,
    candidate_checkpoint: PathLike,
    allowed_prefixes: tuple[str, ...],
) -> list[str]:
    """Return state keys differing outside explicitly allowed sidecar prefixes."""

    base = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=True)
    candidate = torch.load(
        Path(candidate_checkpoint), map_location="cpu", weights_only=True
    )

    def is_allowed(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in allowed_prefixes)

    expected = {name for name in base if not is_allowed(name)}
    actual = {name for name in candidate if not is_allowed(name)}
    mismatches = sorted(expected.symmetric_difference(actual))
    for name in sorted(expected.intersection(actual)):
        left = base[name]
        right = candidate[name]
        if left.shape != right.shape or not torch.equal(left.to(right.dtype), right):
            mismatches.append(name)
    return sorted(set(mismatches))


def _digest_check_name(key: str) -> str:
    return key.removesuffix("_sha256")


def compare_paired_invariance(
    baseline: Mapping,
    candidate: Mapping,
    required_digest_keys: tuple[str, ...],
) -> dict[str, object]:
    """Compare each registered paired digest/evidence entry exactly."""

    checks = {
        _digest_check_name(key): (
            key in baseline and key in candidate and baseline[key] == candidate[key]
        )
        for key in required_digest_keys
    }
    return {"passed": all(checks.values()), "checks": checks}


def append_candidate_atomic(path: PathLike, candidate: Mapping) -> list[dict]:
    """Append one unique, increasing-epoch candidate using atomic replacement."""

    target = Path(path)
    if target.exists():
        try:
            candidates = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing candidate list is unreadable") from exc
        if not isinstance(candidates, list):
            raise ValueError("existing protocol artifact is not a candidate list")
    else:
        candidates = []
    required = {"epoch", "path", "optimizer_path"}
    missing = sorted(required.difference(candidate))
    if missing:
        raise ValueError("candidate missing append-only fields: " + ", ".join(missing))
    epoch = int(candidate["epoch"])
    model_path = Path(candidate["path"]).resolve()
    optimizer_path = Path(candidate["optimizer_path"]).resolve()
    epochs = [int(item["epoch"]) for item in candidates]
    artifact_paths = [
        Path(item[field]).resolve()
        for item in candidates
        for field in ("path", "optimizer_path")
    ]
    if len(epochs) != len(set(epochs)):
        raise ValueError("recorded candidate epochs must be unique")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("recorded candidate artifact paths must be unique")
    if epochs and epoch <= max(epochs):
        raise ValueError(
            "candidate append-only epoch must exceed every recorded epoch"
        )
    if epoch in epochs:
        raise ValueError("candidate epochs must be unique")
    if (
        model_path == optimizer_path
        or model_path in artifact_paths
        or optimizer_path in artifact_paths
    ):
        raise ValueError("candidate artifact paths must be unique")
    appended = [*candidates, _json_safe(dict(candidate))]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(appended, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return appended


def _json_safe(value):
    """Normalize supported metric/evidence values for deterministic JSON."""

    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def monotone_selective_accuracy(direction_metrics: Mapping) -> bool:
    curve = direction_metrics["risk_coverage"]
    values = [
        curve[f"accuracy_at_{point:.2f}"]
        for point in (0.25, 0.50, 0.75, 1.00)
    ]
    return all(left >= right for left, right in zip(values, values[1:]))


def contextual_risk_selection_tuple(candidate: Mapping) -> tuple[float, ...]:
    """Return the registered validation-only A3 selection tuple."""

    return (
        min(candidate["rho"]),
        sum(candidate["auroc"]) / 2.0,
        -sum(candidate["aurc"]) / 2.0,
        -int(candidate["epoch"]),
    )


def _complete_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_complete_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_complete_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return value is not None and value != ""


def _directional_structures_are_complete(value: Any) -> bool:
    if isinstance(value, Mapping):
        directions = {"t2v", "v2t"}
        present = directions.intersection(value)
        if present and (
            present != directions
            or any(not _complete_value(value[key]) for key in directions)
        ):
            return False
        return all(
            _directional_structures_are_complete(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_directional_structures_are_complete(item) for item in value)
    return True


def _valid_a3_paired_digest(paired_digest: Any) -> bool:
    if not isinstance(paired_digest, Mapping):
        return False
    raw_wti = paired_digest.get("raw_wti_sha256")
    if not isinstance(raw_wti, str) or not raw_wti:
        return False
    for key in ("a1_failure_risk_sha256", "a1_vacuity_sha256"):
        directional = paired_digest.get(key)
        if not isinstance(directional, Mapping) or any(
            not isinstance(directional.get(direction), str)
            or not directional[direction]
            for direction in ("t2v", "v2t")
        ):
            return False
    retrieval = paired_digest.get("retrieval")
    if not isinstance(retrieval, Mapping) or any(
        not isinstance(retrieval.get(direction), Mapping)
        or not retrieval[direction]
        or not _complete_value(retrieval[direction])
        for direction in ("t2v", "v2t")
    ):
        return False
    return bool(
        _directional_structures_are_complete(paired_digest)
        and numeric_metrics_are_finite(retrieval)
    )


def contextual_risk_candidate_is_valid(candidate: Mapping) -> bool:
    """Validate the persisted non-artifact evidence needed for A3 selection."""

    paired_digest = candidate.get("paired_digest")
    paired_invariance = candidate.get("paired_invariance")
    evidence = candidate.get(
        "evidence", candidate.get("contextual_risk_evidence")
    )
    required_checks = {
        _digest_check_name(key) for key in A3_REQUIRED_PAIRED_DIGEST_KEYS
    }
    checks = (
        paired_invariance.get("checks", {})
        if isinstance(paired_invariance, Mapping)
        else {}
    )
    complete_hashes = (
        _valid_a3_paired_digest(paired_digest)
        and required_checks.issubset(checks)
    )
    metric_pairs = tuple(candidate.get(name) for name in ("rho", "auroc", "aurc"))
    complete_metric_pairs = all(
        isinstance(values, (list, tuple))
        and len(values) == 2
        and numeric_metrics_are_finite(values)
        for values in metric_pairs
    )
    return bool(
        candidate.get("valid", False)
        and candidate.get("finite", False)
        and complete_metric_pairs
        and _complete_value(evidence)
        and numeric_metrics_are_finite(evidence)
        and complete_hashes
        and paired_invariance.get("passed", False)
        and all(bool(checks[key]) for key in required_checks)
        and not candidate.get("non_a3_mismatches")
        and not candidate.get("runtime_abort", False)
        and not candidate.get("runtime_abort_condition", False)
    )


def choose_contextual_risk_checkpoint(
    candidates: Sequence[dict], *, output_dir: PathLike
) -> dict:
    """Choose the valid A3 winner without consulting test data."""

    validate_candidate_artifacts(candidates, output_dir=output_dir, method_name="A3")
    valid = [
        candidate
        for candidate in candidates
        if contextual_risk_candidate_is_valid(candidate)
    ]
    if not valid:
        raise ValueError("no valid contextual risk checkpoint")
    return max(valid, key=contextual_risk_selection_tuple)


def contextual_risk_acceptance(metrics: Mapping) -> dict[str, object]:
    """Evaluate the registered bidirectional A3 validation gate."""

    directions = (metrics["t2v"], metrics["v2t"])
    checks = {
        "rho": all(
            item["spearman_risk_vs_rank"] >= 0.20 for item in directions
        ),
        "auroc_ci": all(
            item["failure_auroc"]["ci_low"] > 0.50 for item in directions
        ),
        "selective": all(
            monotone_selective_accuracy(item) for item in directions
        ),
        "spread": all(item["risk_stats"]["std"] >= 0.02 for item in directions),
        "invariance": bool(metrics["paired_invariance"]["passed"]),
        "state": not metrics["non_a3_mismatches"],
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "allow_final_test": passed}
