"""Canonical contracts for the run-centric experiment artifact archive."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
import re
from typing import Mapping


ALLOWED_STAGES: frozenset[str] = frozenset(
    {
        "baseline",
        "feature_ablation",
        "loss_ablation",
        "topk_sweep",
        "multiseed",
        "calibration",
        "robustness",
        "efficiency",
        "final",
    }
)
ALLOWED_STATES: frozenset[str] = frozenset(
    {"reserved", "running", "validating", "completed", "failed", "cancelled"}
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{12,40}$")
_RUN_ID = re.compile(
    r"^(?P<dataset>[A-Za-z0-9][A-Za-z0-9._-]*)-"
    r"(?P<stage>baseline|feature_ablation|loss_ablation|topk_sweep|multiseed|calibration|robustness|efficiency|final)-"
    r"(?P<variant>[A-Za-z0-9][A-Za-z0-9._-]*)-"
    r"seed(?P<seed>\d{3})-(?P<timestamp>\d{8}T\d{6}Z)-(?P<git_sha>[0-9a-f]{12})$"
)

_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "study",
        "dataset",
        "stage",
        "variant",
        "seed",
        "artifact_path",
        "registry_path",
        "git_sha",
        "tracked_clean",
        "source_patch_sha256",
        "environment",
        "input_checkpoints",
        "dataset_splits",
        "config",
        "parent_run_ids",
        "selection_lineage",
        "validation_authorized",
        "test_authorized",
        "expected_scalar_outputs",
        "array_references",
    }
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    status: Path
    command: Path
    logs: Path
    checkpoints: Path
    metrics: Path
    arrays: Path
    plots: Path
    provenance: Path


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    study: str
    dataset: str
    stage: str
    variant: str
    seed: int
    artifact_path: str
    registry_path: str
    git_sha: str
    tracked_clean: bool
    source_patch_sha256: str | None
    environment: dict[str, object]
    input_checkpoints: list[dict[str, object]]
    dataset_splits: list[dict[str, object]]
    config: dict[str, object]
    parent_run_ids: list[str]
    selection_lineage: list[object]
    validation_authorized: bool
    test_authorized: bool
    expected_scalar_outputs: list[object]
    array_references: list[dict[str, object]]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RunManifest":
        return cls(**validate_manifest(payload))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_safe_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or "--" in value:
        raise ValueError(f"{field} must be a safe archive name")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _validate_utc_timestamp(timestamp_utc: object) -> str:
    if not isinstance(timestamp_utc, str):
        raise ValueError("timestamp_utc must be a UTC timestamp")
    try:
        parsed = datetime.strptime(timestamp_utc, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("timestamp_utc must use YYYYMMDDTHHMMSSZ UTC format") from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != timestamp_utc:
        raise ValueError("timestamp_utc must use canonical UTC format")
    return timestamp_utc


def build_run_id(
    dataset: str,
    stage: str,
    variant: str,
    seed: int,
    timestamp_utc: str,
    git_sha: str,
) -> str:
    """Build the canonical deterministic identifier for one experiment run."""

    dataset = _require_safe_name(dataset, "dataset")
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {sorted(ALLOWED_STAGES)}")
    variant = _require_safe_name(variant, "variant")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 999:
        raise ValueError("seed must be an integer from 0 through 999")
    timestamp_utc = _validate_utc_timestamp(timestamp_utc)
    if not isinstance(git_sha, str) or not _GIT_SHA.fullmatch(git_sha):
        raise ValueError("git_sha must contain 12 to 40 lowercase hexadecimal characters")
    return f"{dataset}-{stage}-{variant}-seed{seed:03d}-{timestamp_utc}-{git_sha[:12]}"


def make_run_paths(artifact_root: PathLike[str] | str, study: str, run_id: str) -> RunPaths:
    """Return the canonical, not-yet-created paths for a run directory."""

    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("artifact_root must be absolute")
    study = _require_safe_name(study, "study")
    match = _RUN_ID.fullmatch(run_id) if isinstance(run_id, str) else None
    if match is None or "--" in run_id:
        raise ValueError("run_id must be a canonical archive run ID")
    run_root = root / study / match.group("dataset") / match.group("stage") / run_id
    return RunPaths(
        root=run_root,
        manifest=run_root / "manifest.json",
        status=run_root / "status.json",
        command=run_root / "command.sh",
        logs=run_root / "logs",
        checkpoints=run_root / "checkpoints",
        metrics=run_root / "metrics",
        arrays=run_root / "arrays",
        plots=run_root / "plots",
        provenance=run_root / "provenance",
    )


def _validate_hashed_references(value: object, field: str) -> None:
    for index, item in enumerate(_require_list(value, field)):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field}[{index}] must be an object")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"{field}[{index}].path must be a non-empty string")
        _require_sha256(item.get("sha256"), f"{field}[{index}].sha256")


def validate_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy a complete, serializable archive manifest payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be an object")
    unknown = set(payload) - _MANIFEST_FIELDS
    missing = _MANIFEST_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"manifest has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"manifest is missing required fields: {sorted(missing)}")

    normalized = dict(payload)
    for field in ("study", "dataset", "variant"):
        _require_safe_name(normalized[field], field)
    stage = normalized["stage"]
    if not isinstance(stage, str) or stage not in ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {sorted(ALLOWED_STAGES)}")
    seed = normalized["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 999:
        raise ValueError("seed must be an integer from 0 through 999")
    run_id = normalized["run_id"]
    match = _RUN_ID.fullmatch(run_id) if isinstance(run_id, str) else None
    if match is None or "--" in run_id:
        raise ValueError("run_id must be a canonical archive run ID")
    expected_prefix = f"{normalized['dataset']}-{stage}-{normalized['variant']}-seed{seed:03d}-"
    if not run_id.startswith(expected_prefix):
        raise ValueError("run_id must agree with dataset, stage, variant, and seed")
    _validate_utc_timestamp(match.group("timestamp"))
    artifact_path = normalized["artifact_path"]
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_absolute():
        raise ValueError("artifact_path must be absolute")
    registry_path = normalized["registry_path"]
    if not isinstance(registry_path, str) or Path(registry_path).is_absolute() or ".." in Path(registry_path).parts:
        raise ValueError("registry_path must be repository-relative and traversal-free")
    git_sha = normalized["git_sha"]
    if not isinstance(git_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise ValueError("git_sha must be a 40-character lowercase commit SHA")
    if match.group("git_sha") != git_sha[:12]:
        raise ValueError("run_id git SHA-12 must match git_sha")
    if not isinstance(normalized["tracked_clean"], bool):
        raise ValueError("tracked_clean must be a boolean")
    source_patch_sha256 = normalized["source_patch_sha256"]
    if source_patch_sha256 is not None:
        _require_sha256(source_patch_sha256, "source_patch_sha256")
    if not isinstance(normalized["environment"], Mapping):
        raise ValueError("environment must be an object")
    for field in ("python", "pytorch", "cuda", "gpu_model", "precision", "distributed_topology"):
        if not isinstance(normalized["environment"].get(field), str):
            raise ValueError(f"environment.{field} must be a string")
    if isinstance(normalized["environment"].get("gpu_count"), bool) or not isinstance(normalized["environment"].get("gpu_count"), int):
        raise ValueError("environment.gpu_count must be an integer")
    _validate_hashed_references(normalized["input_checkpoints"], "input_checkpoints")
    _validate_hashed_references(normalized["dataset_splits"], "dataset_splits")
    config = normalized["config"]
    if not isinstance(config, Mapping) or set(config) != {"model", "loss", "optimizer", "scheduler", "evaluation"}:
        raise ValueError("config must contain model, loss, optimizer, scheduler, and evaluation")
    parent_run_ids = _require_list(normalized["parent_run_ids"], "parent_run_ids")
    if any(not isinstance(parent, str) or not parent for parent in parent_run_ids):
        raise ValueError("parent_run_ids must contain non-empty run IDs")
    if stage != "baseline" and not parent_run_ids:
        raise ValueError("derived runs require parent lineage")
    _require_list(normalized["selection_lineage"], "selection_lineage")
    for field in ("validation_authorized", "test_authorized"):
        if not isinstance(normalized[field], bool):
            raise ValueError(f"{field} must be a boolean")
    if normalized["test_authorized"] and stage != "final":
        raise ValueError("test_authorized is permitted only for final stage runs")
    _require_list(normalized["expected_scalar_outputs"], "expected_scalar_outputs")
    _validate_hashed_references(normalized["array_references"], "array_references")
    return normalized
