"""Canonical contracts for the run-centric experiment artifact archive."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from os import PathLike
from pathlib import Path
import re
import socket
import stat
import subprocess
import tempfile
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
_REGISTRY_FIELDS = frozenset(
    {
        "run_id",
        "event",
        "artifact_path",
        "manifest_sha256",
        "state",
        "parent_run_ids",
        "selected_validation",
        "test_authorized",
        "summary_metrics",
        "supersedes_sha256",
    }
)
_REGISTRY_REQUIRED_FIELDS = _REGISTRY_FIELDS - {"supersedes_sha256"}
_REGISTRY_EVENTS = frozenset({"reserved", "completed", "failed", "cancelled", "correction"})
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_ALLOWED_TRANSITIONS = {
    "reserved": frozenset({"running", "cancelled"}),
    "running": frozenset({"validating", "cancelled"}),
    "validating": frozenset({"completed", "failed"}),
}


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


_RUN_PATH_RELATIVE_NAMES = {
    "manifest": Path("manifest.json"),
    "status": Path("status.json"),
    "command": Path("command.sh"),
    "logs": Path("logs"),
    "checkpoints": Path("checkpoints"),
    "metrics": Path("metrics"),
    "arrays": Path("arrays"),
    "plots": Path("plots"),
    "provenance": Path("provenance"),
}
_RUN_DIRECTORY_FIELDS = frozenset({"logs", "checkpoints", "metrics", "arrays", "plots", "provenance"})
_RUN_FILE_FIELDS = frozenset({"manifest", "status", "command"})


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


def _unsafe_run_path(message: str) -> ValueError:
    return ValueError(f"unsafe run path: {message}")


def _assert_no_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise _unsafe_run_path("path must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise _unsafe_run_path(f"symlink component: {current}")


def _validate_run_paths(paths: RunPaths, *, require_root: bool = False) -> None:
    """Fail closed unless every existing run member is a private canonical child."""

    if not paths.root.is_absolute():
        raise _unsafe_run_path("run root must be absolute")
    for field, relative_path in _RUN_PATH_RELATIVE_NAMES.items():
        if getattr(paths, field) != paths.root / relative_path:
            raise _unsafe_run_path(f"non-canonical {field} member")
    _assert_no_symlink_components(paths.root)
    try:
        root_info = paths.root.lstat()
    except FileNotFoundError:
        if require_root:
            raise _unsafe_run_path("missing run root")
        return
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise _unsafe_run_path("run root is not a real directory")
    for field in _RUN_DIRECTORY_FIELDS | _RUN_FILE_FIELDS:
        path = getattr(paths, field)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise _unsafe_run_path(f"symlinked {field}")
        if field in _RUN_DIRECTORY_FIELDS and not stat.S_ISDIR(info.st_mode):
            raise _unsafe_run_path(f"{field} is not a directory")
        if field in _RUN_FILE_FIELDS and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise _unsafe_run_path(f"{field} is not a private regular file")
    audit_path = paths.provenance / "audit.jsonl"
    try:
        audit_info = audit_path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(audit_info.st_mode) or not stat.S_ISREG(audit_info.st_mode) or audit_info.st_nlink != 1:
        raise _unsafe_run_path("audit.jsonl is not a private regular file")


def _run_paths_from_root(root: Path) -> RunPaths:
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        status=root / "status.json",
        command=root / "command.sh",
        logs=root / "logs",
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        arrays=root / "arrays",
        plots=root / "plots",
        provenance=root / "provenance",
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


def manifest_sha256(payload: Mapping[str, object]) -> str:
    """Return the stable digest used to bind manifests to registry records."""

    normalized = validate_manifest(payload)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def registry_record_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical digest used by correction records to name prior lines."""

    normalized = _validate_registry_record(payload)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_registry_record(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("registry record must be an object")
    unknown = set(payload) - _REGISTRY_FIELDS
    missing = _REGISTRY_REQUIRED_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"registry record has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"registry record is missing required fields: {sorted(missing)}")

    record = dict(payload)
    _require_safe_name(record["run_id"], "registry.run_id")
    event = record["event"]
    if not isinstance(event, str) or event not in _REGISTRY_EVENTS:
        raise ValueError(f"registry.event must be one of {sorted(_REGISTRY_EVENTS)}")
    artifact_path = record["artifact_path"]
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_absolute():
        raise ValueError("registry.artifact_path must be absolute")
    _require_sha256(record["manifest_sha256"], "registry.manifest_sha256")
    state = record["state"]
    if not isinstance(state, str) or state not in ALLOWED_STATES:
        raise ValueError(f"registry.state must be one of {sorted(ALLOWED_STATES)}")
    if event != "correction" and event != state:
        raise ValueError("registry event must equal state except for correction records")
    parents = _require_list(record["parent_run_ids"], "registry.parent_run_ids")
    if any(not isinstance(parent, str) or not parent for parent in parents):
        raise ValueError("registry.parent_run_ids must contain non-empty run IDs")
    selected_validation = record["selected_validation"]
    if selected_validation is not None and not isinstance(selected_validation, Mapping):
        raise ValueError("registry.selected_validation must be an object or null")
    if not isinstance(record["test_authorized"], bool):
        raise ValueError("registry.test_authorized must be a boolean")
    summary_metrics = record["summary_metrics"]
    if not isinstance(summary_metrics, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in summary_metrics.values()
    ):
        raise ValueError("registry.summary_metrics must map names to numeric scalars")
    if event == "correction":
        _require_sha256(record.get("supersedes_sha256"), "registry.supersedes_sha256")
    elif "supersedes_sha256" in record:
        raise ValueError("only correction records may include supersedes_sha256")
    return record


def _record_key(record: Mapping[str, object]) -> tuple[str, str]:
    return str(record["run_id"]), str(record["event"])


def _parse_registry_text(text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    earlier_by_digest: dict[str, dict[str, object]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"registry line {number} must not be blank")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"registry line {number} is not valid JSON") from exc
        record = _validate_registry_record(payload)
        key = _record_key(record)
        if key in seen:
            raise ValueError(f"registry contains duplicate (run_id, event): {key}")
        if record["event"] == "correction":
            target_digest = str(record["supersedes_sha256"])
            target = earlier_by_digest.get(target_digest)
            if target is None:
                raise ValueError("registry.supersedes_sha256 must name an earlier registry record")
            if target["run_id"] != record["run_id"]:
                raise ValueError("registry.supersedes_sha256 must name a record from the same run")
        seen.add(key)
        records.append(record)
        earlier_by_digest[registry_record_sha256(record)] = record
    return records


def load_registry(registry_path: PathLike[str] | str) -> list[dict[str, object]]:
    """Read and validate the complete append-only JSONL registry."""

    path = Path(registry_path)
    if not path.exists():
        return []
    return _parse_registry_text(path.read_text(encoding="utf-8"))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    _atomic_write_bytes(path, serialized.encode("utf-8"))


def append_registry_records(
    registry_path: PathLike[str] | str, records: list[Mapping[str, object]]
) -> None:
    """Atomically publish new immutable registry records under an advisory lock."""

    path = Path(registry_path)
    if not records:
        raise ValueError("registry publication requires at least one record")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
            existing = _parse_registry_text(existing_text)
            additions = [_validate_registry_record(record) for record in records]
            seen = {_record_key(record) for record in existing}
            for record in additions:
                key = _record_key(record)
                if key in seen:
                    raise ValueError(f"duplicate registry (run_id, event): {key}")
                seen.add(key)
            appended_text = "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
                for record in additions
            )
            separator = "" if not existing_text or existing_text.endswith("\n") else "\n"
            content = existing_text + separator + appended_text
            _parse_registry_text(content)
            _atomic_write_bytes(path, content.encode("utf-8"))
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _require_repo_relative_path(value: PathLike[str] | str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("registry path must be repository-relative and traversal-free")
    return path


def _load_manifest_without_leaf_symlink(path: Path) -> object:
    """Load one run-local manifest while refusing a substituted manifest leaf."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("run-local manifest must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            return json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_private_regular_file(path: Path, flags: int, mode: int = 0o600) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags | nofollow, mode)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise _unsafe_run_path(f"{path.name} is not a private regular file")
    return descriptor


def _read_private_json(path: Path) -> object:
    descriptor = _open_private_regular_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_committed_reservation(
    repo_root: PathLike[str] | str,
    registry_relative_path: PathLike[str] | str,
    manifest: Mapping[str, object],
) -> None:
    """Authorize a worker only when its exact reservation is clean and committed."""

    root = Path(repo_root).resolve()
    registry_path = _require_repo_relative_path(registry_relative_path)
    try:
        normalized_manifest = validate_manifest(manifest)
        expected_digest = manifest_sha256(normalized_manifest)
    except ValueError as exc:
        raise ValueError("committed reservation requires a valid manifest") from exc
    if normalized_manifest["registry_path"] != registry_path.as_posix():
        raise ValueError("committed reservation must name the manifest's declared registry path")
    try:
        artifact_path = Path(normalized_manifest["artifact_path"])
        _assert_no_symlink_components(artifact_path)
        artifact_info = artifact_path.lstat()
        if not stat.S_ISDIR(artifact_info.st_mode) or stat.S_ISLNK(artifact_info.st_mode):
            raise _unsafe_run_path("artifact path is not a real directory")
        _validate_run_paths(_run_paths_from_root(artifact_path), require_root=True)
        disk_manifest = _load_manifest_without_leaf_symlink(artifact_path / "manifest.json")
        disk_manifest = validate_manifest(disk_manifest)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("committed reservation requires a valid run-local manifest") from exc
    if manifest_sha256(disk_manifest) != expected_digest:
        raise ValueError("committed reservation does not match the run-local manifest")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(command, cwd=root, check=False, capture_output=True)
        if result.returncode != 0:
            raise ValueError("committed reservation requires a clean tracked source")
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{registry_path.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("committed reservation registry is unavailable at HEAD") from exc
    records = _parse_registry_text(result.stdout)
    matches = [
        record for record in records
        if record["event"] == "reserved"
        and record["run_id"] == normalized_manifest["run_id"]
        and record["manifest_sha256"] == expected_digest
        and record["artifact_path"] == normalized_manifest["artifact_path"]
        and record["parent_run_ids"] == normalized_manifest["parent_run_ids"]
        and record["test_authorized"] == normalized_manifest["test_authorized"]
    ]
    if len(matches) != 1:
        raise ValueError("worker requires one exact committed reservation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_worker_identity(worker_pid: int | None, host: str | None) -> tuple[int, str]:
    pid = os.getpid() if worker_pid is None else worker_pid
    machine = socket.gethostname() if host is None else host
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
        raise ValueError("worker_pid must be a non-negative integer")
    if not isinstance(machine, str) or not machine:
        raise ValueError("host must be a non-empty string")
    return pid, machine


def _load_run_manifest(paths: RunPaths) -> tuple[dict[str, object], str]:
    _validate_run_paths(paths, require_root=True)
    try:
        payload = _read_private_json(paths.manifest)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("run manifest is not valid JSON") from exc
    manifest = validate_manifest(payload)
    if Path(manifest["artifact_path"]).resolve() != paths.root.resolve():
        raise ValueError("run manifest artifact_path does not match run directory")
    return manifest, manifest_sha256(manifest)


def _append_audit_record(paths: RunPaths, record: Mapping[str, object]) -> None:
    _validate_run_paths(paths, require_root=True)
    audit_path = paths.provenance / "audit.jsonl"
    lock_path = audit_path.with_name(f"{audit_path.name}.lock")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    lock_descriptor = _open_private_regular_file(lock_path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(lock_descriptor, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            audit_descriptor = _open_private_regular_file(
                audit_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT
            )
            with os.fdopen(audit_descriptor, "a", encoding="utf-8") as audit_handle:
                audit_handle.write(encoded)
                audit_handle.flush()
                os.fsync(audit_handle.fileno())
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _load_audit_records(paths: RunPaths) -> list[dict[str, object]]:
    _validate_run_paths(paths, require_root=True)
    audit_path = paths.provenance / "audit.jsonl"
    if not audit_path.exists():
        return []
    records: list[dict[str, object]] = []
    descriptor = _open_private_regular_file(audit_path, os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"audit line {line_number} is not valid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"audit line {line_number} must be an object")
        records.append(record)
    return records


def _ensure_audit_record(paths: RunPaths, record: Mapping[str, object]) -> None:
    transition_id = record.get("transition_id")
    event = record.get("event")
    if not isinstance(transition_id, str) or not isinstance(event, str):
        raise ValueError("audit records require a transition_id and event")
    for previous in _load_audit_records(paths):
        if previous.get("transition_id") == transition_id and previous.get("event") == event:
            if previous != record:
                raise ValueError("existing audit transition record differs from requested transition")
            return
    _append_audit_record(paths, record)


def _find_audit_record(paths: RunPaths, transition_id: str, event: str) -> dict[str, object] | None:
    for record in _load_audit_records(paths):
        if record.get("transition_id") == transition_id and record.get("event") == event:
            return record
    return None


def _transition_id(
    run_id: str,
    expected_state: str,
    next_state: str,
    manifest_digest: str,
    exit_code: int | None,
    output_hashes: Mapping[str, str],
) -> str:
    identity = {
        "run_id": run_id,
        "expected_state": expected_state,
        "next_state": next_state,
        "manifest_sha256": manifest_digest,
        "exit_code": exit_code,
        "output_hashes": dict(output_hashes),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reserve_run_directory(
    paths: RunPaths,
    manifest: Mapping[str, object],
    *,
    worker_pid: int | None = None,
    host: str | None = None,
) -> None:
    """Create one exclusive run directory in the immutable ``reserved`` state."""

    normalized = validate_manifest(manifest)
    _validate_run_paths(paths)
    if Path(normalized["artifact_path"]).resolve() != paths.root.resolve():
        raise ValueError("manifest artifact_path must match canonical run directory")
    worker_pid, host = _validate_worker_identity(worker_pid, host)
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    digest = manifest_sha256(normalized)
    reservation_id = _transition_id(
        str(normalized["run_id"]), "new", "reserved", digest, None, {}
    )

    def reservation_status() -> dict[str, object]:
        return {
        "run_id": normalized["run_id"],
        "state": "reserved",
        "timestamp_utc": _utc_now(),
        "worker_pid": worker_pid,
        "host": host,
        "manifest_sha256": digest,
        "exit_code": None,
        "output_hashes": {},
        "transition_id": reservation_id,
    }

    def reservation_audit(status: Mapping[str, object]) -> dict[str, object]:
        return {
            "event": "reservation_commit",
            "transition_id": reservation_id,
            "status": dict(status),
            "output_hashes": {},
        }

    lock_path = paths.root.parent / f".{normalized['run_id']}.reservation.lock"
    lock_descriptor = _open_private_regular_file(lock_path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(lock_descriptor, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            _validate_run_paths(paths)
            if paths.root.exists():
                try:
                    existing_manifest, existing_digest = _load_run_manifest(paths)
                    if existing_manifest != normalized or existing_digest != digest:
                        raise FileExistsError(f"run directory already exists: {paths.root}")
                    if paths.status.exists():
                        existing_status = _read_private_json(paths.status)
                        if (
                            existing_status.get("state") != "reserved"
                            or existing_status.get("transition_id") != reservation_id
                        ):
                            raise FileExistsError(f"run directory already exists: {paths.root}")
                        for directory in (paths.logs, paths.checkpoints, paths.metrics, paths.arrays, paths.plots, paths.provenance):
                            directory.mkdir(exist_ok=True)
                        _validate_run_paths(paths, require_root=True)
                        audit = reservation_audit(existing_status)
                        if any(
                            item.get("transition_id") == reservation_id
                            and item.get("event") == "reservation_commit"
                            for item in _load_audit_records(paths)
                        ):
                            raise FileExistsError(f"run directory already exists: {paths.root}")
                        _ensure_audit_record(paths, audit)
                        return
                    for directory in (paths.logs, paths.checkpoints, paths.metrics, paths.arrays, paths.plots, paths.provenance):
                        directory.mkdir(exist_ok=True)
                    status = reservation_status()
                    _atomic_write_json(paths.status, status)
                    _ensure_audit_record(paths, reservation_audit(status))
                    return
                except (json.JSONDecodeError, ValueError) as exc:
                    raise FileExistsError(f"run directory already exists: {paths.root}") from exc
            paths.root.mkdir()
            for directory in (paths.logs, paths.checkpoints, paths.metrics, paths.arrays, paths.plots, paths.provenance):
                directory.mkdir()
            _atomic_write_json(paths.manifest, normalized)
            status = reservation_status()
            _atomic_write_json(paths.status, status)
            _ensure_audit_record(paths, reservation_audit(status))
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _validate_output_hashes(output_hashes: Mapping[str, object] | None) -> dict[str, str]:
    if output_hashes is None:
        return {}
    if not isinstance(output_hashes, Mapping):
        raise ValueError("output_hashes must be a mapping")
    normalized: dict[str, str] = {}
    for path, digest in output_hashes.items():
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("output_hashes paths must be non-empty relative paths")
        normalized[path] = _require_sha256(digest, f"output_hashes[{path!r}]")
    return normalized


def _transition_run_status_locked(
    paths: RunPaths,
    expected_state: str,
    next_state: str,
    *,
    worker_pid: int | None = None,
    host: str | None = None,
    exit_code: int | None = None,
    output_hashes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Transition a run while the caller owns its lifecycle lock."""

    _validate_run_paths(paths, require_root=True)
    if expected_state not in ALLOWED_STATES or next_state not in ALLOWED_STATES:
        raise ValueError("expected_state and next_state must be canonical lifecycle states")
    try:
        current = _read_private_json(paths.status)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("run status is not valid JSON") from exc
    current_state = current.get("state") if isinstance(current, Mapping) else None
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise ValueError("exit_code must be an integer or null")
    if next_state in _TERMINAL_STATES and exit_code is None:
        raise ValueError("terminal transitions require an exit_code")
    worker_pid, host = _validate_worker_identity(worker_pid, host)
    manifest, digest = _load_run_manifest(paths)
    if current.get("manifest_sha256") != digest:
        raise ValueError("run status manifest digest does not match manifest")
    hashes = _validate_output_hashes(output_hashes)
    transition_id = _transition_id(
        str(manifest["run_id"]), expected_state, next_state, digest, exit_code, hashes
    )
    if current_state == next_state and current.get("transition_id") == transition_id:
        if current.get("exit_code") != exit_code or current.get("output_hashes") != hashes:
            raise ValueError("existing status differs from requested idempotent transition")
        _ensure_audit_record(
            paths,
            {
                "event": "transition_intent",
                "transition_id": transition_id,
                "from_state": expected_state,
                "to_state": next_state,
                "status": current,
                "output_hashes": hashes,
            },
        )
        _ensure_audit_record(
            paths,
            {
                "event": "transition_commit",
                "transition_id": transition_id,
                "from_state": expected_state,
                "to_state": next_state,
                "status": current,
                "output_hashes": hashes,
            },
        )
        return dict(current)
    if current_state in _TERMINAL_STATES:
        raise FileExistsError("terminal run status is immutable")
    if current_state != expected_state:
        raise ValueError(f"expected current state {expected_state!r}, found {current_state!r}")
    if next_state not in _ALLOWED_TRANSITIONS.get(expected_state, frozenset()):
        raise ValueError(f"transition {expected_state!r} -> {next_state!r} is not allowed")
    prior_intent = _find_audit_record(paths, transition_id, "transition_intent")
    if prior_intent is not None:
        prior_status = prior_intent.get("status")
        if not isinstance(prior_status, Mapping):
            raise ValueError("existing transition intent does not contain a status object")
        status = dict(prior_status)
        if (
            status.get("run_id") != manifest["run_id"]
            or status.get("state") != next_state
            or status.get("manifest_sha256") != digest
            or status.get("exit_code") != exit_code
            or status.get("output_hashes") != hashes
            or status.get("transition_id") != transition_id
        ):
            raise ValueError("existing transition intent differs from requested transition")
    else:
        status = {
            "run_id": manifest["run_id"],
            "state": next_state,
            "timestamp_utc": _utc_now(),
            "worker_pid": worker_pid,
            "host": host,
            "manifest_sha256": digest,
            "exit_code": exit_code,
            "output_hashes": hashes,
            "transition_id": transition_id,
        }
    intent = {
        "event": "transition_intent",
        "transition_id": transition_id,
        "from_state": expected_state,
        "to_state": next_state,
        "status": status,
        "output_hashes": hashes,
    }
    commit = {**intent, "event": "transition_commit"}
    _ensure_audit_record(paths, intent)
    _atomic_write_json(paths.status, status)
    _ensure_audit_record(paths, commit)
    return status


def transition_run_status(
    paths: RunPaths,
    expected_state: str,
    next_state: str,
    *,
    worker_pid: int | None = None,
    host: str | None = None,
    exit_code: int | None = None,
    output_hashes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically transition a live run while preserving an administrative audit trail."""

    _validate_run_paths(paths, require_root=True)
    lock_path = paths.status.with_name(f"{paths.status.name}.lock")
    lock_descriptor = _open_private_regular_file(lock_path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(lock_descriptor, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _transition_run_status_locked(
                paths,
                expected_state,
                next_state,
                worker_pid=worker_pid,
                host=host,
                exit_code=exit_code,
                output_hashes=output_hashes,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
