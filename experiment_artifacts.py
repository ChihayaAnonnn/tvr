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
import secrets
import socket
import stat
import subprocess
import tempfile
from typing import Mapping

import numpy as np
from jsonschema import Draft202012Validator, ValidationError, validators


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
_CANONICAL_NUMERIC_DTYPE = re.compile(
    r"^(?:bool|u?int(?:8|16|32|64)|float(?:16|32|64)|complex(?:64|128))$"
)
_GIT_SHA = re.compile(r"^[0-9a-f]{12,40}$")
_RUN_ID = re.compile(
    r"^(?P<dataset>[A-Za-z0-9][A-Za-z0-9._-]*)-"
    r"(?P<stage>baseline|feature_ablation|loss_ablation|topk_sweep|multiseed|calibration|robustness|efficiency|final)-"
    r"(?P<variant>[A-Za-z0-9][A-Za-z0-9._-]*)-"
    r"seed(?P<seed>\d{3})-(?P<timestamp>\d{8}T\d{6}Z)-(?P<git_sha>[0-9a-f]{12})$"
)


def _validate_canonical_content_address(
    validator: object,
    enabled: object,
    instance: object,
    schema: Mapping[str, object],
):
    """Implement the cross-field invariant JSON Schema cannot express itself."""

    del validator, schema
    if not enabled or not isinstance(instance, Mapping):
        return
    path = instance.get("path")
    digest = instance.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return
    expected_path = f"arrays/sha256/{digest[:2]}/{digest}.npy"
    if path != expected_path:
        yield ValidationError(
            "path must equal the canonical content-address derived from sha256"
        )


ArtifactSchemaValidator = validators.extend(
    Draft202012Validator,
    {"x-canonical-content-address": _validate_canonical_content_address},
)


def artifact_schema_validator(
    schema: Mapping[str, object],
) -> Draft202012Validator:
    """Build the required validator for archive schemas and custom invariants."""

    ArtifactSchemaValidator.check_schema(schema)
    return ArtifactSchemaValidator(schema)

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
        "command_sha256",
        "artifact_locator",
    }
)
_MANIFEST_REQUIRED_FIELDS = _MANIFEST_FIELDS - {"artifact_locator"}
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
        "artifact_locator",
    }
)
_REGISTRY_REQUIRED_FIELDS = _REGISTRY_FIELDS - {
    "supersedes_sha256", "artifact_locator",
}
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
    command_sha256: str
    artifact_locator: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RunManifest":
        return cls(**validate_manifest(payload))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if payload["artifact_locator"] is None:
            del payload["artifact_locator"]
        return payload


_MAX_ARTIFACT_JSON_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ArrayReference:
    """Small, JSON-safe pointer to an immutable numeric artifact."""

    path: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    format: str
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "format": self.format,
            "sha256": self.sha256,
        }


def _array_sha256(array: object) -> str:
    """Hash dtype, shape, and contiguous bytes (the protocol convention)."""

    value = np.asarray(array)
    if value.ndim and not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_numeric_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim and not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise ValueError("artifact arrays must reject object dtype")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("artifact arrays must have numeric or boolean dtype")
    if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
        raise ValueError("artifact arrays must reject non-finite values")
    return array


def _safe_array_key(value: object) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or ".." in value:
        raise ValueError("array key must be a safe semantic name")
    return value


def _array_reference_from_mapping(value: ArrayReference | Mapping[str, object]) -> ArrayReference:
    if isinstance(value, ArrayReference):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise ValueError("array reference must be an object")
    expected = {"path", "key", "shape", "dtype", "byte_size", "format", "sha256"}
    if set(value) != expected:
        raise ValueError("array reference has unexpected fields")
    path = value["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("array reference path must be a non-empty relative path")
    _safe_array_key(value["key"])
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise ValueError("array reference shape must contain non-negative integers")
    dtype = value["dtype"]
    if not isinstance(dtype, str) or not _CANONICAL_NUMERIC_DTYPE.fullmatch(dtype):
        raise ValueError("array reference dtype must use a canonical numeric name")
    try:
        parsed_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError("array reference dtype is invalid") from exc
    if parsed_dtype.hasobject or not (
        np.issubdtype(parsed_dtype, np.number) or np.issubdtype(parsed_dtype, np.bool_)
    ):
        raise ValueError("array reference dtype is unsafe")
    byte_size = value["byte_size"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
        raise ValueError("array reference byte_size must be a non-negative integer")
    if value["format"] != "npy":
        raise ValueError("array reference format must be npy")
    _require_sha256(value["sha256"], "array reference sha256")
    return ArrayReference(
        path=path,
        key=str(value["key"]),
        shape=tuple(shape),
        dtype=str(parsed_dtype),
        byte_size=byte_size,
        format="npy",
        sha256=str(value["sha256"]),
    )


class ArtifactArrayStore:
    """Content-addressed NPY store rooted at one artifact run's arrays dir."""

    def __init__(self, array_dir: PathLike[str] | str):
        root = _canonical_absolute_path(array_dir, "array_dir")
        if root.name != "arrays":
            raise ValueError("array_dir must be the canonical arrays directory")
        self.root = root
        self.run_root = root.parent
        _assert_no_symlink_components(self.root)
        _ensure_safe_directory_chain(self.root)
        _assert_no_symlink_components(self.root)
        root_info = self.root.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise ValueError("array_dir must be a real directory")

    def _reference_for(self, key: str, array: np.ndarray) -> ArrayReference:
        digest = _array_sha256(array)
        return ArrayReference(
            path=(Path(self.root.name) / "sha256" / digest[:2] / f"{digest}.npy").as_posix(),
            key=key,
            shape=tuple(int(item) for item in array.shape),
            dtype=str(array.dtype),
            byte_size=int(array.nbytes),
            format="npy",
            sha256=digest,
        )

    def write(self, key: str, values: object) -> ArrayReference:
        """Atomically publish one finite numeric array, deduplicated by content."""

        key = _safe_array_key(key)
        array = _validate_numeric_array(values)
        reference = self._reference_for(key, array)
        target = self.root / "sha256" / reference.sha256[:2] / f"{reference.sha256}.npy"
        _assert_no_symlink_components(target.parent)
        if target.exists():
            verify_array_reference(self.root.parent, reference)
            return reference
        _ensure_safe_directory_chain(target.parent)
        parent_descriptor = _open_safe_directory(
            self.root, ("sha256", reference.sha256[:2])
        )
        temporary_name = f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except BaseException:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_descriptor)
        verify_array_reference(self.root.parent, reference)
        return reference


def _reference_file_path(
    artifact_root: PathLike[str] | str, reference: ArrayReference
) -> tuple[Path, Path]:
    try:
        root = _canonical_absolute_path(artifact_root, "artifact root")
        _assert_no_symlink_components(root)
    except ValueError as exc:
        raise ValueError("artifact root must be an existing real absolute directory") from exc
    if not root.exists() or root.is_symlink():
        raise ValueError("artifact root must be an existing real absolute directory")
    parts = Path(reference.path).parts
    if not parts or Path(reference.path).is_absolute() or ".." in parts:
        raise ValueError("array reference path traversal is forbidden")
    relative = Path(reference.path)
    expected = Path("arrays") / "sha256" / reference.sha256[:2] / f"{reference.sha256}.npy"
    if root.name == "arrays":
        if parts[0] != "arrays":
            raise ValueError("array reference path must be rooted at arrays")
        relative = Path(*parts[1:])
        expected = expected.relative_to("arrays")
    if relative != expected:
        raise ValueError("array reference path is not canonical content-addressed storage")
    return root, relative


def _open_reference_file(root: Path, relative: Path) -> tuple[int, os.stat_result]:
    """Open a canonical referenced file through no-follow directory FDs."""

    current = _open_safe_directory(root, relative.parts[:-1])
    try:
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.close(descriptor)
            raise ValueError("array reference file is not a private regular file")
        return descriptor, info
    except FileNotFoundError as exc:
        raise ValueError("array reference file is missing") from exc
    except OSError as exc:
        raise ValueError("array reference path is unsafe") from exc
    finally:
        os.close(current)


def _open_safe_directory(root: Path, components: tuple[str, ...]) -> int:
    """Return a held no-follow FD for a child directory chain."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        current = os.open(root, flags)
    except OSError as exc:
        raise ValueError("artifact root is not a safe directory") from exc
    try:
        for component in components:
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise ValueError("array reference directory is unsafe")
        return current
    except FileNotFoundError as exc:
        os.close(current)
        raise ValueError("array reference file is missing") from exc
    except (OSError, ValueError) as exc:
        os.close(current)
        if isinstance(exc, ValueError):
            raise
        raise ValueError("array reference path is unsafe") from exc


def verify_array_reference(
    artifact_root: PathLike[str] | str, reference: ArrayReference | Mapping[str, object]
) -> np.ndarray:
    """Load a referenced NPY without pickle and fail closed on any mismatch."""

    normalized = _array_reference_from_mapping(reference)
    root, relative = _reference_file_path(artifact_root, normalized)
    expected_bytes = int(np.prod(normalized.shape, dtype=object)) * np.dtype(normalized.dtype).itemsize
    if expected_bytes != normalized.byte_size:
        raise ValueError("array reference shape and byte_size disagree")
    descriptor, info = _open_reference_file(root, relative)
    file_size = info.st_size
    if file_size < normalized.byte_size or file_size > normalized.byte_size + 10_000:
        os.close(descriptor)
        raise ValueError("array reference file size is inconsistent with safe NPY storage")
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            array = np.load(handle, allow_pickle=False, max_header_size=10_000)
    except (OSError, ValueError) as exc:
        raise ValueError("array reference is not a safe NPY payload") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(array, np.ndarray):
        raise ValueError("array reference NPY did not contain an array")
    array = _validate_numeric_array(array)
    if str(array.dtype) != normalized.dtype:
        raise ValueError("array reference dtype mismatch")
    if tuple(array.shape) != normalized.shape:
        raise ValueError("array reference shape mismatch")
    if int(array.nbytes) != normalized.byte_size:
        raise ValueError("array reference byte_size mismatch")
    if _array_sha256(array) != normalized.sha256:
        raise ValueError("array reference SHA-256 mismatch")
    return array


def _tensor_to_numpy(value: object) -> np.ndarray | None:
    detach = getattr(value, "detach", None)
    if not callable(detach):
        return None
    cpu = detach().cpu()
    numpy = getattr(cpu, "numpy", None)
    return np.asarray(numpy()) if callable(numpy) else None


def externalize_numeric_payload(
    payload: object,
    store: ArtifactArrayStore,
    *,
    semantic_key: str = "payload",
) -> object:
    """Copy a JSON-shaped payload while replacing numeric arrays with pointers."""

    _safe_array_key(semantic_key)

    def visit(value: object, path: tuple[str, ...]) -> object:
        array = _tensor_to_numpy(value)
        if array is None and isinstance(value, np.ndarray):
            array = value
        if array is not None:
            key = ".".join(path) if path else semantic_key
            return store.write(key, array).as_dict()
        if isinstance(value, Mapping):
            converted: dict[str, object] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("artifact JSON mapping keys must be strings")
                converted[key] = visit(child, path + (key,))
            return converted
        if isinstance(value, (list, tuple)):
            return [visit(child, path + (str(index),)) for index, child in enumerate(value)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return visit(payload, (semantic_key,))


def artifact_json_bytes(
    payload: object, *, ensure_ascii: bool = False, indent: int = 2
) -> bytes:
    """Serialize exactly like an artifact writer after path-aware validation."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, np.ndarray) or _tensor_to_numpy(value) is not None:
            raise ValueError(f"artifact JSON cannot embed numeric array at {path}")
        if isinstance(value, (bytes, bytearray, memoryview)):
            if len(value) > _MAX_ARTIFACT_JSON_BYTES:
                raise ValueError(f"artifact JSON byte string over 1 MiB at {path}")
            raise ValueError(f"artifact JSON cannot embed byte strings at {path}")
        if isinstance(value, np.generic):
            if isinstance(value, np.floating) and not np.isfinite(value):
                raise ValueError(f"artifact JSON contains non-finite number at {path}")
            raise ValueError(f"artifact JSON cannot embed NumPy scalar at {path}")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"artifact JSON contains non-finite number at {path}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"artifact JSON key at {path} must be a string")
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=ensure_ascii,
                allow_nan=False,
                sort_keys=True,
                indent=indent,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact JSON payload is not strict JSON") from exc


def validate_artifact_json_payload(
    payload: object, *, ensure_ascii: bool = False, indent: int = 2
) -> None:
    """Reject payloads that do not fit the exact writer's strict JSON bytes."""

    encoded = artifact_json_bytes(
        payload, ensure_ascii=ensure_ascii, indent=indent
    )
    if len(encoded) >= _MAX_ARTIFACT_JSON_BYTES:
        raise ValueError("artifact JSON document must remain below 1 MiB")


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

    root = _canonical_absolute_path(artifact_root, "artifact_root")
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


def _canonical_absolute_path(value: PathLike[str] | str, field: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not os.path.isabs(raw):
        raise ValueError(f"{field} must be absolute")
    if raw != os.path.normpath(raw):
        raise ValueError(f"{field} must use canonical lexical path spelling")
    return Path(raw)


def _assert_no_symlink_components(path: Path) -> None:
    try:
        canonical_path = _canonical_absolute_path(path, "run path")
    except ValueError as exc:
        raise _unsafe_run_path("path must use canonical lexical spelling") from exc
    current = Path(canonical_path.anchor)
    for component in canonical_path.parts[1:]:
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
    try:
        _canonical_absolute_path(paths.root, "run root")
    except ValueError as exc:
        raise _unsafe_run_path("run root must use canonical lexical spelling") from exc
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


def _validate_artifact_locator(value: object) -> str:
    """Validate a portable path relative to the repository common root."""

    if not isinstance(value, str) or not value:
        raise ValueError("artifact_locator must be a non-empty repository-relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or value.startswith("./")
        or ".." in path.parts
        or "." in path.parts
        or any(not part for part in value.split("/"))
    ):
        raise ValueError("artifact_locator must be normalized and traversal-free")
    return value


def validate_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy a complete, serializable archive manifest payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be an object")
    unknown = set(payload) - _MANIFEST_FIELDS
    missing = _MANIFEST_REQUIRED_FIELDS - set(payload)
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
    try:
        _canonical_absolute_path(artifact_path, "artifact_path")
    except ValueError as exc:
        raise ValueError("artifact_path must be absolute and lexically canonical") from exc
    if "artifact_locator" in normalized:
        _validate_artifact_locator(normalized["artifact_locator"])
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
    _require_sha256(normalized["command_sha256"], "command_sha256")
    return normalized


def manifest_sha256(payload: Mapping[str, object]) -> str:
    """Return the stable digest used to bind manifests to registry records."""

    normalized = validate_manifest(payload)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reservation_record(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return the one canonical registry authorization for a manifest."""

    normalized = validate_manifest(manifest)
    record: dict[str, object] = {
        "run_id": normalized["run_id"],
        "event": "reserved",
        "artifact_path": normalized["artifact_path"],
        "manifest_sha256": manifest_sha256(normalized),
        "state": "reserved",
        "parent_run_ids": normalized["parent_run_ids"],
        "selected_validation": None,
        "test_authorized": normalized["test_authorized"],
        "summary_metrics": {},
    }
    if "artifact_locator" in normalized:
        record["artifact_locator"] = normalized["artifact_locator"]
    return _validate_registry_record(record)


def _validate_summary_metrics(value: object, field: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    metrics: dict[str, int | float] = {}
    for name, metric in value.items():
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"{field} names must be safe strings")
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not np.isfinite(metric)
        ):
            raise ValueError(f"{field} must map names to finite numeric scalars")
        metrics[name] = metric
    return metrics


def _validate_selected_validation(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(
            "registry.selected_validation must be null or an exact path/hash reference"
        )
    path = value["path"]
    if (
        not isinstance(path, str)
        or Path(path).is_absolute()
        or Path(path).parts[:1] != ("metrics",)
        or ".." in Path(path).parts
        or path != Path(path).as_posix()
        or not path.endswith(".json")
    ):
        raise ValueError("registry.selected_validation.path must name canonical metrics JSON")
    digest = _require_sha256(value["sha256"], "registry.selected_validation.sha256")
    return {"path": path, "sha256": digest}


def validate_metrics_document(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the scalar/array-reference document written below ``metrics/``."""

    if not isinstance(payload, Mapping):
        raise ValueError("metrics document must be an object")
    allowed = {"run_id", "split", "summary_metrics", "array_references"}
    required = {"split", "summary_metrics", "array_references"}
    if set(payload) - allowed:
        raise ValueError("metrics document has unknown fields")
    if required - set(payload):
        raise ValueError("metrics document is missing required fields")
    normalized = dict(payload)
    if "run_id" in normalized:
        _require_safe_name(normalized["run_id"], "metrics.run_id")
    if normalized["split"] not in {"train", "validation", "test"}:
        raise ValueError("metrics.split must be train, validation, or test")
    normalized["summary_metrics"] = _validate_summary_metrics(
        normalized["summary_metrics"], "metrics.summary_metrics"
    )
    references = _require_list(normalized["array_references"], "metrics.array_references")
    normalized_references = []
    for index, reference in enumerate(references):
        parsed = _array_reference_from_mapping(reference)
        expected_path = (
            f"arrays/sha256/{parsed.sha256[:2]}/{parsed.sha256}.npy"
        )
        if parsed.path != expected_path:
            raise ValueError(
                f"metrics.array_references[{index}].path is not canonical content-addressed storage"
            )
        normalized_references.append(parsed.as_dict())
    normalized["array_references"] = normalized_references
    validate_artifact_json_payload(normalized)
    return normalized


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
    if "artifact_locator" in record:
        _validate_artifact_locator(record["artifact_locator"])
    _require_sha256(record["manifest_sha256"], "registry.manifest_sha256")
    state = record["state"]
    if not isinstance(state, str) or state not in ALLOWED_STATES:
        raise ValueError(f"registry.state must be one of {sorted(ALLOWED_STATES)}")
    if event != "correction" and event != state:
        raise ValueError("registry event must equal state except for correction records")
    parents = _require_list(record["parent_run_ids"], "registry.parent_run_ids")
    if any(not isinstance(parent, str) or not parent for parent in parents):
        raise ValueError("registry.parent_run_ids must contain non-empty run IDs")
    selected_validation = _validate_selected_validation(record["selected_validation"])
    record["selected_validation"] = selected_validation
    if not isinstance(record["test_authorized"], bool):
        raise ValueError("registry.test_authorized must be a boolean")
    summary_metrics = _validate_summary_metrics(
        record["summary_metrics"], "registry.summary_metrics"
    )
    record["summary_metrics"] = summary_metrics
    if event == "reserved" and (selected_validation is not None or summary_metrics):
        raise ValueError(
            "reserved registry records require empty selection and summary metrics"
        )
    if event == "correction":
        _require_sha256(record.get("supersedes_sha256"), "registry.supersedes_sha256")
    elif "supersedes_sha256" in record:
        raise ValueError("only correction records may include supersedes_sha256")
    validate_artifact_json_payload(record, ensure_ascii=True, indent=None)
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


def parse_registry_text(text: str) -> list[dict[str, object]]:
    """Validate registry content obtained from an immutable source such as Git."""

    return _parse_registry_text(text)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_safe_directory_chain(path: Path) -> None:
    """Create a canonical directory chain without following existing symlinks.

    Every newly linked child is fsynced in both its parent and itself before the
    caller may publish a reference below it.
    """

    canonical = _canonical_absolute_path(path, "directory path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(canonical.anchor, flags)
    try:
        for component in canonical.parts[1:]:
            created = False
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    created = True
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=current)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise _unsafe_run_path(f"directory component is unsafe: {component}")
            if created:
                os.fsync(current)
                os.fsync(child)
            os.close(current)
            current = child
    except OSError as exc:
        raise _unsafe_run_path("directory component is unsafe") from exc
    finally:
        os.close(current)


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


def _private_file_sha256(path: Path) -> str:
    descriptor = _open_private_regular_file(path, os.O_RDONLY)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def hash_run_outputs(paths: RunPaths) -> dict[str, str]:
    """Hash the complete immutable output inventory without following links."""

    _validate_run_paths(paths, require_root=True)
    hashes: dict[str, str] = {}
    for directory in (
        paths.logs, paths.checkpoints, paths.metrics, paths.arrays, paths.plots,
        paths.provenance,
    ):
        for current, directory_names, file_names, descriptor in os.fwalk(
            directory, follow_symlinks=False
        ):
            for name in directory_names:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    raise _unsafe_run_path(f"unsafe output directory: {Path(current) / name}")
            for name in sorted(file_names):
                if directory == paths.provenance and (
                    name == "audit.jsonl" or name.endswith(".lock")
                    or name.startswith("controller-pid.")
                ):
                    continue
                file_descriptor = os.open(
                    name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    info = os.fstat(file_descriptor)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise _unsafe_run_path(
                            f"unsafe output file: {Path(current) / name}"
                        )
                    digest = hashlib.sha256()
                    while chunk := os.read(file_descriptor, 1024 * 1024):
                        digest.update(chunk)
                finally:
                    os.close(file_descriptor)
                relative = (Path(current) / name).relative_to(paths.root).as_posix()
                hashes[relative] = digest.hexdigest()
    hashes["command.sh"] = _private_file_sha256(paths.command)
    return hashes


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
    source_commit = str(normalized_manifest["git_sha"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root, check=False, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("committed reservation git_sha is not an ancestor of HEAD")
    source_changes = subprocess.run(
        ["git", "diff", "--name-only", f"{source_commit}..HEAD", "--"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    forbidden_source_changes = set(source_changes) - {registry_path.as_posix()}
    if forbidden_source_changes:
        raise ValueError(
            "committed reservation source differs from manifest.git_sha: "
            + ", ".join(sorted(forbidden_source_changes))
        )
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
    expected_record = reservation_record(normalized_manifest)
    matches = [record for record in records if record == expected_record]
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
            lock_handle.seek(0)
            reservation_intent = json.dumps(
                {"manifest_sha256": digest, "reservation_id": reservation_id},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ) + "\n"
            existing_intent = lock_handle.read()
            if existing_intent:
                if existing_intent != reservation_intent:
                    raise FileExistsError(f"run reservation intent differs: {paths.root}")
            else:
                if paths.root.exists():
                    raise FileExistsError(f"run directory already exists: {paths.root}")
                lock_handle.write(reservation_intent)
                lock_handle.flush()
                os.fsync(lock_handle.fileno())
                _fsync_directory(lock_path.parent)
            _validate_run_paths(paths)
            if paths.root.exists():
                try:
                    if not paths.manifest.exists():
                        allowed = {path.name for path in (
                            paths.logs, paths.checkpoints, paths.metrics,
                            paths.arrays, paths.plots, paths.provenance,
                        )}
                        for entry in os.scandir(paths.root):
                            if entry.name not in allowed or not entry.is_dir(
                                follow_symlinks=False
                            ):
                                raise FileExistsError(
                                    f"run directory already exists: {paths.root}"
                                )
                            with os.scandir(entry.path) as children:
                                if next(children, None) is not None:
                                    raise FileExistsError(
                                        f"run directory already exists: {paths.root}"
                                    )
                        for directory in (
                            paths.logs, paths.checkpoints, paths.metrics,
                            paths.arrays, paths.plots, paths.provenance,
                        ):
                            directory.mkdir(exist_ok=True)
                        _validate_run_paths(paths, require_root=True)
                        _atomic_write_json(paths.manifest, normalized)
                        status = reservation_status()
                        _atomic_write_json(paths.status, status)
                        _ensure_audit_record(paths, reservation_audit(status))
                        return
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


def load_validated_run_lifecycle(
    paths: RunPaths,
    *,
    require_command: bool = True,
    allow_pending_commit: bool = False,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Load and fully authenticate one run-local manifest/status/audit chain."""

    manifest, digest = _load_run_manifest(paths)
    if require_command and _private_file_sha256(paths.command) != manifest["command_sha256"]:
        raise ValueError("sealed command SHA-256 differs from manifest")
    try:
        status_payload = _read_private_json(paths.status)
    except json.JSONDecodeError as exc:
        raise ValueError("run status is not valid JSON") from exc
    if not isinstance(status_payload, Mapping):
        raise ValueError("run status must be an object")
    required = {
        "run_id", "state", "timestamp_utc", "worker_pid", "host",
        "manifest_sha256", "exit_code", "output_hashes", "transition_id",
    }
    if set(status_payload) != required:
        raise ValueError("run status fields do not match the status schema")
    status = dict(status_payload)
    if status["run_id"] != manifest["run_id"]:
        raise ValueError("run status run_id differs from manifest")
    state = status["state"]
    if not isinstance(state, str) or state not in ALLOWED_STATES:
        raise ValueError("run status state is invalid")
    try:
        parsed = datetime.fromisoformat(str(status["timestamp_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("run status timestamp_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("run status timestamp_utc must be timezone-aware")
    _validate_worker_identity(status["worker_pid"], status["host"])
    if status["manifest_sha256"] != digest:
        raise ValueError("run status manifest digest does not match manifest")
    exit_code = status["exit_code"]
    if state in _TERMINAL_STATES:
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("terminal run status requires an integer exit_code")
    elif exit_code is not None:
        raise ValueError("non-terminal run status requires a null exit_code")
    hashes = _validate_output_hashes(status["output_hashes"])
    transition_id = _require_sha256(status["transition_id"], "status.transition_id")
    audits = _load_audit_records(paths)
    seen_audits: set[tuple[str, str]] = set()
    for index, record in enumerate(audits):
        event = record.get("event")
        audit_id = _require_sha256(
            record.get("transition_id"), f"audit[{index}].transition_id"
        )
        key = (str(event), audit_id)
        if key in seen_audits:
            raise ValueError("audit contains a duplicate event/transition identity")
        seen_audits.add(key)
        if event == "reservation_commit":
            expected_fields = {"event", "transition_id", "status", "output_hashes"}
            from_state, to_state = "new", "reserved"
        elif event in {"transition_intent", "transition_commit"}:
            expected_fields = {
                "event", "transition_id", "from_state", "to_state", "status",
                "output_hashes",
            }
            from_state, to_state = record.get("from_state"), record.get("to_state")
            if not isinstance(from_state, str) or not isinstance(to_state, str):
                raise ValueError("audit transition states must be strings")
            if to_state not in _ALLOWED_TRANSITIONS.get(from_state, frozenset()):
                raise ValueError("audit contains a forbidden lifecycle transition")
        else:
            raise ValueError("audit event is not recognized")
        if set(record) != expected_fields:
            raise ValueError("audit record fields do not match its schema")
        embedded = record.get("status")
        if not isinstance(embedded, Mapping) or set(embedded) != required:
            raise ValueError("audit embedded status fields do not match the status schema")
        if embedded.get("run_id") != manifest["run_id"] or embedded.get("state") != to_state:
            raise ValueError("audit embedded status identity is invalid")
        if embedded.get("manifest_sha256") != digest:
            raise ValueError("audit embedded manifest digest is invalid")
        try:
            audit_time = datetime.fromisoformat(
                str(embedded.get("timestamp_utc")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("audit embedded timestamp is invalid") from exc
        if audit_time.tzinfo is None:
            raise ValueError("audit embedded timestamp must be timezone-aware")
        _validate_worker_identity(embedded.get("worker_pid"), embedded.get("host"))
        embedded_exit = embedded.get("exit_code")
        if to_state in _TERMINAL_STATES:
            if isinstance(embedded_exit, bool) or not isinstance(embedded_exit, int):
                raise ValueError("terminal audit status requires an integer exit_code")
        elif embedded_exit is not None:
            raise ValueError("non-terminal audit status requires a null exit_code")
        embedded_hashes = _validate_output_hashes(embedded.get("output_hashes"))
        if record.get("output_hashes") != embedded_hashes:
            raise ValueError("audit output hashes differ from embedded status")
        if embedded.get("transition_id") != audit_id:
            raise ValueError("audit embedded transition identity differs")
        if _transition_id(
            str(manifest["run_id"]), str(from_state), str(to_state), digest,
            embedded_exit, embedded_hashes,
        ) != audit_id:
            raise ValueError("audit transition identity is invalid")
    if not audits or audits[0].get("event") != "reservation_commit":
        raise ValueError("audit chain must begin with one reservation commit")
    chain_status = audits[0]["status"]
    chain_state = "reserved"
    pending_intent = None
    audit_index = 1
    while audit_index < len(audits):
        intent = audits[audit_index]
        if intent.get("event") != "transition_intent":
            raise ValueError("audit chain requires intent/commit ordering")
        if intent.get("from_state") != chain_state:
            raise ValueError("audit state chain is not contiguous")
        if audit_index + 1 < len(audits):
            commit = audits[audit_index + 1]
            expected_commit = {**intent, "event": "transition_commit"}
            if commit != expected_commit:
                raise ValueError("every audit intent requires its matching commit")
            chain_status = intent["status"]
            chain_state = str(intent["to_state"])
            audit_index += 2
            continue
        if not allow_pending_commit:
            raise ValueError("every audit intent requires its matching commit")
        pending_intent = intent
        audit_index += 1
    if pending_intent is None:
        if status != chain_status:
            raise ValueError("current status differs from the committed audit chain")
    elif status not in (chain_status, pending_intent["status"]):
        raise ValueError("current status is unrelated to the pending audit transition")
    if state == "reserved":
        matches = [
            record for record in audits
            if record.get("event") == "reservation_commit"
            and record.get("transition_id") == transition_id
            and record.get("status") == status
            and record.get("output_hashes") == hashes
        ]
        if len(matches) != 1:
            raise ValueError("reserved status lacks one matching audit commit")
    else:
        commits = [
            record for record in audits
            if record.get("event") == "transition_commit"
            and record.get("transition_id") == transition_id
            and record.get("status") == status
            and record.get("output_hashes") == hashes
            and record.get("to_state") == state
        ]
        intents = [
            record for record in audits
            if record.get("event") == "transition_intent"
            and record.get("transition_id") == transition_id
            and record.get("status") == status
            and record.get("to_state") == state
            and record.get("output_hashes") == hashes
        ]
        if len(commits) == 1:
            transition_record = commits[0]
        elif allow_pending_commit and not commits and len(intents) == 1:
            transition_record = intents[0]
        else:
            raise ValueError("run status lacks one matching transition audit commit")
        from_state = transition_record.get("from_state")
        if not isinstance(from_state, str) or state not in _ALLOWED_TRANSITIONS.get(from_state, frozenset()):
            raise ValueError("run status audit transition is not allowed")
        expected_id = _transition_id(
            str(manifest["run_id"]), from_state, state, digest, exit_code, hashes
        )
        if expected_id != transition_id:
            raise ValueError("run status transition identity is invalid")
        matching_intents = [
            record for record in intents if record.get("from_state") == from_state
        ]
        if len(matching_intents) != 1:
            raise ValueError("run status lacks one matching transition audit intent")
    return manifest, status, audits


def pending_audit_transition(
    audits: list[Mapping[str, object]],
) -> dict[str, object] | None:
    """Return the single validated trailing intent without a commit, if any."""

    committed = {
        record.get("transition_id") for record in audits
        if record.get("event") == "transition_commit"
    }
    pending = [
        dict(record) for record in audits
        if record.get("event") == "transition_intent"
        and record.get("transition_id") not in committed
    ]
    if len(pending) > 1:
        raise ValueError("audit contains multiple pending transitions")
    return pending[0] if pending else None
