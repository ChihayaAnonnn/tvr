#!/usr/bin/env python3
"""Administrative CLI for the run-centric experiment artifact archive.

This command deliberately separates tracked, phase-batched publication from
worker-local lifecycle changes.  It never launches a training process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_artifacts import (
    RunPaths,
    StrictJSONError,
    append_registry_records,
    build_run_id,
    hash_run_outputs,
    load_registry,
    load_validated_run_lifecycle,
    make_run_paths,
    manifest_sha256,
    parse_registry_text,
    pending_audit_transition,
    reservation_record,
    reserve_run_directory,
    strict_json_dump,
    strict_json_dumps,
    strict_json_load,
    strict_json_loads,
    transition_run_status,
    validate_artifact_json_payload,
    validate_manifest,
    validate_metrics_document,
    verify_committed_reservation,
)


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
REGISTERED_A3_FEATURE_SETS = frozenset(
    {"full", "score_only", "semantic_only", "no_context", "no_score_geometry"}
)
REGISTERED_A3_TOPK = frozenset({1, 4, 8, 16, 32})
REGISTERED_A3_LOSSES = frozenset(
    {(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25)}
)
REGISTERED_A3_SEEDS = frozenset({0, 1, 2})
REGISTERED_A3_EPOCHS = frozenset({3, 5})
RUN_SPEC_FIELDS = frozenset(
    {
        "dataset", "stage", "variant", "seed", "environment",
        "input_checkpoints", "dataset_splits", "config", "parent_run_ids",
        "selection_lineage", "validation_authorized", "test_authorized",
        "expected_scalar_outputs", "array_references",
    }
)
RUN_SPEC_TIMESTAMP_FIELDS = frozenset({"timestamp_utc", "scheduled_timestamp_utc"})


def registered_a3_variant_argv(manifest: Mapping[str, object]) -> list[str]:
    """Derive the only mutable scientific A3 argv from a committed manifest."""

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("registered A3 manifest config is required")
    categories = {}
    for name in ("model", "loss", "optimizer", "scheduler", "evaluation"):
        value = config.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"registered A3 config.{name} must be an object")
        categories[name] = value
    duplicated = sorted(
        name
        for name, count in Counter(
            field for values in categories.values() for field in values
        ).items()
        if count > 1
    )
    if duplicated:
        raise ValueError(
            "registered A3 config duplicates effective field "
            + ", ".join(duplicated)
        )
    scenario = categories["evaluation"].get("archive_qualification_scenario")
    if scenario not in (None, "none"):
        if scenario not in {
            "success", "non_owner_failure", "owner_collective_failure",
            "sigterm_different_collectives",
        }:
            raise ValueError("archive qualification scenario is not allowed")
        return []

    fields = {
        "feature_set": categories["model"].get("contextual_risk_feature_set"),
        "topk": categories["model"].get("contextual_risk_topk"),
        "fit_weight": categories["loss"].get("w_contextual_risk_fit"),
        "rank_weight": categories["loss"].get("w_contextual_risk_rank"),
        "seed": categories["optimizer"].get("seed"),
        "epochs": categories["scheduler"].get("epochs"),
    }
    if type(fields["feature_set"]) is not str or fields["feature_set"] not in REGISTERED_A3_FEATURE_SETS:
        raise ValueError("registered A3 feature set is outside the frozen allowlist")
    if type(fields["topk"]) is not int or fields["topk"] not in REGISTERED_A3_TOPK:
        raise ValueError("registered A3 topk is outside the frozen allowlist")
    if type(fields["fit_weight"]) is not float or type(fields["rank_weight"]) is not float:
        raise ValueError("registered A3 loss weights must be explicit floats")
    if (fields["fit_weight"], fields["rank_weight"]) not in REGISTERED_A3_LOSSES:
        raise ValueError("registered A3 loss setting is outside the frozen allowlist")
    if type(fields["seed"]) is not int or fields["seed"] not in REGISTERED_A3_SEEDS:
        raise ValueError("registered A3 seed is outside the frozen allowlist")
    if fields["seed"] != manifest.get("seed"):
        raise ValueError("registered A3 optimizer seed must equal manifest seed")
    if type(fields["epochs"]) is not int or fields["epochs"] not in REGISTERED_A3_EPOCHS:
        raise ValueError("registered A3 epochs is outside the frozen allowlist")
    if manifest.get("stage") == "final" and fields["feature_set"] != "full":
        raise ValueError("registered A3 final stage allows only the full feature set")
    return [
        "--contextual_risk_feature_set", str(fields["feature_set"]),
        "--contextual_risk_topk", str(fields["topk"]),
        "--w_contextual_risk_fit", str(fields["fit_weight"]),
        "--w_contextual_risk_rank", str(fields["rank_weight"]),
        "--seed", str(fields["seed"]),
        "--epochs", str(fields["epochs"]),
    ]


def _repo_relative(value: str | os.PathLike[str], field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be repository-relative and traversal-free")
    return path


def _canonical_root(value: str | os.PathLike[str], field: str) -> Path:
    raw = os.fspath(value)
    if not os.path.isabs(raw) or raw != os.path.normpath(raw):
        raise ValueError(f"{field} must be an absolute canonical path")
    return Path(raw)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout


def _git_common_root(repo: Path) -> Path:
    common_dir = Path(
        _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    )
    return common_dir.parent.resolve()


def _require_clean(repo: Path) -> None:
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", *arguments], cwd=repo, check=False)
        if result.returncode != 0:
            raise ValueError("operation requires clean tracked source")


def _load_committed_phase(repo: Path, relative: Path) -> dict[str, object]:
    disk = repo / relative
    try:
        committed = _git(repo, "show", f"HEAD:{relative.as_posix()}")
    except subprocess.CalledProcessError as exc:
        raise ValueError("phase JSON must exist in HEAD") from exc
    try:
        disk_text = disk.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("phase JSON is unavailable in the working tree") from exc
    if disk_text != committed:
        raise ValueError("phase JSON must exactly match its committed HEAD version")
    try:
        payload = strict_json_loads(committed)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        raise ValueError("phase JSON is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {"phase", "study", "runs"}:
        raise ValueError("phase JSON must contain exactly phase, study, and runs")
    if not isinstance(payload["phase"], str) or not payload["phase"]:
        raise ValueError("phase must be a non-empty string")
    if not isinstance(payload["study"], str) or not payload["study"]:
        raise ValueError("study must be a non-empty string")
    runs = payload["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("phase runs must be a non-empty list")
    for index, spec in enumerate(runs):
        timestamp_fields = (
            set(spec) & RUN_SPEC_TIMESTAMP_FIELDS if isinstance(spec, dict) else set()
        )
        if (
            not isinstance(spec, dict)
            or not RUN_SPEC_FIELDS.issubset(spec)
            or set(spec) - RUN_SPEC_FIELDS - RUN_SPEC_TIMESTAMP_FIELDS != {"command"}
            or len(timestamp_fields) != 1
        ):
            raise ValueError(f"phase runs[{index}] has an invalid manifest template")
        command = spec.get("command")
        if (
            not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item or "\0" in item or "\n" in item for item in command)
        ):
            raise ValueError(f"phase runs[{index}].command must be a non-empty argv list")
    return payload


def _run_identity_timestamp(spec: Mapping[str, object]) -> str:
    """Return the scheduled identity time, accepting the historical field spelling."""

    field = (
        "scheduled_timestamp_utc"
        if "scheduled_timestamp_utc" in spec
        else "timestamp_utc"
    )
    return str(spec[field])


def _build_manifest(
    spec: Mapping[str, object], *, study: str, git_sha: str,
    artifact_root: Path, registry: Path, common_root: Path,
) -> dict[str, object]:
    run_id = build_run_id(
        str(spec["dataset"]), str(spec["stage"]), str(spec["variant"]),
        spec["seed"], _run_identity_timestamp(spec), git_sha,  # type: ignore[arg-type]
    )
    paths = make_run_paths(artifact_root, study, run_id)
    command = _render_command(spec["command"], run_id=run_id, run_dir=paths.root)
    generated = {
        key: value for key, value in spec.items()
        if key not in RUN_SPEC_TIMESTAMP_FIELDS | {"command"}
    }
    payload = {
            **generated,
            "run_id": run_id,
            "study": study,
            "artifact_path": str(paths.root),
            "registry_path": registry.as_posix(),
            "git_sha": git_sha,
            "tracked_clean": True,
            "source_patch_sha256": None,
            "command_sha256": hashlib.sha256(command).hexdigest(),
    }
    try:
        payload["artifact_locator"] = paths.root.relative_to(common_root).as_posix()
    except ValueError:
        pass
    return validate_manifest(payload)


def _render_command(command: Sequence[str], *, run_id: str, run_dir: Path) -> bytes:
    replacements = {"{run_id}": run_id, "{artifact_run_dir}": str(run_dir)}
    command_argv = []
    for original in command:
        item = original
        if "{manifest_sha256}" in item:
            raise ValueError("command cannot contain self-referential manifest SHA")
        for marker, value in replacements.items():
            item = item.replace(marker, value)
        command_argv.append(item)
    return (shlex.join(command_argv) + "\n").encode("utf-8")


def _registry_record(
    manifest: Mapping[str, object], state: str, *,
    summary_metrics: Mapping[str, object] | None = None,
    selected_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if state == "reserved" and not summary_metrics and not selected_validation:
        return reservation_record(manifest)
    record = {
        "run_id": manifest["run_id"],
        "event": state,
        "artifact_path": manifest["artifact_path"],
        "manifest_sha256": manifest_sha256(manifest),
        "state": state,
        "parent_run_ids": manifest["parent_run_ids"],
        "selected_validation": dict(selected_validation) if selected_validation else None,
        "test_authorized": manifest["test_authorized"],
        "summary_metrics": dict(summary_metrics or {}),
    }
    if "artifact_locator" in manifest:
        record["artifact_locator"] = manifest["artifact_locator"]
    return record


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.close(descriptor)
            raise ValueError(f"{description} must be a private regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = strict_json_load(handle)
    except (OSError, json.JSONDecodeError, StrictJSONError) as exc:
        raise ValueError(f"{description} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be an object")
    return payload


def _load_run(
    run_dir: Path, *, allow_pending_commit: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    if not run_dir.is_absolute() or os.fspath(run_dir) != os.path.normpath(run_dir):
        raise ValueError("run-dir must be an absolute canonical path")
    paths = RunPaths(
        root=run_dir, manifest=run_dir / "manifest.json",
        status=run_dir / "status.json", command=run_dir / "command.sh",
        logs=run_dir / "logs", checkpoints=run_dir / "checkpoints",
        metrics=run_dir / "metrics", arrays=run_dir / "arrays",
        plots=run_dir / "plots", provenance=run_dir / "provenance",
    )
    manifest, status, _audits = load_validated_run_lifecycle(
        paths, allow_pending_commit=allow_pending_commit
    )
    return manifest, status


def _assert_identity(
    manifest: Mapping[str, object], *, run_id: str | None, digest: str | None
) -> None:
    if run_id is not None and manifest["run_id"] != run_id:
        raise ValueError("run ID differs from the requested exact identity")
    actual = manifest_sha256(manifest)
    if digest is not None and actual != digest:
        raise ValueError("manifest SHA-256 differs from the requested exact identity")


def _reservation_for_spec(
    spec: Mapping[str, object], records: Sequence[Mapping[str, object]], artifact_root: Path,
) -> Mapping[str, object]:
    prefix = (
        f"{spec['dataset']}-{spec['stage']}-{spec['variant']}-"
        f"seed{int(spec['seed']):03d}-{_run_identity_timestamp(spec)}-"
    )
    matches = [
        record for record in records
        if record["event"] == "reserved"
        and str(record["run_id"]).startswith(prefix)
        and Path(str(record["artifact_path"])).is_relative_to(artifact_root)
    ]
    if len(matches) != 1:
        raise ValueError(f"phase run {prefix} requires exactly one reservation")
    return matches[0]


def _safe_file_sha256(run_dir: Path, relative_name: str) -> str:
    relative = Path(relative_name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("output paths must be relative and traversal-free")
    current = run_dir
    for part in relative.parts[:-1]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unsafe output path: {relative_name}")
    path = run_dir / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"unsafe output file: {relative_name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _all_output_hashes(
    run_dir: Path, *, allow_pending_commit: bool = False
) -> dict[str, str]:
    manifest, _ = _load_run(
        run_dir, allow_pending_commit=allow_pending_commit
    )
    paths = make_run_paths(
        run_dir.parents[3], str(manifest["study"]), str(manifest["run_id"])
    )
    return hash_run_outputs(paths)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    validate_artifact_json_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            strict_json_dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite artifact JSON: {path}") from exc
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reserve_phase(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    phase_path = _repo_relative(args.phase, "phase")
    registry = _repo_relative(args.registry, "registry")
    artifact_root = _canonical_root(args.artifact_root, "artifact-root")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
        raise ValueError("reservation recovery requires a clean staged index")
    phase = _load_committed_phase(repo, phase_path)
    git_sha = _git(repo, "rev-parse", "HEAD").strip()
    common_root = _git_common_root(repo)
    manifests = [
        _build_manifest(
            spec, study=str(phase["study"]), git_sha=git_sha,
            artifact_root=artifact_root, registry=registry, common_root=common_root,
        )
        for spec in phase["runs"]  # type: ignore[union-attr]
    ]
    run_ids = [str(manifest["run_id"]) for manifest in manifests]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("phase contains duplicate run identities")
    committed_text = _git(repo, "show", f"HEAD:{registry.as_posix()}")
    working = load_registry(repo / registry)
    expected_records = [_registry_record(manifest, "reserved") for manifest in manifests]
    committed_records = parse_registry_text(committed_text)
    committed_exact = [record in committed_records for record in expected_records]
    if any(committed_exact) and not all(committed_exact):
        raise ValueError("phase reservation is only partially committed at HEAD")
    additions = [] if all(committed_exact) else expected_records
    if working == committed_records:
        registry_already_appended = not additions
    elif additions and working == committed_records + additions:
        registry_already_appended = True
    else:
        raise ValueError(
            "reservation requires the committed registry or its exact crash-recovery append"
        )
    changed = set(_git(repo, "diff", "--name-only").splitlines())
    if changed - {registry.as_posix()}:
        raise ValueError("reservation recovery requires clean tracked source")
    existing = committed_records
    for spec in phase["runs"]:  # type: ignore[union-attr]
        prefix = (
            f"{spec['dataset']}-{spec['stage']}-{spec['variant']}-"
            f"seed{int(spec['seed']):03d}-{_run_identity_timestamp(spec)}-"
        )
        if any(
            record["event"] == "reserved"
            and str(record["run_id"]).startswith(prefix)
            and record not in expected_records
            for record in existing
        ):
            raise ValueError(
                "phase contains an already registered identity; silent retry rejected"
            )
    for manifest, spec in zip(manifests, phase["runs"]):  # type: ignore[arg-type]
        paths = make_run_paths(artifact_root, str(phase["study"]), str(manifest["run_id"]))
        command = _render_command(
            spec["command"], run_id=str(manifest["run_id"]), run_dir=paths.root
        )
        if paths.root.exists():
            if not registry_already_appended:
                try:
                    reserve_run_directory(paths, manifest)
                except FileExistsError:
                    pass
            recovered_manifest, recovered_status, _ = load_validated_run_lifecycle(
                paths, require_command=False
            )
            if recovered_manifest != manifest or recovered_status["state"] != "reserved":
                raise ValueError("existing run directory is not the exact reserved recovery")
        else:
            if registry_already_appended:
                raise ValueError("committed reservation is missing its run directory")
            reserve_run_directory(paths, manifest)
        if paths.command.exists():
            if _safe_file_sha256(paths.root, "command.sh") != hashlib.sha256(command).hexdigest():
                raise ValueError("existing recovery command differs from sealed command")
        else:
            if registry_already_appended:
                raise ValueError("published reservation is missing its sealed command")
            descriptor = os.open(
                paths.command,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o700,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(command)
                handle.flush()
                os.fsync(handle.fileno())
    if not registry_already_appended:
        append_registry_records(repo / registry, additions)
    return {
        "phase": phase["phase"],
        "study": phase["study"],
        "artifact_root": str(artifact_root),
        "runs": [
            {
                "run_id": manifest["run_id"],
                "artifact_path": manifest["artifact_path"],
                "manifest_sha256": manifest_sha256(manifest),
            }
            for manifest in manifests
        ],
    }


def _verify_run_identity(
    args: argparse.Namespace, *, allow_pending_commit: bool = False
) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    registry = _repo_relative(args.registry, "registry")
    manifest, status = _load_run(
        Path(args.run_dir), allow_pending_commit=allow_pending_commit
    )
    _assert_identity(manifest, run_id=args.run_id, digest=args.manifest_sha256)
    verify_committed_reservation(repo, registry, manifest)
    command_digest = _safe_file_sha256(Path(args.run_dir), "command.sh")
    if command_digest != manifest["command_sha256"]:
        raise ValueError("sealed command SHA-256 mismatch")
    descriptor = os.open(
        Path(args.run_dir) / "command.sh",
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        command_argv = shlex.split(handle.read())
    if not command_argv:
        raise ValueError("sealed command must contain an executable argv")
    registered_a3_command = [
        "env", f"ARTIFACT_RUN_ID={manifest['run_id']}", "bash",
        "run_train_msrvtt_contextual_risk_bg.sh",
    ]
    return {
        "run_id": manifest["run_id"],
        "run_dir": manifest["artifact_path"],
        "manifest_sha256": manifest_sha256(manifest),
        "state": status["state"],
        "stage": manifest["stage"],
        "test_authorized": manifest["test_authorized"],
        "command_argv": command_argv,
        "registered_a3_variant_argv": (
            registered_a3_variant_argv(manifest)
            if manifest["study"] == "contextual-risk-v2"
            and command_argv == registered_a3_command
            else []
        ),
    }


def verify_run(args: argparse.Namespace) -> dict[str, object]:
    return _verify_run_identity(args)


def start_run(args: argparse.Namespace) -> dict[str, object]:
    result = _verify_run_identity(args, allow_pending_commit=True)
    run_dir = Path(args.run_dir)
    paths = make_run_paths_from_dir(run_dir, allow_pending_commit=True)
    _manifest, _status, audits = load_validated_run_lifecycle(
        paths, allow_pending_commit=True
    )
    pending = pending_audit_transition(audits)
    if pending is not None and (
        pending.get("from_state") != "reserved"
        or pending.get("to_state") != "running"
        or pending.get("status", {}).get("exit_code") is not None
        or pending.get("output_hashes") != {}
    ):
        raise ValueError("pending lifecycle transition must be repaired before start-run")
    status = transition_run_status(
        paths, "reserved", "running"
    )
    return {**result, "state": status["state"]}


def make_run_paths_from_dir(run_dir: Path, *, allow_pending_commit: bool = False):
    manifest, _ = _load_run(run_dir, allow_pending_commit=allow_pending_commit)
    artifact_root = run_dir.parents[3]
    return make_run_paths(artifact_root, str(manifest["study"]), str(manifest["run_id"]))


def transition_run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    _verify_run_identity(args, allow_pending_commit=True)
    if args.expected == "reserved" and args.next_state == "running":
        raise ValueError("reserved runs may only be started through start-run")
    paths = make_run_paths_from_dir(run_dir, allow_pending_commit=True)
    hashes = {
        relative: _safe_file_sha256(run_dir, relative)
        for relative in (args.output or [])
    }
    if args.next_state in TERMINAL_STATES:
        complete_inventory = _all_output_hashes(
            run_dir, allow_pending_commit=True
        )
        for relative, digest in hashes.items():
            if complete_inventory.get(relative) != digest:
                raise ValueError(f"terminal output changed while hashing: {relative}")
        hashes = complete_inventory
    _manifest, _status, audits = load_validated_run_lifecycle(
        paths, allow_pending_commit=True
    )
    pending = pending_audit_transition(audits)
    if pending is not None:
        pending_status = pending.get("status", {})
        if (
            pending.get("from_state") != args.expected
            or pending.get("to_state") != args.next_state
            or pending_status.get("exit_code") != args.exit_code
            or pending.get("output_hashes") != hashes
        ):
            raise ValueError(
                "pending lifecycle transition must be repaired before advancing"
            )
    return transition_run_status(
        paths, args.expected, args.next_state,
        exit_code=args.exit_code, output_hashes=hashes,
    )


def _summary_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "metrics" / "selected_validation.json"
    if not path.exists():
        return {}
    payload = validate_metrics_document(
        _read_json(path, "selected validation metrics")
    )
    values = payload["summary_metrics"]
    return {str(name): float(value) for name, value in values.items()}


def finalize_phase(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    phase_path = _repo_relative(args.phase, "phase")
    registry_path = _repo_relative(args.registry, "registry")
    summary_path = _repo_relative(args.summary, "summary")
    artifact_root = _canonical_root(args.artifact_root, "artifact-root")
    phase = _load_committed_phase(repo, phase_path)
    try:
        committed_registry = _git(
            repo, "show", f"HEAD:{registry_path.as_posix()}"
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("finalization requires a committed HEAD registry") from exc
    committed_records = parse_registry_text(committed_registry)
    working_records = load_registry(repo / registry_path)
    terminal_records: list[dict[str, object]] = []
    summary_runs: list[dict[str, object]] = []
    states: Counter[str] = Counter()
    for spec in phase["runs"]:  # type: ignore[union-attr]
        reservation = _reservation_for_spec(spec, committed_records, artifact_root)
        run_dir = Path(str(reservation["artifact_path"]))
        manifest, status = _load_run(run_dir)
        if manifest_sha256(manifest) != reservation["manifest_sha256"]:
            raise ValueError("terminal manifest hash differs from reservation")
        state = status.get("state")
        if state not in TERMINAL_STATES:
            raise ValueError(f"phase run {manifest['run_id']} is not terminal")
        output_hashes = status.get("output_hashes")
        if not isinstance(output_hashes, dict):
            raise ValueError("terminal status output_hashes must be an object")
        actual_output_hashes = _all_output_hashes(run_dir)
        if output_hashes != actual_output_hashes:
            raise ValueError("terminal output hash inventory mismatch")
        metrics = _summary_metrics(run_dir)
        selected_path = "metrics/selected_validation.json"
        selected_validation = (
            {"path": selected_path, "sha256": output_hashes[selected_path]}
            if selected_path in output_hashes else None
        )
        terminal_records.append(
            _registry_record(
                manifest, str(state), summary_metrics=metrics,
                selected_validation=selected_validation,
            )
        )
        states[str(state)] += 1
        summary_runs.append(
            {
                "run_id": manifest["run_id"], "state": state,
                "manifest_sha256": manifest_sha256(manifest),
                "output_hashes": output_hashes, "summary_metrics": metrics,
            }
        )
    summary: dict[str, object] = {
        "phase": phase["phase"], "study": phase["study"],
        "phase_path": phase_path.as_posix(),
        "terminal_counts": dict(sorted(states.items())), "runs": summary_runs,
    }
    committed_terminal_records = [
        record for record in terminal_records if record in committed_records
    ]
    if committed_terminal_records and len(committed_terminal_records) != len(terminal_records):
        raise ValueError(
            "finalization rejects a committed partial terminal batch for the phase"
        )
    try:
        committed_summary_text = _git(
            repo, "show", f"HEAD:{summary_path.as_posix()}"
        )
    except subprocess.CalledProcessError:
        committed_summary = None
    else:
        try:
            committed_summary = strict_json_loads(committed_summary_text)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError("committed phase summary is invalid") from exc
    all_terminal_committed = (
        len(committed_terminal_records) == len(terminal_records)
    )
    if all_terminal_committed and committed_summary != summary:
        raise ValueError(
            "terminal registry batch requires its exact committed phase summary"
        )
    if not all_terminal_committed and committed_summary is not None:
        raise ValueError(
            "phase summary cannot be committed before its terminal registry batch"
        )
    terminal_additions = [
        record for record in terminal_records if record not in committed_records
    ]
    if working_records == committed_records:
        registry_already_appended = not terminal_additions
    elif terminal_additions and working_records == committed_records + terminal_additions:
        registry_already_appended = True
    else:
        raise ValueError(
            "finalization requires the committed HEAD registry or its exact terminal append"
        )
    changed = set(_git(repo, "diff", "--name-only").splitlines())
    allowed_changes = {registry_path.as_posix(), summary_path.as_posix()}
    if changed - allowed_changes or subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo
    ).returncode:
        raise ValueError("finalization requires clean tracked source")
    if not registry_already_appended:
        append_registry_records(repo / registry_path, terminal_additions)
    if (repo / summary_path).exists():
        existing_summary = _read_json(repo / summary_path, "phase summary")
        if existing_summary != summary:
            raise ValueError("existing phase summary differs from exact recovery")
    else:
        _atomic_json(repo / summary_path, summary)
    return summary


def _identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve = commands.add_parser("reserve-phase", allow_abbrev=False)
    reserve.add_argument("--repo-root", required=True)
    reserve.add_argument("--phase", required=True)
    reserve.add_argument("--registry", required=True)
    reserve.add_argument("--artifact-root", required=True)
    reserve.set_defaults(handler=reserve_phase)
    verify = commands.add_parser("verify-run", allow_abbrev=False)
    _identity_arguments(verify)
    verify.set_defaults(handler=verify_run)
    start = commands.add_parser("start-run", allow_abbrev=False)
    _identity_arguments(start)
    start.set_defaults(handler=start_run)
    transition = commands.add_parser("transition-run", allow_abbrev=False)
    _identity_arguments(transition)
    transition.add_argument("--expected", required=True)
    transition.add_argument("--next", dest="next_state", required=True)
    transition.add_argument("--exit-code", type=int)
    transition.add_argument("--output", action="append")
    transition.set_defaults(handler=transition_run)
    finalize = commands.add_parser("finalize-phase", allow_abbrev=False)
    finalize.add_argument("--repo-root", required=True)
    finalize.add_argument("--phase", required=True)
    finalize.add_argument("--registry", required=True)
    finalize.add_argument("--artifact-root", required=True)
    finalize.add_argument("--summary", required=True)
    finalize.set_defaults(handler=finalize_phase)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, ValueError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(strict_json_dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
