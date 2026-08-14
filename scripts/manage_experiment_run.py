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
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from experiment_artifacts import (
    append_registry_records,
    build_run_id,
    load_registry,
    make_run_paths,
    manifest_sha256,
    reserve_run_directory,
    transition_run_status,
    validate_artifact_json_payload,
    validate_manifest,
    verify_committed_reservation,
)


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
RUN_SPEC_FIELDS = frozenset(
    {
        "dataset", "stage", "variant", "seed", "timestamp_utc", "environment",
        "input_checkpoints", "dataset_splits", "config", "parent_run_ids",
        "selection_lineage", "validation_authorized", "test_authorized",
        "expected_scalar_outputs", "array_references",
    }
)


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
        payload = json.loads(committed)
    except json.JSONDecodeError as exc:
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
        if (
            not isinstance(spec, dict)
            or not RUN_SPEC_FIELDS.issubset(spec)
            or set(spec) - RUN_SPEC_FIELDS != {"command"}
        ):
            raise ValueError(f"phase runs[{index}] has an invalid manifest template")
        command = spec.get("command")
        if (
            not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item or "\0" in item or "\n" in item for item in command)
        ):
            raise ValueError(f"phase runs[{index}].command must be a non-empty argv list")
    return payload


def _build_manifest(
    spec: Mapping[str, object], *, study: str, git_sha: str,
    artifact_root: Path, registry: Path,
) -> dict[str, object]:
    run_id = build_run_id(
        str(spec["dataset"]), str(spec["stage"]), str(spec["variant"]),
        spec["seed"], str(spec["timestamp_utc"]), git_sha,  # type: ignore[arg-type]
    )
    paths = make_run_paths(artifact_root, study, run_id)
    generated = {
        key: value for key, value in spec.items()
        if key not in {"timestamp_utc", "command"}
    }
    return validate_manifest(
        {
            **generated,
            "run_id": run_id,
            "study": study,
            "artifact_path": str(paths.root),
            "registry_path": registry.as_posix(),
            "git_sha": git_sha,
            "tracked_clean": True,
            "source_patch_sha256": None,
        }
    )


def _registry_record(
    manifest: Mapping[str, object], state: str, *,
    summary_metrics: Mapping[str, object] | None = None,
    selected_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
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


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be an object")
    return payload


def _load_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    if not run_dir.is_absolute() or os.fspath(run_dir) != os.path.normpath(run_dir):
        raise ValueError("run-dir must be an absolute canonical path")
    manifest = validate_manifest(_read_json(run_dir / "manifest.json", "run manifest"))
    if Path(str(manifest["artifact_path"])) != run_dir:
        raise ValueError("run-dir differs from manifest artifact_path")
    status = _read_json(run_dir / "status.json", "run status")
    if status.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("run status manifest hash mismatch")
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
        f"seed{int(spec['seed']):03d}-{spec['timestamp_utc']}-"
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


def _all_output_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory_name in (
        "logs", "checkpoints", "metrics", "arrays", "plots", "provenance"
    ):
        directory = run_dir / directory_name
        for path in sorted(directory.rglob("*")):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"unsafe artifact output: {path}")
            if stat.S_ISREG(info.st_mode):
                if directory_name == "provenance" and (
                    path.name == "audit.jsonl" or path.name.endswith(".lock")
                    or path.name.startswith("controller-pid.")
                ):
                    continue
                relative = path.relative_to(run_dir).as_posix()
                hashes[relative] = _safe_file_sha256(run_dir, relative)
    command = run_dir / "command.sh"
    if command.exists():
        hashes["command.sh"] = _safe_file_sha256(run_dir, "command.sh")
    return hashes


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    validate_artifact_json_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
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
    _require_clean(repo)
    phase = _load_committed_phase(repo, phase_path)
    git_sha = _git(repo, "rev-parse", "HEAD").strip()
    manifests = [
        _build_manifest(
            spec, study=str(phase["study"]), git_sha=git_sha,
            artifact_root=artifact_root, registry=registry,
        )
        for spec in phase["runs"]  # type: ignore[union-attr]
    ]
    run_ids = [str(manifest["run_id"]) for manifest in manifests]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("phase contains duplicate run identities")
    existing = load_registry(repo / registry)
    for spec in phase["runs"]:  # type: ignore[union-attr]
        prefix = (
            f"{spec['dataset']}-{spec['stage']}-{spec['variant']}-"
            f"seed{int(spec['seed']):03d}-{spec['timestamp_utc']}-"
        )
        if any(
            record["event"] == "reserved"
            and str(record["run_id"]).startswith(prefix)
            for record in existing
        ):
            raise ValueError(
                "phase contains an already registered identity; silent retry rejected"
            )
    for manifest in manifests:
        paths = make_run_paths(
            artifact_root, str(phase["study"]), str(manifest["run_id"])
        )
        if paths.root.exists():
            raise FileExistsError(f"run directory already exists: {paths.root}")
    for manifest, spec in zip(manifests, phase["runs"]):  # type: ignore[arg-type]
        paths = make_run_paths(artifact_root, str(phase["study"]), str(manifest["run_id"]))
        reserve_run_directory(paths, manifest)
        replacements = {
            "{run_id}": str(manifest["run_id"]),
            "{artifact_run_dir}": str(paths.root),
            "{manifest_sha256}": manifest_sha256(manifest),
        }
        command_argv = []
        for item in spec["command"]:
            for marker, value in replacements.items():
                item = item.replace(marker, value)
            command_argv.append(item)
        command = shlex.join(command_argv) + "\n"
        descriptor = os.open(
            paths.command,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o700,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(command)
            handle.flush()
            os.fsync(handle.fileno())
    append_registry_records(
        repo / registry, [_registry_record(manifest, "reserved") for manifest in manifests]
    )
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


def verify_run(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    registry = _repo_relative(args.registry, "registry")
    manifest, status = _load_run(Path(args.run_dir))
    _assert_identity(manifest, run_id=args.run_id, digest=args.manifest_sha256)
    verify_committed_reservation(repo, registry, manifest)
    return {
        "run_id": manifest["run_id"],
        "run_dir": manifest["artifact_path"],
        "manifest_sha256": manifest_sha256(manifest),
        "state": status["state"],
    }


def start_run(args: argparse.Namespace) -> dict[str, object]:
    result = verify_run(args)
    run_dir = Path(args.run_dir)
    status = transition_run_status(
        make_run_paths_from_dir(run_dir), "reserved", "running"
    )
    return {**result, "state": status["state"]}


def make_run_paths_from_dir(run_dir: Path):
    manifest, _ = _load_run(run_dir)
    artifact_root = run_dir.parents[3]
    return make_run_paths(artifact_root, str(manifest["study"]), str(manifest["run_id"]))


def transition_run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    paths = make_run_paths_from_dir(run_dir)
    hashes = {
        relative: _safe_file_sha256(run_dir, relative)
        for relative in (args.output or [])
    }
    if args.next_state in TERMINAL_STATES:
        complete_inventory = _all_output_hashes(run_dir)
        for relative, digest in hashes.items():
            if complete_inventory.get(relative) != digest:
                raise ValueError(f"terminal output changed while hashing: {relative}")
        hashes = complete_inventory
    return transition_run_status(
        paths, args.expected, args.next_state,
        exit_code=args.exit_code, output_hashes=hashes,
    )


def _summary_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "metrics" / "selected_validation.json"
    if not path.exists():
        return {}
    payload = _read_json(path, "selected validation metrics")
    values = payload.get("summary_metrics", {})
    if not isinstance(values, dict) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values.values()
    ):
        raise ValueError("selected validation summary_metrics must contain scalars")
    return {str(name): float(value) for name, value in values.items()}


def finalize_phase(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    phase_path = _repo_relative(args.phase, "phase")
    registry_path = _repo_relative(args.registry, "registry")
    summary_path = _repo_relative(args.summary, "summary")
    if (repo / summary_path).exists():
        raise FileExistsError(f"phase summary already exists: {summary_path}")
    artifact_root = _canonical_root(args.artifact_root, "artifact-root")
    phase = _load_committed_phase(repo, phase_path)
    records = load_registry(repo / registry_path)
    terminal_records: list[dict[str, object]] = []
    summary_runs: list[dict[str, object]] = []
    states: Counter[str] = Counter()
    for spec in phase["runs"]:  # type: ignore[union-attr]
        reservation = _reservation_for_spec(spec, records, artifact_root)
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
    append_registry_records(repo / registry_path, terminal_records)
    summary: dict[str, object] = {
        "phase": phase["phase"], "study": phase["study"],
        "phase_path": phase_path.as_posix(),
        "terminal_counts": dict(sorted(states.items())), "runs": summary_runs,
    }
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
    transition.add_argument("--run-dir", required=True)
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
