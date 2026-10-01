#!/usr/bin/env python3
"""Plan and execute committed pilot batches using the existing Scheme A manager."""

import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from experiment_artifacts import artifact_json_bytes, load_validated_run_lifecycle, manifest_sha256, strict_json_loads
from scripts import manage_experiment_run as manager
from semantic_retrieval.runtime.pilot_protocol import CONTROL, PILOT, _marker, build_phase


def _commit(repo, paths, message):
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
        raise ValueError("orchestration cannot include unrelated staged changes")
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)


def reserve_and_commit(repo_root, phase_path, artifact_root):
    repo = Path(repo_root).resolve()
    phase = manager._load_committed_phase(repo, Path(phase_path))
    if phase["study"] != PILOT:
        raise ValueError("orchestrator only owns the new pilot namespace")
    result = manager.reserve_phase(SimpleNamespace(repo_root=repo, phase=phase_path,
        registry=f"{CONTROL}/registry.jsonl", artifact_root=Path(artifact_root).resolve()))
    if subprocess.run(["git", "diff", "--quiet", "--", f"{CONTROL}/registry.jsonl"], cwd=repo).returncode:
        _commit(repo, [f"{CONTROL}/registry.jsonl"], f"reserve: commit complete {phase['phase']} pilot batch")
    for run in result["runs"]:
        manager.verify_run(SimpleNamespace(repo_root=repo, registry=f"{CONTROL}/registry.jsonl",
            run_dir=run["artifact_path"], run_id=run["run_id"], manifest_sha256=run["manifest_sha256"]))
    return result


def runtime_payload(repo_root, manifest):
    repo = Path(repo_root).resolve()
    evaluation = manifest["config"]["evaluation"]
    return dict(study_path=str(repo / CONTROL / "study.yaml"), registry_path=str(repo / CONTROL / "registry.jsonl"),
        run_dir=manifest["artifact_path"], run_id=manifest["run_id"], manifest_sha256=manifest_sha256(manifest),
        source_sha=manifest["git_sha"], eval_split=evaluation["eval_split"])


def system_template_from_manifest(manifest):
    config = manifest["config"]
    evaluation = config["evaluation"]
    train = dict(**config["optimizer"], **config["scheduler"], loss_weights=config["loss"])
    train["pilot"] = evaluation.get("pilot")
    return dict(runtime={}, model=config["model"], train=train, policy=evaluation["policy"],
                data=evaluation["data"], deployment=evaluation.get("deployment"), resume=evaluation.get("resume"))


def provision_run_config(repo_root, run_dir, template):
    """Bind a run-local SystemConfig without a circular raw-config/manifest hash."""
    import tempfile
    from semantic_retrieval.config import effective_manifest_config, load_system_config

    root = Path(run_dir).resolve()
    manifest, status = manager._load_run(root)
    if status["state"] != "reserved":
        raise ValueError("configuration provisioning requires an untouched reserved run")
    payload = dict(template, runtime=runtime_payload(repo_root, manifest))
    with tempfile.TemporaryDirectory(prefix="pilot-config-") as directory:
        check = Path(directory) / "config.json"
        check.write_bytes(artifact_json_bytes(payload))
        try:
            config = load_system_config(check)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError("configuration template is not a valid bound SystemConfig") from exc
        if effective_manifest_config(config, eval_split=config.runtime.eval_split) != manifest["config"]:
            raise ValueError("configuration template differs from the reserved effective config")
    target = root / "provenance/system-config.json"
    raw = artifact_json_bytes(payload)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != raw:
            raise ValueError("existing run-local configuration is immutable and differs")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    return target


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # A launcher may exit before its children; close only this worker's group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _transition(repo, run_dir, expected, next_state, exit_code=None):
    manifest, _ = manager._load_run(run_dir)
    return manager.transition_run(SimpleNamespace(repo_root=repo, registry=f"{CONTROL}/registry.jsonl",
        run_dir=run_dir, run_id=manifest["run_id"], manifest_sha256=manifest_sha256(manifest),
        expected=expected, next_state=next_state, exit_code=exit_code, output=[]))


def _close_worker(repo, run_dir, code, *, reason=None):
    manifest, status = manager._load_run(run_dir)
    if status["state"] in {"completed", "failed", "cancelled"}:
        raise ValueError("worker wrote a terminal state; controller cannot rewrite immutable history")
    observed_exit_code = code
    if code == 0 and status["state"] != "validating":
        reason, code = "worker exited without a validating completion", 1
    report = dict(observed_exit_code=observed_exit_code, terminal_exit_code=code, reason=reason,
                  closed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (Path(run_dir) / "provenance/controller-result.json").open("xb") as stream:
        stream.write(artifact_json_bytes(report))
    if status["state"] == "reserved":
        _transition(repo, run_dir, "reserved", "cancelled", code or 1)
    else:
        if status["state"] == "running":
            _transition(repo, run_dir, "running", "validating")
        _transition(repo, run_dir, "validating", "completed" if code == 0 else "failed", code)
    return code


def _finalize_and_commit(repo, phase_path, artifact_root):
    phase = manager._load_committed_phase(repo, Path(phase_path))
    summary_path = f"{CONTROL}/summaries/{phase['phase']}.json"
    summary = manager.finalize_phase(SimpleNamespace(repo_root=repo, phase=phase_path,
        registry=f"{CONTROL}/registry.jsonl", artifact_root=artifact_root, summary=summary_path))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", f"{CONTROL}/registry.jsonl", summary_path],
                           cwd=repo, check=True, capture_output=True, text=True).stdout
    if dirty:
        _commit(repo, [f"{CONTROL}/registry.jsonl", summary_path], f"archive: finalize complete {phase['phase']} pilot batch")
    return summary


def execute_phase(repo_root, phase_path, artifact_root, *, config_templates=None):
    """Execute each exact reservation once; failure cancels untouched siblings."""
    repo, artifact_root = Path(repo_root).resolve(), Path(artifact_root).resolve()
    phase = manager._load_committed_phase(repo, Path(phase_path))
    if phase["study"] != PILOT:
        raise ValueError("orchestrator only owns the new pilot namespace")
    records = manager.load_registry(repo / CONTROL / "registry.jsonl")
    identities = [manager._reservation_for_spec(spec, records, artifact_root) for spec in phase["runs"]]
    verified = []
    # Validate the entire batch before starting its first member.
    for record in identities:
        manifest, status = manager._load_run(Path(record["artifact_path"]))
        if status["state"] != "reserved":
            raise ValueError("pilot workers must be reserved; terminal/retry execution is forbidden")
        identity = manager.verify_run(SimpleNamespace(repo_root=repo, registry=f"{CONTROL}/registry.jsonl",
            run_dir=record["artifact_path"], run_id=record["run_id"], manifest_sha256=record["manifest_sha256"]))
        if identity["state"] != "reserved":
            raise ValueError("pilot workers must be reserved; terminal/retry execution is forbidden")
        from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized
        assert_pilot_authorized(repo_root=repo, registry_path=repo / CONTROL / "registry.jsonl", manifest=manifest, operation="start")
        if any("provenance/system-config.json" in arg for arg in identity["command_argv"]):
            key = f"{manifest['variant']}:{manifest['seed']}"
            template = (config_templates or {}).get(key) or system_template_from_manifest(manifest)
            provision_run_config(repo, identity["run_dir"], template)
        verified.append((identity, manifest))
    failed = False
    for identity, manifest in verified:
        run = Path(identity["run_dir"])
        if failed:
            _close_worker(repo, run, 130, reason="prior phase worker failed; no automatic retry")
            continue
        process = None
        started = time.monotonic()
        marker = _marker(manifest)
        max_seconds = marker["gpu_hours"] * 3600 / 4 if marker["gpu_hours"] else None
        code, reason = 1, None
        try:
            with (run / "logs/worker.log").open("xb") as log:
                process = subprocess.Popen(identity["command_argv"], cwd=repo, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with (run / "provenance/controller-launch.json").open("xb") as stream:
                    stream.write(artifact_json_bytes(dict(pid=process.pid, command_sha256=manifest["command_sha256"],
                        gpu_hours_ceiling=marker["gpu_hours"], timeout_seconds=max_seconds)))
                try:
                    code = process.wait(timeout=max_seconds)
                except subprocess.TimeoutExpired:
                    _stop_group(process)
                    code, reason = 124, "registered GPU-hour timeout exceeded"
                if code < 0:
                    code = 128 - code
                _stop_group(process)
        except (OSError, KeyboardInterrupt) as exc:
            if process is not None:
                _stop_group(process)
            code, reason = 130 if isinstance(exc, KeyboardInterrupt) else 1, str(exc)
        with (run / "provenance/controller-cost.json").open("xb") as stream:
            elapsed = time.monotonic() - started
            stream.write(artifact_json_bytes(dict(elapsed_seconds=elapsed, allocated_gpu_hours=elapsed * manifest["environment"]["gpu_count"] / 3600)))
        failed = _close_worker(repo, run, code, reason=reason) != 0
    summary = _finalize_and_commit(repo, phase_path, artifact_root)
    if failed:
        raise RuntimeError("pilot phase stopped; failure evidence committed and manual successor review is required")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path)
    commands = parser.add_subparsers(dest="operation", required=True)
    plan = commands.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--phase-id", required=True)
    plan.add_argument("--phase-path", required=True)
    plan.add_argument("--templates", type=Path, required=True)
    plan.add_argument("--evidence", type=Path, required=True)
    plan.add_argument("--scheduled-timestamp-utc")
    for operation in ("reserve", "run", "finalize"):
        item = commands.add_parser(operation, allow_abbrev=False)
        item.add_argument("--phase", required=True)
        if operation == "run":
            item.add_argument("--config-templates", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    artifact_root = (args.artifact_root or repo / "artifacts").resolve()
    if args.operation == "plan":
        manager._require_clean(repo)
        timestamp = args.scheduled_timestamp_utc or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        phase = build_phase(args.phase_id, phase_path=args.phase_path, scheduled_timestamp_utc=timestamp,
            run_templates=strict_json_loads(args.templates.read_text()), evidence=strict_json_loads(args.evidence.read_text()))
        path = repo / args.phase_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(artifact_json_bytes(phase))
        _commit(repo, [args.phase_path], f"plan: freeze {phase['phase']} pilot batch")
        result = phase
    elif args.operation == "reserve":
        result = reserve_and_commit(repo, args.phase, artifact_root)
    elif args.operation == "run":
        templates = strict_json_loads(args.config_templates.read_text()) if args.config_templates else None
        result = execute_phase(repo, args.phase, artifact_root, config_templates=templates)
    else:
        result = _finalize_and_commit(repo, args.phase, artifact_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
