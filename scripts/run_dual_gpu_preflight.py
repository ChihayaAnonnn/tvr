#!/usr/bin/env python3
"""Once-only controller for approved, committed dual-GPU software phases."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import manage_experiment_run as manager
from semantic_retrieval.runtime.dual_gpu_preflight import PROTOCOL
from semantic_retrieval.runtime.pilot_candidates import write_json


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def launch_once(argv, *, cwd, run_dir, command_sha256, timeout):
    root = Path(run_dir)
    (root / "logs").mkdir(exist_ok=True)
    (root / "provenance").mkdir(exist_ok=True)
    # Record stable manifest identity BEFORE spawn; never look up nonexistent verifier fields afterward.
    write_json(root / "provenance/controller-plan.json", dict(command_sha256=command_sha256, timeout_seconds=timeout))
    process = None
    code = 1
    reason = None
    started = time.monotonic()
    try:
        with (root / "logs/worker.log").open("xb") as log:
            process = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root / "provenance/controller-launch.json", dict(pid=process.pid, command_sha256=command_sha256))
            code = process.wait(timeout=timeout)
            if code < 0:
                code = 128 - code
    except subprocess.TimeoutExpired:
        code, reason = 124, "registered software budget exceeded"
    except (Exception, KeyboardInterrupt) as exc:
        code, reason = 1, str(exc)
    finally:
        if process is not None:
            _stop(process)
    result = dict(
        terminal_exit_code=code,
        observed_launcher_exit_code=process.returncode if process else None,
        reason=reason,
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(root / "provenance/controller-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    control = f"experiments/{PROTOCOL}"
    registry = f"{control}/registry.jsonl"
    phase = manager._load_committed_phase(repo, Path(args.phase))
    if phase["study"] != PROTOCOL or len(phase["runs"]) != 1:
        raise ValueError("controller requires one approved software run")
    artifact_root = manager._git_common_root(repo) / "artifacts"
    records = manager.load_registry(repo / registry)
    record = manager._reservation_for_spec(phase["runs"][0], records, artifact_root)
    identity = SimpleNamespace(
        repo_root=repo,
        registry=registry,
        run_dir=record["artifact_path"],
        run_id=record["run_id"],
        manifest_sha256=record["manifest_sha256"],
    )
    root = Path(record["artifact_path"])
    manifest, status = manager._load_run(root)
    if status["state"] != "reserved":
        raise ValueError("only a new reserved run may launch; retries of terminal runs are forbidden")
    verified = manager.verify_run(identity)

    def interrupted(signum, frame):
        raise InterruptedError(f"controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    result = launch_once(
        verified["command_argv"], cwd=repo, run_dir=root, command_sha256=manifest["command_sha256"], timeout=3600
    )
    _, status = manager._load_run(root)
    code = result["terminal_exit_code"]
    if code == 0 and status["state"] != "validating":
        code = 1

    def transition(expected, next_state, exit_code=None):
        return manager.transition_run(
            SimpleNamespace(**vars(identity), expected=expected, next_state=next_state, exit_code=exit_code, output=[])
        )

    if status["state"] == "reserved":
        transition("reserved", "cancelled", code or 1)
    else:
        if status["state"] == "running":
            transition("running", "validating")
        transition("validating", "completed" if code == 0 else "failed", code)
    summary_path = f"{control}/summaries/{phase['phase']}.json"
    summary = manager.finalize_phase(
        SimpleNamespace(
            repo_root=repo, phase=args.phase, registry=registry, artifact_root=artifact_root, summary=summary_path
        )
    )
    subprocess.run(["git", "add", registry, summary_path], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", f"archive: finalize {phase['phase']} software phase"], cwd=repo, check=True)
    print(json.dumps(dict(terminal_counts=summary["terminal_counts"], exit_code=code)))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
