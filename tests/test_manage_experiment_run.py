import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "manage_experiment_run.py"
REGISTRY = Path("experiments/contextual-risk-v2/registry.jsonl")


def _run(repo, *arguments, check=True):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


def _phase_payload():
    gpu_count = torch.cuda.device_count()
    return {
        "phase": "stage0-smoke",
        "study": "fixture-study",
        "runs": [
            {
                "dataset": "msrvtt",
                "stage": "baseline",
                "variant": "smoke",
                "seed": 0,
                "timestamp_utc": "20260814T020304Z",
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda": str(torch.version.cuda or "none"),
                    "gpu_model": torch.cuda.get_device_name(0) if gpu_count else "none",
                    "gpu_count": gpu_count,
                    "precision": "fp32",
                    "distributed_topology": "single-process",
                },
                "input_checkpoints": [],
                "dataset_splits": [],
                "config": {
                    "model": {},
                    "loss": {},
                    "optimizer": {},
                    "scheduler": {},
                    "evaluation": {},
                },
                "parent_run_ids": [],
                "selection_lineage": [],
                "validation_authorized": True,
                "test_authorized": False,
                "expected_scalar_outputs": [],
                "array_references": [],
                "command": ["python", "fixture_worker.py", "--run-id", "{run_id}"],
            }
        ],
    }


@pytest.fixture
def campaign_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / REGISTRY.parent).mkdir(parents=True)
    (repo / REGISTRY).write_text("", encoding="utf-8")
    phase = Path("experiments/contextual-risk-v2/phases/stage0-smoke.json")
    (repo / phase.parent).mkdir(parents=True)
    (repo / phase).write_text(json.dumps(_phase_payload()), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "phase"], cwd=repo, check=True)
    return repo, phase, tmp_path / "artifacts"


def _reserve(campaign_repo):
    repo, phase, artifact_root = campaign_repo
    result = _run(
        repo,
        "reserve-phase",
        "--repo-root", repo,
        "--phase", phase,
        "--registry", REGISTRY,
        "--artifact-root", artifact_root,
    )
    payload = json.loads(result.stdout)
    return repo, phase, artifact_root, payload


def test_reserve_phase_builds_runs_without_launching_and_verify_requires_commit(campaign_repo):
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    assert payload["phase"] == "stage0-smoke"
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    assert json.loads((run_dir / "status.json").read_text())["state"] == "reserved"
    assert (run_dir / "command.sh").read_text() == (
        f"python fixture_worker.py --run-id {run['run_id']}\n"
    )
    assert not (run_dir / "logs" / "train.log").exists()

    failed = _run(
        repo, "verify-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"], check=False,
    )
    assert failed.returncode != 0
    assert "committed" in failed.stderr or "clean" in failed.stderr

    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    verified = _run(
        repo, "verify-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"],
    )
    assert json.loads(verified.stdout)["state"] == "reserved"


def test_lifecycle_cli_finalizes_only_terminal_runs_and_verifies_output_hashes(campaign_repo):
    repo, phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)

    _run(
        repo, "start-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"],
    )
    premature = _run(
        repo, "finalize-phase", "--repo-root", repo, "--phase", phase,
        "--registry", REGISTRY, "--artifact-root", payload["artifact_root"],
        "--summary", Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json"),
        check=False,
    )
    assert premature.returncode != 0
    assert "terminal" in premature.stderr

    metric = run_dir / "metrics" / "selected_validation.json"
    metric.write_text(json.dumps({"summary_metrics": {"t2v_r1": 52.6}}), encoding="utf-8")
    digest = hashlib.sha256(metric.read_bytes()).hexdigest()
    identity = ["--repo-root", repo, "--registry", REGISTRY, "--run-dir", run_dir,
                "--run-id", run["run_id"], "--manifest-sha256", run["manifest_sha256"]]
    _run(repo, "transition-run", *identity,
         "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", *identity,
         "--expected", "validating", "--next", "completed", "--exit-code", "0",
         "--output", "metrics/selected_validation.json")

    summary_path = Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json")
    result = _run(
        repo, "finalize-phase", "--repo-root", repo, "--phase", phase,
        "--registry", REGISTRY, "--artifact-root", payload["artifact_root"],
        "--summary", summary_path,
    )
    summary = json.loads(result.stdout)
    assert summary["terminal_counts"] == {"completed": 1}
    assert summary["runs"][0]["output_hashes"][
        "metrics/selected_validation.json"
    ] == digest
    assert "command.sh" in summary["runs"][0]["output_hashes"]
    assert json.loads((repo / summary_path).read_text()) == summary
    terminal = json.loads((repo / REGISTRY).read_text().splitlines()[-1])
    assert terminal["event"] == terminal["state"] == "completed"
    assert terminal["summary_metrics"] == {"t2v_r1": 52.6}
    assert terminal["selected_validation"] == {
        "path": "metrics/selected_validation.json", "sha256": digest
    }


def test_finalize_rejects_output_tampering(campaign_repo):
    repo, phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    _run(repo, "start-run", "--repo-root", repo, "--registry", REGISTRY,
         "--run-dir", run_dir, "--run-id", run["run_id"],
         "--manifest-sha256", run["manifest_sha256"])
    evidence = run_dir / "metrics" / "selected_validation.json"
    evidence.write_text("{}", encoding="utf-8")
    identity = ["--repo-root", repo, "--registry", REGISTRY, "--run-dir", run_dir,
                "--run-id", run["run_id"], "--manifest-sha256", run["manifest_sha256"]]
    _run(repo, "transition-run", *identity,
         "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", *identity,
         "--expected", "validating", "--next", "completed", "--exit-code", "0",
         "--output", "metrics/selected_validation.json")
    evidence.write_text('{"tampered": true}', encoding="utf-8")

    failed = _run(
        repo, "finalize-phase", "--repo-root", repo, "--phase", phase,
        "--registry", REGISTRY, "--artifact-root", payload["artifact_root"],
        "--summary", Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json"),
        check=False,
    )
    assert failed.returncode != 0
    assert "hash" in failed.stderr.lower()


def test_retrieval_parser_requires_complete_artifact_identity(monkeypatch, tmp_path):
    import main_task_retrieval

    base = ["prog", "--output_dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base + ["--artifact_run_dir", str(tmp_path)])
    with pytest.raises(ValueError, match="together"):
        main_task_retrieval.get_args()


def test_python_entrypoint_binds_committed_manifest_paths_and_array_store(campaign_repo):
    import main_task_retrieval

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
    )

    context = main_task_retrieval._prepare_artifact_run(args, repo_root=repo)

    assert context.paths.root == Path(run["artifact_path"])
    assert args.artifact_array_store.run_root == Path(run["artifact_path"])
    assert args.artifact_metrics_dir == context.paths.metrics

    args.output_dir = str(context.paths.root)
    with pytest.raises(ValueError, match="checkpoints"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_rank_zero_artifact_worker_stops_at_validating_before_log_owner_exits(campaign_repo):
    import main_task_retrieval

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
    )
    context = main_task_retrieval._prepare_artifact_run(args, repo_root=repo)

    main_task_retrieval._start_artifact_run(context)
    metric_path = main_task_retrieval._publish_artifact_eval_metrics(
        args, {"t2v": {"R1": 52.6}}
    )
    main_task_retrieval._complete_artifact_run(context)

    assert metric_path == context.paths.metrics / "val_eval_000.json"
    status = json.loads(context.paths.status.read_text())
    assert status["state"] == "validating"
    assert status["exit_code"] is None
    assert status["output_hashes"] == {}
    before = context.paths.status.read_bytes()
    main_task_retrieval._complete_artifact_run(context)
    assert context.paths.status.read_bytes() == before
    commits = [
        json.loads(line) for line in
        (context.paths.provenance / "audit.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "transition_commit"
        and json.loads(line).get("to_state") == "validating"
    ]
    assert len(commits) == 1


def test_artifact_failure_defers_terminal_hashing_to_closed_log_owner(campaign_repo):
    import main_task_retrieval

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
    )
    context = main_task_retrieval._prepare_artifact_run(args, repo_root=repo)
    main_task_retrieval._start_artifact_run(context)
    log_path = context.paths.logs / "train.log"
    log_path.write_text("worker interrupted\n", encoding="utf-8")

    main_task_retrieval._fail_artifact_run(context, exit_code=143)

    status = json.loads(context.paths.status.read_text())
    assert status["state"] == "running"
    assert status["exit_code"] is None
    assert status["output_hashes"] == {}
    failure = json.loads(
        (context.paths.provenance / "worker_failure.json").read_text()
    )
    assert failure["exit_code"] == 143


def test_artifact_manifest_must_explicitly_authorize_test(campaign_repo):
    import main_task_retrieval

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=True, eval_split="val", local_rank=0,
    )

    with pytest.raises(ValueError, match="authorize test"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_transition_run_requires_exact_committed_identity(campaign_repo):
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])

    uncommitted = _run(
        repo, "transition-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"],
        "--expected", "reserved", "--next", "running", check=False,
    )
    assert uncommitted.returncode != 0
    assert json.loads((run_dir / "status.json").read_text())["state"] == "reserved"

    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    bypass = _run(
        repo, "transition-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"],
        "--expected", "reserved", "--next", "running", check=False,
    )
    assert bypass.returncode != 0
    assert "start-run" in bypass.stderr


def test_verify_run_rejects_command_tampering(campaign_repo):
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    (run_dir / "command.sh").write_text("python forged.py\n", encoding="utf-8")

    failed = _run(
        repo, "verify-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"], check=False,
    )
    assert failed.returncode != 0
    assert "command" in failed.stderr.lower()


def test_artifact_runtime_rejects_manifest_config_mismatch(campaign_repo):
    import main_task_retrieval

    repo, phase, _artifact_root = campaign_repo
    phase_payload = _phase_payload()
    phase_payload["runs"][0]["config"]["model"] = {"contextual_risk_topk": 8}
    (repo / phase).write_text(json.dumps(phase_payload), encoding="utf-8")
    subprocess.run(["git", "add", phase], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bind topk"], cwd=repo, check=True)
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=str(run_dir), artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(run_dir / "checkpoints"), run_final_test=False,
        eval_split="val", local_rank=0, contextual_risk_topk=16,
        resume_model=None,
    )

    with pytest.raises(ValueError, match="effective config"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_artifact_resume_is_explicitly_rejected_until_child_import_exists(campaign_repo):
    import main_task_retrieval

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
        resume_model="parent-opt.bin",
    )

    with pytest.raises(ValueError, match="child-run lineage"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_artifact_runtime_rejects_changed_hashed_input(campaign_repo):
    import main_task_retrieval

    repo, phase, _artifact_root = campaign_repo
    checkpoint = repo.parent / "base.bin"
    checkpoint.write_bytes(b"original")
    phase_payload = _phase_payload()
    phase_payload["runs"][0]["input_checkpoints"] = [{
        "path": str(checkpoint),
        "sha256": hashlib.sha256(b"original").hexdigest(),
    }]
    (repo / phase).write_text(json.dumps(phase_payload), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bind input"], cwd=repo, check=True)
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve"], cwd=repo, check=True)
    checkpoint.write_bytes(b"changed")
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
        clip_layer_norm_precision="fp32", resume_model=None,
    )

    with pytest.raises(ValueError, match="input.*SHA-256"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_registered_a3_worker_rejects_incomplete_manifest_config(campaign_repo):
    import main_task_retrieval

    repo, phase, _artifact_root = campaign_repo
    phase_payload = _phase_payload()
    phase_payload["study"] = "contextual-risk-v2"
    phase_payload["runs"][0]["command"] = [
        "env", "ARTIFACT_RUN_ID={run_id}", "bash",
        "run_train_msrvtt_contextual_risk_bg.sh",
    ]
    (repo / phase).write_text(json.dumps(phase_payload), encoding="utf-8")
    subprocess.run(["git", "add", phase], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "registered fixture"], cwd=repo, check=True)
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve"], cwd=repo, check=True)
    args = SimpleNamespace(
        artifact_run_dir=run["artifact_path"], artifact_run_id=run["run_id"],
        artifact_manifest_sha256=run["manifest_sha256"],
        output_dir=str(Path(run["artifact_path"]) / "checkpoints"),
        run_final_test=False, eval_split="val", local_rank=0,
        registered_contextual_risk_a3=True,
    )

    with pytest.raises(ValueError, match="config is incomplete"):
        main_task_retrieval._prepare_artifact_run(args, repo_root=repo)


def test_finalize_rejects_forged_terminal_without_committed_audit(campaign_repo):
    repo, phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    status = json.loads((run_dir / "status.json").read_text())
    status.update({
        "state": "completed", "exit_code": 0, "transition_id": "f" * 64,
        "output_hashes": {
            "command.sh": hashlib.sha256((run_dir / "command.sh").read_bytes()).hexdigest()
        },
    })
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

    failed = _run(
        repo, "finalize-phase", "--repo-root", repo, "--phase", phase,
        "--registry", REGISTRY, "--artifact-root", payload["artifact_root"],
        "--summary", Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json"),
        check=False,
    )
    assert failed.returncode != 0
    assert "audit" in failed.stderr.lower() or "transition" in failed.stderr.lower()


def test_verify_rejects_symlinked_status(campaign_repo):
    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    outside = run_dir.parent / "forged-status.json"
    outside.write_bytes((run_dir / "status.json").read_bytes())
    (run_dir / "status.json").unlink()
    (run_dir / "status.json").symlink_to(outside)

    failed = _run(
        repo, "verify-run", "--repo-root", repo, "--registry", REGISTRY,
        "--run-dir", run_dir, "--run-id", run["run_id"],
        "--manifest-sha256", run["manifest_sha256"], check=False,
    )
    assert failed.returncode != 0


def test_finalize_rejects_uncommitted_registry_mutation(campaign_repo):
    repo, phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True)
    _run(repo, "start-run", "--repo-root", repo, "--registry", REGISTRY,
         "--run-dir", run_dir, "--run-id", run["run_id"],
         "--manifest-sha256", run["manifest_sha256"])
    identity = ["--repo-root", repo, "--registry", REGISTRY, "--run-dir", run_dir,
                "--run-id", run["run_id"], "--manifest-sha256", run["manifest_sha256"]]
    _run(repo, "transition-run", *identity,
         "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", *identity,
         "--expected", "validating", "--next", "completed", "--exit-code", "0")
    forged = json.loads((repo / REGISTRY).read_text().splitlines()[0])
    forged.update({
        "run_id": "forged", "artifact_path": str(run_dir.parent / "forged")
    })
    with (repo / REGISTRY).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(forged) + "\n")

    failed = _run(
        repo, "finalize-phase", "--repo-root", repo, "--phase", phase,
        "--registry", REGISTRY, "--artifact-root", payload["artifact_root"],
        "--summary", Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json"),
        check=False,
    )
    assert failed.returncode != 0
    assert "head" in failed.stderr.lower() or "committed" in failed.stderr.lower()
    assert failed.returncode != 0


def test_reserve_recovers_exact_directory_crash_without_duplicate_registry(
    campaign_repo, monkeypatch
):
    import scripts.manage_experiment_run as manage

    repo, phase, artifact_root = campaign_repo
    args = SimpleNamespace(
        repo_root=repo, phase=phase, registry=REGISTRY,
        artifact_root=artifact_root,
    )
    real_append = manage.append_registry_records

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(manage, "append_registry_records", crash)
    with pytest.raises(RuntimeError, match="crash"):
        manage.reserve_phase(args)
    monkeypatch.setattr(manage, "append_registry_records", real_append)

    result = manage.reserve_phase(args)
    retried = manage.reserve_phase(args)

    assert len(result["runs"]) == 1
    assert retried == result
    assert len((repo / REGISTRY).read_text().splitlines()) == 1


def test_reserve_recovers_crash_between_manifest_and_command(campaign_repo, monkeypatch):
    import scripts.manage_experiment_run as manage

    repo, phase, artifact_root = campaign_repo
    args = SimpleNamespace(
        repo_root=repo, phase=phase, registry=REGISTRY,
        artifact_root=artifact_root,
    )
    real_render = manage._render_command
    calls = 0

    def crash_on_publication(*render_args, **render_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("command crash")
        return real_render(*render_args, **render_kwargs)

    monkeypatch.setattr(manage, "_render_command", crash_on_publication)
    with pytest.raises(RuntimeError, match="command crash"):
        manage.reserve_phase(args)
    monkeypatch.setattr(manage, "_render_command", real_render)

    result = manage.reserve_phase(args)

    run_dir = Path(result["runs"][0]["artifact_path"])
    assert (run_dir / "command.sh").is_file()
    assert len((repo / REGISTRY).read_text().splitlines()) == 1


def test_reserve_recovers_crash_after_manifest_before_status(campaign_repo, monkeypatch):
    import experiment_artifacts
    import scripts.manage_experiment_run as manage

    repo, phase, artifact_root = campaign_repo
    args = SimpleNamespace(
        repo_root=repo, phase=phase, registry=REGISTRY,
        artifact_root=artifact_root,
    )
    real_write = experiment_artifacts._atomic_write_json

    def fail_status(path, payload):
        if path.name == "status.json":
            raise OSError("status crash")
        return real_write(path, payload)

    monkeypatch.setattr(experiment_artifacts, "_atomic_write_json", fail_status)
    with pytest.raises(OSError, match="status crash"):
        manage.reserve_phase(args)
    monkeypatch.setattr(experiment_artifacts, "_atomic_write_json", real_write)

    result = manage.reserve_phase(args)

    assert len(result["runs"]) == 1
    assert len((repo / REGISTRY).read_text().splitlines()) == 1


def test_start_run_repairs_status_written_before_audit_commit(
    campaign_repo, monkeypatch
):
    import experiment_artifacts
    import scripts.manage_experiment_run as manage

    repo, _phase, _artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve"], cwd=repo, check=True)
    args = SimpleNamespace(
        repo_root=repo, registry=REGISTRY, run_dir=run["artifact_path"],
        run_id=run["run_id"], manifest_sha256=run["manifest_sha256"],
    )
    real_append = experiment_artifacts._append_audit_record

    def fail_commit(paths, record):
        if record.get("event") == "transition_commit":
            raise OSError("commit crash")
        return real_append(paths, record)

    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", fail_commit)
    with pytest.raises(OSError, match="commit crash"):
        manage.start_run(args)
    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", real_append)

    result = manage.start_run(args)

    assert result["state"] == "running"
    verified = manage.verify_run(args)
    assert verified["state"] == "running"


def test_finalize_recovers_exact_registry_append_crash_without_duplicate_event(
    campaign_repo, monkeypatch
):
    import scripts.manage_experiment_run as manage

    repo, phase, artifact_root, payload = _reserve(campaign_repo)
    run = payload["runs"][0]
    run_dir = Path(run["artifact_path"])
    subprocess.run(["git", "add", REGISTRY], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve"], cwd=repo, check=True)
    identity = ["--repo-root", repo, "--registry", REGISTRY, "--run-dir", run_dir,
                "--run-id", run["run_id"], "--manifest-sha256", run["manifest_sha256"]]
    _run(repo, "start-run", *identity)
    _run(repo, "transition-run", *identity, "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", *identity, "--expected", "validating", "--next", "completed", "--exit-code", "0")
    summary = Path("experiments/contextual-risk-v2/summaries/stage0-smoke.json")
    args = SimpleNamespace(
        repo_root=repo, phase=phase, registry=REGISTRY,
        artifact_root=artifact_root, summary=summary,
    )
    real_atomic = manage._atomic_json

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(manage, "_atomic_json", crash)
    with pytest.raises(RuntimeError, match="crash"):
        manage.finalize_phase(args)
    monkeypatch.setattr(manage, "_atomic_json", real_atomic)

    result = manage.finalize_phase(args)

    assert result["terminal_counts"] == {"completed": 1}
    assert len((repo / REGISTRY).read_text().splitlines()) == 2
