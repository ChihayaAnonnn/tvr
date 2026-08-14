import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


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
    return {
        "phase": "stage0-smoke",
        "study": "contextual-risk-v2",
        "runs": [
            {
                "dataset": "msrvtt",
                "stage": "baseline",
                "variant": "smoke",
                "seed": 0,
                "timestamp_utc": "20260814T020304Z",
                "environment": {
                    "python": "3.10",
                    "pytorch": "2.0",
                    "cuda": "12.0",
                    "gpu_model": "test-gpu",
                    "gpu_count": 1,
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
                "command": [
                    "env", "ARTIFACT_RUN_ID={run_id}", "bash",
                    "run_train_msrvtt_contextual_risk_bg.sh",
                ],
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
        f"env ARTIFACT_RUN_ID={run['run_id']} bash "
        "run_train_msrvtt_contextual_risk_bg.sh\n"
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
    _run(repo, "transition-run", "--run-dir", run_dir,
         "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", "--run-dir", run_dir,
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
    _run(repo, "transition-run", "--run-dir", run_dir,
         "--expected", "running", "--next", "validating")
    _run(repo, "transition-run", "--run-dir", run_dir,
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


def test_rank_zero_artifact_lifecycle_publishes_metrics_and_one_terminal_status(campaign_repo):
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
    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    relative = metric_path.relative_to(context.paths.root).as_posix()
    assert status["output_hashes"][relative] == hashlib.sha256(metric_path.read_bytes()).hexdigest()
    before = context.paths.status.read_bytes()
    main_task_retrieval._complete_artifact_run(context)
    assert context.paths.status.read_bytes() == before
    commits = [
        json.loads(line) for line in
        (context.paths.provenance / "audit.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "transition_commit"
        and json.loads(line).get("to_state") == "completed"
    ]
    assert len(commits) == 1


def test_artifact_failure_and_signal_evidence_becomes_terminal(campaign_repo):
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
    assert status["state"] == "cancelled"
    assert status["exit_code"] == 143
    assert "logs/train.log" in status["output_hashes"]
    assert "command.sh" in status["output_hashes"]


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
