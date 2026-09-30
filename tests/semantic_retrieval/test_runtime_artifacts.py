import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import manage_experiment_run as manager
from semantic_retrieval.config import RuntimeConfig


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def reservation(tmp_path, request):
    repo = tmp_path / "repo"
    repo.mkdir()
    control = repo / "experiments/fixture-study"
    control.mkdir(parents=True)
    registry = control / "registry.jsonl"
    registry.touch()
    study = control / "study.yaml"
    study.write_text("study: fixture-study\n")
    phase = control / "phase.json"
    gpu_count = torch.cuda.device_count()
    run = dict(
        dataset="msrvtt",
        stage="baseline",
        variant="cpu",
        seed=0,
        timestamp_utc="20260814T020304Z",
        environment=dict(
            python=platform.python_version(),
            pytorch=str(torch.__version__),
            cuda=str(torch.version.cuda or "none"),
            gpu_model=torch.cuda.get_device_name(0) if gpu_count else "none",
            gpu_count=gpu_count,
            precision="fp32",
            distributed_topology="single-process",
        ),
        input_checkpoints=[],
        dataset_splits=[],
        config=dict(
            model={},
            loss={},
            optimizer={},
            scheduler={},
            evaluation={"eval_split": "val", "qualification_scenario": getattr(request, "param", "success")},
        ),
        parent_run_ids=[],
        selection_lineage=[],
        validation_authorized=True,
        test_authorized=False,
        expected_scalar_outputs=[],
        array_references=[],
        command=["python", "fixture.py", "--run-id", "{run_id}"],
    )
    phase.write_text(json.dumps(dict(phase="cpu", study="fixture-study", runs=[run])))
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source")
    payload = manager.reserve_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase.relative_to(repo),
            registry=registry.relative_to(repo),
            artifact_root=tmp_path / "artifacts",
        )
    )
    identity = payload["runs"][0]
    manifest = json.loads((Path(identity["artifact_path"]) / "manifest.json").read_text())
    config = RuntimeConfig(
        study,
        registry,
        Path(identity["artifact_path"]),
        identity["run_id"],
        identity["manifest_sha256"],
        manifest["git_sha"],
        "val",
    )
    return repo, config


def commit_reservation(repo, config):
    git(repo, "add", str(config.registry_path.relative_to(repo)))
    git(repo, "commit", "-qm", "reserved batch")


def test_uncommitted_and_dirty_source_cannot_start(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run

    repo, config = reservation
    with pytest.raises(ValueError):
        prepare_run(config, repo_root=repo)
    commit_reservation(repo, config)
    config.study_path.write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        prepare_run(config, repo_root=repo)


@pytest.mark.parametrize("defect", ["identity", "source", "split", "command", "path"])
def test_runtime_identity_config_and_command_binding(reservation, defect):
    from semantic_retrieval.runtime.artifacts import prepare_run

    repo, config = reservation
    commit_reservation(repo, config)
    if defect == "identity":
        config = replace(config, run_id="wrong")
    elif defect == "source":
        config = replace(config, source_sha="0" * 40)
    elif defect == "split":
        config = replace(config, eval_split="test")
    elif defect == "command":
        (config.run_dir / "command.sh").write_text("forged\n")
    else:
        config = replace(config, run_dir=config.run_dir / "..")
    with pytest.raises(ValueError):
        prepare_run(config, repo_root=repo)


def test_cpu_lifecycle_large_array_and_terminal_reuse(reservation):
    from semantic_retrieval.runtime.artifacts import complete_run, prepare_run, publish_metrics, start_run
    from semantic_retrieval.runtime.qualification import run_cpu_qualification

    repo, config = reservation
    commit_reservation(repo, config)
    context = prepare_run(config, repo_root=repo)
    start_run(context)
    run_cpu_qualification(context, scenario="success")
    path = publish_metrics(context, {"summary_metrics": {"loss": 1.0}})
    assert path == context.paths.metrics / "val_eval_001.json"
    complete_run(context)
    complete_run(context)
    assert json.loads(context.paths.status.read_text())["state"] == "validating"
    with pytest.raises(ValueError):
        start_run(context)
    assert next(context.paths.arrays.rglob("*.npy")).stat().st_size > 1024 * 1024
    with pytest.raises(ValueError):
        prepare_run(config, repo_root=repo)


@pytest.mark.parametrize("reservation", ["array_tamper", "signal"], indirect=True)
def test_cpu_fault_preserves_failure_evidence(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run
    from semantic_retrieval.runtime.qualification import run_cpu_qualification

    repo, config = reservation
    commit_reservation(repo, config)
    context = prepare_run(config, repo_root=repo)
    start_run(context)
    scenario = context.manifest["config"]["evaluation"]["qualification_scenario"]
    with pytest.raises((ValueError, InterruptedError)):
        run_cpu_qualification(context, scenario=scenario)
    evidence = json.loads((context.paths.provenance / "worker_failure.json").read_text())
    assert evidence["exit_code"] == (143 if scenario == "signal" else 1)
    assert json.loads(context.paths.status.read_text())["state"] == "running"
