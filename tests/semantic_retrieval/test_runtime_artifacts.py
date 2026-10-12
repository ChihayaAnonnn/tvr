import hashlib
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
    if isinstance(getattr(request, "param", None), dict):
        run.update(request.param)
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


def test_missing_reserved_split_is_not_a_wildcard(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run

    repo, config = reservation
    # Use a new uniquely reserved fixture, not an in-place mutation of an existing run.
    phase = config.study_path.parent / "phase.json"
    payload = json.loads(phase.read_text())
    payload["runs"][0]["variant"] = "missing-split"
    payload["runs"][0]["config"]["evaluation"].pop("eval_split")
    phase.write_text(json.dumps(payload))
    git(repo, "add", str(phase.relative_to(repo)), str(config.registry_path.relative_to(repo)))
    git(repo, "commit", "-qm", "new fixture phase source")
    result = manager.reserve_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase.relative_to(repo),
            registry=config.registry_path.relative_to(repo),
            artifact_root=config.run_dir.parents[3],
        )
    )
    run = result["runs"][0]
    manifest = json.loads((Path(run["artifact_path"]) / "manifest.json").read_text())
    config = replace(
        config,
        run_dir=Path(run["artifact_path"]),
        run_id=run["run_id"],
        manifest_sha256=run["manifest_sha256"],
        source_sha=manifest["git_sha"],
        eval_split="val_locked",
    )
    commit_reservation(repo, config)
    with pytest.raises(ValueError, match="eval_split"):
        prepare_run(config, repo_root=repo)


def test_start_rechecks_source_after_prepare(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run

    repo, config = reservation
    commit_reservation(repo, config)
    context = prepare_run(config, repo_root=repo)
    config.study_path.write_text("changed after preparation\n")
    with pytest.raises(ValueError, match="clean"):
        start_run(context)
    assert json.loads(context.paths.status.read_text())["state"] == "reserved"


def test_start_rechecks_input_after_prepare(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run

    repo, config = reservation
    checkpoint = repo.parent / "fixture.bin"
    checkpoint.write_bytes(b"bound")
    phase = config.study_path.parent / "phase.json"
    payload = json.loads(phase.read_text())
    payload["runs"][0]["variant"] = "input-drift"
    payload["runs"][0]["input_checkpoints"] = [
        {"path": str(checkpoint), "sha256": hashlib.sha256(b"bound").hexdigest()}
    ]
    phase.write_text(json.dumps(payload))
    git(repo, "add", str(phase.relative_to(repo)), str(config.registry_path.relative_to(repo)))
    git(repo, "commit", "-qm", "input fixture phase")
    result = manager.reserve_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase.relative_to(repo),
            registry=config.registry_path.relative_to(repo),
            artifact_root=config.run_dir.parents[3],
        )
    )
    run = result["runs"][0]
    manifest = json.loads((Path(run["artifact_path"]) / "manifest.json").read_text())
    config = replace(
        config,
        run_dir=Path(run["artifact_path"]),
        run_id=run["run_id"],
        manifest_sha256=run["manifest_sha256"],
        source_sha=manifest["git_sha"],
    )
    commit_reservation(repo, config)
    context = prepare_run(config, repo_root=repo)
    checkpoint.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="SHA-256"):
        start_run(context)
    assert json.loads(context.paths.status.read_text())["state"] == "reserved"


def test_fp32_gate_rejects_ambient_autocast(reservation):
    from semantic_retrieval.runtime.artifacts import prepare_run

    repo, config = reservation
    commit_reservation(repo, config)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="precision"):
        prepare_run(config, repo_root=repo)


def test_final_typed_consumer_never_enters_training_resume_validators(tmp_path,monkeypatch):
    # This test isolates runtime dispatch after an explicit permission grant.
    # Production final authorization is separately exercised at reserve/start.
    from semantic_retrieval.full_data import authorization
    from semantic_retrieval.full_data.final import FinalSystemConfig
    from semantic_retrieval.runtime.artifacts import prepare_run
    from semantic_retrieval.runtime.distributed import configure_fp32
    configure_fp32()
    monkeypatch.setattr(authorization,'assert_full_data_authorized',lambda **kwargs:None)
    payload=dict(stage='final',test_authorized=True,validation_authorized=False,parent_run_ids=['fixture-parent'],
                 config=dict(model={},loss={},optimizer={},scheduler={},evaluation=dict(eval_split='test',config_kind='full-data-test-v1')))
    repo,config=reservation.__wrapped__(tmp_path,SimpleNamespace(param=payload))
    config=replace(config,eval_split='test');commit_reservation(repo,config)
    manifest=json.loads((config.run_dir/'manifest.json').read_text())
    context=prepare_run(config,repo_root=repo,system_config=FinalSystemConfig(config,manifest['config']))
    assert context.config.eval_split=='test'
    with pytest.raises(ValueError,match='typed consumer'):prepare_run(config,repo_root=repo)
