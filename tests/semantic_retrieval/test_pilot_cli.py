"""CPU preparation uses the real reservation/lifecycle and never accepts free outputs."""

import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_pilot_catalog import source_fixture

from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.data.pilot_catalog import identity_digest


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def protected(tmp_path):
    from scripts import manage_experiment_run as manager

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    catalog, sha, _ = source_fixture(data)
    prep = dict(
        mode="inventory",
        protocol_id="fixture",
        relevance_space="benchmark_annotation",
        allowed_root=str(data),
        catalog={"path": str(catalog), "sha256": sha},
        annotation=None,
        source_confirmation=None,
        allowlist_csv=None,
        source_annotation=None,
        video_directory=None,
        candidate_orders=None,
    )
    name = "semantic-answerability-pilot-v1"
    control = repo / "experiments" / name
    control.mkdir(parents=True)
    study = control / "study.yaml"
    study.write_text(
        f"study: {name}\nstatus: design_approved\ncontrol_plane: experiments/{name}\ndata_plane: artifacts/{name}\nregistry: experiments/{name}/registry.jsonl\n"
    )
    registry = control / "registry.jsonl"
    registry.touch()
    run = dict(
        dataset="msrvtt",
        stage="baseline",
        variant="source_inventory",
        seed=0,
        timestamp_utc="20261001T120000Z",
        environment=dict(
            python=platform.python_version(),
            pytorch=str(torch.__version__),
            cuda=str(torch.version.cuda or "none"),
            gpu_model="none",
            gpu_count=0,
            precision="fp32",
            distributed_topology="single-process",
        ),
        input_checkpoints=[],
        dataset_splits=[{"path": str(p), "sha256": file_sha256(p)} for p in data.iterdir()],
        config=dict(
            model={},
            loss={},
            optimizer={},
            scheduler={},
            evaluation=dict(eval_split="val", qualification_scenario=f"pilot_prepare:{identity_digest(prep)}"),
        ),
        parent_run_ids=[],
        selection_lineage=[],
        validation_authorized=True,
        test_authorized=False,
        expected_scalar_outputs=[],
        array_references=[],
        command=["python", "fixture.py", "--run-id", "{run_id}"],
    )
    phase = control / "phase.json"
    phase.write_text(json.dumps(dict(phase="D1", study=name, runs=[run])))
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source")
    result = manager.reserve_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase.relative_to(repo),
            registry=registry.relative_to(repo),
            artifact_root=repo / "artifacts",
        )
    )["runs"][0]
    manifest = json.loads((Path(result["artifact_path"]) / "manifest.json").read_text())
    runtime = RuntimeConfig(
        study,
        registry,
        Path(result["artifact_path"]),
        result["run_id"],
        result["manifest_sha256"],
        manifest["git_sha"],
        "val",
    )
    return repo, runtime, prep


def test_uncommitted_reservation_and_changed_preparation_rejected(protected):
    from scripts.prepare_semantic_pilot import execute_preparation

    repo, runtime, prep = protected
    with pytest.raises(ValueError):
        execute_preparation(runtime, prep, repo_root=repo)
    git(repo, "add", str(runtime.registry_path.relative_to(repo)))
    git(repo, "commit", "-qm", "reserved")
    with pytest.raises(ValueError):
        execute_preparation(runtime, {**prep, "mode": "compile"}, repo_root=repo)
    assert json.loads((runtime.run_dir / "status.json").read_text())["state"] == "reserved"


def test_inventory_publishes_run_local_evidence_without_tracked_edits(protected):
    from scripts.prepare_semantic_pilot import execute_preparation

    repo, runtime, prep = protected
    git(repo, "add", str(runtime.registry_path.relative_to(repo)))
    git(repo, "commit", "-qm", "reserved")
    result = execute_preparation(runtime, prep, repo_root=repo)
    assert result["official_video_count"] == 1 and result["query_count"] == 1
    assert list((runtime.run_dir / "metrics").glob("*.json"))
    assert git(repo, "diff", "--name-only") == ""
    assert json.loads((runtime.run_dir / "status.json").read_text())["state"] == "validating"
