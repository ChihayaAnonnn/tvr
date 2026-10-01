"""Committed archive evidence, rather than YAML flags, unlocks pilot phases."""

import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiment_artifacts import (
    append_registry_records,
    artifact_json_bytes,
    hash_run_outputs,
    make_run_paths,
    manifest_sha256,
    reserve_run_directory,
    transition_run_status,
)
from scripts import manage_experiment_run as manager
from test_runtime_artifacts import git

NAME = "semantic-answerability-pilot-v1"
CONTROL = f"experiments/{NAME}"


@pytest.fixture
def pilot_repo(tmp_path):
    repo = tmp_path / "repo"
    control = repo / CONTROL
    control.mkdir(parents=True)
    (control / "registry.jsonl").touch()
    (repo / ".gitignore").write_text("artifacts/\n")
    (control / "study.yaml").write_text(
        yaml.safe_dump(
            dict(
                study=NAME,
                status="design_approved",
                scientific_authorized=True,
                stage0_passed=True,
                control_plane=CONTROL,
                data_plane=f"artifacts/{NAME}",
                registry=f"{CONTROL}/registry.jsonl",
                test_authorized=False,
            )
        )
    )
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source")
    return repo


def template(variant="source_inventory", seed=0, gpu_count=0):
    return dict(
        dataset="msrvtt",
        stage="baseline",
        variant=variant,
        seed=seed,
        scheduled_timestamp_utc="20261001T120000Z",
        environment=dict(
            python=platform.python_version(),
            pytorch=str(torch.__version__),
            cuda=str(torch.version.cuda or "none"),
            gpu_model="A800" if gpu_count else "none",
            gpu_count=gpu_count,
            precision="fp32",
            distributed_topology="torchrun-4" if gpu_count else "single-process",
        ),
        input_checkpoints=[],
        dataset_splits=[],
        config=dict(
            model={},
            loss={},
            optimizer={},
            scheduler={},
            evaluation=dict(eval_split="val", qualification_scenario="success"),
        ),
        parent_run_ids=[],
        selection_lineage=[],
        validation_authorized=True,
        test_authorized=False,
        expected_scalar_outputs=[],
        array_references=[],
        command=["python", "fixture.py", "--run-id", "{run_id}"],
    )


def completed_evidence(repo, phase_id="D1", *, state="completed", commit_terminal=True, outputs=None):
    """Use the real immutable lifecycle and phase finalizer, with a CPU fixture worker."""
    spec = template("benchmark_compile" if phase_id == "D1" else "pilot-analysis-freeze")
    phase_path = Path(CONTROL) / "phases" / f"{phase_id}.json"
    (repo / phase_path).parent.mkdir(exist_ok=True)
    (repo / phase_path).write_bytes(artifact_json_bytes(dict(phase=phase_id, study=NAME, runs=[spec])))
    git(repo, "add", str(phase_path))
    git(repo, "commit", "-qm", "phase")
    manifest = manager._build_manifest(
        spec,
        study=NAME,
        git_sha=git(repo, "rev-parse", "HEAD"),
        artifact_root=repo / "artifacts",
        registry=Path(CONTROL) / "registry.jsonl",
        common_root=repo,
    )
    paths = make_run_paths(repo / "artifacts", NAME, manifest["run_id"])
    reserve_run_directory(paths, manifest)
    paths.command.write_bytes(manager._render_command(spec["command"], run_id=manifest["run_id"], run_dir=paths.root))
    append_registry_records(repo / CONTROL / "registry.jsonl", [manager._registry_record(manifest, "reserved")])
    git(repo, "add", f"{CONTROL}/registry.jsonl")
    git(repo, "commit", "-qm", "reserved batch")
    transition_run_status(paths, "reserved", "running")
    for name, payload in (outputs or {"provenance/input-summary.json": {"ready": True}}).items():
        (paths.root / name).write_bytes(artifact_json_bytes(payload))
    if state == "completed":
        transition_run_status(paths, "running", "validating")
        transition_run_status(paths, "validating", state, exit_code=0, output_hashes=hash_run_outputs(paths))
    elif state == "failed":
        transition_run_status(paths, "running", "validating")
        transition_run_status(paths, "validating", state, exit_code=1, output_hashes=hash_run_outputs(paths))
    else:
        transition_run_status(paths, "running", state, exit_code=1, output_hashes=hash_run_outputs(paths))
    summary_path = Path(CONTROL) / "summaries" / f"{phase_id}.json"
    manager.finalize_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase_path,
            registry=Path(CONTROL) / "registry.jsonl",
            artifact_root=repo / "artifacts",
            summary=summary_path,
        )
    )
    if commit_terminal:
        git(repo, "add", str(summary_path), f"{CONTROL}/registry.jsonl")
        git(repo, "commit", "-qm", "terminal batch")
    return dict(
        path=summary_path.as_posix(),
        sha256=hashlib.sha256((repo / summary_path).read_bytes()).hexdigest(),
        commit=git(repo, "rev-parse", "HEAD"),
    ), paths


def test_phase_contract_requires_all_prior_batches_and_exact_seed_arms():
    from semantic_retrieval.runtime.pilot_protocol import phase_contract

    assert phase_contract("P1a")["runs"] == (("A0", 0), ("P1", 0))
    assert phase_contract("P1b")["runs"] == (("A1", 0), ("A2", 0), ("A3", 0), ("A4", 0))
    assert phase_contract("P2a")["runs"] == (("A0", 1), ("P1", 1), ("A0", 2), ("P1", 2))
    assert len(phase_contract("P2b")["runs"]) == 8
    assert "Q0" in phase_contract("E0")["requires"]
    assert set(phase_contract("C0")["requires"]) >= {"P1a", "P1b", "P2a", "P2b"}
    assert phase_contract("V0")["requires"] == ("C0", "V0S")


def test_authenticate_completed_committed_phase_and_all_output_hashes(pilot_repo):
    from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence

    evidence, paths = completed_evidence(pilot_repo)
    result = verify_phase_evidence(pilot_repo, evidence, "D1")
    assert result["runs"][0]["manifest"]["run_id"] == paths.root.name
    assert result["runs"][0]["run_dir"] == str(paths.root)
    (paths.provenance / "input-summary.json").write_text('{"ready":false}')
    with pytest.raises(ValueError, match="output"):
        verify_phase_evidence(pilot_repo, evidence, "D1")


@pytest.mark.parametrize(
    "dependency", ["experiment_tracking.py", "experiment_artifacts.py", "src/semantic_retrieval/runtime/artifacts.py"]
)
def test_qualified_source_rejects_changed_executable_dependencies(pilot_repo, dependency):
    from semantic_retrieval.runtime.pilot_protocol import _same_q0_source

    path = pilot_repo / dependency
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("VALUE = 1\n")
    git(pilot_repo, "add", ".")
    git(pilot_repo, "commit", "-qm", "qualified source")
    qualified = git(pilot_repo, "rev-parse", "HEAD")
    path.write_text("VALUE = 2\n")
    git(pilot_repo, "add", ".")
    git(pilot_repo, "commit", "-qm", "changed executable")
    with pytest.raises(ValueError, match="source changed"):
        _same_q0_source(pilot_repo, qualified, git(pilot_repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("defect", ["dirty", "digest", "foreign", "phase", "terminal", "failed", "cancelled"])
def test_forged_or_incomplete_predecessor_refused(pilot_repo, defect):
    from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence

    evidence, _ = completed_evidence(
        pilot_repo,
        state=defect if defect in {"failed", "cancelled"} else "completed",
        commit_terminal=defect != "terminal",
    )
    if defect == "dirty":
        (pilot_repo / evidence["path"]).write_text("{}")
    elif defect == "digest":
        evidence["sha256"] = "0" * 64
    elif defect == "foreign":
        evidence["path"] = "experiments/contextual-risk-v2/summaries/Q0.json"
    with pytest.raises(ValueError):
        verify_phase_evidence(pilot_repo, evidence, "E0" if defect == "phase" else "D1")


def test_boolean_flags_never_allow_unqualified_scientific_start(pilot_repo):
    from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized

    spec = template("A0", gpu_count=4)
    manifest = manager._build_manifest(
        spec,
        study=NAME,
        git_sha=git(pilot_repo, "rev-parse", "HEAD"),
        artifact_root=pilot_repo / "artifacts",
        registry=Path(CONTROL) / "registry.jsonl",
        common_root=pilot_repo,
    )
    with pytest.raises(ValueError):
        assert_pilot_authorized(
            repo_root=pilot_repo,
            registry_path=pilot_repo / CONTROL / "registry.jsonl",
            manifest=manifest,
            operation="start",
        )


def test_q0_rejects_old_cpu_or_incomplete_qualification_reports():
    from semantic_retrieval.runtime.pilot_protocol import validate_q0_report

    with pytest.raises(ValueError):
        validate_q0_report({"checks_passed": 15, "world_size": 4}, source_sha="a" * 40)
    with pytest.raises(ValueError):
        validate_q0_report({"protocol": NAME, "world_size": 1, "device": "cpu"}, source_sha="a" * 40)


def test_build_phase_refuses_missing_arm_duplicate_seed_and_budget_overrun():
    from semantic_retrieval.runtime.pilot_protocol import build_phase

    a0, p1 = template("A0", gpu_count=4), template("P1", gpu_count=4)
    evidence = {
        key: dict(path=f"{CONTROL}/summaries/{key}.json", sha256="a" * 64, commit="b" * 40)
        for key in ("D1", "Q0", "E0")
    }
    kwargs = dict(
        phase_path=f"{CONTROL}/phases/P1a.json", scheduled_timestamp_utc="20261001T120000Z", evidence=evidence
    )
    with pytest.raises(ValueError, match="batch"):
        build_phase("P1a", run_templates=[a0], **kwargs)
    with pytest.raises(ValueError, match="batch"):
        build_phase("P1a", run_templates=[a0, a0], **kwargs)
    phase = build_phase("P1a", run_templates=[a0, p1], **kwargs)
    assert phase["study"] == NAME
    assert len(phase["runs"]) == 2
    assert all(r["test_authorized"] is False for r in phase["runs"])
    assert all(r["selection_lineage"][-1]["gpu_hours"] == 8 for r in phase["runs"])


def test_mechanical_subphases_preserve_all_176_gpu_hours():
    from semantic_retrieval.runtime.pilot_protocol import phase_contract

    phases = ("D1", "Q0P", "Q0", "E0", "P1a", "P1b", "P2a", "P2b", "C0S", "C0", "V0E", "V0S", "V0")
    totals = dict(training=0.0, qualification=0.0, evaluation=0.0)
    for name in phases:
        c = phase_contract(name)
        totals[c["category"]] += c["run_gpu_hours"] * len(c["runs"])
    assert totals == dict(training=144.0, qualification=8.0, evaluation=24.0)
    assert sum(totals.values()) == 176.0
    assert phase_contract("V0E")["requires"] == ("D1", "Q0", "C0")


def test_locked_gate_requires_an_actual_hash_bound_freeze(pilot_repo):
    from semantic_retrieval.runtime.pilot_protocol import validate_frozen_prerequisite, verify_phase_evidence

    evidence, _ = completed_evidence(pilot_repo, phase_id="C0", outputs={"provenance/only-ready.json": {"ready": True}})
    predecessor = verify_phase_evidence(pilot_repo, evidence, "C0")
    with pytest.raises(ValueError, match="frozen"):
        validate_frozen_prerequisite(pilot_repo, predecessor)


def test_predecessor_cannot_relabel_an_incomplete_scientific_batch(pilot_repo):
    from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence

    evidence, _ = completed_evidence(pilot_repo, phase_id="P1a")
    with pytest.raises(ValueError, match="registered|batch"):
        verify_phase_evidence(pilot_repo, evidence, "P1a")
