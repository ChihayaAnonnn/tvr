import json
import multiprocessing
from pathlib import Path
import subprocess

import pytest
from jsonschema import Draft202012Validator

from experiment_artifacts import (
    ALLOWED_STAGES,
    ALLOWED_STATES,
    append_registry_records,
    build_run_id,
    load_registry,
    make_run_paths,
    reserve_run_directory,
    transition_run_status,
    validate_manifest,
    verify_committed_reservation,
)


REGISTRY = Path("experiments/contextual-risk-v2/registry.jsonl")


def _manifest_sha256(payload):
    import hashlib

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _reservation_record(manifest):
    return {
        "run_id": manifest["run_id"],
        "event": "reserved",
        "artifact_path": manifest["artifact_path"],
        "manifest_sha256": _manifest_sha256(manifest),
        "state": "reserved",
        "parent_run_ids": manifest["parent_run_ids"],
        "selected_validation": None,
        "test_authorized": manifest["test_authorized"],
        "summary_metrics": {},
    }


def _append_worker(registry_path, record):
    append_registry_records(Path(registry_path), [record])


def _manifest(**overrides):
    payload = {
        "run_id": "msrvtt-baseline-reference-seed000-20260814T020304Z-000000000000",
        "study": "contextual-risk-v2",
        "dataset": "msrvtt",
        "stage": "baseline",
        "variant": "reference",
        "seed": 0,
        "artifact_path": "/tmp/artifacts/contextual-risk-v2/msrvtt/baseline/example",
        "registry_path": "experiments/contextual-risk-v2/registry.jsonl",
        "git_sha": "0" * 40,
        "tracked_clean": True,
        "source_patch_sha256": None,
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
            "model": {}, "loss": {}, "optimizer": {}, "scheduler": {}, "evaluation": {},
        },
        "parent_run_ids": [],
        "selection_lineage": [],
        "validation_authorized": True,
        "test_authorized": False,
        "expected_scalar_outputs": [],
        "array_references": [],
    }
    payload.update(overrides)
    return payload


def test_build_run_id_and_paths_are_canonical(tmp_path):
    run_id = build_run_id(
        dataset="msrvtt", stage="feature_ablation", variant="no_context",
        seed=0, timestamp_utc="20260814T020304Z",
        git_sha="0123456789abcdef",
    )
    assert run_id == (
        "msrvtt-feature_ablation-no_context-seed000-"
        "20260814T020304Z-0123456789ab"
    )
    paths = make_run_paths(tmp_path, "contextual-risk-v2", run_id)
    assert paths.root == tmp_path / "contextual-risk-v2" / "msrvtt" / "feature_ablation" / run_id
    assert paths.manifest == paths.root / "manifest.json"
    assert paths.checkpoints == paths.root / "checkpoints"


@pytest.mark.parametrize("variant", ["../escape", "bad/name", "", "a b", "a--b"])
def test_run_id_rejects_unsafe_variant(variant):
    with pytest.raises(ValueError):
        build_run_id("msrvtt", "baseline", variant, 0, "20260814T020304Z", "0" * 40)


@pytest.mark.parametrize("stage", ["unknown", "baseline/escape"])
def test_run_id_rejects_unknown_or_unsafe_stage(stage):
    with pytest.raises(ValueError):
        build_run_id("msrvtt", stage, "reference", 0, "20260814T020304Z", "0" * 40)


@pytest.mark.parametrize("timestamp", ["20260814T020304", "20260814T020304+00:00", "20261314T020304Z"])
def test_run_id_requires_utc_timestamp(timestamp):
    with pytest.raises(ValueError):
        build_run_id("msrvtt", "baseline", "reference", 0, timestamp, "0" * 40)


@pytest.mark.parametrize("field, value", [
    ("artifact_path", "relative/artifacts"),
    ("source_patch_sha256", "f" * 63),
    ("input_checkpoints", [{"path": "checkpoints/a.pt", "sha256": "f" * 63}]),
])
def test_manifest_rejects_noncanonical_paths_and_hashes(field, value):
    with pytest.raises(ValueError):
        validate_manifest(_manifest(**{field: value}))


def test_manifest_rejects_unknown_fields_and_unauthorized_test_access():
    with pytest.raises(ValueError, match="unknown"):
        validate_manifest(_manifest(unexpected=True))
    with pytest.raises(ValueError, match="final"):
        validate_manifest(_manifest(test_authorized=True))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("run_id", "msrvtt-baseline-reference-seed000-20261314T020304Z-000000000000", "UTC"),
        ("git_sha", "f" * 40, "git_sha"),
    ],
)
def test_manifest_rejects_invalid_or_mismatched_embedded_run_identity(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_manifest(_manifest(**{field: value}))


def test_manifest_requires_parent_lineage_for_derived_runs():
    derived_run_id = "msrvtt-feature_ablation-reference-seed000-20260814T020304Z-000000000000"
    with pytest.raises(ValueError, match="parent"):
        validate_manifest(_manifest(stage="feature_ablation", run_id=derived_run_id))
    assert validate_manifest(
        _manifest(
            stage="feature_ablation",
            run_id=derived_run_id,
            parent_run_ids=["prior-run"],
            selection_lineage=["selected from baseline"],
        )
    )["stage"] == "feature_ablation"


def test_schemas_match_canonical_enums():
    schema_dir = Path("experiments/contextual-risk-v2/schemas")
    manifest = json.loads((schema_dir / "manifest.schema.json").read_text())
    status = json.loads((schema_dir / "status.schema.json").read_text())
    assert set(manifest["properties"]["stage"]["enum"]) == ALLOWED_STAGES
    assert set(status["properties"]["state"]["enum"]) == ALLOWED_STATES


def test_manifest_schema_rejects_duplicate_separators_and_nonfinal_test_access():
    schema = json.loads(Path("experiments/contextual-risk-v2/schemas/manifest.schema.json").read_text())
    validator = Draft202012Validator(schema)
    duplicate_separator = _manifest(
        variant="reference--copy",
        run_id="msrvtt-baseline-reference--copy-seed000-20260814T020304Z-000000000000",
    )
    assert not validator.is_valid(duplicate_separator)
    assert not validator.is_valid(_manifest(test_authorized=True))


def test_registry_is_append_only_and_rejects_duplicate_events(tmp_path):
    registry = tmp_path / "registry.jsonl"
    manifest = _manifest()
    record = _reservation_record(manifest)

    append_registry_records(registry, [record])

    assert load_registry(registry) == [record]
    with pytest.raises(ValueError, match="duplicate"):
        append_registry_records(registry, [record])
    assert load_registry(registry) == [record]


def test_registry_corrections_must_name_superseded_record(tmp_path):
    registry = tmp_path / "registry.jsonl"
    correction = {
        **_reservation_record(_manifest()),
        "event": "correction",
        "state": "reserved",
    }

    with pytest.raises(ValueError, match="supersedes_sha256"):
        append_registry_records(registry, [correction])


def test_registry_append_uses_atomic_replacement(tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _reservation_record(_manifest())
    second = {
        **first,
        "run_id": "a-second-run",
        "event": "completed",
        "state": "completed",
    }
    append_registry_records(registry, [first])
    before_inode = registry.stat().st_ino

    append_registry_records(registry, [second])

    assert registry.stat().st_ino != before_inode
    assert load_registry(registry) == [first, second]
    assert not list(tmp_path.glob(".registry.jsonl.*.tmp"))


def test_registry_concurrent_append_preserves_both_records(tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _reservation_record(_manifest())
    second = {
        **first,
        "run_id": "a-second-run",
        "event": "completed",
        "state": "completed",
    }
    first_process = multiprocessing.Process(target=_append_worker, args=(str(registry), first))
    second_process = multiprocessing.Process(target=_append_worker, args=(str(registry), second))

    first_process.start()
    second_process.start()
    first_process.join(timeout=10)
    second_process.join(timeout=10)

    assert first_process.exitcode == 0
    assert second_process.exitcode == 0
    assert {record["run_id"] for record in load_registry(registry)} == {
        first["run_id"], second["run_id"],
    }


def _init_git_repo(repo_root):
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    registry = repo_root / REGISTRY
    registry.parent.mkdir(parents=True)
    registry.touch()
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial registry"], cwd=repo_root, check=True)


def test_worker_requires_exact_committed_reservation(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    manifest = _manifest()
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    verify_committed_reservation(repo_root, REGISTRY, manifest)
    changed = {**manifest, "seed": 1}
    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, changed)


def test_worker_rejects_missing_or_uncommitted_reservation(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    manifest = _manifest()

    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    with pytest.raises(ValueError, match="clean"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_reservation_is_exclusive_and_initializes_runtime_files(tmp_path):
    run_id = _manifest()["run_id"]
    paths = make_run_paths(tmp_path, "contextual-risk-v2", run_id)
    manifest = _manifest(artifact_path=str(paths.root))

    reserve_run_directory(paths, manifest, worker_pid=17, host="test-host")

    assert json.loads(paths.manifest.read_text()) == manifest
    assert json.loads(paths.status.read_text())["state"] == "reserved"
    assert (paths.provenance / "audit.jsonl").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_run_directory(paths, manifest)


def test_transition_allows_only_forward_lifecycle_and_terminal_runs_are_immutable(tmp_path):
    run_id = _manifest()["run_id"]
    paths = make_run_paths(tmp_path, "contextual-risk-v2", run_id)
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)

    transition_run_status(paths, "reserved", "running", worker_pid=18, host="worker")
    transition_run_status(paths, "running", "validating", worker_pid=18, host="worker")
    with pytest.raises(ValueError, match="not allowed"):
        transition_run_status(paths, "validating", "running")
    transition_run_status(
        paths, "validating", "completed", worker_pid=18, host="worker", exit_code=0,
        output_hashes={"metrics/selected_validation.json": "a" * 64},
    )
    with pytest.raises(FileExistsError, match="terminal"):
        transition_run_status(paths, "completed", "running")
    audit_records = [json.loads(line) for line in (paths.provenance / "audit.jsonl").read_text().splitlines()]
    assert audit_records[-1]["output_hashes"] == {"metrics/selected_validation.json": "a" * 64}
