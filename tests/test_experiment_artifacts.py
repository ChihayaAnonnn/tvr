import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import experiment_artifacts
from experiment_artifacts import (
    ALLOWED_STAGES,
    ALLOWED_STATES,
    RunPaths,
    append_registry_records,
    artifact_schema_validator,
    build_run_id,
    load_registry,
    make_run_paths,
    reserve_run_directory,
    transition_run_status,
    validate_manifest,
    verify_committed_reservation,
)


REGISTRY = Path("experiments/contextual-risk-v2/registry.jsonl")


def test_archive_schema_runtime_dependency_is_declared_and_importable():
    requirements = Path("requirements.txt").read_text().splitlines()
    assert "jsonschema==4.23.0" in requirements

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiment_artifacts import artifact_schema_validator; "
            "assert callable(artifact_schema_validator)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_large_arrays_are_externalized_and_deduplicated(tmp_path):
    """Array references remain small while content lives once on disk."""

    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    values = np.arange(2_000_000, dtype=np.float32)

    first = store.write("raw_wti", values)
    second = store.write("raw_wti_copy", values)

    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert len(json.dumps(first.as_dict()).encode()) < 1024
    np.testing.assert_array_equal(
        experiment_artifacts.verify_array_reference(tmp_path, first), values
    )


@pytest.mark.parametrize(
    "values, message",
    [
        (np.asarray([1.0, np.nan], dtype=np.float32), "non-finite"),
        (np.asarray(["unsafe"], dtype=object), "object"),
    ],
)
def test_array_store_rejects_untrusted_array_encodings(tmp_path, values, message):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")

    with pytest.raises(ValueError, match=message):
        store.write("raw_wti", values)


def test_externalized_payload_rejects_embedded_arrays_and_large_json(tmp_path):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    payload = {
        "evidence": {"risk": np.arange(8, dtype=np.float32)},
        "scalar": 1.0,
    }

    externalized = experiment_artifacts.externalize_numeric_payload(payload, store)

    assert "path" in externalized["evidence"]["risk"]
    assert payload["evidence"]["risk"].dtype == np.float32
    experiment_artifacts.validate_artifact_json_payload(externalized)
    with pytest.raises(ValueError, match=r"evidence\.risk"):
        experiment_artifacts.validate_artifact_json_payload(payload)
    with pytest.raises(ValueError, match="1 MiB"):
        experiment_artifacts.validate_artifact_json_payload({"large": "x" * (1024 * 1024)})


def test_array_reference_rejects_traversal_partial_writes_and_hash_mismatch(tmp_path):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    reference = store.write("raw_wti", np.arange(32, dtype=np.float32))
    traversal = reference.as_dict() | {"path": "../escape.npy"}
    with pytest.raises(ValueError, match="traversal"):
        experiment_artifacts.verify_array_reference(tmp_path, traversal)

    path = tmp_path / reference.path
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(ValueError, match="file size"):
        experiment_artifacts.verify_array_reference(tmp_path, reference)


def test_artifact_json_limit_uses_the_pretty_utf8_bytes_written():
    payload = {"values": ["x"] * 120_000}

    with pytest.raises(ValueError, match="1 MiB"):
        experiment_artifacts.validate_artifact_json_payload(payload)


def test_artifact_json_boundary_matches_exact_writer_bytes():
    empty_size = len(experiment_artifacts.artifact_json_bytes({"value": ""}))
    accepted = {"value": "x" * (1_048_575 - empty_size)}
    rejected = {"value": accepted["value"] + "x"}

    assert len(experiment_artifacts.artifact_json_bytes(accepted)) == 1_048_575
    experiment_artifacts.validate_artifact_json_payload(accepted)
    assert len(experiment_artifacts.artifact_json_bytes(rejected)) == 1_048_576
    with pytest.raises(ValueError, match="1 MiB"):
        experiment_artifacts.validate_artifact_json_payload(rejected)


def test_artifact_json_boundary_counts_unescaped_unicode_utf8():
    empty_size = len(experiment_artifacts.artifact_json_bytes({"value": ""}))
    accepted = {"value": "é" * ((1_048_575 - empty_size) // 2)}

    assert len(experiment_artifacts.artifact_json_bytes(accepted)) < 1_048_576
    experiment_artifacts.validate_artifact_json_payload(accepted)


def test_array_store_refuses_symlinked_ancestor_before_creating_children(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        experiment_artifacts.ArtifactArrayStore(tmp_path / "redirect" / "arrays")

    assert not (outside / "arrays").exists()


def test_zero_dimensional_array_reference_preserves_its_shape(tmp_path):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    reference = store.write("scalar", np.asarray(1.5, dtype=np.float32))

    assert reference.shape == ()
    assert experiment_artifacts.verify_array_reference(tmp_path, reference).shape == ()


def test_zero_dimensional_and_vector_arrays_have_distinct_content_addresses(tmp_path):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")

    scalar = store.write("scalar", np.asarray(1.5, dtype=np.float32))
    vector = store.write("vector", np.asarray([1.5], dtype=np.float32))

    assert scalar.sha256 != vector.sha256
    assert scalar.path != vector.path
    assert experiment_artifacts.verify_array_reference(tmp_path, scalar).shape == ()
    assert experiment_artifacts.verify_array_reference(tmp_path, vector).shape == (1,)


def test_array_store_write_does_not_escape_after_prefix_symlink_swap(tmp_path, monkeypatch):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    values = np.arange(4, dtype=np.float32)
    reference = store._reference_for("safe", values)
    prefix = tmp_path / "arrays" / "sha256" / reference.sha256[:2]
    outside = tmp_path / "outside"
    outside.mkdir()
    original = experiment_artifacts._open_safe_directory
    swapped = False

    def swap_after_open(root, components):
        nonlocal swapped
        descriptor = original(root, components)
        if not swapped and components == ("sha256", reference.sha256[:2]):
            swapped = True
            held = prefix.with_name(f".{prefix.name}.held")
            prefix.rename(held)
            prefix.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(experiment_artifacts, "_open_safe_directory", swap_after_open)
    try:
        with pytest.raises(ValueError, match="symlink|unsafe"):
            store.write("safe", values)
    finally:
        if prefix.is_symlink():
            prefix.unlink()
        held = prefix.with_name(f".{prefix.name}.held")
        if held.exists():
            held.rename(prefix)

    assert not list(outside.iterdir())


def test_array_reference_dataclass_is_validated_like_a_mapping(tmp_path):
    store = experiment_artifacts.ArtifactArrayStore(tmp_path / "arrays")
    reference = store.write("safe", np.arange(4, dtype=np.float32))

    with pytest.raises(ValueError, match="format"):
        experiment_artifacts.verify_array_reference(
            tmp_path, replace(reference, format="npz")
        )


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


def _paths_for_root(root):
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        status=root / "status.json",
        command=root / "command.sh",
        logs=root / "logs",
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        arrays=root / "arrays",
        plots=root / "plots",
        provenance=root / "provenance",
    )


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
        "command_sha256": "0" * 64,
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


def test_metrics_schema_and_python_validator_accept_production_archive_payload():
    payload = {
        "split": "validation",
        "summary_metrics": {"qualification_loss": 7.5},
        "array_references": [{
            "path": "arrays/sha256/99/" + "9" * 64 + ".npy",
            "key": "qualification_values",
            "shape": [524288],
            "dtype": "float32",
            "byte_size": 2097152,
            "format": "npy",
            "sha256": "9" * 64,
        }],
    }
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/metrics.schema.json").read_text()
    )

    assert artifact_schema_validator(schema).is_valid(payload)
    assert experiment_artifacts.validate_metrics_document(payload) == payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload | {"metrics": payload["summary_metrics"]},
        lambda payload: payload | {"summary_metrics": {"loss": float("nan")}},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"byte_size": "2097152"}
        ]},
    ],
)
def test_metrics_contract_rejects_unknown_nonfinite_and_malformed_documents(mutate):
    payload = {
        "split": "validation",
        "summary_metrics": {"loss": 1.0},
        "array_references": [{
            "path": "arrays/sha256/aa/" + "a" * 64 + ".npy",
            "key": "values",
            "shape": [1],
            "dtype": "float32",
            "byte_size": 4,
            "format": "npy",
            "sha256": "a" * 64,
        }],
    }

    with pytest.raises(ValueError):
        experiment_artifacts.validate_metrics_document(mutate(payload))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload | {"run_id": "../unsafe"},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"key": "unsafe..key"}
        ]},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"dtype": "object"}
        ]},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"dtype": "f4"}
        ]},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {
                "path": "arrays/sha256/bb/" + "a" * 64 + ".npy"
            }
        ]},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"sha256": "b" * 64}
        ]},
        lambda payload: payload | {"array_references": [
            payload["array_references"][0] | {"sha256": "aa" + "b" * 62}
        ]},
    ],
)
def test_metrics_schema_and_python_reject_the_same_unsafe_syntax(mutate):
    payload = {
        "split": "validation",
        "summary_metrics": {"loss": 1.0},
        "array_references": [{
            "path": "arrays/sha256/aa/" + "a" * 64 + ".npy",
            "key": "values",
            "shape": [1],
            "dtype": "float32",
            "byte_size": 4,
            "format": "npy",
            "sha256": "a" * 64,
        }],
    }
    invalid = mutate(payload)
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/metrics.schema.json").read_text()
    )

    assert not artifact_schema_validator(schema).is_valid(invalid)
    with pytest.raises(ValueError):
        experiment_artifacts.validate_metrics_document(invalid)


def test_status_schema_requires_null_exit_code_until_terminal():
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/status.schema.json").read_text()
    )
    status = {
        "run_id": _manifest()["run_id"],
        "state": "running",
        "timestamp_utc": "2026-08-15T00:00:00Z",
        "worker_pid": 1,
        "host": "fixture",
        "manifest_sha256": "a" * 64,
        "exit_code": None,
        "output_hashes": {},
        "transition_id": "b" * 64,
    }

    validator = artifact_schema_validator(schema)
    assert validator.is_valid(status)
    assert not validator.is_valid(status | {"exit_code": 1})
    assert validator.is_valid(
        status | {"state": "cancelled", "exit_code": 1}
    )


@pytest.mark.parametrize("path", ["/absolute.json", "metrics/../escape.json"])
def test_status_schema_and_python_reject_unsafe_output_hash_paths(path):
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/status.schema.json").read_text()
    )
    status = {
        "run_id": _manifest()["run_id"],
        "state": "completed",
        "timestamp_utc": "2026-08-15T00:00:00Z",
        "worker_pid": 1,
        "host": "fixture",
        "manifest_sha256": "a" * 64,
        "exit_code": 0,
        "output_hashes": {path: "b" * 64},
        "transition_id": "c" * 64,
    }

    assert not artifact_schema_validator(schema).is_valid(status)
    with pytest.raises(ValueError, match="output_hashes"):
        experiment_artifacts._validate_output_hashes(status["output_hashes"])


def test_registry_schema_and_python_validator_have_strict_selected_metric_contract():
    record = _reservation_record(_manifest())
    terminal = record | {
        "event": "completed",
        "state": "completed",
        "selected_validation": {
            "path": "metrics/selected_validation.json",
            "sha256": "a" * 64,
        },
        "summary_metrics": {"loss": 1.0},
    }
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/registry.schema.json").read_text()
    )
    validator = artifact_schema_validator(schema)

    assert validator.is_valid(terminal)
    assert experiment_artifacts.parse_registry_text(json.dumps(terminal) + "\n") == [terminal]
    for invalid in (
        terminal | {"state": "failed"},
        terminal | {"selected_validation": {"values": [1, 2, 3]}},
        terminal | {"summary_metrics": {"loss": float("inf")}},
        record | {"selected_validation": terminal["selected_validation"]},
        record | {"summary_metrics": {"premature": 1.0}},
    ):
        assert not validator.is_valid(invalid) or invalid["summary_metrics"] == {"loss": float("inf")}
        with pytest.raises(ValueError):
            experiment_artifacts.parse_registry_text(json.dumps(invalid) + "\n")


@pytest.mark.parametrize(
    "path", ["metrics//selected_validation.json", "metrics/./selected_validation.json"]
)
def test_registry_schema_and_python_reject_noncanonical_metric_paths(path):
    record = _reservation_record(_manifest()) | {
        "event": "completed",
        "state": "completed",
        "selected_validation": {"path": path, "sha256": "a" * 64},
        "summary_metrics": {"loss": 1.0},
    }
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/registry.schema.json").read_text()
    )

    assert not artifact_schema_validator(schema).is_valid(record)
    with pytest.raises(ValueError, match="selected_validation.path"):
        experiment_artifacts.parse_registry_text(json.dumps(record) + "\n")


def test_registry_schema_and_python_reject_unsafe_run_id():
    record = _reservation_record(_manifest()) | {"run_id": "bad/name"}
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/registry.schema.json").read_text()
    )

    assert not artifact_schema_validator(schema).is_valid(record)
    with pytest.raises(ValueError, match="run_id"):
        experiment_artifacts.parse_registry_text(json.dumps(record) + "\n")


def test_registry_schema_requires_correction_target_and_matching_event_state():
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/registry.schema.json").read_text()
    )
    validator = artifact_schema_validator(schema)
    reservation = _reservation_record(_manifest())
    assert not validator.is_valid(reservation | {"state": "completed"})
    assert not validator.is_valid(
        reservation | {"event": "correction", "state": "reserved"}
    )


def test_manifest_schema_rejects_duplicate_separators_and_nonfinal_test_access():
    schema = json.loads(Path("experiments/contextual-risk-v2/schemas/manifest.schema.json").read_text())
    validator = artifact_schema_validator(schema)
    duplicate_separator = _manifest(
        variant="reference--copy",
        run_id="msrvtt-baseline-reference--copy-seed000-20260814T020304Z-000000000000",
    )
    assert not validator.is_valid(duplicate_separator)
    assert not validator.is_valid(_manifest(test_authorized=True))


def test_portable_artifact_locator_is_optional_normalized_and_schema_valid():
    legacy = _manifest()
    legacy_digest = experiment_artifacts.manifest_sha256(legacy)
    located = legacy | {
        "artifact_locator": "artifacts/contextual-risk-v2/msrvtt/baseline/example"
    }

    assert "artifact_locator" not in validate_manifest(legacy)
    assert experiment_artifacts.manifest_sha256(legacy) == legacy_digest
    assert validate_manifest(located)["artifact_locator"] == located["artifact_locator"]
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/manifest.schema.json").read_text()
    )
    assert artifact_schema_validator(schema).is_valid(located)

    for invalid in ("/absolute/path", "artifacts/../escape", "artifacts//run", "./artifacts/run"):
        with pytest.raises(ValueError, match="artifact_locator"):
            validate_manifest(legacy | {"artifact_locator": invalid})


def test_registry_accepts_and_preserves_optional_artifact_locator(tmp_path):
    manifest = _manifest(
        artifact_locator="artifacts/contextual-risk-v2/msrvtt/baseline/example"
    )
    record = _reservation_record(manifest) | {
        "artifact_locator": manifest["artifact_locator"]
    }
    registry = tmp_path / "registry.jsonl"

    append_registry_records(registry, [record])

    assert load_registry(registry)[0]["artifact_locator"] == manifest["artifact_locator"]
    schema = json.loads(
        Path("experiments/contextual-risk-v2/schemas/registry.schema.json").read_text()
    )
    assert artifact_schema_validator(schema).is_valid(record)


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


def _write_run_local_manifest(manifest):
    artifact_path = Path(manifest["artifact_path"])
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_worker_requires_exact_committed_reservation(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = _manifest(
        artifact_path=str(tmp_path / "artifact"), git_sha=source_sha,
        run_id=(
            "msrvtt-baseline-reference-seed000-20260814T020304Z-"
            f"{source_sha[:12]}"
        ),
    )
    _write_run_local_manifest(manifest)
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    verify_committed_reservation(repo_root, REGISTRY, manifest)
    changed = {
        **manifest,
        "variant": "changed",
        "run_id": "msrvtt-baseline-changed-seed000-20260814T020304Z-000000000000",
    }
    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, changed)


@pytest.mark.parametrize("locator", [None, "artifacts/contextual-risk-v2/wrong"])
def test_worker_rejects_reservation_with_missing_or_mismatched_locator(
    tmp_path, locator
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    artifact_path = repo_root / "artifacts/contextual-risk-v2/msrvtt/baseline/run"
    manifest = _manifest(
        artifact_path=str(artifact_path), git_sha=source_sha,
        run_id=(
            "msrvtt-baseline-reference-seed000-20260814T020304Z-"
            f"{source_sha[:12]}"
        ),
        artifact_locator=artifact_path.relative_to(repo_root).as_posix(),
    )
    _write_run_local_manifest(manifest)
    record = _reservation_record(manifest)
    if locator is not None:
        record["artifact_locator"] = locator
    append_registry_records(repo_root / REGISTRY, [record])
    subprocess.run(["git", "add", REGISTRY], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "malformed reservation"], cwd=repo_root, check=True)

    with pytest.raises(ValueError, match="exact committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_worker_rejects_missing_or_uncommitted_reservation(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = _manifest(
        artifact_path=str(tmp_path / "artifact"), git_sha=source_sha,
        run_id=(
            "msrvtt-baseline-reference-seed000-20260814T020304Z-"
            f"{source_sha[:12]}"
        ),
    )
    _write_run_local_manifest(manifest)

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


def test_committed_reservation_rejects_run_local_manifest_mismatch(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    manifest = _manifest(artifact_path=str(tmp_path / "artifact"))
    _write_run_local_manifest(manifest)
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    changed_local_manifest = {
        **manifest,
        "config": {"model": {"changed": True}, "loss": {}, "optimizer": {}, "scheduler": {}, "evaluation": {}},
    }
    _write_run_local_manifest(changed_local_manifest)

    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_committed_reservation_rejects_mismatched_registry_authorization_metadata(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    manifest = _manifest(artifact_path=str(tmp_path / "artifact"))
    _write_run_local_manifest(manifest)
    record = _reservation_record(manifest)
    record["test_authorized"] = True
    append_registry_records(repo_root / REGISTRY, [record])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_committed_reservation_refuses_a_symlinked_run_local_manifest(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    artifact_path = tmp_path / "artifact"
    manifest = _manifest(artifact_path=str(artifact_path))
    artifact_path.mkdir()
    target = tmp_path / "other-manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    os.symlink(target, artifact_path / "manifest.json")
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    with pytest.raises(ValueError, match="run-local manifest"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_registry_correction_must_resolve_prior_same_run_record(tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _reservation_record(_manifest())
    append_registry_records(registry, [first])
    correction = {
        **first,
        "event": "correction",
        "state": "reserved",
        "supersedes_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="supersedes_sha256"):
        append_registry_records(registry, [correction])

    other = {**first, "run_id": "other-run", "event": "completed", "state": "completed"}
    other_registry = tmp_path / "other.jsonl"
    append_registry_records(other_registry, [other])
    wrong_run_correction = {
        **first,
        "event": "correction",
        "state": "reserved",
        "supersedes_sha256": experiment_artifacts.registry_record_sha256(other),
    }
    with pytest.raises(ValueError, match="same run"):
        append_registry_records(other_registry, [wrong_run_correction])


def test_registry_preserves_existing_raw_bytes_when_appending(tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _reservation_record(_manifest())
    raw_prefix = json.dumps(first, separators=(", ", ": "), sort_keys=False) + "\n"
    registry.write_text(raw_prefix, encoding="utf-8")
    second = {**first, "run_id": "a-second-run", "event": "completed", "state": "completed"}

    append_registry_records(registry, [second])

    assert registry.read_text(encoding="utf-8").startswith(raw_prefix)


@pytest.mark.parametrize("first_state", ["reserved", "running"])
def test_cancelled_transition_requires_exit_code(tmp_path, first_state):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)
    if first_state == "running":
        transition_run_status(paths, "reserved", "running")

    with pytest.raises(ValueError, match="exit_code"):
        transition_run_status(paths, first_state, "cancelled")


def _advance_to_validating(paths):
    transition_run_status(paths, "reserved", "running")
    transition_run_status(paths, "running", "validating")


def test_terminal_transition_retry_repairs_missing_commit_audit_after_status_write(tmp_path, monkeypatch):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)
    _advance_to_validating(paths)
    original_append = experiment_artifacts._append_audit_record

    def fail_terminal_commit(run_paths, record):
        if record.get("event") == "transition_commit":
            raise OSError("injected audit failure")
        return original_append(run_paths, record)

    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", fail_terminal_commit)
    hashes = {"metrics/selected_validation.json": "a" * 64}
    with pytest.raises(OSError, match="injected"):
        transition_run_status(paths, "validating", "completed", exit_code=0, output_hashes=hashes)
    assert json.loads(paths.status.read_text())["output_hashes"] == hashes

    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", original_append)
    transition_run_status(paths, "validating", "completed", exit_code=0, output_hashes=hashes)
    audit = (paths.provenance / "audit.jsonl").read_text(encoding="utf-8")
    assert '"event":"transition_commit"' in audit


def test_terminal_transition_retry_recovers_from_status_replace_failure(tmp_path, monkeypatch):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)
    _advance_to_validating(paths)
    original_write = experiment_artifacts._atomic_write_json

    def fail_terminal_status(path, payload):
        if path == paths.status and payload.get("state") == "completed":
            raise OSError("injected status failure")
        return original_write(path, payload)

    monkeypatch.setattr(experiment_artifacts, "_atomic_write_json", fail_terminal_status)
    with pytest.raises(OSError, match="injected"):
        transition_run_status(paths, "validating", "completed", exit_code=0)
    assert json.loads(paths.status.read_text())["state"] == "validating"

    monkeypatch.setattr(experiment_artifacts, "_atomic_write_json", original_write)
    transition_run_status(paths, "validating", "completed", exit_code=0)
    assert json.loads(paths.status.read_text())["state"] == "completed"


def test_reservation_retry_repairs_missing_audit_after_status_write(tmp_path, monkeypatch):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    original_append = experiment_artifacts._append_audit_record

    def fail_reservation_commit(run_paths, record):
        if record.get("event") == "reservation_commit":
            raise OSError("injected reservation audit failure")
        return original_append(run_paths, record)

    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", fail_reservation_commit)
    with pytest.raises(OSError, match="injected"):
        reserve_run_directory(paths, manifest)
    assert json.loads(paths.status.read_text())["state"] == "reserved"

    monkeypatch.setattr(experiment_artifacts, "_append_audit_record", original_append)
    reserve_run_directory(paths, manifest)
    audit = (paths.provenance / "audit.jsonl").read_text(encoding="utf-8")
    assert '"event":"reservation_commit"' in audit
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_run_directory(paths, manifest)


def test_reservation_rejects_symlinked_run_root_without_writing_outside(tmp_path):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.root.parent.mkdir(parents=True)
    os.symlink(outside, paths.root, target_is_directory=True)
    manifest = _manifest(artifact_path=str(paths.root))

    with pytest.raises(ValueError, match="safe run path"):
        reserve_run_directory(paths, manifest)

    assert list(outside.iterdir()) == []


def test_committed_verification_rejects_symlinked_run_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.root.parent.mkdir(parents=True)
    os.symlink(outside, paths.root, target_is_directory=True)
    manifest = _manifest(artifact_path=str(paths.root))
    (outside / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    append_registry_records(repo_root / REGISTRY, [_reservation_record(manifest)])
    subprocess.run(["git", "add", REGISTRY.as_posix()], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-qm", "reserve phase"], cwd=repo_root, check=True)

    with pytest.raises(ValueError, match="run-local manifest"):
        verify_committed_reservation(repo_root, REGISTRY, manifest)


def test_lifecycle_rejects_symlinked_run_child_and_audit_file(tmp_path):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.logs.rmdir()
    os.symlink(outside, paths.logs, target_is_directory=True)

    with pytest.raises(ValueError, match="safe run path"):
        transition_run_status(paths, "reserved", "running")

    paths.logs.unlink()
    paths.logs.mkdir()
    audit_target = outside / "audit.jsonl"
    audit_target.write_text("external evidence\n", encoding="utf-8")
    (paths.provenance / "audit.jsonl").unlink()
    os.symlink(audit_target, paths.provenance / "audit.jsonl")

    with pytest.raises(ValueError, match="safe run path"):
        transition_run_status(paths, "reserved", "running")
    assert audit_target.read_text(encoding="utf-8") == "external evidence\n"


def test_lifecycle_rejects_hardlinked_status_file(tmp_path):
    paths = make_run_paths(tmp_path, "contextual-risk-v2", _manifest()["run_id"])
    manifest = _manifest(artifact_path=str(paths.root))
    reserve_run_directory(paths, manifest)
    outside = tmp_path / "outside-status.json"
    outside.write_text(paths.status.read_text(encoding="utf-8"), encoding="utf-8")
    paths.status.unlink()
    os.link(outside, paths.status)

    with pytest.raises(ValueError, match="safe run path"):
        transition_run_status(paths, "reserved", "running")
    assert json.loads(outside.read_text(encoding="utf-8"))["state"] == "reserved"


def test_reservation_rejects_noncanonical_run_root_before_side_effects(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = base / "redirect"
    os.symlink(outside, redirect, target_is_directory=True)
    run_id = _manifest()["run_id"]
    unsafe_root = Path(f"{base}/missing/../redirect") / "contextual-risk-v2" / "msrvtt" / "baseline" / run_id
    paths = _paths_for_root(unsafe_root)
    manifest = _manifest(artifact_path=str(outside / "contextual-risk-v2" / "msrvtt" / "baseline" / run_id))

    with pytest.raises(ValueError, match="safe run path"):
        reserve_run_directory(paths, manifest)

    assert not (base / "missing").exists()
    assert list(outside.iterdir()) == []


def test_make_run_paths_rejects_relative_and_noncanonical_artifact_roots(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        make_run_paths("relative/artifacts", "contextual-risk-v2", _manifest()["run_id"])
    with pytest.raises(ValueError, match="canonical"):
        make_run_paths(tmp_path / "one" / ".." / "two", "contextual-risk-v2", _manifest()["run_id"])
    with pytest.raises(ValueError, match="canonical"):
        make_run_paths(f"{tmp_path}/./two", "contextual-risk-v2", _manifest()["run_id"])
