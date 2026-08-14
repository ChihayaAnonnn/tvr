import json
from pathlib import Path

import pytest

from experiment_artifacts import (
    ALLOWED_STAGES,
    ALLOWED_STATES,
    build_run_id,
    make_run_paths,
    validate_manifest,
)


def _manifest(**overrides):
    payload = {
        "run_id": "msrvtt-baseline-reference-seed000-20260814T020304Z-0123456789ab",
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


def test_manifest_requires_parent_lineage_for_derived_runs():
    derived_run_id = "msrvtt-feature_ablation-reference-seed000-20260814T020304Z-0123456789ab"
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
