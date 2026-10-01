"""Large real galleries retain individual file identity through small bound indices."""

import hashlib
import json

import pytest


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_bound_index_expands_children_and_keeps_index_identity(tmp_path):
    from semantic_retrieval.runtime.input_bindings import manifest_input_references

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    index = tmp_path / "training.input-index.json"
    index.write_text(json.dumps({"schema_version": 1, "kind": "file_sha256_index", "files": [ref(video)]}))
    manifest = {"dataset_splits": [ref(index)], "input_checkpoints": []}
    assert manifest_input_references(tmp_path, manifest) == {
        str(index): ref(index)["sha256"],
        str(video): ref(video)["sha256"],
    }
    index.write_text(index.read_text() + " ")
    with pytest.raises(ValueError, match="index SHA"):
        manifest_input_references(tmp_path, manifest)


@pytest.mark.parametrize("failure", ["duplicate", "nested", "relative", "symlink"])
def test_index_rejects_ambiguous_or_unsafe_child_identity(tmp_path, failure):
    from semantic_retrieval.runtime.input_bindings import manifest_input_references

    child = tmp_path / ("nested.input-index.json" if failure == "nested" else "video.mp4")
    child.write_bytes(b"video")
    item = ref(child)
    if failure == "relative":
        item["path"] = "video.mp4"
    if failure == "symlink":
        link = tmp_path / "link.mp4"
        link.symlink_to(child)
        item["path"] = str(link)
    files = [item, item] if failure == "duplicate" else [item]
    index = tmp_path / "training.input-index.json"
    index.write_text(json.dumps({"schema_version": 1, "kind": "file_sha256_index", "files": files}))
    with pytest.raises(ValueError):
        manifest_input_references(tmp_path, {"dataset_splits": [ref(index)], "input_checkpoints": []})


def test_nine_thousand_inputs_fit_sealed_manifest_with_full_identity(tmp_path):
    from experiment_artifacts import artifact_json_bytes
    from semantic_retrieval.runtime.input_bindings import manifest_input_references, write_input_indices

    refs = [{"path": str(tmp_path / f"video{i}.mp4"), "sha256": "a" * 64} for i in range(9000)]
    indices = write_input_indices(tmp_path, "train", refs)
    manifest = {"dataset_splits": indices, "input_checkpoints": []}
    assert len(indices) == 10
    assert len(artifact_json_bytes(manifest)) < 1024 * 1024
    resolved = manifest_input_references(tmp_path, manifest)
    assert all(resolved[r["path"]] == r["sha256"] for r in refs)
    with pytest.raises(FileExistsError):
        write_input_indices(tmp_path, "train", refs)
