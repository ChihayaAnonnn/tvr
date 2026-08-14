from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from scripts import inventory_legacy_artifacts as inventory
from scripts import migrate_critical_artifacts as migration


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _inventory_record(root: Path, path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "schema_version": 1,
        "root": str(root),
        "relative_path": path.relative_to(root).as_posix(),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "classification": "historical",
        "state": "completed",
    }


def test_inventory_resumes_after_a_verified_record_and_truncated_tail(tmp_path):
    root = tmp_path / "logs"
    first = _write(root / "a.log", b"first")
    second = _write(root / "b.log", b"second")
    output = tmp_path / "artifacts" / "inventory.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(_inventory_record(root, first), sort_keys=True)
        + "\n"
        + '{"relative_path": "interrupted',
        encoding="utf-8",
    )

    summary = inventory.inventory_legacy_artifacts(
        [root], output, tmp_path / "inventory-summary.json"
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["relative_path"] for record in records] == ["a.log", "b.log"]
    assert records[1]["sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()
    assert summary["completed_files"] == 2
    assert summary["completed_bytes"] == len(b"firstsecond")


def test_inventory_refuses_resume_when_last_completed_record_changed(tmp_path):
    root = tmp_path / "logs"
    source = _write(root / "a.log", b"before")
    output = tmp_path / "inventory.jsonl"
    output.write_text(
        json.dumps(_inventory_record(root, source), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source.write_bytes(b"after!")

    with pytest.raises(ValueError, match="last verified inventory record"):
        inventory.inventory_legacy_artifacts(
            [root], output, tmp_path / "summary.json"
        )


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_inventory_refuses_symlinks(tmp_path, kind):
    root = tmp_path / "logs"
    root.mkdir()
    outside = _write(tmp_path / "outside" / "payload", b"secret")
    if kind == "file":
        (root / "link").symlink_to(outside)
    else:
        (root / "link").symlink_to(outside.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        inventory.inventory_legacy_artifacts(
            [root], tmp_path / "inventory.jsonl", tmp_path / "summary.json"
        )


def test_inventory_refuses_hardlinked_files(tmp_path):
    root = tmp_path / "ckpts"
    source = _write(root / "model.bin", b"weights")
    os.link(source, root / "duplicate.bin")

    with pytest.raises(ValueError, match="hardlink"):
        inventory.inventory_legacy_artifacts(
            [root], tmp_path / "inventory.jsonl", tmp_path / "summary.json"
        )


def test_inventory_detects_source_mutation_during_hash(tmp_path, monkeypatch):
    root = tmp_path / "ckpts"
    source = _write(root / "model.bin", b"before")
    original = inventory._hash_fd

    def mutate_after_hash(fd, chunk_size):
        digest = original(fd, chunk_size)
        source.write_bytes(b"after!")
        return digest

    monkeypatch.setattr(inventory, "_hash_fd", mutate_after_hash)
    with pytest.raises(ValueError, match="changed while hashing"):
        inventory.inventory_legacy_artifacts(
            [root], tmp_path / "inventory.jsonl", tmp_path / "summary.json"
        )


def test_inventory_classifies_whitelisted_sources_as_critical(tmp_path):
    root = tmp_path / "ckpts"
    _write(root / "selected" / "model.bin", b"weights")
    output = tmp_path / "inventory.jsonl"

    inventory.inventory_legacy_artifacts(
        [root],
        output,
        tmp_path / "summary.json",
        classifications={str(root / "selected"): "critical"},
    )

    record = json.loads(output.read_text().strip())
    assert record["classification"] == "critical"


def test_inventory_summary_uses_portable_path_inside_current_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "logs"
    _write(root / "run.log", b"evidence")

    summary = inventory.inventory_legacy_artifacts(
        [root],
        tmp_path / "artifacts" / "contextual-risk-v2" / "legacy" / "inventory.jsonl",
        tmp_path / "inventory-summary.json",
    )

    assert summary["inventory_path"] == (
        "artifacts/contextual-risk-v2/legacy/inventory.jsonl"
    )


def test_migration_copies_and_verifies_without_deleting_source(tmp_path):
    source = tmp_path / "ckpts" / "legacy-run"
    _write(source / "model.bin", b"weights")
    _write(source / "metrics.json", b'{"r1":52.9}')

    result = migration.migrate_critical_run(
        source,
        tmp_path / "artifacts",
        move=False,
        destination="msrvtt/baseline/legacy-run/checkpoints",
    )

    assert source.exists()
    assert result.destination.exists()
    assert result.source_sha256 == result.destination_sha256
    assert result.file_count == 2
    assert result.byte_count == len(b"weights") + len(b'{"r1":52.9}')


def test_migration_is_idempotent_but_fails_closed_on_collision(tmp_path):
    source = tmp_path / "ckpts" / "legacy-run"
    _write(source / "model.bin", b"weights")
    artifact_root = tmp_path / "artifacts"
    kwargs = {
        "move": False,
        "destination": "msrvtt/baseline/legacy-run/checkpoints",
    }
    first = migration.migrate_critical_run(source, artifact_root, **kwargs)
    second = migration.migrate_critical_run(source, artifact_root, **kwargs)
    assert second.source_sha256 == first.destination_sha256

    (first.destination / "model.bin").write_bytes(b"different")
    with pytest.raises(FileExistsError, match="destination collision"):
        migration.migrate_critical_run(source, artifact_root, **kwargs)


def test_migration_fails_closed_when_destination_tree_has_an_extra_file(tmp_path):
    source = tmp_path / "ckpts" / "legacy-run"
    _write(source / "model.bin", b"weights")
    artifact_root = tmp_path / "artifacts"
    destination = "msrvtt/baseline/legacy-run/checkpoints"
    result = migration.migrate_critical_run(
        source, artifact_root, destination=destination
    )
    _write(result.destination / "unexpected.bin", b"foreign")

    with pytest.raises(FileExistsError, match="destination collision"):
        migration.migrate_critical_run(
            source, artifact_root, destination=destination
        )


def test_migration_dry_run_has_no_side_effects(tmp_path):
    source = tmp_path / "ckpts" / "legacy-run"
    _write(source / "model.bin", b"weights")
    artifact_root = tmp_path / "artifacts"

    result = migration.migrate_critical_run(
        source,
        artifact_root,
        destination="msrvtt/baseline/legacy-run/checkpoints",
        dry_run=True,
    )

    assert result.dry_run is True
    assert not artifact_root.exists()
    assert source.read_bytes() if source.is_file() else source.exists()


@pytest.mark.parametrize("destination", ["../escape", "/absolute", "a/../../b"])
def test_migration_rejects_destination_path_escape(tmp_path, destination):
    source = _write(tmp_path / "ckpts" / "model.bin", b"weights")
    with pytest.raises(ValueError, match="relative.*traversal"):
        migration.migrate_critical_run(
            source, tmp_path / "artifacts", destination=destination
        )


def test_migration_refuses_source_symlinks_and_hardlinks(tmp_path):
    source = tmp_path / "ckpts" / "legacy-run"
    payload = _write(source / "model.bin", b"weights")
    (source / "link.bin").symlink_to(payload)
    with pytest.raises(ValueError, match="symlink"):
        migration.migrate_critical_run(source, tmp_path / "artifacts")
    (source / "link.bin").unlink()
    os.link(payload, source / "hardlink.bin")
    with pytest.raises(ValueError, match="hardlink"):
        migration.migrate_critical_run(source, tmp_path / "artifacts")


def test_migration_rejects_identical_destination_below_symlink_parent(tmp_path):
    source = _write(tmp_path / "ckpts" / "train.log", b"same")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside"
    _write(outside / "train.log", b"same")
    (artifact_root / "logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|unsafe artifact"):
        migration.migrate_critical_run(
            source, artifact_root, destination="logs/train.log"
        )


def test_migration_rejects_identical_hardlinked_destination(tmp_path):
    source = _write(tmp_path / "ckpts" / "model.bin", b"same")
    artifact_root = tmp_path / "artifacts"
    destination = _write(artifact_root / "checkpoints" / "model.bin", b"same")
    os.link(destination, tmp_path / "outside-hardlink.bin")

    with pytest.raises(FileExistsError, match="destination collision"):
        migration.migrate_critical_run(
            source, artifact_root, destination="checkpoints/model.bin"
        )


def test_provenance_publication_rejects_symlinked_parent(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / "dataset" / "stage" / "run"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = {"schema_version": 1, "operation": "copy-plus-verify"}
    (outside / "legacy-migration.json").write_bytes(migration._json_bytes(payload))
    (run_root / "provenance").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|unsafe artifact"):
        migration._publish_json_equal_or_fail(
            artifact_root,
            Path("dataset/stage/run/provenance/legacy-migration.json"),
            payload,
        )


def test_migration_detects_source_mutation_during_copy(tmp_path, monkeypatch):
    source = _write(tmp_path / "ckpts" / "model.bin", b"before")
    original = migration._copy_open_file

    def mutate_after_copy(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(b"after!")
        return result

    monkeypatch.setattr(migration, "_copy_open_file", mutate_after_copy)
    with pytest.raises(ValueError, match="changed while copying"):
        migration.migrate_critical_run(source, tmp_path / "artifacts")


def test_migration_rehashes_all_sources_after_multifile_copy(tmp_path, monkeypatch):
    source = tmp_path / "ckpts" / "legacy-run"
    first = _write(source / "a.bin", b"first")
    _write(source / "b.bin", b"second")
    original = migration._copy_open_file

    def mutate_earlier_file_during_later_copy(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == "b.bin":
            first.write_bytes(b"FIRST")
        return result

    monkeypatch.setattr(migration, "_copy_open_file", mutate_earlier_file_during_later_copy)
    with pytest.raises(ValueError, match="changed after copying"):
        migration.migrate_critical_run(
            source, tmp_path / "artifacts", destination="checkpoints"
        )
    assert (tmp_path / "artifacts" / "checkpoints" / "a.bin").exists()


def test_migration_fsyncs_parents_for_new_directory_chain(tmp_path, monkeypatch):
    source = _write(tmp_path / "ckpts" / "model.bin", b"weights")
    original = migration.os.fsync
    directory_fsyncs = 0

    def count_directory_fsyncs(fd):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fsyncs += 1
        return original(fd)

    monkeypatch.setattr(migration.os, "fsync", count_directory_fsyncs)
    migration.migrate_critical_run(
        source,
        tmp_path / "artifacts",
        destination="dataset/stage/run/checkpoints/model.bin",
    )

    assert directory_fsyncs >= 6


def test_mapping_records_unresolved_run_and_copies_only_resolved_whitelist(tmp_path):
    source_root = tmp_path / "repo"
    _write(source_root / "ckpts" / "a1" / "model.bin", b"weights")
    mapping = tmp_path / "critical-runs.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "name": "msrvtt-a1",
                        "dataset": "msrvtt",
                        "stage": "baseline",
                        "run_id": "legacy-msrvtt-a1",
                        "classification": "critical",
                        "resolution": "resolved",
                        "sources": [
                            {
                                "path": "ckpts/a1",
                                "destination": "checkpoints",
                            }
                        ],
                    },
                    {
                        "name": "msvd-a1",
                        "dataset": "msvd",
                        "stage": "baseline",
                        "run_id": None,
                        "classification": "unresolved",
                        "resolution": "unresolved",
                        "sources": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = migration.migrate_mapping(
        mapping, tmp_path / "artifacts", source_root=source_root
    )

    assert report["resolved_runs"] == 1
    assert report["unresolved_runs"] == ["msvd-a1"]
    copied = tmp_path / "artifacts" / "msrvtt" / "baseline" / "legacy-msrvtt-a1"
    assert (copied / "checkpoints" / "model.bin").read_bytes() == b"weights"
    provenance = json.loads(
        (copied / "provenance" / "legacy-migration.json").read_text()
    )
    assert provenance["source_sha256"] == provenance["destination_sha256"]


def test_mapping_rejects_source_path_escape(tmp_path):
    mapping = tmp_path / "critical-runs.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "name": "bad",
                        "dataset": "msrvtt",
                        "stage": "baseline",
                        "run_id": "legacy-bad",
                        "classification": "critical",
                        "resolution": "resolved",
                        "sources": [
                            {"path": "../secret", "destination": "checkpoints"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source path.*traversal"):
        migration.migrate_mapping(
            mapping, tmp_path / "artifacts", source_root=tmp_path / "repo"
        )
