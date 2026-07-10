import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from dataloaders.msrvtt_protocol import (
    build_trusted_manifest,
    load_trusted_manifest,
    sha256_file,
    validate_trusted_manifest,
    write_generated_split_files,
    write_trusted_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_msrvtt_trusted_split.py"
COMMITTED_MANIFEST = (
    PROJECT_ROOT / "dataloaders" / "splits" / "msrvtt_trusted_v1_seed42.json"
)
REAL_TRAIN_CSV = Path("/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv")
REAL_ANNOTATION_JSON = Path("/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json")
REAL_TEST_CSV = Path("/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv")
REAL_SOURCES_AVAILABLE = all(
    path.is_file() for path in (REAL_TRAIN_CSV, REAL_ANNOTATION_JSON, REAL_TEST_CSV)
)
COMMITTED_MANIFEST_SHA256 = (
    "b0a9127a23514d15a1b4e3d702faaceb50d10aaa32bc1de5fd8f27c10a57b611"
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _snapshot_files(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            path.stat().st_size,
            sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def trusted_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("trusted_sources")
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    annotation_json = root / "annotation.json"
    _write_csv(
        train_csv,
        ["video_id"],
        ({"video_id": f"video{i}"} for i in range(9000)),
    )
    _write_csv(
        test_csv,
        ["key", "video_id", "sentence"],
        (
            {
                "key": f"ret{i}",
                "video_id": f"video{i + 9000}",
                "sentence": f"test caption {i}",
            }
            for i in range(1000)
        ),
    )
    payload = {
        "videos": [{"video_id": f"video{i}"} for i in range(10000)],
        "sentences": [
            {
                "video_id": f"video{i}",
                "caption": f"caption {i}-{caption_index}",
            }
            for i in range(10000)
            for caption_index in range(20)
        ],
    }
    annotation_json.write_text(json.dumps(payload), encoding="utf-8")
    return train_csv, annotation_json, test_csv


@pytest.fixture(scope="module")
def trusted_manifest(trusted_sources):
    return build_trusted_manifest(*trusted_sources)


def test_trusted_split_is_deterministic_and_has_exact_counts(
    trusted_sources, trusted_manifest
):
    assert build_trusted_manifest(*trusted_sources) == trusted_manifest
    assert trusted_manifest["protocol_version"] == "trusted-v1"
    assert trusted_manifest["seed"] == 42
    assert trusted_manifest["counts"] == {
        "source_train_videos": 9000,
        "train_videos": 8500,
        "val_videos": 500,
        "val_sentences": 10000,
        "test_csv_rows": 1000,
        "test_videos": 1000,
    }
    assert trusted_manifest["overlap_counts"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    train_ids = set(trusted_manifest["train_video_ids"])
    val_ids = set(trusted_manifest["val_video_ids"])
    assert len(train_ids) == 8500
    assert len(val_ids) == 500
    assert train_ids.isdisjoint(val_ids)


def test_generated_val_contains_twenty_grouped_captions_per_video(
    tmp_path, trusted_sources, trusted_manifest
):
    paths = write_generated_split_files(
        trusted_manifest, trusted_sources[1], tmp_path / "generated"
    )
    with Path(paths["train_csv"]).open(encoding="utf-8", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    with Path(paths["val_csv"]).open(encoding="utf-8", newline="") as handle:
        val_rows = list(csv.DictReader(handle))
    assert len(train_rows) == 8500
    assert [row["video_id"] for row in train_rows] == trusted_manifest[
        "train_video_ids"
    ]
    assert len(val_rows) == 10000
    assert Counter(row["video_id"] for row in val_rows) == {
        video_id: 20 for video_id in trusted_manifest["val_video_ids"]
    }
    annotation = json.loads(trusted_sources[1].read_text(encoding="utf-8"))
    captions_by_video = {}
    for sentence in annotation["sentences"]:
        captions_by_video.setdefault(sentence["video_id"], []).append(
            sentence["caption"]
        )
    expected_val_rows = [
        (video_id, caption)
        for video_id in trusted_manifest["val_video_ids"]
        for caption in captions_by_video[video_id]
    ]
    assert [(row["video_id"], row["sentence"]) for row in val_rows] == (
        expected_val_rows
    )


def test_test_csv_rejects_duplicate_video_ids_with_path_and_lines(
    tmp_path, trusted_sources
):
    train_csv, annotation_json, _ = trusted_sources
    test_csv = tmp_path / "duplicate_test_ids.csv"
    _write_csv(
        test_csv,
        ["video_id", "sentence"],
        [
            {
                "video_id": "video9000",
                "sentence": "first duplicate caption",
            },
            {
                "video_id": "video9000",
                "sentence": "second duplicate caption",
            },
            *(
                {
                    "video_id": f"video{i}",
                    "sentence": f"test caption {i}",
                }
                for i in range(9001, 10000)
            ),
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        build_trusted_manifest(train_csv, annotation_json, test_csv)
    message = str(exc_info.value)
    assert str(test_csv) in message
    assert "duplicate video_id" in message
    assert "video9000" in message
    assert "CSV lines 2, 3" in message


def test_train_video_list_rejects_duplicate_ids(tmp_path, trusted_sources):
    _, annotation_json, test_csv = trusted_sources
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        ["video_id"],
        ({"video_id": "video0"} for _ in range(9000)),
    )
    with pytest.raises(ValueError, match=r"duplicate video_id.*video0"):
        build_trusted_manifest(train_csv, annotation_json, test_csv)


def test_validation_rejects_train_test_overlap(tmp_path, trusted_sources):
    train_csv, annotation_json, _ = trusted_sources
    test_csv = tmp_path / "test.csv"
    _write_csv(
        test_csv,
        ["video_id", "sentence"],
        (
            {
                "video_id": "video0" if i == 0 else f"video{i + 9000}",
                "sentence": f"test {i}",
            }
            for i in range(1000)
        ),
    )
    with pytest.raises(ValueError, match=r"train/test overlap.*video0"):
        build_trusted_manifest(train_csv, annotation_json, test_csv)


def test_validation_rejects_missing_caption(tmp_path, trusted_sources):
    train_csv, _, test_csv = trusted_sources
    annotation_json = tmp_path / "annotation.json"
    annotation_json.write_text(
        json.dumps(
            {
                "videos": [{"video_id": f"video{i}"} for i in range(10000)],
                "sentences": [
                    {"video_id": "video0", "caption": f"caption {i}"}
                    for i in range(19)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"video_id=video0 expected 20 captions"):
        build_trusted_manifest(train_csv, annotation_json, test_csv)


@pytest.mark.parametrize("missing_key", ["videos", "sentences"])
def test_annotation_missing_required_key_has_clear_error(
    tmp_path, trusted_sources, missing_key
):
    train_csv, _, test_csv = trusted_sources
    annotation_json = tmp_path / f"missing_{missing_key}.json"
    payload = {"videos": [], "sentences": []}
    del payload[missing_key]
    annotation_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"annotation.*missing.*{missing_key}"):
        build_trusted_manifest(train_csv, annotation_json, test_csv)


def test_csv_missing_video_id_header_has_clear_error(tmp_path, trusted_sources):
    _, annotation_json, test_csv = trusted_sources
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("id\nvideo0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required column 'video_id'"):
        build_trusted_manifest(train_csv, annotation_json, test_csv)


def test_manifest_validation_reports_missing_and_malformed_keys(
    trusted_sources, trusted_manifest
):
    missing = dict(trusted_manifest)
    missing.pop("protocol_version")
    with pytest.raises(ValueError, match=r"manifest missing required key.*protocol_version"):
        validate_trusted_manifest(missing, *trusted_sources)

    malformed = dict(trusted_manifest)
    malformed["val_video_ids"] = "not-a-list"
    with pytest.raises(ValueError, match=r"manifest key 'val_video_ids'.*list"):
        validate_trusted_manifest(malformed, *trusted_sources)


def test_manifest_validation_detects_source_or_content_drift(
    trusted_sources, trusted_manifest
):
    changed = json.loads(json.dumps(trusted_manifest))
    changed["source_sha256"]["train_csv"] = "0" * 64
    with pytest.raises(ValueError, match=r"manifest mismatch.*source_sha256"):
        validate_trusted_manifest(changed, *trusted_sources)


def test_manifest_validation_rejects_noncanonical_protocol_parameters(trusted_sources):
    noncanonical = build_trusted_manifest(*trusted_sources, seed=43)
    with pytest.raises(ValueError, match=r"trusted-v1 requires seed=42"):
        validate_trusted_manifest(noncanonical, *trusted_sources)


def test_check_only_validates_without_writing_derived_files(
    tmp_path, trusted_sources, trusted_manifest
):
    train_csv, annotation_json, test_csv = trusted_sources
    managed_root = tmp_path / "managed"
    manifest_path = managed_root / "manifest.json"
    output_dir = managed_root / "generated"
    outside_cwd = tmp_path / "outside-repository"
    outside_cwd.mkdir()
    write_trusted_manifest(manifest_path, trusted_manifest)
    write_generated_split_files(trusted_manifest, annotation_json, output_dir)
    (output_dir / "sentinel.keep").write_text("preserve me\n", encoding="utf-8")
    for index, path in enumerate(
        sorted(path for path in managed_root.rglob("*") if path.is_file())
    ):
        timestamp_ns = 1_600_000_000_000_000_000 + index
        os.utime(path, ns=(timestamp_ns, timestamp_ns))
    before = _snapshot_files(managed_root)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-csv",
            str(train_csv),
            "--annotation-json",
            str(annotation_json),
            "--test-csv",
            str(test_csv),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--check-only",
        ],
        cwd=outside_cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "trusted-v1 validated: train=8500 val=500 test=1000" in result.stdout
    assert _snapshot_files(managed_root) == before


@pytest.mark.skipif(
    not REAL_SOURCES_AVAILABLE,
    reason="canonical MSRVTT source files are unavailable",
)
def test_committed_manifest_validates_against_canonical_sources():
    manifest = load_trusted_manifest(COMMITTED_MANIFEST)
    summary = validate_trusted_manifest(
        manifest,
        REAL_TRAIN_CSV,
        REAL_ANNOTATION_JSON,
        REAL_TEST_CSV,
    )
    assert summary["manifest_sha256"] == COMMITTED_MANIFEST_SHA256
    assert summary["train_videos"] == 8500
    assert summary["val_videos"] == 500
    assert summary["val_sentences"] == 10000
    assert summary["test_videos"] == 1000
