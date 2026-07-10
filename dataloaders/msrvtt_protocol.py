"""Build and validate the versioned MSRVTT trusted-v1 data split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

PROTOCOL_VERSION = "trusted-v1"
SOURCE_TRAIN_VIDEO_COUNT = 9000
TEST_VIDEO_COUNT = 1000
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 500
DEFAULT_CAPTIONS_PER_VIDEO = 20
ALGORITHM = (
    "sort unique source train video_id strings; shuffle with an isolated "
    "random.Random(seed); first val_size IDs are val and the remainder are train"
)

_MANIFEST_KEYS = {
    "protocol_version",
    "seed",
    "algorithm",
    "val_size",
    "expected_captions_per_video",
    "source_sha256",
    "counts",
    "overlap_counts",
    "test_video_ids_sha256",
    "train_video_ids",
    "val_video_ids",
}
_SOURCE_HASH_KEYS = {"train_csv", "annotation_json", "test_csv"}
_COUNT_KEYS = {
    "source_train_videos",
    "train_videos",
    "val_videos",
    "val_sentences",
    "test_csv_rows",
    "test_videos",
}
_OVERLAP_KEYS = {"train_val", "train_test", "val_test"}


def sha256_file(path):
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_ids(path):
    """Read a CSV whose video IDs must be non-empty and unique."""

    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise ValueError(f"CSV {path} missing required column 'video_id'")
        ids = []
        first_line_by_id = {}
        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            raw_video_id = row.get("video_id")
            video_id = raw_video_id.strip() if isinstance(raw_video_id, str) else ""
            if not video_id:
                raise ValueError(f"empty video_id in {path} at CSV line {line_number}")
            first_line = first_line_by_id.get(video_id)
            if first_line is not None:
                raise ValueError(
                    f"duplicate video_id in CSV {path}: {video_id!r} at "
                    f"CSV lines {first_line}, {line_number}"
                )
            first_line_by_id[video_id] = line_number
            ids.append(video_id)
    return ids, row_count


def _load_annotation(annotation_json):
    path = Path(annotation_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"annotation JSON is malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"annotation JSON root must be an object: {path}")
    for key in ("videos", "sentences"):
        if key not in payload:
            raise ValueError(f"annotation JSON {path} missing required key '{key}'")
        if not isinstance(payload[key], list):
            raise ValueError(f"annotation key '{key}' must be a list in {path}")
    return payload


def _annotation_index(annotation_json):
    payload = _load_annotation(annotation_json)
    video_ids = []
    for index, row in enumerate(payload["videos"]):
        if not isinstance(row, dict):
            raise ValueError(f"annotation videos[{index}] must be an object")
        raw_video_id = row.get("video_id")
        video_id = raw_video_id.strip() if isinstance(raw_video_id, str) else ""
        if not video_id:
            raise ValueError(f"annotation videos[{index}] has empty video_id")
        video_ids.append(video_id)
    duplicates = sorted(
        video_id for video_id, count in Counter(video_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"annotation videos contains duplicate video_id: {duplicates[:5]}")

    known_video_ids = set(video_ids)
    captions = defaultdict(list)
    for index, row in enumerate(payload["sentences"]):
        if not isinstance(row, dict):
            raise ValueError(f"annotation sentences[{index}] must be an object")
        raw_video_id = row.get("video_id")
        video_id = raw_video_id.strip() if isinstance(raw_video_id, str) else ""
        caption = row.get("caption")
        if not video_id:
            raise ValueError(f"annotation sentences[{index}] has empty video_id")
        if video_id not in known_video_ids:
            raise ValueError(
                f"annotation sentences[{index}] references unknown video_id={video_id}"
            )
        if not isinstance(caption, str):
            raise ValueError(
                f"annotation sentences[{index}] caption must be a string"
            )
        captions[video_id].append(caption)
    return video_ids, captions


def _raise_overlap(left_name, left_ids, right_name, right_ids):
    overlap = sorted(set(left_ids) & set(right_ids))
    if overlap:
        raise ValueError(
            f"{left_name}/{right_name} overlap count={len(overlap)} "
            f"examples={overlap[:5]}"
        )


def _validate_build_parameters(seed, val_size, expected_captions):
    for name, value in (
        ("seed", seed),
        ("val_size", val_size),
        ("expected_captions", expected_captions),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if not 0 < val_size < SOURCE_TRAIN_VIDEO_COUNT:
        raise ValueError(
            f"val_size must be between 1 and {SOURCE_TRAIN_VIDEO_COUNT - 1}, "
            f"got {val_size}"
        )
    if expected_captions <= 0:
        raise ValueError(
            f"expected_captions must be positive, got {expected_captions}"
        )


def build_trusted_manifest(
    train_csv,
    annotation_json,
    test_csv,
    seed=DEFAULT_SEED,
    val_size=DEFAULT_VAL_SIZE,
    expected_captions=DEFAULT_CAPTIONS_PER_VIDEO,
):
    """Build the deterministic trusted-v1 manifest from official sources."""

    _validate_build_parameters(seed, val_size, expected_captions)
    train_source_ids, train_row_count = _read_csv_ids(train_csv)
    test_ids, test_row_count = _read_csv_ids(test_csv)
    if train_row_count != SOURCE_TRAIN_VIDEO_COUNT:
        raise ValueError(
            f"expected {SOURCE_TRAIN_VIDEO_COUNT} source train CSV rows, "
            f"got {train_row_count}"
        )
    if len(train_source_ids) != SOURCE_TRAIN_VIDEO_COUNT:
        raise ValueError(
            f"expected {SOURCE_TRAIN_VIDEO_COUNT} source train videos, "
            f"got {len(train_source_ids)}"
        )
    if len(test_ids) != TEST_VIDEO_COUNT:
        raise ValueError(
            f"expected {TEST_VIDEO_COUNT} unique test videos, got {len(test_ids)} "
            f"from {test_row_count} CSV rows"
        )
    _raise_overlap("train", train_source_ids, "test", test_ids)

    annotation_video_ids, captions = _annotation_index(annotation_json)
    expected_annotation_ids = set(train_source_ids) | set(test_ids)
    actual_annotation_ids = set(annotation_video_ids)
    if actual_annotation_ids != expected_annotation_ids:
        missing = sorted(expected_annotation_ids - actual_annotation_ids)
        extra = sorted(actual_annotation_ids - expected_annotation_ids)
        raise ValueError(
            "annotation video IDs do not match train/test sources: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for video_id in train_source_ids:
        count = len(captions.get(video_id, []))
        if count != expected_captions:
            raise ValueError(
                f"video_id={video_id} expected {expected_captions} captions, "
                f"got {count}"
            )

    shuffled = sorted(train_source_ids)
    random.Random(seed).shuffle(shuffled)
    val_ids = shuffled[:val_size]
    train_ids = shuffled[val_size:]
    _raise_overlap("train", train_ids, "val", val_ids)
    _raise_overlap("train", train_ids, "test", test_ids)
    _raise_overlap("val", val_ids, "test", test_ids)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "algorithm": ALGORITHM,
        "val_size": val_size,
        "expected_captions_per_video": expected_captions,
        "source_sha256": {
            "train_csv": sha256_file(train_csv),
            "annotation_json": sha256_file(annotation_json),
            "test_csv": sha256_file(test_csv),
        },
        "counts": {
            "source_train_videos": len(train_source_ids),
            "train_videos": len(train_ids),
            "val_videos": len(val_ids),
            "val_sentences": len(val_ids) * expected_captions,
            "test_csv_rows": test_row_count,
            "test_videos": len(test_ids),
        },
        "overlap_counts": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "test_video_ids_sha256": hashlib.sha256(
            ("\n".join(test_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "train_video_ids": train_ids,
        "val_video_ids": val_ids,
    }


def _require_mapping_keys(value, name, expected_keys):
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest key '{name}' must be an object")
    missing = sorted(expected_keys - set(value))
    extra = sorted(set(value) - expected_keys)
    if missing:
        raise ValueError(f"manifest key '{name}' missing required keys: {missing}")
    if extra:
        raise ValueError(f"manifest key '{name}' has unexpected keys: {extra}")


def _validate_manifest_shape(manifest):
    if not isinstance(manifest, Mapping):
        raise ValueError("trusted split manifest root must be an object")
    missing = sorted(_MANIFEST_KEYS - set(manifest))
    extra = sorted(set(manifest) - _MANIFEST_KEYS)
    if missing:
        raise ValueError(f"manifest missing required key(s): {missing}")
    if extra:
        raise ValueError(f"manifest has unexpected key(s): {extra}")

    for name in ("protocol_version", "algorithm", "test_video_ids_sha256"):
        if not isinstance(manifest[name], str):
            raise ValueError(f"manifest key '{name}' must be a string")
    for name in ("seed", "val_size", "expected_captions_per_video"):
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"manifest key '{name}' must be an integer")
    for name in ("train_video_ids", "val_video_ids"):
        value = manifest[name]
        if not isinstance(value, list):
            raise ValueError(f"manifest key '{name}' must be a list")
        if any(not isinstance(video_id, str) or not video_id for video_id in value):
            raise ValueError(f"manifest key '{name}' must contain non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError(f"manifest key '{name}' contains duplicate video IDs")

    _require_mapping_keys(manifest["source_sha256"], "source_sha256", _SOURCE_HASH_KEYS)
    _require_mapping_keys(manifest["counts"], "counts", _COUNT_KEYS)
    _require_mapping_keys(manifest["overlap_counts"], "overlap_counts", _OVERLAP_KEYS)
    for name, digest in manifest["source_sha256"].items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                f"manifest source_sha256 key '{name}' must be a 64-character string"
            )
    for section in ("counts", "overlap_counts"):
        for name, value in manifest[section].items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"manifest {section} key '{name}' must be a non-negative integer"
                )


def _canonical_json_bytes(payload):
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _manifest_sha256(manifest):
    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def validate_trusted_manifest(manifest, train_csv, annotation_json, test_csv):
    """Rebuild the manifest and reject any schema, source, or split drift."""

    _validate_manifest_shape(manifest)
    required_values = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": DEFAULT_SEED,
        "val_size": DEFAULT_VAL_SIZE,
        "expected_captions_per_video": DEFAULT_CAPTIONS_PER_VIDEO,
    }
    for key, required_value in required_values.items():
        if manifest[key] != required_value:
            raise ValueError(
                f"trusted-v1 requires {key}={required_value!r}, "
                f"got {manifest[key]!r}"
            )
    expected = build_trusted_manifest(
        train_csv,
        annotation_json,
        test_csv,
        seed=manifest["seed"],
        val_size=manifest["val_size"],
        expected_captions=manifest["expected_captions_per_video"],
    )
    for key in sorted(_MANIFEST_KEYS):
        if manifest[key] != expected[key]:
            raise ValueError(
                f"trusted split manifest mismatch at '{key}'; "
                "manifest does not match current source files or protocol"
            )
    return {
        "protocol_version": manifest["protocol_version"],
        "seed": manifest["seed"],
        "source_sha256": dict(manifest["source_sha256"]),
        **manifest["counts"],
        **manifest["overlap_counts"],
        "manifest_sha256": _manifest_sha256(manifest),
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(_canonical_json_bytes(payload))
    os.replace(temporary_path, path)


def _atomic_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def write_generated_split_files(manifest, annotation_json, output_dir):
    """Write reproducible train/val CSVs and a validation summary."""

    _validate_manifest_shape(manifest)
    _, captions = _annotation_index(annotation_json)
    expected_captions = manifest["expected_captions_per_video"]
    for video_id in manifest["val_video_ids"]:
        count = len(captions.get(video_id, []))
        if count != expected_captions:
            raise ValueError(
                f"video_id={video_id} expected {expected_captions} captions, got {count}"
            )

    output_dir = Path(output_dir)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    summary_json = output_dir / "validation_summary.json"
    _atomic_csv(
        train_csv,
        ["video_id"],
        ({"video_id": video_id} for video_id in manifest["train_video_ids"]),
    )
    _atomic_csv(
        val_csv,
        ["video_id", "sentence"],
        (
            {"video_id": video_id, "sentence": caption}
            for video_id in manifest["val_video_ids"]
            for caption in captions[video_id]
        ),
    )
    _atomic_json(
        summary_json,
        {
            "protocol_version": manifest["protocol_version"],
            "seed": manifest["seed"],
            "counts": manifest["counts"],
            "overlap_counts": manifest["overlap_counts"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": _manifest_sha256(manifest),
        },
    )
    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "summary_json": str(summary_json),
    }


def load_trusted_manifest(path):
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"trusted split manifest is malformed at {path}: {exc}") from exc
    _validate_manifest_shape(manifest)
    return manifest


def write_trusted_manifest(path, manifest):
    _validate_manifest_shape(manifest)
    _atomic_json(path, manifest)
