"""Task 1 — Frozen CRF protocol contract tests."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from crf_ladder import EVAL_CRFS, TRAIN_CRFS, X264_RECIPE, crf_output_path
from crf_ladder.protocol import AUDIT_SIZE, DEV_SIZE, X264Recipe, split_crf_dev_audit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRUSTED_PATH = _REPO_ROOT / "dataloaders" / "splits" / "msrvtt_trusted_v1_seed0.json"
_CRF_SPLIT_PATH = _REPO_ROOT / "dataloaders" / "splits" / "crf_dev_audit_seed0.json"


def test_frozen_crf_contract():
    assert TRAIN_CRFS == (30, 38, 46)
    assert EVAL_CRFS == (34, 36, 42, 44, 50)
    assert DEV_SIZE == 250
    assert AUDIT_SIZE == 250
    assert isinstance(X264_RECIPE, X264Recipe)
    assert asdict(X264_RECIPE) == {
        "preset": "veryfast",
        "pixel_format": "yuv420p",
        "color_range": "mpeg",
        "color_primaries": "smpte170m",
        "color_trc": "smpte170m",
        "colorspace": "smpte170m",
        "time_base_policy": "inverse_input_fps",
        "keyint": 48,
        "min_keyint": 48,
        "scenecut": 0,
        "b_frames": 2,
    }


def test_dev_audit_preserves_manifest_order_and_is_disjoint():
    ids = [f"video{i}" for i in range(500)]
    dev, audit = split_crf_dev_audit(ids)
    assert dev == ids[:250]
    assert audit == ids[250:]
    assert set(dev).isdisjoint(audit)


def test_output_path_is_canonical(tmp_path):
    assert crf_output_path(tmp_path, 34, "video12") == tmp_path / "crf34" / "video12.mp4"


def test_versioned_manifest_contains_frozen_crf_partition():
    payload = json.loads(_TRUSTED_PATH.read_text())
    dev, audit = split_crf_dev_audit(payload["val_video_ids"])
    assert dev == payload["val_video_ids"][:250]
    assert audit == payload["val_video_ids"][250:]


def test_checked_in_crf_split_matches_manifest_and_records_source_sha256():
    trusted_bytes = _TRUSTED_PATH.read_bytes()
    trusted_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
    split = json.loads(_CRF_SPLIT_PATH.read_text())
    assert split["source_manifest"] == "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    assert split["source_manifest_sha256"] == trusted_sha256
    dev, audit = split_crf_dev_audit(json.loads(trusted_bytes)["val_video_ids"])
    assert split["crf_dev_video_ids"] == dev
    assert split["crf_audit_video_ids"] == audit
