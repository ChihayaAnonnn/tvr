"""Task 2 — PyAV atomic transcode + preflight + manifest tests.

Unit-only tests must not touch PyAV; only the opt-in
`test_real_libx264_preflight_when_video_is_configured` exercises the real
encoder, and it is skipped unless the env var ``CRF_PREFLIGHT_VIDEO`` names a
readable source video.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from crf_ladder import codec
from crf_ladder.codec import (
    ManifestRow,
    append_manifest_row,
    build_manifest_row,
    encoder_options,
    run_preflight,
    should_rebuild,
    transcode_atomic,
)


def _row(
    *,
    video_id: str = "video1",
    success: bool = True,
    source_hash: str = "a",
    output_hash: str = "b",
) -> ManifestRow:
    return ManifestRow(
        video_id=video_id,
        source_path="/tmp/src.mp4",
        output_path="/tmp/out.mp4",
        source_sha256=source_hash,
        output_sha256=output_hash if success else None,
        crf=30,
        preset="veryfast",
        encoder="libx264",
        pyav_version="17.1.0",
        av_library_versions='{"libavcodec": "62.28.101"}',
        pixel_format="yuv420p",
        color_range="mpeg",
        color_primaries="smpte170m",
        color_trc="smpte170m",
        colorspace="smpte170m",
        fps_num=12,
        fps_den=1,
        time_base_num=1,
        time_base_den=12,
        frame_count=12 if success else 0,
        width=298 if success else 0,
        height=224 if success else 0,
        keyint=48,
        min_keyint=48,
        scenecut=0,
        b_frames=2,
        size_bytes=12345 if success else 0,
        bitrate_bps=98760 if success else 0,
        success=success,
        error=None if success else "boom",
    )


def test_encoder_options_matches_frozen_recipe():
    opts = encoder_options(34)
    assert opts["crf"] == "34"
    assert opts["preset"] == "veryfast"
    assert opts["x264-params"] == (
        "keyint=48:min-keyint=48:scenecut=0:bframes=2:force-cfr=1:"
        "colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m:range=limited"
    )


def test_manifest_append_is_one_complete_json_object(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_manifest_row(path, _row())
    payload = json.loads(path.read_text().strip())
    assert payload["video_id"] == "video1"
    assert payload["success"] is True


def test_manifest_append_serializes_sorted_keys(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_manifest_row(path, _row(video_id="video2"))
    line = path.read_text().strip()
    keys = list(json.loads(line, object_pairs_hook=lambda pairs: [k for k, _ in pairs]))
    assert keys == sorted(keys)


def test_manifest_append_two_rows_yield_two_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_manifest_row(path, _row(video_id="a"))
    append_manifest_row(path, _row(video_id="b"))
    lines = [line for line in path.read_text().splitlines() if line]
    assert len(lines) == 2
    assert [json.loads(line)["video_id"] for line in lines] == ["a", "b"]


def test_resume_rebuilds_missing_failed_or_hash_mismatched_output(tmp_path):
    output = tmp_path / "video1.mp4"
    assert should_rebuild(output, None, "a") is True
    output.write_bytes(b"payload")
    assert should_rebuild(output, _row(success=False), "a") is True
    assert should_rebuild(output, _row(source_hash="wrong"), "a") is True


def test_resume_keeps_valid_row(tmp_path, monkeypatch):
    output = tmp_path / "video1.mp4"
    output.write_bytes(b"payload")
    good_row = _row(source_hash="a", output_hash="b")
    monkeypatch.setattr(codec, "sha256_file", lambda p: "b" if Path(p) == output else "x")
    assert should_rebuild(output, good_row, "a") is False


def test_should_rebuild_when_output_hash_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "video1.mp4"
    output.write_bytes(b"payload")
    stale_row = _row(source_hash="a", output_hash="expected")
    monkeypatch.setattr(codec, "sha256_file", lambda p: "different")
    assert should_rebuild(output, stale_row, "a") is True


def test_transcode_failure_never_replaces_destination(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "out.mp4"
    destination.write_bytes(b"known-good")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(codec, "_transcode", boom)
    with pytest.raises(RuntimeError, match="boom"):
        transcode_atomic(source, destination, 30)
    assert destination.read_bytes() == b"known-good"
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_transcode_atomic_cleans_up_tmp_when_invariants_fail(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "out.mp4"

    def fake_transcode(source_path, tmp_path_out, crf):
        Path(tmp_path_out).write_bytes(b"encoded")

    def fail_invariants(src, out):
        raise RuntimeError("shape mismatch: " + str(src))

    monkeypatch.setattr(codec, "_transcode", fake_transcode)
    monkeypatch.setattr(codec, "assert_transcode_invariants", fail_invariants)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        transcode_atomic(source, destination, 30)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_build_manifest_row_success_populates_full_row(monkeypatch, tmp_path):
    output = tmp_path / "video7.mp4"
    output.write_bytes(b"data")
    metadata = codec.VideoMetadata(
        width=298,
        height=224,
        fps_num=12,
        fps_den=1,
        time_base_num=1,
        time_base_den=12,
        frame_count=12,
        duration_seconds=1.0,
    )
    monkeypatch.setattr(codec, "probe_video", lambda p: metadata)
    monkeypatch.setattr(codec, "sha256_file", lambda p: "outhash")
    job = SimpleNamespace(video_id="video7", crf=30)
    row = build_manifest_row(
        job=job,
        source_path=Path("/src/video7.mp4"),
        source_sha256="srchash",
        output_path=output,
        versions={"pyav": "17.1.0", "av_libraries": '{"libavcodec": "62.28.101"}'},
        success=True,
    )
    assert row.success is True
    assert row.video_id == "video7"
    assert row.crf == 30
    assert row.width == 298 and row.height == 224
    assert row.frame_count == 12
    assert row.source_sha256 == "srchash"
    assert row.output_sha256 == "outhash"
    assert row.bitrate_bps == round(len(b"data") * 8 / 1.0)
    assert row.encoder == "libx264"
    assert row.preset == "veryfast"


def test_build_manifest_row_failure_never_touches_output(monkeypatch, tmp_path):
    def boom(_):
        raise AssertionError("probe_video must not be called on failure")

    monkeypatch.setattr(codec, "probe_video", boom)
    monkeypatch.setattr(codec, "sha256_file", boom)
    job = SimpleNamespace(video_id="video9", crf=46)
    row = build_manifest_row(
        job=job,
        source_path=Path("/src/video9.mp4"),
        source_sha256="srchash",
        output_path=tmp_path / "video9.mp4",
        versions={"pyav": "17.1.0", "av_libraries": "{}"},
        success=False,
        error="boom",
    )
    assert row.success is False
    assert row.error == "boom"
    assert row.output_sha256 is None
    assert row.frame_count == 0
    assert row.width == 0 and row.height == 0
    assert row.size_bytes == 0
    assert row.bitrate_bps == 0


def test_real_libx264_preflight_when_video_is_configured(tmp_path):
    source = os.environ.get("CRF_PREFLIGHT_VIDEO")
    if not source:
        pytest.skip("set CRF_PREFLIGHT_VIDEO for runtime codec capability check")
    result = run_preflight(Path(source), tmp_path / "preflight.mp4", frame_limit=16)
    assert result.encoder == "libx264"
    assert result.decoded_frames == 16
    assert (result.source_width, result.source_height) == (
        result.output_width,
        result.output_height,
    )
    assert result.source_fps == result.output_fps
