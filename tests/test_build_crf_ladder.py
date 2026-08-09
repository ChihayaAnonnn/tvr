"""Task 3 — CRF ladder builders (MSRVTT + MSVD).

Unit-only: PyAV is never invoked because ``codec.transcode_atomic``,
``codec.probe_video``, ``codec.sha256_file``, and ``codec.run_preflight`` are
monkeypatched. On Linux ``ProcessPoolExecutor`` uses ``fork`` by default so
these patches are inherited by worker subprocesses.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from crf_ladder import EVAL_CRFS, TRAIN_CRFS, codec
from scripts import build_msrvtt_crf_ladder, build_msvd_crf_ladder
from scripts.build_msrvtt_crf_ladder import build_jobs

mp.set_start_method("fork", force=True)


def _fake_manifest(tmp_path, train, val):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "trusted.json"
    path.write_text(
        json.dumps(
            {
                "train_video_ids": list(train),
                "val_video_ids": list(val),
                "test_video_ids": ["video9999"],
            }
        )
    )
    return path


def _fake_source_root(tmp_path, video_ids, extension=".mp4"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    for vid in video_ids:
        target = root / f"{vid}{extension}"
        if not target.exists():
            target.write_bytes(f"source_bytes_{vid}".encode())
    return root


def _install_fake_codec(monkeypatch):
    """Replace all PyAV-touching primitives with deterministic fakes."""

    def fake_sha(path):
        return f"srchash:{Path(path).name}"

    def fake_probe(path):
        return codec.VideoMetadata(
            width=298,
            height=224,
            fps_num=12,
            fps_den=1,
            time_base_num=1,
            time_base_den=12288,
            frame_count=12,
            duration_seconds=1.0,
        )

    def fake_transcode(source, destination, crf):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        payload = f"encoded:{Path(source).name}:crf{int(crf)}".encode()
        Path(destination).write_bytes(payload)
        return Path(destination)

    def fake_preflight(source, destination, frame_limit):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"preflight")
        from fractions import Fraction

        return codec.PreflightResult(
            encoder="libx264",
            source_width=298,
            source_height=224,
            source_fps=Fraction(12, 1),
            output_width=298,
            output_height=224,
            output_fps=Fraction(12, 1),
            decoded_frames=frame_limit,
        )

    monkeypatch.setattr(codec, "sha256_file", fake_sha)
    monkeypatch.setattr(codec, "probe_video", fake_probe)
    monkeypatch.setattr(codec, "transcode_atomic", fake_transcode)
    monkeypatch.setattr(codec, "run_preflight", fake_preflight)


def test_msrvtt_jobs_use_only_train_and_val_ids():
    manifest = {
        "train_video_ids": ["video1", "video2"],
        "val_video_ids": ["video3"],
        "test_video_ids": ["video9999"],
    }
    jobs = build_jobs(manifest)
    assert {(j.video_id, j.crf) for j in jobs} == {
        ("video1", 30), ("video1", 38), ("video1", 46),
        ("video2", 30), ("video2", 38), ("video2", 46),
        ("video3", 34), ("video3", 36), ("video3", 42), ("video3", 44), ("video3", 50),
    }
    assert all(j.video_id != "video9999" for j in jobs)


def test_build_jobs_ordering_is_deterministic():
    manifest = {
        "train_video_ids": ["b", "a"],
        "val_video_ids": ["z"],
        "test_video_ids": [],
    }
    jobs = build_jobs(manifest)
    # Train jobs come first in manifest order (b then a), each expanded across TRAIN_CRFS
    # in TRAIN_CRFS order, then val jobs across EVAL_CRFS in EVAL_CRFS order.
    expected = (
        [("b", crf) for crf in TRAIN_CRFS]
        + [("a", crf) for crf in TRAIN_CRFS]
        + [("z", crf) for crf in EVAL_CRFS]
    )
    assert [(j.video_id, j.crf) for j in jobs] == expected


def test_workers_never_open_the_manifest_file(monkeypatch, tmp_path):
    train_ids = ["video1", "video2"]
    val_ids = ["video3"]
    _install_fake_codec(monkeypatch)
    manifest = _fake_manifest(tmp_path, train_ids, val_ids)
    source_root = _fake_source_root(tmp_path, train_ids + val_ids)
    output_root = tmp_path / "out"

    parent_pid = os.getpid()
    writer_pids: set[int] = set()
    real_append = codec.append_manifest_row

    def spy(path, row):
        writer_pids.add(os.getpid())
        real_append(path, row)

    monkeypatch.setattr(codec, "append_manifest_row", spy)
    monkeypatch.setattr(build_msrvtt_crf_ladder.codec, "append_manifest_row", spy)

    rc = build_msrvtt_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--manifest", str(manifest),
            "--output-root", str(output_root),
            "--workers", "4",
            "--limit", "8",
        ]
    )
    assert rc == 0
    assert writer_pids == {parent_pid}


def _run_build(tmp_path, workers, limit, output_root=None, monkeypatch=None):
    train_ids = [f"video{i}" for i in range(1, 20)]
    val_ids = [f"videoV{i}" for i in range(1, 20)]
    manifest = _fake_manifest(tmp_path, train_ids, val_ids)
    source_root = _fake_source_root(tmp_path, train_ids + val_ids)
    if output_root is None:
        output_root = tmp_path / "out"
    rc = build_msrvtt_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--manifest", str(manifest),
            "--output-root", str(output_root),
            "--workers", str(workers),
            "--limit", str(limit),
        ]
    )
    assert rc == 0
    return output_root


def _load_jsonl(path):
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line]
    return [json.loads(line) for line in lines]


def _expected_jobs_up_to(limit):
    """Return the first `limit` (video_id, crf) tuples following build_jobs order."""
    train_ids = [f"video{i}" for i in range(1, 20)]
    val_ids = [f"videoV{i}" for i in range(1, 20)]
    manifest = {"train_video_ids": train_ids, "val_video_ids": val_ids, "test_video_ids": []}
    return [(j.video_id, j.crf) for j in build_jobs(manifest)[:limit]]


def test_multi_worker_jsonl_has_one_row_per_completed_job_and_is_deterministic(
    monkeypatch, tmp_path
):
    """Byte-identical manifest across reruns when source + output paths are stable."""

    _install_fake_codec(monkeypatch)
    train_ids = [f"video{i}" for i in range(1, 20)]
    val_ids = [f"videoV{i}" for i in range(1, 20)]
    shared = tmp_path / "shared"
    manifest = _fake_manifest(shared, train_ids, val_ids)
    source_root = _fake_source_root(shared, train_ids + val_ids)

    def run_once(output_root: Path) -> dict[int, bytes]:
        rc = build_msrvtt_crf_ladder.main(
            argv=[
                "--source-root", str(source_root),
                "--manifest", str(manifest),
                "--output-root", str(output_root),
                "--workers", "4",
                "--limit", "24",
            ]
        )
        assert rc == 0
        return {
            crf: (output_root / f"crf{crf}" / "manifest.jsonl").read_bytes()
            for crf in (30, 38, 46) + EVAL_CRFS
            if (output_root / f"crf{crf}" / "manifest.jsonl").exists()
        }

    # Same output dir, wiped between runs → paths are identical, so are the bytes.
    out = tmp_path / "out"
    a_bytes = run_once(out)
    import shutil
    shutil.rmtree(out)
    b_bytes = run_once(out)
    assert set(a_bytes) == set(b_bytes)
    for crf in a_bytes:
        assert a_bytes[crf] == b_bytes[crf]

    expected = _expected_jobs_up_to(24)
    per_crf_expected: dict[int, list[tuple[str, int]]] = {}
    for vid, crf in expected:
        per_crf_expected.setdefault(crf, []).append((vid, crf))
    for crf, expected_rows in per_crf_expected.items():
        path = out / f"crf{crf}" / "manifest.jsonl"
        rows = _load_jsonl(path)
        assert len(rows) == len(expected_rows)
        assert [(r["video_id"], r["crf"]) for r in rows] == expected_rows
        for line in path.read_text().splitlines():
            if line:
                json.loads(line)  # each line parses standalone


def test_resume_after_partial_run_never_duplicates_rows(monkeypatch, tmp_path):
    _install_fake_codec(monkeypatch)
    out = tmp_path / "out"
    _run_build(tmp_path, workers=2, limit=8, output_root=out)
    initial_rows = {
        crf: len(_load_jsonl(out / f"crf{crf}" / "manifest.jsonl"))
        for crf in (30, 38, 46)
    }
    _run_build(tmp_path, workers=2, limit=16, output_root=out)
    for crf in (30, 38, 46):
        rows = _load_jsonl(out / f"crf{crf}" / "manifest.jsonl")
        expected = sum(1 for vid, c in _expected_jobs_up_to(16) if c == crf)
        assert len(rows) == expected
        # All video_ids unique per crf: no duplicate rebuild rows.
        ids = [r["video_id"] for r in rows]
        assert len(set(ids)) == len(ids)
    # Ensure the first-batch rows are still present.
    assert initial_rows[30] > 0


def test_check_only_verifies_existing_manifest_without_writing(monkeypatch, tmp_path):
    _install_fake_codec(monkeypatch)
    out = tmp_path / "out"
    _run_build(tmp_path, workers=2, limit=6, output_root=out)
    # Capture original manifest bytes.
    manifest_bytes = {
        crf: (out / f"crf{crf}" / "manifest.jsonl").read_bytes()
        for crf in (30, 38, 46)
        if (out / f"crf{crf}" / "manifest.jsonl").exists()
    }
    train_ids = [f"video{i}" for i in range(1, 20)]
    val_ids = [f"videoV{i}" for i in range(1, 20)]
    manifest = _fake_manifest(tmp_path / "check", train_ids, val_ids)
    source_root = _fake_source_root(tmp_path / "check", train_ids + val_ids)
    rc = build_msrvtt_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--manifest", str(manifest),
            "--output-root", str(out),
            "--workers", "1",
            "--limit", "6",
            "--check-only",
        ]
    )
    assert rc == 0
    for crf, expected in manifest_bytes.items():
        assert (out / f"crf{crf}" / "manifest.jsonl").read_bytes() == expected


def test_msvd_builder_writes_subset_json_and_only_34_42(monkeypatch, tmp_path):
    _install_fake_codec(monkeypatch)
    test_list = tmp_path / "test_list.txt"
    ids = [f"msvd{i:03d}" for i in range(20)]
    test_list.write_text("\n".join(ids) + "\n")
    source_root = _fake_source_root(tmp_path, ids, extension=".avi")
    out = tmp_path / "msvd_out"
    rc = build_msvd_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--test-list", str(test_list),
            "--output-root", str(out),
            "--subset-size", "10",
            "--workers", "2",
        ]
    )
    assert rc == 0
    subset = json.loads((out / "subset.json").read_text())
    assert len(subset["video_ids"]) == 10
    assert subset["selection_rule"] == "first_n_from_test_list_txt"
    # Only CRF 34 and 42 built.
    assert (out / "crf34" / "manifest.jsonl").exists()
    assert (out / "crf42" / "manifest.jsonl").exists()
    assert not (out / "crf30").exists()
    assert not (out / "crf50").exists()


def test_msvd_builder_reuses_existing_subset_json_on_rerun(monkeypatch, tmp_path):
    _install_fake_codec(monkeypatch)
    test_list = tmp_path / "test_list.txt"
    ids = [f"msvd{i:03d}" for i in range(20)]
    test_list.write_text("\n".join(ids) + "\n")
    source_root = _fake_source_root(tmp_path, ids, extension=".avi")
    out = tmp_path / "msvd_out"
    build_msvd_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--test-list", str(test_list),
            "--output-root", str(out),
            "--subset-size", "10",
            "--workers", "1",
        ]
    )
    subset_before = (out / "subset.json").read_bytes()
    # Even with a larger subset-size, the checked-in subset.json wins.
    build_msvd_crf_ladder.main(
        argv=[
            "--source-root", str(source_root),
            "--test-list", str(test_list),
            "--output-root", str(out),
            "--subset-size", "15",
            "--workers", "1",
        ]
    )
    subset_after = (out / "subset.json").read_bytes()
    assert subset_before == subset_after


def test_msvd_builder_refuses_when_source_missing(monkeypatch, tmp_path):
    _install_fake_codec(monkeypatch)
    test_list = tmp_path / "test_list.txt"
    ids = [f"msvd{i:03d}" for i in range(5)]
    test_list.write_text("\n".join(ids) + "\n")
    # Create only some of the sources.
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / f"{ids[0]}.avi").write_bytes(b"x")
    out = tmp_path / "msvd_out"
    with pytest.raises(FileNotFoundError):
        build_msvd_crf_ladder.main(
            argv=[
                "--source-root", str(source_root),
                "--test-list", str(test_list),
                "--output-root", str(out),
                "--subset-size", "5",
                "--workers", "1",
            ]
        )
