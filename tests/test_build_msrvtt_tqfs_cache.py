import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_msrvtt_tqfs_cache as builder


def test_cache_space_gate_rejects_insufficient_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(
        builder.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=49 * 1024**3),
    )

    with pytest.raises(RuntimeError, match="at least 50 GiB"):
        builder.ensure_cache_space(tmp_path / "cache", 50 * 1024**3)


def test_cache_space_gate_creates_parent_and_accepts_capacity(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "nested" / "cache"
    monkeypatch.setattr(
        builder.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=60 * 1024**3),
    )

    builder.ensure_cache_space(cache_dir, 50 * 1024**3)

    assert cache_dir.is_dir()


def test_cache_cli_defaults_to_nvme_location(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_msrvtt_tqfs_cache.py"])

    args = builder.parse_args()

    assert Path(args.cache_dir) == (
        Path.home()
        / ".cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224"
    )


def test_main_checks_space_before_creating_worker_pool(
    monkeypatch, tmp_path
):
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "train_video_ids": ["video1"],
                "val_video_ids": ["video2"],
            }
        ),
        encoding="utf-8",
    )
    events = []

    def reject_space(_cache_dir, _minimum_free_bytes=0):
        events.append("space")
        raise RuntimeError("insufficient cache space")

    def fail_executor(*_args, **_kwargs):
        events.append("executor")
        raise AssertionError("worker pool must not be created")

    monkeypatch.setattr(builder, "ensure_cache_space", reject_space)
    monkeypatch.setattr(builder, "ProcessPoolExecutor", fail_executor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_msrvtt_tqfs_cache.py",
            "--split-manifest",
            str(manifest),
            "--features-path",
            str(tmp_path / "videos"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--workers",
            "1",
        ],
    )

    with pytest.raises(RuntimeError, match="insufficient cache space"):
        builder.main()

    assert events == ["space"]
