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
