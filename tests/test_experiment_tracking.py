"""Generic provenance survives retirement of model-specific configuration."""

import json
from types import SimpleNamespace

import pytest

from experiment_tracking import (
    append_batch_protocol_stats,
    atomic_write_json,
    collect_git_state,
    compute_batch_semantics,
    is_global_rank_zero,
)


def test_tracking_atomic_writer_rejects_nested_nonstandard_json_number(tmp_path):
    target = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="strict JSON"):
        atomic_write_json(target, {"config": {"risk": float("inf")}})
    assert not target.exists()


def test_atomic_writer_and_batch_telemetry(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"source": "exact"})
    assert json.loads(target.read_text()) == {"source": "exact"}
    stats = dict(unique_video_count=2, duplicate_sample_count=0, mean_positive_count=1)
    path = tmp_path / "stats.tsv"
    append_batch_protocol_stats(path, 0, 0, 0, stats)
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError):
        append_batch_protocol_stats(path, 0, 0, 0, {**stats, "unexpected": 1})
    assert is_global_rank_zero(SimpleNamespace(rank=0))
    assert not is_global_rank_zero(SimpleNamespace(rank=1))


def test_batch_and_source_contract(tmp_path):
    import subprocess

    result = compute_batch_semantics(16, 2, 2, 10, 3)
    assert result["per_rank_micro_batch"] == 4
    assert result["total_optimizer_steps"] == 15
    with pytest.raises(ValueError):
        compute_batch_semantics(15, 2, 2, 10, 3)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    path = tmp_path / "source.py"
    path.write_text("one")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)
    assert not collect_git_state(tmp_path)["dirty"]
    path.write_text("two")
    assert collect_git_state(tmp_path)["modified_paths"] == ["source.py"]
