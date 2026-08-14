"""Contract tests for the tracked experiment-archive guidance."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    ["README.md", "AGENTS.md", "docs/agent/experiment-artifacts.md"],
)
def test_docs_name_archive_and_forbid_legacy_outputs(repo_root: Path, path: str) -> None:
    text = (repo_root / path).read_text(encoding="utf-8")
    assert "artifacts/contextual-risk-v2" in text
    assert "experiments/contextual-risk-v2/registry.jsonl" in text
    assert "不得将新实验直接写入 `logs/` 或 `ckpts/`" in text
    assert "validation" in text.lower()


def test_agent_rules_cover_every_mandatory_archive_guardrail(repo_root: Path) -> None:
    text = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
    for rule in (
        "读取 `experiments/contextual-risk-v2/study.yaml` 和 `experiments/contextual-risk-v2/registry.jsonl`",
        "reserve-phase",
        "artifacts/contextual-risk-v2",
        "不得将新实验直接写入 `logs/` 或 `ckpts/`",
        "1 MiB",
        "validation",
        "terminal registry",
        "不得静默重试",
    ):
        assert rule in text


def test_study_is_locked_to_approved_campaign_protocol(repo_root: Path) -> None:
    text = (repo_root / "experiments/contextual-risk-v2/study.yaml").read_text(
        encoding="utf-8"
    )
    for required in (
        "stages:",
        "baseline",
        "feature_ablation",
        "loss_ablation",
        "topk_sweep",
        "multiseed",
        "calibration",
        "robustness",
        "efficiency",
        "final",
        "seeds: [0, 1, 2]",
        "samples: 10000",
        "seed: 0",
        "test_sealed",
        "canonical_a1",
        "a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607",
        "hardware_policy",
        "stop_gates:",
    ):
        assert required in text


def test_git_ignores_data_plane_but_tracks_control_plane(repo_root: Path) -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "artifacts/example/file.bin"],
        cwd=repo_root,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "check-ignore", "-q", "experiments/contextual-risk-v2/study.yaml"],
        cwd=repo_root,
        check=False,
    )
    assert ignored.returncode == 0
    assert tracked.returncode == 1


def test_study_readme_exposes_lifecycle_commands_and_stage_zero_gate(repo_root: Path) -> None:
    text = (repo_root / "experiments/contextual-risk-v2/README.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "reserve-phase",
        "verify-run",
        "start-run",
        "transition-run",
        "finalize-phase",
        "Stage 0",
        "四卡",
        "test",
    ):
        assert required in text
