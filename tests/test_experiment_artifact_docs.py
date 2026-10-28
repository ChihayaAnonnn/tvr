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
        "不再要求每次失败后人工审核和授权重试",
        "不能证明身份、源码或输出完整性",
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
        "terminal_in_place_retry: forbidden",
        "authorized_successor: subsequent_phase_with_new_unique_run_id",
        "predecessor_lineage: parent_run_ids",
        "authorization_reason: selection_lineage",
        "canonical_a1",
        "a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607",
        "hardware_policy",
        "stop_gates:",
    ):
        assert required in text


def test_study_and_scientific_plan_lock_registered_a3_variant_composition(
    repo_root: Path,
) -> None:
    study = (repo_root / "experiments/contextual-risk-v2/study.yaml").read_text(
        encoding="utf-8"
    )
    plan = (
        repo_root
        / "docs/superpowers/plans/2026-08-14-reliable-retrieval-experiment-campaign.md"
    ).read_text(encoding="utf-8")
    for required in (
        "feature_sets: [full, score_only, semantic_only, no_context, no_score_geometry]",
        "topk: [1, 4, 8, 16, 32]",
        "seeds: [0, 1, 2]",
        "epochs: [3, 5]",
        "loss_weights:",
    ):
        assert required in study
    for required in (
        "run_train_msrvtt_contextual_risk_bg.sh",
        "manifest-derived allowlist",
        "config.model.contextual_risk_feature_set",
        "config.loss.w_contextual_risk_fit",
        "config.optimizer.seed",
        "config.scheduler.epochs",
        "不得接受任意 trailing argv",
    ):
        assert required in plan


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


def test_git_ignores_only_the_registry_administrative_lock(repo_root: Path) -> None:
    lock = subprocess.run(
        ["git", "check-ignore", "-q", "experiments/contextual-risk-v2/registry.jsonl.lock"],
        cwd=repo_root,
        check=False,
    )
    registry = subprocess.run(
        ["git", "check-ignore", "-q", "experiments/contextual-risk-v2/registry.jsonl"],
        cwd=repo_root,
        check=False,
    )
    sibling = subprocess.run(
        ["git", "check-ignore", "-q", "experiments/contextual-risk-v2/study.yaml.lock"],
        cwd=repo_root,
        check=False,
    )

    assert lock.returncode == 0
    assert registry.returncode == 1
    assert sibling.returncode == 1


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


@pytest.mark.parametrize(
    "path",
    ["docs/agent/experiment-artifacts.md"],
)
def test_docs_forbid_terminal_in_place_retry_and_describe_existing_lineage(
    repo_root: Path, path: str
) -> None:
    text = (repo_root / path).read_text(encoding="utf-8")
    for required in (
        "failed/cancelled",
        "不可原地重试",
        "新的 phase/run ID",
        "`parent_run_ids`",
        "`selection_lineage`",
        "`supersedes_sha256`",
    ):
        assert required in text


def test_root_readme_describes_all_ignored_runtime_writes(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "受跟踪控制面不会在运行时写入" in text
    for location in (
        "<run-dir>/status.json",
        "<run-dir>/provenance/audit.jsonl",
        "<run-dir>/metrics/",
        "<run-dir>/checkpoints/",
        "<run-dir>/logs/",
        "<run-dir>/arrays/",
    ):
        assert location in text


@pytest.mark.parametrize(
    "path", ["README.md", "docs/agent/experiment-artifacts.md"]
)
def test_reserve_phase_docs_pass_parent_data_plane_root(repo_root: Path, path: str) -> None:
    text = (repo_root / path).read_text(encoding="utf-8")
    assert '--artifact-root "$PWD/artifacts"' in text
    assert '--artifact-root "$PWD/artifacts/contextual-risk-v2"' not in text
    assert "父级数据面根目录" in text


def test_docs_define_future_scheduled_run_identity_field(repo_root: Path) -> None:
    guide = (repo_root / "docs/agent/experiment-artifacts.md").read_text(
        encoding="utf-8"
    )
    campaign = (
        repo_root
        / "docs/superpowers/plans/2026-08-14-reliable-retrieval-experiment-campaign.md"
    ).read_text(encoding="utf-8")
    assert "`scheduled_timestamp_utc`" in guide
    assert "`scheduled_timestamp_utc`" in campaign
    assert "预留时间" in guide


def test_production_qualification_plan_maps_delivered_paths(repo_root: Path) -> None:
    plan = (
        repo_root
        / "docs/superpowers/plans/2026-08-15-production-archive-qualification.md"
    ).read_text(encoding="utf-8")
    for planned, delivered in (
        ("`tools/experiment_artifacts.py`", "`experiment_artifacts.py`"),
        (
            "`schemas/experiment_manifest.schema.json`",
            "`experiments/contextual-risk-v2/schemas/manifest.schema.json`",
        ),
        (
            "`schemas/experiment_registry_record.schema.json`",
            "`experiments/contextual-risk-v2/schemas/registry.schema.json`",
        ),
        ("`tools/archive_qualification.py`", "`main_task_retrieval.py`"),
    ):
        assert f"{planned} -> {delivered}" in plan


def test_qualification_summary_supersedes_git_admin_lineage_for_future_runs(
    repo_root: Path,
) -> None:
    summary = (
        repo_root
        / "experiments/contextual-risk-v2/summaries/archive-qualification.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(summary.split())
    assert "`sdd/task-7-review.md`" in summary
    assert "Git-tracked report or commit" in normalized
    assert "must not cite worktree Git administrative state" in normalized


def test_removed_manual_retry_rule_is_not_reintroduced(repo_root: Path) -> None:
    text = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "不再要求每次失败后人工审核和授权重试" in text
    assert "8. 保留失败运行" not in text
    assert "只有人工审查和授权后" not in text
    assert "不允许覆盖历史产物或绕过研究 gate" in text
