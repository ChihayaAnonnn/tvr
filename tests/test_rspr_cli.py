import logging
import sys
from types import SimpleNamespace

import pytest
import torch

import main_task_retrieval
from experiment_tracking import build_experiment_manifest

RSPR_DEFAULTS = {
    "rspr_mode": "legacy",
    "rspr_sample_count": 4,
    "rspr_eval_sample_count": 8,
    "rspr_match_mode": "soft",
    "rspr_detach_samples": False,
    "rspr_match_temperature": 0.07,
    "rspr_prob_temperature": 0.07,
    "rspr_rank_temperature": 0.07,
    "rspr_hard_negatives": 8,
    "rspr_prior_std": 0.1,
    "rspr_prob_weight": 0.1,
    "rspr_rank_weight": 0.1,
    "rspr_anchor_weight": 1e-4,
    "rspr_warmup_epochs": 1.0,
    "rspr_eval_seed": 0,
    "rspr_top_r": 100,
    "rspr_det_temperature": 1.0,
    "rspr_rerank_temperature": 1.0,
    "rspr_rerank_weight": 0.1,
    "rspr_pair_chunk_size": 4096,
    "rspr_freeze_clip": False,
    "rspr_freeze_dsa": False,
}


def _args(**overrides):
    values = dict(RSPR_DEFAULTS)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_get_args_exposes_exact_rspr_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_task_retrieval.py",
            "--do_eval",
            "--init_model",
            "checkpoint.bin",
            "--output_dir",
            str(tmp_path),
        ],
    )

    parsed = main_task_retrieval.get_args()

    assert {name: getattr(parsed, name) for name in RSPR_DEFAULTS} == RSPR_DEFAULTS


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"rspr_mode": "mean", "rspr_sample_count": 2}, "sample_count=1"),
        ({"rspr_mode": "stochastic", "rspr_sample_count": 3}, "positive even"),
        ({"rspr_mode": "stochastic", "rspr_eval_sample_count": 3}, "positive even"),
        ({"rspr_mode": "stochastic", "rspr_hard_negatives": 0}, "hard_negatives"),
        (
            {
                "rspr_mode": "mean",
                "rspr_sample_count": 1,
                "rspr_hard_negatives": -1,
            },
            "hard_negatives",
        ),
        ({"rspr_mode": "off", "rspr_match_temperature": 0.0}, "positive"),
        ({"rspr_mode": "off", "rspr_prob_temperature": -0.1}, "positive"),
        ({"rspr_mode": "off", "rspr_rank_temperature": 0.0}, "positive"),
        ({"rspr_mode": "off", "rspr_prior_std": 0.0}, "positive"),
        ({"rspr_mode": "off", "rspr_pair_chunk_size": 0}, "positive"),
        ({"rspr_mode": "off", "rspr_prob_weight": -0.1}, "nonnegative"),
        ({"rspr_mode": "off", "rspr_rank_weight": -0.1}, "nonnegative"),
        ({"rspr_mode": "off", "rspr_anchor_weight": -0.1}, "nonnegative"),
        ({"rspr_mode": "off", "rspr_rerank_weight": -0.1}, "nonnegative"),
        ({"rspr_mode": "off", "rspr_warmup_epochs": -0.1}, "nonnegative"),
        ({"rspr_mode": "off", "rspr_top_r": -1}, "nonnegative"),
        ({"rspr_mode": "legacy", "rspr_freeze_clip": True}, "freeze"),
        ({"rspr_mode": "legacy", "rspr_freeze_dsa": True}, "freeze"),
    ),
)
def test_validate_rspr_cli_rejects_invalid_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        main_task_retrieval.validate_rspr_cli(_args(**overrides))


def test_validate_rspr_cli_allows_top_r_zero():
    main_task_retrieval.validate_rspr_cli(_args(rspr_mode="off", rspr_top_r=0))


def test_validate_rspr_cli_legacy_ignores_numeric_rspr_values():
    main_task_retrieval.validate_rspr_cli(
        _args(
            rspr_sample_count=-3,
            rspr_eval_sample_count=-5,
            rspr_match_temperature=0.0,
            rspr_prob_temperature=-1.0,
            rspr_rank_temperature=0.0,
            rspr_prior_std=-1.0,
            rspr_prob_weight=-1.0,
            rspr_rank_weight=-1.0,
            rspr_anchor_weight=-1.0,
            rspr_warmup_epochs=-1.0,
            rspr_top_r=-1,
            rspr_pair_chunk_size=0,
        )
    )


def test_effective_parameter_log_has_explicit_rspr_group(
    monkeypatch, caplog, tmp_path
):
    args = _args(
        seed=3,
        output_dir=str(tmp_path),
        local_rank=0,
        experiment_desc="",
    )
    logger = logging.getLogger("test.rspr.effective.parameters")
    monkeypatch.setattr(main_task_retrieval, "get_logger", lambda _path: logger)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caplog.set_level(logging.INFO, logger=logger.name)

    main_task_retrieval.set_seed_logger(args)

    rspr_lines = [record.getMessage() for record in caplog.records if "[RSPR]" in record.getMessage()]
    assert len(rspr_lines) == 1
    for name, value in RSPR_DEFAULTS.items():
        assert f"{name}={value}" in rspr_lines[0]


def test_experiment_manifest_records_all_rspr_fields_explicitly():
    args = _args()

    manifest = build_experiment_manifest(
        args, split_summary=None, batch_semantics={}, git_state={}
    )

    assert manifest["rspr"] == RSPR_DEFAULTS
