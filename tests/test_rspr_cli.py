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
    "rspr_prob_loss": "soft_bce",
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
    "rspr_recall_source": "deterministic",
    "rspr_rerank_scale": "logit_scale",
    "rspr_pair_chunk_size": 4096,
    "rspr_freeze_clip": False,
    "rspr_freeze_dsa": False,
    "rspr_grad_diagnostics": False,
}


# The knobs that decide how much of the backbone actually trains. A silent
# change to any of them moves R@1 by more than the whole RSPR module does, so
# the manifest has to carry them next to the RSPR settings.
OPTIMIZATION_DEFAULTS = {
    "lr": 0.0001,
    "coef_lr": 1.0,
    "lr_decay": 0.9,
    "warmup_proportion": 0.1,
    "epochs": 20,
    "freeze_layer_num": 0,
    "max_words": 20,
    "max_frames": 100,
    "feature_framerate": 1,
    "slice_framepos": 0,
    "expand_msrvtt_sentences": False,
}


def _args(**overrides):
    values = dict(RSPR_DEFAULTS)
    values.update(OPTIMIZATION_DEFAULTS)
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
        ({"rspr_mode": "off", "rspr_prob_loss": "bogus"}, "prob_loss"),
        ({"rspr_mode": "off", "rspr_recall_source": "bogus"}, "recall_source"),
        ({"rspr_mode": "off", "rspr_rerank_scale": "bogus"}, "rerank_scale"),
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


def _trusted_args(**overrides):
    values = {
        "experiment_profile": "parity",
        "datatype": "msrvtt",
        "do_train": True,
        "do_eval": False,
        "init_model": "",
        "eval_split": "val",
        "expand_msrvtt_sentences": True,
        "run_final_test": False,
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "freeze_layer_num": 0,
        "max_frames": 12,
        "slice_framepos": 2,
        "lr": 5e-5,
        "coef_lr": 1e-3,
        "fold_val_into_train": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_trusted_cli_accepts_the_official_parity_recipe():
    main_task_retrieval.validate_trusted_cli(_trusted_args())


def test_validate_trusted_cli_keeps_the_held_out_split_out_of_other_profiles():
    """Only parity may train on the 500 videos trusted-v1 holds out.

    Folding them in is what makes a parity number comparable to a published
    one; doing it anywhere else would silently contaminate our own val set.
    """

    with pytest.raises(ValueError, match="fold_val_into_train"):
        main_task_retrieval.validate_trusted_cli(
            _trusted_args(experiment_profile="hygiene", batch_size=256)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        # Accumulation restores the optimizer batch but not the contrastive
        # one, so 256x2 trains a 256-way InfoNCE and is not parity.
        ({"batch_size": 256, "gradient_accumulation_steps": 2}, "batch_size=512"),
        ({"gradient_accumulation_steps": 2}, "gradient_accumulation_steps=1"),
        ({"freeze_layer_num": 8}, "freeze_layer_num=0"),
        ({"max_frames": 8}, "max_frames=12"),
        ({"slice_framepos": 3}, "slice_framepos=2"),
        # The official recipe trains on all 9000 source videos.
        ({"fold_val_into_train": False}, "fold_val_into_train"),
    ),
)
def test_validate_trusted_cli_rejects_parity_that_is_not_parity(overrides, message):
    """A 'parity' run that quietly differs is worse than no parity run at all.

    The point of the profile is to make one number comparable to a published
    one, so every knob the comparison rests on is checked rather than assumed.
    """

    with pytest.raises(ValueError, match=message):
        main_task_retrieval.validate_trusted_cli(_trusted_args(**overrides))


def test_effective_parameter_log_shows_how_much_of_the_backbone_trains(
    monkeypatch, caplog, tmp_path
):
    """freeze_layer_num and slice_framepos changed silently and were never logged.

    They belong in the run's own log, not only in the manifest, because that
    is where a run is read while it is still worth killing.
    """

    args = _args(
        seed=3,
        output_dir=str(tmp_path),
        local_rank=0,
        experiment_desc="",
        freeze_layer_num=8,
        slice_framepos=3,
    )
    logger = logging.getLogger("test.rspr.training.parameters")
    monkeypatch.setattr(main_task_retrieval, "get_logger", lambda _path: logger)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caplog.set_level(logging.INFO, logger=logger.name)

    main_task_retrieval.set_seed_logger(args)

    training_lines = [
        record.getMessage()
        for record in caplog.records
        if "[Training]" in record.getMessage()
    ]
    assert len(training_lines) == 1
    for expected in ("freeze_layer_num=8", "slice_framepos=3"):
        assert expected in training_lines[0]


def test_effective_parameter_log_shows_which_videos_training_may_see(
    monkeypatch, caplog, tmp_path
):
    """Training scope belongs in the run log next to the split manifest."""

    args = _args(
        seed=3,
        output_dir=str(tmp_path),
        local_rank=0,
        experiment_desc="",
        fold_val_into_train=True,
    )
    logger = logging.getLogger("test.rspr.protocol.parameters")
    monkeypatch.setattr(main_task_retrieval, "get_logger", lambda _path: logger)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caplog.set_level(logging.INFO, logger=logger.name)

    main_task_retrieval.set_seed_logger(args)

    protocol_lines = [
        record.getMessage()
        for record in caplog.records
        if "[Protocol]" in record.getMessage()
    ]
    assert len(protocol_lines) == 1
    assert "fold_val_into_train=True" in protocol_lines[0]


def test_experiment_manifest_records_the_training_split_scope():
    manifest = build_experiment_manifest(
        _args(fold_val_into_train=True),
        split_summary=None,
        batch_semantics={},
        git_state={},
    )

    assert manifest["data"]["fold_val_into_train"] is True


def test_experiment_manifest_defaults_the_training_split_scope_to_trusted():
    manifest = build_experiment_manifest(
        _args(), split_summary=None, batch_semantics={}, git_state={}
    )

    assert manifest["data"]["fold_val_into_train"] is False


def test_get_args_defaults_to_the_trusted_training_split(monkeypatch, tmp_path):
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

    assert main_task_retrieval.get_args().fold_val_into_train is False


def test_experiment_manifest_records_all_rspr_fields_explicitly():
    args = _args()

    manifest = build_experiment_manifest(
        args, split_summary=None, batch_semantics={}, git_state={}
    )

    assert manifest["rspr"] == RSPR_DEFAULTS


def test_experiment_manifest_records_the_optimization_recipe():
    """The manifest logged 24 RSPR knobs and no learning rate.

    A freeze_layer_num default that changed from 0 to 8 therefore sat
    unnoticed for six days while every arm was compared against a baseline it
    had quietly crippled. These fields make that regression visible in the
    diff between two runs' manifests.
    """

    args = _args()

    manifest = build_experiment_manifest(
        args, split_summary=None, batch_semantics={}, git_state={}
    )

    assert manifest["optimization"] == OPTIMIZATION_DEFAULTS


def test_experiment_manifest_reports_the_run_s_own_optimization_values():
    args = _args(freeze_layer_num=8, max_frames=8, lr=5e-5, slice_framepos=3)

    manifest = build_experiment_manifest(
        args, split_summary=None, batch_semantics={}, git_state={}
    )

    assert manifest["optimization"]["freeze_layer_num"] == 8
    assert manifest["optimization"]["max_frames"] == 8
    assert manifest["optimization"]["lr"] == pytest.approx(5e-5)
    assert manifest["optimization"]["slice_framepos"] == 3


def test_get_args_exposes_exact_optimization_defaults(monkeypatch, tmp_path):
    """Pins the parser defaults to the values the manifest falls back on."""

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

    assert {
        name: getattr(parsed, name) for name in OPTIMIZATION_DEFAULTS
    } == OPTIMIZATION_DEFAULTS
