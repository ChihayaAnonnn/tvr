"""--run_final_test without --do_train finishes a run whose training is done.

hygiene_a0_seed0 trained five epochs, saved five checkpoints, and then lost
its final test to a NameError. Every input the test needs was already on
disk -- the selected checkpoints and the val scores that selected them -- but
--run_final_test was gated behind --do_train, so recovering a ten-minute eval
meant a three-hour retrain.

That gate made sense when the trackers only existed as loop locals. They are
a file now (best_validation_checkpoints.json, written per epoch for
--resume_from), which makes the selection readable without rerunning the loop
that produced it.

The standalone path reads that file rather than accepting checkpoints on the
command line, and that is the point: a hand-picked checkpoint is how you get
a test number that no validation run actually chose. Selection stays a
property of the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main_task_retrieval


def _snapshot(directory: Path, *, t2v_bin: str = "bin.1", v2t_bin: str = "bin.2"):
    """Write a best-val snapshot plus the checkpoint files it points at."""

    directory.mkdir(parents=True, exist_ok=True)
    for name in {t2v_bin, v2t_bin}:
        (directory / name).write_text(name)
    payload = {
        "selection_split": "val",
        "tie_break": "later_epoch",
        "epochs_completed": 5,
        "t2v": {
            "selection_metric": "t2v_r1",
            "selection_score": 59.0,
            "checkpoint": str(directory / t2v_bin),
        },
        "v2t": {
            "selection_metric": "v2t_r1",
            "selection_score": 87.8,
            "checkpoint": str(directory / v2t_bin),
        },
    }
    (directory / "best_validation_checkpoints.json").write_text(json.dumps(payload))
    return payload


def _final_test_args(**overrides):
    values = {
        "experiment_profile": "hygiene",
        "datatype": "msrvtt",
        "do_train": False,
        "do_eval": False,
        "run_final_test": True,
        "init_model": "",
        "resume_model": "",
        "resume_from": "",
        "eval_split": "val",
        "expand_msrvtt_sentences": True,
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "freeze_layer_num": 0,
        "max_frames": 12,
        "slice_framepos": 2,
        "lr": 5e-5,
        "coef_lr": 1e-3,
        "fold_val_into_train": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- validate_trusted_cli ---


def test_run_final_test_no_longer_requires_do_train():
    """The selection lives in a file now, so the loop is not a prerequisite."""

    main_task_retrieval.validate_trusted_cli(_final_test_args())


def test_run_final_test_rejects_do_eval():
    """--do_eval scores one checkpoint on --eval_split; --run_final_test scores
    the val-selected ones on test. Running both in one process would emit two
    unlabelled metric blocks into the same log, which is how a val number gets
    read as a test number."""

    args = _final_test_args(do_eval=True, init_model="/tmp/bin.1")

    with pytest.raises(ValueError, match="run_final_test.*do_eval"):
        main_task_retrieval.validate_trusted_cli(args)


def test_standalone_run_final_test_rejects_init_model():
    """Pointing --init_model at a checkpoint would suggest that checkpoint is
    what gets tested. It is not -- the snapshot decides -- so the flag is
    refused rather than silently ignored."""

    args = _final_test_args(init_model="/tmp/bin.3")

    with pytest.raises(ValueError, match="run_final_test.*init_model"):
        main_task_retrieval.validate_trusted_cli(args)


def test_run_final_test_with_do_train_still_allowed():
    """The in-loop path is unchanged."""

    main_task_retrieval.validate_trusted_cli(
        _final_test_args(do_train=True, eval_split="val")
    )


def test_get_args_accepts_final_test_as_a_mode_of_its_own(monkeypatch, tmp_path):
    """Exercised through get_args, not just validate_trusted_cli.

    The first attempt at this only pinned validate_trusted_cli and passed,
    while the real entry point still died on a separate
    "at least one of do_train or do_eval" guard further down get_args. Two
    gates, one tested. This test goes through the front door.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_task_retrieval.py",
            "--run_final_test",
            "--eval_split",
            "test",
            "--output_dir",
            str(tmp_path),
        ],
    )

    parsed = main_task_retrieval.get_args()

    assert parsed.run_final_test is True
    assert parsed.do_train is False
    assert parsed.do_eval is False


def test_get_args_still_rejects_a_run_that_does_nothing(monkeypatch, tmp_path):
    """No train, no eval, no final test is not a run."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["main_task_retrieval.py", "--output_dir", str(tmp_path)],
    )

    with pytest.raises(ValueError, match="do_train.*do_eval.*run_final_test"):
        main_task_retrieval.get_args()


# --- load_best_validation_snapshot ---


def test_load_best_validation_snapshot_returns_the_selection(tmp_path):
    written = _snapshot(tmp_path)

    loaded = main_task_retrieval.load_best_validation_snapshot(tmp_path)

    assert loaded == written


def test_load_best_validation_snapshot_errors_when_missing(tmp_path):
    """A directory with no snapshot never completed a validation epoch."""

    with pytest.raises(ValueError, match="best_validation_checkpoints.json"):
        main_task_retrieval.load_best_validation_snapshot(tmp_path)


@pytest.mark.parametrize("direction", ("t2v", "v2t"))
def test_load_best_validation_snapshot_errors_on_a_missing_checkpoint(
    tmp_path, direction
):
    """Fail in a second, not after the test dataloader is built.

    A snapshot outliving its checkpoints is normal -- the bins are 400 MB
    each and get cleaned up -- so this is a real path, not a corruption case.
    """

    _snapshot(tmp_path)
    payload = json.loads(
        (tmp_path / "best_validation_checkpoints.json").read_text()
    )
    Path(payload[direction]["checkpoint"]).unlink()

    with pytest.raises(ValueError, match="missing checkpoint"):
        main_task_retrieval.load_best_validation_snapshot(tmp_path)


@pytest.mark.parametrize("missing", ("t2v", "v2t"))
def test_load_best_validation_snapshot_errors_on_a_truncated_snapshot(
    tmp_path, missing
):
    """Both directions are required; a half-written snapshot is not a
    selection we can test against."""

    _snapshot(tmp_path)
    path = tmp_path / "best_validation_checkpoints.json"
    payload = json.loads(path.read_text())
    del payload[missing]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=missing):
        main_task_retrieval.load_best_validation_snapshot(tmp_path)


def test_load_best_validation_snapshot_accepts_one_checkpoint_for_both(tmp_path):
    """When one epoch wins both directions there is one bin, tested once."""

    written = _snapshot(tmp_path, t2v_bin="bin.1", v2t_bin="bin.1")

    loaded = main_task_retrieval.load_best_validation_snapshot(tmp_path)

    assert loaded["t2v"]["checkpoint"] == loaded["v2t"]["checkpoint"]
    assert loaded == written
