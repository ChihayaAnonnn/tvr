"""--resume_from picks up a training run that was interrupted mid-loop.

hygiene_a0_seed0 lost epochs 4-5 to a CUDA OOM caused by another user's
vLLM job landing on GPU 0. The recipe was healthy, the split was healthy,
five hours of training were already on disk, and rerunning from epoch 1 to
recover the last two costs another three hours. This test module pins the
contract for the minimal resume path -- reload the latest saved (model,
optimizer, best-val) triple, continue the epoch loop, refuse anything that
would silently change the trajectory.

What "minimal" means and does not:

* We rely on BertAdam carrying its own step counter in the optimizer state,
  so loading opt.bin.N restores the warmup+cosine LR position exactly. No
  separate scheduler snapshot is written.
* We do NOT snapshot RNG state, DDP sampler internals, or dataloader worker
  state. A resumed run is not bit-for-bit identical to an uninterrupted one;
  it is one round of training that was paused. RSPR arm comparability rests
  on comparing to the same baseline, not on the baseline being bitwise
  reproducible against itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main_task_retrieval


# Enough of the manifest for resolve_resume_target to find its recipe fields.
# The real manifest carries a lot more (rspr knobs, git state, split hashes)
# but resume only rejects the drift that would change the training curve.
_MANIFEST_TEMPLATE = {
    "batch": {
        "requested_effective_batch": 512,
        "gradient_accumulation_steps": 1,
    },
    "optimization": {
        "coef_lr": 1e-3,
        "epochs": 5,
        "freeze_layer_num": 0,
        "lr": 5e-5,
        "max_frames": 12,
        "max_words": 32,
        "slice_framepos": 2,
    },
    "profile": "hygiene",
    "seed": 0,
    "split": {
        "manifest_sha256": "a" * 64,
    },
}


def _write_manifest(directory: Path, **section_overrides) -> Path:
    """Persist a manifest that only differs from the healthy template in the
    supplied section overrides -- each override is a (section, key, value)
    triple applied on top of a deep-copied template.
    """

    manifest = json.loads(json.dumps(_MANIFEST_TEMPLATE))
    for section, key, value in section_overrides.get("overrides", ()):
        if section is None:
            manifest[key] = value
        else:
            manifest[section][key] = value
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _touch(path: Path, payload: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _healthy_ckpt_dir(tmp_path: Path, epochs_completed: int = 3) -> Path:
    """A checkpoint directory that looks like a run of ours that got N epochs
    in before something killed it. bin.k / opt.bin.k for k in [0, N-1].
    """

    directory = tmp_path / "ckpt_run"
    _write_manifest(directory)
    for epoch in range(epochs_completed):
        _touch(directory / f"pytorch_model.bin.{epoch}", f"model-{epoch}")
        _touch(directory / f"pytorch_opt.bin.{epoch}", f"opt-{epoch}")
    return directory


def _healthy_args(**overrides):
    values = {
        "resume_from": "",
        "init_model": "",
        "resume_model": "",
        "experiment_profile": "hygiene",
        "batch_size": 512,
        "gradient_accumulation_steps": 1,
        "epochs": 5,
        "lr": 5e-5,
        "coef_lr": 1e-3,
        "freeze_layer_num": 0,
        "max_frames": 12,
        "max_words": 32,
        "slice_framepos": 2,
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- resolve_resume_target ---


def test_resolve_resume_target_picks_the_highest_saved_epoch(tmp_path):
    """The 'latest' epoch is the newest one whose model AND opt files exist."""

    directory = _healthy_ckpt_dir(tmp_path, epochs_completed=3)

    resume = main_task_retrieval.resolve_resume_target(directory, _healthy_args())

    assert resume.model_path == directory / "pytorch_model.bin.2"
    assert resume.opt_path == directory / "pytorch_opt.bin.2"
    # We resumed from epoch 2 completed, so the loop should start at epoch 3.
    # The "resume" semantics attach to the epoch counter, not the loop index.
    assert resume.start_epoch == 3
    assert resume.epochs_completed == 3


def test_resolve_resume_target_ignores_bins_without_a_matching_optimizer(tmp_path):
    """A bin.k without opt.bin.k cannot restore the BertAdam step counter.

    Silently skipping it and picking bin.k-1 would resume with a mid-epoch
    LR and no way to tell -- so bin.k is discarded and the next candidate
    down is tried instead.
    """

    directory = _healthy_ckpt_dir(tmp_path, epochs_completed=3)
    _touch(directory / "pytorch_model.bin.3", "model-3")
    # No opt.bin.3.

    resume = main_task_retrieval.resolve_resume_target(directory, _healthy_args())

    assert resume.start_epoch == 3  # still resumes at 3, not 4


def test_resolve_resume_target_errors_when_no_bin_survives(tmp_path):
    directory = tmp_path / "ckpt_run"
    _write_manifest(directory)

    with pytest.raises(ValueError, match="no saved epochs"):
        main_task_retrieval.resolve_resume_target(directory, _healthy_args())


def test_resolve_resume_target_errors_when_the_manifest_is_missing(tmp_path):
    directory = tmp_path / "ckpt_run"
    directory.mkdir()
    _touch(directory / "pytorch_model.bin.0", "model-0")
    _touch(directory / "pytorch_opt.bin.0", "opt-0")

    with pytest.raises(ValueError, match="experiment_manifest.json"):
        main_task_retrieval.resolve_resume_target(directory, _healthy_args())


def test_resolve_resume_target_errors_when_the_directory_does_not_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        main_task_retrieval.resolve_resume_target(
            tmp_path / "nope", _healthy_args()
        )


def test_resolve_resume_target_errors_when_the_run_already_finished(tmp_path):
    """A directory where every epoch is on disk has nothing to resume.

    Overwriting it with another training loop would burn a fresh 3 hours to
    reproduce the same checkpoints. Refuse with a message that says so.
    """

    directory = _healthy_ckpt_dir(tmp_path, epochs_completed=5)  # epochs=5

    with pytest.raises(ValueError, match="already completed"):
        main_task_retrieval.resolve_resume_target(directory, _healthy_args())


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        (None, "seed", 1, "seed"),
        (None, "profile", "parity", "profile"),
        ("batch", "requested_effective_batch", 256, "batch"),
        ("batch", "gradient_accumulation_steps", 2, "gradient_accumulation_steps"),
        ("optimization", "epochs", 10, "epochs"),
        ("optimization", "lr", 1e-4, "lr"),
        ("optimization", "coef_lr", 1.0, "coef_lr"),
        ("optimization", "freeze_layer_num", 8, "freeze_layer_num"),
        ("optimization", "max_frames", 8, "max_frames"),
        ("optimization", "max_words", 20, "max_words"),
        ("optimization", "slice_framepos", 3, "slice_framepos"),
    ),
)
def test_resolve_resume_target_rejects_recipe_drift(
    tmp_path, section, key, value, message
):
    """Resume is not a license to change the recipe.

    An OOM at epoch 4 means we get epochs 4-5 with the SAME recipe or we
    admit we're producing a different number. Anything else is exactly the
    provenance failure that cost 3.2 R@1 to diagnose.
    """

    directory = tmp_path / "ckpt_run"
    _write_manifest(directory, overrides=((section, key, value),))
    _touch(directory / "pytorch_model.bin.0", "model-0")
    _touch(directory / "pytorch_opt.bin.0", "opt-0")

    with pytest.raises(ValueError, match=message):
        main_task_retrieval.resolve_resume_target(directory, _healthy_args())


def test_resolve_resume_target_loads_best_val_trackers_when_present(tmp_path):
    """Per-epoch snapshots of the trackers survive an OOM.

    Without them the resumed run would restart best_val at -inf and could
    'select' an epoch that scores lower than one already discarded.
    """

    directory = _healthy_ckpt_dir(tmp_path, epochs_completed=3)
    (directory / "best_validation_checkpoints.json").write_text(
        json.dumps(
            {
                "selection_split": "val",
                "epochs_completed": 3,
                "t2v": {
                    "selection_metric": "t2v_r1",
                    "selection_score": 59.0,
                    "checkpoint": str(directory / "pytorch_model.bin.1"),
                },
                "v2t": {
                    "selection_metric": "v2t_r1",
                    "selection_score": 87.8,
                    "checkpoint": str(directory / "pytorch_model.bin.2"),
                },
            }
        )
    )

    resume = main_task_retrieval.resolve_resume_target(directory, _healthy_args())

    assert resume.best_val is not None
    assert resume.best_val["t2v"]["selection_score"] == 59.0
    assert resume.best_val["v2t"]["selection_score"] == 87.8


def test_resolve_resume_target_returns_none_best_val_when_absent(tmp_path):
    """If the source run died before any epoch's tracker was written, resume
    starts the trackers over rather than pretending they were preserved.
    """

    directory = _healthy_ckpt_dir(tmp_path, epochs_completed=1)

    resume = main_task_retrieval.resolve_resume_target(directory, _healthy_args())

    assert resume.best_val is None


# --- validate_trusted_cli mutual exclusion ---


def _trusted_args_with_resume(**overrides):
    values = {
        "experiment_profile": "hygiene",
        "datatype": "msrvtt",
        "do_train": True,
        "do_eval": False,
        "init_model": "",
        "resume_model": "",
        "resume_from": "",
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
        "fold_val_into_train": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resume_from_and_init_model_are_mutually_exclusive():
    """Both flags try to seed the model weights; letting both through would
    silently pick one and hide the ambiguity."""

    args = _trusted_args_with_resume(
        resume_from="/tmp/some_ckpt_dir", init_model="/tmp/other/bin.0"
    )

    with pytest.raises(ValueError, match="resume_from.*init_model"):
        main_task_retrieval.validate_trusted_cli(args)


def test_resume_from_and_resume_model_are_mutually_exclusive():
    """resume_model is the older opt-only path; resume_from supersedes it."""

    args = _trusted_args_with_resume(
        resume_from="/tmp/some_ckpt_dir", resume_model="/tmp/other/opt.bin.0"
    )

    with pytest.raises(ValueError, match="resume_from.*resume_model"):
        main_task_retrieval.validate_trusted_cli(args)


# --- write_best_validation_snapshot ---


def test_write_best_validation_snapshot_writes_after_every_epoch(tmp_path):
    """Per-epoch write is what makes resume possible after an OOM.

    Writing only at end-of-loop -- as the old code did -- leaves nothing on
    disk when the loop is interrupted, and the resumed run cannot tell that
    epoch 2 already beat what epoch 4 might produce.
    """

    main_task_retrieval.write_best_validation_snapshot(
        tmp_path,
        epochs_completed=2,
        best_t2v_score=59.0,
        best_t2v_checkpoint=str(tmp_path / "pytorch_model.bin.1"),
        best_v2t_score=87.0,
        best_v2t_checkpoint=str(tmp_path / "pytorch_model.bin.1"),
    )

    payload = json.loads(
        (tmp_path / "best_validation_checkpoints.json").read_text()
    )
    assert payload["epochs_completed"] == 2
    assert payload["t2v"]["selection_score"] == 59.0
    assert payload["v2t"]["selection_score"] == 87.0
    assert payload["selection_split"] == "val"
    assert payload["tie_break"] == "later_epoch"


def test_write_best_validation_snapshot_returns_what_it_wrote(tmp_path):
    """The writer returns the payload so callers need not re-read the file."""

    returned = main_task_retrieval.write_best_validation_snapshot(
        tmp_path,
        epochs_completed=2,
        best_t2v_score=59.0,
        best_t2v_checkpoint=str(tmp_path / "pytorch_model.bin.1"),
        best_v2t_score=87.0,
        best_v2t_checkpoint=str(tmp_path / "pytorch_model.bin.1"),
    )

    on_disk = json.loads(
        (tmp_path / "best_validation_checkpoints.json").read_text()
    )
    assert returned == on_disk


# --- build_best_validation_payload ---


@pytest.mark.parametrize("direction", ("t2v", "v2t"))
def test_build_best_validation_payload_carries_the_final_test_keys(direction):
    """The final test spreads each direction's dict into its own record.

    It reads "checkpoint" to decide what to load and keeps the rest as
    provenance, so those keys are a contract between the two call sites --
    not an implementation detail of the snapshot file.
    """

    payload = main_task_retrieval.build_best_validation_payload(
        epochs_completed=5,
        best_t2v_score=59.0,
        best_t2v_checkpoint="ckpts/run/pytorch_model.bin.1",
        best_v2t_score=87.8,
        best_v2t_checkpoint="ckpts/run/pytorch_model.bin.2",
    )

    assert set(payload[direction]) == {
        "selection_metric",
        "selection_score",
        "checkpoint",
    }


def test_build_best_validation_payload_keeps_the_directions_apart():
    """T2V and V2T select independently; a run usually ends with two different
    checkpoints, and swapping them would silently report the wrong number."""

    payload = main_task_retrieval.build_best_validation_payload(
        epochs_completed=5,
        best_t2v_score=59.0,
        best_t2v_checkpoint="ckpts/run/pytorch_model.bin.1",
        best_v2t_score=87.8,
        best_v2t_checkpoint="ckpts/run/pytorch_model.bin.2",
    )

    assert payload["t2v"]["checkpoint"] == "ckpts/run/pytorch_model.bin.1"
    assert payload["t2v"]["selection_score"] == 59.0
    assert payload["v2t"]["checkpoint"] == "ckpts/run/pytorch_model.bin.2"
    assert payload["v2t"]["selection_score"] == 87.8
