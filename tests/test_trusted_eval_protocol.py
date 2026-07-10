import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

cv2_stub = types.ModuleType("cv2")
cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
sys.modules.setdefault("cv2", cv2_stub)
spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules.setdefault("modules.spatial_enhancer", spatial_enhancer_stub)

import main_task_retrieval as retrieval  # noqa: E402


def _args(**overrides):
    values = {
        "datatype": "msrvtt",
        "do_train": True,
        "do_eval": False,
        "eval_split": "val",
        "init_model": None,
        "expand_msrvtt_sentences": True,
        "experiment_profile": "hygiene",
        "final_score_mode": "wti",
        "w_mil": 0.0,
        "w_evidential": 0.0,
        "w_neg_reg": 0.0,
        "w_orth": 0.0,
        "uncertainty_mode": "none",
        "use_hard_negative_packing": False,
        "use_explicit_hard_negative_loss": False,
        "use_uacl_intra_alignment": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_rejects_test_split():
    with pytest.raises(ValueError, match="training cannot use eval_split=test"):
        retrieval.validate_trusted_cli(_args(eval_split="test"))


def test_training_requires_expanded_captions():
    with pytest.raises(ValueError, match="requires --expand_msrvtt_sentences"):
        retrieval.validate_trusted_cli(_args(expand_msrvtt_sentences=False))


def test_eval_only_requires_initial_checkpoint():
    with pytest.raises(ValueError, match="--do_eval requires --init_model"):
        retrieval.validate_trusted_cli(
            _args(do_train=False, do_eval=True, eval_split="test")
        )


@pytest.mark.parametrize("name", ["w_mil", "w_evidential", "w_neg_reg", "w_orth"])
def test_hygiene_rejects_active_auxiliary_loss(name):
    with pytest.raises(ValueError, match=rf"hygiene requires {name}=0"):
        retrieval.validate_trusted_cli(_args(**{name: 0.01}))


def test_hygiene_rejects_uncertainty_mode():
    with pytest.raises(ValueError, match="hygiene requires uncertainty_mode=none"):
        retrieval.validate_trusted_cli(_args(uncertainty_mode="evidential"))


@pytest.mark.parametrize(
    "flag",
    [
        "use_hard_negative_packing",
        "use_explicit_hard_negative_loss",
        "use_uacl_intra_alignment",
    ],
)
def test_hygiene_rejects_hn_and_uacl_paths(flag):
    with pytest.raises(ValueError, match="hygiene forbids HN and UACL paths"):
        retrieval.validate_trusted_cli(_args(**{flag: True}))


def test_hygiene_rejects_non_wti_final_score():
    with pytest.raises(ValueError, match="requires final_score_mode=wti"):
        retrieval.validate_trusted_cli(_args(final_score_mode="wti_anchor_wti"))


def test_get_args_does_not_silently_normalize_msrvtt_hygiene(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--datatype",
            "msrvtt",
            "--expand_msrvtt_sentences",
            "--experiment_profile",
            "hygiene",
            "--w_mil",
            "0.01",
        ],
    )

    with pytest.raises(ValueError, match="hygiene requires w_mil=0"):
        retrieval.get_args()


def test_get_args_exposes_trusted_data_paths_and_eval_split(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_eval",
            "--init_model",
            "checkpoint.bin",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--eval_split",
            "test",
            "--source_train_csv",
            "source-train.csv",
            "--test_csv",
            "blind-test.csv",
            "--split_manifest",
            "trusted.json",
        ],
    )

    args = retrieval.get_args()

    assert args.eval_split == "test"
    assert args.source_train_csv == "source-train.csv"
    assert args.test_csv == "blind-test.csv"
    assert args.split_manifest == "trusted.json"


def test_training_constructs_val_but_not_test(monkeypatch):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {
            "train": lambda args, tokenizer: (
                calls.append("train") or ("train", 1, "sampler")
            ),
            "val": lambda args, tokenizer, subset="val": (
                calls.append("val") or ("val", 2)
            ),
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(_args(), object())

    assert result == (("train", 1, "sampler"), "val", 2, "val")
    assert calls == ["train", "val"]


@pytest.mark.parametrize("eval_split", ["val", "test"])
def test_eval_only_constructs_only_requested_split(monkeypatch, eval_split):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {
            "train": lambda args, tokenizer: calls.append("train"),
            "val": lambda args, tokenizer, subset="val": (
                calls.append("val") or ("val", 2)
            ),
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(
        _args(
            do_train=False,
            do_eval=True,
            eval_split=eval_split,
            init_model="checkpoint.bin",
        ),
        object(),
    )

    expected_length = 2 if eval_split == "val" else 3
    assert result == (None, eval_split, expected_length, eval_split)
    assert calls == [eval_split]


def test_msrvtt_training_rejects_missing_val_loader(monkeypatch):
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {"train": lambda args, tokenizer: ("train", 1, "sampler"), "test": None},
    )

    with pytest.raises(ValueError, match="msrvtt has no val dataloader"):
        retrieval.prepare_requested_dataloaders(_args(), object())


def test_eval_only_rejects_missing_requested_loader(monkeypatch):
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {"train": object(), "val": object(), "test": None},
    )

    with pytest.raises(ValueError, match="msrvtt has no test dataloader"):
        retrieval.prepare_requested_dataloaders(
            _args(
                do_train=False,
                do_eval=True,
                eval_split="test",
                init_model="checkpoint.bin",
            ),
            object(),
        )


def test_non_msrvtt_training_retains_test_fallback_when_val_is_missing(monkeypatch):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "legacy_dataset",
        {
            "train": lambda args, tokenizer: (
                calls.append("train") or ("train", 1, "sampler")
            ),
            "val": None,
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(
        _args(datatype="legacy_dataset", experiment_profile="default"), object()
    )

    assert result == (("train", 1, "sampler"), "test", 3, "test")
    assert calls == ["train", "test"]


def test_checkpoint_selection_prefers_higher_internal_val_score():
    assert retrieval.select_best_checkpoint(
        48.0, "epoch1.bin", 48.5, "epoch2.bin"
    ) == (48.5, "epoch2.bin")
    assert retrieval.select_best_checkpoint(
        48.5, "epoch2.bin", 48.1, "epoch3.bin"
    ) == (48.5, "epoch2.bin")


def test_checkpoint_selection_tie_prefers_later_checkpoint():
    assert retrieval.select_best_checkpoint(
        48.5, "epoch2.bin", 48.5, "epoch3.bin"
    ) == (48.5, "epoch3.bin")


def test_training_evaluation_errors_propagate(monkeypatch):
    def fail_eval(*args, **kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(retrieval, "eval_epoch", fail_eval)

    with pytest.raises(RuntimeError, match="validation failed"):
        retrieval.evaluate_training_checkpoint(
            args=object(),
            model=object(),
            eval_dataloader=object(),
            device=object(),
            n_gpu=1,
            best_score=48.0,
            best_path="epoch1.bin",
            candidate_path="epoch2.bin",
        )


def test_multi_sentence_eval_selects_one_video_row_per_caption_group():
    cut_off_points = [20, 40]
    assert retrieval.select_multi_sentence_video_rows(0, 25, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(25, 15, cut_off_points) == [14]


def test_multi_sentence_eval_handles_group_ends_on_batch_boundaries():
    cut_off_points = [20, 40, 60]
    assert retrieval.select_multi_sentence_video_rows(0, 20, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(20, 20, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(40, 20, cut_off_points) == [19]
