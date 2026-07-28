"""Which videos a run trains on, and which set it then selects on.

The trusted-v1 protocol holds 500 of the 9000 source train videos out as a
local validation set, and the train dataloader enforces that the train CSV is
exactly the remaining 8500. The official UATVR recipe trains on all 9000 and
reports on JSFUSION test, so an official-parity run has to be able to say
"fold the held-out 500 back in" -- explicitly, and without loosening the
equality check that makes the protocol auditable. Folding them in also means
there is no held-out val left, so the eval loader is the reported test set.
"""

import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

import dataloaders.data_dataloaders as data_dataloaders
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader

CAPTIONS_PER_VIDEO = 2
TRAIN_IDS = ["video7", "video3", "video5"]
VAL_IDS = ["video1", "video9"]


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest():
    """A shape-valid trusted manifest. Only the ID lists matter to the loader."""

    return {
        "protocol_version": "trusted-v1",
        "seed": 0,
        "algorithm": "test fixture",
        "val_size": len(VAL_IDS),
        "expected_captions_per_video": CAPTIONS_PER_VIDEO,
        "source_sha256": {
            "train_csv": _digest("train"),
            "annotation_json": _digest("annotation"),
            "test_csv": _digest("test"),
        },
        "counts": {
            "source_train_videos": len(TRAIN_IDS) + len(VAL_IDS),
            "train_videos": len(TRAIN_IDS),
            "val_videos": len(VAL_IDS),
            "val_sentences": len(VAL_IDS) * CAPTIONS_PER_VIDEO,
            "test_csv_rows": 0,
            "test_videos": 0,
        },
        "overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "test_video_ids_sha256": _digest("test-ids"),
        "train_video_ids": list(TRAIN_IDS),
        "val_video_ids": list(VAL_IDS),
    }


def _write_csv(path, video_ids):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id"])
        writer.writeheader()
        writer.writerows({"video_id": video_id} for video_id in video_ids)
    return str(path)


def _write_annotation(path, video_ids):
    payload = {
        "videos": [{"video_id": video_id} for video_id in video_ids],
        "sentences": [
            {"video_id": video_id, "caption": f"{video_id} caption {index}"}
            for video_id in video_ids
            for index in range(CAPTIONS_PER_VIDEO)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _build(tmp_path, csv_video_ids, **overrides):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    features_path = tmp_path / "videos"
    features_path.mkdir(exist_ok=True)
    kwargs = {
        "csv_path": _write_csv(tmp_path / "train.csv", csv_video_ids),
        "json_path": _write_annotation(
            tmp_path / "annotation.json", TRAIN_IDS + VAL_IDS
        ),
        "features_path": str(features_path),
        "tokenizer": None,
        "unfold_sentences": True,
        "split_manifest_path": str(manifest_path),
        "expected_captions_per_video": CAPTIONS_PER_VIDEO,
        "slice_framepos": 2,
    }
    kwargs.update(overrides)
    return MSRVTT_TrainDataLoader(**kwargs)


def test_trusted_scope_still_rejects_the_full_source_train_csv(tmp_path):
    """The default scope is unchanged: 8500 means 8500."""

    with pytest.raises(ValueError, match="train_video_ids"):
        _build(tmp_path, TRAIN_IDS + VAL_IDS)


def test_folding_val_into_train_accepts_the_full_source_train_csv(tmp_path):
    """Official parity trains on all 9000 source videos."""

    dataset = _build(tmp_path, TRAIN_IDS + VAL_IDS, fold_val_into_train=True)

    assert set(dataset.video_group_ids) == set(TRAIN_IDS) | set(VAL_IDS)


def test_folding_val_into_train_keeps_the_original_train_group_ids(tmp_path):
    """Group IDs stay manifest-derived and deterministic.

    The held-out videos are appended after the trusted train IDs rather than
    interleaved, so a video's group ID means the same thing in both scopes.
    """

    dataset = _build(tmp_path, TRAIN_IDS + VAL_IDS, fold_val_into_train=True)

    assert dataset.video_group_ids == {
        "video7": 0,
        "video3": 1,
        "video5": 2,
        "video1": 3,
        "video9": 4,
    }


def test_folding_val_into_train_trains_on_every_folded_caption(tmp_path):
    dataset = _build(tmp_path, TRAIN_IDS + VAL_IDS, fold_val_into_train=True)

    assert len(dataset.sentences_dict) == (
        (len(TRAIN_IDS) + len(VAL_IDS)) * CAPTIONS_PER_VIDEO
    )


def test_folding_val_into_train_still_rejects_a_csv_that_is_not_the_source(tmp_path):
    """Widening the scope is not the same as dropping the check."""

    with pytest.raises(ValueError, match="extra=\\['video404'\\]"):
        _build(tmp_path, TRAIN_IDS + VAL_IDS + ["video404"], fold_val_into_train=True)


def test_folding_val_into_train_names_the_scope_it_expected(tmp_path):
    """The error has to say which of the two ID sets it compared against."""

    with pytest.raises(ValueError, match="train_video_ids\\+val_video_ids"):
        _build(tmp_path, TRAIN_IDS, fold_val_into_train=True)


def _eval_loader_kwargs(monkeypatch, **argument_overrides):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(data_dataloaders, "MSRVTT_DataLoader", _capture)
    monkeypatch.setattr(data_dataloaders, "_build_msrvtt_eval_loader", lambda *_: None)
    arguments = {
        "val_csv": "val.csv",
        "features_path": "videos",
        "max_words": 32,
        "feature_framerate": 1,
        "max_frames": 12,
        "eval_frame_order": 0,
        "slice_framepos": 2,
        "num_thread_reader": 0,
        "batch_size_val": 16,
        "fold_val_into_train": False,
    }
    arguments.update(argument_overrides)
    data_dataloaders.dataloader_msrvtt_val(SimpleNamespace(**arguments), tokenizer=None)
    return captured


def test_trusted_val_loader_requires_twenty_contiguous_captions(monkeypatch):
    kwargs = _eval_loader_kwargs(monkeypatch)

    assert kwargs["multi_sentence_per_video"] is True
    assert kwargs["expected_captions_per_video"] == 20


def test_parity_selects_on_jsfusion_which_has_one_caption_per_video(monkeypatch):
    """Folding val into train leaves no held-out set to select on.

    The parity eval CSV is JSFUSION test: 1000 rows, one caption each. Asking
    it for the trusted val set's 20-caption layout is what killed the run.
    """

    kwargs = _eval_loader_kwargs(monkeypatch, fold_val_into_train=True)

    assert kwargs["multi_sentence_per_video"] is False
    assert kwargs["expected_captions_per_video"] is None
