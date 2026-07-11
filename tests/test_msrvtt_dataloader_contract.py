import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler

import dataloaders.data_dataloaders as builders
import dataloaders.dataloader_msrvtt_retrieval as msrvtt
from dataloaders.dataloader_msrvtt_retrieval import (
    MSRVTT_DataLoader,
    MSRVTT_TrainDataLoader,
)
from dataloaders.rawvideo_util import RawVideoExtractor


class Tokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, words):
        return list(range(1, len(words) + 1))


def _write_manifest(path: Path, train_video_ids):
    path.write_text(
        json.dumps(
            {
                "protocol_version": "trusted-v1",
                "seed": 42,
                "algorithm": "test fixture",
                "val_size": 0,
                "expected_captions_per_video": 20,
                "source_sha256": {
                    "train_csv": "0" * 64,
                    "annotation_json": "1" * 64,
                    "test_csv": "2" * 64,
                },
                "counts": {
                    "source_train_videos": len(train_video_ids),
                    "train_videos": len(train_video_ids),
                    "val_videos": 0,
                    "val_sentences": 0,
                    "test_csv_rows": 0,
                    "test_videos": 0,
                },
                "overlap_counts": {
                    "train_val": 0,
                    "train_test": 0,
                    "val_test": 0,
                },
                "test_video_ids_sha256": "3" * 64,
                "train_video_ids": train_video_ids,
                "val_video_ids": [],
            }
        ),
        encoding="utf-8",
    )


def _write_train_fixture(root: Path, captions_per_video=20):
    train_csv = root / "train.csv"
    train_csv.write_text("video_id\nvideo_b\nvideo_a\n", encoding="utf-8")
    annotation = root / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "videos": [],
                "sentences": [
                    {"video_id": video_id, "caption": f"{video_id}-{index}"}
                    for video_id in ("video_b", "video_a")
                    for index in range(captions_per_video)
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    _write_manifest(manifest, ["video_a", "video_b"])
    return train_csv, annotation, manifest


def _fake_rawvideo(_self, video_ids):
    return (
        np.zeros((len(video_ids), 1, 1, 3, 2, 2), dtype=np.float32),
        np.ones((len(video_ids), 1), dtype=np.int64),
    )


def _build_train_dataset(
    tmp_path,
    monkeypatch,
    *,
    use_attributes=False,
    return_sample_index=False,
    return_hard_negative=False,
):
    train_csv, annotation, manifest = _write_train_fixture(tmp_path)
    monkeypatch.setattr(MSRVTT_TrainDataLoader, "_get_rawvideo", _fake_rawvideo)
    monkeypatch.setattr(
        msrvtt,
        "load_hard_negative_index",
        lambda _path, sample_len: [-1] * sample_len,
    )
    return MSRVTT_TrainDataLoader(
        csv_path=train_csv,
        json_path=annotation,
        features_path=tmp_path,
        tokenizer=Tokenizer(),
        max_frames=1,
        unfold_sentences=True,
        use_attributes=use_attributes,
        return_sample_index=return_sample_index,
        return_hard_negative=return_hard_negative,
        split_manifest_path=manifest,
    )


def test_train_sample_returns_stable_manifest_group_id(tmp_path, monkeypatch):
    dataset = _build_train_dataset(tmp_path, monkeypatch)

    assert len(dataset) == 40
    assert isinstance(dataset[0][-1], np.int64)
    assert int(dataset[0][-1]) == 1
    assert int(dataset[20][-1]) == 0


@pytest.mark.parametrize(
    (
        "use_attributes",
        "return_sample_index",
        "return_hard_negative",
        "expected_length",
        "sample_index_position",
    ),
    [
        (False, False, False, 6, None),
        (True, False, False, 9, None),
        (False, False, True, 10, 5),
        (True, False, True, 13, 8),
    ],
)
def test_train_return_contract_always_appends_group_id(
    tmp_path,
    monkeypatch,
    use_attributes,
    return_sample_index,
    return_hard_negative,
    expected_length,
    sample_index_position,
):
    dataset = _build_train_dataset(
        tmp_path,
        monkeypatch,
        use_attributes=use_attributes,
        return_sample_index=return_sample_index,
        return_hard_negative=return_hard_negative,
    )

    sample = dataset[20]

    assert len(sample) == expected_length
    assert isinstance(sample[-1], np.int64)
    assert int(sample[-1]) == 0
    if sample_index_position is not None:
        assert isinstance(sample[sample_index_position], np.int64)
        assert int(sample[sample_index_position]) == 20
        assert sample[sample_index_position] != sample[-1]


def test_train_rejects_sample_index_without_explicit_hard_negative(
    tmp_path, monkeypatch
):
    train_csv, annotation, manifest = _write_train_fixture(tmp_path)
    monkeypatch.setattr(MSRVTT_TrainDataLoader, "_get_rawvideo", _fake_rawvideo)

    with pytest.raises(
        ValueError, match="sample_index is only available with explicit hard negatives"
    ):
        MSRVTT_TrainDataLoader(
            csv_path=train_csv,
            json_path=annotation,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            unfold_sentences=True,
            return_sample_index=True,
            return_hard_negative=False,
            split_manifest_path=manifest,
        )


def test_train_requires_unfolded_sentences(tmp_path):
    train_csv, annotation, manifest = _write_train_fixture(tmp_path)

    with pytest.raises(ValueError, match="requires unfold_sentences=True"):
        MSRVTT_TrainDataLoader(
            csv_path=train_csv,
            json_path=annotation,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            unfold_sentences=False,
            split_manifest_path=manifest,
        )


def test_train_requires_manifest(tmp_path):
    train_csv, annotation, _manifest = _write_train_fixture(tmp_path)

    with pytest.raises(ValueError, match="requires split_manifest_path"):
        MSRVTT_TrainDataLoader(
            csv_path=train_csv,
            json_path=annotation,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            unfold_sentences=True,
        )


def test_train_rejects_csv_manifest_mismatch(tmp_path):
    train_csv, annotation, manifest = _write_train_fixture(tmp_path)
    _write_manifest(manifest, ["video_a", "video_c"])

    with pytest.raises(ValueError, match="train CSV video IDs do not match"):
        MSRVTT_TrainDataLoader(
            csv_path=train_csv,
            json_path=annotation,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            unfold_sentences=True,
            split_manifest_path=manifest,
        )


def test_train_rejects_wrong_caption_count(tmp_path):
    train_csv, annotation, manifest = _write_train_fixture(
        tmp_path, captions_per_video=19
    )

    with pytest.raises(ValueError, match="expected 20 captions, got 19"):
        MSRVTT_TrainDataLoader(
            csv_path=train_csv,
            json_path=annotation,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            unfold_sentences=True,
            split_manifest_path=manifest,
        )


def _write_eval_csv(path: Path, groups):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        for video_id, count in groups:
            for index in range(count):
                writer.writerow(
                    {"video_id": video_id, "sentence": f"{video_id}-{index}"}
                )


def test_val_loader_exposes_multi_sentence_metadata(tmp_path):
    val_csv = tmp_path / "val.csv"
    _write_eval_csv(val_csv, [("video_a", 20), ("video_b", 20)])

    dataset = MSRVTT_DataLoader(
        csv_path=val_csv,
        features_path=tmp_path,
        tokenizer=Tokenizer(),
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )

    assert dataset.multi_sentence_per_video is True
    assert dataset.cut_off_points == [20, 40]
    assert dataset.sentence_num == 40
    assert dataset.video_num == 2


def test_val_loader_rejects_non_contiguous_video_rows(tmp_path):
    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "video_id,sentence\nvideo_a,a\nvideo_b,b\nvideo_a,c\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        MSRVTT_DataLoader(
            csv_path=val_csv,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            multi_sentence_per_video=True,
            expected_captions_per_video=20,
        )


def test_val_loader_rejects_wrong_caption_count(tmp_path):
    val_csv = tmp_path / "val.csv"
    _write_eval_csv(val_csv, [("video_a", 20), ("video_b", 19)])

    with pytest.raises(ValueError, match="each val video must have 20 captions"):
        MSRVTT_DataLoader(
            csv_path=val_csv,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            multi_sentence_per_video=True,
            expected_captions_per_video=20,
        )


def _builder_args(tmp_path, val_csv, test_csv):
    return SimpleNamespace(
        train_csv=str(tmp_path / "train.csv"),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        split_manifest=str(tmp_path / "manifest.json"),
        data_path=str(tmp_path / "annotation.json"),
        features_path=str(tmp_path),
        max_words=30,
        max_words_attrs=30,
        feature_framerate=1.0,
        max_frames=1,
        train_frame_order=0,
        eval_frame_order=0,
        slice_framepos=0,
        strategy=1,
        use_attributes=False,
        msrvtt_attributes_path="",
        attr_num_blocks=4,
        use_explicit_hard_negative_loss=False,
        use_hard_negative_packing=False,
        hard_negative_path="",
        expand_msrvtt_sentences=True,
        batch_size=4,
        batch_size_val=7,
        n_gpu=2,
        world_size=2,
        rank=0,
        num_thread_reader=0,
        tqfs_cache_dir="",
    )


def test_val_and_test_builders_use_distinct_csv_and_sequential_sampling(tmp_path):
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _write_eval_csv(val_csv, [("video_a", 20), ("video_b", 20)])
    _write_eval_csv(test_csv, [("video_test", 1)])
    args = _builder_args(tmp_path, val_csv, test_csv)

    val_loader, val_length = builders.dataloader_msrvtt_val(
        args, Tokenizer(), subset="val"
    )
    test_loader, test_length = builders.dataloader_msrvtt_test(
        args, Tokenizer(), subset="test"
    )

    assert val_length == 40
    assert val_loader.dataset.video_ids[:20] == ["video_a"] * 20
    assert val_loader.dataset.multi_sentence_per_video is True
    assert val_loader.dataset.expected_captions_per_video == 20
    assert test_length == 1
    assert test_loader.dataset.video_ids == ["video_test"]
    assert test_loader.dataset.multi_sentence_per_video is False
    for loader in (val_loader, test_loader):
        assert isinstance(loader.sampler, SequentialSampler)
        assert loader.batch_size == 7
        assert loader.drop_last is False

    assert builders.DATALOADER_DICT["msrvtt"]["val"] is builders.dataloader_msrvtt_val
    assert builders.DATALOADER_DICT["msrvtt"]["test"] is builders.dataloader_msrvtt_test


class _DummyTrainDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor(index)


def test_train_builder_passes_manifest_and_uses_distributed_sampler(
    tmp_path, monkeypatch
):
    captured = {}
    dataset = _DummyTrainDataset()

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return dataset

    sampler_calls = []

    def fake_distributed_sampler(value):
        sampler_calls.append(value)
        return SequentialSampler(value)

    monkeypatch.setattr(builders, "MSRVTT_TrainDataLoader", fake_dataset)
    monkeypatch.setattr(
        builders.torch.utils.data.distributed,
        "DistributedSampler",
        fake_distributed_sampler,
    )
    args = _builder_args(tmp_path, tmp_path / "val.csv", tmp_path / "test.csv")

    loader, length, sampler = builders.dataloader_msrvtt_train(args, Tokenizer())

    assert captured["split_manifest_path"] == args.split_manifest
    assert captured["tqfs_cache_dir"] == ""
    assert captured["unfold_sentences"] is True
    assert sampler_calls == [dataset]
    assert sampler is loader.sampler
    assert length == 4
    assert loader.drop_last is True
    assert loader.batch_size == 2
    assert loader.pin_memory is torch.cuda.is_available()
    assert loader.persistent_workers is False


def test_video_loader_worker_settings_limit_oversubscription():
    kwargs = builders._video_loader_kwargs(
        SimpleNamespace(num_thread_reader=8, prefetch_factor=4)
    )

    assert kwargs["num_workers"] == 8
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4
    assert kwargs["worker_init_fn"] is builders._configure_video_worker


def test_video_loader_zero_workers_omits_multiprocessing_options():
    kwargs = builders._video_loader_kwargs(
        SimpleNamespace(num_thread_reader=0, prefetch_factor=4)
    )

    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs
    assert "worker_init_fn" not in kwargs


def test_tqfs_dependency_fails_fast_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setattr(msrvtt.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires scikit-learn"):
        msrvtt._validate_tqfs_dependency(3)

    msrvtt._validate_tqfs_dependency(2)


def test_tqfs_video_path_decodes_once_and_preprocesses_selected_frames():
    extractor = RawVideoExtractor.__new__(RawVideoExtractor)
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(3)]
    calls = []

    def get_raw_video_data(_path, start_time=None, end_time=None):
        calls.append((start_time, end_time))
        return {"video": frames}

    extractor.get_raw_video_data = get_raw_video_data
    extractor.preprocess_raw_frames = lambda selected: torch.zeros(
        len(selected), 1, 3, 2, 2
    )

    result = extractor.get_tqfs_video_data("video.mp4", num_frames=2)

    assert calls == [(None, None)]
    assert result["video"].shape == (2, 3, 2, 2)


def test_msvd_builder_registry_is_unchanged():
    assert builders.DATALOADER_DICT["msvd"] == {
        "train": builders.dataloader_msvd_train,
        "val": builders.dataloader_msvd_test,
        "test": builders.dataloader_msvd_test,
    }
