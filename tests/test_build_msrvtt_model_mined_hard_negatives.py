import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

import scripts.build_msrvtt_model_mined_hard_negatives as model_miner
from scripts.build_msrvtt_model_mined_hard_negatives import (
    CaptionSample,
    ModelHardNegativeConfig,
    build_model_mined_mapping,
    build_task_args_for_checkpoint,
    mine_mapping_with_model,
    select_hard_negative_candidate,
    select_query_samples,
    text_pair_metrics,
)


def test_parse_args_requires_checkpoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])

    with pytest.raises(SystemExit) as exc_info:
        model_miner.parse_args()

    assert exc_info.value.code == 2


def test_parse_args_defaults_to_generated_trusted_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--checkpoint", "checkpoint.bin"],
    )

    args = model_miner.parse_args()

    assert args.train_csv.endswith("data/generated/msrvtt_trusted_v1/train.csv")
    assert args.val_csv.endswith("data/generated/msrvtt_trusted_v1/val.csv")
    assert "JSFUSION_test" not in args.val_csv
    assert args.source_train_csv.endswith("csv/MSRVTT_train.9k.csv")
    assert args.test_csv.endswith("csv/MSRVTT_JSFUSION_test.csv")
    assert args.split_manifest.endswith(
        "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
    )


def test_builder_task_args_use_supported_trusted_protocol_fields(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--checkpoint",
            "checkpoint.bin",
            "--output",
            str(tmp_path / "mapping.json"),
            "--train_csv",
            str(tmp_path / "train.csv"),
            "--source_train_csv",
            str(tmp_path / "source.csv"),
            "--test_csv",
            str(tmp_path / "test.csv"),
            "--split_manifest",
            str(tmp_path / "manifest.json"),
            "--val_csv",
            str(tmp_path / "val.csv"),
        ],
    )
    args = model_miner.parse_args()

    task_args = build_task_args_for_checkpoint(args)

    assert task_args.train_csv == args.train_csv
    assert task_args.source_train_csv == args.source_train_csv
    assert task_args.test_csv == args.test_csv
    assert task_args.split_manifest == args.split_manifest
    assert task_args.val_csv == args.val_csv
    assert task_args.eval_split == "val"
    for retired_name in (
        "final_score_mode",
        "uncertainty_mode",
        "n_video_embeddings",
        "w_evidential",
    ):
        assert not hasattr(task_args, retired_name)


def test_main_runs_trusted_gate_before_dataset_or_checkpoint_access(monkeypatch):
    events = []

    class StopAtModel(RuntimeError):
        pass

    args = Namespace(
        top_k=1,
        min_rank=1,
        seed=42,
        device="cpu",
        query_start=0,
        query_end=0,
        limit_queries=0,
    )
    monkeypatch.setattr(model_miner, "parse_args", lambda: args)
    monkeypatch.setattr(
        model_miner,
        "validate_trusted_diagnostic_inputs",
        lambda _args: events.append("gate"),
        raising=False,
    )

    def build_dataset(_args):
        events.append("dataset")
        return Namespace(csv_video_ids=["video1"])

    monkeypatch.setattr(model_miner, "build_train_dataset", build_dataset)
    sample = CaptionSample(0, "video1", "caption")
    monkeypatch.setattr(
        model_miner,
        "samples_from_dataset",
        lambda _dataset: [sample],
    )
    monkeypatch.setattr(
        model_miner,
        "select_query_samples",
        lambda *_args, **_kwargs: ([sample], 0, 1),
    )
    monkeypatch.setattr(model_miner, "build_meta", lambda _args: {})
    monkeypatch.setattr(
        model_miner,
        "load_resume_state",
        lambda _args, _meta: ({}, set()),
    )

    def stop_at_model(_args, _device):
        events.append("model")
        raise StopAtModel

    monkeypatch.setattr(model_miner, "load_model", stop_at_model)

    with pytest.raises(StopAtModel):
        model_miner.main()

    assert events == ["gate", "dataset", "model"]


def test_select_hard_negative_candidate_skips_same_video_and_near_duplicates():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a man plays guitar"),
        CaptionSample(2, "video0", "same source video"),
        CaptionSample(3, "video3", "a woman cooks pasta"),
        CaptionSample(4, "video4", "children play soccer"),
    ]
    video_to_sample_indices = {
        "video0": [0, 2],
        "video1": [1],
        "video3": [3],
        "video4": [4],
    }
    config = ModelHardNegativeConfig(top_k=4, min_rank=1, max_jaccard=0.8, max_overlap=0.9)

    candidate = select_hard_negative_candidate(
        anchor=samples[0],
        ranked_video_indices=[0, 1, 2, 3],
        video_ids=["video0", "video1", "video3", "video4"],
        samples=samples,
        video_to_sample_indices=video_to_sample_indices,
        config=config,
    )

    assert candidate is not None
    assert candidate.hard_index == 3
    assert candidate.hard_video_id == "video3"
    assert candidate.model_rank == 3
    assert candidate.skipped_same_video == 1
    assert candidate.skipped_text_risk == 1


def test_select_hard_negative_candidate_respects_min_rank_window():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a woman cooks pasta"),
        CaptionSample(2, "video2", "children play soccer"),
    ]
    video_to_sample_indices = {"video0": [0], "video1": [1], "video2": [2]}
    config = ModelHardNegativeConfig(top_k=2, min_rank=2)

    candidate = select_hard_negative_candidate(
        anchor=samples[0],
        ranked_video_indices=[1, 2],
        video_ids=["video0", "video1", "video2"],
        samples=samples,
        video_to_sample_indices=video_to_sample_indices,
        config=config,
    )

    assert candidate is not None
    assert candidate.hard_index == 2
    assert candidate.model_rank == 2


def test_text_pair_metrics_flags_exact_and_overlap():
    metrics = text_pair_metrics("A man plays guitar on stage", "man playing guitar on a stage")

    assert metrics["shared_tokens"] >= 2
    assert metrics["jaccard"] > 0.25
    assert metrics["overlap"] >= 0.5
    assert metrics["exact_caption"] is False


def test_build_model_mined_mapping_writes_compatible_entries_and_stats():
    samples = [
        CaptionSample(0, "video0", "a man plays guitar"),
        CaptionSample(1, "video1", "a woman cooks pasta"),
        CaptionSample(2, "video2", "children play soccer"),
    ]
    video_ids = ["video0", "video1", "video2"]
    scores = [
        [10.0, 9.5, 3.0],
        [2.0, 8.0, 7.5],
        [5.0, 4.5, 9.0],
    ]
    config = ModelHardNegativeConfig(top_k=3, min_rank=1)

    result = build_model_mined_mapping(
        samples=samples,
        video_ids=video_ids,
        scores=scores,
        config=config,
        include_captions=True,
    )

    assert result["stats"]["num_samples"] == 3
    assert result["stats"]["mapping_size"] == 3
    assert result["stats"]["fallback_count"] == 0
    assert result["mapping"]["0"]["anchor_index"] == 0
    assert result["mapping"]["0"]["hard_index"] == 1
    assert result["mapping"]["0"]["hard_video_id"] == "video1"
    assert result["mapping"]["0"]["model_rank"] == 2
    assert result["mapping"]["0"]["anchor_caption"] == "a man plays guitar"


def test_select_query_samples_uses_exclusive_range_then_limit():
    samples = [CaptionSample(i, f"video{i}", f"caption {i}") for i in range(10)]

    selected, query_start, query_end = select_query_samples(
        samples,
        query_start=3,
        query_end=8,
        limit_queries=2,
    )

    assert [sample.sample_index for sample in selected] == [3, 4]
    assert query_start == 3
    assert query_end == 5


def test_select_query_samples_rejects_invalid_ranges():
    samples = [CaptionSample(i, f"video{i}", f"caption {i}") for i in range(3)]

    try:
        select_query_samples(samples, query_start=2, query_end=1, limit_queries=0)
    except ValueError as exc:
        assert "query_end" in str(exc)
    else:
        raise AssertionError("expected invalid query range to fail")


def test_mine_mapping_scores_single_tensor_visual_outputs():
    class FakeDataset:
        max_words = 1

        def _get_text(self, _video_id, _caption, max_words):
            assert max_words == self.max_words
            return (
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                None,
            )

        def _get_rawvideo(self, video_ids):
            values = {"video0": 1.0, "video1": 2.0}
            video = np.array([values[video_id] for video_id in video_ids], dtype=np.float32).reshape(-1, 1, 1)
            video_mask = np.ones((len(video_ids), 1), dtype=np.int64)
            return video, video_mask

    class FakeModel:
        loose_type = True

        def __init__(self):
            self.similarity_calls = 0

        def get_visual_output(self, video, _video_mask, shaped=False):
            assert shaped is False
            return video[:, 0]

        def get_sequence_output(self, input_ids, _segment_ids, _input_mask):
            sequence_output = input_ids.float().unsqueeze(-1)
            return sequence_output, sequence_output

        def get_similarity_logits(
            self,
            sequence_output,
            _text_token,
            visual_output,
            _input_mask,
            _video_mask,
            loose_type,
        ):
            assert loose_type is self.loose_type
            self.similarity_calls += 1
            logits = sequence_output[:, 0, 0].unsqueeze(1) + visual_output[:, 0, 0].unsqueeze(0)
            return logits, None

    args = Namespace(
        top_k=2,
        min_rank=1,
        max_jaccard=0.8,
        max_overlap=0.9,
        min_token_len=2,
        keep_stopwords=False,
        video_batch_size=1,
        text_batch_size=1,
        video_chunk_size=2,
        include_captions=False,
        progress_interval=0,
        checkpoint_interval=0,
    )
    samples = [
        CaptionSample(0, "video0", "guitar solo"),
        CaptionSample(1, "video1", "cooking pasta"),
    ]
    model = FakeModel()

    mapping, processed = mine_mapping_with_model(
        args,
        model,
        FakeDataset(),
        query_samples=samples[:1],
        all_samples=samples,
        video_ids=["video0", "video1"],
        device=torch.device("cpu"),
    )

    assert processed == {0}
    assert mapping["0"]["hard_video_id"] == "video1"
    assert mapping["0"]["positive_score"] == 1.0
    assert mapping["0"]["model_score"] == 2.0
    assert model.similarity_calls == 1
