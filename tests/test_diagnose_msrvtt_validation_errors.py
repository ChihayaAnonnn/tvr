import sys
from argparse import Namespace

import pytest
import torch

import scripts.diagnose_msrvtt_hard_negative_runtime as runtime_diagnostics
import scripts.diagnose_msrvtt_validation_errors as validation_diagnostics
from scripts.diagnose_msrvtt_validation_errors import (
    ValidationItem,
    build_validation_error_rows,
    compute_validation_sim_matrix,
    is_hard_like_pair,
    rank_of_ground_truth,
    summarize_error_rows,
)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--baseline_checkpoint", "baseline.bin"],
        ["--target_checkpoint", "target.bin"],
    ],
)
def test_parse_args_requires_both_checkpoints(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])

    with pytest.raises(SystemExit) as exc_info:
        validation_diagnostics.parse_args()

    assert exc_info.value.code == 2


def test_parse_args_defaults_to_trusted_internal_val(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--baseline_checkpoint",
            "baseline.bin",
            "--target_checkpoint",
            "target.bin",
        ],
    )

    args = validation_diagnostics.parse_args()

    assert args.train_csv.endswith("data/generated/msrvtt_trusted_v1/train.csv")
    assert args.val_csv.endswith("data/generated/msrvtt_trusted_v1/val.csv")
    assert "JSFUSION_test" not in args.val_csv
    assert args.source_train_csv.endswith("csv/MSRVTT_train.9k.csv")
    assert args.test_csv.endswith("csv/MSRVTT_JSFUSION_test.csv")
    assert args.split_manifest.endswith(
        "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
    )


def test_validation_task_args_only_use_internal_val_and_supported_fields(
    tmp_path,
):
    args = Namespace(
        output_dir=str(tmp_path / "out"),
        train_csv=str(tmp_path / "train.csv"),
        source_train_csv=str(tmp_path / "source.csv"),
        test_csv=str(tmp_path / "test.csv"),
        split_manifest=str(tmp_path / "manifest.json"),
        val_csv=str(tmp_path / "val.csv"),
        data_path=str(tmp_path / "annotation.json"),
        features_path=str(tmp_path / "videos"),
    )

    task_args = runtime_diagnostics.build_task_args(
        args, "checkpoint.bin"
    )

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


def test_main_runs_trusted_gate_before_reading_scored_csv_or_models(
    monkeypatch,
):
    events = []

    class StopAtCheckpoint(RuntimeError):
        pass

    args = Namespace(
        device="cpu",
        val_csv="internal-val.csv",
        max_queries=0,
        baseline_name="baseline",
        target_name="target",
        baseline_checkpoint="baseline.bin",
        target_checkpoint="target.bin",
    )
    monkeypatch.setattr(validation_diagnostics, "parse_args", lambda: args)

    def record_gate(gate_args, scored_csv=None):
        assert gate_args is args
        assert scored_csv == args.val_csv
        events.append("gate")

    monkeypatch.setattr(
        validation_diagnostics,
        "validate_trusted_diagnostic_inputs",
        record_gate,
        raising=False,
    )

    def stop_at_read(_csv_path, max_queries=0):
        assert max_queries == 0
        events.append("read")
        return [ValidationItem(0, "video1", "caption")]

    monkeypatch.setattr(
        validation_diagnostics,
        "read_validation_items",
        stop_at_read,
    )

    def stop_at_checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        raise StopAtCheckpoint

    monkeypatch.setattr(
        validation_diagnostics,
        "compute_validation_sim_matrix",
        stop_at_checkpoint,
    )

    with pytest.raises(StopAtCheckpoint):
        validation_diagnostics.main()

    assert events == ["gate", "read", "checkpoint"]


def test_rank_of_ground_truth_uses_one_based_descending_rank():
    scores = [0.4, 0.9, 0.7, 0.7]

    assert rank_of_ground_truth(scores, 1) == 1
    assert rank_of_ground_truth(scores, 2) == 2
    assert rank_of_ground_truth(scores, 3) == 2
    assert rank_of_ground_truth(scores, 0) == 4


def test_hard_like_pair_detects_near_duplicate_validation_errors():
    assert is_hard_like_pair(
        "someone is cooking in the kitchen",
        "a person is cooking in a kitchen",
    )
    assert is_hard_like_pair(
        "two men are competitive wrestling",
        "a wrestling match at a gym",
    )
    assert not is_hard_like_pair(
        "a dog runs outside",
        "a basketball game is being played",
    )


def test_build_validation_error_rows_classifies_fixed_and_regressed_queries():
    items = [
        ValidationItem(0, "video0", "someone is cooking in the kitchen"),
        ValidationItem(1, "video1", "a person is cooking in a kitchen"),
        ValidationItem(2, "video2", "a dog runs outside"),
        ValidationItem(3, "video3", "a basketball game is being played"),
    ]
    baseline_sim = [
        [0.8, 0.9, 0.1, 0.2],  # wrong to hard-like video1
        [0.2, 0.9, 0.3, 0.4],  # correct
        [0.7, 0.1, 0.6, 0.8],  # wrong to video3
        [0.2, 0.1, 0.3, 0.9],  # correct
    ]
    target_sim = [
        [0.95, 0.7, 0.1, 0.2],  # fixed
        [0.2, 0.7, 0.3, 0.8],   # regressed to video3
        [0.7, 0.1, 0.6, 0.8],   # both wrong same
        [0.2, 0.1, 0.3, 0.9],   # both correct
    ]

    rows = build_validation_error_rows(items, baseline_sim, target_sim)
    summary = summarize_error_rows(rows)

    assert [row["transition"] for row in rows] == [
        "fixed_by_hn",
        "regressed_by_hn",
        "both_wrong_same",
        "both_correct",
    ]
    assert rows[0]["baseline_pred_hard_like"] is True
    assert rows[1]["target_pred_hard_like"] is False
    assert rows[0]["baseline_gt_rank"] == 2
    assert rows[0]["target_gt_rank"] == 1
    assert summary["num_queries"] == 4
    assert summary["baseline_error_count"] == 2
    assert summary["target_error_count"] == 2
    assert summary["fixed_by_hn_count"] == 1
    assert summary["regressed_by_hn_count"] == 1
    assert summary["baseline_error_hard_like_rate"] == 0.5


def test_build_validation_error_rows_maps_captions_to_unique_video_candidates():
    items = [
        ValidationItem(0, "video0", "the first caption for video zero"),
        ValidationItem(1, "video0", "the second caption for video zero"),
        ValidationItem(2, "video1", "the caption for video one"),
    ]
    baseline_sim = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.95],
    ]
    target_sim = [
        [0.85, 0.15],
        [0.75, 0.25],
        [0.05, 0.9],
    ]

    rows = build_validation_error_rows(items, baseline_sim, target_sim)

    assert [row["baseline_correct"] for row in rows] == [True, True, True]
    assert [row["target_correct"] for row in rows] == [True, True, True]
    assert [row["baseline_gt_rank"] for row in rows] == [1, 1, 1]
    assert [row["target_gt_rank"] for row in rows] == [1, 1, 1]
    assert rows[1]["baseline_top1_video_id"] == "video0"
    assert rows[1]["baseline_top1_caption"] == items[0].caption


def test_build_validation_dataloader_locks_trusted_multicaption_protocol(
    monkeypatch,
):
    import dataloaders.dataloader_msrvtt_retrieval as msrvtt_dataloader
    import modules.tokenization_clip as tokenization_clip

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.video_ids = ["video0"] * 20 + ["video1"] * 20
            self.sentences = ["caption"] * 40

    monkeypatch.setattr(msrvtt_dataloader, "MSRVTT_DataLoader", FakeDataset)
    monkeypatch.setattr(tokenization_clip, "SimpleTokenizer", object)
    monkeypatch.setattr(
        runtime_diagnostics,
        "build_task_args",
        lambda _args, _checkpoint: Namespace(
            max_words=32,
            feature_framerate=1,
            max_frames=12,
            eval_frame_order=0,
            slice_framepos=2,
        ),
    )
    monkeypatch.setattr(
        torch.utils.data,
        "DataLoader",
        lambda dataset, **kwargs: Namespace(dataset=dataset, **kwargs),
    )
    args = Namespace(
        baseline_checkpoint="baseline.bin",
        val_csv="trusted-val.csv",
        features_path="videos",
        max_queries=1,
        batch_size=4,
    )

    dataloader = validation_diagnostics.build_validation_dataloader(args)

    assert captured["multi_sentence_per_video"] is True
    assert captured["expected_captions_per_video"] == 20
    assert dataloader.dataset.video_ids == ["video0"]
    assert dataloader.dataset.sentences == ["caption"]


def test_compute_validation_sim_matrix_uses_first_row_per_unique_video(
    monkeypatch,
):
    class FakeDataset:
        video_ids = ["video_b", "video_b", "video_a"]

    class FakeDataloader:
        dataset = FakeDataset()

        def __iter__(self):
            input_ids = torch.tensor([[0], [100], [200]], dtype=torch.long)
            input_mask = torch.ones_like(input_ids)
            segment_ids = torch.zeros_like(input_ids)
            video = torch.tensor([20.0, 999.0, 10.0]).reshape(3, 1, 1, 1)
            video_mask = torch.ones((3, 1), dtype=torch.long)
            yield input_ids, input_mask, segment_ids, video, video_mask

    class FakeModel:
        loose_type = True

        def __init__(self):
            self.sequence_calls = 0
            self.visual_call_values = []

        def eval(self):
            return self

        def get_sequence_output(
            self,
            input_ids,
            _segment_ids,
            _input_mask,
        ):
            self.sequence_calls += 1
            sequence_output = input_ids.float().unsqueeze(-1)
            return sequence_output, sequence_output

        def get_visual_output(self, video, _video_mask):
            self.visual_call_values.append(
                video.reshape(video.size(0), -1)[:, 0].tolist()
            )
            return video[:, 0]

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
            logits = (
                sequence_output[:, 0, 0].unsqueeze(1)
                + visual_output[:, 0, 0].unsqueeze(0)
            )
            return logits, None

    model = FakeModel()
    monkeypatch.setattr(
        validation_diagnostics,
        "build_validation_dataloader",
        lambda _args: FakeDataloader(),
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "load_model_for_checkpoint",
        lambda _args, _checkpoint, _device: (model, None),
    )

    scores = compute_validation_sim_matrix(
        Namespace(video_chunk_size=2),
        "checkpoint.bin",
        torch.device("cpu"),
    )

    assert scores == [
        [20.0, 10.0],
        [120.0, 110.0],
        [220.0, 210.0],
    ]
    assert model.sequence_calls == 1
    assert model.visual_call_values == [[20.0, 10.0]]


def test_compute_validation_sim_matrix_uses_independent_model_outputs(monkeypatch):
    class FakeModel:
        loose_type = True

        def __init__(self):
            self.similarity_calls = 0

        def eval(self):
            return self

        def get_sequence_output(
            self,
            input_ids,
            _segment_ids,
            _input_mask,
        ):
            sequence_output = input_ids.float().unsqueeze(-1)
            return sequence_output, sequence_output

        def get_visual_output(self, video, _video_mask):
            return video[:, 0]

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

    input_ids = torch.tensor([[0], [10]], dtype=torch.long)
    input_mask = torch.ones_like(input_ids)
    segment_ids = torch.zeros_like(input_ids)
    video = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    video_mask = torch.ones((2, 1), dtype=torch.long)
    class FakeDataset:
        video_ids = ["video0", "video1"]

    class FakeDataloader:
        dataset = FakeDataset()

        def __iter__(self):
            yield input_ids, input_mask, segment_ids, video, video_mask

    dataloader = FakeDataloader()
    model = FakeModel()

    monkeypatch.setattr(validation_diagnostics, "build_validation_dataloader", lambda _args: dataloader)
    monkeypatch.setattr(
        runtime_diagnostics,
        "load_model_for_checkpoint",
        lambda _args, _checkpoint, _device: (model, None),
    )

    scores = compute_validation_sim_matrix(
        Namespace(video_chunk_size=2),
        "checkpoint.bin",
        torch.device("cpu"),
    )

    assert scores == [[1.0, 2.0], [11.0, 12.0]]
    assert model.similarity_calls == 1
