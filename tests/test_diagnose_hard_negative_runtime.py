import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.diagnose_msrvtt_hard_negative_runtime as runtime_diagnostics
from scripts.diagnose_msrvtt_hard_negative_runtime import (
    RuntimeSample,
    build_task_args,
    compute_checkpoint_scores,
    compute_rank,
    summarize_runtime_rows,
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
        runtime_diagnostics.parse_args()

    assert exc_info.value.code == 2


def test_parse_args_defaults_to_generated_trusted_split(monkeypatch):
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

    args = runtime_diagnostics.parse_args()

    assert args.train_csv.endswith("data/generated/msrvtt_trusted_v1/train.csv")
    assert args.val_csv.endswith("data/generated/msrvtt_trusted_v1/val.csv")
    assert "JSFUSION_test" not in args.val_csv
    assert args.source_train_csv.endswith("csv/MSRVTT_train.9k.csv")
    assert args.test_csv.endswith("csv/MSRVTT_JSFUSION_test.csv")
    assert args.split_manifest.endswith(
        "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
    )


def test_build_task_args_uses_supported_deterministic_protocol_args(tmp_path):
    cli_args = argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        train_csv=str(tmp_path / "train.csv"),
        source_train_csv=str(tmp_path / "source.csv"),
        test_csv=str(tmp_path / "test.csv"),
        split_manifest=str(tmp_path / "manifest.json"),
        val_csv=str(tmp_path / "val.csv"),
        data_path=str(tmp_path / "annotation.json"),
        features_path=str(tmp_path / "videos"),
    )

    args = build_task_args(cli_args, "checkpoint.bin")

    assert args.train_csv == cli_args.train_csv
    assert args.source_train_csv == cli_args.source_train_csv
    assert args.test_csv == cli_args.test_csv
    assert args.split_manifest == cli_args.split_manifest
    assert args.val_csv == cli_args.val_csv
    assert args.eval_split == "val"
    assert args.experiment_profile == "hygiene"
    for retired_name in (
        "final_score_mode",
        "uncertainty_mode",
        "n_video_embeddings",
        "w_evidential",
    ):
        assert not hasattr(args, retired_name)


def test_trusted_diagnostic_gate_rejects_non_internal_val_csv(
    tmp_path, monkeypatch
):
    from dataloaders import msrvtt_protocol

    monkeypatch.setattr(
        msrvtt_protocol,
        "load_trusted_manifest",
        lambda _path: {
            "train_video_ids": ["train1"],
            "val_video_ids": ["video1"],
        },
    )
    monkeypatch.setattr(
        msrvtt_protocol,
        "validate_trusted_manifest",
        lambda *_args, **_kwargs: None,
    )
    args = argparse.Namespace(
        split_manifest="manifest.json",
        train_csv=str(tmp_path / "train.csv"),
        source_train_csv="source.csv",
        data_path="annotation.json",
        test_csv="test.csv",
    )
    Path(args.train_csv).write_text(
        "video_id\ntrain1\n", encoding="utf-8"
    )
    wrong = tmp_path / "wrong.csv"
    wrong.write_text(
        "video_id,sentence\nvideo2,caption\n", encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="exactly match trusted-v1 internal val"
    ):
        runtime_diagnostics.validate_trusted_diagnostic_inputs(
            args, scored_csv=wrong
        )

    valid = tmp_path / "valid.csv"
    valid.write_text(
        "video_id,sentence\nvideo1,caption\n", encoding="utf-8"
    )
    runtime_diagnostics.validate_trusted_diagnostic_inputs(
        args, scored_csv=valid
    )

    Path(args.train_csv).write_text(
        "video_id\ntrain1\nvideo1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly match trusted-v1 train"):
        runtime_diagnostics.validate_trusted_diagnostic_inputs(
            args, scored_csv=valid
        )


def test_main_runs_trusted_gate_before_dataset_or_checkpoint_access(monkeypatch):
    events = []

    class StopAtCheckpoint(RuntimeError):
        pass

    args = Namespace(
        device="cpu",
        hard_negative_path="mapping.json",
        max_anchors=1,
        seed=42,
        max_rank_videos=1,
        baseline_name="baseline",
        target_name="target",
        baseline_checkpoint="baseline.bin",
        target_checkpoint="target.bin",
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "parse_args",
        lambda: args,
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "validate_trusted_diagnostic_inputs",
        lambda _args: events.append("gate"),
        raising=False,
    )

    class FakeDataset:
        def __len__(self):
            return 1

    def build_dataset(_args):
        events.append("dataset")
        return FakeDataset()

    monkeypatch.setattr(
        runtime_diagnostics,
        "build_msrvtt_train_dataset",
        build_dataset,
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "load_hard_mapping",
        lambda _path, _length: [0],
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "select_runtime_samples",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        runtime_diagnostics,
        "build_video_pool",
        lambda *_args, **_kwargs: ["video1"],
    )

    def stop_at_checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        raise StopAtCheckpoint

    monkeypatch.setattr(
        runtime_diagnostics,
        "compute_checkpoint_scores",
        stop_at_checkpoint,
    )

    with pytest.raises(StopAtCheckpoint):
        runtime_diagnostics.main()

    assert events == ["gate", "dataset", "checkpoint"]


def test_compute_rank_uses_one_based_descending_rank():
    scores = [0.1, 0.9, 0.4, 0.4]

    assert compute_rank(scores, 1) == 1
    assert compute_rank(scores, 2) == 2
    assert compute_rank(scores, 3) == 2
    assert compute_rank(scores, 0) == 4


def test_summarize_runtime_rows_reports_hardness_and_checkpoint_delta():
    rows = [
        {
            "baseline_pos_logit": 10.0,
            "baseline_hard_logit": 9.0,
            "baseline_hard_rank": 3,
            "target_pos_logit": 11.0,
            "target_hard_logit": 7.0,
            "target_hard_rank": 20,
        },
        {
            "baseline_pos_logit": 8.0,
            "baseline_hard_logit": 8.5,
            "baseline_hard_rank": 1,
            "target_pos_logit": 8.2,
            "target_hard_logit": 8.0,
            "target_hard_rank": 6,
        },
    ]

    summary = summarize_runtime_rows(rows, margin=0.5)

    assert summary["num_rows"] == 2
    assert summary["baseline_gap_mean"] == 0.25
    assert summary["target_gap_mean"] == 2.1
    assert summary["gap_delta_mean"] == 1.85
    assert summary["baseline_margin_fail_rate"] == 0.5
    assert summary["target_margin_fail_rate"] == 0.5
    assert summary["baseline_hard_rank_top1_rate"] == 0.5
    assert summary["target_hard_rank_top5_rate"] == 0.0
    assert summary["target_hard_rank_top10_rate"] == 0.5


def test_compute_checkpoint_scores_accepts_single_tensor_visual_output(monkeypatch):
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

        def get_sequence_output(self, input_ids, _segment_ids, _input_mask):
            sequence_output = input_ids.float().unsqueeze(-1)
            return sequence_output, sequence_output

        def get_visual_output(self, video, _video_mask, shaped=False):
            assert shaped is False
            return video[:, 0]

        def get_similarity_logits(
            self,
            sequence_output,
            _text_token,
            visual_output,
            _text_mask,
            _video_mask,
            loose_type,
        ):
            assert loose_type is self.loose_type
            self.similarity_calls += 1
            logits = sequence_output[:, 0, 0].unsqueeze(1) + visual_output[:, 0, 0].unsqueeze(0)
            return logits, None

    model = FakeModel()

    def fake_load_model(_cli_args, _checkpoint, _device):
        return model, None

    monkeypatch.setattr(runtime_diagnostics, "load_model_for_checkpoint", fake_load_model)
    args = Namespace(text_batch_size=1, video_batch_size=1)
    samples = [RuntimeSample(0, "video0", "caption", 1, "video1", "hard caption")]

    scores = compute_checkpoint_scores(
        args,
        "checkpoint.bin",
        FakeDataset(),
        samples,
        ["video0", "video1"],
        torch.device("cpu"),
    )

    assert scores == [[1.0, 2.0]]
    assert model.similarity_calls == 2
