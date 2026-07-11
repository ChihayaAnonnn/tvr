from argparse import Namespace

import numpy as np
import torch

import scripts.diagnose_msrvtt_hard_negative_runtime as runtime_diagnostics
from scripts.diagnose_msrvtt_hard_negative_runtime import (
    RuntimeSample,
    compute_checkpoint_scores,
    compute_rank,
    summarize_runtime_rows,
)


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
