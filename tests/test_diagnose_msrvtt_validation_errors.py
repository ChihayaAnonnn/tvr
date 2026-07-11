from argparse import Namespace

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


def test_compute_validation_sim_matrix_accepts_three_model_outputs(monkeypatch):
    class FakeModel:
        loose_type = True

        def __init__(self):
            self.similarity_calls = 0

        def eval(self):
            return self

        def get_sequence_visual_output(
            self,
            input_ids,
            _segment_ids,
            _input_mask,
            video,
            _video_mask,
        ):
            sequence_output = input_ids.float().unsqueeze(-1)
            return sequence_output, sequence_output, video[:, 0]

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
    dataloader = [(input_ids, input_mask, segment_ids, video, video_mask)]
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
