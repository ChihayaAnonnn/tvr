from scripts.diagnose_msrvtt_hard_negative_runtime import (
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
