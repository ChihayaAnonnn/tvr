import numpy as np
import pytest

from prob_models.retrieval_risk_metrics import (
    build_eval_positive_mask,
    evaluate_retrieval_risk,
    positive_ranks,
)


def test_multi_sentence_mask_maps_caption_ranges_to_videos():
    mask = build_eval_positive_mask(5, 2, cut_off_points=[2, 5])

    assert mask.tolist() == [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [False, True],
    ]


def test_one_to_one_mask_is_identity():
    assert build_eval_positive_mask(3, 3).tolist() == np.eye(
        3, dtype=bool
    ).tolist()


@pytest.mark.parametrize(
    ("num_text", "num_video", "cut_off_points"),
    [
        (3, 2, None),
        (5, 2, [2]),
        (5, 2, [3, 4]),
        (5, 2, [3, 6]),
    ],
)
def test_eval_mask_rejects_inconsistent_shapes(
    num_text, num_video, cut_off_points
):
    with pytest.raises(ValueError):
        build_eval_positive_mask(num_text, num_video, cut_off_points)


def test_positive_ranks_support_multiple_positives():
    scores = np.array([[0.8, 0.9, 0.1], [0.7, 0.6, 0.5]])
    mask = np.array([[True, True, False], [False, False, True]])

    ranks = positive_ranks(scores, mask)

    assert ranks.tolist() == [0, 2]


def test_perfect_risk_has_perfect_auroc_and_selective_accuracy():
    scores = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 4.0, 3.0, 0.0],
            [4.0, 0.0, 0.0, 3.0],
        ]
    )
    risk = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluate_retrieval_risk(
        scores,
        np.eye(4, dtype=bool),
        risk,
        np.full(4, 0.3),
        bootstrap_samples=100,
        seed=3,
    )

    assert result["failure_auroc"]["value"] == 1.0
    assert result["failure_auroc"]["ci_low"] == 1.0
    assert result["failure_auroc"]["ci_high"] == 1.0
    assert result["risk_coverage"]["accuracy_at_0.50"] == 1.0
    assert result["spearman_risk_vs_rank"] > 0
    assert result["risk_stats"]["mean"] == pytest.approx(0.5)


def test_risk_metrics_reject_constant_failure_labels():
    with pytest.raises(ValueError, match="success and failure"):
        evaluate_retrieval_risk(
            np.eye(3),
            np.eye(3, dtype=bool),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.4, 0.4]),
            bootstrap_samples=10,
        )
