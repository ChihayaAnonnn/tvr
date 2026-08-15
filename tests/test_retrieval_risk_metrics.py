import numpy as np
import pytest

from prob_models.retrieval_risk_metrics import (
    binary_calibration_metrics,
    build_eval_positive_mask,
    evaluate_failure_risk_scores,
    evaluate_retrieval_risk,
    paired_bootstrap_delta,
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


def test_binary_calibration_metrics_match_hand_computation():
    labels = np.array([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    result = binary_calibration_metrics(labels, probabilities, bins=2)

    assert result["ece"] == pytest.approx(0.15)
    assert result["brier"] == pytest.approx(0.025)
    assert result["nll"] == pytest.approx(
        -np.mean(np.log([0.9, 0.8, 0.8, 0.9]))
    )


def test_failure_risk_summary_treats_constant_risk_as_neutral():
    result = evaluate_failure_risk_scores(
        np.array([False, True]),
        np.array([0, 1]),
        np.array([0.5, 0.5]),
    )

    assert result["rho"] == 0.0
    assert result["auroc"] == 0.5
    assert result["aurc"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([0, 1], [0.2], "shape"),
        ([0, 1], [0.2, np.nan], "finite"),
        ([0, 1], [-0.1, 0.8], r"\[0, 1\]"),
    ],
)
def test_binary_calibration_metrics_reject_invalid_inputs(
    labels, probabilities, message
):
    with pytest.raises(ValueError, match=message):
        binary_calibration_metrics(labels, probabilities)


@pytest.mark.parametrize("metric", ["auroc", "aurc"])
def test_paired_bootstrap_is_deterministic_and_paired(metric):
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    first_risk = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    second_risk = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])

    first = paired_bootstrap_delta(
        labels, first_risk, second_risk, metric, 10_000, 0
    )
    second = paired_bootstrap_delta(
        labels, first_risk, second_risk, metric, 10_000, 0
    )

    assert first == second
    assert first["metric"] == metric
    assert first["samples"] == 10_000
    assert first["seed"] == 0
    assert first["valid_replicates"] == 10_000
    if metric == "auroc":
        assert first["point_delta"] > 0
    else:
        assert first["point_delta"] < 0
    assert first["ci_low"] <= first["ci_high"]


def test_paired_bootstrap_rejects_invalid_inputs_and_constant_labels():
    labels = np.array([False, False, False])
    risks = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="both success and failure"):
        paired_bootstrap_delta(labels, risks, risks, "auroc", 10, 0)

    labels = np.array([False, True, True])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_delta(
            labels, risks, np.array([0.1, np.nan, 0.3]), "auroc", 10, 0
        )
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap_delta(labels, risks, risks, "rho", 10, 0)
    with pytest.raises(ValueError, match="positive"):
        paired_bootstrap_delta(labels, risks, risks, "auroc", 0, 0)
