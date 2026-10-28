import numpy as np
import pytest


def test_ties_use_candidate_identity_and_one_rank_per_query():
    from semantic_retrieval.evaluation.ranking import positive_ranks, retrieval_metrics

    scores = np.array([[1.0, 1.0], [0.0, 1.0]])
    relevance = np.array([[1, 0], [0, 1]])
    assert positive_ranks(scores, relevance == 1, ["a", "b"]).tolist() == [0, 0]
    assert positive_ranks(scores, relevance == 1, ["b", "a"]).tolist() == [1, 0]
    result = retrieval_metrics(scores, relevance, ["a", "b"])
    assert result["R1"] == 100
    assert result["num_eligible"] == 2
    assert positive_ranks(scores.T, relevance.T == 1, ["a", "b"]).tolist() == [0, 1]


def test_mult_positive_and_incomplete_annotation_denominators():
    from semantic_retrieval.evaluation.ranking import retrieval_metrics

    result = retrieval_metrics(
        np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0], [2.0, 4.0, 1.0]]),
        np.array([[1, 1, 0], [0, 0, 0], [1, -1, 0]]),
        ["a", "b", "c"],
    )
    assert result["R1"] == 0
    assert result["MeanR"] == 2
    assert result["num_eligible"] == 1
    assert result["num_no_answer"] == 1
    assert result["num_unknown"] == 1
    empty = retrieval_metrics(np.ones((1, 2)), np.zeros((1, 2)), ["a", "b"])
    assert empty["R1"] is None
    assert empty["num_eligible"] == 0


@pytest.mark.parametrize(
    "scores,mask,ids",
    [
        (np.array([[np.nan, 0.0]]), np.array([[True, False]]), ["a", "b"]),
        (np.array([[np.inf, 0.0]]), np.array([[True, False]]), ["a", "b"]),
        (np.ones((1, 2)), np.array([[False, False]]), ["a", "b"]),
        (np.ones((1, 2)), np.array([[True, False]]), ["a", "a"]),
    ],
)
def test_rank_rejects_undefined_or_ambiguous_input(scores, mask, ids):
    from semantic_retrieval.evaluation.ranking import positive_ranks

    with pytest.raises(ValueError):
        positive_ranks(scores, mask, ids)


def test_single_class_calibration_and_undefined_auroc():
    from semantic_retrieval.evaluation.binary import binary_metrics

    result = binary_metrics(np.array([0, 0]), np.array([0.25, 0.25]))
    assert result["auroc"] is None
    assert result["brier"] == 0.0625
    assert result["num_positive"] == 0
    assert result["num_negative"] == 2
    assert binary_metrics(np.array([0, 1]), np.array([0.25, 0.75]))["auroc"] == 1.0


def test_single_candidate_baselines_have_zero_entropy_and_no_margin():
    from semantic_retrieval.evaluation.baselines import score_risk_baselines

    result = score_risk_baselines(np.ones((2, 1)), "t2v")
    assert result["entropy"].tolist() == [0.0, 0.0]
    assert "negative_margin" not in result
    assert score_risk_baselines(np.ones((1, 2)), "v2t")["entropy"].tolist() == [0.0, 0.0]
