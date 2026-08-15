import numpy as np
import pytest

from prob_models.retrieval_risk_baselines import score_risk_baselines


METHODS = {
    "one_minus_top1",
    "negative_margin",
    "entropy",
    "negative_energy",
    "score_variance",
}


def test_score_baselines_order_ambiguous_query_above_clear_query():
    scores = np.array(
        [[8.0, 1.0, 0.0], [3.0, 2.9, 0.0]], dtype=np.float32
    )

    risks = score_risk_baselines(scores, "t2v")

    assert set(risks) == METHODS
    for method, risk in risks.items():
        assert risk.dtype == np.float64
        assert risk[1] > risk[0], method


def test_score_baselines_have_explicit_shift_behavior():
    scores = np.array(
        [[4.0, 1.0, -2.0], [2.0, 1.5, 0.0]], dtype=np.float64
    )

    original = score_risk_baselines(scores, "t2v")
    shifted = score_risk_baselines(scores + 7.0, "t2v")

    for method in METHODS - {"negative_energy"}:
        np.testing.assert_allclose(shifted[method], original[method])
    assert np.all(shifted["negative_energy"] < original["negative_energy"])


def test_score_baselines_follow_query_permutations_and_direction_transpose():
    scores = np.array(
        [[4.0, 1.0, 0.0], [2.0, 1.9, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    permutation = np.array([2, 0, 1])

    original = score_risk_baselines(scores, "t2v")
    permuted = score_risk_baselines(scores[permutation], "t2v")
    transposed = score_risk_baselines(scores.T, "v2t")

    for method in METHODS:
        np.testing.assert_array_equal(
            permuted[method], original[method][permutation]
        )
        np.testing.assert_array_equal(transposed[method], original[method])


def test_score_baselines_handle_ties_deterministically():
    scores = np.array([[2.0, 2.0, 0.0], [1.0, 1.0, 1.0]])

    first = score_risk_baselines(scores, "t2v")
    second = score_risk_baselines(scores, "t2v")

    assert first["negative_margin"].tolist() == [0.5, 0.5]
    for method in METHODS:
        np.testing.assert_array_equal(first[method], second[method])
        assert np.isfinite(first[method]).all()


@pytest.mark.parametrize("direction", ["t2v", "v2t"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_score_baselines_reject_nonfinite_scores(direction, nonfinite):
    scores = np.eye(3)
    scores[0, 1] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        score_risk_baselines(scores, direction)


def test_score_baselines_reject_invalid_shape_direction_and_candidate_count():
    with pytest.raises(ValueError, match="direction"):
        score_risk_baselines(np.eye(2), "invalid")
    with pytest.raises(ValueError, match="2-D"):
        score_risk_baselines(np.ones(3), "t2v")
    with pytest.raises(ValueError, match="query"):
        score_risk_baselines(np.ones((0, 3)), "t2v")
    with pytest.raises(ValueError, match="two candidates"):
        score_risk_baselines(np.ones((3, 1)), "t2v")
