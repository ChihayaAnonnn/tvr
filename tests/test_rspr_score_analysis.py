import numpy as np
import pytest

from modules.rspr_score_analysis import (
    candidate_recall_ceiling,
    count_rerank_flips,
    permute_probability_within_candidates,
    rank_auc,
    recover_rerank_coefficient,
    replay_rerank,
    top1_uncertainty_and_correctness,
)


def _dump(size=6, top_r=3, coefficient=2.5, seed=0):
    """Build a dump shaped exactly like an eval run's, with a known coefficient."""

    generator = np.random.default_rng(seed)
    recall = generator.normal(size=(size, size)).astype(np.float64)
    probability = np.full((size, size), np.nan)
    t2v = np.full((size, size), -np.inf)
    v2t = np.full((size, size), -np.inf)
    rows = np.argsort(-recall, axis=1, kind="stable")[:, :top_r]
    columns = np.argsort(-recall, axis=0, kind="stable")[:top_r]
    for text, videos in enumerate(rows):
        for video in videos:
            probability[text, video] = generator.uniform(-1, 1)
    for video in range(size):
        for text in columns[:, video]:
            if np.isnan(probability[text, video]):
                probability[text, video] = generator.uniform(-1, 1)
    for text, videos in enumerate(rows):
        for video in videos:
            t2v[text, video] = recall[text, video] + coefficient * probability[text, video]
    for video in range(size):
        for text in columns[:, video]:
            v2t[text, video] = recall[text, video] + coefficient * probability[text, video]
    return {
        "t2v": t2v,
        "v2t": v2t,
        "recall": recall,
        "probability": probability,
        "uncertainty": np.where(np.isnan(probability), np.nan, 0.1),
    }


def test_recovered_coefficient_matches_the_scale_the_run_actually_applied():
    """Replaying a rerank needs the coefficient, and the dump does not store it.

    Recovering it from the scores keeps the control honest: a hand-typed lambda
    that disagrees with the run would silently change what is being compared.
    """

    dump = _dump(coefficient=2.5)

    coefficient = recover_rerank_coefficient(dump)

    assert coefficient == pytest.approx(2.5)


def test_recovering_the_coefficient_rejects_scores_it_cannot_explain():
    dump = _dump()
    row, column = np.argwhere(np.isfinite(dump["t2v"]))[0]
    dump["t2v"][row, column] += 1.0

    with pytest.raises(ValueError, match="residual"):
        recover_rerank_coefficient(dump)


def test_replaying_the_recorded_probabilities_reproduces_the_recorded_scores():
    dump = _dump(coefficient=2.5)

    t2v, v2t = replay_rerank(dump, dump["probability"], 2.5)

    np.testing.assert_allclose(t2v, dump["t2v"])
    np.testing.assert_allclose(v2t, dump["v2t"])


def test_replaying_with_a_zero_coefficient_leaves_the_deterministic_order():
    dump = _dump()

    t2v, _ = replay_rerank(dump, dump["probability"], 0.0)

    scored = np.isfinite(dump["t2v"])
    np.testing.assert_allclose(t2v[scored], dump["recall"][scored])
    assert np.isneginf(t2v[~scored]).all()


def test_permutation_keeps_each_rows_probabilities_and_moves_them():
    """A control that changed the score multiset would not isolate tie-breaking."""

    dump = _dump(size=8, top_r=4)

    permuted = permute_probability_within_candidates(
        dump["probability"], np.isfinite(dump["t2v"]), seed=0
    )

    scored = np.isfinite(dump["t2v"])
    for row in range(dump["probability"].shape[0]):
        mask = scored[row]
        np.testing.assert_allclose(
            np.sort(permuted[row, mask]), np.sort(dump["probability"][row, mask])
        )
    assert np.isnan(permuted[~scored]).all()
    assert not np.allclose(permuted[scored], dump["probability"][scored])


def test_permutation_is_reproducible_per_seed():
    dump = _dump(size=8, top_r=4)
    mask = np.isfinite(dump["t2v"])

    first = permute_probability_within_candidates(dump["probability"], mask, seed=3)
    second = permute_probability_within_candidates(dump["probability"], mask, seed=3)
    other = permute_probability_within_candidates(dump["probability"], mask, seed=4)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[mask], other[mask])


def test_auc_reports_perfect_and_inverted_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])

    assert rank_auc(scores, labels) == pytest.approx(1.0)
    assert rank_auc(-scores, labels) == pytest.approx(0.0)


def test_auc_of_a_constant_score_is_uninformative():
    """A frozen uncertainty cannot predict anything, and must not look like it can."""

    scores = np.full(6, 0.0003)
    labels = np.array([0, 1, 0, 1, 1, 0])

    assert rank_auc(scores, labels) == pytest.approx(0.5)


def test_auc_needs_both_classes_present():
    with pytest.raises(ValueError, match="both classes"):
        rank_auc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_top1_uncertainty_is_read_off_the_retrieved_pair():
    scores = np.array([[3.0, 1.0], [2.0, 5.0]])
    uncertainty = np.array([[0.1, 0.9], [0.4, 0.2]])

    values, correct = top1_uncertainty_and_correctness(scores, uncertainty)

    np.testing.assert_allclose(values, [0.1, 0.2])
    np.testing.assert_array_equal(correct, [True, True])


def test_top1_correctness_is_false_when_the_wrong_video_wins():
    scores = np.array([[1.0, 3.0], [2.0, 5.0]])
    uncertainty = np.array([[0.1, 0.9], [0.4, 0.2]])

    values, correct = top1_uncertainty_and_correctness(scores, uncertainty)

    np.testing.assert_allclose(values, [0.9, 0.2])
    np.testing.assert_array_equal(correct, [False, True])


def test_rerank_flips_are_split_into_repairs_and_breakages():
    """A net gain of +2 can be 2 repairs or 32 repairs against 30 breakages.

    Only the split says whether reranking is deciding or coin-flipping.
    """

    baseline = np.array([[1.0, 3.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 3.0]])
    reranked = np.array([[3.0, 1.0, 0.0], [3.0, 0.0, 1.0], [0.0, 1.0, 3.0]])

    repaired, broken = count_rerank_flips(baseline, reranked)

    assert (repaired, broken) == (1, 1)


def test_candidate_recall_ceiling_counts_queries_whose_answer_is_reachable():
    """Reranking cannot beat the pool it was handed."""

    scored = np.array(
        [[True, True, False], [True, False, True], [True, False, True]]
    )

    assert candidate_recall_ceiling(scored) == pytest.approx(2 / 3)
