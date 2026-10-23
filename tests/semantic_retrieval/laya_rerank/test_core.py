import numpy as np
import pytest
from semantic_retrieval.laya_rerank.core import evidence_indices, rerank, compare


def test_frames():
    assert evidence_indices([0, 1, 1, 2]) == [0, 1, 2]
    assert evidence_indices(list(range(12))) == [0, 3, 7, 11]
    with pytest.raises(ValueError): evidence_indices([])


def test_stable_full_ranking():
    original = np.arange(15)
    result = rerank(original, np.array([.1, .9, .9, .2, .3, .4, .5, .6, .7, .8]))
    assert list(result[:2]) == [1, 2]
    assert np.array_equal(result[10:], original[10:])
    assert set(result) == set(original)
    with pytest.raises(ValueError): rerank(original, np.full(10, np.nan))
    with pytest.raises(ValueError): rerank(np.zeros(15, dtype=int), np.ones(10))


def test_pair_metrics():
    a = np.tile(np.arange(12), (2, 1))
    b = a.copy(); b[0, :2] = [1, 0]
    r = compare(a, b, np.array([1, 0]))
    assert r['corrected'] == 1 and r['worsened'] == 0
    assert r['baseline']['r1'] == 50 and r['reranked']['r1'] == 100
    assert r['baseline']['r10'] == r['reranked']['r10']
