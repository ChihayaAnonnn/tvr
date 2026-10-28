import torch
from semantic_retrieval.evidence.views import build_frame_pool, view_frame_ids
from semantic_retrieval.evidence.encoding import pack_observations
from semantic_retrieval.models.temporal import TemporalEncoder

def test_nested_and_single_frame():
    pool = build_frame_pool(16, 16)
    assert set(view_frame_ids(pool, 4)) < set(view_frame_ids(pool, 8))
    assert view_frame_ids(build_frame_pool(1, 16), 16) == (0,)

def test_same_video_different_queries_stays_separate(state_factory, query_features, frame_reader):
    a = state_factory(query_id="q0", canonical_ids=("a",), observed=(0,))
    b = state_factory(query_id="q1", canonical_ids=("a",), observed=(15,))
    out = pack_observations((a, b), query_features, frame_reader)
    assert len(out.observation_keys) == 2
    assert out.pair_indices[:, 1].tolist() == [0, 1]
    assert out.position_ids[:, 0].tolist() == [0, 15]

def test_hidden_frames_cannot_change_temporal_output():
    t = TemporalEncoder(8, 1, 2, 16).eval()
    x = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, False, True]])
    ids = torch.tensor([[0, 7, 15]])
    before = t(x, mask, position_ids=ids)
    x[:, 1] = 999
    after = t(x, mask, position_ids=ids)
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])

def test_stable_positions_duplicate_rejected_and_padding_safe():
    import pytest
    t = TemporalEncoder(8, 1, 2, 16).eval()
    x = torch.randn(1, 3, 8); mask = torch.tensor([[True, False, True]])
    ids = torch.tensor([[0, -999, 15]])
    a = t(x, mask, position_ids=ids)
    perm = torch.tensor([2, 1, 0])
    b = t(x[:, perm], mask[:, perm], position_ids=ids[:, perm])
    torch.testing.assert_close(a[0][:, perm], b[0])
    torch.testing.assert_close(a[1], b[1])
    with pytest.raises(ValueError):
        t(x, mask, position_ids=torch.tensor([[0, 0, 0]]))


def test_same_observation_shared_but_never_union(state_factory, query_features, frame_reader):
    a = state_factory(query_id='q0', canonical_ids=('a',), observed=(0,))
    b = state_factory(query_id='q1', canonical_ids=('a',), observed=(0,))
    out = pack_observations((a,b), query_features, frame_reader)
    assert len(out.observation_keys) == 1
    assert frame_reader.requests == [('a', (0,))]
