from dataclasses import replace
import pytest
from semantic_retrieval.evidence.views import build_frame_pool, view_frame_ids, propose_actions, coverage_vector


def test_nested_short_and_unique_pool(state_factory):
    assert build_frame_pool(3, 16) == (0, 1, 2)
    assert view_frame_ids(tuple(range(16)), 4) == (0, 7, 11, 15)
    s = state_factory(observed=(0,), frame_count=1)
    assert not propose_actions(s)
    assert coverage_vector(s.candidates[0]).tolist() == [1., 0., 0., 0.]


def test_actions_skip_failed_attempts_and_bind_costs(state_factory):
    s = state_factory(observed=(0, 15))
    c = replace(s.candidates[0], attempted_frame_ids=(0, 7, 15))
    s = replace(s, candidates=(c,))
    action, = propose_actions(s)
    assert action.frame_ids == (11,)
    assert action.position_ids == (11,)
    assert action.reservation.frame_requests == action.reservation.clip_frames == 1
    assert action.reservation.temporal_tokens == 3
    assert action.reservation.mc_elements == 4
    assert action.estimated_cost == 9.
    with pytest.raises(ValueError):
        build_frame_pool(True, 16)
