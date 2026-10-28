import torch
from semantic_retrieval.evidence.types import ActionValueDistribution, StopDecision
from semantic_retrieval.evidence.policy import decide

def test_harmful_actions_stop(state_factory):
    state = state_factory()
    from semantic_retrieval.evidence.views import propose_actions
    actions = propose_actions(state)
    probs = torch.tensor([[.7, .2, .1]]).repeat(len(actions), 1)
    result = decide(state, actions, ActionValueDistribution(probs.log()),
                    state.budget, stop_threshold=0.)
    assert isinstance(result, StopDecision)

def test_failed_acquisition_charges_requests_and_exhausts_attempts(state_factory):
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.types import ActionOutcome, ResourceCounts
    from semantic_retrieval.evidence.policy import acquire, apply_outcome
    state = state_factory()
    action = propose_actions(state)[0]
    def fail_source(a, s):
        return ActionOutcome(a.action_id, (), a.frame_ids,
                             ResourceCounts(len(a.frame_ids), 0, 0, 0), 0., ())
    outcome = acquire(action, state, fail_source)
    updated = apply_outcome(state, action, outcome)
    assert updated.budget.spent.frame_requests == len(action.frame_ids)
    assert updated.budget.spent.clip_frames == 0
    for other in propose_actions(updated):
        if other.candidate_id == action.candidate_id:
            assert not set(other.frame_ids) & set(action.frame_ids)

def test_execution_without_commit_reconciles_exactly_once(journal_case):
    c = journal_case
    c.journal.prepare(c.state, c.decision)
    c.journal.record_execution(c.decision.action.action_id, c.outcome)
    once = c.journal.recover(c.state)
    twice = c.journal.recover(once)
    assert once == twice == c.next_state
    assert once.budget.spent == c.outcome.actual_counts

def test_all_decode_failures_stop_without_free_frames(state_factory):
    from dataclasses import replace
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.types import ActionOutcome, ResourceCounts
    from semantic_retrieval.evidence.policy import acquire, apply_outcome
    state = state_factory(canonical_ids=('a',), observed=(0,), frame_count=3)
    before = state.candidates[0].observed_frame_ids
    while propose_actions(state):
        action = propose_actions(state)[0]
        def source(a,s):
            return ActionOutcome(a.action_id, (), a.frame_ids, ResourceCounts(len(a.frame_ids),0,0,0), 0., ())
        state = apply_outcome(state, action, acquire(action,state,source))
    assert state.budget.spent.frame_requests == 2
    assert state.budget.spent.clip_frames == 0
    assert state.candidates[0].observed_frame_ids == before


def test_cost_changes_action_choice_and_tampering_rejected(state_factory):
    from dataclasses import replace
    import pytest
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.policy import reserve_action
    from semantic_retrieval.evidence.types import ExecutionProfile
    s = state_factory(observed=(0,15))
    b = replace(s.candidates[1], observed_frame_ids=(0,7,15), attempted_frame_ids=(0,7,15))
    s = replace(s, candidates=(s.candidates[0], b), profile=ExecutionProfile(4,(100.,100.,.01,.01)))
    actions = propose_actions(s)
    # a: 2 new frames, b: 1. b has lower absolute gain but higher gain/cost.
    probs = torch.tensor([[.1,.2,.7],[.1,.45,.45]])
    decision = decide(s,actions,ActionValueDistribution(probs.log()),s.budget,stop_threshold=0.)
    assert decision.action.candidate_id == 'b'
    with pytest.raises(ValueError):
        reserve_action(s, replace(actions[0],estimated_cost=.1), predicted_counts=actions[0].reservation)
