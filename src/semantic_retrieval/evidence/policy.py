"""Hard resource accounting and one-step risk-neutral evidence acquisition."""
from dataclasses import replace

from semantic_retrieval.evidence.types import AcquireDecision, ActionOutcome, StopDecision, ZERO, positive
from semantic_retrieval.evidence.views import propose_actions


def _legal(state, action):
    if action.state_sha256 != state.state_sha256 or action not in propose_actions(state):
        raise ValueError('stale/invalid acquisition action')


def reserve_action(state, action, *, predicted_counts):
    _legal(state, action)
    if predicted_counts != action.reservation:
        raise ValueError('reservation must equal validated action upper bound')
    return AcquireDecision(action, state.budget.reserve(predicted_counts))


def decide(state, actions, values, budget, *, stop_threshold):
    positive(stop_threshold, zero=True)
    if budget != state.budget or len(values.logits) != len(actions):
        raise ValueError('policy budget/action mismatch')
    ranked = []
    for action, gain in zip(actions, values.expected_gain.detach().cpu().tolist()):
        _legal(state, action)
        if gain > stop_threshold and (budget.spent+budget.reserved+action.reservation).fits(budget.limits):
            ranked.append((-gain/action.estimated_cost, action.candidate_id, action.action_id, action))
    if not ranked:
        return StopDecision('no_positive_affordable_action')
    action = min(ranked, key=lambda x:x[:3])[3]
    return reserve_action(state, action, predicted_counts=action.reservation)


def validate_outcome(action, outcome):
    if not isinstance(outcome, ActionOutcome) or outcome.action_id != action.action_id:
        raise ValueError('acquisition outcome identity mismatch')
    if set(outcome.succeeded_ids)|set(outcome.failed_ids) != set(action.frame_ids):
        raise ValueError('outcome must account for every requested frame')
    if not outcome.actual_counts.fits(action.reservation):
        raise ValueError('actual counts exceed pre-reserved resources')
    if outcome.actual_counts.frame_requests != len(action.frame_ids) or outcome.actual_counts.clip_frames != len(outcome.succeeded_ids):
        raise ValueError('failed requests charged; successful frame encodings charged')
    if len(outcome.feature_references) != len(outcome.succeeded_ids):
        raise ValueError('one verified feature reference required per successful frame')


def acquire(action, state, frame_source):
    reserve_action(state, action, predicted_counts=action.reservation)
    outcome = frame_source(action, state)
    validate_outcome(action, outcome)
    return outcome


def apply_outcome(state, action, outcome):
    _legal(state, action)
    validate_outcome(action, outcome)
    ledger = state.budget.reserve(action.reservation).settle(action.reservation, outcome.actual_counts)
    candidates = tuple(replace(c, observed_frame_ids=tuple(sorted(set(c.observed_frame_ids)|set(outcome.succeeded_ids))),
                               attempted_frame_ids=tuple(sorted(set(c.attempted_frame_ids)|set(action.frame_ids))))
                       if c.canonical_id == action.candidate_id else c for c in state.candidates)
    return replace(state, candidates=candidates, budget=ledger, version=state.version+1)
