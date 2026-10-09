"""Information-matched model variants and common-budget acquisition rules."""
from dataclasses import replace

import torch
from torch.nn import functional as F

from semantic_retrieval.evidence.model import EvidenceMatcherModel
from semantic_retrieval.evidence.policy import decide, reserve_action
from semantic_retrieval.evidence.types import StopDecision, ActionValueDistribution, digest
from semantic_retrieval.evidence.value import build_value_features
from semantic_retrieval.evidence.views import propose_actions, view_frame_ids
from semantic_retrieval.retrieval.index import CanonicalIndex


def build_matcher_variant(config, *, variant):
    base = config.model if hasattr(config,'model') else config
    return EvidenceMatcherModel(replace(base,variant=variant))


def build_initial_index(states, frame_reader, *, identity):
    rows, ids = [], []
    for state in states:
        if state.gallery_version != identity.gallery_version:
            raise ValueError('initial index gallery mismatch')
        for c in state.candidates:
            initial = view_frame_ids(c.pool_frame_ids,state.view_levels[0])
            if not c.observed_frame_ids or not set(c.observed_frame_ids) <= set(initial) or c.attempted_frame_ids != initial:
                raise ValueError('index may only consume initial sparse observations')
            feature = frame_reader(c,c.observed_frame_ids)
            if feature.requires_grad or feature.ndim != 2 or len(feature) != len(c.observed_frame_ids) or not torch.isfinite(feature).all():
                raise ValueError('invalid frozen initial frame features')
            pooled = feature.mean(0)
            if pooled.norm() == 0:
                raise ValueError('zero initial embedding')
            rows.append(F.normalize(pooled,dim=0)); ids.append(c.canonical_id)
    if not rows:
        raise ValueError('empty initial gallery')
    return CanonicalIndex(ids,torch.stack(rows),identity)


def search_initial_candidates(query_features,index,k,*,expected_identity):
    pooled = query_features.pooled_text
    if (pooled.norm(dim=-1) == 0).any():
        raise ValueError('zero query embedding')
    return tuple(tuple(row) for row in index.search(F.normalize(pooled,dim=-1),k,expected_identity=expected_identity))


def build_acquisition_baseline(name, *, seed, config):
    if name not in {'fixed_frames','random','margin','entropy','mean_regressor','distribution'}:
        raise ValueError('unknown acquisition baseline')
    def policy(state, scores, *, value_model=None):
        actions = propose_actions(state)
        affordable = tuple(a for a in actions if (state.budget.spent+state.budget.reserved+a.reservation).fits(state.budget.limits))
        if not affordable:
            return StopDecision('no_affordable_action')
        if name in {'distribution','mean_regressor'}:
            if value_model is None:
                raise ValueError('trained value model required')
            values = value_model(build_value_features(state,actions,scores))
            if name == 'mean_regressor':
                # Direct means use the same greedy rule; never invent class probabilities.
                gains = values.detach().cpu().tolist()
                ranked = [( -g/a.estimated_cost,a.candidate_id,a.action_id,a) for a,g in zip(actions,gains)
                          if g > config.policy.stop_threshold and a in affordable]
                if not ranked:
                    return StopDecision('no_positive_affordable_action')
                action = min(ranked,key=lambda x:x[:3])[3]
                return reserve_action(state,action,predicted_counts=action.reservation)
            return decide(state,actions,values,state.budget,stop_threshold=config.policy.stop_threshold)
        lp = scores.ranking_log_probability.detach() if name in {'margin','entropy'} else None
        if name == 'margin':
            ordered = lp.sort(descending=True).values
            if len(ordered) == 1 or ordered[0]-ordered[1] >= config.policy.stop_threshold:
                return StopDecision('margin_stop')
        if name == 'entropy' and -(lp.exp()*lp).sum() <= config.policy.stop_threshold:
            return StopDecision('entropy_stop')
        # Fixed views advance all candidates through the declared levels in rounds.
        # Attempted frames, including decode failures, determine progress.
        attempted = {c.canonical_id:len(c.attempted_frame_ids) for c in state.candidates}
        ordered = sorted(affordable,key=lambda a:(attempted[a.candidate_id],a.candidate_id) if name == 'fixed_frames' else (0,a.candidate_id))
        identity = dict(seed=seed,query=state.query_id,observations=sorted((c.canonical_id,c.observed_frame_ids,c.attempted_frame_ids) for c in state.candidates))
        action = ordered[int(digest(identity),16)%len(ordered)] if name == 'random' else ordered[0]
        return reserve_action(state,action,predicted_counts=action.reservation)
    return policy
