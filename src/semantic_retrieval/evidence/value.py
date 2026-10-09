"""Predict acquisition outcomes using only current observations and action metadata."""
from dataclasses import dataclass

import torch
from torch import nn

from semantic_retrieval.evidence.types import ActionValueDistribution, TransitionTargets, positive
from semantic_retrieval.evidence.views import coverage_vector, propose_actions


FEATURE_SCHEMA = 'observable-action-v1'


def build_value_features(state, actions, pair_scores):
    expected = tuple((state.query_id, c.canonical_id) for c in state.candidates)
    if set(pair_scores.pair_ids) != set(expected):
        raise ValueError('value features require current fixed candidate predictions')
    allowed = {a.action_id: a for a in propose_actions(state)}
    lp = pair_scores.ranking_log_probability.detach()
    loc = pair_scores.loc.detach()
    probability = lp.exp()
    sorted_lp = lp.sort(descending=True).values
    margin = sorted_lp[0]-sorted_lp[1] if len(lp) > 1 else lp.new_zeros(())
    geometry = torch.stack((margin, -(probability*lp).sum(), lp.mean(), lp.std(unbiased=False)))
    remaining = [0. if limit == 0 else (limit-spent-reserved)/limit
                 for limit, spent, reserved in zip(state.budget.limits.values(), state.budget.spent.values(), state.budget.reserved.values())]
    rows = []
    for action in actions:
        if action.action_id not in allowed or action != allowed[action.action_id]:
            raise ValueError('stale or altered action')
        i = pair_scores.pair_ids.index((state.query_id, action.candidate_id))
        c = next(c for c in state.candidates if c.canonical_id == action.candidate_id)
        f = max(1, c.frame_count-1)
        action_values = [len(action.frame_ids)/len(c.pool_frame_ids), min(action.frame_ids)/f, max(action.frame_ids)/f, action.estimated_cost]
        rows.append(torch.cat((pair_scores.features[i].detach(), torch.stack((loc[i], pair_scores.scale[i].detach().log(), lp[i])),
                               geometry, coverage_vector(c).to(lp), lp.new_tensor(action_values), lp.new_tensor(remaining))))
    return torch.stack(rows) if rows else lp.new_empty((0, pair_scores.features.shape[1]+19))


class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        positive(input_dim, integer=True); positive(hidden_dim, integer=True)
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, features):
        return ActionValueDistribution(self.network(features.detach()))


class MeanRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.network(features.detach()).squeeze(-1)


class GeometryValue(ValueHead):
    def forward(self, features):
        masked = features.detach().clone()
        masked[:, features.shape[1]-18] = 0  # h then loc then log_scale
        return super().forward(masked)


class EvidenceValueModel(nn.Module):
    def __init__(self, matcher, value_head):
        super().__init__()
        self.matcher = matcher.requires_grad_(False).eval()
        self.value_head = value_head
        self.stage = 'value'

    def train(self, mode=True):
        super().train(mode)
        self.matcher.eval()
        return self

    def forward(self, features):
        return self.value_head(features)


@dataclass(frozen=True)
class ValueTrainingBatch:
    features: torch.Tensor
    targets: TransitionTargets
    importance_weights: torch.Tensor | None
