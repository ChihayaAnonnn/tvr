"""Canonical sets; grouping is optional and never establishes answerability."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SemanticSetOutput:
    canonical_ids: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    supports: tuple[float, ...]
    entropy: float | None
    conflict_signal: bool


class SemanticSet:
    @staticmethod
    def build(candidate_ids, pair_scores, *, group_predictor=None, support_threshold=None):
        if pair_scores.logits.shape != (len(candidate_ids),) or not torch.isfinite(pair_scores.logits).all():
            raise ValueError("invalid candidate scores")
        values = {}
        for i, p in zip(candidate_ids, pair_scores.logits.sigmoid().detach().cpu().tolist()):
            if not isinstance(i, str) or not i:
                raise ValueError("invalid canonical ID")
            if i in values and values[i] != p:
                raise ValueError("alias scores conflict")
            values[i] = p
        ids = tuple(sorted(values))
        groups = tuple((i,) for i in ids)
        if group_predictor is not None:
            if not getattr(group_predictor, "validated", False):
                raise ValueError("group predictor requires validation")
            groups = group_predictor.groups(candidate_ids, pair_scores)
            flat = [i for g in groups for i in g]
            if sorted(flat) != list(ids) or len(flat) != len(set(flat)):
                raise ValueError("group predictor must partition canonical IDs")
        supports = tuple(max(values[i] for i in g) for g in groups)
        entropy = None
        if support_threshold is not None:
            if not math.isfinite(support_threshold) or not 0 <= support_threshold <= 1:
                raise ValueError("invalid support threshold")
            accepted = [x for x in supports if x >= support_threshold]
            total = sum(accepted)
            entropy = -sum((x / total) * math.log(x / total) for x in accepted if x > 0) if total else 0.0
        return SemanticSetOutput(ids, groups, supports, entropy, False)
