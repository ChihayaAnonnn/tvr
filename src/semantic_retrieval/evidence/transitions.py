"""Frozen, provenance-bound one-step evidence acquisition supervision."""
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import nn

from experiment_artifacts import strict_json_load, verify_array_reference
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.evidence.frame_cache import publish_json, safe_path
from semantic_retrieval.evidence.types import TransitionTargets, digest, require_digest
from semantic_retrieval.evidence.views import propose_actions


@dataclass(frozen=True)
class TransitionProducerIdentity:
    matcher_sha256: str
    config_sha256: str
    index_sha256: str
    label_sha256: str
    view_sha256: str
    mc_sha256: str
    state_sampling_sha256: str
    feature_schema_sha256: str

    def __post_init__(self):
        for v in asdict(self).values():
            require_digest(v)

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class TransitionRecord:
    state_sha256: str
    action_id: str
    before_id: str
    after_id: str
    source_components: tuple[str, ...]


@dataclass(frozen=True)
class TransitionReferences:
    index_path: Path
    index_sha256: str
    producer: TransitionProducerIdentity
    arrays: tuple


@dataclass(frozen=True)
class TransitionBatch:
    features: torch.Tensor
    targets: TransitionTargets
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrozenTransitionModel:
    model: nn.Module
    predict: Callable
    observe: Callable
    build_features: Callable

    def validate(self, producer):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('transition producer must be frozen/eval')
        if model_sha256(self.model) != producer.matcher_sha256:
            raise ValueError('transition producer weight hash mismatch')


def label_transition(before_id, after_id, relevance: Mapping[str,int], *, producer_sha256):
    before = relevance.get(before_id, -1); after = relevance.get(after_id, -1)
    if before not in {-1,0,1} or after not in {-1,0,1}:
        raise ValueError('invalid transition relevance')
    valid = before != -1 and after != -1
    cls = after-before+1 if valid else 1
    return TransitionTargets(torch.tensor([cls], dtype=torch.long), torch.tensor([valid]), producer_sha256)


def sample_states(initial_states, *, seed, max_steps, observe):
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError('invalid state sampling steps')
    generator = torch.Generator().manual_seed(seed)
    for initial in initial_states:
        state = initial
        yield state
        for _ in range(max_steps):
            actions = tuple(a for a in propose_actions(state) if (state.budget.spent+state.budget.reserved+a.reservation).fits(state.budget.limits))
            if not actions:
                break
            action = actions[int(torch.randint(len(actions), (), generator=generator))]
            updated = observe(state, action)
            if updated.state_sha256 == state.state_sha256:
                raise ValueError('acquisition made no state progress')
            state = updated
            yield state


def _top1(scores):
    if not scores.pair_ids or len({q for q, _ in scores.pair_ids}) != 1:
        raise ValueError('transition prediction must cover one query')
    return min(zip(scores.ranking_log_probability.detach().cpu().tolist(), scores.pair_ids), key=lambda x: (-x[0], x[1][1]))[1][1]


def compile_transitions(states, actions, frozen_matcher, targets, *, store, producer_identity):
    frozen_matcher.validate(producer_identity)
    records, features, classes, valid, components = [], [], [], [], []
    with torch.no_grad():
        for state in sorted(states, key=lambda s: s.state_sha256):
            prediction = frozen_matcher.predict(state)
            before = _top1(prediction)
            # The fixed candidate set and all legal actions are compulsory.
            affordable = lambda a:(state.budget.spent+state.budget.reserved+a.reservation).fits(state.budget.limits)
            legal = tuple(a for a in propose_actions(state) if affordable(a))
            supplied = actions(state) if callable(actions) else tuple(a for a in actions if a.state_sha256 == state.state_sha256)
            supplied = tuple(a for a in supplied if affordable(a))
            if {a.action_id for a in supplied} != {a.action_id for a in legal}:
                raise ValueError('transition actions must enumerate all legal actions')
            relevance, source_components = targets[state.query_id]
            if not source_components:
                raise ValueError('transition source components required')
            component = digest(dict(sources=tuple(sorted(source_components))))
            for action in sorted(legal, key=lambda a: a.action_id):
                after_state = frozen_matcher.observe(state, action)
                if tuple(c.canonical_id for c in after_state.candidates) != tuple(c.canonical_id for c in state.candidates):
                    raise ValueError('transition changed candidate set')
                after_scores = frozen_matcher.predict(after_state)
                if prediction.sampling_sha256 != after_scores.sampling_sha256:
                    raise ValueError('before/after integration identity mismatch')
                after = _top1(after_scores)
                target = label_transition(before, after, relevance, producer_sha256=producer_identity.sha256)
                feature = frozen_matcher.build_features(state, (action,), prediction)
                if feature.ndim != 2 or len(feature) != 1 or feature.requires_grad or not torch.isfinite(feature).all():
                    raise ValueError('invalid observable transition features')
                features.append(feature[0].cpu()); classes.append(target.outcome_class.item()); valid.append(target.valid_mask.item())
                components.append(component)
                records.append(TransitionRecord(state.state_sha256, action.action_id, before, after, tuple(sorted(source_components))))
    frozen_matcher.validate(producer_identity)
    if not records:
        raise ValueError('no legal transition records')
    refs = tuple(store.write(key, value) for key, value in (
        ('transition_features', torch.stack(features)), ('transition_class', torch.tensor(classes, dtype=torch.long)),
        ('transition_valid', torch.tensor(valid, dtype=torch.bool))))
    # Keep only small provenance in JSON; partitioning large indices is required before writing.
    payload = dict(schema_version='evidence-transitions-v1', producer=asdict(producer_identity),
                   arrays=[asdict(r) for r in refs], records=[asdict(r) for r in records], component_ids=components)
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    validate_evidence_document('transition_index',payload)
    p = store.run_root/'provenance'/'transitions.json'
    if p.exists():
        raise ValueError('transition output already exists')
    publish_json(p, payload)
    return TransitionReferences(p, hashlib.sha256(p.read_bytes()).hexdigest(), producer_identity, refs)


def read_transitions(references, *, expected_producer):
    safe_path(references.index_path)
    if references.producer != expected_producer or hashlib.sha256(references.index_path.read_bytes()).hexdigest() != references.index_sha256:
        raise ValueError('transition identity/index mismatch')
    with references.index_path.open() as h:
        p = strict_json_load(h)
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    validate_evidence_document('transition_index',p)
    if p['schema_version'] != 'evidence-transitions-v1' or p['producer'] != asdict(expected_producer):
        raise ValueError('transition producer mismatch')
    if p['arrays'] != [dict(asdict(r), shape=list(r.shape)) for r in references.arrays]:
        raise ValueError('transition array binding mismatch')
    root = references.index_path.parent.parent
    features, classes, valid = [torch.from_numpy(verify_array_reference(root, r).copy()) for r in references.arrays]
    if len(features) != len(p['records']) or len(p['component_ids']) != len(features):
        raise ValueError('transition record count mismatch')
    return TransitionBatch(features, TransitionTargets(classes, valid, expected_producer.sha256), tuple(p['component_ids']))
