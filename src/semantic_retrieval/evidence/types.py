"""Explicit evidence identities; observations never carry hidden supervision."""
import hashlib
import math
import re
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.types import EncodedFeatures


def digest(value):
    return hashlib.sha256(artifact_json_bytes(value)).hexdigest()


def require_digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('expected SHA-256 identity')


def positive(value, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite numeric value')
    if integer and type(value) is not int or (value < 0 if zero else value <= 0):
        raise ValueError('invalid numeric range or type')


@dataclass(frozen=True)
class ResourceCounts:
    frame_requests: int
    clip_frames: int
    temporal_tokens: int
    mc_elements: int

    def __post_init__(self):
        for v in self.values():
            positive(v, integer=True, zero=True)

    def values(self):
        return tuple(asdict(self).values())

    def __add__(self, other):
        return ResourceCounts(*(a+b for a, b in zip(self.values(), other.values())))

    def __sub__(self, other):
        return ResourceCounts(*(a-b for a, b in zip(self.values(), other.values())))

    def fits(self, other):
        return all(a <= b for a, b in zip(self.values(), other.values()))


ZERO = ResourceCounts(0, 0, 0, 0)


@dataclass(frozen=True)
class BudgetLedger:
    limits: ResourceCounts
    spent: ResourceCounts
    reserved: ResourceCounts

    def __post_init__(self):
        if not all(isinstance(x, ResourceCounts) for x in (self.limits, self.spent, self.reserved)):
            raise ValueError('invalid budget counts')
        if not (self.spent+self.reserved).fits(self.limits):
            raise ValueError('budget exceeded')


    def reserve(self, counts):
        return BudgetLedger(self.limits, self.spent, self.reserved+counts)

    def settle(self, reservation, actual):
        if not actual.fits(reservation) or not reservation.fits(self.reserved):
            raise ValueError('invalid budget settlement')
        return BudgetLedger(self.limits, self.spent+actual, self.reserved-reservation)


@dataclass(frozen=True)
class CandidateObservation:
    canonical_id: str
    content_sha256: str
    frame_count: int
    pool_frame_ids: tuple[int, ...]
    observed_frame_ids: tuple[int, ...]
    attempted_frame_ids: tuple[int, ...]

    def __post_init__(self):
        require_digest(self.content_sha256)
        positive(self.frame_count, integer=True)
        if not self.canonical_id:
            raise ValueError('empty candidate identity')
        for ids in (self.pool_frame_ids, self.observed_frame_ids, self.attempted_frame_ids):
            if not isinstance(ids, tuple) or tuple(sorted(set(ids))) != ids:
                raise ValueError('frame identities must be sorted and unique')
            if any(type(i) is not int or not 0 <= i < self.frame_count for i in ids):
                raise ValueError('invalid frame identity')
        if not self.observed_frame_ids or not set(self.observed_frame_ids) <= set(self.attempted_frame_ids) <= set(self.pool_frame_ids):
            raise ValueError('observed ⊆ attempted ⊆ pool required with nonempty observations')


@dataclass(frozen=True)
class ExecutionProfile:
    mc_samples: int
    cost_weights: tuple[float, ...]

    def __post_init__(self):
        positive(self.mc_samples, integer=True)
        if len(self.cost_weights) != 4:
            raise ValueError('four cost weights required')
        for x in self.cost_weights:
            positive(x)


@dataclass(frozen=True)
class ObservationState:
    query_id: str
    gallery_version: str
    candidates: tuple[CandidateObservation, ...]
    view_levels: tuple[int, ...]
    profile: ExecutionProfile
    budget: BudgetLedger
    version: int

    def __post_init__(self):
        positive(self.version, integer=True, zero=True)
        if not self.query_id or not self.gallery_version or not self.candidates:
            raise ValueError('nonempty query/gallery/candidates required')
        if len({c.canonical_id for c in self.candidates}) != len(self.candidates):
            raise ValueError('duplicate canonical candidate')
        if not self.view_levels or tuple(sorted(set(self.view_levels))) != self.view_levels:
            raise ValueError('strictly increasing view levels required')
        for x in self.view_levels:
            positive(x, integer=True)

    @property
    def state_sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class EvidenceAction:
    action_id: str
    state_sha256: str
    candidate_id: str
    frame_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    reservation: ResourceCounts
    estimated_cost: float

    def __post_init__(self):
        require_digest(self.state_sha256)
        positive(self.estimated_cost)
        if not self.action_id or not self.candidate_id or not self.frame_ids or len(self.frame_ids) != len(self.position_ids):
            raise ValueError('invalid action identity/frames')
        for ids in (self.frame_ids, self.position_ids):
            if tuple(sorted(set(ids))) != ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('action frames/positions must be unique nonnegative integers')


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    succeeded_ids: tuple[int, ...]
    failed_ids: tuple[int, ...]
    actual_counts: ResourceCounts
    elapsed_seconds: float
    feature_references: tuple

    def __post_init__(self):
        positive(self.elapsed_seconds, zero=True)
        if not self.action_id or set(self.succeeded_ids) & set(self.failed_ids):
            raise ValueError('invalid outcome')
        for ids in (self.succeeded_ids, self.failed_ids):
            if tuple(sorted(set(ids))) != ids:
                raise ValueError('outcome frames must be unique and sorted')


@dataclass(frozen=True)
class ObservedFeatures:
    features: EncodedFeatures
    position_ids: Tensor
    pair_indices: Tensor
    query_offsets: Tensor
    pair_ids: tuple[tuple[str, str], ...]
    observation_keys: tuple
    pair_coverage: Tensor

    def __post_init__(self):
        p = len(self.pair_ids)
        f = self.features
        if self.pair_coverage.shape != (p, 4) or not torch.isfinite(self.pair_coverage).all():
            raise ValueError("invalid observed coverage")
        if self.position_ids.dtype != torch.long or self.position_ids.shape != f.frame_mask.shape:
            raise ValueError('invalid observed positions')
        if f.text_mask.dtype != torch.bool or f.frame_mask.dtype != torch.bool or not f.frame_mask.any(-1).all():
            raise ValueError('invalid observed masks')
        if self.pair_indices.dtype != torch.long or self.pair_indices.shape != (p, 2):
            raise ValueError('invalid pair indices')
        off = self.query_offsets
        if off.dtype != torch.long or off.ndim != 1 or len(off) != len(f.text_tokens)+1 or off[0] != 0 or off[-1] != p or not (off[1:] > off[:-1]).all():
            raise ValueError('invalid query offsets')
        if (self.pair_indices < 0).any() or (self.pair_indices[:, 0] >= len(f.text_tokens)).any() or (self.pair_indices[:, 1] >= len(f.frame_tokens)).any():
            raise ValueError('pair index out of range')
        for q, (a, b) in enumerate(zip(off[:-1].tolist(), off[1:].tolist())):
            if not (self.pair_indices[a:b, 0] == q).all() or len(set(self.pair_ids[a:b])) != b-a:
                raise ValueError('pair rows must be contiguous and unique per query')
        for t in (f.text_tokens, f.pooled_text, f.frame_tokens, f.pooled_video):
            if not torch.isfinite(t).all():
                raise ValueError('nonfinite observed features')


@dataclass(frozen=True)
class PairPredictiveScores:
    loc: Tensor
    scale: Tensor
    features: Tensor
    ranking_log_probability: Tensor
    pair_ids: tuple
    sampling_sha256: str

    def __post_init__(self):
        require_digest(self.sampling_sha256)
        n = len(self.pair_ids)
        if any(x.shape != (n,) for x in (self.loc, self.scale, self.ranking_log_probability)) or self.features.ndim != 2 or len(self.features) != n:
            raise ValueError('invalid predictive shapes')
        if any(not torch.isfinite(x).all() for x in (self.loc, self.scale, self.features, self.ranking_log_probability)) or not (self.scale > 0).all():
            raise ValueError('nonfinite scores or nonpositive scale')


@dataclass(frozen=True)
class EvidenceTargets:
    relevance: Tensor
    rich_scores: Tensor
    rich_valid: Tensor
    reference_sha256: str

    def __post_init__(self):
        require_digest(self.reference_sha256)
        if self.relevance.ndim != 1 or self.relevance.dtype != torch.long or not ((self.relevance >= -1) & (self.relevance <= 1)).all():
            raise ValueError('invalid relevance labels')
        if self.rich_scores.shape != self.relevance.shape or self.rich_valid.shape != self.relevance.shape or self.rich_valid.dtype != torch.bool or not torch.isfinite(self.rich_scores).all():
            raise ValueError('invalid reference supervision')


@dataclass(frozen=True)
class TransitionTargets:
    outcome_class: Tensor
    valid_mask: Tensor
    producer_sha256: str

    def __post_init__(self):
        require_digest(self.producer_sha256)
        if self.outcome_class.ndim != 1 or self.outcome_class.dtype != torch.long or self.valid_mask.dtype != torch.bool or self.valid_mask.shape != self.outcome_class.shape or not ((self.outcome_class >= 0) & (self.outcome_class <= 2)).all():
            raise ValueError('invalid transition target; unknown uses valid_mask')


@dataclass(frozen=True)
class LossTerm:
    numerator: Tensor
    denominator: Tensor

    def __post_init__(self):
        if self.numerator.ndim or self.denominator.ndim or not torch.isfinite(self.numerator) or not torch.isfinite(self.denominator) or self.denominator < 0 or self.denominator.requires_grad:
            raise ValueError('invalid loss sum/count')


@dataclass(frozen=True)
class ActionValueDistribution:
    logits: Tensor

    def __post_init__(self):
        if self.logits.ndim != 2 or self.logits.shape[1] != 3 or not torch.isfinite(self.logits).all():
            raise ValueError('expected finite [A,3] logits')

    @property
    def expected_gain(self):
        p = self.logits.softmax(-1)
        return p[:, 2]-p[:, 0]


@dataclass(frozen=True)
class AcquireDecision:
    action: EvidenceAction
    ledger: BudgetLedger


@dataclass(frozen=True)
class StopDecision:
    reason: str


def state_from_payload(payload):
    p = dict(payload)
    p['candidates'] = tuple(CandidateObservation(**{**c, **{k: tuple(c[k]) for k in ('pool_frame_ids', 'observed_frame_ids', 'attempted_frame_ids')}}) for c in p['candidates'])
    p['view_levels'] = tuple(p['view_levels'])
    p['profile'] = ExecutionProfile(p['profile']['mc_samples'], tuple(p['profile']['cost_weights']))
    p['budget'] = BudgetLedger(**{k: ResourceCounts(**v) for k,v in p['budget'].items()})
    return ObservationState(**p)


def action_from_payload(payload):
    p = dict(payload)
    p['frame_ids'] = tuple(p['frame_ids']); p['position_ids'] = tuple(p['position_ids'])
    p['reservation'] = ResourceCounts(**p['reservation'])
    return EvidenceAction(**p)


def outcome_from_payload(payload):
    from experiment_artifacts import ArrayReference
    p = dict(payload)
    p['succeeded_ids'] = tuple(p['succeeded_ids']); p['failed_ids'] = tuple(p['failed_ids'])
    p['actual_counts'] = ResourceCounts(**p['actual_counts'])
    p['feature_references'] = tuple(ArrayReference(**{**r, 'shape': tuple(r['shape'])}) for r in p['feature_references'])
    return ActionOutcome(**p)
