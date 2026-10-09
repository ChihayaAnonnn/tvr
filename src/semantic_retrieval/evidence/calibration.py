"""Multiclass calibration is fitted only on the bound calibration role."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from semantic_retrieval.evidence.types import digest, positive, require_digest


def validate_identity(identity):
    keys = {'producer_sha256', 'model_sha256', 'role', 'relevance_space', 'feature_schema_sha256'}
    if set(identity) != keys or identity['role'] != 'V_calibration' or identity['relevance_space'] not in {'reviewed_semantic', 'benchmark_annotation'}:
        raise ValueError('invalid calibration identity/scope')
    for k in ('producer_sha256', 'model_sha256', 'feature_schema_sha256'):
        require_digest(identity[k])


@dataclass(frozen=True)
class ValueCalibration:
    temperature: float
    identity: dict
    count: int
    schema_version: str = 'evidence-calibration-v1'

    def __post_init__(self):
        positive(self.temperature)
        positive(self.count, integer=True)
        validate_identity(self.identity)
        if self.schema_version != 'evidence-calibration-v1':
            raise ValueError('unknown calibration schema')

    def apply(self, logits, *, expected_identity):
        validate_identity(expected_identity)
        if digest(expected_identity) != digest(self.identity):
            raise ValueError('calibration identity mismatch')
        if logits.ndim != 2 or logits.shape[-1] != 3 or not torch.isfinite(logits).all():
            raise ValueError('invalid outcome logits')
        return logits/self.temperature


def fit_value_temperature(logits, targets, *, split, identity):
    validate_identity(identity)
    if split != 'V_calibration' or identity['producer_sha256'] != targets.producer_sha256:
        raise ValueError('calibration requires bound V_calibration transitions')
    if logits.shape != (len(targets.valid_mask),3) or not torch.isfinite(logits).all() or not targets.valid_mask.any():
        raise ValueError('no valid calibration logits/labels')
    x = logits.detach()[targets.valid_mask].double()
    y = targets.outcome_class[targets.valid_mask]
    log_t = torch.zeros((), device=x.device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x/log_t.clamp(-10,10).exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    temperature = log_t.detach().clamp(-10,10).exp()
    if F.cross_entropy(x/temperature, y) > F.cross_entropy(x,y):
        temperature = x.new_tensor(1.)
    return ValueCalibration(float(temperature), dict(identity), int(targets.valid_mask.sum()))
