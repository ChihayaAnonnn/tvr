"""Proper objectives expose sums and effective weights, never local averages."""
import torch
from semantic_retrieval.evidence.matcher import mc_log_probabilities, validate_predictive_inputs
from semantic_retrieval.evidence.types import LossTerm


def probabilistic_contrastive(loc, scale, offsets, labels, noise, *, temperature):
    validate_predictive_inputs(loc, scale, offsets, noise, temperature)
    if labels.dtype != torch.long or labels.shape != loc.shape or not ((labels >= -1) & (labels <= 1)).all():
        raise ValueError('invalid relevance labels')
    total = loc.sum()*0+scale.sum()*0
    count = 0
    for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        y = labels[a:b]
        known = y != -1
        if not (y == 1).any() or not (y == 0).any():
            continue
        selected = y[known]
        off = torch.tensor([0, int(known.sum())], device=offsets.device)
        lp = mc_log_probabilities(loc[a:b][known], scale[a:b][known], off, noise[:, a:b][:, known], temperature=temperature)
        total = total-lp[selected == 1].mean()
        count += 1
    return LossTerm(total, loc.new_tensor(float(count)))


def gaussian_score_nll(loc, scale, target, valid_mask):
    if loc.shape != scale.shape or target.shape != loc.shape or valid_mask.shape != loc.shape or valid_mask.dtype != torch.bool:
        raise ValueError('invalid Gaussian target shapes')
    if any(not torch.isfinite(x).all() for x in (loc, scale, target)) or not (scale > 0).all():
        raise ValueError('invalid Gaussian parameters/target')
    # Slice before squaring: unused extremes must not overflow or poison gradients.
    mu, sigma, y = loc[valid_mask], scale[valid_mask], target.detach()[valid_mask]
    loss = sigma.log()+.5*((y-mu)/sigma).square()
    return LossTerm(loss.sum()+loc.sum()*0+scale.sum()*0, valid_mask.sum().to(loc.dtype))


def _effective_weights(values, targets, importance_weights):
    if len(values) != len(targets.outcome_class):
        raise ValueError('value target count mismatch')
    weights = torch.ones(len(values), device=values.device, dtype=values.dtype) if importance_weights is None else importance_weights.detach().to(values)
    if weights.shape != targets.valid_mask.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid importance weights')
    return weights[targets.valid_mask]


def value_nll(logits, targets, *, importance_weights=None):
    if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
        raise ValueError('finite [A,3] outcome logits required')
    w = _effective_weights(logits, targets, importance_weights)
    valid = targets.valid_mask
    nll = -logits[valid].log_softmax(-1).gather(1, targets.outcome_class[valid,None]).squeeze(-1)
    return LossTerm((w*nll).sum()+logits.sum()*0, w.sum())


def mean_gain_mse(predictions, targets, *, importance_weights=None):
    if predictions.ndim != 1 or not torch.isfinite(predictions).all():
        raise ValueError('finite mean gain vector required')
    w = _effective_weights(predictions, targets, importance_weights)
    valid = targets.valid_mask
    gain = targets.outcome_class[valid].to(predictions.dtype)-1
    return LossTerm((w*(predictions[valid]-gain).square()).sum()+predictions.sum()*0, w.sum())
