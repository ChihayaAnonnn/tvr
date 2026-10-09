"""Stage-specific optimization and frozen richer-view score supervision."""
from dataclasses import dataclass

import torch
from torch import nn

from semantic_retrieval.evidence.objectives import gaussian_score_nll, probabilistic_contrastive
from semantic_retrieval.evidence.types import EvidenceTargets, LossTerm, ObservedFeatures


@dataclass(frozen=True)
class MatcherTrainingBatch:
    observed: ObservedFeatures
    rich: ObservedFeatures
    targets: EvidenceTargets


def freeze_reference(model):
    return model.requires_grad_(False).eval()


def build_stage_optimizer(model, config):
    stage = getattr(model, 'stage', 'value')
    allowed = {'reference': ('temporal.', 'matcher.base.'),
               'matcher': ('temporal.', 'matcher.', 'log_temperature'), 'value': ('value_head.',)}
    if stage not in allowed:
        raise ValueError('no trainable optimizer for this stage')
    prefixes = config.trainable_prefixes
    names = dict(model.named_parameters())
    variant = getattr(getattr(model, 'config', None), 'variant', None)
    selected = [n for n in names if n.startswith(prefixes) and not (variant in {'deterministic','fixed_scale','learned_temperature'} and n.startswith('matcher.scale_head.'))]
    if not selected or any(not n.startswith(allowed[stage]) for n in selected) or any(not any(n.startswith(p) for n in names) for p in prefixes):
        raise ValueError('stage parameter allowlist violation')
    for n, p in names.items():
        p.requires_grad_(n in selected)
    optimizer = torch.optim.AdamW([names[n] for n in selected], lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. if config.scheduler_name == 'constant' else max(0., 1-step/config.scheduler_steps))
    return optimizer, scheduler


def rich_score_targets(reference, rich_observed, *, current_observed):
    if reference.training or any(p.requires_grad for p in reference.parameters()):
        raise ValueError('reference must be frozen in eval mode')
    if rich_observed.pair_ids != current_observed.pair_ids:
        raise ValueError('reference/current pair identities differ')
    valid = []
    for (_, r), (_, c) in zip(rich_observed.pair_indices.tolist(), current_observed.pair_indices.tolist()):
        rich_key = rich_observed.observation_keys[r]
        current_key = current_observed.observation_keys[c]
        if rich_key[:2] != current_key[:2] or not set(current_key[2]) <= set(rich_key[2]):
            raise ValueError('rich target must share source and contain current evidence')
        valid.append(rich_key[2] != current_key[2])
    with torch.no_grad():
        scores = reference.point_scores(rich_observed).logits.detach()
    return scores, torch.tensor(valid, device=scores.device, dtype=torch.bool)


def pair_offsets(pair_ids, device):
    if not pair_ids:
        return torch.tensor([0,0], dtype=torch.long, device=device)
    offsets = [0]
    seen = set()
    for i, (query, _) in enumerate(pair_ids):
        if i == 0 or query != pair_ids[i-1][0]:
            if query in seen:
                raise ValueError('noncontiguous query pairs')
            seen.add(query)
            if i:
                offsets.append(i)
    return torch.tensor(offsets+[len(pair_ids)], dtype=torch.long, device=device)


def matcher_loss(student_scores, targets, *, noise, config):
    offsets = pair_offsets(student_scores.pair_ids, student_scores.loc.device)
    if config.model.variant == 'standard_probabilistic' and config.train.loss_weights['score'] != 0:
        raise ValueError('standard probabilistic baseline requires score weight zero')
    if config.model.variant in {'deterministic', 'learned_temperature'}:
        pc = deterministic_contrastive(student_scores.loc, offsets, targets.relevance,
                                       temperature=student_scores.point_temperature)
    else:
        pc = probabilistic_contrastive(student_scores.loc, student_scores.scale, offsets, targets.relevance, noise,
                                      temperature=config.sampling.temperature)
    score = gaussian_score_nll(student_scores.loc, student_scores.scale, targets.rich_scores, targets.rich_valid)
    return {'pc': pc, 'score': score}


def deterministic_contrastive(logits, offsets, labels, *, temperature):
    from semantic_retrieval.models.pair_matcher import validate_offsets
    validate_offsets(offsets,len(logits))
    temperature = torch.as_tensor(temperature,device=logits.device)
    if temperature.ndim or not torch.isfinite(temperature) or temperature <= 0 or labels.shape != logits.shape or labels.dtype != torch.long or not ((labels >= -1)&(labels <= 1)).all():
        raise ValueError('invalid point contrastive inputs')
    total = logits.sum()*0+temperature*0
    count = 0
    for a,b in zip(offsets[:-1].tolist(),offsets[1:].tolist()):
        y = labels[a:b]
        if (y == 1).any() and (y == 0).any():
            known = y != -1
            total = total-(logits[a:b][known]/temperature).log_softmax(0)[y[known] == 1].mean()
            count += 1
    return LossTerm(total,logits.new_tensor(float(count)))


@dataclass(frozen=True)
class StepMetrics:
    loss_sums: dict
    weight_sums: dict
    step: int
    stepped: bool


def global_loss(terms, weights):
    import torch.distributed as dist
    from semantic_retrieval.evidence.types import positive
    if not terms or set(terms)-set(weights):
        raise ValueError('explicit nonempty loss terms and weights required')
    for value in weights.values():
        positive(value,zero=True)
    identity = (tuple(sorted(terms)),tuple(sorted(weights.items())))
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        gathered = [None]*world
        dist.all_gather_object(gathered,identity)
        if any(x != identity for x in gathered):
            raise ValueError('rank loss keys/weights differ')
    total = sum(term.numerator*0 for term in terms.values())
    active = False
    for key in sorted(terms):
        term = terms[key]
        count = term.denominator.detach().clone()
        if world > 1:
            dist.all_reduce(count)
        if count > 0 and weights[key] > 0:
            total = total+weights[key]*term.numerator*(world/count)
            active = True
    return total, active


class EvidenceStepper:
    def __init__(self, model, optimizer, scheduler, *, weights, loss_builder=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.weights, self.loss_builder = weights, loss_builder
        self.steps = 0

    def step(self, batch):
        import torch.distributed as dist
        if dist.is_initialized():
            stage = getattr(self.model.module if hasattr(self.model,'module') else self.model,'stage',None)
            stages = [None]*dist.get_world_size()
            dist.all_gather_object(stages,stage)
            if any(x != stage for x in stages):
                raise ValueError('rank stages differ')
        self.optimizer.zero_grad(set_to_none=True)
        terms = self.loss_builder(self.model,batch) if self.loss_builder is not None else batch
        loss,active = global_loss(terms,self.weights)
        if active:
            loss.backward()
            parameters = [p for group in self.optimizer.param_groups for p in group['params']]
            finite = torch.tensor(int(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)),device=loss.device)
            if dist.is_initialized():
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite:
                raise ValueError('nonfinite evidence gradient')
            self.optimizer.step(); self.scheduler.step(); self.steps += 1
        return StepMetrics({k:float(t.numerator.detach()) for k,t in terms.items()},
                           {k:float(t.denominator) for k,t in terms.items()},self.steps,active)
