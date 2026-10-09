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
               'matcher': ('temporal.', 'matcher.'), 'value': ('value_head.',)}
    if stage not in allowed:
        raise ValueError('no trainable optimizer for this stage')
    prefixes = config.trainable_prefixes
    names = dict(model.named_parameters())
    selected = [n for n in names if n.startswith(prefixes)]
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
    pc = probabilistic_contrastive(student_scores.loc, student_scores.scale, offsets, targets.relevance, noise,
                                  temperature=config.sampling.temperature)
    score = gaussian_score_nll(student_scores.loc, student_scores.scale, targets.rich_scores, targets.rich_valid)
    return {'pc': pc, 'score': score}


def deterministic_contrastive(logits, offsets, labels, *, temperature):
    # Zero integration noise yields the point softmax while scale stays valid.
    return probabilistic_contrastive(logits, torch.ones_like(logits), offsets, labels,
                                    torch.zeros((1,len(logits)), device=logits.device), temperature=temperature)
