"""Candidate ranking and scoped outcome/episode evaluation."""
from dataclasses import dataclass
import math
import numpy as np
import torch

from semantic_retrieval.evidence.types import ResourceCounts, positive


@dataclass(frozen=True)
class EvidenceReport:
    summary_metrics: dict[str,float]
    array_references: tuple
    undefined_reasons: dict[str,str]
    identity: dict


@dataclass(frozen=True)
class EvidenceEvaluationRecord:
    query_id: str
    initial_top1_label: int
    final_top1_label: int
    candidate_presence: int
    costs: ResourceCounts
    elapsed_seconds: float

    def __post_init__(self):
        if not self.query_id or any(x not in {-1,0,1} for x in (self.initial_top1_label,self.final_top1_label,self.candidate_presence)):
            raise ValueError('invalid evaluation record labels')
        positive(self.elapsed_seconds,zero=True)


def rank_candidates(log_probability, canonical_ids):
    if log_probability.shape != (len(canonical_ids),) or not torch.isfinite(log_probability).all() or len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError('invalid finite candidate scores/identities')
    return tuple(i for _,i in sorted(zip(log_probability.detach().cpu().tolist(),canonical_ids),key=lambda x:(-x[0],x[1])))


def score_outcomes(logits, targets):
    if logits.shape != (len(targets.valid_mask),3) or not torch.isfinite(logits).all():
        raise ValueError('invalid outcome predictions')
    valid = targets.valid_mask
    result = {'valid_count':float(valid.sum()),'unknown_count':float((~valid).sum())}
    if not valid.any():
        return result
    lp = logits[valid].log_softmax(-1)
    y = targets.outcome_class[valid]
    result['nll'] = float(-lp.gather(1,y[:,None]).mean())
    target = torch.nn.functional.one_hot(y,3).to(lp)
    result['brier'] = float((lp.exp()-target).square().sum(-1).mean())
    return result


def evaluate_evidence(records, *, relevance_space, source_components):
    if relevance_space not in {'benchmark_annotation','reviewed_semantic'} or len(source_components) != len(records) or not records:
        raise ValueError('explicit evaluation scope/components required')
    labels = [r.final_top1_label for r in records]
    known = [x for x in labels if x != -1]
    summary = dict(episode_count=float(len(records)), known_count=float(len(known)), unknown_count=float(labels.count(-1)),
                   corrections=float(sum(r.initial_top1_label == 0 and r.final_top1_label == 1 for r in records)),
                   worsenings=float(sum(r.initial_top1_label == 1 and r.final_top1_label == 0 for r in records)),
                   elapsed_seconds=sum(r.elapsed_seconds for r in records), component_count=float(len(set(source_components))))
    undefined = {}
    if known:
        summary['known_top1_accuracy'] = sum(known)/len(known)
    else:
        undefined['known_top1_accuracy'] = 'no known final relevance labels'
    if len(known) == len(records):
        summary['overall_top1_accuracy'] = sum(known)/len(records)
    else:
        undefined['overall_top1_accuracy'] = 'incomplete final relevance labels'
    pres = [r.candidate_presence for r in records]
    if -1 not in pres:
        summary['initial_miss_count'] = float(pres.count(0))
    else:
        undefined['initial_miss_count'] = 'incomplete candidate relevance'
    for i,name in enumerate(('frame_requests','clip_frames','temporal_tokens','mc_elements')):
        summary[name] = float(sum(r.costs.values()[i] for r in records))
    return EvidenceReport(summary,(),undefined,dict(relevance_space=relevance_space,split='validation'))


def paired_component_interval(before, after, component_ids, *, seed, samples, confidence_level):
    positive(samples,integer=True)
    if not 0 < confidence_level < 1 or len(before) != len(after) or len(after) != len(component_ids) or not component_ids:
        raise ValueError('invalid paired component bootstrap inputs')
    delta = np.asarray(after,dtype=float)-np.asarray(before,dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError('nonfinite bootstrap observations')
    # Components receive equal weight; captions/actions do not create new independent units.
    units = np.array([delta[np.array(component_ids) == c].mean() for c in sorted(set(component_ids))])
    out = dict(component_count=len(units), mean_delta=float(units.mean()), defined=len(units) >= 2)
    if len(units) < 2:
        return dict(out,reason='fewer than two independent source components')
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for i in range(samples):
        boot[i] = units[rng.integers(len(units),size=len(units))].mean()
    alpha = (1-confidence_level)/2
    return dict(out,lower=float(np.quantile(boot,alpha)),upper=float(np.quantile(boot,1-alpha)))
