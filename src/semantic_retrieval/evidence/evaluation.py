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


def prediction_evidence(*,store,population,selected,rich,before,after,components,seed,mean_population=None,mean_selected=None):
    """Publish numeric observations externally with explicit probability/label scopes."""
    metrics,undefined,refs = {},{},[]
    def save(name,value):
        refs.append(store.write(name,value.detach().cpu() if isinstance(value,torch.Tensor) else np.asarray(value)))
    mean_batches = dict(population=mean_population,selected=mean_selected)
    for scope,batch in (('population',population),('selected',selected)):
        mean_batch = mean_batches[scope]
        if batch is not None and mean_batch is not None:
            raise ValueError('ambiguous classification/regression evidence scope')
        if mean_batch is not None:
            gain,targets = mean_batch
            if gain.shape != (len(targets.valid_mask),) or not torch.isfinite(gain).all():
                raise ValueError('invalid finite mean-gain predictions')
            valid = targets.valid_mask
            metrics[scope+'_valid_count'] = float(valid.sum())
            metrics[scope+'_unknown_count'] = float((~valid).sum())
            metrics[scope+'_valid_coverage'] = float(valid.sum())/len(valid) if len(valid) else 0.
            save(scope+'_mean_gain',gain);save(scope+'_class',targets.outcome_class);save(scope+'_valid',valid)
            undefined[scope+'_categorical_scores'] = 'mean regression does not provide categorical probabilities'
            if valid.any():
                value = (gain[valid].double()-(targets.outcome_class[valid].to(gain).double()-1)).square().mean()
                if not torch.isfinite(value):
                    raise ValueError('nonfinite mean-gain MSE')
                metrics[scope+'_gain_mse'] = float(value)
            else:
                undefined[scope+'_gain_mse'] = 'no known action outcomes'
            continue
        if batch is None:
            undefined[scope+'_outcomes'] = 'no bound outcome predictions/labels for this scope'
            continue
        logits,targets = batch
        values = score_outcomes(logits,targets)
        metrics.update({scope+'_'+k:v for k,v in values.items()})
        save(scope+'_logits',logits);save(scope+'_class',targets.outcome_class);save(scope+'_valid',targets.valid_mask)
        count = len(targets.valid_mask)
        metrics[scope+'_valid_coverage'] = float(targets.valid_mask.sum())/count if count else 0.
        valid = targets.valid_mask
        if valid.any():
            probability = logits[valid].double().softmax(-1)
            gain = probability[:,2]-probability[:,0]
            metrics[scope+'_gain_mse'] = float((gain-(targets.outcome_class[valid].to(gain)-1)).square().mean())
            for category,name in ((0,'worsen'),(2,'correct')):
                p = probability[:,category];y = (targets.outcome_class[valid] == category).to(p)
                error = p.new_zeros(())
                # Fixed equal-width diagnostic bins, never fit on evaluation labels.
                for i in range(10):
                    mask = (p >= i/10)&((p <= 1.) if i == 9 else (p < (i+1)/10))
                    if mask.any():
                        error += mask.double().mean()*(p[mask].mean()-y[mask].mean()).abs()
                metrics[scope+'_'+name+'_ece10'] = float(error)
        else:
            undefined[scope+'_proper_scores'] = 'no known action outcomes'
            undefined[scope+'_gain_mse'] = 'no known action outcomes'
    if rich is None:
        undefined['rich_scores'] = 'no bound frozen rich-score reference'
    else:
        loc,scale,target,valid = rich
        if loc.shape != scale.shape or target.shape != loc.shape or valid.shape != loc.shape or valid.dtype != torch.bool or not all(torch.isfinite(x).all() for x in (loc,scale,target)) or not (scale > 0).all():
            raise ValueError('invalid rich-score evidence')
        save('rich_loc',loc);save('rich_scale',scale);save('rich_target',target);save('rich_valid',valid)
        metrics['rich_valid_count'] = float(valid.sum());metrics['rich_unknown_count'] = float((~valid).sum())
        if valid.any():
            error = (target[valid].double()-loc[valid].double())/scale[valid].double()
            metrics['rich_gaussian_nll'] = float((.5*error.square()+scale[valid].double().log()+.5*math.log(2*math.pi)).mean())
            metrics['rich_interval_coverage_95'] = float((error.abs() <= 1.959963984540054).double().mean())
            metrics['rich_interval_width_95'] = float((2*1.959963984540054*scale[valid].double()).mean())
        else:
            undefined['rich_proper_scores'] = 'no valid rich-score targets'
    if before:
        interval = paired_component_interval(before,after,components,seed=seed,samples=2000,confidence_level=.95)
        metrics.update({'paired_'+k:float(v) for k,v in interval.items() if k not in {'defined','reason'}})
        save('paired_initial',before);save('paired_final',after)
        # Stable numeric grouping metadata without embedding per-query arrays in JSON.
        mapping = {key:i for i,key in enumerate(sorted(set(components)))}
        save('paired_component',[mapping[key] for key in components])
        if not interval['defined']:
            undefined['paired_interval'] = interval['reason']
    else:
        undefined['paired_interval'] = 'no episodes with both initial and final known labels'
    return EvidenceReport(metrics,tuple(refs),undefined,dict(relevance='explicit source role',bootstrap_unit='query source component',rich_scope='fixed-reference scalar score',outcome_scopes=['scripted population','policy-selected']))
