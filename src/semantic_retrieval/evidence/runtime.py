"""Typed evidence consumer of the shared Scheme A worker lifecycle."""
from dataclasses import dataclass, fields
from pathlib import Path

from semantic_retrieval.evidence.config import EvidenceSystemConfig, evidence_manifest_config
from semantic_retrieval.evidence.types import ObservationState, ResourceCounts


@dataclass(frozen=True)
class EpisodeResult:
    ranking: tuple[str,...]
    action_ids: tuple[str,...]
    final_state: ObservationState
    costs: ResourceCounts
    timings: dict[str,float]
    initial_ranking: tuple[str,...] = ()


def validate_evidence_binding(repository,manifest,system_config,runtime):
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    from semantic_retrieval.evidence.data import audit_component_roles, read_split_binding
    cfg = system_config
    if not isinstance(cfg,EvidenceSystemConfig) or cfg.runtime != runtime or manifest['config'] != evidence_manifest_config(cfg,eval_split=runtime.eval_split):
        raise ValueError('effective evidence configuration mismatch')
    refs = manifest_input_references(repository,manifest)
    expected = [(cfg.model.base.clip_weights,cfg.model.base.clip_sha256)]
    expected += [(s.manifest.path,s.manifest.sha256) for s in cfg.splits]
    expected += [(v.path,v.sha256) for f in fields(cfg.parents) if (v := getattr(cfg.parents,f.name)) is not None]
    for item in (cfg.index.reference,cfg.transitions.reference,cfg.resume.checkpoint if cfg.resume else None):
        if item is not None:
            expected.append((item.path,item.sha256))
    if any(refs.get(str(p)) != h for p,h in expected):
        raise ValueError('evidence input/parent is not reserved')
    if not cfg.synthetic:
        episodes = tuple(e for s in cfg.splits for e in read_split_binding(s))
        audit_component_roles(episodes)
        for wrapped in episodes:
            if not wrapped.episode.query_source_ids or not wrapped.episode.query_component_id:
                raise ValueError("scientific roles require explicit query source and connected component identities")
            for p,h in zip(wrapped.episode.video_paths,wrapped.episode.content_sha256):
                if refs.get(str(p)) != h:
                    raise ValueError('every consumed video must be an individually hash-bound input')
        from semantic_retrieval.evidence.execution import validate_parent_lineage
        validate_parent_lineage(repository,manifest,cfg)


def prepare_evidence_run(config, *, repo_root):
    from semantic_retrieval.runtime.artifacts import prepare_run
    if not isinstance(config,EvidenceSystemConfig) or config.runtime is None:
        raise ValueError('evidence worker requires explicit bound runtime')
    return prepare_run(config.runtime,repo_root=repo_root,system_config=config)


def run_evidence_stage(context,config):
    if context.system_config != config:
        raise ValueError('worker context/config mismatch')
    if config.stage in {'cpu_fixture','gpu_qualification'}:
        from semantic_retrieval.evidence.qualification import run_cpu_evidence_qualification, run_gpu_evidence_qualification
        return (run_cpu_evidence_qualification(context,config) if config.stage == 'cpu_fixture'
                else run_gpu_evidence_qualification(context,config))
    from semantic_retrieval.evidence.execution import execute_scientific_stage
    return execute_scientific_stage(context,config)


def run_policy_episode(context,state,models,frame_source,*,journal):
    import time
    import torch
    from dataclasses import replace
    from semantic_retrieval.evidence.evaluation import rank_candidates
    from semantic_retrieval.evidence.policy import acquire, apply_outcome, decide
    from semantic_retrieval.evidence.types import StopDecision
    from semantic_retrieval.evidence.value import build_value_features
    from semantic_retrieval.evidence.views import propose_actions
    started = time.monotonic()
    state = journal.recover(state)
    first_state,history_ids = journal.episode_history(state.query_id)
    action_ids = list(history_ids)
    initial_ranking = ()
    if first_state is not None:
        first_scores = journal.prediction_for(first_state,expected_sampling=models.sampling_sha256)
        initial_ranking = rank_candidates(first_scores.ranking_log_probability,tuple(i for _,i in first_scores.pair_ids))
    while True:
        scores = journal.prediction_for(state,expected_sampling=models.sampling_sha256)
        if scores is None:
            upper = models.prediction_counts(state)
            reserved = state.budget.reserve(upper)  # Must succeed before computing anything.
            journal.prepare_prediction(state,upper)
            scores,counts = models.predict(state)
            if not counts.fits(upper):
                raise ValueError('prediction actual counts exceed reserved upper bound')
            metered = replace(state,budget=reserved.settle(upper,counts),version=state.version+1)
            journal.record_prediction(state,metered,scores,counts)
            state = metered
        device = getattr(models,'device',scores.loc.device)
        scores = replace(scores,**{k:getattr(scores,k).to(device) for k in ('loc','scale','features','ranking_log_probability')})
        actions = propose_actions(state)
        with torch.no_grad():
            if hasattr(models,'policy'):
                decision = models.policy(state,scores,value_model=models.value_head)
            else:
                values = models.value_head(build_value_features(state,actions,scores))
                decision = decide(state,actions,values,state.budget,stop_threshold=models.stop_threshold)
        ranking = rank_candidates(scores.ranking_log_probability,tuple(i for _,i in scores.pair_ids))
        if not initial_ranking:
            initial_ranking = ranking
        if isinstance(decision,StopDecision):
            break
        journal.prepare(state,decision)
        outcome = acquire(decision.action,state,frame_source)
        journal.record_execution(decision.action.action_id,outcome)
        state = apply_outcome(state,decision.action,outcome)
        journal.commit(decision.action.action_id,state)
        action_ids.append(decision.action.action_id)
    return EpisodeResult(ranking,tuple(action_ids),state,state.budget.spent,dict(total_seconds=time.monotonic()-started),initial_ranking)
