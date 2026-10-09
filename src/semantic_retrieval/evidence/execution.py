"""Real video adapters and protected scientific stage execution."""
import hashlib
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from experiment_artifacts import ArtifactArrayStore, ArrayReference, load_validated_run_lifecycle, make_run_paths, parse_registry_text, strict_json_load
from semantic_retrieval.config import config_payload
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.data.video import decode_video
from semantic_retrieval.evidence.calibration import ValueCalibration, fit_value_temperature
from semantic_retrieval.evidence.checkpoint import EvidenceProgress, evidence_config_sha256, load_evidence_checkpoint, save_evidence_checkpoint
from semantic_retrieval.evidence.config import InputReference
from semantic_retrieval.evidence.data import read_split_binding
from semantic_retrieval.evidence.encoding import pack_observations
from semantic_retrieval.evidence.evaluation import EvidenceEvaluationRecord, evaluate_evidence, rank_candidates
from semantic_retrieval.evidence.frame_cache import FrameCacheIdentity, FrameFeatureCache, publish_json, safe_path
from semantic_retrieval.evidence.model import EvidenceMatcherModel, load_reference_parts
from semantic_retrieval.evidence.objectives import value_nll, mean_gain_mse
from semantic_retrieval.evidence.policy import acquire, apply_outcome
from semantic_retrieval.evidence.sampling import SamplingState
from semantic_retrieval.evidence.training import EvidenceStepper, build_stage_optimizer, deterministic_contrastive, freeze_reference, matcher_loss, rich_score_targets
from semantic_retrieval.evidence.transitions import FrozenTransitionModel, TransitionProducerIdentity, TransitionReferences, compile_transitions, read_transitions, sample_states
from semantic_retrieval.evidence.types import ActionOutcome, BudgetLedger, CandidateObservation, EvidenceTargets, ExecutionProfile, ObservationState, ResourceCounts, ZERO, digest
from semantic_retrieval.evidence.value import ValueHead, build_value_head, build_value_features
from semantic_retrieval.evidence.views import build_frame_pool, propose_actions, view_frame_ids
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.input_bindings import context_input_references
from semantic_retrieval.types import QueryFeatures, QueryInputs


class DecodeFailure(ValueError):
    """A requested frame could not be decoded; identity failures are separate."""


class VideoFrameSource:
    def __init__(self, videos, image_size, preprocessing_sha256):
        self.videos, self.image_size = videos, image_size
        self.preprocessing_sha256 = preprocessing_sha256

    def __call__(self,candidate,frame_ids):
        if candidate.canonical_id not in self.videos:
            raise ValueError('video absent from bound source inventory')
        path,expected = self.videos[candidate.canonical_id]
        safe_path(path)
        if expected != candidate.content_sha256 or file_sha256(path) != expected:
            raise ValueError('video source identity mismatch')
        try:
            frames,mask = decode_video(path,frame_ids,self.image_size)
        except ValueError as exc:
            if str(exc).startswith(('video has no decodable requested frames:', 'cannot open video:')):
                raise DecodeFailure(str(exc)) from exc
            raise
        if not mask.all():
            raise DecodeFailure('partial frame decoding; request individual frames')
        return frames


class EpisodeAdapter:
    def __init__(self,model,query,source,cache,sampling):
        self.model,self.query,self.source,self.cache,self.sampling = model,query,source,cache,sampling
        self.clip_identity = digest(config_payload(model.config.base)['clip'])

    def features(self,state):
        return self.model.encode_inputs(self.query,(state,),self.source,self.cache)

    def predict(self,state):
        with torch.no_grad():
            return self.model(self.features(state),sampling=self.sampling)

    def prediction_counts(self,state):
        m = 1 if self.model.config.variant in {'deterministic','learned_temperature'} else self.sampling.config.eval_samples
        return ResourceCounts(0,0,sum(len(c.observed_frame_ids) for c in state.candidates),m*len(state.candidates))

    def acquire(self,action,state):
        started = time.monotonic()
        c = next(c for c in state.candidates if c.canonical_id == action.candidate_id)
        succeeded,failed,refs = [],[],[]
        device = next(self.model.parameters()).device
        identity = FrameCacheIdentity(c.content_sha256,self.model.config.base.clip_sha256,self.clip_identity,self.source.preprocessing_sha256)
        for i in action.frame_ids:
            try:
                image = self.source(c,(i,)).to(device)
            except DecodeFailure:
                failed.append(i); continue
            with torch.no_grad():
                feature = self.model.clip.encode_frames(image[None],torch.ones((1,1),dtype=torch.bool,device=device))[0,0]
            refs.append(self.cache.write(c.canonical_id,i,feature,identity=identity)); succeeded.append(i)
        return ActionOutcome(action.action_id,tuple(succeeded),tuple(failed),ResourceCounts(len(action.frame_ids),len(succeeded),0,0),time.monotonic()-started,tuple(refs))

    def observe(self,state,action):
        outcome = acquire(action,state,self.acquire)
        updated = apply_outcome(state,action,outcome)
        counts = self.prediction_counts(updated)
        return replace(updated,budget=updated.budget.reserve(counts).settle(counts,counts))


def validate_parent_lineage(repository,manifest,cfg):
    import subprocess
    registry = repository/manifest['registry_path']
    text = subprocess.run(['git','show','HEAD:'+manifest['registry_path']],cwd=repository,check=True,capture_output=True,text=True).stdout
    records = parse_registry_text(text)
    parents = manifest['parent_run_ids']
    for reference in [getattr(cfg.parents,k) for k in ('reference','matcher','value','calibration')]+[cfg.transitions.reference,cfg.index.reference,cfg.resume.checkpoint if cfg.resume else None]:
        if reference is None:
            continue
        matched = False
        for run_id in parents:
            terminal = [r for r in records if r['run_id'] == run_id and r['event'] in {'completed','failed','cancelled'}]
            if len(terminal) != 1:
                continue
            record = terminal[0]
            root = Path(record['artifact_path'])
            if not reference.path.is_relative_to(root):
                continue
            pm,status,_ = load_validated_run_lifecycle(make_run_paths(root.parents[3],manifest['study'],run_id))
            if status['state'] != 'completed' and not (cfg.resume and reference == cfg.resume.checkpoint and status['state'] in {'failed','cancelled'}):
                raise ValueError('model/input parent is not completed')
            if status['output_hashes'].get(reference.path.relative_to(root).as_posix()) != reference.sha256:
                raise ValueError('parent input differs from terminal artifact')
            matched = True
        if not matched:
            raise ValueError('every scientific parent must have committed terminal lineage')


def _load_model(reference,models):
    safe_path(reference.path)
    if file_sha256(reference.path) != reference.sha256:
        raise ValueError('parent checkpoint hash mismatch')
    with reference.path.with_suffix('.json').open() as h:
        meta = strict_json_load(h)
    return load_evidence_checkpoint(reference.path,models,expected_config_sha256=meta['config_sha256'],expected_parents=meta['parents'])


def _read_transition_input(cfg):
    ref = cfg.transitions.reference
    if ref is None or file_sha256(ref.path) != ref.sha256:
        raise ValueError('bound transition input required')
    with ref.path.open() as h:
        p = strict_json_load(h)
    producer = TransitionProducerIdentity(**p['producer'])
    if producer.sha256 != cfg.transitions.producer_sha256:
        raise ValueError('transition producer differs from config')
    role = 'V_calibration' if cfg.stage == 'calibration' else 'T_value'
    bindings = [s for s in cfg.splits if s.role == role]
    if not cfg.synthetic:
        if len(bindings) != 1 or producer.label_sha256 != bindings[0].manifest.sha256:
            raise ValueError('transition supervision source role mismatch')
        if cfg.index.reference is None or producer.index_sha256 != cfg.index.reference.sha256 or producer.view_sha256 != digest(config_payload(cfg.views)) or producer.mc_sha256 != digest(config_payload(cfg.sampling)) or producer.feature_schema_sha256 != digest(cfg.value.feature_schema):
            raise ValueError('transition view/index/integration/schema mismatch')
    arrays = tuple(ArrayReference(**{**r,'shape':tuple(r['shape'])}) for r in p['arrays'])
    return read_transitions(TransitionReferences(ref.path,ref.sha256,producer,arrays),expected_producer=producer)


def _query(model,text):
    from semantic_retrieval.data.tokenizer import ClipTokenizer
    ids,mask = ClipTokenizer().encode_batch([text],model.config.base.clip.context_length)
    return QueryInputs(ids,mask)


def initial_episode_state(episode,cfg,source,*,cache=None,model=None):
    candidates = []
    requests = successes = 0
    initial_requests = sum(len(view_frame_ids(build_frame_pool(n,cfg.views.pool_size),cfg.views.levels[0])) for n in episode.frame_counts)
    upper = ResourceCounts(initial_requests,initial_requests,0,0)
    reserved = BudgetLedger(cfg.policy.limits,ZERO,ZERO).reserve(upper)
    for canonical,content,count in zip(episode.canonical_ids,episode.content_sha256,episode.frame_counts):
        pool = build_frame_pool(count,cfg.views.pool_size)
        requested = view_frame_ids(pool,cfg.views.levels[0])
        provisional = CandidateObservation(canonical,content,count,pool,(requested[0],),(requested[0],))
        valid = []
        for i in requested:
            requests += 1
            try:
                image = source(provisional,(i,))
            except DecodeFailure:
                continue
            valid.append(i); successes += 1
            if cache is not None and model is not None:
                device = next(model.parameters()).device
                with torch.no_grad():
                    feature = model.clip.encode_frames(image.to(device)[None],torch.ones((1,1),device=device,dtype=torch.bool))[0,0]
                identity = FrameCacheIdentity(content,cfg.model.base.clip_sha256,digest(config_payload(cfg.model.base)['clip']),cfg.views.preprocessing_sha256)
                cache.write(canonical,i,feature,identity=identity)
        if not valid:
            raise DecodeFailure('candidate has empty initial observation: '+canonical)
        candidates.append(CandidateObservation(canonical,content,count,pool,tuple(valid),requested))
    budget = reserved.settle(upper,ResourceCounts(requests,successes,0,0))
    mc_samples = 1 if cfg.model.variant in {'deterministic','learned_temperature'} else cfg.sampling.eval_samples
    return ObservationState(episode.query_id,cfg.index.gallery_version,tuple(candidates),cfg.views.levels,
                            ExecutionProfile(mc_samples,cfg.policy.cost_weights),budget,0)


def bind_candidate_order(episode,canonical_order):
    reviewed = dict(zip(episode.canonical_ids,episode.relevance))
    if episode.relevance_space == 'benchmark_annotation':
        positive_ids = set(episode.benchmark_positive_ids)
        labels = tuple(int(c in positive_ids) for c in canonical_order)
    else:
        labels = tuple(reviewed.get(c,-1) for c in canonical_order)
    presence = 1 if 1 in labels else (-1 if -1 in labels else 0)
    return labels,presence


def _episodes_and_sources(cfg,role,model,context):
    from semantic_retrieval.evidence.baselines import build_initial_index,search_initial_candidates
    from semantic_retrieval.retrieval.index import IndexIdentity
    selected = [s for s in cfg.splits if s.role == role]
    if len(selected) != 1 or cfg.index.reference is None:
        raise ValueError('one source-role manifest and frozen initial-index specification required')
    wrapped = read_split_binding(selected[0]); videos = {}; inventory = {}
    for w in wrapped:
        e = w.episode
        if e.gallery_version != cfg.index.gallery_version:
            raise ValueError('episode gallery differs from configured fixed index')
        for i,(c,p,h,count,source_id) in enumerate(zip(e.canonical_ids,e.video_paths,e.content_sha256,e.frame_counts,e.source_ids)):
            if c in videos and videos[c] != (p,h):
                raise ValueError('conflicting canonical source inventory')
            videos[c] = (p,h); inventory[c] = (p,h,count,source_id)
    ref = cfg.index.reference
    if file_sha256(ref.path) != ref.sha256:
        raise ValueError('initial index specification hash mismatch')
    with ref.path.open() as handle:
        index_spec = strict_json_load(handle)
    expected_keys = {'schema_version','clip_sha256','preprocessing_sha256','view_sha256','gallery_version','roles'}
    if set(index_spec) != expected_keys or index_spec['schema_version'] != 'evidence-initial-index-v1' or index_spec['clip_sha256'] != cfg.model.base.clip_sha256 or index_spec['preprocessing_sha256'] != cfg.views.preprocessing_sha256 or index_spec['view_sha256'] != digest(config_payload(cfg.views)) or index_spec['gallery_version'] != cfg.index.gallery_version:
        raise ValueError('initial index protocol/encoder binding mismatch')
    if role not in index_spec['roles'] or tuple(index_spec['roles'][role]) != tuple(sorted(inventory)):
        raise ValueError('role gallery must equal the declared label-independent inventory')
    source = VideoFrameSource(videos,cfg.model.base.clip.image_size,cfg.views.preprocessing_sha256)
    cache = FrameFeatureCache(context.array_store)
    candidates = tuple(CandidateObservation(c,h,count,build_frame_pool(count,cfg.views.pool_size),
                      view_frame_ids(build_frame_pool(count,cfg.views.pool_size),cfg.views.levels[0]),
                      view_frame_ids(build_frame_pool(count,cfg.views.pool_size),cfg.views.levels[0]))
                      for c,(p,h,count,source_id) in sorted(inventory.items()))
    gallery_state = ObservationState('__initial_index__',cfg.index.gallery_version,candidates,cfg.views.levels,
                       ExecutionProfile(cfg.sampling.eval_samples,cfg.policy.cost_weights),BudgetLedger(cfg.policy.limits,ZERO,ZERO),0)
    device = next(model.parameters()).device
    def reader(candidate,ids):
        identity = FrameCacheIdentity(candidate.content_sha256,cfg.model.base.clip_sha256,digest(config_payload(cfg.model.base)['clip']),cfg.views.preprocessing_sha256)
        rows = []
        for i in ids:
            path = cache._path(candidate.canonical_id,i,identity)
            if path.exists():
                feature = cache.read(candidate.canonical_id,i,identity=identity).to(device)
            else:
                image = source(candidate,(i,)).to(device)
                with torch.no_grad():
                    feature = model.clip.encode_frames(image[None],torch.ones((1,1),device=device,dtype=torch.bool))[0,0]
                cache.write(candidate.canonical_id,i,feature,identity=identity)
            rows.append(feature)
        return torch.stack(rows)
    index_identity = IndexIdentity(cfg.model.base.clip_sha256,cfg.index.gallery_version,
                                  digest(dict(preprocessing=cfg.views.preprocessing_sha256,view=digest(config_payload(cfg.views)))))
    started = time.monotonic()
    index = build_initial_index((gallery_state,),reader,identity=index_identity)
    result = []
    for w in wrapped:
        e = w.episode
        query = _query(model,e.query_text)
        with torch.no_grad():
            tokens,pooled = model.clip.encode_text(query.input_ids.to(device),query.text_mask.to(device))
        qf = QueryFeatures(tokens,pooled,query.text_mask.to(device))
        order = search_initial_candidates(qf,index,cfg.index.candidate_count,expected_identity=index_identity)[0]
        labels,presence = bind_candidate_order(e,order)
        selected_episode = replace(e,canonical_ids=order,candidate_ids=order,relevance=labels,answer_in_candidates=presence,
                  video_paths=tuple(inventory[c][0] for c in order),content_sha256=tuple(inventory[c][1] for c in order),
                  frame_counts=tuple(inventory[c][2] for c in order),source_ids=tuple(inventory[c][3] for c in order))
        result.append(replace(w,episode=selected_episode))
    context.array_store.write('initial_index_embeddings',index.embeddings.detach().cpu())
    publish_json(context.paths.provenance/('initial-index-'+role+'.json'),dict(schema_version='evidence-initial-index-report-v1',
                 identity=asdict(index_identity),candidate_count=cfg.index.candidate_count,gallery_count=len(index.ids),
                 elapsed_seconds=time.monotonic()-started,role=role,specification_sha256=ref.sha256))
    return tuple(result),{w.episode.query_id:w for w in result},source,cache


def _write_reference_parts(context,model):
    from semantic_retrieval.runtime.checkpoint import _publish_file
    path = context.paths.checkpoints/'reference-parts.pt'
    if path.exists():
        raise ValueError('reference snapshot already exists')
    _publish_file(path,lambda h:torch.save({'clip':model.clip.state_dict(),'temporal':model.temporal.state_dict(),'matcher.base':model.matcher.base.state_dict()},h))
    return path


def _producer(cfg,model,role):
    # Full producer contract includes source role through configuration plus label manifest hash.
    bindings = [s for s in cfg.splits if s.role == role]
    if len(bindings) != 1:
        raise ValueError('transition role not bound')
    return TransitionProducerIdentity(model_sha256(model),evidence_config_sha256(cfg),cfg.index.reference.sha256 if cfg.index.reference else digest(config_payload(cfg.index)),
                                     bindings[0].manifest.sha256,digest(config_payload(cfg.views)),digest(config_payload(cfg.sampling)),
                                     digest(dict(seed=cfg.transitions.state_seed,trajectories=cfg.transitions.trajectories,role=role)),digest('observable-action-v1'))


def _training(context,cfg,model,device):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    model.stage = cfg.stage
    reference = None
    if cfg.stage == 'matcher':
        reference = EvidenceMatcherModel(cfg.model).to(device)
        load_reference_parts(cfg.parents.reference.path,reference,expected_sha256=cfg.parents.reference.sha256)
        load_reference_parts(cfg.parents.reference.path,model,expected_sha256=cfg.parents.reference.sha256)
        freeze_reference(reference)
    optimizer,scheduler = build_stage_optimizer(model,cfg.train)
    sampling = SamplingState(cfg.sampling,mode='train',model_sha256=model_sha256(model))
    start = 0
    if cfg.stage == 'value':
        transition_batch = _read_transition_input(cfg)
        if transition_batch.targets.producer_sha256 != cfg.transitions.producer_sha256:
            raise ValueError('value transition producer mismatch')
        features = transition_batch.features.to(device)
        target = replace(transition_batch.targets,outcome_class=transition_batch.targets.outcome_class.to(device),valid_mask=transition_batch.targets.valid_mask.to(device))
    else:
        episodes,lookup,source,cache = _episodes_and_sources(cfg,'T_match',model,context)
    wrapped_model = DistributedDataParallel(model,device_ids=[device.index] if device.type == 'cuda' else None,find_unused_parameters=True) if world > 1 else model
    stepper = EvidenceStepper(wrapped_model,optimizer,scheduler,weights=cfg.train.loss_weights)
    if cfg.resume:
        restored = load_evidence_checkpoint(cfg.resume.checkpoint.path,{'matcher' if cfg.stage != 'value' else 'value':model.value_head if cfg.stage == 'value' else model},
                                            expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)
        if restored.progress.step > cfg.resume.cumulative_steps:
            raise ValueError('resume cumulative budget identity mismatch')
        optimizer.load_state_dict(restored.optimizer_state); scheduler.load_state_dict(restored.scheduler_state)
        sampling.load_state_dict(restored.progress.mc_state); start = restored.progress.cursor; stepper.steps = restored.progress.step
    summary = {}
    for iteration in range(start,cfg.train.scheduler_steps):
        if cfg.stage == 'value':
            indices = torch.arange(rank,len(features),world,device=device)
            out = wrapped_model(features[indices])
            t = replace(target,outcome_class=target.outcome_class[indices],valid_mask=target.valid_mask[indices])
            terms = {'value':mean_gain_mse(out,t) if cfg.value.kind == 'mean_regressor' else value_nll(out.logits,t)}
        else:
            w = episodes[(iteration*world+rank)%len(episodes)]
            e = w.episode
            state = initial_episode_state(e,cfg,source,cache=cache,model=model)
            query = _query(model,e.query_text)
            level = cfg.views.levels[iteration%len(cfg.views.levels)]
            # Training may read richer views, but the student receives only its explicit observed subset.
            current_candidates = []
            rich_candidates = []
            for c in state.candidates:
                valid = []
                for i in c.pool_frame_ids:
                    try:
                        source(c,(i,))
                    except DecodeFailure:
                        continue
                    valid.append(i)
                current_ids = tuple(i for i in view_frame_ids(c.pool_frame_ids,level) if i in valid)
                if not current_ids:
                    raise DecodeFailure('empty student training view')
                current_candidates.append(replace(c,observed_frame_ids=current_ids,attempted_frame_ids=current_ids))
                rich_candidates.append(replace(c,observed_frame_ids=tuple(valid),attempted_frame_ids=c.pool_frame_ids))
            state = replace(state,candidates=tuple(current_candidates))
            observed = model.encode_inputs(query,(state,),source,cache)
            labels = torch.tensor(e.relevance,device=device,dtype=torch.long)
            if cfg.stage == 'reference':
                # DDP forward must own the trainable graph, including its point-score path.
                pred = wrapped_model(observed,sampling=sampling)
                terms = {'pc':deterministic_contrastive(pred.loc,observed.query_offsets,labels,temperature=cfg.sampling.temperature)}
            else:
                rich_state = replace(state,candidates=tuple(rich_candidates))
                rich = model.encode_inputs(query,(rich_state,),source,cache)
                rich_scores,valid = rich_score_targets(reference,rich,current_observed=observed)
                rng = sampling.state_dict()
                pred = wrapped_model(observed,sampling=sampling)
                after = sampling.state_dict(); sampling.load_state_dict(rng)
                noise = sampling.noise(pred.pair_ids,device=device,dtype=torch.float32); sampling.load_state_dict(after)
                terms = matcher_loss(pred,EvidenceTargets(labels,rich_scores,valid,model_sha256(reference)),noise=noise,config=cfg)
        result = stepper.step(terms)
        summary = {k+'_sum':v for k,v in result.loss_sums.items()}
        if (iteration+1)%cfg.train.checkpoint_interval == 0 or iteration+1 == cfg.train.scheduler_steps:
            progress = EvidenceProgress(cfg.stage,stepper.steps,0,iteration+1,dict(cursor=iteration+1,order_seed=cfg.train.seed),sampling.state_dict(),
                                        cfg.transitions.producer_sha256 or model_sha256(model),None,None)
            save_evidence_checkpoint(context,{'value' if cfg.stage == 'value' else 'matcher':model.value_head if cfg.stage == 'value' else model},optimizer,scheduler,cfg,progress)
    if cfg.stage == 'reference' and rank == 0:
        _write_reference_parts(context,model)
    summary['optimizer_steps'] = float(stepper.steps)
    return dict(summary_metrics=summary)


def execute_scientific_stage(context,cfg):
    from semantic_retrieval.evidence.authorization import assert_evidence_authorized
    manifest,status,_ = load_validated_run_lifecycle(context.paths)
    if manifest != context.manifest or status['state'] != 'running':
        raise ValueError('scientific stage requires a verified running identity')
    assert_evidence_authorized(repo_root=context.repo_root,registry_path=context.config.registry_path,manifest=manifest,operation='start')
    device = torch.device('cuda',int(__import__('os').environ.get('LOCAL_RANK','0'))) if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(cfg.train.seed)
    model = EvidenceMatcherModel(cfg.model).to(device)
    if cfg.stage in {'reference','matcher'}:
        return _training(context,cfg,model,device)
    _load_model(cfg.parents.matcher,{'matcher':model})
    freeze_reference(model)
    head = build_value_head(cfg.value,model.matcher.feature_dim+19).to(device)
    if cfg.stage == 'value':
        from semantic_retrieval.evidence.value import EvidenceValueModel
        return _training(context,cfg,EvidenceValueModel(model,head).to(device),device)
    if cfg.stage == 'transitions':
        role = {'train':'T_value','val_calibration':'V_calibration','val_development':'V_development','val_locked':'V_locked'}.get(context.config.eval_split)
        if role is None:
            raise ValueError('unsupported transition role')
        episodes,lookup,source,cache = _episodes_and_sources(cfg,role,model,context)
        producer = _producer(cfg,model,role)
        initial = []; adapters = {}
        sampling = SamplingState(cfg.sampling,mode='eval',model_sha256=producer.matcher_sha256)
        for w in episodes:
            state = initial_episode_state(w.episode,cfg,source,cache=cache,model=model)
            adapters[state.query_id] = EpisodeAdapter(model,_query(model,w.episode.query_text),source,cache,sampling)
            counts = adapters[state.query_id].prediction_counts(state)
            state = replace(state,budget=state.budget.reserve(counts).settle(counts,counts))
            initial.append(state)
        observe = lambda s,a:adapters[s.query_id].observe(s,a)
        states = tuple(sample_states(initial,seed=cfg.transitions.state_seed,max_steps=cfg.transitions.trajectories,observe=observe))
        adapter = FrozenTransitionModel(model,lambda s:adapters[s.query_id].predict(s),observe,build_value_features)
        truth = {w.episode.query_id:(dict(zip(w.episode.canonical_ids,w.episode.relevance)),(w.episode.query_component_id,)) for w in episodes}
        refs = compile_transitions(states,propose_actions,adapter,truth,store=context.array_store,producer_identity=producer)
        return dict(summary_metrics={'transition_count':float(len(read_transitions(refs,expected_producer=producer).features))})
    from semantic_retrieval.evidence.value import EvidenceValueModel
    value_container = EvidenceValueModel(model,head)
    if cfg.parents.value is not None:
        _load_model(cfg.parents.value,{'value':head})
    if cfg.stage == 'calibration':
        batch = _read_transition_input(cfg)
        identity = dict(producer_sha256=batch.targets.producer_sha256,model_sha256=model_sha256(head),role='V_calibration',
                        relevance_space=next(s.relevance_space for s in cfg.splits if s.role == 'V_calibration'),feature_schema_sha256=digest('observable-action-v1'))
        targets = replace(batch.targets,outcome_class=batch.targets.outcome_class.to(device),valid_mask=batch.targets.valid_mask.to(device))
        calibration = fit_value_temperature(head(batch.features.to(device)).logits,targets,split='V_calibration',identity=identity)
        from semantic_retrieval.evidence.schemas import validate_evidence_document
        payload = asdict(calibration); validate_evidence_document('calibration',payload)
        publish_json(context.paths.provenance/'value-calibration.json',payload)
        return dict(summary_metrics={'calibration_temperature':calibration.temperature,'calibration_count':float(calibration.count)})
    if cfg.stage == 'evaluation':
        return _evaluation(context,cfg,model,head)
    raise ValueError('unsupported scientific evidence stage')


def _evaluation(context,cfg,model,head):
    from types import SimpleNamespace
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.runtime import run_policy_episode
    role = {'val_development':'V_development','val_locked':'V_locked'}.get(context.config.eval_split)
    if role is None:
        raise ValueError('evaluation must explicitly select development or locked validation')
    episodes,lookup,source,cache = _episodes_and_sources(cfg,role,model,context)
    calibration = None
    if cfg.policy.baseline == 'distribution':
        with cfg.parents.calibration.path.open() as h:
            payload = strict_json_load(h)
        from semantic_retrieval.evidence.schemas import validate_evidence_document
        validate_evidence_document('calibration',payload)
        calibration = ValueCalibration(**payload)
        relevance = next(s.relevance_space for s in cfg.splits if s.role == role)
        if calibration.identity['model_sha256'] != model_sha256(head) or calibration.identity['role'] != 'V_calibration' or calibration.identity['relevance_space'] != relevance or calibration.identity['feature_schema_sha256'] != digest(cfg.value.feature_schema):
            raise ValueError('calibration model/role/relevance/schema differs from evaluation')
    sampling = SamplingState(cfg.sampling,mode='eval',model_sha256=model_sha256(model))
    records,components = [],[]
    for w in episodes:
        e = w.episode
        state = initial_episode_state(e,cfg,source,cache=cache,model=model)
        adapter = EpisodeAdapter(model,_query(model,e.query_text),source,cache,sampling)
        def predict(s):
            return adapter.predict(s),adapter.prediction_counts(s)
        def value(features):
            out = head(features)
            return replace(out,logits=calibration.apply(out.logits,expected_identity=calibration.identity)) if calibration is not None else out
        from semantic_retrieval.evidence.baselines import build_acquisition_baseline
        policy = build_acquisition_baseline(cfg.policy.baseline,seed=cfg.sampling.seed,config=cfg)
        models = SimpleNamespace(device=next(model.parameters()).device,predict=predict,prediction_counts=adapter.prediction_counts,sampling_sha256=sampling.identity,value_head=value,policy=policy,stop_threshold=cfg.policy.stop_threshold)
        # Each query has its own cache/journal cost account; cross-query feature reuse grants no free actions.
        journal = ActionJournal(context.array_store.run_root/'provenance'/'evidence-actions',context.array_store)
        result = run_policy_episode(context,state,models,adapter.acquire,journal=journal)
        initial_id = result.initial_ranking[0]
        truth = dict(zip(e.canonical_ids,e.relevance))
        records.append(EvidenceEvaluationRecord(e.query_id,truth[initial_id],truth[result.ranking[0]],e.answer_in_candidates,result.costs,result.timings['total_seconds']))
        components.append(e.query_component_id)
    report = evaluate_evidence(records,relevance_space=next(s.relevance_space for s in cfg.splits if s.role == role),source_components=components)
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    payload = asdict(report); validate_evidence_document('report',payload)
    publish_json(context.paths.provenance/'evidence-report.json',payload)
    return dict(summary_metrics=report.summary_metrics)
