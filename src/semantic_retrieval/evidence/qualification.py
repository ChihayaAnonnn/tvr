"""Archived software qualification with real CLIP, video I/O and exact replay."""
import copy
import os
import time
from dataclasses import asdict,replace
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from experiment_artifacts import load_validated_run_lifecycle
from semantic_retrieval.config import config_payload
from semantic_retrieval.evidence.baselines import build_acquisition_baseline
from semantic_retrieval.evidence.checkpoint import EvidenceProgress,evidence_config_sha256,load_evidence_checkpoint,save_evidence_checkpoint
from semantic_retrieval.evidence.execution import EpisodeAdapter,VideoFrameSource,initial_episode_state
from semantic_retrieval.evidence.frame_cache import FrameFeatureCache,publish_json
from semantic_retrieval.evidence.journal import ActionJournal
from semantic_retrieval.evidence.matcher import mc_log_probabilities
from semantic_retrieval.evidence.model import EvidenceMatcherModel
from semantic_retrieval.evidence.objectives import value_nll,mean_gain_mse
from semantic_retrieval.evidence.policy import acquire,apply_outcome,reserve_action
from semantic_retrieval.evidence.runtime import run_policy_episode
from semantic_retrieval.evidence.sampling import SamplingState
from semantic_retrieval.evidence.training import EvidenceStepper,build_stage_optimizer,deterministic_contrastive,freeze_reference,matcher_loss,rich_score_targets
from semantic_retrieval.evidence.transitions import FrozenTransitionModel,TransitionProducerIdentity,compile_transitions,label_transition,read_transitions
from semantic_retrieval.evidence.types import EvidenceTargets,ResourceCounts,digest
from semantic_retrieval.evidence.value import EvidenceValueModel,ValueHead,build_value_head,build_value_features
from semantic_retrieval.evidence.views import propose_actions
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.types import QueryInputs


def _require(condition,message):
    if not condition:
        raise ValueError('qualification failed: '+message)
    return True


def _videos(context,cfg,rank):
    """Deterministic generated input recipe, with every materialized byte hashed."""
    from semantic_retrieval.data.episodes import file_sha256
    root = context.paths.provenance/f'synthetic-rank-{rank}'
    root.mkdir(exist_ok=False)
    videos = {}
    for j in range(3):
        path = root/f'candidate-{j}.avi'
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(32,32))
        _require(writer.isOpened(),'synthetic video writer')
        try:
            for i in range(cfg.views.pool_size):
                yy,xx = np.mgrid[:32,:32]
                image = np.stack(((xx*7+i*17+j*31)%256,(yy*9+i*11+j*53)%256,((xx+yy)*3+i*23+j*71)%256),axis=-1).astype(np.uint8)
                writer.write(image)
        finally:
            writer.release()
        videos[f'r{rank}-c{j}'] = (path,file_sha256(path))
    return videos


def _wrap(model,device):
    return DistributedDataParallel(model,device_ids=[device.index] if device.type == 'cuda' else None,find_unused_parameters=True) if dist.is_initialized() else model


def _grad(module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum() > 0 for g in gradients)


def _episode_payload(result,model_hash):
    return dict(model_sha256=model_hash,action_ids=result.action_ids,cost_counts=asdict(result.costs),initial_ranking=result.initial_ranking,ranking=result.ranking,state_sha256=result.final_state.state_sha256)


def _qualify_variants(model,observed,sampling):
    results = {}
    for variant in ('gaussian','deterministic','fixed_scale','learned_temperature','standard_probabilistic'):
        probe = copy.deepcopy(model)
        probe.config = replace(probe.config,variant=variant)
        probe.matcher.base.requires_grad_(True)
        probe.matcher.scale_head.requires_grad_(variant in {'gaussian','standard_probabilistic'})
        if variant == 'learned_temperature':
            probe.log_temperature = torch.nn.Parameter(torch.zeros((),device=next(probe.parameters()).device))
        prediction = probe(observed,sampling=sampling)
        (-prediction.ranking_log_probability[0]).backward()
        results[variant] = _require(_grad(probe.matcher.base) and torch.isfinite(prediction.ranking_log_probability).all(),'variant '+variant)
        if variant == 'learned_temperature':
            results[variant] &= _require(probe.log_temperature.grad is not None and torch.isfinite(probe.log_temperature.grad) and probe.log_temperature.grad.abs() > 0,'temperature gradient')
        del probe
    return results


def _exercise(context,cfg,device):
    started = time.monotonic()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    manifest,status,_ = load_validated_run_lifecycle(context.paths)
    _require(status['state'] == 'running' and manifest == context.manifest,'running protected identity')
    torch.manual_seed(cfg.train.seed)
    model = EvidenceMatcherModel(cfg.model).to(device)
    videos = _videos(context,cfg,rank)
    source = VideoFrameSource(videos,cfg.model.base.clip.image_size,cfg.views.preprocessing_sha256)
    from experiment_artifacts import ArtifactArrayStore
    local_store = ArtifactArrayStore(context.paths.root/f'provenance/rank-{rank}/arrays')
    cache = FrameFeatureCache(local_store)
    ids = tuple(videos)
    episode = SimpleNamespace(query_id=f'r{rank}-q0',canonical_ids=ids,content_sha256=tuple(videos[k][1] for k in ids),frame_counts=(cfg.views.pool_size,)*3)
    state = initial_episode_state(episode,cfg,source,cache=cache,model=model)
    length = cfg.model.base.clip.context_length
    # Fixture tokens are explicitly synthetic, and always valid under the bound vocabulary.
    tokens = torch.zeros((1,length),dtype=torch.long);tokens[0,:3] = torch.tensor([cfg.model.base.clip.vocab_size-2,2,cfg.model.base.clip.vocab_size-1])
    query = QueryInputs(tokens,tokens != 0)
    sampling = SamplingState(cfg.sampling,mode='train',model_sha256=model_sha256(model))
    adapter = EpisodeAdapter(model,query,source,cache,sampling)
    observed = adapter.features(state)
    rich_state = replace(state,candidates=tuple(replace(c,observed_frame_ids=c.pool_frame_ids,attempted_frame_ids=c.pool_frame_ids) for c in state.candidates))
    rich = adapter.features(rich_state)
    checks = {'real_clip_forward':_require(observed.features.frame_tokens.shape[-1] == cfg.model.base.dim,'real CLIP dimension')}
    # Changing hidden evidence cannot change a student's observation-only encoder result.
    model.eval()
    with torch.no_grad():
        before = model.point_scores(observed).logits
        changed_rich = replace(rich,features=replace(rich.features,frame_tokens=rich.features.frame_tokens+100))
        _ = model.point_scores(changed_rich)
        checks['hidden_evidence_isolated'] = _require(torch.equal(before,model.point_scores(observed).logits),'hidden evidence isolation')
    labels = torch.tensor([-1,-1,-1] if world > 1 and rank == 0 else [1,0,0],device=device)
    # Reference: train only temporal and deterministic pair matcher, then freeze a distinct teacher.
    model.stage = 'reference'
    ref_train = replace(cfg.train,trainable_prefixes=('temporal.','matcher.base.'),loss_weights={'pc':1.,'score':0.,'value':0.})
    opt,sched = build_stage_optimizer(model,ref_train)
    wrapped = _wrap(model,device)
    ref_stepper = EvidenceStepper(wrapped,opt,sched,weights=ref_train.loss_weights)
    prediction = wrapped(rich,sampling=sampling)
    ref_stepper.step({'pc':deterministic_contrastive(prediction.loc,rich.query_offsets,labels,temperature=cfg.sampling.temperature)})
    del wrapped
    teacher = freeze_reference(copy.deepcopy(model))
    teacher_hash = model_sha256(teacher)
    model.stage = 'matcher'
    train = replace(cfg.train,trainable_prefixes=('temporal.','matcher.'),loss_weights={'pc':1.,'score':1.,'value':0.})
    opt,sched = build_stage_optimizer(model,train)
    wrapped = _wrap(model,device)
    stepper = EvidenceStepper(wrapped,opt,sched,weights=train.loss_weights)
    scores,valid = rich_score_targets(teacher,rich,current_observed=observed)
    if world > 1 and rank == 0:
        valid = torch.zeros_like(valid)
    targets = EvidenceTargets(labels,scores,valid,teacher_hash)
    def step():
        rng = sampling.state_dict()
        predicted = wrapped(observed,sampling=sampling)
        after = sampling.state_dict();sampling.load_state_dict(rng)
        noise = sampling.noise(predicted.pair_ids,device=device,dtype=torch.float32);sampling.load_state_dict(after)
        terms = matcher_loss(predicted,targets,noise=noise,config=replace(cfg,train=train))
        stepper.step(terms)
        return predicted
    prediction = step()
    checks['loc_gradient'] = _require(_grad(model.matcher.base),'loc gradient')
    checks['scale_gradient'] = _require(_grad(model.matcher.scale_head),'scale gradient')
    checks['unknown_rank'] = _require(world == 1 or (rank != 0 or (labels == -1).all()),'unknown-label rank')
    progress = EvidenceProgress(cfg.stage,1,0,1,dict(cursor=1,order_seed=cfg.train.seed),sampling.state_dict(),teacher_hash,None,None)
    path = save_evidence_checkpoint(context,{'matcher':model},opt,sched,cfg,progress)
    step(); continuous_hash = model_sha256(model); next_noise = sampling.noise(prediction.pair_ids,device=device,dtype=torch.float32)
    restored = load_evidence_checkpoint(path,{'matcher':model},expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)
    opt.load_state_dict(restored.optimizer_state);sched.load_state_dict(restored.scheduler_state);sampling.load_state_dict(restored.progress.mc_state);stepper.steps = restored.progress.step
    step(); restored_noise = sampling.noise(prediction.pair_ids,device=device,dtype=torch.float32)
    checks['exact_resume'] = _require(model_sha256(model) == continuous_hash and torch.equal(next_noise,restored_noise),'exact optimizer/RNG/MC resume')
    checks['teacher_frozen'] = _require(teacher_hash == model_sha256(teacher) and all(p.grad is None for p in teacher.parameters()),'frozen teacher')
    with torch.no_grad():
        noise = SamplingState(cfg.sampling,mode='eval',model_sha256=continuous_hash).noise(prediction.pair_ids,device=device,dtype=torch.float32)
        p1 = mc_log_probabilities(prediction.loc,prediction.scale,observed.query_offsets,noise,temperature=cfg.sampling.temperature)
        p2 = mc_log_probabilities(prediction.loc,prediction.scale*7,observed.query_offsets,noise,temperature=cfg.sampling.temperature)
        checks['scale_affects_prediction'] = _require(not torch.allclose(p1,p2,rtol=1e-5,atol=1e-7),'scale participates in ranking probabilities')
    del wrapped,teacher
    freeze_reference(model)
    eval_sampling = SamplingState(cfg.sampling,mode='eval',model_sha256=continuous_hash)
    adapter = EpisodeAdapter(model,query,source,cache,eval_sampling)
    # Each rank writes its own transition artifacts; avoid collisions in the shared run namespace.
    producer = TransitionProducerIdentity(continuous_hash,evidence_config_sha256(cfg),digest('synthetic-index'),digest('synthetic-labels'),digest(config_payload(cfg.views)),digest(config_payload(cfg.sampling)),digest('initial-states'),digest('observable-action-v1'))
    truth = {state.query_id:(dict(zip(ids,(1,0,0))),(f'rank-{rank}-component',))}
    refs = compile_transitions((state,),propose_actions,FrozenTransitionModel(model,adapter.predict,adapter.observe,build_value_features),truth,store=local_store,producer_identity=producer)
    transitions = read_transitions(refs,expected_producer=producer)
    head = ValueHead(model.matcher.feature_dim+19,cfg.value.hidden_dim).to(device)
    value_model = EvidenceValueModel(model,head)
    value_train = replace(cfg.train,trainable_prefixes=('value_head.',),loss_weights={'pc':0.,'score':0.,'value':1.})
    vopt,vsched = build_stage_optimizer(value_model,value_train)
    vwrapped = _wrap(value_model,device)
    vtargets = replace(transitions.targets,outcome_class=transitions.targets.outcome_class.to(device),valid_mask=transitions.targets.valid_mask.to(device))
    if world > 1 and rank == 0:
        vtargets = replace(vtargets,valid_mask=torch.zeros_like(vtargets.valid_mask))
    output = vwrapped(transitions.features.to(device))
    EvidenceStepper(vwrapped,vopt,vsched,weights=value_train.loss_weights).step({'value':value_nll(output.logits,vtargets)})
    checks['value_gradient'] = _require(_grad(head),'value gradient')
    value_kinds = {'distribution':checks['value_gradient']}
    for kind in ('geometry','mean_regressor'):
        probe_head = build_value_head(replace(cfg.value,kind=kind),model.matcher.feature_dim+19).to(device)
        probe_model = EvidenceValueModel(model,probe_head)
        probe_opt,probe_sched = build_stage_optimizer(probe_model,value_train)
        probe_wrapped = _wrap(probe_model,device)
        probe_output = probe_wrapped(transitions.features.to(device))
        term = mean_gain_mse(probe_output,vtargets) if kind == 'mean_regressor' else value_nll(probe_output.logits,vtargets)
        EvidenceStepper(probe_wrapped,probe_opt,probe_sched,weights=value_train.loss_weights).step({'value':term})
        value_kinds[kind] = _require(_grad(probe_head),'value kind '+kind)
        hashes = [None]*world
        if dist.is_initialized():
            dist.all_gather_object(hashes,model_sha256(probe_head))
        else:
            hashes = [model_sha256(probe_head)]
        _require(len(set(hashes)) == 1,'value kind rank equality '+kind)
        del probe_wrapped,probe_model,probe_head,probe_opt,probe_sched

    checks['three_classes'] = _require(tuple(label_transition(a,b,{'positive':1,'negative':0},producer_sha256=producer.sha256).outcome_class.item() for a,b in [('positive','negative'),('positive','positive'),('negative','positive')]) == (0,1,2),'three-class labels')
    head.eval();model.eval()
    policy = build_acquisition_baseline('random',seed=cfg.sampling.seed,config=cfg)
    facade = SimpleNamespace(device=device,prediction_counts=adapter.prediction_counts,predict=lambda s:(adapter.predict(s),adapter.prediction_counts(s)),sampling_sha256=eval_sampling.identity,value_head=head,policy=policy,stop_threshold=cfg.policy.stop_threshold)
    # Continuous run, then repeated completed replay: neither frames nor model computation may run twice.
    journal = ActionJournal(local_store.run_root/'provenance/evidence-actions',local_store,query_id=state.query_id)
    continuous = run_policy_episode(context,state,facade,adapter.acquire,journal=journal)
    resumed = run_policy_episode(context,state,facade,lambda *a:(_ for _ in ()).throw(AssertionError('repeated I/O')),journal=journal)
    checks['action_recovery'] = _require(continuous.final_state == resumed.final_state and continuous.action_ids == resumed.action_ids and continuous.initial_ranking == resumed.initial_ranking,'action/prediction recovery')
    # Separate query: simulate interruption after physical execution, before the commit record.
    interrupted = replace(state,query_id=state.query_id+'-interrupted')
    journal = ActionJournal(local_store.run_root/'provenance/evidence-actions',local_store,query_id=interrupted.query_id)
    action = propose_actions(interrupted)[0]
    journal.prepare(interrupted,reserve_action(interrupted,action,predicted_counts=action.reservation))
    outcome = acquire(action,interrupted,adapter.acquire);journal.record_execution(action.action_id,outcome)
    checks['action_recovery'] &= _require(journal.recover(interrupted) == apply_outcome(interrupted,action,outcome),'execution-before-commit recovery')
    checks['finite_fp32'] = _require(all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters()) and all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in head.parameters()),'finite FP32')
    hashes = [None]*world
    local_hash = (model_sha256(model),model_sha256(head))
    if dist.is_initialized():
        dist.all_gather_object(hashes,local_hash)
    else:
        hashes = [local_hash]
    checks['rank_weights_equal'] = _require(all(h == local_hash for h in hashes),'equal rank parameters')
    variants = _qualify_variants(model,observed,eval_sampling)
    gpu_uuid = None
    if device.type == 'cuda':
        import subprocess
        physical = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[device.index]
        gpu_uuid = subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','--id='+physical],capture_output=True,text=True,check=True).stdout.strip()
    gathered = [None]*world
    local = dict(checks=checks,variants=variants,value_kinds=value_kinds,gpu_uuid=gpu_uuid,continuous=_episode_payload(continuous,continuous_hash),resumed=_episode_payload(resumed,continuous_hash),video_sha256={k:v[1] for k,v in videos.items()},seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
    if dist.is_initialized():
        dist.all_gather_object(gathered,local)
    else:
        gathered = [local]
    if device.type == 'cuda':
        _require(len({x['gpu_uuid'] for x in gathered}) == 4,'four distinct physical GPUs')
    report = dict(schema_version='evidence-qualification-v1',world_size=world,device_type=device.type,production_capacity=not cfg.model.base.clip.fixture,
                  source_sha=context.config.source_sha,manifest_sha256=context.config.manifest_sha256,
                  scientific_authorized=False,test_authorized=False,checks={k:all(x['checks'][k] for x in gathered) for k in checks},
                  qualified_variants=[v for v in variants if all(x['variants'][v] for x in gathered)],
                  qualified_value_kinds=[v for v in value_kinds if all(x['value_kinds'][v] for x in gathered)],
                  continuous=gathered[0]['continuous'],resumed=gathered[0]['resumed'],ranks=gathered,
                  summary_metrics={'qualification_passed':1.,'elapsed_seconds':max(x['seconds'] for x in gathered),'peak_cuda_bytes':float(max(x['peak_cuda_bytes'] for x in gathered))})
    if rank == 0:
        publish_json(context.paths.provenance/'evidence-qualification.json',report)
    return report


def run_cpu_evidence_qualification(context,config,*,interrupt_at=None):
    if interrupt_at is not None:
        raise ValueError('interruptions are verified internally at checkpoint and execution boundaries')
    _require(config.stage == 'cpu_fixture' and config.synthetic and config.model.base.clip.fixture and not torch.cuda.is_available(),'tiny CPU-only qualification')
    return _exercise(context,config,torch.device('cpu'))


def run_gpu_evidence_qualification(context,config):
    _require(config.stage == 'gpu_qualification' and config.synthetic and not config.model.base.clip.fixture and dist.is_initialized() and dist.get_world_size() == 4 and torch.cuda.is_available(),'production four-rank CUDA qualification')
    return _exercise(context,config,torch.device('cuda',int(os.environ['LOCAL_RANK'])))
