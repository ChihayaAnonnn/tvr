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
from semantic_retrieval.evidence.checkpoint import EvidenceProgress, evidence_config_sha256, load_evidence_checkpoint, load_evidence_model_weights, save_evidence_checkpoint
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
from semantic_retrieval.evidence.timing import measure,synchronize
from semantic_retrieval.evidence.views import build_frame_pool, propose_actions, view_frame_ids
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.input_bindings import context_input_references
from semantic_retrieval.types import QueryFeatures, QueryInputs


class DecodeFailure(ValueError):
    """A requested frame could not be decoded; identity failures are separate."""


class InitialObservationFailure(DecodeFailure):
    """No initial prediction exists; preserve actual work before failing the run."""
    def __init__(self,query_id,candidate_id,actual_counts,*,budget=None):
        super().__init__('candidate has empty initial observation: '+candidate_id)
        self.query_id,self.candidate_id = query_id,candidate_id
        self.actual_counts,self.budget = actual_counts,budget


def _record_input_failure(provenance,failure,query_ids,*,scope):
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    rank = dist.get_rank() if dist.is_initialized() else 0
    payload = dict(schema_version='evidence-input-failure-v1',scope=scope,
        query_id=failure.query_id,candidate_id=failure.candidate_id,
        actual_counts=asdict(failure.actual_counts),budget=asdict(failure.budget) if failure.budget is not None else None,
        affected_query_count=len(query_ids),affected_query_ids_sha256=digest(tuple(sorted(query_ids))),
        evaluation_complete=False,overall_metrics_defined=False)
    validate_evidence_document('input_failure',payload)
    publish_json(provenance/'input-failures'/f'{scope}-rank-{rank}-{digest(failure.query_id)}.json',payload)


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
    def __init__(self,model,query,source,cache,sampling,*,timings=None):
        self.model,self.query,self.source,self.cache,self.sampling = model,query,source,cache,sampling
        self.timings = timings
        self.clip_identity = digest(config_payload(model.config.base)['clip'])

    def features(self,state):
        return self.model.encode_inputs(self.query,(state,),self.source,self.cache)

    def predict(self,state):
        with torch.no_grad():
            with measure(self.timings,'query_cache_seconds',next(self.model.parameters()).device):
                features = self.features(state)
            return self.model(features,sampling=self.sampling,timings=self.timings)

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
                with measure(self.timings,'decode_seconds',device):
                    image = self.source(c,(i,)).to(device)
            except DecodeFailure:
                failed.append(i); continue
            with torch.no_grad(),measure(self.timings,'clip_seconds',device):
                feature = self.model.clip.encode_frames(image[None],torch.ones((1,1),dtype=torch.bool,device=device))[0,0]
            refs.append(self.cache.write(c.canonical_id,i,feature,identity=identity)); succeeded.append(i)
        return ActionOutcome(action.action_id,tuple(succeeded),tuple(failed),ResourceCounts(len(action.frame_ids),len(succeeded),0,0),time.monotonic()-started,tuple(refs))

    def observe(self,state,action):
        outcome = acquire(action,state,self.acquire)
        updated = apply_outcome(state,action,outcome)
        counts = self.prediction_counts(updated)
        return replace(updated,budget=updated.budget.reserve(counts).settle(counts,counts))


def validate_parent_contract(cfg,kind,parent_manifest):
    """Check semantics against the committed producer manifest, not its sidecar claims."""
    from semantic_retrieval.evidence.config import evidence_manifest_config
    parent = parent_manifest['config']; expected = evidence_manifest_config(cfg,eval_split='train')
    pe = parent['evaluation']; ce = expected['evaluation']
    if pe.get('config_kind') != 'evidence-retrieval-v1' or pe.get('evidence_stage') != kind:
        raise ValueError('parent stage/protocol mismatch')
    def model_contract(model):
        model = dict(model); base = dict(model['base']);base.pop('clip_weights')
        model['base'] = base
        return base if kind == 'reference' else model
    if model_contract(parent['model']) != model_contract(expected['model']):
        raise ValueError('parent model/CLIP/variant/floor mismatch')
    if pe['views'] != ce['views'] or pe['splits'] != ce['splits'] or pe['index'] != ce['index']:
        raise ValueError('parent views/source-role/index mismatch')
    if kind != 'reference' and pe['sampling'] != ce['sampling']:
        raise ValueError('parent integration protocol mismatch')
    if kind in {'value','calibration'}:
        if pe['value'] != ce['value'] or pe['parents']['matcher'] != ce['parents']['matcher']:
            raise ValueError('parent value capacity/schema/frozen matcher mismatch')
    if kind == 'calibration' and pe['parents']['value'] != ce['parents']['value']:
        raise ValueError('parent calibration belongs to a different value model')


def _manifest_evidence_config_sha256(manifest):
    c = manifest['config'];e = c['evaluation']
    p = {k:e[k] for k in ('schema_version','synthetic','views','sampling','splits','index','parents','transitions','value','policy')}
    p.update(stage=e['evidence_stage'],model=c['model'],train=dict(c['optimizer'],**c['scheduler'],loss_weights=c['loss']))
    return digest(p)


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
            if cfg.resume and reference == cfg.resume.checkpoint and cfg.stage == 'evaluation':
                with reference.path.open() as handle:
                    recovery = strict_json_load(handle)
                if recovery.get('config_sha256') != evidence_config_sha256(cfg) or recovery.get('source_sha') != pm['git_sha']:
                    raise ValueError('evaluation recovery differs from terminal producer identity')
                changed = subprocess.run(['git','diff','--name-only',pm['git_sha'],manifest['git_sha'],'--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py'],cwd=repository,check=True,capture_output=True,text=True).stdout.strip()
                if changed:
                    raise ValueError('evaluation recovery executable source changed')
            kind = next((name for name in ('reference','matcher','value','calibration') if getattr(cfg.parents,name) == reference),None)
            if kind is not None:
                validate_parent_contract(cfg,kind,pm)
                if reference.path.suffix == '.pt' and reference.path.with_suffix('.json').exists():
                    sidecar = reference.path.with_suffix('.json')
                    if status['output_hashes'].get(sidecar.relative_to(root).as_posix()) != file_sha256(sidecar):
                        raise ValueError('parent checkpoint sidecar is not terminal evidence')
                    with sidecar.open() as handle:
                        metadata = strict_json_load(handle)
                    if metadata['config_sha256'] != _manifest_evidence_config_sha256(pm) or metadata['stage'] != kind or metadata['source_sha'] != pm['git_sha'] or metadata['manifest_sha256'] != record['manifest_sha256']:
                        raise ValueError('parent checkpoint configuration differs from committed producer')
            matched = True
        if not matched:
            raise ValueError('every scientific parent must have committed terminal lineage')


def _load_model(reference,models):
    safe_path(reference.path)
    if file_sha256(reference.path) != reference.sha256:
        raise ValueError('parent checkpoint hash mismatch')
    with reference.path.with_suffix('.json').open() as h:
        meta = strict_json_load(h)
    return load_evidence_model_weights(reference.path,models,expected_config_sha256=meta['config_sha256'],expected_parents=meta['parents'])


def _read_transition_input(cfg,*,matcher_sha256):
    ref = cfg.transitions.reference
    if ref is None or file_sha256(ref.path) != ref.sha256:
        raise ValueError('bound transition input required')
    with ref.path.open() as h:
        p = strict_json_load(h)
    producer = TransitionProducerIdentity(**p['producer'])
    if producer.sha256 != cfg.transitions.producer_sha256:
        raise ValueError('transition producer differs from config')
    if producer.matcher_sha256 != matcher_sha256:
        raise ValueError('transitions do not belong to the loaded frozen matcher')
    role = ('V_calibration' if cfg.stage == 'calibration' else {'val_development':'V_development','val_locked':'V_locked'}[cfg.runtime.eval_split] if cfg.stage == 'evaluation' else 'T_value')
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


def initial_episode_state(episode,cfg,source,*,cache=None,model=None,timings=None,failure_context=None):
    candidates = []
    requests = encodes = 0
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
                with measure(timings,'decode_seconds',next(model.parameters()).device if model is not None else 'cpu'):
                    image = source(provisional,(i,))
            except DecodeFailure:
                continue
            valid.append(i)
            if cache is not None and model is not None:
                device = next(model.parameters()).device
                with torch.no_grad(),measure(timings,'clip_seconds',device):
                    feature = model.clip.encode_frames(image.to(device)[None],torch.ones((1,1),device=device,dtype=torch.bool))[0,0]
                encodes += 1
                identity = FrameCacheIdentity(content,cfg.model.base.clip_sha256,digest(config_payload(cfg.model.base)['clip']),cfg.views.preprocessing_sha256)
                cache.write(canonical,i,feature,identity=identity)
        if not valid:
            actual = ResourceCounts(requests,encodes,0,0)
            failure = InitialObservationFailure(episode.query_id,canonical,actual,budget=reserved.settle(upper,actual))
            if failure_context is not None:
                _record_input_failure(failure_context.paths.provenance,failure,(episode.query_id,),scope='episode')
            raise failure
        candidates.append(CandidateObservation(canonical,content,count,pool,tuple(valid),requested))
    budget = reserved.settle(upper,ResourceCounts(requests,encodes,0,0))
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
        # Episode.source_ids is aggregate deduplicated lineage, not a video-aligned column.
        for c,p,h,count in zip(e.canonical_ids,e.video_paths,e.content_sha256,e.frame_counts):
            if c in videos and videos[c] != (p,h):
                raise ValueError('conflicting canonical source inventory')
            if c in inventory and inventory[c][2] != count:
                raise ValueError('conflicting canonical frame count')
            lineage = set(e.source_ids) | (set(inventory[c][3]) if c in inventory else set())
            videos[c] = (p,h); inventory[c] = (p,h,count,tuple(sorted(lineage)))
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
    started = time.monotonic()
    index_requests = index_encodes = 0
    source = VideoFrameSource(videos,cfg.model.base.clip.image_size,cfg.views.preprocessing_sha256)
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed_training = dist.is_initialized() and cfg.stage in {'reference','matcher'}
    index_store = ArtifactArrayStore(context.paths.provenance/f'index-rank-{rank}'/'arrays') if distributed_training else context.array_store
    cache = FrameFeatureCache(index_store)
    candidates = tuple(CandidateObservation(c,h,count,build_frame_pool(count,cfg.views.pool_size),
                      view_frame_ids(build_frame_pool(count,cfg.views.pool_size),cfg.views.levels[0]),
                      view_frame_ids(build_frame_pool(count,cfg.views.pool_size),cfg.views.levels[0]))
                      for c,(p,h,count,source_id) in sorted(inventory.items()))
    gallery_state = ObservationState('__initial_index__',cfg.index.gallery_version,candidates,cfg.views.levels,
                       ExecutionProfile(cfg.sampling.eval_samples,cfg.policy.cost_weights),BudgetLedger(cfg.policy.limits,ZERO,ZERO),0)
    device = next(model.parameters()).device
    def reader(candidate,ids):
        nonlocal index_requests, index_encodes
        identity = FrameCacheIdentity(candidate.content_sha256,cfg.model.base.clip_sha256,digest(config_payload(cfg.model.base)['clip']),cfg.views.preprocessing_sha256)
        rows = []
        for i in ids:
            path = cache._path(candidate.canonical_id,i,identity)
            if path.exists():
                feature = cache.read(candidate.canonical_id,i,identity=identity).to(device)
            else:
                index_requests += 1
                image = source(candidate,(i,)).to(device)
                with torch.no_grad():
                    feature = model.clip.encode_frames(image[None],torch.ones((1,1),device=device,dtype=torch.bool))[0,0]
                index_encodes += 1
                cache.write(candidate.canonical_id,i,feature,identity=identity)
            rows.append(feature)
        return torch.stack(rows)
    # Persist the successful/attempted sparse inventory so a successor does not
    # retry failed index decodes before recovering its online observations.
    from semantic_retrieval.evidence.types import state_from_payload
    gallery_dir = index_store.run_root/'provenance'/'evidence-gallery'/digest(role)
    gallery_path = gallery_dir/'state.json'
    binding = digest(dict(index=ref.sha256,views=config_payload(cfg.views),clip=cfg.model.base.clip_sha256,role=role))
    safe_path(gallery_path)
    if gallery_path.exists():
        with gallery_path.open() as handle:
            payload = strict_json_load(handle)
        if set(payload) != {'binding_sha256','state','candidate_count','candidates_sha256'} or payload['binding_sha256'] != binding:
            raise ValueError('recovered initial gallery binding mismatch')
        rows = []
        for candidate in candidates:
            path = gallery_dir/(digest(candidate.canonical_id)+'.json');safe_path(path)
            with path.open() as handle:
                rows.append(strict_json_load(handle))
        if payload['candidate_count'] != len(rows) or payload['candidates_sha256'] != digest(rows):
            raise ValueError('recovered gallery candidate metadata hash mismatch')
        restored = state_from_payload(dict(payload['state'],candidates=rows))
        original = replace(restored,candidates=tuple(replace(c,observed_frame_ids=c.attempted_frame_ids) for c in restored.candidates))
        if original != gallery_state:
            raise ValueError('recovered initial gallery source inventory mismatch')
        gallery_state = restored
    else:
        successful = []
        for candidate in candidates:
            valid = []
            for frame_id in candidate.attempted_frame_ids:
                try:
                    reader(candidate,(frame_id,))
                except DecodeFailure:
                    continue
                valid.append(frame_id)
            if not valid:
                failure = InitialObservationFailure('__initial_index__:'+role,candidate.canonical_id,ResourceCounts(index_requests,index_encodes,0,0))
                _record_input_failure(index_store.run_root/'provenance',failure,tuple(w.episode.query_id for w in wrapped),scope='index')
                raise failure
            successful.append(replace(candidate,observed_frame_ids=tuple(valid)))
        gallery_state = replace(gallery_state,candidates=tuple(successful))
        payload = asdict(gallery_state);rows = payload.pop('candidates')
        for row in rows:
            path = gallery_dir/(digest(row['canonical_id'])+'.json')
            if path.exists():
                raise ValueError('partial initial gallery publication')
            publish_json(path,row)
        publish_json(gallery_path,dict(binding_sha256=binding,state=payload,candidate_count=len(rows),candidates_sha256=digest(rows)))
    index_identity = IndexIdentity(cfg.model.base.clip_sha256,cfg.index.gallery_version,
                                  digest(dict(preprocessing=cfg.views.preprocessing_sha256,view=digest(config_payload(cfg.views)))))
    prepared = time.monotonic()
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
                  frame_counts=tuple(inventory[c][2] for c in order),
                  source_ids=tuple(sorted({s for c in order for s in inventory[c][3]})))
        result.append(replace(w,episode=selected_episode))
    index_store.write('initial_index_embeddings',index.embeddings.detach().cpu())
    finished = time.monotonic()
    publish_json(index_store.run_root/'provenance'/('initial-index-'+role+'.json'),dict(schema_version='evidence-initial-index-report-v2',
                 identity=asdict(index_identity),candidate_count=cfg.index.candidate_count,gallery_count=len(index.ids),
                 elapsed_seconds=finished-started,preparation_seconds=prepared-started,build_search_publish_seconds=finished-prepared,
                 resource_counts=asdict(ResourceCounts(index_requests,index_encodes,0,0)),
                 timing_scope='current_worker_gallery_preparation_build_query_search_embedding_publication',
                 role=role,specification_sha256=ref.sha256))
    return tuple(result),{w.episode.query_id:w for w in result},source,cache


def publish_transition_coverage(cfg,role,episodes,producer,references,store):
    """Publish actual selection for the full input cohort after successful compilation."""
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    bindings = [s for s in cfg.splits if s.role == role]
    if role not in {'T_value','V_development','V_calibration'} or len(bindings) != 1:
        raise ValueError('transition coverage requires one permitted role')
    binding = bindings[0]
    expected = read_split_binding(binding)
    if not expected or len(episodes) != len(expected) or references.producer != producer or producer.label_sha256 != binding.manifest.sha256 or producer.index_sha256 != cfg.index.reference.sha256 or file_sha256(references.index_path) != references.index_sha256:
        raise ValueError('transition coverage cohort/producer mismatch')
    safe_path(cfg.index.reference.path)
    if file_sha256(cfg.index.reference.path) != cfg.index.reference.sha256:
        raise ValueError('transition coverage index mismatch')
    with cfg.index.reference.path.open() as handle:
        gallery = strict_json_load(handle)['roles'][role]
    if not gallery or gallery != sorted(set(gallery)):
        raise ValueError('transition coverage requires sorted unique gallery')
    positions = {canonical:i for i,canonical in enumerate(gallery)}
    count = min(cfg.index.candidate_count,len(gallery))
    selections,presence,queries,components = [],[],[],[]
    for original,selected in zip(expected,episodes):
        before,after = original.episode,selected.episode
        if selected.role != role or after.query_id != before.query_id or after.query_component_id != before.query_component_id or len(after.canonical_ids) != count or len(set(after.canonical_ids)) != count or any(c not in positions for c in after.canonical_ids):
            raise ValueError('transition coverage query/selection mismatch')
        labels,actual_presence = bind_candidate_order(before,after.canonical_ids)
        if after.relevance != labels or after.answer_in_candidates != actual_presence:
            raise ValueError('transition coverage label/presence mismatch')
        queries.append(after.query_id);components.append(after.query_component_id)
        selections.append([positions[c] for c in after.canonical_ids]);presence.append(actual_presence)
    if len(set(queries)) != len(queries):
        raise ValueError('transition coverage duplicate query')
    path = store.run_root/'provenance/transition-coverage.json'
    if path.exists():
        raise ValueError('transition coverage already published')
    arrays = [store.write('candidate_presence',torch.tensor(presence,dtype=torch.int8)),
              store.write('selected_gallery_indices',torch.tensor(selections,dtype=torch.int64))]
    payload = dict(schema_version='evidence-transition-coverage-v1',role=role,query_count=len(queries),candidate_count=count,
        producer_sha256=producer.sha256,transition_index_sha256=references.index_sha256,
        role_manifest_sha256=binding.manifest.sha256,index_specification_sha256=cfg.index.reference.sha256,
        query_order_sha256=digest(queries),component_order_sha256=digest(components),gallery_order_sha256=digest(gallery),
        cohort_complete=True,array_references=[asdict(a) for a in arrays])
    validate_evidence_document('transition_coverage',payload)
    publish_json(path,payload)
    return path


def _write_reference_parts(context,model):
    from semantic_retrieval.runtime.checkpoint import _publish_file, _externalize_large_tensors
    path = context.paths.checkpoints/'reference-parts.pt'
    if path.exists():
        raise ValueError('reference snapshot already exists')
    payload = _externalize_large_tensors({'clip':model.clip.state_dict(),'temporal':model.temporal.state_dict(),'matcher.base':model.matcher.base.state_dict()},context.array_store)
    _publish_file(path,lambda h:torch.save(payload,h))
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
        transition_batch = _read_transition_input(cfg,matcher_sha256=model_sha256(model.matcher))
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
            state = initial_episode_state(e,cfg,source,cache=cache,model=model,failure_context=context)
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


def single_publisher(work):
    """Run non-training publication once and propagate its result or error to every rank."""
    if not dist.is_initialized():
        return work()
    envelope = [None]
    if dist.get_rank() == 0:
        try:
            envelope[0] = dict(result=work(),error=None)
        except Exception as error:
            envelope[0] = dict(result=None,error=f'{type(error).__name__}: {error}')
    dist.broadcast_object_list(envelope,src=0)
    if envelope[0]['error'] is not None:
        raise RuntimeError('scientific publisher failed: '+envelope[0]['error'])
    return envelope[0]['result']


def execute_scientific_stage(context,cfg):
    # All ranks validate authorization; only training stages run distributed work.
    from semantic_retrieval.evidence.authorization import assert_evidence_authorized
    assert_evidence_authorized(repo_root=context.repo_root,registry_path=context.config.registry_path,manifest=context.manifest,operation='start')
    if cfg.stage in {'transitions','calibration','evaluation'}:
        return single_publisher(lambda:_execute_scientific_stage(context,cfg))
    return _execute_scientific_stage(context,cfg)


def _execute_scientific_stage(context,cfg):
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
            state = initial_episode_state(w.episode,cfg,source,cache=cache,model=model,failure_context=context)
            adapters[state.query_id] = EpisodeAdapter(model,_query(model,w.episode.query_text),source,cache,sampling)
            counts = adapters[state.query_id].prediction_counts(state)
            state = replace(state,budget=state.budget.reserve(counts).settle(counts,counts))
            initial.append(state)
        observe = lambda s,a:adapters[s.query_id].observe(s,a)
        states = tuple(sample_states(initial,seed=cfg.transitions.state_seed,max_steps=cfg.transitions.trajectories,observe=observe))
        adapter = FrozenTransitionModel(model,lambda s:adapters[s.query_id].predict(s),observe,build_value_features)
        truth = {w.episode.query_id:(dict(zip(w.episode.canonical_ids,w.episode.relevance)),(w.episode.query_component_id,)) for w in episodes}
        refs = compile_transitions(states,propose_actions,adapter,truth,store=context.array_store,producer_identity=producer)
        batch = read_transitions(refs,expected_producer=producer)
        publish_transition_coverage(cfg,role,episodes,producer,refs,context.array_store)
        return dict(summary_metrics={'transition_count':float(len(batch.features))})
    from semantic_retrieval.evidence.value import EvidenceValueModel
    value_container = EvidenceValueModel(model,head)
    if cfg.parents.value is not None:
        _load_model(cfg.parents.value,{'value':head})
    if cfg.stage == 'calibration':
        batch = _read_transition_input(cfg,matcher_sha256=model_sha256(model))
        identity = dict(producer_sha256=batch.targets.producer_sha256,model_sha256=model_sha256(head),role='V_calibration',
                        relevance_space=next(s.relevance_space for s in cfg.splits if s.role == 'V_calibration'),feature_schema_sha256=digest('observable-action-v1'))
        targets = replace(batch.targets,outcome_class=batch.targets.outcome_class.to(device),valid_mask=batch.targets.valid_mask.to(device))
        with torch.no_grad():
            raw_logits = head(batch.features.to(device)).logits
        calibration = fit_value_temperature(raw_logits,targets,split='V_calibration',identity=identity)
        from semantic_retrieval.evidence.schemas import validate_evidence_document
        payload = asdict(calibration); validate_evidence_document('calibration',payload)
        publish_json(context.paths.provenance/'value-calibration.json',payload)
        from semantic_retrieval.evidence.evaluation import prediction_evidence
        evidence = prediction_evidence(store=context.array_store,population=(raw_logits,targets),
            selected=(calibration.apply(raw_logits,expected_identity=identity),targets),rich=None,before=[],after=[],components=(),seed=cfg.sampling.seed)
        names = {'population':'uncalibrated_population','selected':'calibrated_population'}
        metrics = {names.get(k.split('_',1)[0],k.split('_',1)[0])+'_'+k.split('_',1)[1]:v for k,v in evidence.summary_metrics.items()}
        metrics.update(calibration_temperature=calibration.temperature,calibration_count=float(calibration.count))
        evidence = replace(evidence,summary_metrics=metrics,undefined_reasons={},identity=dict(identity,scope='scripted calibration population; no policy-selected actions'))
        publish_json(context.paths.provenance/'calibration-evidence.json',asdict(evidence))
        return dict(summary_metrics=metrics)
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
    from semantic_retrieval.evidence.recovery import import_recovery,export_recovery,read_initial_state,save_initial_state
    if cfg.resume is not None:
        with cfg.resume.checkpoint.path.open() as handle:
            snapshot = strict_json_load(handle)
        # Terminal lineage and executable equivalence have already been checked at prepare/start.
        import_recovery(cfg.resume.checkpoint,context.array_store,config_sha256=evidence_config_sha256(cfg),source_sha=snapshot['source_sha'])
    episodes,lookup,source,cache = _episodes_and_sources(cfg,role,model,context)
    def snapshot_progress():
        return export_recovery(context.array_store,config_sha256=evidence_config_sha256(cfg),source_sha=context.config.source_sha)
    journals = {}
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
    records,components,completed = [],[],[]
    for w in episodes:
        e = w.episode
        journal = ActionJournal(context.array_store.run_root/'provenance'/'evidence-actions',context.array_store,
                                after_write=snapshot_progress,query_id=e.query_id)
        journals[e.query_id] = journal
        timings = {}
        synchronize(next(model.parameters()).device);episode_started = time.monotonic()
        state = read_initial_state(context.array_store,e.query_id)
        if state is None:
            state = initial_episode_state(e,cfg,source,cache=cache,model=model,timings=timings,failure_context=context)
            save_initial_state(context.array_store,state);snapshot_progress()
        elif tuple(c.canonical_id for c in state.candidates) != e.canonical_ids:
            raise ValueError('recovered initial candidate selection differs from fixed index')
        adapter = EpisodeAdapter(model,_query(model,e.query_text),source,cache,sampling,timings=timings)
        def predict(s):
            return adapter.predict(s),adapter.prediction_counts(s)
        def value(features):
            with measure(timings,'value_seconds',features.device):
                out = head(features)
            return replace(out,logits=calibration.apply(out.logits,expected_identity=calibration.identity)) if calibration is not None else out
        from semantic_retrieval.evidence.baselines import build_acquisition_baseline
        policy = build_acquisition_baseline(cfg.policy.baseline,seed=cfg.sampling.seed,config=cfg)
        models = SimpleNamespace(device=next(model.parameters()).device,predict=predict,prediction_counts=adapter.prediction_counts,sampling_sha256=sampling.identity,value_head=value,policy=policy,stop_threshold=cfg.policy.stop_threshold)
        # Each query has its own cache/journal cost account; cross-query feature reuse grants no free actions.
        result = run_policy_episode(context,state,models,adapter.acquire,journal=journal)
        synchronize(next(model.parameters()).device)
        timings['total_seconds'] = time.monotonic()-episode_started
        result = replace(result,timings=timings)
        initial_id = result.initial_ranking[0]
        truth = dict(zip(e.canonical_ids,e.relevance))
        records.append(EvidenceEvaluationRecord(e.query_id,truth[initial_id],truth[result.ranking[0]],e.answer_in_candidates,result.costs,result.timings['total_seconds']))
        components.append(e.query_component_id)
        completed.append((e,result))
    report = evaluate_evidence(records,relevance_space=next(s.relevance_space for s in cfg.splits if s.role == role),source_components=components)
    # Diagnostics run after every online episode, with separate storage and accounting.
    # Counterfactual evidence never enters the online cache or policy state.
    from semantic_retrieval.evidence.evaluation import prediction_evidence
    from semantic_retrieval.evidence.transitions import label_transition
    from semantic_retrieval.evidence.types import TransitionTargets
    population = selected = rich = None
    if cfg.transitions.reference is not None and cfg.value.kind != 'mean_regressor' and cfg.parents.value is not None:
        batch = _read_transition_input(cfg,matcher_sha256=model_sha256(model))
        with torch.no_grad():
            out = head(batch.features.to(next(head.parameters()).device)).logits
            if calibration is not None:
                out = calibration.apply(out,expected_identity=calibration.identity)
        population = (out.cpu(),batch.targets)
    selected_logits,selected_classes,selected_valid = [],[],[]
    if cfg.value.kind != 'mean_regressor' and cfg.parents.value is not None:
        for episode,result in completed:
            journal = journals[episode.query_id]
            truth = dict(zip(episode.canonical_ids,episode.relevance))
            for action_id in result.action_ids:
                before,after,action = journal.action_prediction_states(action_id)
                prior = journal.prediction_for(before,expected_sampling=sampling.identity)
                following = journal.prediction_for(after,expected_sampling=sampling.identity)
                with torch.no_grad():
                    logits = head(build_value_features(before,(action,),prior).to(next(head.parameters()).device)).logits
                    if calibration is not None:
                        logits = calibration.apply(logits,expected_identity=calibration.identity)
                target = label_transition(rank_candidates(prior.ranking_log_probability,tuple(c for _,c in prior.pair_ids))[0],
                    rank_candidates(following.ranking_log_probability,tuple(c for _,c in following.pair_ids))[0],truth,producer_sha256=model_sha256(model))
                selected_logits.append(logits.cpu());selected_classes.append(target.outcome_class);selected_valid.append(target.valid_mask)
        if selected_logits:
            selected = (torch.cat(selected_logits),TransitionTargets(torch.cat(selected_classes),torch.cat(selected_valid),model_sha256(model)))
    diagnostics_started = time.monotonic()
    if cfg.parents.reference is not None and cfg.model.variant in {'gaussian','fixed_scale','standard_probabilistic'}:
        reference = EvidenceMatcherModel(cfg.model).to(next(model.parameters()).device)
        load_reference_parts(cfg.parents.reference.path,reference,expected_sha256=cfg.parents.reference.sha256);freeze_reference(reference)
        diagnostic_cache = FrameFeatureCache(ArtifactArrayStore(context.paths.provenance/'rich-diagnostics'/'arrays'))
        locs,scales,targets,masks = [],[],[],[]
        for episode,result in completed:
            journal = journals[episode.query_id]
            state = result.final_state
            rows = []
            for candidate in state.candidates:
                valid = []
                for frame_id in candidate.pool_frame_ids:
                    try:
                        source(candidate,(frame_id,))
                    except DecodeFailure:
                        continue
                    valid.append(frame_id)
                if not valid:
                    raise DecodeFailure('rich diagnostic has no decodable frames')
                rows.append(replace(candidate,observed_frame_ids=tuple(valid),attempted_frame_ids=candidate.pool_frame_ids))
            rich_state = replace(state,candidates=tuple(rows))
            with torch.no_grad():
                query = _query(model,episode.query_text)
                observed = model.encode_inputs(query,(state,),source,diagnostic_cache)
                rich_observed = reference.encode_inputs(query,(rich_state,),source,diagnostic_cache)
                scores,valid = rich_score_targets(reference,rich_observed,current_observed=observed)
            predicted = journal.prediction_for(state,expected_sampling=sampling.identity)
            locs.append(predicted.loc.cpu());scales.append(predicted.scale.cpu());targets.append(scores.cpu());masks.append(valid.cpu())
        rich = tuple(torch.cat(x) for x in (locs,scales,targets,masks))
    known = [(r,c) for r,c in zip(records,components) if r.initial_top1_label != -1 and r.final_top1_label != -1]
    evidence = prediction_evidence(store=context.array_store,population=population,selected=selected,rich=rich,
        before=[r.initial_top1_label for r,c in known],after=[r.final_top1_label for r,c in known],components=tuple(c for r,c in known),seed=cfg.sampling.seed)
    for name in ('decode_seconds','clip_seconds','query_cache_seconds','temporal_seconds','pair_seconds','scale_seconds','mc_seconds','value_seconds'):
        report.summary_metrics[name] = sum(result.timings.get(name,0.) for episode,result in completed)
    episode_arrays = (('episode_labels',torch.tensor([[r.initial_top1_label,r.final_top1_label,r.candidate_presence] for r in records])),
        ('episode_costs',torch.tensor([r.costs.values() for r in records])),('episode_seconds',torch.tensor([r.elapsed_seconds for r in records],dtype=torch.float64)))
    refs = tuple(context.array_store.write(key,value) for key,value in episode_arrays)
    report = replace(report,summary_metrics=dict(report.summary_metrics,**evidence.summary_metrics,offline_diagnostic_seconds=time.monotonic()-diagnostics_started),
        array_references=refs+evidence.array_references,undefined_reasons=dict(report.undefined_reasons,**evidence.undefined_reasons),
        identity=dict(report.identity,latency_scope='current worker segment including initial acquisition; prior segment latency excluded on successor recovery' if cfg.resume else 'online query including initial acquisition; shared index and offline diagnostics excluded',probability_scopes=evidence.identity,episode_order_sha256=digest([e.query_id for e,result in completed])))
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    payload = asdict(report); validate_evidence_document('report',payload)
    publish_json(context.paths.provenance/'evidence-report.json',payload)
    return dict(summary_metrics=report.summary_metrics)
