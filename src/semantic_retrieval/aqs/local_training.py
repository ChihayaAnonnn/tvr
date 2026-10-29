"""Software-only local-query training and identity-bound exact recovery."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import torch
import torch.distributed as dist
from experiment_artifacts import artifact_json_bytes, strict_json_load
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import _publish_file, restore_rng
from semantic_retrieval.full_data.training import save_state, _progress
from semantic_retrieval.full_data.worker import _collective_call
from .training import CheckpointBinding, model_recipe, _validate_optimizer_binding
from .local_model import LocalQueryAqsModel
from .local_checkpoint import stage_checkpoint, stage_local, validate_local_optimizer

ARCHITECTURE='uatvr-aqs-local-query-v1'


@dataclass(frozen=True,init=False)
class LocalBinding(CheckpointBinding):
    """Immutable canonical bytes, independent of legacy/KL checkpoint binding."""


def local_recipe(model):
    if not isinstance(model,LocalQueryAqsModel):raise ValueError('expected active local query model')
    return dict(architecture=ARCHITECTURE,model=model_recipe(model),local=model.local_config.recipe(),
        fusion=model.fusion_config.recipe(),aqs=model.aqs_config.recipe(),
        chunks=dict(query=model.query_chunk,video=model.video_chunk),
        parent_initialization=getattr(model,'parent_initialization',None))


def local_source_identity():
    root=Path(__file__).resolve().parents[1];base=root.parent.parent
    files=set((root/'aqs').glob('*.py'))
    files.update((root/'full_data/upstream_uatvr').rglob('*.py'))
    files.update((root/'runtime').glob('*.py'))
    files.update(root/name for name in ('full_data/models.py','full_data/training.py','full_data/worker.py',
        'evidence/frame_cache.py','types.py','models/clip.py','config.py'))
    files.add(base/'experiment_artifacts.py')
    return {p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def _record(model,optimizer,binding):
    def prepare():
        if not isinstance(binding,LocalBinding):raise ValueError('expected LocalBinding')
        model._validate_local_context();_validate_optimizer_binding(model,optimizer)
        validate_local_optimizer(optimizer)
        if any(binding.base_recipe.get(k)!=v for k,v in local_recipe(model).items()):
            raise ValueError('local checkpoint recipe mismatch')
        if binding.id_mapping_sha256!=model.id_mapping_sha256:raise ValueError('local ID mapping mismatch')
        identity=binding.base_identity
        world=dist.get_world_size() if dist.is_initialized() else 1
        if identity['world_size']!=world or identity['method']!='uatvr':raise ValueError('local world/method mismatch')
        names={id(p):n for n,p in model.named_parameters()}
        opt_recipe=[dict(parameters=[names[id(p)] for p in g['params']],
            **{k:v for k,v in g.items() if k!='params'}) for g in optimizer.param_groups]
        payload=dict(architecture=ARCHITECTURE,base_recipe=binding.base_recipe,
            id_mapping_sha256=binding.id_mapping_sha256,sources=local_source_identity(),optimizer=opt_recipe,
            optimizer_implementation=type(optimizer).__module__+'.'+type(optimizer).__qualname__)
        identity['recipe_sha256']=hashlib.sha256(artifact_json_bytes(payload)).hexdigest()
        return dict(schema='uatvr-aqs-local-checkpoint-v1',architecture=ARCHITECTURE,identity=identity,recipe=payload)
    record=_collective_call(prepare)
    if dist.is_initialized():
        records=[None]*dist.get_world_size();dist.all_gather_object(records,record)
        if any(r!=record for r in records):raise ValueError('local checkpoint rank identity mismatch')
    return record


def save_local_state(path,model,optimizer,binding,progress,store):
    record=_record(model,optimizer,binding);path=Path(path)
    def validate():
        _progress(progress);validate_local_optimizer(optimizer)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('nonfinite local state')
    _collective_call(validate)
    def publish():
        if not dist.is_initialized() or dist.get_rank()==0:
            safe_path(path);safe_path(path.with_suffix('.local.json'))
            if path.parent.parent!=store.run_root:raise ValueError('checkpoint and arrays must share run root')
            if any(p.exists() for p in (path,path.with_suffix('.json'),path.with_suffix('.local.json'))):
                raise FileExistsError('local checkpoint publication is exclusive')
            _publish_file(path.with_suffix('.local.json'),lambda f:f.write(artifact_json_bytes(record)))
    _collective_call(publish)
    return save_state(path,model,optimizer,record['identity'],progress,store)


def restore_local_state(path,model,optimizer,binding):
    record=_record(model,optimizer,binding);path=Path(path)
    def verify():
        sidecar=path.with_suffix('.local.json');safe_path(sidecar)
        with sidecar.open() as stream:actual=strict_json_load(stream)
        if actual!=record:raise ValueError('local checkpoint provenance mismatch')
    _collective_call(verify)
    payload,content_sha256=_collective_call(lambda:stage_local(path,model,optimizer,record['identity']))
    if dist.is_initialized():
        hashes=[None]*dist.get_world_size()
        dist.all_gather_object(hashes,content_sha256)
        if any(value!=content_sha256 for value in hashes):
            raise ValueError('local checkpoint content differs across ranks')
    # Every rank has staged every field before the first live mutation.
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    rank=dist.get_rank() if dist.is_initialized() else 0
    restore_rng(payload['rng'][rank])
    return payload['progress']


def initialize_local_parent(path,model,*,expected_sha256):
    def stage():
        if (not isinstance(model,LocalQueryAqsModel) or not isinstance(expected_sha256,str)
                or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None):
            raise ValueError('invalid parent initialization identity')
        model._validate_local_context()
        parent=Path(path);sidecar=parent.with_suffix('.aqs.json');safe_path(sidecar)
        payload,meta=stage_checkpoint(parent)
        if meta['sha256']!=expected_sha256:raise ValueError('parent content hash mismatch')
        with sidecar.open() as stream:side=strict_json_load(stream)
        if (not isinstance(side,dict) or set(side)!={'schema_version','identity','recipe'}
                or side['schema_version']!='uatvr-aqs-recipe-v1' or side['identity']!=meta['identity']
                or side['recipe'].get('architecture')!='uatvr-aqs-v1'
                or side['recipe'].get('id_mapping_sha256')!=model.id_mapping_sha256
                or hashlib.sha256(artifact_json_bytes(side['recipe'])).hexdigest()!=meta['identity']['recipe_sha256']):
            raise ValueError('parent must be an authenticated old AQS checkpoint')
        current=model.state_dict();common={k:v for k,v in current.items() if not k.startswith('local_queries.')}
        saved=payload['model']
        if saved.keys()!=common.keys() or any(saved[k].shape!=v.shape or saved[k].dtype!=v.dtype for k,v in common.items()):
            raise ValueError('parent common parameter shape/key mismatch')
        merged=dict(current);merged.update(saved)
        record=dict(architecture='uatvr-aqs-v1',sha256=expected_sha256,model_sha256=meta['model_sha256'],
                    source_identity=meta['identity'])
        return merged,record
    merged,record=_collective_call(stage)
    if dist.is_initialized():
        records=[None]*dist.get_world_size();dist.all_gather_object(records,record)
        if any(r!=record for r in records):raise ValueError('parent rank identity mismatch')
    model.load_state_dict(merged,strict=True)
    model.parent_initialization=record
    return record


def local_step(model,network,optimizer,batch):
    def prepare():
        if not isinstance(model,LocalQueryAqsModel) or getattr(network,'module',network) is not model:
            raise ValueError('network must wrap supplied local model')
        model._validate_local_context();_validate_optimizer_binding(model,optimizer)
        validate_local_optimizer(optimizer)
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite local parameter')
    _collective_call(prepare)
    model.train();optimizer.zero_grad(set_to_none=True)
    output=network(*batch);output['loss'].backward()
    def validate():
        for name,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise ValueError('nonfinite local gradient: '+name)
        intended=[*model.local_queries.parameters(),model.clip.visual.proj,model.clip.text_projection]
        # A one-video tail batch has zero CE/AQS signal; zero is legitimate,
        # whereas None indicates a disconnected trainable branch.
        if any(p.grad is None for p in intended):raise ValueError('missing local query/encoder gradient')
    _collective_call(validate)
    diagnostics=dict(query_gradient_norm=float(model.local_queries.query.grad.norm()))
    _collective_call(lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    with torch.no_grad():model.clip.logit_scale.clamp_(max=math.log(100))
    def finite():
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite local update')
        validate_local_optimizer(optimizer)
    _collective_call(finite)
    result={k:float(output[k].detach()) for k in ('loss','contrastive','mil','kl','aqs')}
    result.update(output['aqs_stats']);result.update(output['local_stats']);result.update(diagnostics)
    result.update(local_active=1.,aqs_active=float(model.aqs_config.active),
        aqs_weighted=model.aqs_config.weight*result['aqs'],diagnostic_rank=float(dist.get_rank() if dist.is_initialized() else 0))
    return result
