"""Software training and strict, versioned recovery for fused scoring."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import torch
import torch.distributed as dist
from experiment_artifacts import artifact_json_bytes,strict_json_load
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import _publish_file
from semantic_retrieval.full_data.training import save_state,restore_state,_optimizer_state,_progress
from semantic_retrieval.full_data.worker import _collective_call
from .training import CheckpointBinding,model_recipe,_validate_optimizer_binding
from .fusion_model import FusionAqsModel


@dataclass(frozen=True,init=False)
class FusionBinding(CheckpointBinding):
    """Canonical immutable base recipe; scoring semantics are checked at use."""


def fusion_recipe(model):
    if not isinstance(model,FusionAqsModel): raise ValueError('fusion requires FusionAqsModel')
    return dict(architecture='uatvr-aqs-kl-v1',model=model_recipe(model),
                fusion=model.fusion_config.recipe(),aqs=model.aqs_config.recipe())


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files=set((root/'aqs').glob('*.py'))
    files.update((root/'full_data/upstream_uatvr').rglob('*.py'))
    files.update(root/name for name in ('full_data/models.py','full_data/training.py','full_data/worker.py',
        'runtime/checkpoint.py','runtime/distributed.py','types.py','models/clip.py'))
    files.add(root.parent.parent/'experiment_artifacts.py')
    base=root.parent.parent
    return {p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def _record(model,optimizer,binding):
    def prepare():
        if not isinstance(binding,FusionBinding): raise ValueError('expected FusionBinding')
        model._validate_fusion_context()
        _validate_optimizer_binding(model,optimizer)
        recipe=fusion_recipe(model)
        if any(binding.base_recipe.get(k)!=v for k,v in recipe.items()):
            raise ValueError('fusion model recipe mismatch')
        if binding.id_mapping_sha256!=model.id_mapping_sha256:
            raise ValueError('fusion ID mapping mismatch')
        identity=binding.base_identity
        world=dist.get_world_size() if dist.is_initialized() else 1
        if identity['world_size']!=world or identity['method']!='uatvr': raise ValueError('fusion world/method mismatch')
        payload=dict(architecture='uatvr-aqs-kl-v1',base_recipe=binding.base_recipe,
            id_mapping_sha256=binding.id_mapping_sha256,sources=source_identity())
        identity['recipe_sha256']=hashlib.sha256(artifact_json_bytes(payload)).hexdigest()
        return dict(schema='uatvr-aqs-kl-checkpoint-v1',architecture='uatvr-aqs-kl-v1',identity=identity,recipe=payload)
    record=_collective_call(prepare) if dist.is_initialized() else prepare()
    if dist.is_initialized():
        records=[None]*dist.get_world_size();dist.all_gather_object(records,record)
        if any(r!=record for r in records): raise ValueError('fusion checkpoint rank identity mismatch')
    return record


def save_fusion_state(path,model,optimizer,binding,progress,store):
    record=_record(model,optimizer,binding);path=Path(path)
    def validate():
        _progress(progress);_optimizer_state(optimizer.state_dict(),optimizer)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()): raise ValueError('nonfinite checkpoint model')
    _collective_call(validate)
    def publish():
        if not dist.is_initialized() or dist.get_rank()==0:
            safe_path(path);safe_path(path.with_suffix('.fusion.json'))
            if path.parent.parent!=store.run_root: raise ValueError('fusion checkpoint must share run-local array store')
            if any(p.exists() for p in (path,path.with_suffix('.json'),path.with_suffix('.fusion.json'))):
                raise FileExistsError('fusion checkpoint publication is exclusive')
            _publish_file(path.with_suffix('.fusion.json'),lambda f:f.write(artifact_json_bytes(record)))
    if dist.is_initialized(): _collective_call(publish)
    else: publish()
    return save_state(path,model,optimizer,record['identity'],progress,store)


def restore_fusion_state(path,model,optimizer,binding):
    record=_record(model,optimizer,binding);path=Path(path)
    def verify():
        sidecar=path.with_suffix('.fusion.json');safe_path(sidecar)
        with sidecar.open() as stream: actual=strict_json_load(stream)
        if actual!=record: raise ValueError('fusion checkpoint recipe/provenance mismatch')
    if dist.is_initialized(): _collective_call(verify)
    else: verify()
    # The low-level loader validates arrays, optimizer, state and each RNG before mutation.
    return restore_state(path,model,optimizer,record['identity'])


def fusion_step(model,network,optimizer,batch):
    def prepare():
        if not isinstance(model,FusionAqsModel) or getattr(network,'module',network) is not model:
            raise ValueError('network must wrap the supplied fusion model')
        model._validate_fusion_context();_validate_optimizer_binding(model,optimizer)
    _collective_call(prepare)
    model.train();optimizer.zero_grad(set_to_none=True)
    output=network(*batch);output['loss'].backward()
    heads=dict(text_variance=model.core.uncertain_net_text.fc.weight,video_variance=model.core.uncertain_net_video.fc.weight,
               text_mean=model.core.pie_net_text.fc.weight,video_mean=model.core.pie_net_video.fc.weight)
    def validate():
        for name,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise ValueError('nonfinite fusion gradient: '+name)
        intended=[*heads.values(),model.clip.text_projection,model.clip.visual.proj,model.core.frame_position_embeddings.weight]
        if any(p.grad is None or p.grad.abs().sum()==0 for p in intended):
            raise ValueError('intended fusion parameters lack a nonzero gradient')
    _collective_call(validate)
    before={k:p.detach().clone() for k,p in heads.items()}
    diagnostics={k+'_gradient_norm':float(p.grad.norm()) for k,p in heads.items()}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad(): model.clip.logit_scale.clamp_(max=math.log(100))
    def finite():
        if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('nonfinite fusion parameter after update')
    _collective_call(finite)
    result={k:float(output[k].detach()) for k in ('loss','contrastive','mil','kl')}
    active=model.aqs_config.active
    result.update(aqs=float(output['aqs'].detach()) if active else 0.,aqs_active=float(active),
                  fusion_active=float(model.fusion_config.active),diagnostic_rank=float(dist.get_rank() if dist.is_initialized() else 0))
    result['aqs_weighted']=model.aqs_config.weight*result['aqs']
    result.update(output['aqs_stats'] if active else dict(mean_negatives=0.,mean_difficulty=0.,mean_uniqueness=0.,empty_rows=0.))
    if model.fusion_config.active: result.update(output['fusion_stats'])
    result.update(diagnostics)
    result.update({k+'_update_norm':float((p.detach()-before[k]).norm()) for k,p in heads.items()})
    return result
