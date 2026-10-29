"""Read-only staging of authenticated local/parent checkpoint payloads."""
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from experiment_artifacts import strict_json_load
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import _hydrate_array_references
from semantic_retrieval.full_data.training import _identity, _progress, _optimizer_state


def stage_checkpoint(path):
    path=Path(path);safe_path(path);safe_path(path.with_suffix('.json'))
    with path.with_suffix('.json').open() as stream:meta=strict_json_load(stream)
    if (not isinstance(meta,dict) or set(meta)!={'schema_version','identity','sha256','model_sha256'}
            or meta['schema_version']!='full-data-checkpoint-v1'):
        raise ValueError('checkpoint metadata schema mismatch')
    _identity(meta['identity'])
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('checkpoint content hash mismatch')
    # Deserialize exactly the bytes whose hash was checked.
    import io
    payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if (not isinstance(payload,dict) or set(payload)!={'schema_version','identity','progress','model','optimizer','rng'}
            or payload['schema_version']!=meta['schema_version'] or payload['identity']!=meta['identity']):
        raise ValueError('checkpoint payload identity/schema mismatch')
    payload=_hydrate_array_references(payload,path.parent.parent)
    _progress(payload['progress'])
    state=payload['model']
    if (not isinstance(state,dict) or not state or any(type(k) is not str or not isinstance(v,torch.Tensor)
            or not torch.isfinite(v).all() for k,v in state.items())):
        raise ValueError('checkpoint model must contain finite tensors')
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        tensor=tensor.contiguous()
        digest.update(name.encode());digest.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    if digest.hexdigest()!=meta['model_sha256']:raise ValueError('checkpoint model digest mismatch')
    return payload,meta


def stage_local(path,model,optimizer,identity):
    payload,meta=stage_checkpoint(path)
    if meta['identity']!=identity:raise ValueError('checkpoint identity mismatch')
    current=model.state_dict();saved=payload['model']
    if (saved.keys()!=current.keys() or any(saved[k].shape!=v.shape or saved[k].dtype!=v.dtype for k,v in current.items())):
        raise ValueError('checkpoint strict model state mismatch')
    _optimizer_state(payload['optimizer'],optimizer)
    world=dist.get_world_size() if dist.is_initialized() else 1
    states=payload['rng']
    if not isinstance(states,list) or len(states)!=world or world!=identity['world_size']:
        raise ValueError('checkpoint RNG world size mismatch')
    for rng in states:
        if not isinstance(rng,dict) or set(rng)!={'torch','python','numpy','cuda'}:
            raise ValueError('checkpoint RNG schema mismatch')
        torch.Generator().set_state(rng['torch'])
        random.Random().setstate(rng['python'])
        ns=rng['numpy']
        np.random.RandomState().set_state((ns[0],ns[1].numpy().astype(np.uint32),ns[2],ns[3],ns[4]))
        if not isinstance(rng['cuda'],list) or len(rng['cuda'])!=torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG count mismatch')
        for i,state in enumerate(rng['cuda']):torch.Generator(device=torch.device('cuda',i)).set_state(state)
    return payload
