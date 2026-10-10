"""Full-caption paired batches and exact, identity-bound baseline recovery."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import re

import numpy as np
import torch
import torch.distributed as dist

from experiment_artifacts import artifact_json_bytes, strict_json_load, verify_array_reference
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import (
    _publish_file, _externalize_large_tensors, _hydrate_array_references, model_sha256,
    rng_state, restore_rng,
)


def unique_video_batches(identities, *, global_batch, world_size, seed, epoch):
    if (not identities or any(not isinstance(i, str) or not i for i in identities)
            or any(type(i) is not int or i < 1 for i in (global_batch, world_size))
            or global_batch % world_size or type(seed) is not int or seed < 0
            or type(epoch) is not int or epoch < 0):
        raise ValueError('invalid unique-video sampler configuration')
    groups = defaultdict(list)
    for index, video in enumerate(identities):
        groups[video].append(index)
    counts = {len(indices) for indices in groups.values()}
    if len(counts) != 1 or len(identities) % world_size or len(groups) < world_size:
        raise ValueError('complete paired sampler requires equal caption counts and divisible complete cohort')
    limit = min(global_batch, len(groups)//world_size*world_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    groups = {i: list(rng.permutation(groups[i])) for i in sorted(groups)}
    videos = list(groups)
    batches, current, seen = [], [], set()
    def add(i, caption):
        if i in seen:
            raise ValueError('unique-video sampler cannot complete without repeated identities')
        current.append(int(caption)); seen.add(i)
        if len(current) == limit:
            batches.append(tuple(current)); current.clear(); seen.clear()
    for cycle in range(next(iter(counts))):
        deferred = []
        for i in rng.permutation(videos):
            if i in seen:
                deferred.append(i)
            else:
                add(i, groups[i][cycle])
        for i in deferred:
            add(i, groups[i][cycle])
    if current:
        if len(current) % world_size:
            raise ValueError('partial global batch cannot be equally partitioned without padding/dropping')
        batches.append(tuple(current))
    if sorted(i for batch in batches for i in batch) != list(range(len(identities))):
        raise ValueError('incomplete epoch caption coverage')
    return batches


def make_optimizer(model, *, learning_rate, total_steps):
    """Pinned official BertAdam recipe, including its uncorrected moments."""
    from .upstream_uatvr.nn_components.optimization import BertAdam
    if (type(learning_rate) not in (float, int) or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(total_steps) is not int or total_steps < 2):
        raise ValueError('invalid baseline optimizer schedule')
    clip_ids = {id(p) for p in model.clip.parameters()}
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if p.requires_grad:
            decay = 0.0 if any(s in name for s in ('bias', 'LayerNorm.bias', 'LayerNorm.weight')) else 0.2
            groups[(id(p) in clip_ids, decay)].append(p)
    parameters = [dict(params=params, weight_decay=decay, lr=learning_rate*(0.001 if clip else 1.0))
                  for (clip, decay), params in sorted(groups.items())]
    return BertAdam(parameters, lr=learning_rate, warmup=0.1, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=total_steps, weight_decay=0.2, max_grad_norm=1.0)


def _identity(identity):
    if (not isinstance(identity, dict)
            or set(identity) != {'source_sha', 'recipe_sha256', 'input_sha256', 'method', 'world_size'}
            or re.fullmatch('[0-9a-f]{40}', str(identity['source_sha'])) is None
            or any(re.fullmatch('[0-9a-f]{64}', str(identity[k])) is None for k in ('recipe_sha256', 'input_sha256'))
            or identity['method'] not in {'clip4clip', 'uatvr'}
            or type(identity['world_size']) is not int or identity['world_size'] < 1):
        raise ValueError('invalid checkpoint identity')


def _progress(progress):
    if (not isinstance(progress, dict) or set(progress) != {'epoch', 'batch_cursor', 'update'}
            or any(type(x) is not int or x < 0 for x in progress.values())):
        raise ValueError('invalid checkpoint progress')


def _optimizer_state(state, optimizer):
    expected = optimizer.state_dict()
    if (not isinstance(state, dict) or set(state) != {'state', 'param_groups'}
            or not isinstance(state['state'], dict) or not isinstance(state['param_groups'], list)
            or len(state['param_groups']) != len(expected['param_groups'])):
        raise ValueError('checkpoint optimizer state mismatch')
    parameters = {}
    for old, new, live in zip(state['param_groups'], expected['param_groups'], optimizer.param_groups):
        if (not isinstance(old, dict) or set(old) != set(new) or old['params'] != new['params']
                or any(old[k] != new[k] for k in old if k != 'params')):
            raise ValueError('checkpoint optimizer recipe/parameter order mismatch')
        parameters.update(zip(old['params'], live['params']))
    if not set(state['state']) <= parameters.keys():
        raise ValueError('checkpoint optimizer has unknown parameters')
    for index, values in state['state'].items():
        if (not isinstance(values, dict)
                or set(values) not in ({'step', 'next_m', 'next_v'}, {'step', 'exp_avg', 'exp_avg_sq'},
                                      {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'})):
            raise ValueError('checkpoint optimizer parameter state schema mismatch')
        for key, value in values.items():
            if key == 'step':
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1 or not torch.isfinite(value).all() or value.item() < 0 or value.item() != int(value.item()):
                        raise ValueError('invalid optimizer step')
                elif type(value) is not int or value < 0:
                    raise ValueError('invalid optimizer step')
            elif (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                  or value.dtype != parameters[index].dtype or not torch.isfinite(value).all()):
                raise ValueError('checkpoint optimizer moment shape/dtype/finite mismatch')


def save_state(path, model, optimizer, identity, progress, store):
    path = Path(path)
    safe_path(path); safe_path(path.with_suffix('.json'))
    _identity(identity); _progress(progress)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('checkpoint publication is exclusive')
    if path.parent.parent != store.run_root:
        raise ValueError('checkpoint must share array store run root')
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    if world != identity['world_size']:
        raise ValueError('checkpoint training world size differs from identity')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint model state')
    _optimizer_state(optimizer.state_dict(), optimizer)
    states = [rng_state()]
    if world > 1:
        states = [None]*world
        dist.all_gather_object(states, rng_state())
    error = [None]
    if rank == 0:
        try:
            payload = dict(schema_version='full-data-checkpoint-v1', identity=identity, progress=progress,
                           model=model.state_dict(), optimizer=optimizer.state_dict(), rng=states)
            payload = _externalize_large_tensors(payload, store)
            _publish_file(path, lambda stream: torch.save(payload, stream))
            meta = dict(schema_version='full-data-checkpoint-v1', identity=identity,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_sha256=model_sha256(model))
            _publish_file(path.with_suffix('.json'), lambda stream: stream.write(artifact_json_bytes(meta)))
        except Exception as exc:
            if world == 1:
                raise
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError('checkpoint publication failed: '+error[0])
    return path


def restore_state(path, model, optimizer, identity):
    path = Path(path); _identity(identity)
    safe_path(path); safe_path(path.with_suffix('.json'))
    with path.with_suffix('.json').open() as stream:
        meta = strict_json_load(stream)
    if (set(meta) != {'schema_version', 'identity', 'sha256', 'model_sha256'}
            or meta['schema_version'] != 'full-data-checkpoint-v1'
            or meta['identity'] != identity):
        raise ValueError('checkpoint identity/schema mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('checkpoint content hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'identity', 'progress', 'model', 'optimizer', 'rng'}
            or payload['identity'] != identity or payload['schema_version'] != meta['schema_version']):
        raise ValueError('checkpoint payload identity/schema mismatch')
    payload = _hydrate_array_references(payload, path.parent.parent)
    _progress(payload['progress'])
    current, saved = model.state_dict(), payload['model']
    if (not isinstance(saved, dict) or saved.keys() != current.keys()
            or any(not isinstance(saved[k], torch.Tensor) or saved[k].shape != v.shape
                   or saved[k].dtype != v.dtype or not torch.isfinite(saved[k]).all() for k, v in current.items())):
        raise ValueError('checkpoint strict model state mismatch')
    digest = hashlib.sha256()
    for name, tensor in sorted(saved.items()):
        tensor = tensor.contiguous()
        digest.update(name.encode()); digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    if digest.hexdigest() != meta['model_sha256']:
        raise ValueError('checkpoint model identity mismatch')
    state = payload['optimizer']
    _optimizer_state(state, optimizer)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if not isinstance(payload['rng'], list) or len(payload['rng']) != world or world != identity['world_size']:
        raise ValueError('checkpoint RNG world size mismatch')
    for rng in payload['rng']:
        if not isinstance(rng, dict) or set(rng) != {'torch', 'python', 'numpy', 'cuda'}:
            raise ValueError('checkpoint RNG schema mismatch')
        torch.Generator().set_state(rng['torch'])
        random.Random().setstate(rng['python'])
        ns = rng['numpy']
        np.random.RandomState().set_state((ns[0], ns[1].numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
        if len(rng['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG device count mismatch')
        for index, state_tensor in enumerate(rng['cuda']):
            # Validate with a private generator before mutating the live model/RNG.
            torch.Generator(device=torch.device('cuda', index)).set_state(state_tensor)
    model.load_state_dict(saved, strict=True)
    optimizer.load_state_dict(state)
    restore_rng(payload['rng'][rank])
    return payload['progress']


def carry_checkpoint(source, target, store):
    """Copy authenticated historical state and all arrays into a new run, exclusively."""
    source, target = Path(source), Path(target)
    safe_path(source); safe_path(source.with_suffix('.json')); safe_path(target)
    if target.parent.parent != store.run_root:
        raise ValueError('historical checkpoint must share successor array store root')
    with source.with_suffix('.json').open() as stream: meta = strict_json_load(stream)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != meta['sha256']:
        raise ValueError('historical checkpoint content hash mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=True)
    def copy_arrays(value):
        if isinstance(value, dict):
            if set(value) == {'__array_reference__'}:
                reference = value['__array_reference__']
                copied = store.write(reference['key'], verify_array_reference(source.parent.parent, reference)).as_dict()
                if copied != reference: raise ValueError('historical array identity changed during carry')
            else:
                for item in value.values(): copy_arrays(item)
        elif isinstance(value, (tuple, list)):
            for item in value: copy_arrays(item)
    copy_arrays(payload)
    _publish_file(target, lambda stream: stream.write(data))
    _publish_file(target.with_suffix('.json'), lambda stream: stream.write(artifact_json_bytes(meta)))
    return target
