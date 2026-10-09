"""Strict evidence checkpoint publication, per-rank progress and exact RNG recovery."""
import hashlib
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from experiment_artifacts import artifact_json_bytes, load_validated_run_lifecycle, strict_json_load
from semantic_retrieval.config import config_payload
from semantic_retrieval.runtime.checkpoint import (
    _externalize_large_tensors, _hydrate_array_references, _publish_file,
    model_sha256, rng_state, restore_rng,
)
from semantic_retrieval.evidence.config import evidence_manifest_config
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.evidence.types import digest, positive, require_digest

SCHEMA = 'evidence-checkpoint-v1'


def evidence_config_sha256(config):
    p = config.to_payload()
    p.pop('runtime'); p.pop('resume')
    return digest(p)


@dataclass(frozen=True)
class EvidenceProgress:
    stage: str
    step: int
    epoch: int
    cursor: int
    sampler_state: dict
    mc_state: dict
    producer_sha256: str
    observation_state: dict | None
    pending_action: dict | None

    def __post_init__(self):
        if self.stage not in {'cpu_fixture','gpu_qualification','reference','matcher','transitions','value','calibration','evaluation'}:
            raise ValueError('invalid checkpoint stage')
        for x in (self.step,self.epoch,self.cursor):
            positive(x,integer=True,zero=True)
        require_digest(self.producer_sha256)
        if not isinstance(self.sampler_state,dict) or not isinstance(self.mc_state,dict) or self.sampler_state.get('cursor') != self.cursor:
            raise ValueError('checkpoint must carry exact sampler cursor')


@dataclass(frozen=True)
class EvidenceTrainingState:
    progress: EvidenceProgress
    optimizer_state: dict
    scheduler_state: dict
    rng_states: tuple


def save_evidence_checkpoint(context, models, optimizer, scheduler, config, progress):
    manifest,status,_ = load_validated_run_lifecycle(context.paths)
    if status['state'] != 'running' or manifest != context.manifest or context.system_config != config or manifest['config'] != evidence_manifest_config(config,eval_split=context.config.eval_split):
        raise ValueError('checkpoint requires exact running evidence identity')
    if progress.stage != config.stage or not models:
        raise ValueError('checkpoint stage/models mismatch')
    rank = dist.get_rank() if dist.is_initialized() else 0
    local = dict(progress=asdict(progress),rng=rng_state())
    ranks = [local]
    if dist.is_initialized():
        ranks = [None]*dist.get_world_size(); dist.all_gather_object(ranks,local)
    hashes = {k:model_sha256(v) for k,v in models.items()}
    if dist.is_initialized():
        all_hashes = [None]*dist.get_world_size(); dist.all_gather_object(all_hashes,hashes)
        if any(x != hashes for x in all_hashes):
            raise ValueError('checkpoint model weights differ across ranks')
    path = context.paths.checkpoints/f'evidence-step-{progress.step:08d}-cursor-{progress.cursor:08d}.pt'
    error = [None]
    if rank == 0:
        try:
            if path.exists() or path.with_suffix('.json').exists():
                raise ValueError('checkpoint publication cannot overwrite historical output')
            payload = dict(schema_version=SCHEMA,config_sha256=evidence_config_sha256(config),parents=config_payload(config.parents),
                           models={k:v.state_dict() for k,v in models.items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                           ranks=ranks,world_size=len(ranks))
            payload = _externalize_large_tensors(payload,context.array_store)
            _publish_file(path,lambda handle:torch.save(payload,handle))
            meta = dict(schema_version=SCHEMA,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        config_sha256=payload['config_sha256'],parents=payload['parents'],model_sha256=hashes,
                        stage=progress.stage,producer_sha256=progress.producer_sha256,world_size=len(ranks),
                        source_sha=context.config.source_sha,manifest_sha256=context.config.manifest_sha256)
            from semantic_retrieval.evidence.schemas import validate_evidence_document
            validate_evidence_document('checkpoint_sidecar',meta)
            _publish_file(path.with_suffix('.json'),lambda handle:handle.write(artifact_json_bytes(meta)))
        except Exception as exc:
            if not dist.is_initialized():
                raise
            error[0] = str(exc)
    if dist.is_initialized():
        dist.broadcast_object_list(error,src=0)
    if error[0] is not None:
        raise ValueError('checkpoint publication failed: '+error[0])
    return path


def load_evidence_checkpoint(path, models, *, expected_config_sha256, expected_parents):
    safe_path(path); safe_path(path.with_suffix('.json'))
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError('partial checkpoint publication')
    with path.with_suffix('.json').open() as h:
        meta = strict_json_load(h)
    from semantic_retrieval.evidence.schemas import validate_evidence_document
    validate_evidence_document('checkpoint_sidecar',meta)
    keys = {'schema_version','sha256','config_sha256','parents','model_sha256','stage','producer_sha256','world_size','source_sha','manifest_sha256'}
    parents = expected_parents if isinstance(expected_parents,dict) else config_payload(expected_parents)
    if set(meta) != keys or meta['schema_version'] != SCHEMA or meta['config_sha256'] != expected_config_sha256 or meta['parents'] != parents or meta['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('checkpoint schema/hash/config/parents mismatch')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if set(payload) != {'schema_version','config_sha256','parents','models','optimizer','scheduler','ranks','world_size'} or payload['schema_version'] != SCHEMA or payload['config_sha256'] != expected_config_sha256 or payload['parents'] != parents:
        raise ValueError('checkpoint payload identity mismatch')
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if payload['world_size'] != world or meta['world_size'] != world or len(payload['ranks']) != world or set(payload['models']) != set(models) or set(meta['model_sha256']) != set(models):
        raise ValueError('checkpoint world size/model set mismatch')
    payload = _hydrate_array_references(payload,path.parent.parent)
    progress = EvidenceProgress(**payload['ranks'][rank]['progress'])
    if progress.stage != meta['stage'] or progress.producer_sha256 != meta['producer_sha256']:
        raise ValueError('checkpoint progress producer/stage mismatch')
    # Validate every model before mutating any destination module.
    for key,model in models.items():
        current = model.state_dict(); stored = payload['models'][key]
        if set(current) != set(stored) or any(current[k].shape != stored[k].shape or current[k].dtype != stored[k].dtype or not torch.isfinite(stored[k]).all() for k in current):
            raise ValueError('checkpoint model schema mismatch')
        h = hashlib.sha256()
        for name,tensor in sorted(stored.items()):
            x = tensor.contiguous(); h.update(name.encode()); h.update(str((tuple(x.shape),str(x.dtype))).encode()); h.update(x.numpy().tobytes())
        if h.hexdigest() != meta['model_sha256'][key]:
            raise ValueError('checkpoint model digest mismatch')
    for key,model in models.items():
        model.load_state_dict(payload['models'][key],strict=True)
    states = tuple(p['rng'] for p in payload['ranks'])
    restore_rng(states[rank])
    return EvidenceTrainingState(progress,payload['optimizer'],payload['scheduler'],states)
