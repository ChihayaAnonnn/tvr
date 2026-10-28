"""AQS software training and strict recovery APIs, without experiment launchers."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re

import torch
import torch.distributed as dist

from experiment_artifacts import artifact_json_bytes, strict_json_loads, strict_json_load
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import _publish_file
from semantic_retrieval.full_data.training import _identity, save_state, restore_state
from semantic_retrieval.full_data.worker import _collective_call
from .config import AqsConfig


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('invalid canonical ID mapping digest')


def _canonical(value):
    def visit(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item: visit(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values(): visit(child)
            return
        raise ValueError('checkpoint recipe requires finite canonical JSON values')
    visit(value)
    return artifact_json_bytes(value)


def video_id_mapping(video_ids):
    """Use the same lexicographic ordering as CaptionDataset.video_ids."""
    if (not isinstance(video_ids, (list, tuple)) or not video_ids
            or any(type(value) is not str or not value for value in video_ids)
            or len(set(video_ids)) != len(video_ids)):
        raise ValueError('video ID table must contain unique nonempty strings')
    ordered = sorted(video_ids)
    return ({video: index for index, video in enumerate(ordered)},
            hashlib.sha256(_canonical(dict(schema='canonical-video-id-v1', video_ids=ordered))).hexdigest())


def model_recipe(model):
    from .model import AqsBaselineModel
    if not isinstance(model, AqsBaselineModel) or model.method != 'uatvr':
        raise ValueError('AQS checkpoint requires the AQS UATVR model')
    task = model.core.task_config
    return dict(clip=asdict(model.clip_config), batching=model.batching,
                temporal_layers=task.cross_num_hidden_layers,
                extra_video_tokens=model.core.extra_cls_frame_num,
                extra_text_tokens=model.core.extra_cls_text_num,
                video_samples=model.core.n_video_samples, text_samples=model.core.n_text_samples,
                trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad])


def _source_identity():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('config.py', 'selection.py', 'loss.py', 'core.py', 'model.py', 'training.py')}


def _bound_payload(base_recipe, aqs, id_mapping_sha256):
    return dict(architecture='uatvr-aqs-v1', base_recipe=base_recipe, aqs=aqs.recipe(),
                id_mapping_sha256=id_mapping_sha256, extension_sources=_source_identity())


def bound_identity(base_identity, base_recipe, aqs, id_mapping_sha256):
    _identity(base_identity)
    _digest(id_mapping_sha256)
    if base_identity['method'] != 'uatvr' or not isinstance(aqs, AqsConfig) or not isinstance(base_recipe, dict):
        raise ValueError('invalid AQS checkpoint binding')
    recipe_bytes = _canonical(base_recipe)
    if hashlib.sha256(recipe_bytes).hexdigest() != base_identity['recipe_sha256']:
        raise ValueError('base recipe identity mismatch')
    payload = _bound_payload(base_recipe, aqs, id_mapping_sha256)
    result = dict(base_identity)
    result['recipe_sha256'] = hashlib.sha256(_canonical(payload)).hexdigest()
    return result


@dataclass(frozen=True, init=False)
class CheckpointBinding:
    """Store canonical bytes so nested dictionary mutations cannot alter identity."""
    _identity_bytes: bytes
    _recipe_bytes: bytes
    id_mapping_sha256: str

    def __init__(self, base_identity, base_recipe, id_mapping_sha256):
        _identity(base_identity)
        _digest(id_mapping_sha256)
        if not isinstance(base_recipe, dict):
            raise ValueError('base recipe must be a dictionary')
        recipe = _canonical(base_recipe)
        if hashlib.sha256(recipe).hexdigest() != base_identity['recipe_sha256']:
            raise ValueError('base recipe identity mismatch')
        object.__setattr__(self, '_identity_bytes', _canonical(base_identity))
        object.__setattr__(self, '_recipe_bytes', recipe)
        object.__setattr__(self, 'id_mapping_sha256', id_mapping_sha256)

    @property
    def base_identity(self):
        return strict_json_loads(self._identity_bytes.decode('utf-8'))

    @property
    def base_recipe(self):
        return strict_json_loads(self._recipe_bytes.decode('utf-8'))


def _model_identity(model, binding):
    if not isinstance(binding, CheckpointBinding) or model.id_mapping_sha256 != binding.id_mapping_sha256:
        raise ValueError('AQS model ID mapping identity mismatch')
    recipe = binding.base_recipe
    if recipe.get('model') != model_recipe(model) or model.batching != 'unique_video_all_captions_v1':
        raise ValueError('AQS model recipe mismatch')
    identity = bound_identity(binding.base_identity, recipe, model.aqs_config, binding.id_mapping_sha256)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if identity['world_size'] != world:
        raise ValueError('checkpoint world size identity mismatch')
    return identity


def _checkpoint_identity(model, optimizer, binding):
    def prepare():
        _validate_optimizer_binding(model, optimizer)
        return _model_identity(model, binding)
    identity = _collective_call(prepare) if dist.is_initialized() else prepare()
    if dist.is_initialized():
        identities = [None]*dist.get_world_size()
        dist.all_gather_object(identities, identity)
        if any(other != identity for other in identities):
            raise ValueError('checkpoint rank identity mismatch')
    return identity


def _recipe_record(model, binding, identity):
    payload = _bound_payload(binding.base_recipe, model.aqs_config, binding.id_mapping_sha256)
    if hashlib.sha256(_canonical(payload)).hexdigest() != identity['recipe_sha256']:
        raise ValueError('AQS sources or recipe changed during checkpoint transaction')
    return dict(schema_version='uatvr-aqs-recipe-v1', identity=identity, recipe=payload)


def save_aqs_state(path, model, optimizer, binding, progress, store):
    identity = _checkpoint_identity(model, optimizer, binding)
    record = _recipe_record(model, binding, identity)
    path = Path(path)

    def publish_recipe():
        if not dist.is_initialized() or dist.get_rank() == 0:
            safe_path(path); safe_path(path.with_suffix('.aqs.json'))
            if path.parent.parent != store.run_root:
                raise ValueError('AQS checkpoint must share the run-local array store')
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError('AQS checkpoint publication is exclusive')
            _publish_file(path.with_suffix('.aqs.json'), lambda stream: stream.write(_canonical(record)))
    if dist.is_initialized():
        _collective_call(publish_recipe)
    else:
        publish_recipe()
    return save_state(path, model, optimizer, identity, progress, store)


def restore_aqs_state(path, model, optimizer, binding):
    identity = _checkpoint_identity(model, optimizer, binding)
    expected = _recipe_record(model, binding, identity)
    sidecar = Path(path).with_suffix('.aqs.json')

    def verify_recipe():
        safe_path(sidecar)
        with sidecar.open() as stream:
            actual = strict_json_load(stream)
        if actual != expected:
            raise ValueError('AQS checkpoint recipe/provenance mismatch')
    if dist.is_initialized():
        _collective_call(verify_recipe)
    else:
        verify_recipe()
    return restore_state(path, model, optimizer, identity)


def _validate_optimizer_binding(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    supplied = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(supplied) != len(set(supplied)) or set(supplied) != expected:
        raise ValueError('optimizer parameters must exactly match the trainable AQS model')


def aqs_step(model, network, optimizer, batch):
    def prepare():
        if getattr(network, 'module', network) is not model:
            raise ValueError('optimizer step network must wrap the supplied AQS model')
        _validate_optimizer_binding(model, optimizer)
    if dist.is_initialized():
        _collective_call(prepare)
    else:
        prepare()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = network(*batch)
    output['loss'].backward()

    def validate_gradients():
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('nonfinite AQS gradient: '+name)
        intended = [model.clip.text_projection, model.clip.visual.proj,
                    model.core.uncertain_net_text.fc.weight, model.core.pie_net_video.fc.weight,
                    model.core.frame_position_embeddings.weight]
        if any(p.grad is None or p.grad.abs().sum() == 0 for p in intended):
            raise ValueError('intended AQS training parameters lack a nonzero gradient')
    _collective_call(validate_gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        model.clip.logit_scale.clamp_(max=math.log(100))
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite AQS parameter after optimizer step')
    result = {key: float(output[key].detach()) for key in ('loss', 'contrastive', 'mil', 'kl')}
    active = model.aqs_config.active
    result.update(aqs=float(output['aqs'].detach()) if active else 0., aqs_active=float(active))
    result['aqs_weighted'] = model.aqs_config.weight*result['aqs']
    result.update(output['aqs_stats'] if active else dict(mean_negatives=0., mean_difficulty=0., mean_uniqueness=0., empty_rows=0.))
    return result
