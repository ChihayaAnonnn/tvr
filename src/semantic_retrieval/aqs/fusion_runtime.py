"""Strict offline software recipes. Parsing never authorizes or launches a run."""
from dataclasses import dataclass,fields
import re
from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.models.clip import ClipConfig
from .config import AqsConfig
from .fusion_config import FusionConfig


@dataclass(frozen=True)
class FusionRuntimeSpec:
    fusion: FusionConfig
    aqs: AqsConfig
    model_recipe_json: bytes
    sampler_recipe_json: bytes
    source_sha: str
    input_sha256: str
    world_size: int
    dsl: bool
    test_authorized: bool


def _keys(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys): raise ValueError('software recipe fields mismatch')


def _integer(value,minimum=1):
    if type(value) is not int or value<minimum: raise ValueError('invalid software recipe integer')


def fusion_spec_from_recipe(recipe):
    _keys(recipe,('schema','fusion','aqs','model','sampler','identity','dsl','test_authorized'))
    if (recipe['schema']!='uatvr-aqs-kl-software-recipe-v1' or recipe['dsl'] is not False
            or recipe['test_authorized'] is not False): raise ValueError('unsupported software scoring/validation scope')
    fusion=FusionConfig.from_recipe(recipe['fusion']);aqs=AqsConfig.from_recipe(recipe['aqs'])
    if any(type(recipe['aqs'][k]) is not type(v) for k,v in aqs.recipe().items()): raise ValueError('invalid AQS recipe types')
    m=recipe['model']
    _keys(m,('clip','batching','temporal_layers','extra_video_tokens','extra_text_tokens','video_samples','text_samples','trainable_parameters'))
    try:
        _keys(m['clip'],(f.name for f in fields(ClipConfig)))
        clip=ClipConfig(**m['clip'])
    except TypeError as exc: raise ValueError('invalid CLIP recipe') from exc
    if m['batching']!='unique_video_all_captions_v1': raise ValueError('fusion requires unique-video batching')
    for k in ('temporal_layers','extra_video_tokens','extra_text_tokens','video_samples','text_samples'): _integer(m[k])
    if m['temporal_layers']>clip.text_layers or max(m['extra_video_tokens'],m['extra_text_tokens'])>=clip.context_length:
        raise ValueError('invalid semantic adaptation dimensions')
    names=m['trainable_parameters']
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) or re.fullmatch(r'[A-Za-z_][\w.]*',n) is None for n in names)
            or len(set(names))!=len(names)): raise ValueError('invalid trainable parameter names')
    sampler=recipe['sampler'];_keys(sampler,('policy','global_batch','seed','epochs'))
    if sampler['policy']!='unique_video_all_captions_v1': raise ValueError('invalid sampler policy')
    for k in ('global_batch','epochs'): _integer(sampler[k])
    _integer(sampler['seed'],minimum=0)
    identity=recipe['identity'];_keys(identity,('source_sha','input_sha256','world_size'))
    _integer(identity['world_size'])
    for k,size in (('source_sha',40),('input_sha256',64)):
        if not isinstance(identity[k],str) or re.fullmatch('[0-9a-f]{'+str(size)+'}',identity[k]) is None: raise ValueError('invalid source/input identity')
    if sampler['global_batch']%identity['world_size']: raise ValueError('batch cannot shard completely')
    return FusionRuntimeSpec(fusion,aqs,artifact_json_bytes(m),artifact_json_bytes(sampler),
        identity['source_sha'],identity['input_sha256'],identity['world_size'],False,False)
