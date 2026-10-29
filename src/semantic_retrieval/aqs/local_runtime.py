"""Strict production software recipes; no file access or launch authorization."""
from dataclasses import dataclass, fields
import re
from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.models.clip import ClipConfig
from .config import AqsConfig
from .fusion_config import FusionConfig
from .local_config import LocalQueryConfig


@dataclass(frozen=True)
class LocalRuntimeSpec:
    local: LocalQueryConfig
    fusion: FusionConfig
    aqs: AqsConfig
    recipe_json: bytes
    source_sha: str
    input_sha256: str
    world_size: int
    id_mapping_sha256: str

    @property
    def architecture(self):
        return 'uatvr-aqs-local-query-v1' if self.local.active else 'uatvr-aqs-v1'


def _keys(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('software recipe fields mismatch')


def _integer(value,minimum=1):
    if type(value) is not int or value<minimum:raise ValueError('invalid recipe integer')


def _digest(value,length):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is None:
        raise ValueError('invalid source/input/ID digest')


def local_spec_from_recipe(recipe):
    _keys(recipe,('schema','local','fusion','aqs','model','chunks','sampler','identity',
                 'id_mapping_sha256','initialization','dsl','test_authorized'))
    if (recipe['schema']!='uatvr-aqs-local-query-software-recipe-v1'
            or recipe['dsl'] is not False or recipe['test_authorized'] is not False):
        raise ValueError('unsupported local software scope')
    local=LocalQueryConfig.from_recipe(recipe['local']);fusion=FusionConfig.from_recipe(recipe['fusion'])
    aqs=AqsConfig.from_recipe(recipe['aqs'])
    if any(type(recipe['aqs'][k]) is not type(v) for k,v in aqs.recipe().items()):raise ValueError('invalid AQS recipe types')
    if fusion.enabled or fusion.eta!=0:raise ValueError('local v1 requires KL fusion off')
    model=recipe['model']
    _keys(model,('clip','batching','temporal_layers','extra_video_tokens','extra_text_tokens',
                 'video_samples','text_samples','trainable_parameters'))
    _keys(model['clip'],(f.name for f in fields(ClipConfig)))
    clip=ClipConfig(**model['clip'])
    if clip.fixture:raise ValueError('production software recipe forbids fixture CLIP')
    if model['batching']!='unique_video_all_captions_v1':raise ValueError('invalid local batching')
    for key in ('temporal_layers','extra_video_tokens','extra_text_tokens','video_samples','text_samples'):_integer(model[key])
    if (model['temporal_layers']>clip.text_layers or model['video_samples']!=model['text_samples']
            or model['extra_video_tokens']+12>clip.context_length
            or model['extra_text_tokens']+min(32,clip.context_length)>clip.context_length):
        raise ValueError('invalid UATVR adaptation dimensions')
    names=model['trainable_parameters']
    if (not isinstance(names,list) or not names or any(not isinstance(n,str)
            or re.fullmatch(r'[A-Za-z_][\w.]*',n) is None for n in names) or len(set(names))!=len(names)):
        raise ValueError('invalid trainable parameter names')
    if local.active and not any(n.startswith('local_queries.') for n in names):raise ValueError('missing local parameters')
    if not local.active and any(n.startswith('local_queries.') for n in names):
        raise ValueError('inactive local recipe must describe legacy parameters')
    chunks=recipe['chunks'];_keys(chunks,('query','video'))
    for v in chunks.values():_integer(v)
    sampler=recipe['sampler'];_keys(sampler,('policy','global_batch','seed','epochs'))
    if sampler['policy']!='unique_video_all_captions_v1':raise ValueError('invalid sampler policy')
    for key in ('global_batch','epochs'):_integer(sampler[key])
    _integer(sampler['seed'],0)
    identity=recipe['identity'];_keys(identity,('source_sha','input_sha256','world_size'))
    _integer(identity['world_size']);_digest(identity['source_sha'],40);_digest(identity['input_sha256'],64)
    _digest(recipe['id_mapping_sha256'],64)
    if sampler['global_batch']%identity['world_size']:raise ValueError('batch does not shard equally')
    initialization=recipe['initialization'];_keys(initialization,('parent','resume'))
    if all(initialization[k] is not None for k in initialization):raise ValueError('parent and resume are mutually exclusive')
    for key,architecture in (('parent','uatvr-aqs-v1'),('resume','uatvr-aqs-local-query-v1' if local.active else 'uatvr-aqs-v1')):
        value=initialization[key]
        if value is not None:
            _keys(value,('sha256','architecture'));_digest(value['sha256'],64)
            if value['architecture']!=architecture:raise ValueError('initialization architecture mismatch')
    return LocalRuntimeSpec(local,fusion,aqs,artifact_json_bytes(recipe),identity['source_sha'],
        identity['input_sha256'],identity['world_size'],recipe['id_mapping_sha256'])
