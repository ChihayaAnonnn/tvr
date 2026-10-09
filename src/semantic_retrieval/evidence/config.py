"""Versioned explicit configuration for the evidence path."""
from dataclasses import dataclass, fields
from pathlib import Path

from experiment_artifacts import strict_json_load
from semantic_retrieval.config import ModelConfig, RuntimeConfig, config_payload
from semantic_retrieval.models.clip import ClipConfig
from semantic_retrieval.evidence.types import ResourceCounts, positive, require_digest

LOSS_KEYS = {'pc', 'score', 'value'}
ROLES = {'T_match', 'T_value', 'V_development', 'V_calibration', 'V_locked'}


@dataclass(frozen=True)
class InputReference:
    path: Path
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, Path):
            raise ValueError('explicit input Path required')
        require_digest(self.sha256)


@dataclass(frozen=True)
class ViewConfig:
    pool_size: int
    levels: tuple[int, ...]
    preprocessing_sha256: str
    decode_protocol: str

    def __post_init__(self):
        positive(self.pool_size, integer=True)
        if not self.levels or tuple(sorted(set(self.levels))) != self.levels or self.levels[-1] > self.pool_size:
            raise ValueError('invalid nested view levels')
        for x in self.levels:
            positive(x, integer=True)
        require_digest(self.preprocessing_sha256)
        if self.decode_protocol != 'opencv-rgb-v1':
            raise ValueError('unsupported decoder protocol')


@dataclass(frozen=True)
class SamplingConfig:
    train_samples: int
    eval_samples: int
    temperature: float
    seed: int
    protocol: str

    def __post_init__(self):
        positive(self.train_samples, integer=True)
        positive(self.eval_samples, integer=True)
        positive(self.temperature)
        positive(self.seed, integer=True, zero=True)
        if self.protocol != 'identity-normal-v1':
            raise ValueError('unsupported integration protocol')


@dataclass(frozen=True)
class StageTrainConfig:
    learning_rate: float
    weight_decay: float
    seed: int
    scheduler_name: str
    scheduler_steps: int
    checkpoint_interval: int
    trainable_prefixes: tuple[str, ...]
    loss_weights: dict[str, float]

    def __post_init__(self):
        positive(self.learning_rate)
        positive(self.weight_decay, zero=True)
        positive(self.seed, integer=True, zero=True)
        positive(self.scheduler_steps, integer=True)
        positive(self.checkpoint_interval, integer=True)
        if self.scheduler_name not in {'constant', 'linear'} or not self.trainable_prefixes or len(set(self.trainable_prefixes)) != len(self.trainable_prefixes):
            raise ValueError('explicit scheduler and parameter prefixes required')
        if set(self.loss_weights) != LOSS_KEYS:
            raise ValueError('explicit pc/score/value weights required')
        for v in self.loss_weights.values():
            positive(v, zero=True)
        if not any(self.loss_weights.values()):
            raise ValueError('at least one loss must be active')


@dataclass(frozen=True)
class SplitBinding:
    role: str
    manifest: InputReference
    relevance_space: str

    def __post_init__(self):
        if self.role not in ROLES or self.relevance_space not in {'benchmark_annotation', 'reviewed_semantic'}:
            raise ValueError('unknown split role/relevance space')


@dataclass(frozen=True)
class EvidenceModelConfig:
    base: ModelConfig
    variant: str
    scale_floor: float

    def __post_init__(self):
        if not isinstance(self.base, ModelConfig) or self.base.grouping is not None or self.base.alignment is not None or self.base.pilot_arm is not None:
            raise ValueError('evidence model requires base CLIP/temporal/pair configuration')
        if self.variant not in {'gaussian', 'deterministic', 'fixed_scale', 'learned_temperature', 'standard_probabilistic'}:
            raise ValueError('unknown matcher variant')
        positive(self.scale_floor)


@dataclass(frozen=True)
class IndexConfig:
    reference: InputReference | None
    candidate_count: int
    gallery_version: str

    def __post_init__(self):
        positive(self.candidate_count, integer=True)
        if not self.gallery_version:
            raise ValueError('gallery identity required')


@dataclass(frozen=True)
class ParentConfig:
    reference: InputReference | None
    matcher: InputReference | None
    value: InputReference | None
    calibration: InputReference | None


@dataclass(frozen=True)
class TransitionConfig:
    reference: InputReference | None
    producer_sha256: str | None
    state_seed: int
    trajectories: int

    def __post_init__(self):
        positive(self.state_seed, integer=True, zero=True)
        positive(self.trajectories, integer=True)
        if self.producer_sha256 is not None:
            require_digest(self.producer_sha256)


@dataclass(frozen=True)
class ValueConfig:
    hidden_dim: int
    feature_schema: str
    temperature: float

    def __post_init__(self):
        positive(self.hidden_dim, integer=True)
        positive(self.temperature)
        if self.feature_schema != 'observable-action-v1':
            raise ValueError('unknown value feature schema')


@dataclass(frozen=True)
class EvidencePolicyConfig:
    stop_threshold: float
    limits: ResourceCounts
    cost_weights: tuple[float, ...]

    def __post_init__(self):
        positive(self.stop_threshold, zero=True)
        if len(self.cost_weights) != 4:
            raise ValueError('four explicit cost weights required')
        for v in self.cost_weights:
            positive(v)


@dataclass(frozen=True)
class EvidenceResumeConfig:
    checkpoint: InputReference
    cumulative_steps: int

    def __post_init__(self):
        positive(self.cumulative_steps, integer=True, zero=True)


@dataclass(frozen=True)
class EvidenceSystemConfig:
    schema_version: int
    runtime: RuntimeConfig | None
    stage: str
    synthetic: bool
    model: EvidenceModelConfig
    views: ViewConfig
    sampling: SamplingConfig
    train: StageTrainConfig
    splits: tuple[SplitBinding, ...]
    index: IndexConfig
    parents: ParentConfig
    transitions: TransitionConfig
    value: ValueConfig
    policy: EvidencePolicyConfig
    resume: EvidenceResumeConfig | None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or type(self.synthetic) is not bool:
            raise ValueError('invalid config version or synthetic marker')
        if self.stage not in {'cpu_fixture', 'reference', 'matcher', 'transitions', 'value', 'calibration', 'evaluation', 'gpu_qualification'}:
            raise ValueError('unknown evidence stage')
        if self.views.pool_size > self.model.base.max_frames:
            raise ValueError('frame pool exceeds temporal capacity')
        if len({s.role for s in self.splits}) != len(self.splits):
            raise ValueError('duplicate split role')
        if self.model.base.clip.fixture != self.synthetic:
            raise ValueError('synthetic/CLIP fixture mismatch')
        required = {'matcher': ('reference',), 'transitions': ('matcher',), 'value': ('matcher',), 'calibration': ('matcher', 'value'), 'evaluation': ('matcher', 'value', 'calibration')}.get(self.stage, ())
        if any(getattr(self.parents, k) is None for k in required):
            raise ValueError('missing stage parent model')
        if self.stage in {'value', 'calibration'} and (self.transitions.reference is None or self.transitions.producer_sha256 is None):
            raise ValueError('bound transitions required')
        if not self.synthetic and {s.role for s in self.splits} != ROLES:
            raise ValueError('all disjoint source-role manifests required for science')
        if self.stage in {'cpu_fixture', 'gpu_qualification'} and not self.synthetic:
            raise ValueError('qualification is synthetic software only')

    def to_payload(self):
        return config_payload(self)


def load_evidence_config(path: Path):
    with path.open() as handle:
        payload = strict_json_load(handle)

    def construct(cls, values):
        if not isinstance(values, dict) or set(values) != {f.name for f in fields(cls)}:
            raise ValueError(f'invalid {cls.__name__} fields')
        try:
            return cls(**values)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f'malformed {cls.__name__}') from exc

    def local(raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError('explicit path required')
        p = Path(raw)
        return (p if p.is_absolute() else path.parent/p).resolve()

    def ref(raw):
        if raw is None:
            return None
        raw = dict(raw)
        raw['path'] = local(raw['path'])
        return construct(InputReference, raw)

    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(EvidenceSystemConfig)}:
        raise ValueError('invalid evidence system fields')
    p = dict(payload)
    if p['runtime'] is not None:
        r = dict(p['runtime'])
        for name in ('study_path', 'registry_path', 'run_dir'):
            r[name] = local(r[name])
        p['runtime'] = construct(RuntimeConfig, r)
    m = dict(p['model']); b = dict(m['base'])
    b['clip'] = construct(ClipConfig, b['clip']); b['clip_weights'] = local(b['clip_weights'])
    m['base'] = construct(ModelConfig, b); p['model'] = construct(EvidenceModelConfig, m)
    for name, cls, seq in [('views', ViewConfig, 'levels'), ('train', StageTrainConfig, 'trainable_prefixes'), ('policy', EvidencePolicyConfig, 'cost_weights')]:
        r = dict(p[name]); r[seq] = tuple(r[seq])
        if name == 'policy':
            r['limits'] = construct(ResourceCounts, r['limits'])
        p[name] = construct(cls, r)
    p['sampling'] = construct(SamplingConfig, p['sampling'])
    p['value'] = construct(ValueConfig, p['value'])
    p['parents'] = construct(ParentConfig, {k: ref(v) for k, v in p['parents'].items()})
    for name, cls in [('index', IndexConfig), ('transitions', TransitionConfig)]:
        r = dict(p[name]); r['reference'] = ref(r['reference']); p[name] = construct(cls, r)
    p['splits'] = tuple(construct(SplitBinding, {**s, 'manifest': ref(s['manifest'])}) for s in p['splits'])
    if p['resume'] is not None:
        r = dict(p['resume']); r['checkpoint'] = ref(r['checkpoint']); p['resume'] = construct(EvidenceResumeConfig, r)
    return construct(EvidenceSystemConfig, p)


def evidence_manifest_config(config, *, eval_split):
    p = config.to_payload(); t = p['train']
    return dict(model=p['model'], loss=t['loss_weights'],
                optimizer={k: t[k] for k in ('learning_rate', 'weight_decay', 'seed', 'trainable_prefixes')},
                scheduler={k: t[k] for k in ('scheduler_name', 'scheduler_steps', 'checkpoint_interval')},
                evaluation=dict(config_kind='evidence-retrieval-v1', evidence_stage=p['stage'], eval_split=eval_split,
                                **{k: p[k] for k in ('schema_version', 'synthetic', 'views', 'sampling', 'splits', 'index', 'parents', 'transitions', 'value', 'policy', 'resume')}))
