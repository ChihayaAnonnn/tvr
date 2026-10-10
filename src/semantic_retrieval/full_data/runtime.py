"""Immutable worker settings and exhaustive manifest consumption."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.models.clip import ClipConfig
from semantic_retrieval.runtime.input_bindings import manifest_input_references

KIND = 'full-data-retrieval-v1'


@dataclass(frozen=True)
class ResumeReference:
    parent_run_id: str
    checkpoint: Path
    checkpoint_sha256: str
    sidecar_sha256: str

    def __post_init__(self):
        if (re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', self.parent_run_id) is None
                or not isinstance(self.checkpoint, Path) or not self.checkpoint.is_absolute()
                or self.checkpoint.resolve() != self.checkpoint
                or any(re.fullmatch('[0-9a-f]{64}', h) is None for h in (self.checkpoint_sha256, self.sidecar_sha256))):
            raise ValueError('invalid checkpoint resume binding')


@dataclass(frozen=True)
class FullDataSettings:
    stage: str
    method: str
    clip_weights: Path
    clip_sha256: str
    clip: ClipConfig
    seed: int
    source_report: Path | None = None
    source_report_sha256: str | None = None
    resume: ResumeReference | None = None
    world_size: int | None = None
    memory_limit_gib: int = 60

    def __post_init__(self):
        if self.world_size is None:
            object.__setattr__(self, 'world_size', 1 if self.stage == 'cpu_fixture' else 4)
        if (type(self.world_size) is not int or self.world_size not in ({1} if self.stage == 'cpu_fixture' else {2, 4})
                or type(self.memory_limit_gib) is not int or self.memory_limit_gib != 60):
            raise ValueError('invalid sealed topology/memory limit')
        if (self.stage not in {'cpu_fixture', 'gpu_qualification', 'baseline'}
                or self.method not in {'both', 'uatvr', 'clip4clip'}
                or not isinstance(self.clip_weights, Path) or not self.clip_weights.is_absolute()
                or self.clip_weights.resolve() != self.clip_weights
                or re.fullmatch('[0-9a-f]{64}', self.clip_sha256) is None
                or not isinstance(self.clip, ClipConfig) or type(self.seed) is not int or self.seed < 0):
            raise ValueError('invalid full-data settings')
        if self.stage == 'cpu_fixture':
            expected = ClipConfig(64, 16, 64, 1, 4, 32, 128, 64, 1, 1, fixture=True)
            if self.clip != expected or self.method != 'both':
                raise ValueError('CPU fixture requires exact tiny synthetic capacity and both methods')
        elif self.clip.fixture:
            raise ValueError('GPU qualification/science requires production capacity')
        if self.stage == 'gpu_qualification' and self.method != 'both':
            raise ValueError('qualification must cover both actual methods')
        if self.stage == 'baseline':
            if (self.method == 'both' or not isinstance(self.source_report, Path) or not self.source_report.is_absolute()
                    or self.source_report.resolve() != self.source_report
                    or not isinstance(self.source_report_sha256, str)
                    or re.fullmatch('[0-9a-f]{64}', self.source_report_sha256) is None):
                raise ValueError('scientific baseline requires one method and complete input certification')
        elif self.source_report is not None or self.source_report_sha256 is not None:
            raise ValueError('synthetic qualification cannot consume scientific data')
        if self.resume is not None and (self.stage != 'baseline' or not isinstance(self.resume, ResumeReference)):
            raise ValueError('resume is restricted to explicitly bound scientific checkpoints')

    @property
    def methods(self):
        return ('uatvr', 'clip4clip') if self.method == 'both' else (self.method,)

    @property
    def frames(self):
        return 3 if self.clip.fixture else 12

    @property
    def text_length(self):
        return 8 if self.clip.fixture else 32

    def recipe(self, method):
        if method not in self.methods:
            raise ValueError('method is not part of sealed worker configuration')
        return dict(epochs=5, global_batch=4 if self.clip.fixture else (64 if method == 'uatvr' else 128),
                    learning_rate=5e-5 if method == 'uatvr' else 1e-4)


@dataclass(frozen=True)
class FullDataSystemConfig:
    runtime: RuntimeConfig
    settings: FullDataSettings


def manifest_config(settings):
    source = None if settings.source_report is None else dict(path=str(settings.source_report), sha256=settings.source_report_sha256)
    resume = None if settings.resume is None else dict(parent_run_id=settings.resume.parent_run_id,
        checkpoint=str(settings.resume.checkpoint), checkpoint_sha256=settings.resume.checkpoint_sha256,
        sidecar_sha256=settings.resume.sidecar_sha256)
    return dict(
        model=dict(architecture='full-data-baseline-v1', method=settings.method, clip=asdict(settings.clip),
                   clip_weights=str(settings.clip_weights), clip_sha256=settings.clip_sha256,
                   temporal_layers=1 if settings.clip.fixture else 4, extra_video_tokens=2,
                   extra_text_tokens=2, samples=7, freeze_layer_num=0),
        loss=dict(contrastive='official_symmetric_diagonal', mil_alpha=0.01, vib_beta=0.0001,
                  batching='unique_video_all_captions_v1'),
        optimizer=dict(kind='pinned_bert_adam', beta1=0.9, beta2=0.98, epsilon=1e-6, weight_decay=0.2,
                       clip_coefficient=0.001, gradient_norm=1.0,
                       recipes={method: settings.recipe(method) for method in settings.methods}),
        scheduler=dict(kind='warmup_cosine', warmup=0.1),
        evaluation=dict(config_kind=KIND, full_data_stage=settings.stage, eval_split='val',
                        synthetic=settings.stage != 'baseline', seed=settings.seed, frames=settings.frames,
                        text_length=settings.text_length, source_report=source, dsl=False,
                        checkpoint_selection='earliest_raw_t2v_r1', qualification_steps=3 if settings.stage != 'baseline' else 0,
                        loader_workers=0,
                        world_size=settings.world_size, memory_limit_gib=settings.memory_limit_gib,
                        deterministic_algorithms=True, resume=resume))


def settings_from_manifest(manifest):
    try:
        cfg = manifest['config']; model = cfg['model']; ev = cfg['evaluation']
        source = ev['source_report']
        resumed = ev['resume']
        resume = None if resumed is None else ResumeReference(resumed['parent_run_id'], Path(resumed['checkpoint']),
                                                              resumed['checkpoint_sha256'], resumed['sidecar_sha256'])
        settings = FullDataSettings(ev['full_data_stage'], model['method'], Path(model['clip_weights']), model['clip_sha256'],
                                    ClipConfig(**model['clip']), ev['seed'],
                                    None if source is None else Path(source['path']), None if source is None else source['sha256'], resume,
                                    ev['world_size'], ev['memory_limit_gib'])
        if source is not None and set(source) != {'path', 'sha256'}:
            raise ValueError('source input configuration has unconsumed fields')
        if cfg != manifest_config(settings):
            raise ValueError('full-data effective config has unconsumed or mismatched fields')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed full-data config') from exc
    return settings


def validate_full_data_binding(repository, manifest, system, runtime):
    if system.runtime != runtime or manifest['config'] != manifest_config(system.settings):
        raise ValueError('typed full-data config differs from sealed manifest/runtime')
    refs = manifest_input_references(repository, manifest)
    settings = system.settings
    if refs.get(str(settings.clip_weights)) != settings.clip_sha256:
        raise ValueError('full-data CLIP input is not reserved')
    if settings.source_report is not None and refs.get(str(settings.source_report)) != settings.source_report_sha256:
        raise ValueError('full-data source certification is not reserved')
    if settings.resume is not None:
        ref = settings.resume
        if (refs.get(str(ref.checkpoint)) != ref.checkpoint_sha256
                or refs.get(str(ref.checkpoint.with_suffix('.json'))) != ref.sidecar_sha256):
            raise ValueError('resume checkpoint and final publication marker must both be reserved')
