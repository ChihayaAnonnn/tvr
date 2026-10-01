"""Explicit run identity; no legacy model defaults or output overrides."""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

from semantic_retrieval.models.clip import ClipConfig


@dataclass(frozen=True)
class RuntimeConfig:
    study_path: Path
    registry_path: Path
    run_dir: Path
    run_id: str
    manifest_sha256: str
    source_sha: str
    eval_split: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_sha):
            raise ValueError("source_sha must be an exact Git SHA")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_id):
            raise ValueError("run_id must be a safe nonempty identity")
        if self.eval_split not in {"train", "val", "val_development", "val_calibration", "val_locked", "test"}:
            raise ValueError("unknown eval_split")
        for name in ("study_path", "registry_path", "run_dir"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a Path")


def load_runtime_config(path: Path) -> RuntimeConfig:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(RuntimeConfig)}:
        raise ValueError("runtime configuration must contain exactly the declared keys")
    for name in ("study_path", "registry_path", "run_dir"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"{name} must be a nonempty path")
        value = Path(payload[name])
        payload[name] = (value if value.is_absolute() else path.parent / value).resolve()
    try:
        return RuntimeConfig(**payload)
    except (TypeError, AttributeError) as exc:
        raise ValueError("malformed runtime configuration") from exc


@dataclass(frozen=True)
class ModelConfig:
    clip: ClipConfig
    clip_weights: Path
    clip_sha256: str
    dim: int
    num_layers: int
    num_heads: int
    max_frames: int
    coarse_dim: int
    grouping: "GroupingConfig | None" = None
    alignment: "AlignmentConfig | None" = None

    def __post_init__(self):
        if not isinstance(self.clip, ClipConfig) or not isinstance(self.clip_weights, Path):
            raise ValueError("explicit CLIP configuration and local weights required")
        if not re.fullmatch("[0-9a-f]{64}", self.clip_sha256):
            raise ValueError("invalid CLIP hash")
        if (
            any(type(x) is not int or x <= 0 for x in (self.dim, self.num_heads, self.max_frames, self.coarse_dim))
            or type(self.num_layers) is not int
            or self.num_layers < 0
        ):
            raise ValueError("invalid model capacities")
        if self.dim != self.clip.embed_dim or self.dim % self.num_heads:
            raise ValueError("incompatible representation dimensions")


@dataclass(frozen=True)
class PolicyConfig:
    pair_threshold: float
    presence_threshold: float
    k_sequence: tuple[int, ...]
    max_budget: int

    def __post_init__(self):
        import math

        if any(
            not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1
            for x in (self.pair_threshold, self.presence_threshold)
        ):
            raise ValueError("invalid policy thresholds")
        if (
            not isinstance(self.k_sequence, tuple)
            or not self.k_sequence
            or any(type(x) is not int or x <= 0 for x in self.k_sequence)
            or any(a >= b for a, b in zip(self.k_sequence, self.k_sequence[1:]))
            or type(self.max_budget) is not int
            or self.max_budget < self.k_sequence[-1]
        ):
            raise ValueError("invalid K sequence or budget")


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    seed: int
    scheduler_name: str
    scheduler_steps: int
    trainable_prefixes: tuple[str, ...]
    loss_weights: dict[str, float]

    def __post_init__(self):
        import math

        from semantic_retrieval.objectives.retrieval import LOSS_NAMES

        if (
            any(
                not isinstance(x, (int, float)) or not math.isfinite(x) for x in (self.learning_rate, self.weight_decay)
            )
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("invalid optimizer configuration")
        if (
            self.scheduler_name not in {"linear", "constant"}
            or type(self.scheduler_steps) is not int
            or self.scheduler_steps <= 0
        ):
            raise ValueError("explicit scheduler required")
        if (
            not isinstance(self.trainable_prefixes, tuple)
            or not self.trainable_prefixes
            or len(set(self.trainable_prefixes)) != len(self.trainable_prefixes)
            or not all(isinstance(x, str) and x for x in self.trainable_prefixes)
        ):
            raise ValueError("invalid trainable prefixes")
        if (
            set(self.loss_weights) != set(LOSS_NAMES)
            or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in self.loss_weights.values())
            or not any(self.loss_weights.values())
        ):
            raise ValueError("all objective weights must be explicit")


@dataclass(frozen=True)
class DataConfig:
    episode_manifest: Path
    episode_sha256: str
    gallery_version: str
    split: str

    def __post_init__(self):
        if (
            not isinstance(self.episode_manifest, Path)
            or not re.fullmatch("[0-9a-f]{64}", self.episode_sha256)
            or not self.gallery_version
            or self.split not in {"train", "val", "val_development", "val_calibration", "val_locked"}
        ):
            raise ValueError("invalid or sealed data configuration")


@dataclass(frozen=True)
class SystemConfig:
    runtime: RuntimeConfig
    model: ModelConfig
    train: TrainConfig
    policy: PolicyConfig
    data: DataConfig | None = None
    deployment: "DeploymentConfig | None" = None


def config_payload(config):
    from dataclasses import asdict

    def normalize(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: normalize(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [normalize(v) for v in x]
        return x

    return normalize(asdict(config))


def effective_manifest_config(config, *, eval_split):
    payload = config_payload(config)
    t = payload["train"]
    return dict(
        model=payload["model"],
        loss=t["loss_weights"],
        optimizer={k: t[k] for k in ("learning_rate", "weight_decay", "seed", "trainable_prefixes")},
        scheduler={k: t[k] for k in ("scheduler_name", "scheduler_steps")},
        evaluation=dict(
            eval_split=eval_split, policy=payload["policy"], data=payload["data"], deployment=payload["deployment"]
        ),
    )


def load_system_config(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"runtime", "model", "train", "policy", "data", "deployment"}:
        raise ValueError("system requires explicit runtime/model/train/policy/data")

    def construct(cls, values):
        if not isinstance(values, dict) or set(values) != {f.name for f in fields(cls)}:
            raise ValueError(f"invalid {cls.__name__} fields")
        return cls(**values)

    def local(raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("explicit path required")
        p = Path(raw)
        return (p if p.is_absolute() else path.parent / p).resolve()

    rt = payload["runtime"].copy()
    for name in ("study_path", "registry_path", "run_dir"):
        rt[name] = local(rt[name])
    m = payload["model"].copy()
    m["clip"] = construct(ClipConfig, m["clip"])
    m["clip_weights"] = local(m["clip_weights"])
    if m["grouping"] is not None:
        group = m["grouping"].copy()
        group["validation_report"] = local(group["validation_report"])
        group["annotation_provenance_sha256"] = tuple(group["annotation_provenance_sha256"])
        m["grouping"] = construct(GroupingConfig, group)
    if m["alignment"] is not None:
        m["alignment"] = construct(AlignmentConfig, m["alignment"])
    t = payload["train"].copy()
    t["trainable_prefixes"] = tuple(t["trainable_prefixes"])
    p = payload["policy"].copy()
    p["k_sequence"] = tuple(p["k_sequence"])
    d = payload["data"].copy()
    d["episode_manifest"] = local(d["episode_manifest"])
    dep = payload["deployment"]
    if dep is not None:
        dep = dep.copy()
        for name in ("checkpoint_path", "gallery_manifest", "calibration_path"):
            if dep[name] is not None:
                dep[name] = local(dep[name])
        dep = construct(DeploymentConfig, dep)
    return SystemConfig(
        construct(RuntimeConfig, rt),
        construct(ModelConfig, m),
        construct(TrainConfig, t),
        construct(PolicyConfig, p),
        construct(DataConfig, d),
        dep,
    )


@dataclass(frozen=True)
class DeploymentConfig:
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_config_sha256: str
    gallery_manifest: Path
    gallery_sha256: str
    calibration_path: Path | None
    calibration_sha256: str | None
    calibration_split_sha256: str
    mode: str

    def __post_init__(self):
        if (
            not isinstance(self.checkpoint_path, Path)
            or not isinstance(self.gallery_manifest, Path)
            or self.mode not in {"calibrate", "evaluate"}
        ):
            raise ValueError("invalid deployment configuration")
        for digest in (
            self.checkpoint_sha256,
            self.checkpoint_config_sha256,
            self.gallery_sha256,
            self.calibration_split_sha256,
        ):
            if not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("invalid deployment hash")
        if self.mode == "evaluate" and (
            not isinstance(self.calibration_path, Path)
            or not isinstance(self.calibration_sha256, str)
            or not re.fullmatch("[0-9a-f]{64}", self.calibration_sha256)
        ):
            raise ValueError("frozen calibration input required")
        if self.mode == "calibrate" and (self.calibration_path is not None or self.calibration_sha256 is not None):
            raise ValueError("calibration fit must not consume prior calibration")


@dataclass(frozen=True)
class GroupingConfig:
    threshold: float
    validation_report: Path
    validation_report_sha256: str
    annotation_provenance_sha256: tuple[str, ...]

    def __post_init__(self):
        import math

        if (
            not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
            or not isinstance(self.validation_report, Path)
            or not re.fullmatch("[0-9a-f]{64}", self.validation_report_sha256)
            or not self.annotation_provenance_sha256
            or not all(re.fullmatch("[0-9a-f]{64}", x) for x in self.annotation_provenance_sha256)
        ):
            raise ValueError("explicit grouping validation and annotation provenance required")


@dataclass(frozen=True)
class AlignmentConfig:
    epsilon: float
    dustbin_cost: float
    max_iterations: int
    tolerance: float
    warmup_steps: int

    def __post_init__(self):
        import math

        if (
            any(
                not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (self.epsilon, self.dustbin_cost, self.tolerance)
            )
            or self.epsilon <= 0
            or self.dustbin_cost < 0
            or not 0 < self.tolerance < 1
            or type(self.max_iterations) is not int
            or self.max_iterations <= 0
            or type(self.warmup_steps) is not int
            or self.warmup_steps < 0
        ):
            raise ValueError("invalid explicit alignment solver configuration")
