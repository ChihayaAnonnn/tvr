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
