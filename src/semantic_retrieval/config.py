"""Explicit run identity; no legacy model defaults or output overrides."""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path


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
