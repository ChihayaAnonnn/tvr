"""Fail-closed permission boundary for the separately registered benchmark pilot."""

import re
from pathlib import Path

import yaml

PILOT = "semantic-answerability-pilot-v1"
CONTROL = f"experiments/{PILOT}"
DATA = f"artifacts/{PILOT}"


def assert_pilot_authorized(*, repo_root: Path, registry_path: Path, manifest: dict, operation: str) -> None:
    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = Path(repo_root) / registry
    if (
        manifest["study"] == "dual-gpu-preflight-v1"
        or "dual-gpu-preflight-v1" in registry.parts
        or "dual-gpu-preflight-v1" in Path(manifest["artifact_path"]).parts
    ):
        from .dual_gpu_preflight import validate_manifest

        validate_manifest(Path(repo_root).resolve(), registry, manifest)
        return
    root = Path(manifest["artifact_path"])
    relevant = manifest["study"] == PILOT or PILOT in registry.parts or PILOT in root.parts
    if not relevant:
        return
    if operation not in {"reserve", "start"}:
        raise ValueError("unsupported pilot operation")
    repository = Path(repo_root).resolve()
    expected_registry = repository / CONTROL / "registry.jsonl"
    if (
        manifest["study"] != PILOT
        or registry.resolve() != expected_registry
        or manifest["registry_path"] != f"{CONTROL}/registry.jsonl"
        or root.parent.parent.parent.name != PILOT
    ):
        raise ValueError("pilot namespace/control/data identity mismatch")
    study_path = expected_registry.with_name("study.yaml")
    if study_path.is_symlink():
        raise ValueError("pilot study cannot be a symlink")
    study = yaml.safe_load(study_path.read_text())
    if not isinstance(study, dict) or any(
        study.get(k) != v
        for k, v in {
            "study": PILOT,
            "status": "design_approved",
            "control_plane": CONTROL,
            "data_plane": DATA,
            "registry": f"{CONTROL}/registry.jsonl",
        }.items()
    ):
        raise ValueError("pilot study is draft or identity is inconsistent")
    if manifest["test_authorized"] is not False or manifest["stage"] != "baseline":
        raise ValueError("pilot requires sealed test and its registered stage")
    if (
        manifest["dataset"] == "msrvtt"
        and manifest["variant"] in {"source_inventory", "benchmark_compile"}
        and type(manifest["environment"]["gpu_count"]) is int
        and manifest["environment"]["gpu_count"] == 0
    ):
        config = manifest.get("config", {})
        evaluation = config.get("evaluation", {})
        if (
            set(config) != {"model", "loss", "optimizer", "scheduler", "evaluation"}
            or any(config[k] for k in ("model", "loss", "optimizer", "scheduler"))
            or set(evaluation) != {"eval_split", "qualification_scenario"}
            or evaluation["eval_split"] != "val"
            or not isinstance(evaluation["qualification_scenario"], str)
            or re.fullmatch(r"pilot_prepare:[0-9a-f]{64}", evaluation["qualification_scenario"]) is None
        ):
            raise ValueError("pilot preparation identity cannot carry scientific configuration")
        return
    from .pilot_protocol import assert_phase_authorized

    assert_phase_authorized(repository, manifest, operation)
