"""Fail-closed permission boundary for the separately registered benchmark pilot."""

from pathlib import Path

import yaml

PILOT = "semantic-answerability-pilot-v1"
CONTROL = f"experiments/{PILOT}"
DATA = f"artifacts/{PILOT}"


def assert_pilot_authorized(*, repo_root: Path, registry_path: Path, manifest: dict, operation: str) -> None:
    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = Path(repo_root) / registry
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
    # Scientific gates are intentionally not implemented by C1. YAML booleans
    # cannot turn CPU preparation permission into scientific permission.
    if (
        manifest["dataset"] != "msrvtt"
        or manifest["stage"] != "baseline"
        or manifest["variant"] not in {"source_inventory", "benchmark_compile"}
        or type(manifest["environment"]["gpu_count"]) is not int
        or manifest["environment"]["gpu_count"] != 0
        or manifest["test_authorized"] is not False
    ):
        raise ValueError("pilot authorizes only sealed-test CPU data preparation")
