"""Model-independent Scheme A worker binding; controller owns terminal transitions."""

import datetime
import hashlib
import os
import platform
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from experiment_artifacts import (
    ArtifactArrayStore,
    RunPaths,
    artifact_json_bytes,
    load_validated_run_lifecycle,
    make_run_paths,
    manifest_sha256,
    transition_run_status,
    validate_artifact_json_payload,
    verify_array_reference,
    verify_committed_reservation,
)
from experiment_tracking import atomic_write_json
from semantic_retrieval.config import RuntimeConfig


@dataclass
class RunContext:
    paths: RunPaths
    manifest: dict
    array_store: ArtifactArrayStore
    config: RuntimeConfig
    sequence: int = 0


def _hash_input(repository, reference):
    raw = Path(str(reference["path"]))
    path = raw if raw.is_absolute() else repository / raw
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f"artifact input path is not canonical/private: {raw}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"artifact input is not a private regular file: {raw}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def prepare_run(config, *, repo_root):
    repository = Path(repo_root).resolve()
    root = config.run_dir
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("run directory must be an absolute canonical path")
    paths = RunPaths(
        root=root,
        manifest=root / "manifest.json",
        status=root / "status.json",
        command=root / "command.sh",
        logs=root / "logs",
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        arrays=root / "arrays",
        plots=root / "plots",
        provenance=root / "provenance",
    )
    manifest, status, _ = load_validated_run_lifecycle(paths)
    if status["state"] != "reserved":
        raise ValueError("worker must start from reserved state")
    if (
        manifest["run_id"] != config.run_id
        or manifest_sha256(manifest) != config.manifest_sha256
        or manifest["git_sha"] != config.source_sha
    ):
        raise ValueError("runtime identity differs from committed manifest")
    if (
        Path(manifest["artifact_path"]) != root
        or make_run_paths(root.parents[3], manifest["study"], manifest["run_id"]).root != root
    ):
        raise ValueError("run path differs from canonical manifest path")
    registry = config.registry_path.resolve()
    if (
        not registry.is_relative_to(repository)
        or registry.relative_to(repository).as_posix() != manifest["registry_path"]
    ):
        raise ValueError("runtime registry differs from manifest")
    study = config.study_path.resolve()
    if not study.is_relative_to(repository) or study.parent != registry.parent or not study.is_file():
        raise ValueError("study control file must share the declared registry directory")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(study.relative_to(repository))],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    if "artifact_locator" in manifest:
        common = Path(
            subprocess.run(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if (common.parent / manifest["artifact_locator"]).resolve() != root:
            raise ValueError("artifact locator differs from run path")
    verify_committed_reservation(repository, manifest["registry_path"], manifest)
    sealed = shlex.split(paths.command.read_text())
    if not sealed:
        raise ValueError("sealed command is empty")
    if any("run_train_" in arg or "main_task_retrieval" in arg for arg in sealed):
        raise ValueError("retired launcher cannot start a new worker")
    if config.eval_split == "test" and not manifest["test_authorized"]:
        raise ValueError("manifest does not authorize test evaluation")
    if config.eval_split != "test" and not manifest["validation_authorized"]:
        raise ValueError("manifest does not authorize validation")
    # Stage A has no scientific model consumer: reject unconsumed settings rather than pretending to bind them.
    for category, values in manifest["config"].items():
        if category != "evaluation" and values:
            raise ValueError(f"effective config has no stage-A consumer: {category}")
    evaluation = manifest["config"]["evaluation"]
    if set(evaluation) - {"eval_split", "qualification_scenario"}:
        raise ValueError("effective config has unsupported evaluation fields")
    if evaluation.get("eval_split", config.eval_split) != config.eval_split:
        raise ValueError("effective config eval_split mismatch")
    gpu_count = torch.cuda.device_count()
    environment = dict(
        python=platform.python_version(),
        pytorch=str(torch.__version__),
        cuda=str(torch.version.cuda or "none"),
        gpu_model=torch.cuda.get_device_name(0) if gpu_count else "none",
        gpu_count=gpu_count,
        precision="fp32",
        distributed_topology=f"torchrun-{int(os.environ['WORLD_SIZE'])}"
        if int(os.environ.get("WORLD_SIZE", "1")) > 1
        else "single-process",
    )
    if environment != manifest["environment"]:
        raise ValueError("runtime environment or precision mismatch")
    for field in ("input_checkpoints", "dataset_splits"):
        for reference in manifest[field]:
            if _hash_input(repository, reference) != reference["sha256"]:
                raise ValueError(f"input {field} SHA-256 mismatch")
    return RunContext(paths, manifest, ArtifactArrayStore(paths.arrays), config)


def start_run(context):
    transition_run_status(context.paths, "reserved", "running")


def complete_run(context):
    _, status, _ = load_validated_run_lifecycle(context.paths)
    if status["state"] == "running":
        transition_run_status(context.paths, "running", "validating")
    elif status["state"] != "validating":
        raise ValueError("worker cannot complete from current state")


def fail_run(context, *, exit_code):
    _, status, _ = load_validated_run_lifecycle(context.paths)
    if status["state"] not in {"running", "validating"} or type(exit_code) is not int or exit_code == 0:
        raise ValueError("failure requires an active worker and nonzero integer exit code")
    atomic_write_json(
        context.paths.provenance / "worker_failure.json",
        dict(exit_code=exit_code, recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),
    )


def publish_metrics(context, payload):
    _, status, _ = load_validated_run_lifecycle(context.paths)
    if status["state"] != "running":
        raise ValueError("only a running worker may publish metrics")
    validate_artifact_json_payload(payload)
    for reference in payload.get("array_references", []):
        verify_array_reference(context.paths.root, reference)
    path = context.paths.metrics / f"{context.config.eval_split}_eval_{context.sequence:03d}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(artifact_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    context.sequence += 1
    return path
