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
    parse_registry_text,
    transition_run_status,
    validate_artifact_json_payload,
    verify_array_reference,
    verify_committed_reservation,
)
from experiment_tracking import atomic_write_json
from semantic_retrieval.config import RuntimeConfig, effective_manifest_config
from semantic_retrieval.runtime.input_bindings import manifest_input_references


@dataclass
class RunContext:
    paths: RunPaths
    manifest: dict
    array_store: ArtifactArrayStore
    config: RuntimeConfig
    repo_root: Path
    sequence: int = 0
    system_config: object | None = None
    input_references: dict[str, str] | None = None


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


def _validate_resume(repository, manifest, system_config):
    resume = system_config.resume
    if resume is None:
        return
    if system_config.data is None or system_config.data.split != "train" or system_config.deployment is not None:
        raise ValueError("resume is only supported for training")
    refs = {str(r["path"]): r["sha256"] for r in manifest["input_checkpoints"]}
    if any(
        refs.get(str(path)) != digest
        for path, digest in (
            (resume.checkpoint_path, resume.checkpoint_sha256),
            (resume.checkpoint_path.with_suffix(".json"), resume.metadata_sha256),
        )
    ):
        raise ValueError("resume checkpoint and metadata must be reserved inputs")
    if resume.parent_run_id == manifest["run_id"] or resume.parent_run_id not in manifest["parent_run_ids"]:
        raise ValueError("resume requires a distinct reserved parent run")
    report = resume.review_report_path
    if not report.is_relative_to(repository):
        raise ValueError("resume review report must be Git tracked")
    relative_report = report.relative_to(repository).as_posix()
    refs = {str(r["path"]): r["sha256"] for r in manifest["dataset_splits"]}
    if refs.get(str(report)) != resume.review_report_sha256:
        raise ValueError("resume review report must be a reserved input")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_report], cwd=repository, check=True, capture_output=True
    )
    if not any(
        isinstance(entry, dict)
        and entry.get("kind") == "training_resume"
        and entry.get("parent_run_id") == resume.parent_run_id
        and entry.get("approved") is True
        and entry.get("report_path") == relative_report
        and entry.get("report_sha256") == resume.review_report_sha256
        and isinstance(entry.get("reason"), str)
        and entry["reason"].strip()
        for entry in manifest["selection_lineage"]
    ):
        raise ValueError("resume requires a tracked approved review lineage")
    root = Path(manifest["artifact_path"])
    parent = make_run_paths(root.parents[3], manifest["study"], resume.parent_run_id)
    if resume.checkpoint_path.parent != parent.checkpoints or resume.checkpoint_path.suffix != ".pt":
        raise ValueError("resume checkpoint must belong to its declared parent")
    parent_manifest, status, _ = load_validated_run_lifecycle(parent)
    if status["state"] not in {"failed", "cancelled"}:
        raise ValueError("resume parent must first be finalized failed/cancelled")
    for path, expected in (
        (resume.checkpoint_path, resume.checkpoint_sha256),
        (resume.checkpoint_path.with_suffix(".json"), resume.metadata_sha256),
    ):
        if status["output_hashes"].get(path.relative_to(parent.root).as_posix()) != expected:
            raise ValueError("resume checkpoint differs from terminal output evidence")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{manifest['registry_path']}"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout
    if not any(
        record["run_id"] == resume.parent_run_id
        and record["state"] == status["state"]
        and record["manifest_sha256"] == manifest_sha256(parent_manifest)
        for record in parse_registry_text(committed)
    ):
        raise ValueError("resume parent terminal record must be committed")


def _validate_pilot(repository, manifest, system_config):
    """All candidate and parent inputs belong to the committed reservation."""
    pilot = system_config.train.pilot
    if pilot is None or system_config.deployment is not None:
        return
    from semantic_retrieval.runtime.pilot import validate_pilot_config

    validate_pilot_config(system_config, world_size=int(os.environ.get("WORLD_SIZE", "1")))
    refs = {str(r["path"]): r["sha256"] for field in ("dataset_splits", "input_checkpoints") for r in manifest[field]}
    inputs = [(pilot.candidate_order_path, pilot.candidate_order_sha256)]
    if pilot.parent_checkpoint_path is not None:
        inputs += [
            (pilot.parent_config_path, pilot.parent_config_sha256),
            (pilot.parent_checkpoint_path, pilot.parent_checkpoint_sha256),
            (pilot.parent_checkpoint_path.with_suffix(".json"), pilot.parent_metadata_sha256),
        ]
    if any(refs.get(str(path)) != digest for path, digest in inputs):
        raise ValueError("pilot candidate ordering/parent inputs must be reserved")
    if pilot.parent_run_id is None:
        return
    if pilot.parent_run_id == manifest["run_id"] or pilot.parent_run_id not in manifest["parent_run_ids"]:
        raise ValueError("pilot branch requires a distinct reserved P1 parent run")
    root = Path(manifest["artifact_path"])
    parent = make_run_paths(root.parents[3], manifest["study"], pilot.parent_run_id)
    if pilot.parent_checkpoint_path.parent != parent.checkpoints:
        raise ValueError("P1 checkpoint does not belong to declared parent")
    parent_manifest, status, _ = load_validated_run_lifecycle(parent)
    if status["state"] != "completed":
        raise ValueError("P1 parent must be completed")
    for path, digest in inputs[2:]:
        if status["output_hashes"].get(path.relative_to(parent.root).as_posix()) != digest:
            raise ValueError("P1 input differs from completed parent evidence")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{manifest['registry_path']}"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout
    if not any(
        record["run_id"] == pilot.parent_run_id
        and record["state"] == "completed"
        and record["manifest_sha256"] == manifest_sha256(parent_manifest)
        for record in parse_registry_text(committed)
    ):
        raise ValueError("P1 completed terminal record must be committed")


def prepare_run(config, *, repo_root, system_config=None):
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
    from .pilot_policy import assert_pilot_authorized

    assert_pilot_authorized(repo_root=repository, registry_path=registry, manifest=manifest, operation="start")
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
    if system_config is not None:
        if system_config.runtime != config or manifest["config"] != effective_manifest_config(
            system_config, eval_split=config.eval_split
        ):
            raise ValueError("effective system config mismatch")
        refs = {str(r["path"]): r["sha256"] for r in manifest["input_checkpoints"]}
        if refs.get(str(system_config.model.clip_weights)) != system_config.model.clip_sha256:
            raise ValueError("CLIP input is not reserved")
        if system_config.model.grouping is not None:
            group = system_config.model.grouping
            refs = {str(r["path"]): r["sha256"] for r in manifest["dataset_splits"]}
            if refs.get(str(group.validation_report)) != group.validation_report_sha256:
                raise ValueError("grouping validation report is not reserved")
        if system_config.deployment is not None:
            deployment = system_config.deployment
            refs = {
                str(r["path"]): r["sha256"]
                for field in ("input_checkpoints", "dataset_splits")
                for r in manifest[field]
            }
            for path, digest in (
                (deployment.checkpoint_path, deployment.checkpoint_sha256),
                (deployment.gallery_manifest, deployment.gallery_sha256),
                (deployment.calibration_path, deployment.calibration_sha256),
            ):
                if path is not None and refs.get(str(path)) != digest:
                    raise ValueError("deployment input is not reserved")
        if system_config.data is not None:
            data = system_config.data
            refs = {str(r["path"]): r["sha256"] for r in manifest["dataset_splits"]}
            if refs.get(str(data.episode_manifest)) != data.episode_sha256:
                raise ValueError("episode input is not reserved")
    else:
        # Stage A has no scientific model consumer: reject unconsumed settings rather than pretending to bind them.
        for category, values in manifest["config"].items():
            if category != "evaluation" and values:
                raise ValueError(f"effective config has no stage-A consumer: {category}")
        evaluation = manifest["config"]["evaluation"]
        if set(evaluation) - {"eval_split", "qualification_scenario"}:
            raise ValueError("effective config has unsupported evaluation fields")
        if evaluation.get("eval_split") != config.eval_split:
            raise ValueError("effective config eval_split mismatch")
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError("runtime precision must be FP32 without ambient autocast")
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("runtime precision forbids reduced-precision float32 matmul")
    if torch.cuda.is_available() and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("runtime precision requires CUDA matmul and cuDNN TF32 disabled")
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
    # This runs inside the actual worker after visibility/environment binding;
    # reserve/controller policy checks intentionally stay CPU-safe.
    if (manifest['study'] == 'semantic-answerability-pilot-v1' and gpu_count > 0
            and manifest['variant'] != 'pilot-q0-v2'):
        from types import SimpleNamespace
        from .pilot_protocol import validate_pilot_gpu_qualification
        validate_pilot_gpu_qualification(SimpleNamespace(manifest=manifest,repo_root=repository,config=config))
    references = manifest_input_references(repository, manifest)
    for name, digest in references.items():
        if _hash_input(repository, {"path": name, "sha256": digest}) != digest:
            raise ValueError("input SHA-256 mismatch")
    if system_config is not None:
        _validate_resume(repository, manifest, system_config)
        _validate_pilot(repository, manifest, system_config)
    return RunContext(
        paths,
        manifest,
        ArtifactArrayStore(paths.arrays),
        config,
        repository,
        system_config=system_config,
        input_references=references,
    )


def start_run(context):
    fresh = prepare_run(context.config, repo_root=context.repo_root, system_config=context.system_config)
    if (
        fresh.paths != context.paths
        or fresh.manifest != context.manifest
        or context.array_store.run_root != fresh.paths.root
    ):
        raise ValueError("run context differs from freshly verified launch binding")
    transition_run_status(fresh.paths, "reserved", "running")


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
