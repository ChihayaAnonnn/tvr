"""Small, dependency-free helpers for trusted experiment provenance.

The tracking sidecar deliberately contains only reproducibility metadata.  It is
kept separate from the training loss path so that telemetry can never affect
autograd or optimizer updates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from numbers import Integral
from pathlib import Path

CSD_LOGVAR_CANONICAL_A1_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
).resolve()
CSD_LOGVAR_CANONICAL_A1_SHA256 = (
    "a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607"
)

_BATCH_STATS = (
    "unique_video_count",
    "duplicate_sample_count",
    "mean_positive_count",
)
_BATCH_PROTOCOL_FIELDS = ("epoch", "forward_step", "global_step", *_BATCH_STATS)


def is_global_rank_zero(args):
    """Return whether *args* belongs to the sole process writing sidecars.

    ``rank`` is populated after distributed initialisation.  A missing rank is
    the single-process compatibility case and therefore defaults to global
    rank zero; ``local_rank`` is intentionally ignored because it repeats on
    every node in multi-node DDP.
    """

    return getattr(args, "rank", 0) == 0


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def compute_batch_semantics(
    requested_effective_batch,
    gradient_accumulation_steps,
    world_size,
    dataloader_steps,
    epochs,
):
    """Return the distinct shell, forward, rank-local, and optimizer batch sizes.

    ``requested_effective_batch`` is the shell value before the historical
    ``args.batch_size`` mutation.  Each forward uses the shell value divided by
    accumulation, while optimizer updates regain the requested effective batch.
    Trusted training uses complete accumulation windows, so a partial final
    window is rejected rather than silently dropped from the accounting.
    """

    requested = _positive_int(requested_effective_batch, "requested effective batch")
    accumulation = _positive_int(gradient_accumulation_steps, "gradient accumulation steps")
    world = _positive_int(world_size, "world size")
    steps = _positive_int(dataloader_steps, "dataloader steps")
    epoch_count = _positive_int(epochs, "epochs")

    if requested % accumulation:
        raise ValueError("requested batch must be divisible by gradient accumulation")
    forward_global = requested // accumulation
    if forward_global % world:
        raise ValueError("forward global batch must be divisible by world size")
    if steps % accumulation:
        raise ValueError("dataloader steps must be divisible by gradient accumulation")

    optimizer_steps = steps // accumulation
    return {
        "requested_effective_batch": requested,
        "forward_global_contrastive_batch": forward_global,
        "per_rank_micro_batch": forward_global // world,
        "gradient_accumulation_steps": accumulation,
        "optimizer_effective_batch": forward_global * accumulation,
        "forward_steps_per_epoch": steps,
        "optimizer_steps_per_epoch": optimizer_steps,
        "total_optimizer_steps": optimizer_steps * epoch_count,
        "world_size": world,
    }


def collect_git_state(project_root):
    """Capture commit and porcelain paths without embedding command-line data."""

    root = Path(project_root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()
    lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).splitlines()
    modified_paths = []
    for line in lines:
        if len(line) >= 4:
            # Porcelain v1 has two status columns followed by a space.
            modified_paths.append(line[3:])
    return {
        "commit": commit,
        "dirty": bool(lines),
        "modified_paths": modified_paths,
    }


def build_experiment_manifest(args, split_summary, batch_semantics, git_state):
    """Build a JSON-serializable provenance payload from explicit safe fields."""

    split = split_summary if split_summary is not None else None
    pretrained_clip_name = getattr(args, "pretrained_clip_name", "")
    backbone = {
        "pretrained_clip_name": pretrained_clip_name,
        "clip_layer_norm_precision": getattr(
            args, "clip_layer_norm_precision", "fp16"
        ),
        "clip_gradient_checkpointing": bool(
            getattr(args, "clip_gradient_checkpointing", False)
        ),
        "clip_visual_checkpoint_layers": int(
            getattr(args, "clip_visual_checkpoint_layers", 4)
        ),
        "name": pretrained_clip_name,
        "path": "",
    }
    data = {
        "source_train_csv": getattr(args, "source_train_csv", ""),
        "train_csv": getattr(args, "train_csv", ""),
        "val_csv": getattr(args, "val_csv", ""),
        "test_csv": getattr(args, "test_csv", ""),
        "annotation_json": getattr(args, "data_path", ""),
        "split_manifest": getattr(args, "split_manifest", ""),
    }
    workers = int(getattr(args, "num_thread_reader", 0))
    runtime = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_dataloader_workers_per_rank": workers,
        "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
        "pin_memory": bool(getattr(args, "pin_memory", False)),
        "persistent_workers": workers > 0,
    }
    payload = {
        "protocol_version": split.get("protocol_version") if split else None,
        "git": git_state,
        "split": split,
        "seed": getattr(args, "seed", None),
        "profile": getattr(args, "experiment_profile", "default"),
        "run_final_test": bool(getattr(args, "run_final_test", False)),
        "backbone": backbone,
        "data": data,
        "batch": batch_semantics,
        "runtime": runtime,
    }
    if bool(getattr(args, "aleatoric_crf_training", False)):
        payload["crf"] = _crf_manifest_section(args)
    if bool(getattr(args, "use_retrieval_risk_sidecar", False)):
        payload["retrieval_risk"] = _retrieval_risk_manifest_section(args)
    if bool(getattr(args, "use_csd_logvar_sidecar", False)):
        payload["csd_logvar"] = _csd_logvar_manifest_section(args)
    return payload


def _retrieval_risk_manifest_section(args) -> dict:
    return {
        "enabled": True,
        "sidecar_only": bool(
            getattr(args, "retrieval_risk_sidecar_only", False)
        ),
        "weight": float(getattr(args, "w_retrieval_risk", 0.0)),
        "temperature": float(
            getattr(args, "retrieval_risk_temperature", 1.0)
        ),
        "hidden_dim": int(getattr(args, "retrieval_risk_hidden_dim", 32)),
        "bootstrap_samples": int(
            getattr(args, "retrieval_risk_bootstrap_samples", 1000)
        ),
        "init_model": getattr(args, "init_model", ""),
        "dsl": bool(getattr(args, "DSL", False)),
        "dsl_temperature": float(getattr(args, "dsl_temperature", 1.0)),
        "score_path": "raw_wti_no_risk_rescaling",
    }


def _csd_logvar_manifest_section(args) -> dict:
    return {
        "enabled": True,
        "sidecar_only": bool(getattr(args, "csd_logvar_sidecar_only", False)),
        "weights": {
            "csd": float(getattr(args, "w_csd", 1.0)),
            "grounding": float(getattr(args, "w_csd_grounding", 0.05)),
            "vib": float(getattr(args, "w_csd_vib", 1e-4)),
        },
        "hidden_dim": int(getattr(args, "csd_logvar_hidden_dim", 128)),
        "bounds": [
            float(getattr(args, "csd_logvar_min", -12.0)),
            float(getattr(args, "csd_logvar_max", 4.0)),
        ],
        "init_logvar": float(getattr(args, "csd_logvar_init", -8.0)),
        "initial_scale": float(getattr(args, "csd_initial_scale", 5.0)),
        "initial_shift": float(getattr(args, "csd_initial_shift", 5.0)),
        "hardness_temperature": float(
            getattr(args, "csd_hardness_temperature", 1.0)
        ),
        "bootstrap_samples": int(getattr(args, "csd_bootstrap_samples", 1000)),
        "run_mode": (
            "resume" if bool(getattr(args, "resume_model", "")) else "fresh"
        ),
        "canonical_a1_checkpoint": str(CSD_LOGVAR_CANONICAL_A1_CHECKPOINT),
        "canonical_a1_sha256": CSD_LOGVAR_CANONICAL_A1_SHA256,
        "init_model": getattr(args, "init_model", ""),
        "resume_model": getattr(args, "resume_model", ""),
        "score_path": "auxiliary_only_raw_wti_unchanged",
    }


def _crf_manifest_section(args) -> dict:
    """Build the CRF-mode section of the experiment manifest.

    Only invoked when ``--aleatoric_crf_training`` is set. Standard hygiene
    runs never gain a ``crf`` key so their manifests remain byte-comparable
    against pre-Task-7 baselines.
    """
    from crf_ladder import TRAIN_CRFS, X264_RECIPE

    ladder_root = Path(getattr(args, "crf_ladder_root", ""))
    ladder_shas: dict[str, str] = {}
    if ladder_root.exists():
        for level_dir in sorted(ladder_root.glob("crf*")):
            manifest_file = level_dir / "manifest.jsonl"
            if manifest_file.exists():
                ladder_shas[level_dir.name] = hashlib.sha256(
                    manifest_file.read_bytes()
                ).hexdigest()

    return {
        "training_crfs": list(TRAIN_CRFS),
        "base_margin": float(getattr(args, "aleatoric_base_margin", 0.05)),
        "order_fraction": float(getattr(args, "crf_order_fraction", 0.20)),
        "lambdas": {
            "cf": float(getattr(args, "aleatoric_lambda_cf", 1.0)),
            "cv": float(getattr(args, "aleatoric_lambda_cv", 1.0)),
            "of": float(getattr(args, "aleatoric_lambda_of", 1.0)),
            "ov": float(getattr(args, "aleatoric_lambda_ov", 1.0)),
        },
        "prior_weight": float(getattr(args, "aleatoric_prior_weight", 0.0)),
        "ranking_weight": float(getattr(args, "aleatoric_ranking_weight", 1.0)),
        "pooling_temperature": float(getattr(args, "aleatoric_pooling_temperature", 0.0)),
        "ladder_root": str(ladder_root),
        "ladder_manifest_sha256_by_level": ladder_shas,
        "x264_recipe": {
            "preset": X264_RECIPE.preset,
            "pixel_format": X264_RECIPE.pixel_format,
            "color_range": X264_RECIPE.color_range,
            "color_primaries": X264_RECIPE.color_primaries,
            "color_trc": X264_RECIPE.color_trc,
            "colorspace": X264_RECIPE.colorspace,
            "time_base_policy": X264_RECIPE.time_base_policy,
            "keyint": X264_RECIPE.keyint,
            "min_keyint": X264_RECIPE.min_keyint,
            "scenecut": X264_RECIPE.scenecut,
            "b_frames": X264_RECIPE.b_frames,
        },
    }


def atomic_write_json(path, payload):
    """Atomically replace *path* with pretty, deterministic JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _finite_number(value, name):
    try:
        number = float(value.item()) if hasattr(value, "item") else float(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def extract_batch_protocol_stats(loss_dict):
    """Extract and validate non-gradient telemetry returned by the model."""

    if not isinstance(loss_dict, dict):
        raise ValueError("loss stats must be returned in a dict")
    missing = [name for name in _BATCH_STATS if name not in loss_dict]
    if missing:
        raise ValueError(f"loss stats missing required keys: {missing}")
    return {
        name: _finite_number(loss_dict[name], f"loss stat {name}")
        for name in _BATCH_STATS
    }


def append_batch_protocol_stats(path, epoch, forward_step, global_step, stats):
    """Append one validated rank-0 telemetry row with an immutable schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(stats, dict):
        raise ValueError("batch stats must be a mapping")
    missing = [name for name in _BATCH_STATS if name not in stats]
    extra = [name for name in stats if name not in _BATCH_STATS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"extra keys: {extra}")
        raise ValueError("batch stats schema mismatch (" + "; ".join(details) + ")")
    values = {
        name: _finite_number(stats[name], f"batch stat {name}")
        for name in _BATCH_STATS
    }
    row = {
        "epoch": _positive_or_zero_int(epoch, "epoch"),
        "forward_step": _positive_or_zero_int(forward_step, "forward step"),
        "global_step": _positive_or_zero_int(global_step, "global step"),
        **values,
    }
    fieldnames = list(_BATCH_PROTOCOL_FIELDS)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                existing_header = next(csv.reader(handle, delimiter="\t"), None)
        except (OSError, csv.Error) as exc:
            raise ValueError(f"unable to read existing batch stats header: {path}") from exc
        if existing_header != fieldnames:
            raise ValueError(
                "batch stats header schema mismatch: "
                f"expected {fieldnames}, got {existing_header}"
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _positive_or_zero_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
