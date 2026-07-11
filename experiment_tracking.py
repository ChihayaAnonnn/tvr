"""Small, dependency-free helpers for trusted experiment provenance.

The tracking sidecar deliberately contains only reproducibility metadata.  It is
kept separate from the training loss path so that telemetry can never affect
autograd or optimizer updates.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
from numbers import Integral
from pathlib import Path

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
    backbone = {
        "type": getattr(args, "backbone_type", ""),
        "pretrained_clip_name": getattr(args, "pretrained_clip_name", ""),
        "clip_layer_norm_precision": getattr(
            args, "clip_layer_norm_precision", "fp16"
        ),
        "clip_gradient_checkpointing": bool(
            getattr(args, "clip_gradient_checkpointing", False)
        ),
        "clip_visual_checkpoint_layers": int(
            getattr(args, "clip_visual_checkpoint_layers", 4)
        ),
        "name": getattr(args, "backbone_name", ""),
        "path": getattr(args, "backbone_path", ""),
    }
    data = {
        "source_train_csv": getattr(args, "source_train_csv", ""),
        "train_csv": getattr(args, "train_csv", ""),
        "val_csv": getattr(args, "val_csv", ""),
        "test_csv": getattr(args, "test_csv", ""),
        "annotation_json": getattr(args, "data_path", ""),
        "split_manifest": getattr(args, "split_manifest", ""),
        "tqfs_cache_dir": getattr(args, "tqfs_cache_dir", ""),
    }
    hard_negative = {
        "packing_enabled": bool(
            getattr(args, "use_hard_negative_packing", False)
        ),
        "explicit_loss_enabled": bool(
            getattr(args, "use_explicit_hard_negative_loss", False)
        ),
        "mapping_path": getattr(args, "hard_negative_path", ""),
        "pack_seed": getattr(args, "hard_negative_pack_seed", None),
        "loss_weight": getattr(args, "w_hard_negative", 0.0),
    }
    return {
        "protocol_version": split.get("protocol_version") if split else None,
        "git": git_state,
        "split": split,
        "seed": getattr(args, "seed", None),
        "profile": getattr(args, "experiment_profile", "default"),
        "backbone": backbone,
        "data": data,
        "batch": batch_semantics,
        "hard_negative": hard_negative,
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
