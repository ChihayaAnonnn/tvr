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


def _contextual_risk_canonical_a1_checkpoint() -> Path:
    """Resolve ignored checkpoints from a linked worktree's common repository."""

    relative = Path(
        "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
    )
    project_root = Path(__file__).resolve().parent
    local = (project_root / relative).resolve()
    if local.exists():
        return local
    dot_git = project_root / ".git"
    if dot_git.is_file():
        prefix = "gitdir:"
        registration = dot_git.read_text().strip()
        if registration.startswith(prefix):
            git_dir = Path(registration[len(prefix):].strip()).resolve()
            if git_dir.parent.name == "worktrees":
                return (git_dir.parents[2] / relative).resolve()
    return local


CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT = (
    _contextual_risk_canonical_a1_checkpoint()
)
CONTEXTUAL_RISK_CANONICAL_A1_SHA256 = CSD_LOGVAR_CANONICAL_A1_SHA256

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
    """Capture the commit and tracked/index changes relative to ``HEAD``.

    Registered experiments intentionally permit ignored and untracked runtime
    artifacts (logs, checkpoints, smoke outputs).  Only tracked worktree or
    index changes make the source state dirty.
    """

    root = Path(project_root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()
    lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
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
        "datatype": getattr(args, "datatype", "msrvtt"),
        "source_train_csv": getattr(args, "source_train_csv", ""),
        "train_csv": getattr(args, "train_csv", ""),
        "val_csv": getattr(args, "val_csv", ""),
        "test_csv": getattr(args, "test_csv", ""),
        "annotation_json": getattr(args, "data_path", ""),
        "features_path": getattr(args, "features_path", ""),
        "split_manifest": getattr(args, "split_manifest", ""),
        "use_attributes": bool(getattr(args, "use_attributes", False)),
        "msrvtt_attributes_path": getattr(args, "msrvtt_attributes_path", ""),
        "msvd_attributes_path": getattr(args, "msvd_attributes_path", ""),
        "max_words_attrs": getattr(args, "max_words_attrs", None),
        "attr_num_blocks": int(getattr(args, "attr_num_blocks", 4)),
        "expand_msrvtt_sentences": bool(
            getattr(args, "expand_msrvtt_sentences", False)
        ),
    }
    workers = int(getattr(args, "num_thread_reader", 0))
    runtime = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "n_gpu": int(getattr(args, "n_gpu", 1)),
        "world_size": int((batch_semantics or {}).get("world_size", 1)),
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
    if bool(getattr(args, "use_contextual_risk_sidecar", False)):
        payload["contextual_risk"] = _contextual_risk_manifest_section(
            args, split, batch_semantics
        )
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


def _contextual_risk_manifest_section(args, split_summary, batch_semantics) -> dict:
    """Return the immutable registered A3 experiment contract."""

    split = split_summary or {}
    batch = batch_semantics or {}
    split_manifest = Path(getattr(args, "split_manifest", "")).resolve()
    output_dir = Path(getattr(args, "output_dir", "")).resolve()
    return {
        "enabled": True,
        "sidecar_only": bool(
            getattr(args, "contextual_risk_sidecar_only", False)
        ),
        "method": {
            "name": "contextual_retrieval_risk",
            "version": "A3-v2",
        },
        "canonical_a1": {
            "path": str(CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT),
            "sha256": CONTEXTUAL_RISK_CANONICAL_A1_SHA256,
        },
        "architecture": {
            "topk": int(getattr(args, "contextual_risk_topk", 16)),
            "projection_dim": int(
                getattr(args, "contextual_risk_projection_dim", 128)
            ),
            "hidden_dim": int(
                getattr(args, "contextual_risk_hidden_dim", 128)
            ),
            "temperature": float(
                getattr(args, "contextual_risk_temperature", 1.0)
            ),
            "dropout": 0.1,
            "shared_directional_head": True,
        },
        "loss": {
            "weights": {
                "bce": 1.0,
                "fit": float(getattr(args, "w_contextual_risk_fit", 0.25)),
                "rank": float(getattr(args, "w_contextual_risk_rank", 0.25)),
            },
            "positive_weight_bounds": [0.25, 4.0],
            "fit": "smooth_l1_to_robust_rank_normalized_hardness",
            "rank_minimum_hardness_gap": 0.05,
            "hardness_temperature": float(
                getattr(args, "contextual_risk_temperature", 1.0)
            ),
            "bootstrap_samples": int(
                getattr(args, "contextual_risk_bootstrap_samples", 1000)
            ),
        },
        "split_identity": {
            "protocol_version": split.get("protocol_version"),
            "seed": split.get("seed"),
            "manifest_path": str(split_manifest),
            "manifest_sha256": split.get("manifest_sha256"),
        },
        "training": {
            "seed": getattr(args, "seed", None),
            "world_size": int(batch.get("world_size", 0)),
            "precision": "fp32",
            "effective_batch": int(
                batch.get(
                    "optimizer_effective_batch",
                    batch.get("requested_effective_batch", 0),
                )
            ),
            "scheduler_horizon_epochs": 5,
            "optimizer_steps_per_epoch": int(
                batch.get("optimizer_steps_per_epoch", 0)
            ),
            "scheduler_total_optimizer_steps": int(
                batch.get("total_optimizer_steps", 0)
            ),
        },
        "continuation_inputs": {
            "optimizer": {
                "name": "torch.optim.AdamW",
                "lr": float(getattr(args, "lr", 1e-4)),
                "coef_lr": float(getattr(args, "coef_lr", 1.0)),
                "warmup_proportion": float(
                    getattr(args, "warmup_proportion", 0.1)
                ),
                "schedule": "warmup_cosine",
                "weight_decay": 0.2,
                "betas": [0.9, 0.98],
                "epsilon": 1e-6,
                "bias_correction": True,
                "fused": False,
                "no_decay_patterns": [
                    "bias", "LayerNorm.bias", "LayerNorm.weight"
                ],
            },
            "epochs": {
                "fresh_stop_after": 3,
                "scheduler_horizon": 5,
                "continuation_total": 5,
            },
            "batch": {
                "train": int(getattr(args, "batch_size", 256)),
                "validation": int(getattr(args, "batch_size_val", 3500)),
                "gradient_accumulation_steps": int(
                    getattr(args, "gradient_accumulation_steps", 1)
                ),
            },
            "input_shape": {
                "max_words": int(getattr(args, "max_words", 20)),
                "max_frames": int(getattr(args, "max_frames", 100)),
                "feature_framerate": int(
                    getattr(args, "feature_framerate", 1)
                ),
                "slice_framepos": int(getattr(args, "slice_framepos", 0)),
                "train_frame_order": int(
                    getattr(args, "train_frame_order", 0)
                ),
                "eval_frame_order": int(getattr(args, "eval_frame_order", 0)),
            },
            "model": {
                "use_uatvr_full": bool(getattr(args, "use_uatvr_full", False)),
                "cross_model": getattr(args, "cross_model", "cross-base"),
                "linear_patch": getattr(args, "linear_patch", "2d"),
                "sim_header": getattr(args, "sim_header", "meanP"),
                "loose_type": bool(getattr(args, "loose_type", True)),
                "text_num_hidden_layers": int(
                    getattr(args, "text_num_hidden_layers", 12)
                ),
                "visual_num_hidden_layers": int(
                    getattr(args, "visual_num_hidden_layers", 12)
                ),
                "cross_num_hidden_layers": int(
                    getattr(args, "cross_num_hidden_layers", 4)
                ),
                "extra_video_cls_num": int(
                    getattr(args, "extra_video_cls_num", 2)
                ),
                "extra_text_cls_num": int(
                    getattr(args, "extra_text_cls_num", 2)
                ),
                "n_video_embeddings": int(
                    getattr(args, "n_video_embeddings", 7)
                ),
                "n_text_embeddings": int(
                    getattr(args, "n_text_embeddings", 7)
                ),
                "strategy": int(getattr(args, "strategy", 1)),
                "video_dim": int(getattr(args, "video_dim", 1024)),
                "use_ada_norm": bool(getattr(args, "use_ada_norm", False)),
                "disable_spatial_enhancer": bool(
                    getattr(args, "disable_spatial_enhancer", False)
                ),
                "rope_mode": getattr(args, "rope_mode", "2d"),
                "fusion_mode": getattr(args, "fusion_mode", "prob_mos"),
                "fusion_temperature": float(
                    getattr(args, "fusion_temperature", 1.5)
                ),
                "freeze_layer_num": int(
                    getattr(args, "freeze_layer_num", 0)
                ),
            },
            "data_loader": {
                "num_thread_reader": int(
                    getattr(args, "num_thread_reader", 0)
                ),
                "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
            },
            "paired_evaluation": {
                "directions": ["t2v", "v2t"],
                "dsl": bool(getattr(args, "DSL", False)),
                "dsl_temperature": float(
                    getattr(args, "dsl_temperature", 1.0)
                ),
                "score_path": "post_dsl_directional_scores",
                "eval_vid_chunk_size": int(
                    getattr(args, "eval_vid_chunk_size", 128)
                ),
            },
        },
        "thresholds": {
            "directional_rho_min": 0.20,
            "directional_auroc_ci_low_strictly_greater_than": 0.50,
            "directional_risk_std_min": 0.02,
            "selective_accuracy": (
                "monotone_non_increasing_at_25_50_75_100_percent"
            ),
            "paired_invariance_required": True,
            "non_a3_state_mismatches_allowed": 0,
        },
        "selection_rule": (
            "max(min_directional_rho,mean_directional_auroc,"
            "-mean_directional_aurc,-epoch)"
        ),
        "continuation_rule": {
            "initial_epoch_indices": [0, 1, 2],
            "continued_epoch_indices": [3, 4],
            "condition": (
                "selection_tuple(epoch_index_2) strictly_greater_than "
                "selection_tuple(epoch_index_1)"
            ),
            "resume": "exact_epoch_2_model_and_optimizer_same_output_directory",
        },
        "candidate_json": str(
            (output_dir / "contextual_risk_candidates.json").resolve()
        ),
        "paired_baseline_json": str(
            (output_dir / "a3_paired_baseline.json").resolve()
        ),
        "selection_json": str(
            (output_dir / "best_contextual_risk_checkpoint.json").resolve()
        ),
        "final_test_claim_json": str(
            (output_dir / "a3_final_test_claim.json").resolve()
        ),
        "final_test_result_json": str(
            (output_dir / "final_test.json").resolve()
        ),
        "final_test_eligibility": (
            "terminal_lifecycle_and_requested_and_selected_validation_"
            "acceptance_passes"
        ),
    }


def validate_contextual_risk_resume_manifest(existing, requested) -> None:
    """Reject continuation when any immutable A3 registration changed."""

    existing_git = existing.get("git") or {}
    requested_git = requested.get("git") or {}
    if existing_git.get("commit") != requested_git.get("commit"):
        raise ValueError("immutable A3 manifest git commit differs on resume")
    if bool(existing_git.get("dirty", False)) or bool(
        requested_git.get("dirty", False)
    ):
        raise ValueError(
            "immutable A3 manifest requires clean tracked source state"
        )

    batch_keys = (
        "requested_effective_batch",
        "forward_global_contrastive_batch",
        "per_rank_micro_batch",
        "gradient_accumulation_steps",
        "optimizer_effective_batch",
        "forward_steps_per_epoch",
        "optimizer_steps_per_epoch",
        "world_size",
    )
    retrieval_keys = (
        "enabled",
        "sidecar_only",
        "weight",
        "temperature",
        "hidden_dim",
        "bootstrap_samples",
        "dsl",
        "dsl_temperature",
        "score_path",
    )

    def immutable_contract(manifest):
        a3 = manifest.get("contextual_risk")
        retrieval = manifest.get("retrieval_risk")
        if not isinstance(a3, dict) or not isinstance(retrieval, dict):
            raise ValueError("immutable A3 manifest dependencies are missing")
        batch = manifest.get("batch") or {}
        return {
            "protocol_version": manifest.get("protocol_version"),
            "split": manifest.get("split"),
            "seed": manifest.get("seed"),
            "profile": manifest.get("profile"),
            "run_final_test": manifest.get("run_final_test"),
            "backbone": manifest.get("backbone"),
            "data": manifest.get("data"),
            "runtime": manifest.get("runtime"),
            "batch": {key: batch.get(key) for key in batch_keys},
            "retrieval_risk": {
                key: retrieval.get(key) for key in retrieval_keys
            },
            "contextual_risk": a3,
        }

    if immutable_contract(existing) != immutable_contract(requested):
        raise ValueError("immutable A3 manifest fields differ on resume")


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
