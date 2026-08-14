from __future__ import division, print_function, unicode_literals

import argparse
import copy
import datetime
import json
import math
import os
import random
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.data_dataloaders import DATALOADER_DICT
from dataloaders.msrvtt_protocol import (
    load_trusted_manifest,
    validate_generated_split_files,
    validate_trusted_manifest,
)
from experiment_artifacts import (
    validate_artifact_json_payload,
    verify_array_reference,
)
from experiment_tracking import (
    CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT,
    CONTEXTUAL_RISK_CANONICAL_A1_SHA256,
    CSD_LOGVAR_CANONICAL_A1_CHECKPOINT,
    CSD_LOGVAR_CANONICAL_A1_SHA256,
    REGISTERED_A3_CLIP_ASSET_PATH,
    REGISTERED_A3_CLIP_ASSET_SHA256,
    REGISTERED_A3_CLIP_CACHE_DIR,
    atomic_write_json,
    build_experiment_manifest,
    collect_git_state,
    compute_batch_semantics,
    is_global_rank_zero,
    validate_contextual_risk_resume_manifest,
)
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UATVR as _UATVR_HEAD
from prob_models.csd_logvar_protocol import (
    array_sha256,
    choose_csd_logvar_checkpoint,
    compare_paired_invariance,
    csd_logvar_acceptance,
    file_sha256,
    non_csd_state_dict_mismatches,
    numeric_metrics_are_finite,
    validate_csd_logvar_candidate_artifacts,
    validate_csd_logvar_resume_lineage,
    validate_csd_logvar_selected_checkpoint,
)
from prob_models.uncertainty_protocol import (
    A3_REQUIRED_PAIRED_DIGEST_KEYS,
    append_candidate_atomic,
    choose_contextual_risk_checkpoint,
    compare_paired_invariance as compare_uncertainty_paired_invariance,
    contextual_risk_acceptance,
    contextual_risk_candidate_is_valid,
    contextual_risk_selection_tuple,
    non_sidecar_state_dict_mismatches,
    validate_candidate_artifacts,
    validate_resume_lineage,
    validate_selected_checkpoint,
)
from prob_models.contextual_retrieval_risk import ContextualRetrievalRiskSidecar
from prob_models.retrieval_risk import RetrievalRiskSidecar
from prob_models.retrieval_risk_metrics import (
    build_eval_positive_mask,
    evaluate_intrinsic_uncertainty,
    evaluate_retrieval_risk,
    summarize_csd_components,
    summarize_logvar,
)


def _import_uatvr(use_full: bool):
    """Return the UATVR class matching the requested arch (see --use_uatvr_full).

    HEAD arch (default) is the simplified 2026-08-08 rewrite; June arch
    (``--use_uatvr_full``) restores ``modules.modeling_mulit.UATVR`` with
    SAP + spatial_enhancer + ada_norm text. Ckpts are not interchangeable.
    """
    if use_full:
        from modules.modeling_mulit import UATVR as _UATVR_FULL
        return _UATVR_FULL
    return _UATVR_HEAD


UATVR = _UATVR_HEAD  # module-level default so existing type refs still work
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from util import get_logger, parallel_apply

global logger


def validate_trusted_cli(args):
    """Validate the strict MSRVTT trusted-v1 command-line contract."""
    if args.experiment_profile not in {"default", "hygiene"}:
        raise ValueError(
            "unsupported experiment_profile; expected default or hygiene"
        )

    if args.datatype != "msrvtt":
        return

    if args.do_train and args.eval_split == "test":
        raise ValueError("trusted-v1 training cannot use eval_split=test")
    if args.do_train and not args.expand_msrvtt_sentences:
        raise ValueError(
            "trusted-v1 MSRVTT training requires --expand_msrvtt_sentences"
        )
    if args.run_final_test and not args.do_train:
        raise ValueError("--run_final_test requires --do_train")
    if args.do_eval and not args.do_train and not args.init_model:
        raise ValueError("--do_eval requires --init_model")

    if args.do_train and args.experiment_profile == "hygiene":
        if args.batch_size != 256:
            raise ValueError("hygiene requires --batch_size=256")
        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                "hygiene requires --gradient_accumulation_steps=1"
            )


def get_args(description="CLIP4Clip on Retrieval Task"):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--do_pretrain", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--train_csv", type=str, default="data/.train.csv", help="")
    parser.add_argument("--val_csv", type=str, default="data/.val.csv", help="")
    parser.add_argument("--test_csv", type=str, default="data/.test.csv", help="")
    parser.add_argument(
        "--source_train_csv", type=str, default="data/.source_train.csv", help=""
    )
    parser.add_argument(
        "--split_manifest",
        type=str,
        default="dataloaders/splits/msrvtt_trusted_v1_seed0.json",
        help="Versioned trusted-v1 MSRVTT split manifest.",
    )
    parser.add_argument(
        "--eval_split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split. Training always evaluates internal val.",
    )
    parser.add_argument(
        "--run_final_test",
        action="store_true",
        help=(
            "After training, reload each checkpoint selected on internal val and "
            "evaluate it on the final test split."
        ),
    )
    parser.add_argument("--data_path", type=str, default="data/caption.pickle", help="data pickle file path")
    parser.add_argument("--features_path", type=str, default="data/videos_feature.pickle", help="feature path")

    # =========================
    # System-2 (attributes) text
    # =========================
    parser.add_argument(
        "--use_attributes",
        action="store_true",
        help="If set, load and return attributes text as an additional System-2 input.",
    )
    parser.add_argument(
        "--msrvtt_attributes_path",
        type=str,
        default="",
        help="Path to MSRVTT attributes file. Supports JSON (video_id->text) or JSONL (with fields video_id/attributes).",
    )
    parser.add_argument(
        "--msvd_attributes_path",
        type=str,
        default="",
        help="Path to MSVD attributes file. Supports JSON (video_id->text) or JSONL (with fields video_id/attributes).",
    )
    parser.add_argument(
        "--max_words_attrs",
        type=int,
        default=None,
        help="Max words for attributes branch. If None, fallback to --max_words.",
    )
    parser.add_argument(
        "--attr_num_blocks",
        type=int,
        default=4,
        help="Split Qwen-generated attributes into N blocks (k sequences per sample) for query branch. "
        "Typically 4 blocks: ENTITIES(+APPEARANCE), ACTIONS, SCENE, TEXT/OCR.",
    )

    parser.add_argument("--num_thread_reader", type=int, default=1, help="")
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each DataLoader worker.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--batch_size_val", type=int, default=3500, help="batch size eval")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay")
    parser.add_argument("--n_display", type=int, default=100, help="Information display frequence")
    parser.add_argument(
        "--log_mus_scores",
        action="store_true",
        help="若设置，在每次评估时将每条查询的 MUS（映射不确定性）写入 logs/mus_scores/ 下的 TSV 文件。",
    )
    parser.add_argument("--video_dim", type=int, default=1024, help="video feature dimension")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--max_frames", type=int, default=100, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument("--n_pair", type=int, default=1, help="Num of pair to output from data loader")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Changed in the execute process.")

    parser.add_argument(
        "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    # parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument("--coef_lr", type=float, default=1.0, help="coefficient for bert branch.")
    parser.add_argument("--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text.")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument("--cross_num_hidden_layers", type=int, default=4, help="Layer NO. of cross.")
    parser.add_argument(
        "--loose_type",
        action="store_true",
        default=True,
        help="Use the loose token-wise retrieval path required by UATVR.",
    )

    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")
    parser.add_argument(
        "--train_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )
    parser.add_argument(
        "--eval_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )

    parser.add_argument("--freeze_layer_num", type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument(
        "--slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 3: TQFS 帧质量采样.",
    )
    parser.add_argument(
        "--linear_patch", type=str, default="2d", choices=["2d", "3d"], help="linear projection of flattened patches."
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="meanP",
        choices=["meanP", "seqTransf"],
        help="choice a similarity header.",
    )

    parser.add_argument("--pretrained_clip_name", default="ViT-B/16", type=str, help="Choose an OpenAI CLIP version")
    parser.add_argument(
        "--clip_layer_norm_precision",
        default="fp16",
        choices=["fp16", "fp32"],
        help=(
            "Execution precision for project OpenAI CLIP LayerNorm modules. "
            "fp16 is used only for CUDA FP16 inputs; fp32 preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--clip_gradient_checkpointing",
        action="store_true",
        help=(
            "Checkpoint selected OpenAI CLIP visual Transformer blocks during training "
            "to reduce activation memory without changing the forward contrastive batch."
        ),
    )
    parser.add_argument(
        "--clip_visual_checkpoint_layers",
        type=int,
        default=4,
        help=(
            "Number of leading OpenAI CLIP visual Transformer blocks to checkpoint. "
            "Used only with --clip_gradient_checkpointing."
        ),
    )
    parser.add_argument("--strategy", default=1, type=int, help="Sampling strategies.")
    parser.add_argument("--extra_video_cls_num", default=2, type=int, help="extra video class aggregation token")
    parser.add_argument("--extra_text_cls_num", default=2, type=int, help="extra sentence class aggregation token")
    parser.add_argument(
        "--n_video_embeddings",
        default=7,
        type=int,
        help="Number of probabilistic video embeddings sampled by UATVR.",
    )
    parser.add_argument(
        "--n_text_embeddings",
        default=7,
        type=int,
        help="Number of probabilistic text embeddings sampled by UATVR.",
    )
    parser.add_argument("--DSL", default=False, type=bool, help="whether using dual softmax in post testing")
    parser.add_argument(
        "--dsl_temperature",
        default=1.0,
        type=float,
        help="Dual-softmax temperature; 1.0 preserves the previous default.",
    )
    parser.add_argument(
        "--eval_vid_chunk_size",
        default=128,
        type=int,
        help="Number of videos per chunk during evaluation to limit peak GPU memory. "
             "Smaller value reduces memory at the cost of slightly slower eval.",
    )

    parser.add_argument(
        "--experiment_desc",
        default="",
        type=str,
        help="实验描述，会写入日志便于后续分析。",
    )
    parser.add_argument(
        "--experiment_profile",
        default="default",
        type=str,
        choices=["default", "hygiene"],
        help=(
            "Experiment profile. hygiene selects the clean WTI baseline and "
            "forbids hard-negative diagnostic paths; default permits "
            "independent diagnostics."
        ),
    )

    # ==============================================================
    # CRF degradation ladder training (see docs/superpowers/plans/
    # 2026-08-09-crf-degradation-ladder.md). All defaults are the frozen
    # values from the plan's Global Constraints [G8, G10, G11].
    # ==============================================================
    parser.add_argument(
        "--aleatoric_crf_training",
        action="store_true",
        help="Enable CRF-ladder training: pull the crf_train dataloader profile "
             "and switch train_epoch to model.forward_crf_train.",
    )
    parser.add_argument(
        "--crf_ladder_root",
        type=str,
        default="data/generated/msrvtt_crf",
        help="Root directory containing crfNN/videoXXX.mp4 + manifest.jsonl per level.",
    )
    parser.add_argument(
        "--crf_order_fraction",
        type=float,
        default=0.20,
        help="Fraction of a local batch dedicated to order pairs (plan [G8]).",
    )
    parser.add_argument(
        "--aleatoric_base_margin",
        type=float,
        default=0.05,
        help="Base ranking margin; effective margin = base_margin * ordinal_rank.",
    )
    parser.add_argument("--aleatoric_lambda_cf", type=float, default=1.0)
    parser.add_argument("--aleatoric_lambda_cv", type=float, default=1.0)
    parser.add_argument("--aleatoric_lambda_of", type=float, default=1.0)
    parser.add_argument("--aleatoric_lambda_ov", type=float, default=1.0)
    parser.add_argument(
        "--aleatoric_prior_weight",
        type=float,
        default=0.0,
        help="Weight of variance-prior loss. Plan [G11] fixes this at 0.0 for CRF runs.",
    )
    parser.add_argument(
        "--aleatoric_ranking_weight",
        type=float,
        default=1.0,
        help="Multiplier applied to the total CRF ranking loss.",
    )
    parser.add_argument(
        "--aleatoric_pooling_temperature",
        type=float,
        default=0.0,
        help="Aleatoric aggregation pooling temperature. Plan [G11] fixes this at 0.0.",
    )
    parser.add_argument(
        "--use_uatvr_full",
        action="store_true",
        help="Use restored June UATVR arch (modules.modeling_mulit.UATVR) with SAP + spatial_enhancer + ada_norm text.",
    )
    parser.add_argument(
        "--use_ada_norm",
        action="store_true",
        help="[--use_uatvr_full only] Enable UncertaintyAdaNorm on text side.",
    )
    parser.add_argument(
        "--disable_spatial_enhancer",
        action="store_true",
        help="[--use_uatvr_full only] Disable SpatialEnhancer (default is enabled).",
    )
    parser.add_argument(
        "--rope_mode",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="[--use_uatvr_full only] SpatialEnhancer RoPE mode.",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="prob_mos",
        help="[--use_uatvr_full only] similarity fusion mode.",
    )
    parser.add_argument(
        "--fusion_temperature",
        type=float,
        default=1.5,
        help="[--use_uatvr_full only] similarity fusion temperature.",
    )
    parser.add_argument(
        "--w_query_sim",
        type=float,
        default=0.01,
        help="[--use_uatvr_full only] weight on query-sim aux loss.",
    )
    parser.add_argument(
        "--w_uncertainty_reg",
        type=float,
        default=0.001,
        help="[--use_uatvr_full only] weight on uncertainty regularization.",
    )
    parser.add_argument(
        "--w_evidential",
        type=float,
        default=0.01,
        help="[--use_uatvr_full only] weight on evidential loss.",
    )
    parser.add_argument(
        "--w_neg_reg",
        type=float,
        default=0.01,
        help="[--use_uatvr_full only] weight on negative regularization.",
    )
    parser.add_argument(
        "--uncertainty_mode",
        type=str,
        default="none",
        help="[--use_uatvr_full only] uncertainty training mode.",
    )
    parser.add_argument(
        "--uncertainty_text_head",
        type=str,
        default="text",
        help="[--use_uatvr_full only] which uncertainty head to use on text.",
    )
    parser.add_argument(
        "--log_sigma_min",
        type=float,
        default=-1.5,
        help="[--use_uatvr_full only] clamp for log-sigma min.",
    )
    parser.add_argument(
        "--log_sigma_max",
        type=float,
        default=4.0,
        help="[--use_uatvr_full only] clamp for log-sigma max.",
    )
    parser.add_argument(
        "--anneal_warmup_epochs",
        type=int,
        default=0,
        help="[--use_uatvr_full only] uncertainty annealing warmup epochs.",
    )
    parser.add_argument(
        "--mamba_lr_ratio",
        type=float,
        default=0.1,
        help="[--use_uatvr_full only] lr ratio for mamba params (if used).",
    )
    parser.add_argument(
        "--w_mil",
        type=float,
        default=0.01,
        help="[--use_uatvr_full only] weight on MIL loss.",
    )
    parser.add_argument(
        "--w_orth",
        type=float,
        default=0.1,
        help="[--use_uatvr_full only] weight on orthogonality loss.",
    )
    parser.add_argument(
        "--use_retrieval_risk_sidecar",
        action="store_true",
        help="Train or evaluate the detached retrieval-risk sidecar.",
    )
    parser.add_argument(
        "--retrieval_risk_sidecar_only",
        action="store_true",
        help="Freeze every parameter except retrieval_risk_sidecar.*.",
    )
    parser.add_argument("--w_retrieval_risk", type=float, default=0.0)
    parser.add_argument(
        "--retrieval_risk_temperature", type=float, default=1.0
    )
    parser.add_argument("--retrieval_risk_hidden_dim", type=int, default=32)
    parser.add_argument(
        "--retrieval_risk_bootstrap_samples", type=int, default=1000
    )
    parser.add_argument("--use_csd_logvar_sidecar", action="store_true")
    parser.add_argument("--csd_logvar_sidecar_only", action="store_true")
    parser.add_argument("--w_csd", type=float, default=1.0)
    parser.add_argument("--w_csd_grounding", type=float, default=0.05)
    parser.add_argument("--w_csd_vib", type=float, default=1e-4)
    parser.add_argument("--csd_logvar_hidden_dim", type=int, default=128)
    parser.add_argument("--csd_logvar_min", type=float, default=-12.0)
    parser.add_argument("--csd_logvar_max", type=float, default=4.0)
    parser.add_argument("--csd_logvar_init", type=float, default=-8.0)
    parser.add_argument("--csd_initial_scale", type=float, default=5.0)
    parser.add_argument("--csd_initial_shift", type=float, default=5.0)
    parser.add_argument("--csd_hardness_temperature", type=float, default=1.0)
    parser.add_argument("--csd_bootstrap_samples", type=int, default=1000)
    parser.add_argument("--use_contextual_risk_sidecar", action="store_true")
    parser.add_argument("--contextual_risk_sidecar_only", action="store_true")
    parser.add_argument(
        "--registered_contextual_risk_a3",
        action="store_true",
        help="Assert the complete immutable registered A3 worker contract.",
    )
    parser.add_argument("--registered_clip_asset_path", type=str, default="")
    parser.add_argument("--registered_clip_asset_sha256", type=str, default="")
    parser.add_argument("--contextual_risk_topk", type=int, default=16)
    parser.add_argument("--contextual_risk_projection_dim", type=int, default=128)
    parser.add_argument("--contextual_risk_hidden_dim", type=int, default=128)
    parser.add_argument("--contextual_risk_temperature", type=float, default=1.0)
    parser.add_argument("--contextual_risk_bootstrap_samples", type=int, default=1000)
    parser.add_argument("--w_contextual_risk_fit", type=float, default=0.25)
    parser.add_argument("--w_contextual_risk_rank", type=float, default=0.25)
    args = parser.parse_args()

    if args.use_retrieval_risk_sidecar and not args.use_uatvr_full:
        raise ValueError(
            "use_retrieval_risk_sidecar requires --use_uatvr_full"
        )
    if args.retrieval_risk_sidecar_only and not args.use_retrieval_risk_sidecar:
        raise ValueError(
            "retrieval_risk_sidecar_only requires --use_retrieval_risk_sidecar"
        )
    if args.w_retrieval_risk < 0:
        raise ValueError("w_retrieval_risk must be non-negative")
    if args.retrieval_risk_temperature <= 0:
        raise ValueError("retrieval_risk_temperature must be positive")
    if args.retrieval_risk_hidden_dim <= 0:
        raise ValueError("retrieval_risk_hidden_dim must be positive")
    if args.retrieval_risk_bootstrap_samples < 0:
        raise ValueError("retrieval_risk_bootstrap_samples must be non-negative")
    if args.use_contextual_risk_sidecar and not args.use_uatvr_full:
        raise ValueError("use_contextual_risk_sidecar requires --use_uatvr_full")
    if args.contextual_risk_sidecar_only and not args.use_contextual_risk_sidecar:
        raise ValueError(
            "contextual_risk_sidecar_only requires --use_contextual_risk_sidecar"
        )
    if (
        args.registered_contextual_risk_a3
        and not args.contextual_risk_sidecar_only
    ):
        raise ValueError(
            "registered_contextual_risk_a3 requires "
            "--contextual_risk_sidecar_only"
        )
    if args.contextual_risk_sidecar_only and not args.use_retrieval_risk_sidecar:
        raise ValueError(
            "contextual_risk_sidecar_only requires --use_retrieval_risk_sidecar"
        )
    if (
        args.do_train
        and args.use_contextual_risk_sidecar
        and not args.contextual_risk_sidecar_only
    ):
        raise ValueError(
            "contextual risk training requires --contextual_risk_sidecar_only"
        )
    _validate_exclusive_sidecar_only_modes(args)
    if min(
        args.contextual_risk_topk,
        args.contextual_risk_projection_dim,
        args.contextual_risk_hidden_dim,
    ) <= 0:
        raise ValueError("contextual risk dimensions must be positive")
    if args.contextual_risk_temperature <= 0:
        raise ValueError("contextual_risk_temperature must be positive")
    if min(args.w_contextual_risk_fit, args.w_contextual_risk_rank) < 0:
        raise ValueError("contextual risk loss weights must be non-negative")
    if args.contextual_risk_bootstrap_samples < 0:
        raise ValueError("contextual risk bootstrap samples must be non-negative")
    if args.contextual_risk_sidecar_only and args.do_train:
        if args.fp16:
            raise ValueError("A3-only training requires FP32 and forbids --fp16")
        if args.resume_model:
            if not args.init_model:
                raise ValueError(
                    "A3 continuation requires both --init_model and --resume_model"
                )
            if args.epochs != 5:
                raise ValueError("A3 continuation must target --epochs=5")
        else:
            if not args.init_model or Path(args.init_model).resolve() != (
                CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT
            ):
                raise ValueError(
                    "fresh A3-only runs require --init_model to resolve to "
                    f"{CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT}"
                )
            if args.epochs != 3:
                raise ValueError("fresh A3-only runs require --epochs=3")
    if args.use_csd_logvar_sidecar and not args.use_uatvr_full:
        raise ValueError("use_csd_logvar_sidecar requires --use_uatvr_full")
    if args.csd_logvar_sidecar_only and not args.use_csd_logvar_sidecar:
        raise ValueError("csd_logvar_sidecar_only requires --use_csd_logvar_sidecar")
    if args.csd_logvar_sidecar_only and not args.use_retrieval_risk_sidecar:
        raise ValueError(
            "csd_logvar_sidecar_only requires --use_retrieval_risk_sidecar"
        )
    if (
        args.do_train
        and args.use_csd_logvar_sidecar
        and not args.csd_logvar_sidecar_only
    ):
        raise ValueError(
            "CSD logvar training requires --csd_logvar_sidecar_only"
        )
    if args.csd_logvar_sidecar_only:
        if args.resume_model:
            if not args.init_model:
                raise ValueError(
                    "A2 continuation requires both --init_model and --resume_model"
                )
        elif not args.init_model or Path(args.init_model).resolve() != (
            CSD_LOGVAR_CANONICAL_A1_CHECKPOINT
        ):
            raise ValueError(
                "fresh A2-only runs require --init_model to resolve to "
                f"{CSD_LOGVAR_CANONICAL_A1_CHECKPOINT}"
            )
    if min(args.w_csd, args.w_csd_grounding, args.w_csd_vib) < 0:
        raise ValueError("CSD loss weights must be non-negative")
    if args.use_csd_logvar_sidecar and args.w_csd == 0 and args.w_csd_grounding == 0:
        raise ValueError("CSD or grounding weight must be positive")
    if args.csd_logvar_hidden_dim <= 0:
        raise ValueError("csd_logvar_hidden_dim must be positive")
    if not args.csd_logvar_min < args.csd_logvar_max:
        raise ValueError("csd_logvar_min must be smaller than csd_logvar_max")
    if not args.csd_logvar_min <= args.csd_logvar_init <= args.csd_logvar_max:
        raise ValueError("csd_logvar_init must lie inside the clamp bounds")
    if args.csd_initial_scale <= 0 or args.csd_hardness_temperature <= 0:
        raise ValueError("CSD scales and temperatures must be positive")
    if args.csd_bootstrap_samples < 0:
        raise ValueError("csd_bootstrap_samples must be non-negative")
    if args.dsl_temperature <= 0:
        raise ValueError("dsl_temperature must be positive")

    validate_trusted_cli(args)

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps)
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.clip_visual_checkpoint_layers < 0:
        raise ValueError("--clip_visual_checkpoint_layers must be non-negative")
    if args.num_thread_reader < 0:
        raise ValueError("--num_thread_reader must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive")

    args.requested_effective_batch_size = args.batch_size
    if args.batch_size % args.gradient_accumulation_steps:
        raise ValueError(
            "--batch_size must be divisible by --gradient_accumulation_steps"
        )
    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    return args


def prepare_requested_dataloaders(args, tokenizer):
    """Build only the train/eval loaders required by the requested run mode."""
    loaders = DATALOADER_DICT.get(args.datatype)
    if loaders is None:
        raise ValueError(f"unsupported datatype: {args.datatype}")

    train_bundle = None
    if args.do_train:
        profile = "train"
        if bool(getattr(args, "aleatoric_crf_training", False)):
            profile = "crf_train"
        train_factory = loaders.get(profile)
        if train_factory is None:
            raise ValueError(f"{args.datatype} has no {profile!r} dataloader")
        train_bundle = train_factory(args, tokenizer)

    if args.do_train:
        split = "val"
        factory = loaders.get(split)
        if factory is None and args.datatype != "msrvtt":
            split = "test"
            factory = loaders.get(split)
    else:
        split = args.eval_split
        factory = loaders.get(split)

    if factory is None:
        raise ValueError(f"{args.datatype} has no {split} dataloader")
    eval_loader, eval_length = factory(args, tokenizer, subset=split)
    return train_bundle, eval_loader, eval_length, split


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    # A800: 打开 cuDNN autotune + TF32，让固定 shape 的 conv/matmul 走 Tensor Core。
    # deterministic 关掉，让 autotune 生效；如果日后要复现严格基线，把这两项翻回来即可。
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    world_size = torch.distributed.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        if args.experiment_desc:
            logger.info("    [Experiment] %s", args.experiment_desc)
        # 按类别分组打印关键参数；完整参数由 experiment manifest 记录。
        key_params = {
            "Training": [
                "epochs",
                "batch_size",
                "lr",
                "coef_lr",
                "gradient_accumulation_steps",
                "fp16",
                "max_frames",
                "max_words",
                "seed",
            ],
            "Model": [
                "pretrained_clip_name",
                "clip_layer_norm_precision",
                "clip_gradient_checkpointing",
                "clip_visual_checkpoint_layers",
                "sim_header",
                "loose_type",
                "linear_patch",
                "extra_video_cls_num",
                "extra_text_cls_num",
                "n_video_embeddings",
                "n_text_embeddings",
                "use_attributes",
                "max_words_attrs",
                "attr_num_blocks",
            ],
            "Protocol": [
                "datatype",
                "do_train",
                "do_eval",
                "eval_split",
                "expand_msrvtt_sentences",
                "source_train_csv",
                "test_csv",
                "split_manifest",
                "experiment_profile",
            ],
        }
        for group, keys in key_params.items():
            vals = []
            for k in keys:
                if k in args.__dict__:
                    vals.append(f"{k}={args.__dict__[k]}")
            if vals:
                logger.info("  [%s] %s", group, " | ".join(vals))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    if local_rank == 0:
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu
            )
        )

    return device, n_gpu


def _normalize_registered_a3_model_fp32(model):
    """Normalize and verify the complete registered A3 model in exact FP32."""

    low_precision = (torch.float16, torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype in low_precision:
                parameter.data = parameter.data.to(torch.float32)
        for buffer in model.buffers():
            if buffer.dtype in low_precision:
                buffer.data = buffer.data.to(torch.float32)
    offenders = [
        f"parameter:{name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.dtype in low_precision
    ]
    offenders.extend(
        f"buffer:{name}={buffer.dtype}"
        for name, buffer in model.named_buffers()
        if buffer.dtype in low_precision
    )
    if offenders:
        raise TypeError(
            "registered A3 exact-FP32 model retains low-precision state: "
            + ", ".join(offenders)
        )
    return model


def init_model(args, device, n_gpu, local_rank):
    if bool(getattr(args, "registered_contextual_risk_a3", False)):
        _preflight_registered_a3_clip_asset(args, verify_load=False)
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location="cpu")
        # Older ckpts (e.g. June bin.4) mix fp16 and fp32 params. Fused AdamW
        # requires uniform dtype across params, so cast everything to fp32.
        for k, v in list(model_state_dict.items()):
            if v.dtype in (torch.float16, torch.bfloat16):
                model_state_dict[k] = v.to(torch.float32)
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")

    UATVR_cls = _import_uatvr(getattr(args, "use_uatvr_full", False))
    model = UATVR_cls.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    if bool(getattr(args, "contextual_risk_sidecar_only", False)):
        _normalize_registered_a3_model_fp32(model)
    else:
        # Preserve the historical optimizer compatibility normalization outside
        # registered A3. (CLIP conversion can leave parameters in FP16.)
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.to(torch.float32)
    model.to(device)

    return model


def _trainable_named_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


_SIDECAR_ONLY_MODE_NAMES = (
    "retrieval_risk_sidecar_only",
    "csd_logvar_sidecar_only",
    "contextual_risk_sidecar_only",
    "intrinsic_concentration_only",
)


def _validate_exclusive_sidecar_only_modes(args):
    active = [
        name
        for name in _SIDECAR_ONLY_MODE_NAMES
        if bool(getattr(args, name, False))
    ]
    if len(active) > 1:
        raise ValueError(f"at most one sidecar_only mode may be enabled: {active}")


def _contextual_risk_lifecycle_owner(args):
    """Return whether this process exclusively owns A3 lifecycle artifacts."""

    return is_global_rank_zero(args)


REGISTERED_A3_SCHEDULER_HORIZON_EPOCHS = 5


def _registered_a3_expected_values():
    """Return every material worker value fixed by the A3 registration."""

    project_root = Path(__file__).resolve().parent
    data_root = Path("/data2/hxj/data/MSRVTT")
    generated = project_root / "data/generated/msrvtt_trusted_v1"
    return {
        "registered_contextual_risk_a3": True,
        "do_train": True,
        "do_eval": False,
        "do_pretrain": False,
        "run_final_test": True,
        "datatype": "msrvtt",
        "eval_split": "val",
        "expand_msrvtt_sentences": True,
        "source_train_csv": str(data_root / "csv/MSRVTT_train.9k.csv"),
        "train_csv": str(generated / "train.csv"),
        "val_csv": str(generated / "val.csv"),
        "test_csv": str(data_root / "csv/MSRVTT_JSFUSION_test.csv"),
        "split_manifest": str(
            project_root / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
        ),
        "data_path": str(data_root / "annotation/MSRVTT_v2.json"),
        "features_path": str(
            data_root / "videos/compressed_videos/msrvtt_224_12fps"
        ),
        "use_attributes": False,
        "msrvtt_attributes_path": "",
        "msvd_attributes_path": "",
        "max_words_attrs": None,
        "attr_num_blocks": 4,
        "num_thread_reader": 12,
        "prefetch_factor": 4,
        "lr": 1e-4,
        "lr_decay": 0.9,
        "coef_lr": 1e-3,
        "warmup_proportion": 0.1,
        "batch_size": 256,
        "requested_effective_batch_size": 256,
        "batch_size_val": 16,
        "gradient_accumulation_steps": 1,
        "n_display": 20,
        "log_mus_scores": False,
        "seed": 0,
        "max_words": 32,
        "max_frames": 8,
        "feature_framerate": 1,
        "train_frame_order": 0,
        "eval_frame_order": 0,
        "slice_framepos": 3,
        "video_dim": 1024,
        "n_pair": 1,
        "margin": 0.1,
        "task_type": "retrieval",
        "cross_model": "cross-base",
        "cache_dir": "",
        "do_lower_case": False,
        "linear_patch": "2d",
        "sim_header": "seqTransf",
        "loose_type": True,
        "text_num_hidden_layers": 12,
        "visual_num_hidden_layers": 12,
        "cross_num_hidden_layers": 4,
        "pretrained_clip_name": "ViT-B/16",
        "registered_clip_asset_path": str(REGISTERED_A3_CLIP_ASSET_PATH),
        "registered_clip_asset_sha256": REGISTERED_A3_CLIP_ASSET_SHA256,
        "clip_layer_norm_precision": "fp32",
        "clip_gradient_checkpointing": True,
        "clip_visual_checkpoint_layers": 4,
        "freeze_layer_num": -1,
        "strategy": 1,
        "extra_video_cls_num": 2,
        "extra_text_cls_num": 2,
        "n_video_embeddings": 7,
        "n_text_embeddings": 7,
        "eval_vid_chunk_size": 128,
        "DSL": False,
        "dsl_temperature": 1.0,
        "experiment_profile": "hygiene",
        "experiment_desc": "A3 contextual retrieval-risk sidecar",
        "use_uatvr_full": True,
        "use_ada_norm": True,
        "disable_spatial_enhancer": False,
        "rope_mode": "2d",
        "fusion_mode": "prob_mos",
        "fusion_temperature": 1.5,
        "w_query_sim": 0.01,
        "w_uncertainty_reg": 0.001,
        "w_evidential": 0.01,
        "w_neg_reg": 0.01,
        "uncertainty_mode": "none",
        "uncertainty_text_head": "text",
        "log_sigma_min": -1.5,
        "log_sigma_max": 4.0,
        "anneal_warmup_epochs": 0,
        "mamba_lr_ratio": 0.1,
        "w_mil": 0.01,
        "w_orth": 0.1,
        "use_retrieval_risk_sidecar": True,
        "retrieval_risk_sidecar_only": False,
        "w_retrieval_risk": 0.0,
        "retrieval_risk_temperature": 1.0,
        "retrieval_risk_hidden_dim": 32,
        "retrieval_risk_bootstrap_samples": 1000,
        "use_contextual_risk_sidecar": True,
        "contextual_risk_sidecar_only": True,
        "contextual_risk_topk": 16,
        "contextual_risk_projection_dim": 128,
        "contextual_risk_hidden_dim": 128,
        "contextual_risk_temperature": 1.0,
        "contextual_risk_bootstrap_samples": 1000,
        "w_contextual_risk_fit": 0.25,
        "w_contextual_risk_rank": 0.25,
        "use_csd_logvar_sidecar": False,
        "csd_logvar_sidecar_only": False,
        "intrinsic_concentration_only": False,
        "aleatoric_crf_training": False,
        "crf_ladder_root": "data/generated/msrvtt_crf",
        "crf_order_fraction": 0.2,
        "aleatoric_base_margin": 0.05,
        "aleatoric_lambda_cf": 1.0,
        "aleatoric_lambda_cv": 1.0,
        "aleatoric_lambda_of": 1.0,
        "aleatoric_lambda_ov": 1.0,
        "aleatoric_prior_weight": 0.0,
        "aleatoric_ranking_weight": 1.0,
        "aleatoric_pooling_temperature": 0.0,
        "w_csd": 1.0,
        "w_csd_grounding": 0.05,
        "w_csd_vib": 1e-4,
        "csd_logvar_hidden_dim": 128,
        "csd_logvar_min": -12.0,
        "csd_logvar_max": 4.0,
        "csd_logvar_init": -8.0,
        "csd_initial_scale": 5.0,
        "csd_initial_shift": 5.0,
        "csd_hardness_temperature": 1.0,
        "csd_bootstrap_samples": 1000,
        "fp16": False,
        "fp16_opt_level": "O1",
        "world_size": 4,
        "n_gpu": 4,
        "pin_memory": True,
    }


_REGISTERED_A3_MISSING = object()
_REGISTERED_A3_DYNAMIC_NAMESPACE_FIELDS = {
    # Validated separately because fresh/continuation values differ.
    "epochs",
    "init_model",
    "resume_model",
    # Owned by the registered lifecycle and torchrun topology gates.
    "output_dir",
    "rank",
}


@dataclass(frozen=True)
class _RegisteredA3ContractField:
    name: str
    source: str
    kind: str
    expected: object
    required: bool = True
    missing_default: object = _REGISTERED_A3_MISSING


def _registered_a3_contract_schema():
    """Declare source, type, presence, and value for each normalized A3 field."""

    path_fields = {
        "source_train_csv",
        "train_csv",
        "val_csv",
        "test_csv",
        "split_manifest",
        "data_path",
        "features_path",
    }
    derived_fields = {
        "requested_effective_batch_size",
        "world_size",
        "n_gpu",
        "pin_memory",
    }
    optional_false_fields = {"intrinsic_concentration_only"}
    fields = []
    for name, expected in _registered_a3_expected_values().items():
        if name in path_fields:
            kind = "path"
        elif expected is None:
            kind = "none"
        elif type(expected) is bool:
            kind = "boolean"
        elif type(expected) is int:
            kind = "integer"
        elif type(expected) is float:
            kind = "float"
        else:
            kind = "string"
        optional = name in optional_false_fields
        fields.append(
            _RegisteredA3ContractField(
                name=name,
                source="derived" if name in derived_fields else "namespace",
                kind=kind,
                expected=expected,
                required=not optional,
                missing_default=False if optional else _REGISTERED_A3_MISSING,
            )
        )
    fields.append(
        _RegisteredA3ContractField(
            name="CUDA_VISIBLE_DEVICES",
            source="environment",
            kind="string",
            expected="0,2,3,4",
        )
    )
    return tuple(fields)


def _registered_a3_expected_config():
    """Return registered Namespace/derived values for manifests and diagnostics."""

    return {
        field.name: field.expected
        for field in _registered_a3_contract_schema()
        if field.source != "environment"
    }


def _normalize_registered_a3_contract_value(field, value):
    if field.kind == "boolean":
        if type(value) is not bool:
            raise TypeError("must be a boolean")
        return value
    if field.kind == "integer":
        if type(value) is not int:
            raise TypeError("must be an integer")
        return value
    if field.kind == "float":
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("must be a finite float")
        return value
    if field.kind == "string":
        if type(value) is not str:
            raise TypeError("must be a string")
        return value
    if field.kind == "path":
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise TypeError("must be a non-empty path")
        return str(Path(value).resolve())
    if field.kind == "none":
        if value is not None:
            raise TypeError("must be None")
        return None
    raise RuntimeError(f"unknown registered A3 field kind: {field.kind}")


def _registered_a3_schema_mismatches(args):
    """Extract, normalize, and compare all schema-declared A3 worker values."""

    mismatches = []
    for field in _registered_a3_contract_schema():
        if field.source == "environment":
            actual = os.environ.get(field.name, _REGISTERED_A3_MISSING)
        else:
            actual = getattr(args, field.name, _REGISTERED_A3_MISSING)
        if actual is _REGISTERED_A3_MISSING:
            if field.required:
                mismatches.append(
                    f"{field.name} is required from registered {field.source}"
                )
                continue
            actual = field.missing_default
        try:
            normalized = _normalize_registered_a3_contract_value(field, actual)
            expected = _normalize_registered_a3_contract_value(
                field, field.expected
            )
        except TypeError as exc:
            mismatches.append(f"{field.name}={actual!r} {exc}")
            continue
        if normalized != expected:
            mismatches.append(
                f"{field.name}={normalized!r} (registered {expected!r})"
            )
    return mismatches


def _registered_a3_expected_environment():
    """Return deterministic process values fixed by the registered launcher."""

    return {
        "CUDA_VISIBLE_DEVICES": "0,2,3,4",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONHASHSEED": "0",
        "CLIP_CACHE_DIR": str(REGISTERED_A3_CLIP_CACHE_DIR),
        "REGISTERED_CLIP_ASSET_PATH": str(REGISTERED_A3_CLIP_ASSET_PATH),
        "REGISTERED_CLIP_ASSET_SHA256": REGISTERED_A3_CLIP_ASSET_SHA256,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_IB_DISABLE": "1",
        "NCCL_DEBUG": "WARN",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_WARN_ONCE": "1",
        "MASTER_ADDR": "127.0.0.9",
        "MASTER_PORT": "29547",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4",
        "GROUP_WORLD_SIZE": "1",
        "ROLE_WORLD_SIZE": "4",
        "GROUP_RANK": "0",
        "NPROC": "4",
        "TRAIN_BATCH_SIZE": "256",
        "TRAIN_GRADIENT_ACCUMULATION_STEPS": "1",
        "TRAIN_NUM_WORKERS": "12",
        "TRAIN_PREFETCH_FACTOR": "4",
        "CLIP_LAYER_NORM_PRECISION": "fp32",
        "CLIP_GRADIENT_CHECKPOINTING": "1",
        "CLIP_VISUAL_CHECKPOINT_LAYERS": "4",
        "FREEZE_LAYER_NUM": "-1",
        "A800_THROUGHPUT_COMPARISON": "0",
        "DATA_PATH": "/data2/hxj/data/MSRVTT",
        "EXPERIMENT_PROFILE": "hygiene",
        "EXPERIMENT_DESC": "A3 contextual retrieval-risk sidecar",
        "TVR_PYTHON": "/home/xujie/.conda/envs/tvr/bin/python",
        "TVR_TORCHRUN": "/home/xujie/.conda/envs/tvr/bin/torchrun",
    }


def _preflight_registered_a3_clip_asset(args, *, verify_load):
    """Bind A3 to one exact CLIP file and optionally parse it through CLIP."""

    if not bool(getattr(args, "registered_contextual_risk_a3", False)):
        return None
    cli_path = Path(getattr(args, "registered_clip_asset_path", "")).resolve()
    cli_sha256 = getattr(args, "registered_clip_asset_sha256", "")
    cache_value = os.environ.get("CLIP_CACHE_DIR")
    mismatches = []
    if cli_path != REGISTERED_A3_CLIP_ASSET_PATH:
        mismatches.append(
            f"CLI asset path {cli_path} is not {REGISTERED_A3_CLIP_ASSET_PATH}"
        )
    if cli_sha256 != REGISTERED_A3_CLIP_ASSET_SHA256:
        mismatches.append("CLI SHA-256 is not the registered CLIP SHA-256")
    if not cache_value or Path(cache_value).resolve() != REGISTERED_A3_CLIP_CACHE_DIR:
        mismatches.append(
            f"CLIP_CACHE_DIR={cache_value!r} is not "
            f"{str(REGISTERED_A3_CLIP_CACHE_DIR)!r}"
        )
    env_asset_path = os.environ.get("REGISTERED_CLIP_ASSET_PATH")
    if (
        not env_asset_path
        or Path(env_asset_path).resolve() != REGISTERED_A3_CLIP_ASSET_PATH
    ):
        mismatches.append("REGISTERED_CLIP_ASSET_PATH is not the registered path")
    if os.environ.get("REGISTERED_CLIP_ASSET_SHA256") != (
        REGISTERED_A3_CLIP_ASSET_SHA256
    ):
        mismatches.append("REGISTERED_CLIP_ASSET_SHA256 is not the registered hash")
    if mismatches:
        raise ValueError("registered A3 CLIP preflight mismatch: " + "; ".join(mismatches))
    if not cli_path.is_file():
        raise ValueError(f"registered A3 CLIP asset is missing: {cli_path}")
    actual_sha256 = file_sha256(cli_path)
    if actual_sha256 != REGISTERED_A3_CLIP_ASSET_SHA256:
        raise ValueError("registered A3 CLIP asset SHA-256 mismatch")
    tensor_count = None
    if verify_load:
        from modules.module_clip import CLIP

        state_dict = CLIP.get_config(
            getattr(args, "pretrained_clip_name", "ViT-B/16"),
            model_path=str(cli_path),
        )
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("registered A3 CLIP asset did not load a state dictionary")
        tensor_count = len(state_dict)
    return {
        "cache_dir": str(REGISTERED_A3_CLIP_CACHE_DIR),
        "path": str(cli_path),
        "sha256": actual_sha256,
        "tensor_count": tensor_count,
    }


def _assert_registered_a3_contract(args, batch_semantics=None):
    """Fail closed before optimization for every registered A3 worker."""

    if not (
        bool(getattr(args, "contextual_risk_sidecar_only", False))
        or bool(getattr(args, "registered_contextual_risk_a3", False))
    ):
        return
    mismatches = _registered_a3_schema_mismatches(args)

    continuing = bool(getattr(args, "resume_model", ""))
    expected_epochs = 5 if continuing else 3
    actual_epochs = getattr(args, "epochs", _REGISTERED_A3_MISSING)
    if type(actual_epochs) is not int:
        mismatches.append(f"epochs={actual_epochs!r} must be an integer")
    elif actual_epochs != expected_epochs:
        mismatches.append(
            f"epochs={actual_epochs!r} (registered {expected_epochs!r})"
        )
    if not continuing:
        init_model = getattr(args, "init_model", _REGISTERED_A3_MISSING)
        if not isinstance(init_model, (str, os.PathLike)) or not str(init_model):
            mismatches.append("init_model is required as a non-empty path")
        elif Path(init_model).resolve() != CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT:
            mismatches.append("init_model is not the canonical A1 checkpoint")

    if batch_semantics is not None:
        required_batch = {
            "world_size": 4,
            "requested_effective_batch": 256,
            "optimizer_effective_batch": 256,
            "gradient_accumulation_steps": 1,
        }
        for name, registered in required_batch.items():
            if batch_semantics.get(name) != registered:
                mismatches.append(
                    f"batch.{name}={batch_semantics.get(name)!r} "
                    f"(registered {registered!r})"
                )
        steps_per_epoch = int(batch_semantics.get("optimizer_steps_per_epoch", 0))
        expected_total = steps_per_epoch * REGISTERED_A3_SCHEDULER_HORIZON_EPOCHS
        if steps_per_epoch <= 0 or int(
            batch_semantics.get("total_optimizer_steps", 0)
        ) != expected_total:
            mismatches.append(
                "batch.total_optimizer_steps does not use the registered "
                "five-epoch scheduler horizon"
            )
    if mismatches:
        raise ValueError(
            "registered A3 worker contract mismatch: " + "; ".join(mismatches)
        )


def _assert_registered_a3_initial_environment(args):
    """Check process-start values before seed setup can normalize them."""

    if not bool(getattr(args, "contextual_risk_sidecar_only", False)):
        return
    expected = _registered_a3_expected_environment()
    mismatches = [
        f"{name}={os.environ.get(name)!r} (registered {value!r})"
        for name, value in expected.items()
        if os.environ.get(name) != value
    ]
    forbidden = (
        "SMOKE_STEPS",
        "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER",
        "A3_LAUNCHER_TEST_MODE",
        "A3_BG_CONTROLLER",
        "A3_TEST_OUTPUT_DIR",
        "A3_TEST_OWNERSHIP_TIMEOUT_SECONDS",
    )
    mismatches.extend(
        f"{name} is forbidden in a registered A3 worker"
        for name in forbidden
        if name in os.environ
    )
    torchrun_topology = {
        "ROLE_NAME": "default",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RUN_ID": "none",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
    }
    topology_mismatches = [
        f"{name}={os.environ.get(name)!r} (registered {value!r})"
        for name, value in torchrun_topology.items()
        if os.environ.get(name) != value
    ]
    if topology_mismatches:
        mismatches.append(
            "torchrun worker topology mismatch: " + ", ".join(topology_mismatches)
        )
    allowed_accelerator_environment = {
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
    }
    accelerator_override_names = sorted(
        name
        for name in os.environ
        if (
            name.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_", "CUBLAS_"))
            or name
            in {
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            }
        )
        and name not in allowed_accelerator_environment
    )
    mismatches.extend(
        f"{name} is not allowed in the registered A3 accelerator environment"
        for name in accelerator_override_names
    )
    dynamic_ranks = {}
    for name in ("RANK", "LOCAL_RANK", "ROLE_RANK"):
        try:
            dynamic_ranks[name] = int(os.environ[name])
        except (KeyError, ValueError):
            mismatches.append(f"{name} must be an integer in [0, 3]")
    if len(dynamic_ranks) == 3:
        if any(not 0 <= value < 4 for value in dynamic_ranks.values()):
            mismatches.append("registered A3 rank values must be in [0, 3]")
        if len(set(dynamic_ranks.values())) != 1:
            mismatches.append("RANK, LOCAL_RANK, and ROLE_RANK must agree")
    if mismatches:
        raise ValueError(
            "registered A3 initial environment mismatch: "
            + "; ".join(mismatches)
        )


def _a3_training_autocast_enabled(args):
    """A3's registered training precision is exact FP32."""

    return not bool(getattr(args, "contextual_risk_sidecar_only", False))


@contextmanager
def _a3_exact_fp32_backend(args):
    """Temporarily disable TF32 for registered A3-only computation."""

    enabled = bool(
        getattr(args, "contextual_risk_sidecar_only", False)
        or getattr(args, "use_contextual_risk_sidecar", False)
    )
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if enabled:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        if enabled:
            torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


def configure_retrieval_risk_sidecar_training(model, args):
    """Freeze the model around a retrieval-risk sidecar-only experiment."""

    use_sidecar = bool(getattr(args, "use_retrieval_risk_sidecar", False))
    sidecar_only = bool(getattr(args, "retrieval_risk_sidecar_only", False))
    if sidecar_only and not use_sidecar:
        raise ValueError(
            "retrieval_risk_sidecar_only requires use_retrieval_risk_sidecar"
        )
    if not sidecar_only:
        return
    if not hasattr(model, "retrieval_risk_sidecar"):
        raise ValueError("model does not expose retrieval_risk_sidecar")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.retrieval_risk_sidecar.parameters():
        parameter.requires_grad = True


def set_retrieval_risk_training_mode(model):
    """Keep a frozen backbone deterministic while the sidecar trains."""

    inner = model.module if hasattr(model, "module") else model
    if not getattr(inner, "retrieval_risk_sidecar_only", False):
        return
    model.eval()
    # UATVR.forward uses the root training flag to enter its loss path. Restore
    # only the roots after eval() has put every frozen child in inference mode.
    model.training = True
    inner.training = True
    inner.retrieval_risk_sidecar.train()


def configure_csd_logvar_sidecar_training(model, args):
    """Freeze every parameter except the CSD log-variance sidecar."""

    enabled = bool(getattr(args, "use_csd_logvar_sidecar", False))
    sidecar_only = bool(getattr(args, "csd_logvar_sidecar_only", False))
    if sidecar_only and not enabled:
        raise ValueError("csd_logvar_sidecar_only requires use_csd_logvar_sidecar")
    if not sidecar_only:
        return
    if not hasattr(model, "csd_logvar_sidecar"):
        raise ValueError("model does not expose csd_logvar_sidecar")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.csd_logvar_sidecar.parameters():
        parameter.requires_grad = True

    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("csd_logvar_sidecar.")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected A2 trainable parameters: {unexpected}")
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def configure_contextual_risk_sidecar_training(model, args):
    """Freeze every parameter except the contextual-risk sidecar."""

    _validate_exclusive_sidecar_only_modes(args)
    enabled = bool(getattr(args, "use_contextual_risk_sidecar", False))
    sidecar_only = bool(getattr(args, "contextual_risk_sidecar_only", False))
    if sidecar_only and not enabled:
        raise ValueError(
            "contextual_risk_sidecar_only requires use_contextual_risk_sidecar"
        )
    if not sidecar_only:
        return
    if not hasattr(model, "contextual_risk_sidecar"):
        raise ValueError("model does not expose contextual_risk_sidecar")

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("contextual_risk_sidecar."):
            parameter.requires_grad = True

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, _ in trainable
        if not name.startswith("contextual_risk_sidecar.")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected A3 trainable parameters: {unexpected}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    if trainable_count == 0:
        raise RuntimeError("A3 sidecar-only mode has zero trainable parameters")
    return trainable_count


def _set_contextual_risk_training_mode(model):
    """Keep only the A3 sidecar in module training mode."""

    inner = model.module if hasattr(model, "module") else model
    if not getattr(inner, "contextual_risk_sidecar_only", False):
        return
    model.eval()
    model.training = True
    inner.training = True
    inner.contextual_risk_sidecar.train()


def set_csd_logvar_training_mode(model):
    """Keep only the A2 sidecar stochastic-training modules enabled."""

    inner = model.module if hasattr(model, "module") else model
    if not getattr(inner, "csd_logvar_sidecar_only", False):
        return
    model.eval()
    model.training = True
    inner.training = True
    inner.csd_logvar_sidecar.train()


def _should_keep_clip_parameter_trainable(name, args):
    if (
        name.find("ln_final.") == 0
        or name.find("text_projection") == 0
        or name.find("logit_scale") == 0
        or name.find("visual.ln_post.") == 0
        or name.find("visual.proj") == 0
        or name.find("visual.norm.") == 0
        or name.find("visual.fc_norm.") == 0
        or name.find("visual.head.") == 0
    ):
        return True

    if name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
        return layer_num >= args.freeze_layer_num

    if name.find("visual.blocks.") == 0:
        layer_num = int(name.split(".blocks.")[1].split(".")[0])
        return layer_num >= args.freeze_layer_num

    if args.linear_patch == "3d" and name.find("conv2."):
        return True

    return False


def _build_warmup_cosine_scheduler(
    optimizer, *, warmup_proportion, total_steps
):
    """Build the registered scheduler with an immutable step horizon."""

    warmup_frac = float(warmup_proportion)
    t_total = max(1, int(total_steps))

    def _warmup_cosine_lambda(step):
        x = step / t_total
        if warmup_frac > 0 and x < warmup_frac:
            return x / warmup_frac
        return 0.5 * (1.0 + math.cos(math.pi * x))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup_cosine_lambda
    )


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.0):
    if hasattr(model, "module"):
        model = model.module

    param_optimizer = _trainable_named_parameters(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_other_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]
    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_other_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": weight_decay,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for _, p in no_decay_clip_param_tp],
            "weight_decay": 0.0,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for _, p in decay_other_param_tp],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for _, p in no_decay_other_param_tp],
            "weight_decay": 0.0,
        },
    ]

    if local_rank == 0:
        n_clip = len(decay_clip_param_tp) + len(no_decay_clip_param_tp)
        n_other = len(decay_other_param_tp) + len(no_decay_other_param_tp)
        logger.info(
            "[Optimizer] CLIP: %d params (lr=%.2e) | Other: %d params (lr=%.2e)",
            n_clip,
            args.lr * coef_lr,
            n_other,
            args.lr,
        )

    # 用 fused AdamW（Ampere+ 上的融合 kernel）替换纯 Python 的 BertAdam。
    # 学习率调度用 LambdaLR 复现 BertAdam 的 warmup_cosine 语义：
    #   x = step / t_total；x<warmup 线性升到 1，之后走 0.5*(1+cos(pi*x)) 衰减到 0。
    # 与 BertAdam 的差异：AdamW 做 bias correction（BertAdam 没做），bf16 训练下这是更稳的选择。
    # fused=True requires strict dtype uniformity across params; disable when
    # init_model is set since older ckpts (e.g. June bin.4) mix fp16/fp32.
    use_fused = not bool(getattr(args, "init_model", None))
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=weight_decay,
        fused=use_fused,
    )
    scheduler = _build_warmup_cosine_scheduler(
        optimizer,
        warmup_proportion=args.warmup_proportion,
        total_steps=num_train_optimization_steps,
    )

    # DDP 三开关：
    #   - gradient_as_bucket_view=True：省显存 + 减一次拷贝
    #   - static_graph=True：DDP 缓存 reducer 计划，配合 grad checkpointing 提速
    #   - broadcast_buffers=False：模型无 BN，广播 buffer 是纯浪费
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers=False,
    )

    return optimizer, scheduler, model


def _publish_experiment_manifest(
    args,
    *,
    tracking_payload,
    manifest_path,
    batch_semantics,
):
    """Register or validate the A3 experiment manifest.

    Rank 0 exclusively owns the fresh check-and-write so non-zero ranks
    cannot race the file into existence that rank 0 itself just wrote and
    then refuse it.  Resume-time read validation runs on every rank so any
    manifest divergence fails the whole process group rather than only
    rank 0.  A group barrier at the end blocks non-zero ranks until rank 0
    has finished publishing training arguments and diagnostics.
    """

    manifest_path = Path(manifest_path)
    a3_only = bool(getattr(args, "contextual_risk_sidecar_only", False))
    resume_a3 = a3_only and bool(getattr(args, "resume_model", ""))
    if resume_a3:
        if not manifest_path.exists():
            raise FileNotFoundError(
                "A3 continuation requires the original experiment manifest"
            )
        validate_contextual_risk_resume_manifest(
            json.loads(manifest_path.read_text()), tracking_payload
        )
    if is_global_rank_zero(args):
        if a3_only and not resume_a3 and manifest_path.exists():
            raise FileExistsError(
                "fresh A3 run refuses an existing experiment manifest"
            )
        if not resume_a3:
            atomic_write_json(manifest_path, tracking_payload)
        _write_training_args_json(
            args, Path(args.output_dir) / "training_args.json"
        )
        for key, value in batch_semantics.items():
            logger.info("  Batch semantics | %s=%s", key, value)
    if a3_only:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def _write_training_args_json(args, path):
    """Persist the training-time args namespace so the audit CLI can rebuild
    the same UATVR architecture at eval time (Task 9 wiring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            try:
                json.dumps(list(value))
                payload[key] = list(value)
            except TypeError:
                payload[key] = repr(value)
        else:
            payload[key] = repr(value)
    atomic_write_json(path, payload)


def _build_training_checkpoint_payload(
    *,
    epoch,
    optimizer,
    scheduler,
    tr_loss,
    global_step,
    scheduler_horizon_epochs=None,
    scheduler_total_steps=None,
    optimizer_steps_per_epoch=None,
    rng_state_by_rank=None,
    contextual_risk_diagnostic_window=None,
):
    payload = {
        "epoch": int(epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": tr_loss,
        "global_step": int(global_step),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scheduler_horizon_epochs is not None:
        payload["scheduler_horizon_epochs"] = int(scheduler_horizon_epochs)
    if scheduler_total_steps is not None:
        payload["scheduler_total_steps"] = int(scheduler_total_steps)
    if optimizer_steps_per_epoch is not None:
        payload["optimizer_steps_per_epoch"] = int(optimizer_steps_per_epoch)
    if rng_state_by_rank is not None:
        payload["rng_state_by_rank"] = rng_state_by_rank
    if contextual_risk_diagnostic_window is not None:
        payload["contextual_risk_diagnostic_window"] = (
            _validate_contextual_risk_diagnostic_window(
                contextual_risk_diagnostic_window
            )
        )
    return payload


def _capture_local_rng_state():
    """Capture every stochastic generator used by an A3 worker process."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _gather_distributed_rng_states():
    """Collect rank-local RNG state on every rank at an epoch checkpoint."""

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local = {"rank": int(rank), "state": _capture_local_rng_state()}
    if not (dist.is_available() and dist.is_initialized()):
        gathered = [local]
    else:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    states = {}
    for item in gathered:
        if not isinstance(item, dict) or "rank" not in item or "state" not in item:
            raise RuntimeError("distributed A3 RNG gather returned malformed state")
        item_rank = int(item["rank"])
        if item_rank in states:
            raise RuntimeError("distributed A3 RNG gather returned duplicate rank")
        states[item_rank] = item["state"]
    if sorted(states) != list(range(len(gathered))):
        raise RuntimeError("distributed A3 RNG gather omitted a global rank")
    return states


def _restore_rng_state_for_current_rank(rng_state_by_rank, *, rank=None):
    """Restore the exact global-rank RNG state before stochastic training."""

    import torch.distributed as dist

    if rank is None:
        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
    state = rng_state_by_rank.get(rank)
    if state is None:
        state = rng_state_by_rank.get(str(rank))
    if not isinstance(state, dict):
        raise ValueError(f"A3 RNG checkpoint is missing global rank {rank}")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError("A3 RNG checkpoint missing " + ", ".join(missing))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "A3 CUDA RNG checkpoint device count differs from current worker"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    elif cuda_states:
        raise ValueError("A3 CUDA RNG checkpoint cannot restore without CUDA")


def _restore_training_checkpoint(
    checkpoint,
    optimizer,
    scheduler,
    *,
    require_complete,
    expected_scheduler_horizon_epochs=None,
    expected_scheduler_total_steps=None,
    expected_optimizer_steps_per_epoch=None,
    restore_rng=True,
    model=None,
):
    if require_complete:
        missing = [
            name
            for name in ("scheduler_state_dict", "global_step")
            if name not in checkpoint
        ]
        if missing:
            raise ValueError(
                "sidecar resume checkpoint missing " + ", ".join(missing)
            )
    registered_a3 = any(
        value is not None
        for value in (
            expected_scheduler_horizon_epochs,
            expected_scheduler_total_steps,
            expected_optimizer_steps_per_epoch,
        )
    )
    validated_diagnostic_window = None
    if registered_a3:
        expected = {
            "scheduler_horizon_epochs": expected_scheduler_horizon_epochs,
            "scheduler_total_steps": expected_scheduler_total_steps,
            "optimizer_steps_per_epoch": expected_optimizer_steps_per_epoch,
        }
        missing = sorted(
            name for name, value in expected.items()
            if value is not None and name not in checkpoint
        )
        if "rng_state_by_rank" not in checkpoint:
            missing.append("rng_state_by_rank")
        if "contextual_risk_diagnostic_window" not in checkpoint:
            missing.append("contextual_risk_diagnostic_window")
        if missing:
            raise ValueError(
                "registered A3 resume checkpoint missing " + ", ".join(missing)
            )
        for name, value in expected.items():
            if value is not None and int(checkpoint[name]) != int(value):
                raise ValueError(f"registered A3 resume {name} mismatch")
        if model is None:
            raise ValueError(
                "registered A3 resume requires a model for diagnostic window restore"
            )
        validated_diagnostic_window = _validate_contextual_risk_diagnostic_window(
            checkpoint["contextual_risk_diagnostic_window"]
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        if scheduler is None:
            raise ValueError("cannot restore scheduler state without a scheduler")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if registered_a3 and restore_rng:
        _restore_rng_state_for_current_rank(checkpoint["rng_state_by_rank"])
    if registered_a3:
        _restore_contextual_risk_diagnostic_window(
            model, validated_diagnostic_window
        )
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0))


def _optimizer_checkpoint_path(epoch, args, type_name=""):
    return os.path.join(
        args.output_dir,
        "pytorch_opt.bin.{}{}".format(
            "" if type_name == "" else type_name + ".",
            epoch,
        ),
    )


def _checkpoint_publication_marker_path(epoch, args, type_name=""):
    stem = "checkpoint_epoch.{}{}".format(
        "" if type_name == "" else type_name + ".",
        epoch,
    )
    return Path(args.output_dir) / f"{stem}.published.json"


def _fsync_directory(path):
    descriptor = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _torch_save_staged(payload, final_path):
    final = Path(final_path)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{final.name}.",
        suffix=".staging",
        dir=str(final.parent),
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Retain staging as crash evidence; startup diagnoses it explicitly.
        raise
    return staging


def _validate_checkpoint_publication(
    marker_path, *, epoch, model_path, optimizer_path
):
    marker_file = Path(marker_path)
    if not marker_file.is_file():
        raise ValueError(
            f"A3 checkpoint publication marker is missing: {marker_file}"
        )
    try:
        marker = json.loads(marker_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("A3 checkpoint publication marker is unreadable") from exc
    expected_model = Path(model_path).resolve()
    expected_optimizer = Path(optimizer_path).resolve()
    if int(marker.get("epoch", -1)) != int(epoch):
        raise ValueError("A3 checkpoint publication marker epoch mismatch")
    if Path(marker.get("model_path", "")).resolve() != expected_model:
        raise ValueError("A3 checkpoint publication marker model path mismatch")
    if Path(marker.get("optimizer_path", "")).resolve() != expected_optimizer:
        raise ValueError("A3 checkpoint publication marker optimizer path mismatch")
    try:
        model_sha = file_sha256(expected_model)
        optimizer_sha = file_sha256(expected_optimizer)
    except OSError as exc:
        raise ValueError("A3 published checkpoint pair is incomplete") from exc
    if marker.get("model_sha256") != model_sha:
        raise ValueError("A3 publication marker model SHA-256 mismatch")
    if marker.get("optimizer_sha256") != optimizer_sha:
        raise ValueError("A3 publication marker optimizer SHA-256 mismatch")
    return marker


def save_model(
    epoch,
    args,
    model,
    optimizer,
    tr_loss,
    type_name="",
    scheduler=None,
    global_step=0,
    scheduler_horizon_epochs=None,
    scheduler_total_steps=None,
    optimizer_steps_per_epoch=None,
    rng_state_by_rank=None,
):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch)
    )
    optimizer_state_file = _optimizer_checkpoint_path(epoch, args, type_name)
    append_only_sidecar = bool(
        getattr(args, "csd_logvar_sidecar_only", False)
        or getattr(args, "contextual_risk_sidecar_only", False)
    )
    a3_only = bool(getattr(args, "contextual_risk_sidecar_only", False))
    marker_path = _checkpoint_publication_marker_path(epoch, args, type_name)
    if append_only_sidecar:
        existing = [
            path
            for path in (output_model_file, optimizer_state_file, marker_path)
            if Path(path).exists()
        ]
        if existing:
            raise FileExistsError(
                "sidecar append-only save refuses existing checkpoint(s): "
                + ", ".join(str(path) for path in existing)
            )
    training_payload = _build_training_checkpoint_payload(
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        tr_loss=tr_loss,
        global_step=global_step,
        scheduler_horizon_epochs=scheduler_horizon_epochs,
        scheduler_total_steps=scheduler_total_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        rng_state_by_rank=rng_state_by_rank,
        contextual_risk_diagnostic_window=(
            _snapshot_contextual_risk_diagnostic_window(model)
            if a3_only
            else None
        ),
    )
    if a3_only:
        missing_registration = [
            name
            for name, value in (
                ("scheduler_horizon_epochs", scheduler_horizon_epochs),
                ("scheduler_total_steps", scheduler_total_steps),
                ("optimizer_steps_per_epoch", optimizer_steps_per_epoch),
                ("rng_state_by_rank", rng_state_by_rank),
            )
            if value is None
        ]
        if missing_registration:
            raise ValueError(
                "registered A3 checkpoint missing "
                + ", ".join(missing_registration)
            )
        output_model_path = Path(output_model_file)
        optimizer_state_path = Path(optimizer_state_file)
        staging = [
            *output_model_path.parent.glob(f".{output_model_path.name}.*.staging"),
            *optimizer_state_path.parent.glob(
                f".{optimizer_state_path.name}.*.staging"
            ),
        ]
        if staging:
            raise FileExistsError(
                "A3 incomplete staged checkpoint retained for diagnosis: "
                + ", ".join(str(path) for path in sorted(staging))
            )
        model_staging = _torch_save_staged(
            model_to_save.state_dict(), output_model_path
        )
        optimizer_staging = _torch_save_staged(
            training_payload, optimizer_state_path
        )
        os.replace(model_staging, output_model_path)
        os.replace(optimizer_staging, optimizer_state_path)
        _fsync_directory(output_model_path.parent)
        atomic_write_json(
            marker_path,
            {
                "epoch": int(epoch),
                "model_path": str(output_model_path.resolve()),
                "model_sha256": file_sha256(output_model_path),
                "optimizer_path": str(optimizer_state_path.resolve()),
                "optimizer_sha256": file_sha256(optimizer_state_path),
            },
        )
        _fsync_directory(output_model_path.parent)
    else:
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.save(training_payload, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def load_model(
    epoch, args, n_gpu, device, model_file=None, *, registered_a3_fp32=False
):  # for evaluation and test
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        # Prepare model
        cache_dir = (
            args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
        )
        UATVR_cls = _import_uatvr(getattr(args, "use_uatvr_full", False))
        model = UATVR_cls.from_pretrained(
            args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args
        )
        if registered_a3_fp32:
            _normalize_registered_a3_model_fp32(model)
        model.to(device)
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
    else:
        model = None
    return model


def _fmt_time(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _move_batch_to_device(batch, device):
    """Move a training batch to ``device`` uniformly across shapes.

    Supports the historical tuple batch, the CRF-mode :class:`CRFTrainBatch`
    dataclass (via its dataclass ``.to`` method), and any object that
    already exposes a ``.to(device, non_blocking=True)`` method.
    """
    from dataloaders.crf_pairing import CRFTrainBatch

    if isinstance(batch, CRFTrainBatch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return tuple(
            t.to(device=device, non_blocking=True) if hasattr(t, "to") else t
            for t in batch
        )
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _crf_dev_evaluate(model, args, device, dev_video_ids):
    """Score CRF-dev {clean, CRF34, CRF42} per video; return three-pair u_video accuracy.

    Rank-0-only. For each of the 250 CRF-dev videos we decode three
    variants (clean training-path video, CRF34 degraded, CRF42 degraded)
    via the same :class:`RawVideoExtractor` the training loop uses, run
    ``model.score_video_risk`` on each, and count three ordering pairs:

    * (u_clean < u_34)
    * (u_34   < u_42)
    * (u_clean < u_42)

    The returned float is the mean over all ``3 * len(dev_video_ids)``
    comparisons — the same three-pair accuracy the plan uses as the
    CRF-dev checkpoint-selection metric.
    """
    import numpy as np

    from crf_ladder import crf_output_path
    from dataloaders.rawvideo_util import RawVideoExtractor

    inner = model.module if hasattr(model, "module") else model
    was_training = inner.training
    inner.eval()

    extractor = RawVideoExtractor(
        framerate=args.feature_framerate,
        size=224,
    )

    def _decode(path):
        raw = extractor.get_video_data(str(path))["video"]
        if len(raw.shape) <= 3:
            raise RuntimeError(f"CRF-dev decode returned no frames: {path}")
        proc = extractor.process_raw_data(raw)  # [T_raw, 1, 3, H, W]
        if args.max_frames < proc.shape[0]:
            idx = np.linspace(0, proc.shape[0] - 1, num=args.max_frames, dtype=int)
            proc = proc[idx]
        return proc  # [T, 1, 3, H, W]

    correct = np.zeros((len(dev_video_ids), 3), dtype=np.float64)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, vid in enumerate(dev_video_ids):
            paths = [
                os.path.join(args.features_path, f"{vid}.mp4"),
                str(crf_output_path(args.crf_ladder_root, 34, vid)),
                str(crf_output_path(args.crf_ladder_root, 42, vid)),
            ]
            u_scores = []
            for path in paths:
                video = _decode(path)  # [T, 1, 3, H, W]
                video_t = torch.as_tensor(video).unsqueeze(0).unsqueeze(0).float().to(
                    device, non_blocking=True
                )
                # shape: [B=1, pair=1, T, 1, 3, H, W] — 7-dim path in score_video_risk
                T = video.shape[0]
                mask = torch.ones(1, T, dtype=torch.long, device=device)
                _, u_v = inner.score_video_risk(video_t, mask)
                u_scores.append(float(u_v.item()))
            u_clean, u_34, u_42 = u_scores
            correct[i, 0] = float(u_clean < u_34)
            correct[i, 1] = float(u_34 < u_42)
            correct[i, 2] = float(u_clean < u_42)

    if was_training:
        inner.train()
    return float(correct.mean())


def _unpack_train_batch(batch):
    group_ids = None
    if len(batch) == 5:
        input_ids, input_mask, segment_ids, video, video_mask = batch
    elif len(batch) == 6:
        input_ids, input_mask, segment_ids, video, video_mask, group_ids = batch
    elif len(batch) == 8:
        (
            input_ids,
            input_mask,
            segment_ids,
            _input_ids_a,
            _input_mask_a,
            _segment_ids_a,
            video,
            video_mask,
        ) = batch
    elif len(batch) == 9:
        (
            input_ids,
            input_mask,
            segment_ids,
            _input_ids_a,
            _input_mask_a,
            _segment_ids_a,
            video,
            video_mask,
            group_ids,
        ) = batch
    else:
        raise ValueError(f"Unexpected training batch size={len(batch)}")
    return input_ids, input_mask, segment_ids, video, video_mask, group_ids


def _should_log_training_step(global_step, log_step, smoke_cap, args):
    return global_step % log_step == 0 or (
        smoke_cap is not None
        and global_step == smoke_cap
        and bool(getattr(args, "use_csd_logvar_sidecar", False))
    )


def _contextual_risk_log_interval(args):
    if getattr(args, "use_contextual_risk_sidecar", False):
        return 20
    return int(args.n_display)


def _validate_contextual_risk_diagnostics(terms):
    """Abort immediately on invalid A3 loss values or one-class targets."""

    for name, value in terms.items():
        if name.startswith("contextual_risk_") and not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite A3 diagnostic: {name}")
    for direction in ("t2v", "v2t"):
        rate = float(terms[f"contextual_risk_{direction}_failure_rate"])
        if not 0.0 < rate < 1.0:
            raise RuntimeError(
                f"A3 {direction} BCE requires both success and failure classes"
            )


def _contextual_risk_gradient_norm(model):
    inner = model.module if hasattr(model, "module") else model
    squared_norm = 0.0
    for name, parameter in inner.named_parameters():
        if name.startswith("contextual_risk_sidecar.") and parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            squared_norm += float(torch.sum(gradient * gradient))
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite A3 gradient norm")
    return norm


def _advance_contextual_zero_gradient_windows(previous, gradient_norm):
    if not math.isfinite(float(gradient_norm)):
        raise FloatingPointError("non-finite A3 gradient norm")
    consecutive = int(previous) + 1 if float(gradient_norm) == 0.0 else 0
    if consecutive >= 2:
        raise RuntimeError("zero A3 gradient for two consecutive windows")
    return consecutive


def _contextual_risk_window_state(model):
    inner = model.module if hasattr(model, "module") else model
    state = getattr(inner, "_contextual_risk_diagnostic_window", None)
    if state is None:
        state = {
            "term_sums": {},
            "term_count": 0,
            "gradient_total": 0.0,
            "optimizer_steps": 0,
            "zero_gradient_windows": 0,
        }
        inner._contextual_risk_diagnostic_window = state
    return state


def _validate_contextual_risk_diagnostic_window(state):
    """Validate and normalize the complete continuation-sensitive A3 window."""

    required = {
        "term_sums",
        "term_count",
        "gradient_total",
        "optimizer_steps",
        "zero_gradient_windows",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("A3 diagnostic window has an invalid schema")
    term_sums = state["term_sums"]
    if not isinstance(term_sums, dict) or any(
        not isinstance(name, str) or not name.startswith("contextual_risk_")
        for name in term_sums
    ):
        raise ValueError("A3 diagnostic window term_sums is invalid")
    normalized_sums = {}
    for name, value in term_sums.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "A3 diagnostic window term_sums contains a non-numeric value"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("A3 diagnostic window term_sums is non-finite or negative")
        normalized_sums[name] = numeric

    normalized = {"term_sums": normalized_sums}
    for name in ("term_count", "optimizer_steps", "zero_gradient_windows"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"A3 diagnostic window {name} is invalid")
        normalized[name] = int(value)
    try:
        gradient_total = float(state["gradient_total"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "A3 diagnostic window gradient_total is non-numeric"
        ) from exc
    if not math.isfinite(gradient_total) or gradient_total < 0.0:
        raise ValueError("A3 diagnostic window gradient_total is invalid")
    normalized["gradient_total"] = gradient_total
    if normalized["optimizer_steps"] >= 20:
        raise ValueError("A3 diagnostic window optimizer_steps must be below 20")
    if normalized["zero_gradient_windows"] > 1:
        raise ValueError("A3 diagnostic window consecutive zero count is invalid")
    if normalized["term_count"] == 0 and normalized_sums:
        raise ValueError("A3 diagnostic window has sums without terms")
    if normalized["term_count"] > 0 and not normalized_sums:
        raise ValueError("A3 diagnostic window has terms without sums")
    if normalized["term_count"] != normalized["optimizer_steps"]:
        raise ValueError("A3 diagnostic window term and optimizer counts differ")
    if normalized["optimizer_steps"] == 0 and gradient_total != 0.0:
        raise ValueError("A3 diagnostic window has gradients without steps")
    return normalized


def _snapshot_contextual_risk_diagnostic_window(model):
    return copy.deepcopy(
        _validate_contextual_risk_diagnostic_window(
            _contextual_risk_window_state(model)
        )
    )


def _restore_contextual_risk_diagnostic_window(model, state):
    inner = model.module if hasattr(model, "module") else model
    inner._contextual_risk_diagnostic_window = copy.deepcopy(
        _validate_contextual_risk_diagnostic_window(state)
    )


def _accumulate_contextual_risk_window(model, terms):
    state = _contextual_risk_window_state(model)
    for name, value in terms.items():
        state["term_sums"][name] = state["term_sums"].get(name, 0.0) + float(
            value
        )
    state["term_count"] += 1


def _finish_contextual_risk_window_step(model, gradient_norm):
    """Finish one optimizer step and emit only a complete 20-step window."""

    state = _contextual_risk_window_state(model)
    state["gradient_total"] += float(gradient_norm)
    state["optimizer_steps"] += 1
    if state["optimizer_steps"] < 20:
        return None
    if state["optimizer_steps"] != 20 or state["term_count"] <= 0:
        raise RuntimeError("invalid A3 diagnostic window state")

    means = {
        name: total / state["term_count"]
        for name, total in state["term_sums"].items()
    }
    means["contextual_risk_gradient_norm"] = (
        state["gradient_total"] / state["optimizer_steps"]
    )
    zero_gradient_windows = _advance_contextual_zero_gradient_windows(
        state["zero_gradient_windows"],
        means["contextual_risk_gradient_norm"],
    )
    inner = model.module if hasattr(model, "module") else model
    inner._contextual_risk_diagnostic_window = {
        "term_sums": {},
        "term_count": 0,
        "gradient_total": 0.0,
        "optimizer_steps": 0,
        "zero_gradient_windows": zero_gradient_windows,
    }
    return means


def _log_contextual_risk_window(means):
    logger.info(
        "  [A3 window] total=%.4f BCE=%.4f fit=%.4f rank=%.4f | grad=%.4e",
        means["contextual_risk_loss"],
        means["contextual_risk_bce"],
        means["contextual_risk_fit"],
        means["contextual_risk_rank"],
        means["contextual_risk_gradient_norm"],
    )
    for direction in ("t2v", "v2t"):
        prefix = f"contextual_risk_{direction}_"
        logger.info(
            "  [A3 window] %s failure=%.4f weight=%.4f | "
            "risk=%.4f/%.4f hardness=%.4f/%.4f | valid-pair=%.4f "
            "saturation=%.4f top-K entropy=%.4f",
            direction.upper(),
            means[prefix + "failure_rate"],
            means[prefix + "positive_weight"],
            means[prefix + "prediction_mean"],
            means[prefix + "prediction_std"],
            means[prefix + "hardness_mean"],
            means[prefix + "hardness_std"],
            means[prefix + "valid_ranking_fraction"],
            means[prefix + "saturation_fraction"],
            means[prefix + "entropy"],
        )


def _time_per_dataloader_step_since_log(
    elapsed_seconds, current_dataloader_step, last_logged_dataloader_step
):
    return elapsed_seconds / (
        current_dataloader_step - last_logged_dataloader_step
    )


def _train_epoch_impl(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    set_retrieval_risk_training_mode(model)
    set_csd_logvar_training_mode(model)
    _set_contextual_risk_training_mode(model)
    log_step = _contextual_risk_log_interval(args)
    last_log_time = time.time()
    last_logged_dataloader_step = 0
    epoch_start_time = time.time()
    total_loss = 0
    num_steps = len(train_dataloader)
    # 每个 optimizer step 对应 gradient_accumulation_steps 个 dataloader step
    total_opt_steps = num_steps // args.gradient_accumulation_steps

    from dataloaders.crf_pairing import CRFTrainBatch

    # SMOKE_STEPS env: cap the training loop at N forward-backward iterations.
    # Task 11 step 6 uses `SMOKE_STEPS=1 bash run_train_msrvtt_crf_bg.sh` to
    # verify the CRF pipeline end-to-end without paying for a full epoch.
    _smoke_steps_env = os.environ.get("SMOKE_STEPS")
    smoke_cap: int | None = int(_smoke_steps_env) if _smoke_steps_env else None
    if smoke_cap is not None and local_rank == 0:
        logger.info("[SMOKE_STEPS] cap=%d — will exit train_epoch after %d step(s)",
                    smoke_cap, smoke_cap)

    crf_window: dict[str, float] = {"cf": 0.0, "cv": 0.0, "of": 0.0, "ov": 0.0,
                                    "rank_total": 0.0, "retrieval": 0.0}
    crf_window_count = 0
    csd_window: dict[str, float] = {}
    csd_window_count = 0
    for step, batch in enumerate(train_dataloader):
        if smoke_cap is not None and step >= smoke_cap:
            if local_rank == 0:
                logger.info("[SMOKE_STEPS] reached step=%d cap=%d — breaking",
                            step, smoke_cap)
            break
        batch = _move_batch_to_device(batch, device)

        loss_terms = None
        if isinstance(batch, CRFTrainBatch):
            # CRF profile: retrieval loss + weighted ranking loss.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inner = model.module if hasattr(model, "module") else model
                total, loss_terms = inner.forward_crf_train(
                    batch,
                    base_margin=float(args.aleatoric_base_margin),
                    lambda_cf=float(args.aleatoric_lambda_cf),
                    lambda_cv=float(args.aleatoric_lambda_cv),
                    lambda_of=float(args.aleatoric_lambda_of),
                    lambda_ov=float(args.aleatoric_lambda_ov),
                )
                loss = total * float(args.aleatoric_ranking_weight)
        else:
            input_ids, input_mask, segment_ids, video, video_mask, group_ids = (
                _unpack_train_batch(batch)
            )
            # A800 原生支持 bf16；autocast 内部激活走 bf16，master 参数仍 fp32，无需 GradScaler。
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=_a3_training_autocast_enabled(args),
            ):
                sidecar_kwargs = {}
                if (
                    getattr(args, "use_retrieval_risk_sidecar", False)
                    or getattr(args, "use_csd_logvar_sidecar", False)
                    or getattr(args, "use_contextual_risk_sidecar", False)
                ):
                    sidecar_kwargs["group_ids"] = group_ids
                loss_output = model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    video,
                    video_mask,
                    **sidecar_kwargs,
                )
            # June arch UATVR (--use_uatvr_full) returns a dict; HEAD arch
            # returns a scalar tensor. Unpack the dict here so downstream
            # scalar-tensor semantics (mean, /accum, backward) still apply.
            if isinstance(loss_output, dict):
                if getattr(args, "use_csd_logvar_sidecar", False):
                    loss_terms = {
                        key: value.detach()
                        for key, value in loss_output.items()
                        if key.startswith("csd_logvar_")
                        and key != "csd_logvar_loss"
                    }
                elif getattr(args, "use_contextual_risk_sidecar", False):
                    loss_terms = {
                        key: value.detach()
                        for key, value in loss_output.items()
                        if key.startswith("contextual_risk_")
                    }
                    _validate_contextual_risk_diagnostics(loss_terms)
                loss = loss_output["total"]
            else:
                loss = loss_output

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if loss_terms is not None:
            if any(name.startswith("csd_logvar_") for name in loss_terms):
                for name, tensor in loss_terms.items():
                    csd_window[name] = csd_window.get(name, 0.0) + float(tensor)
                csd_window_count += 1
            elif any(
                name.startswith("contextual_risk_") for name in loss_terms
            ):
                _accumulate_contextual_risk_window(model, loss_terms)
            else:
                for name, tensor in loss_terms.items():
                    crf_window[name] = crf_window.get(name, 0.0) + float(tensor)
                crf_window_count += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:
            contextual_window_means = None
            if getattr(args, "use_contextual_risk_sidecar", False):
                contextual_window_means = _finish_contextual_risk_window_step(
                    model, _contextual_risk_gradient_norm(model)
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # 先记录本次 step 将要应用的 LR，再 step。
            scheduled_group_lrs = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # 推进到下一步的 LR
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, "module"):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1

            should_log = _should_log_training_step(
                global_step, log_step, smoke_cap, args
            )
            should_log = should_log or contextual_window_means is not None

            if (
                should_log
                and local_rank == 0
            ):
                # --- 计算进度 & ETA ---
                opt_step = global_step % total_opt_steps or total_opt_steps
                progress = opt_step / total_opt_steps * 100
                current_dataloader_step = step + 1
                current_time = time.time()
                time_per_step = _time_per_dataloader_step_since_log(
                    current_time - last_log_time,
                    current_dataloader_step,
                    last_logged_dataloader_step,
                )
                remaining_steps = num_steps - (step + 1)
                eta = time_per_step * remaining_steps
                last_log_time = current_time
                last_logged_dataloader_step = current_dataloader_step

                lr_clip = scheduled_group_lrs[0]
                lr_new = (
                    scheduled_group_lrs[2]
                    if len(scheduled_group_lrs) > 2
                    else lr_clip
                )

                logger.info(
                    "[Epoch %d/%d] step=%d/%d progress=%.0f%% | loss=%.4f | "
                    "lr=%.2e/%.2e | time=%.2fs eta=%s",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    num_steps,
                    progress,
                    float(loss),
                    lr_clip,
                    lr_new,
                    time_per_step,
                    _fmt_time(eta),
                )
                if crf_window_count > 0 and local_rank == 0:
                    means = {k: v / crf_window_count for k, v in crf_window.items()}
                    logger.info(
                        "  [CRF window] retrieval=%.4f rank=%.4f | "
                        "cf=%.4f cv=%.4f of=%.4f ov=%.4f",
                        means.get("retrieval", 0.0),
                        means.get("rank_total", 0.0),
                        means.get("cf", 0.0),
                        means.get("cv", 0.0),
                        means.get("of", 0.0),
                        means.get("ov", 0.0),
                    )
                    crf_window = {k: 0.0 for k in crf_window}
                    crf_window_count = 0
                if csd_window_count > 0 and local_rank == 0:
                    means = {
                        key: value / csd_window_count
                        for key, value in csd_window.items()
                    }
                    logger.info(
                        "  [CSD window] BCE pos=%.4f neg=%.4f | "
                        "grounding fit=%.4f rank=%.4f | VIB=%.4f",
                        means.get("csd_logvar_csd_positive", 0.0),
                        means.get("csd_logvar_csd_negative", 0.0),
                        means.get("csd_logvar_grounding_fit", 0.0),
                        means.get("csd_logvar_grounding_rank", 0.0),
                        means.get("csd_logvar_vib", 0.0),
                    )
                    logger.info(
                        "  [CSD window] distance mean=%.4f variance=%.4f | "
                        "scale=%.4f shift=%.4f",
                        means.get("csd_logvar_mean_distance", 0.0),
                        means.get("csd_logvar_variance_distance", 0.0),
                        means.get("csd_logvar_scale", 0.0),
                        means.get("csd_logvar_shift", 0.0),
                    )
                    logger.info(
                        "  [CSD window] clamp text low/high=%.4f/%.4f | "
                        "video low/high=%.4f/%.4f",
                        means.get("csd_logvar_text_clamp_low", 0.0),
                        means.get("csd_logvar_text_clamp_high", 0.0),
                        means.get("csd_logvar_video_clamp_low", 0.0),
                        means.get("csd_logvar_video_clamp_high", 0.0),
                    )
                    csd_window = {}
                    csd_window_count = 0
                if contextual_window_means is not None:
                    _log_contextual_risk_window(contextual_window_means)

    total_loss = total_loss / len(train_dataloader)
    epoch_time = time.time() - epoch_start_time
    if local_rank == 0:
        logger.info(
            "[Epoch %d/%d] Training complete | Avg Loss: %.4f | Time: %s",
            epoch + 1,
            args.epochs,
            total_loss,
            _fmt_time(epoch_time),
        )
    return total_loss, global_step


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, local_rank=0):
    """Run one epoch under the registered A3 FP32 backend contract."""

    with _a3_exact_fp32_backend(args):
        return _train_epoch_impl(
            epoch, args, model, train_dataloader, device, n_gpu, optimizer,
            scheduler, global_step, local_rank=local_rank,
        )


def _log_mus_scores_tsv(args, sim_matrix: "np.ndarray"):
    """将 sim_matrix 每行的 MUS 写入 TSV 诊断日志（rank0 only）。

    文件路径：logs/mus_scores/YYYYMMDD/<run_id>.tsv
    列：idx, mus_score
    """
    import datetime as _dt

    from modules.mus_util import compute_mus_batch

    date_str = _dt.datetime.now().strftime("%Y%m%d")
    run_id = os.path.basename(os.path.normpath(args.output_dir)) or "eval"
    out_dir = os.path.join("logs", "mus_scores", date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{run_id}.tsv")

    mus_scores = compute_mus_batch(sim_matrix, k=10)
    mean_mus = float(np.mean(mus_scores))
    high_ratio = float(np.mean(mus_scores > 0.5))

    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write("time\tidx\tmus_score\n")
        now = _dt.datetime.now().isoformat(timespec="seconds")
        for i, v in enumerate(mus_scores):
            f.write(f"{now}\t{i}\t{v:.6f}\n")

    logger.info(
        "MUS stats | mean=%.4f  high_ratio(>0.5)=%.3f  n_queries=%d  log=%s",
        mean_mus, high_ratio, len(mus_scores), out_file,
    )


def _run_on_single_gpu(
    model,
    args,
    batch_list_t,
    batch_list_v,
    batch_sequence_output_list,
    batch_visual_output_list,
):
    # Feature batches are cached on CPU by eval_epoch and moved to the worker
    # device in bounded video chunks for UATVR similarity computation.
    # For June arch (--use_uatvr_full) each entry is a (visual_cls, visual_hidden)
    # tuple; HEAD arch stores a single tensor.
    is_june = isinstance(batch_visual_output_list[0], tuple)
    if is_june:
        visual_cls_all = torch.cat([v[0] for v in batch_visual_output_list], dim=0)
        visual_hidden_all = torch.cat([v[1] for v in batch_visual_output_list], dim=0)
    else:
        visual_output_all = torch.cat(batch_visual_output_list, dim=0)
    video_mask_all = torch.cat([v[0] for v in batch_list_v], dim=0)

    device = next(model.parameters()).device
    chunk_size = getattr(args, "eval_vid_chunk_size", 128)
    n_vid = visual_cls_all.size(0) if is_june else visual_output_all.size(0)
    video_chunks = []
    for v_start in range(0, n_vid, chunk_size):
        v_end = min(v_start + chunk_size, n_vid)
        if is_june:
            video_chunk = (
                visual_cls_all[v_start:v_end].to(device),
                visual_hidden_all[v_start:v_end].to(device),
                video_mask_all[v_start:v_end].to(device),
            )
        else:
            video_chunk = (
                visual_output_all[v_start:v_end].to(device),
                video_mask_all[v_start:v_end].to(device),
            )
        video_chunks.append(video_chunk)
    if is_june:
        del visual_cls_all, visual_hidden_all, video_mask_all
    else:
        del visual_output_all, video_mask_all

    collect_csd = bool(getattr(args, "use_csd_logvar_sidecar", False))
    collect_contextual = bool(
        getattr(args, "use_contextual_risk_sidecar", False)
    )
    collect_aux = collect_csd or collect_contextual
    sim_matrix = []
    text_payload = []
    video_payload = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, _segment_ids, *_tmp = b1
        sequence_output, text_token = batch_sequence_output_list[idx1]
        sequence_output = sequence_output.to(device)
        text_token = text_token.to(device)
        input_mask = input_mask.to(device)

        row_logits = []
        for chunk_index, chunk in enumerate(video_chunks):
            if is_june:
                visual_output, visual_hidden, video_mask = chunk
                auxiliary_kwargs = {"return_csd_aux": collect_csd}
                if collect_contextual:
                    auxiliary_kwargs["return_contextual_aux"] = True
                logits, aux = model.get_similarity_logits(
                    sequence_output,
                    text_token,
                    visual_output,
                    visual_hidden,
                    input_mask,
                    video_mask,
                    loose_type=model.loose_type,
                    **auxiliary_kwargs,
                )
            else:
                visual_output, video_mask = chunk
                logits, *_tmp = model.get_similarity_logits(
                    sequence_output,
                    text_token,
                    visual_output,
                    input_mask,
                    video_mask,
                    loose_type=model.loose_type,
                )
            row_logits.append(logits)
            if collect_aux and chunk_index == 0:
                text_payload.append(
                    {
                        name: aux[name].float().cpu()
                        for name in (
                            ("text_mu", "text_logvar")
                            if collect_csd
                            else ("text_mu",)
                        )
                    }
                )
            if collect_aux and idx1 == 0:
                video_payload.append(
                    {
                        name: aux[name].float().cpu()
                        for name in (
                            ("video_mu", "video_logvar")
                            if collect_csd
                            else ("video_mu",)
                        )
                    }
                )

        # bf16 张量不能直接 .numpy()，必须先转 fp32。
        b1_all_v_logits = torch.cat(row_logits, dim=1).float().cpu()
        sim_matrix.append(b1_all_v_logits.detach().numpy())

    if not collect_aux:
        return sim_matrix
    payload = {
        "similarity_rows": sim_matrix,
        "text_mu": torch.cat(
            [payload["text_mu"] for payload in text_payload], dim=0
        ),
        "video_mu": torch.cat(
            [payload["video_mu"] for payload in video_payload], dim=0
        ),
    }
    if collect_csd:
        payload.update(
            {
                "text_logvar": torch.cat(
                    [payload["text_logvar"] for payload in text_payload], dim=0
                ),
                "video_logvar": torch.cat(
                    [payload["video_logvar"] for payload in video_payload], dim=0
                ),
            }
        )
    return payload


def _merge_canonical_eval_means(outputs):
    """Merge ordered text means and deduplicate all-worker video means."""
    similarity_rows = [
        row for output in outputs for row in output["similarity_rows"]
    ]
    matrix = np.concatenate(tuple(similarity_rows), axis=0)
    text_mu = torch.cat([output["text_mu"] for output in outputs], dim=0)
    video_mu = outputs[0]["video_mu"]
    if (
        text_mu.size(0) != matrix.shape[0]
        or video_mu.size(0) != matrix.shape[1]
    ):
        raise ValueError(
            "canonical mean axes do not align with retrieval matrix"
        )
    return matrix, {"text_mu": text_mu, "video_mu": video_mu}


def _merge_contextual_eval_outputs(outputs):
    """Merge the complete WTI matrix with canonical A3 modality means."""

    return _merge_canonical_eval_means(outputs)


def _merge_csd_eval_outputs(outputs):
    """Merge per-worker CSD payloads along retrieval matrix axes."""

    try:
        matrix, means = _merge_canonical_eval_means(outputs)
    except ValueError as error:
        raise ValueError(
            "intrinsic uncertainty axes do not align with retrieval matrix"
        ) from error
    text_logvar = torch.cat(
        [output["text_logvar"] for output in outputs], dim=0
    )
    video_logvar = outputs[0]["video_logvar"]
    if text_logvar.shape != means["text_mu"].shape or video_logvar.shape != means[
        "video_mu"
    ].shape:
        raise ValueError(
            "intrinsic uncertainty axes do not align with retrieval matrix"
        )
    return matrix, {
        **means,
        "text_logvar": text_logvar,
        "video_logvar": video_logvar,
    }


def _select_eval_device_ids(n_gpu, num_text_batches):
    """Avoid launching parallel similarity workers with empty text splits."""

    if n_gpu <= 0 or num_text_batches <= 0:
        raise ValueError("evaluation requires a GPU and at least one text batch")
    split_len = (int(num_text_batches) + int(n_gpu) - 1) // int(n_gpu)
    worker_count = (int(num_text_batches) + split_len - 1) // split_len
    return list(range(worker_count))


def select_best_checkpoint(
    best_score, best_path, candidate_score, candidate_path
):
    """Select by one validation R@1 metric; ties prefer the later epoch."""
    if candidate_score >= best_score:
        return candidate_score, candidate_path
    return best_score, best_path


def select_multi_sentence_video_rows(batch_start, batch_size, cut_off_points):
    """Return local rows for 1-based exclusive caption-group endpoints."""
    batch_end = batch_start + batch_size
    return [
        int(cut_off) - 1 - batch_start
        for cut_off in cut_off_points
        if batch_start < int(cut_off) <= batch_end
    ]


def _reshape_multi_sentence_matrix(sim_matrix, cut_off_points):
    max_length = max(
        end - start
        for start, end in zip([0] + cut_off_points[:-1], cut_off_points)
    )
    grouped = []
    for start, end in zip([0] + cut_off_points[:-1], cut_off_points):
        padding = np.full(
            (max_length - end + start, sim_matrix.shape[1]),
            -np.inf,
        )
        grouped.append(
            np.concatenate((sim_matrix[start:end], padding), axis=0)
        )
    return np.stack(tuple(grouped), axis=0)


def _normalize_eval_directions(directions):
    directions = tuple(dict.fromkeys(directions))
    if not directions or any(direction not in {"t2v", "v2t"} for direction in directions):
        raise ValueError("directions must contain one or both of: t2v, v2t")
    return directions


def _compute_directional_metrics(
    text_to_video_matrix,
    video_to_text_matrix,
    *,
    cut_off_points=None,
    independent_directions=True,
    directions=("t2v", "v2t"),
):
    directions = _normalize_eval_directions(directions)
    if cut_off_points is None:
        metrics = {}
        if "t2v" in directions:
            metrics["t2v"] = compute_metrics(text_to_video_matrix)
        if "v2t" in directions:
            metrics["v2t"] = compute_metrics(video_to_text_matrix.T)
        return metrics

    metrics = {}
    if "t2v" in directions:
        text_to_video_grouped = _reshape_multi_sentence_matrix(
            text_to_video_matrix,
            cut_off_points,
        )
        metrics["t2v"] = tensor_text_to_video_metrics(text_to_video_grouped)
    if "v2t" in directions:
        multi_caption_v2t = (
            video_to_text_matrix
            if independent_directions
            else text_to_video_matrix
        )
        video_to_text_grouped = _reshape_multi_sentence_matrix(
            multi_caption_v2t,
            cut_off_points,
        )
        metrics["v2t"] = compute_metrics(tensor_video_to_text_sim(video_to_text_grouped))
    return metrics


def _serialize_retrieval_metrics(metrics):
    return {
        name: value if isinstance(value, list) else float(value)
        for name, value in metrics.items()
    }


def _compute_retrieval_risk_metrics(
    *,
    model,
    risk_similarity_matrix,
    metric_t2v_matrix,
    metric_v2t_matrix,
    positive_mask,
    directions,
    device,
    bootstrap_samples=1000,
    return_array_digests=False,
    return_array_evidence=False,
    canonical_cpu_replay=False,
    artifact_array_store=None,
):
    """Score the complete candidate matrix without mutating retrieval inputs."""

    directions = _normalize_eval_directions(directions)
    raw_logits = torch.from_numpy(
        np.asarray(risk_similarity_matrix, dtype=np.float32)
    )
    retrieval_sidecar = (
        getattr(model, "retrieval_risk_sidecar", None)
        if canonical_cpu_replay
        else None
    )
    cpu_sidecar = (
        copy.deepcopy(retrieval_sidecar).cpu().eval()
        if retrieval_sidecar is not None
        else None
    )
    if cpu_sidecar is None:
        raw_logits = raw_logits.to(device)
    payload = {}
    array_digests = {}
    with torch.no_grad():
        for offset, direction in enumerate(directions):
            output = (
                cpu_sidecar(raw_logits, direction)
                if cpu_sidecar is not None
                else model.score_retrieval_risk(raw_logits, direction)
            )
            failure_risk = output["failure_risk"].float().cpu().numpy()
            vacuity = output["vacuity"].float().cpu().numpy()
            if direction == "t2v":
                scores = metric_t2v_matrix
                mask = positive_mask
            else:
                scores = metric_v2t_matrix.T
                mask = positive_mask.T
            payload[direction] = evaluate_retrieval_risk(
                scores,
                mask,
                failure_risk,
                vacuity,
                bootstrap_samples=bootstrap_samples,
                seed=offset,
            )
            array_digests[direction] = {
                "failure_risk": array_sha256(failure_risk),
                "vacuity": array_sha256(vacuity),
            }
            if return_array_evidence:
                if artifact_array_store is not None:
                    failure_reference = artifact_array_store.write(
                        f"a1_failure_risk.{direction}", failure_risk
                    ).as_dict()
                    vacuity_reference = artifact_array_store.write(
                        f"a1_vacuity.{direction}", vacuity
                    ).as_dict()
                    array_digests[direction].update({
                        "failure_risk_values": failure_reference,
                        "failure_risk_metadata": failure_reference,
                        "vacuity_values": vacuity_reference,
                        "vacuity_metadata": vacuity_reference,
                    })
                else:
                    array_digests[direction].update({
                        "failure_risk_values": failure_risk.tolist(),
                        "failure_risk_dtype": str(failure_risk.dtype),
                        "failure_risk_shape": list(failure_risk.shape),
                        "vacuity_values": vacuity.tolist(),
                        "vacuity_dtype": str(vacuity.dtype),
                        "vacuity_shape": list(vacuity.shape),
                    })
    if return_array_digests:
        return payload, array_digests
    return payload


def _compute_contextual_retrieval_risk_metrics(
    *,
    model,
    risk_similarity_matrix,
    positive_mask,
    text_means,
    video_means,
    directions,
    device,
    bootstrap_samples=1000,
    return_evidence=False,
    artifact_array_store=None,
):
    """Score A3 once per direction on the complete raw candidate matrix."""

    directions = _normalize_eval_directions(directions)
    raw_wti = np.asarray(risk_similarity_matrix, dtype=np.float32)
    text_array = torch.as_tensor(text_means).detach().float().cpu().numpy()
    video_array = torch.as_tensor(video_means).detach().float().cpu().numpy()
    raw_logits = torch.from_numpy(raw_wti)
    text_tensor = torch.from_numpy(text_array)
    video_tensor = torch.from_numpy(video_array)
    contextual_sidecar = getattr(model, "contextual_risk_sidecar", None)
    cpu_sidecar = (
        copy.deepcopy(contextual_sidecar).cpu().eval()
        if contextual_sidecar is not None
        else None
    )
    if cpu_sidecar is None:
        raw_logits = raw_logits.to(device)
        text_tensor = text_tensor.to(device)
        video_tensor = video_tensor.to(device)
    payload = {}
    failure_risk_arrays = {}
    with torch.no_grad():
        for offset, direction in enumerate(directions):
            if cpu_sidecar is not None:
                output = cpu_sidecar(
                    raw_logits,
                    text_tensor if direction == "t2v" else video_tensor,
                    video_tensor if direction == "t2v" else text_tensor,
                    direction,
                )
            else:
                output = model.score_contextual_retrieval_risk(
                    raw_logits,
                    text_tensor,
                    video_tensor,
                    direction,
                )
            failure_risk = output["failure_risk"].float().cpu().numpy()
            failure_risk_arrays[direction] = failure_risk
            scores = raw_wti if direction == "t2v" else raw_wti.T
            mask = positive_mask if direction == "t2v" else positive_mask.T
            payload[direction] = evaluate_retrieval_risk(
                scores,
                mask,
                failure_risk,
                bootstrap_samples=bootstrap_samples,
                seed=offset,
            )

    if not return_evidence:
        return payload
    mask_array = np.asarray(positive_mask, dtype=bool)

    def metadata(array):
        value = np.asarray(array)
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }

    evidence = {
        "raw_wti": raw_wti.tolist(),
        "text_means": text_array.tolist(),
        "video_means": video_array.tolist(),
        "positive_mask": mask_array.tolist(),
        "failure_risk": {
            direction: failure_risk_arrays[direction].tolist()
            for direction in directions
        },
        "sha256": {
            "raw_wti": array_sha256(raw_wti),
            "text_means": array_sha256(text_array),
            "video_means": array_sha256(video_array),
            "failure_risk": {
                direction: array_sha256(failure_risk_arrays[direction])
                for direction in directions
            },
            "positive_mask": array_sha256(mask_array),
        },
        "array_metadata": {
            "raw_wti": metadata(raw_wti),
            "text_means": metadata(text_array),
            "video_means": metadata(video_array),
            "positive_mask": metadata(mask_array),
            "failure_risk": {
                direction: metadata(failure_risk_arrays[direction])
                for direction in directions
            },
        },
        "metric_config": {
            "bootstrap_samples": int(bootstrap_samples),
            "seed_by_direction": {
                direction: offset for offset, direction in enumerate(directions)
            },
        },
    }
    if artifact_array_store is not None:
        evidence["artifact_array_root"] = str(
            artifact_array_store.root.parent.resolve()
        )
        for name, value in (
            ("raw_wti", raw_wti),
            ("text_means", text_array),
            ("video_means", video_array),
            ("positive_mask", mask_array),
        ):
            reference = artifact_array_store.write(name, value).as_dict()
            evidence[name] = reference
            evidence["array_metadata"][name] = reference
        for direction in directions:
            reference = artifact_array_store.write(
                f"failure_risk.{direction}", failure_risk_arrays[direction]
            ).as_dict()
            evidence["failure_risk"][direction] = reference
            evidence["array_metadata"]["failure_risk"][direction] = reference
        validate_artifact_json_payload(evidence)
    return payload, evidence


def _build_paired_digest(
    result,
    risk_similarity_matrix,
    risk_array_digests,
    directions,
    *,
    text_means=None,
    video_means=None,
):
    """Build exact A1-output evidence for an A2 paired evaluation."""

    payload = {
        "raw_wti_sha256": array_sha256(risk_similarity_matrix),
        "a1_failure_risk_sha256": {
            direction: risk_array_digests[direction]["failure_risk"]
            for direction in directions
        },
        "a1_vacuity_sha256": {
            direction: risk_array_digests[direction]["vacuity"]
            for direction in directions
        },
        "retrieval": {direction: result[direction] for direction in directions},
    }
    if text_means is not None and video_means is not None:
        payload.update({
            "text_means_sha256": array_sha256(
                np.asarray(text_means, dtype=np.float32)
            ),
            "video_means_sha256": array_sha256(
                np.asarray(video_means, dtype=np.float32)
            ),
        })
    return payload


def _compute_csd_logvar_metrics(
    *,
    metric_t2v_matrix,
    metric_v2t_matrix,
    positive_mask,
    intrinsic,
    bootstrap_samples,
    logvar_min,
    logvar_max,
):
    """Evaluate modality-specific intrinsic uncertainty by query axis."""

    return {
        "t2v": evaluate_intrinsic_uncertainty(
            metric_t2v_matrix,
            positive_mask,
            intrinsic["text_logvar"].exp().mean(-1).numpy(),
            bootstrap_samples,
            0,
        ),
        "v2t": evaluate_intrinsic_uncertainty(
            metric_v2t_matrix.T,
            positive_mask.T,
            intrinsic["video_logvar"].exp().mean(-1).numpy(),
            bootstrap_samples,
            1,
        ),
        "text_distribution": summarize_logvar(
            intrinsic["text_logvar"].numpy(), logvar_min, logvar_max
        ),
        "video_distribution": summarize_logvar(
            intrinsic["video_logvar"].numpy(), logvar_min, logvar_max
        ),
        "distance_components": summarize_csd_components(
            intrinsic["text_mu"].numpy(),
            intrinsic["video_mu"].numpy(),
            intrinsic["text_logvar"].numpy(),
            intrinsic["video_logvar"].numpy(),
        ),
    }


def _apply_dual_softmax(similarity_matrix, temperature=1.0):
    """Return independent T2V/V2T DSL matrices without mutating the input."""

    if temperature <= 0:
        raise ValueError("dsl_temperature must be positive")
    matrix = torch.as_tensor(
        np.asarray(similarity_matrix, dtype=np.float32)
    )
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("DSL requires a finite 2-D similarity matrix")

    text_to_video = (
        matrix
        * F.softmax(matrix / temperature, dim=0)
        * matrix.size(0)
    )
    transposed = matrix.T
    video_to_text = (
        transposed
        * F.softmax(transposed / temperature, dim=0)
        * transposed.size(0)
    )
    return (
        text_to_video.detach().cpu().numpy(),
        video_to_text.T.detach().cpu().numpy(),
    )


@contextmanager
def _eval_autocast_context(args):
    """Scope exact FP32 and deterministic cuDNN to paired sidecar evaluation."""

    paired_sidecar_only = bool(
        getattr(args, "csd_logvar_sidecar_only", False)
        or getattr(args, "contextual_risk_sidecar_only", False)
        or getattr(args, "use_contextual_risk_sidecar", False)
    )
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_deterministic = torch.backends.cudnn.deterministic
    if paired_sidecar_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        with _a3_exact_fp32_backend(args):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=not paired_sidecar_only,
            ):
                yield
    finally:
        if paired_sidecar_only:
            torch.backends.cudnn.benchmark = prior_benchmark
            torch.backends.cudnn.deterministic = prior_deterministic


def _eval_epoch_impl(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    directions=("t2v", "v2t"),
    report_csd_logvar=None,
    artifact_array_store=None,
):
    directions = _normalize_eval_directions(directions)
    if report_csd_logvar is None:
        report_csd_logvar = bool(
            getattr(args, "use_csd_logvar_sidecar", False)
        )

    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if (
        hasattr(eval_dataloader.dataset, "multi_sentence_per_video")
        and eval_dataloader.dataset.multi_sentence_per_video
    ):
        multi_sentence_ = True
        cut_off_points_ = eval_dataloader.dataset.cut_off_points
        sentence_num_ = eval_dataloader.dataset.sentence_num
        video_num_ = eval_dataloader.dataset.video_num

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad(), _eval_autocast_context(args):
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        progress_bar = tqdm(
            eval_dataloader, desc="Evaluating", leave=False, disable=getattr(args, "local_rank", 0) != 0
        )
        for bid, batch in enumerate(progress_bar):
            batch = tuple(t.to(device) for t in batch)
            if len(batch) == 5:
                input_ids, input_mask, segment_ids, video, video_mask = batch
            elif len(batch) == 8:
                input_ids, input_mask, segment_ids, _input_ids_a, _input_mask_a, _segment_ids_a, video, video_mask = batch
            else:
                raise ValueError(f"Unexpected batch size={len(batch)} from dataloader.")

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output, text_token = model.get_sequence_output(
                    input_ids,
                    segment_ids,
                    input_mask,
                )
                batch_sequence_output_list.append(
                    (sequence_output.cpu(), text_token.cpu())
                )
                batch_list_t.append((input_mask.cpu(), segment_ids.cpu()))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = select_multi_sentence_video_rows(
                    batch_start=s_,
                    batch_size=b,
                    cut_off_points=cut_off_points_,
                )

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    # June arch (--use_uatvr_full) returns (visual_cls, visual_hidden);
                    # HEAD returns a single tensor. Preserve both parts for June.
                    if isinstance(visual_output, tuple):
                        batch_visual_output_list.append(tuple(t.cpu() for t in visual_output))
                    else:
                        batch_visual_output_list.append(visual_output.cpu())
                    batch_list_v.append((video_mask.cpu(),))
                total_video_num += b
            else:
                gsvo = model.get_sequence_visual_output(
                    input_ids,
                    segment_ids,
                    input_mask,
                    video,
                    video_mask,
                )
                # HEAD returns 3-tuple (seq, text_token, visual);
                # June returns 4-tuple (seq, hidden_word, visual_cls, visual_hidden).
                if len(gsvo) == 4:
                    sequence_output, text_token, visual_output, visual_hidden = gsvo
                    batch_visual_output_list.append(
                        (visual_output.cpu(), visual_hidden.cpu())
                    )
                else:
                    sequence_output, text_token, visual_output = gsvo
                    batch_visual_output_list.append(visual_output.cpu())

                # Move to CPU immediately to reduce peak GPU memory during similarity computation
                batch_sequence_output_list.append(
                    (sequence_output.cpu(), text_token.cpu())
                )
                batch_list_t.append((input_mask.cpu(), segment_ids.cpu()))
                batch_list_v.append((video_mask.cpu(),))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        torch.cuda.empty_cache()
        if (
            n_gpu > 1
            and "LOCAL_RANK" not in os.environ
        ):
            device_ids = _select_eval_device_ids(n_gpu, len(batch_list_t))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device("cuda:{}".format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [
                        tuple(tensor.to(devc) for tensor in output)
                        for output in batch_sequence_output_list[s_:e_]
                    ]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [
                        tuple(tensor.to(devc) for tensor in output)
                        if isinstance(output, tuple)
                        else output.to(devc)
                        for output in batch_visual_output_list
                    ]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [
                (
                    args,
                    batch_list_t_splits[dev_id],
                    batch_list_v_splits[dev_id],
                    batch_t_output_splits[dev_id],
                    batch_v_output_splits[dev_id],
                )
                for dev_id in device_ids
            ]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            if getattr(args, "use_csd_logvar_sidecar", False):
                sim_matrix, intrinsic = _merge_csd_eval_outputs(
                    parallel_outputs
                )
            elif getattr(args, "use_contextual_risk_sidecar", False):
                sim_matrix, intrinsic = _merge_contextual_eval_outputs(
                    parallel_outputs
                )
            else:
                sim_matrix = []
                for idx in range(len(parallel_outputs)):
                    sim_matrix += parallel_outputs[idx]
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
                intrinsic = None
            v2t_directional_matrix = sim_matrix
            mus_matrix = sim_matrix
        else:
            retrieval_output = _run_on_single_gpu(
                model,
                args,
                batch_list_t,
                batch_list_v,
                batch_sequence_output_list,
                batch_visual_output_list,
            )
            if getattr(args, "use_csd_logvar_sidecar", False):
                sim_matrix, intrinsic = _merge_csd_eval_outputs(
                    [retrieval_output]
                )
            elif getattr(args, "use_contextual_risk_sidecar", False):
                sim_matrix, intrinsic = _merge_contextual_eval_outputs(
                    [retrieval_output]
                )
            else:
                sim_matrix = np.concatenate(tuple(retrieval_output), axis=0)
                intrinsic = None
            v2t_directional_matrix = sim_matrix
            mus_matrix = sim_matrix

    risk_similarity_matrix = np.array(sim_matrix, copy=True)

    # MUS 诊断日志：在相似度矩阵整合完成后、指标计算前输出
    if getattr(args, "log_mus_scores", False):
        _log_mus_scores_tsv(args, mus_matrix)

    if args.DSL:  # using dual softmax during evaluation
        logger.info(
            "\t Using Dual Softmax evaluation (temperature=%.3f).",
            args.dsl_temperature,
        )
        sim_matrix, v2t_directional_matrix = _apply_dual_softmax(
            sim_matrix, args.dsl_temperature
        )
    else:
        v2t_directional_matrix = sim_matrix

    metric_t2v_matrix = sim_matrix
    metric_v2t_matrix = v2t_directional_matrix

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        grouped_shape = _reshape_multi_sentence_matrix(
            metric_t2v_matrix,
            cut_off_points_,
        ).shape
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                grouped_shape[0], grouped_shape[1], grouped_shape[2]
            )
        )

        directional_metrics = _compute_directional_metrics(
            metric_t2v_matrix,
            metric_v2t_matrix,
            cut_off_points=cut_off_points_,
            independent_directions=False,
            directions=directions,
        )
    else:
        logger.info("Retrieval Evaluation | #Text: %d, #Video: %d", sim_matrix.shape[0], sim_matrix.shape[1])
        directional_metrics = _compute_directional_metrics(
            metric_t2v_matrix,
            metric_v2t_matrix,
            directions=directions,
        )

    for direction in directions:
        metrics = directional_metrics[direction]
        logger.info(
            "  %s | R@1: %5.1f  R@5: %5.1f  R@10: %5.1f  MdR: %5.1f  MnR: %5.1f",
            direction.upper(),
            metrics["R1"], metrics["R5"], metrics["R10"], metrics["MR"], metrics["MeanR"],
        )

    result = {
        direction: _serialize_retrieval_metrics(directional_metrics[direction])
        for direction in directions
    }
    report_retrieval_risk = bool(
        getattr(model, "use_retrieval_risk_sidecar", False)
    )
    report_contextual_risk = bool(
        getattr(model, "use_contextual_risk_sidecar", False)
    )
    positive_mask = None
    if report_retrieval_risk or report_contextual_risk or report_csd_logvar:
        positive_mask = build_eval_positive_mask(
            metric_t2v_matrix.shape[0],
            metric_t2v_matrix.shape[1],
            cut_off_points_ if multi_sentence_ else None,
        )
    if report_retrieval_risk:
        result["retrieval_risk"], risk_array_digests = (
            _compute_retrieval_risk_metrics(
                model=model,
                risk_similarity_matrix=risk_similarity_matrix,
                metric_t2v_matrix=metric_t2v_matrix,
                metric_v2t_matrix=metric_v2t_matrix,
                positive_mask=positive_mask,
                directions=directions,
                device=device,
                bootstrap_samples=getattr(
                    args, "retrieval_risk_bootstrap_samples", 1000
                ),
                return_array_digests=True,
                return_array_evidence=report_contextual_risk,
                canonical_cpu_replay=report_contextual_risk,
                artifact_array_store=artifact_array_store,
            )
        )
        for direction in directions:
            risk_metrics = result["retrieval_risk"][direction]
            logger.info(
                "  %s risk | rho=%+.4f AUROC=%.4f AURC=%.4f",
                direction.upper(),
                risk_metrics["spearman_risk_vs_rank"],
                risk_metrics["failure_auroc"]["value"],
                risk_metrics["risk_coverage"]["aurc"],
            )
        result["paired_digest"] = _build_paired_digest(
            result,
            risk_similarity_matrix,
            risk_array_digests,
            directions,
            text_means=(intrinsic["text_mu"] if report_contextual_risk else None),
            video_means=(intrinsic["video_mu"] if report_contextual_risk else None),
        )
    if report_contextual_risk:
        assert intrinsic is not None
        (
            result["contextual_risk"],
            result["contextual_risk_evidence"],
        ) = _compute_contextual_retrieval_risk_metrics(
            model=model,
            risk_similarity_matrix=risk_similarity_matrix,
            positive_mask=positive_mask,
            text_means=intrinsic["text_mu"],
            video_means=intrinsic["video_mu"],
            directions=directions,
            device=device,
            bootstrap_samples=getattr(
                args, "contextual_risk_bootstrap_samples", 1000
            ),
            return_evidence=True,
            artifact_array_store=artifact_array_store,
        )
        a3_evidence = result["contextual_risk_evidence"]
        contextual_sidecar = getattr(model, "contextual_risk_sidecar", None)
        if contextual_sidecar is not None:
            a3_evidence["contextual_head_config"] = {
                "embed_dim": int(contextual_sidecar.embed_dim),
                "projection_dim": int(
                    contextual_sidecar.text_projection.out_features
                ),
                "hidden_dim": int(contextual_sidecar.network[0].out_features),
                "topk": int(contextual_sidecar.topk),
                "temperature": float(contextual_sidecar.temperature),
                "dropout": float(contextual_sidecar.network[2].p),
            }
        if report_retrieval_risk:
            a3_evidence["a1_failure_risk"] = {
                direction: risk_array_digests[direction]["failure_risk_values"]
                for direction in directions
            }
            a3_evidence["a1_vacuity"] = {
                direction: risk_array_digests[direction]["vacuity_values"]
                for direction in directions
            }

            def _a1_array_metadata(direction, name):
                stored = risk_array_digests[direction]
                reference = stored.get(f"{name}_metadata")
                if reference is not None:
                    return reference
                return {
                    "dtype": stored[f"{name}_dtype"],
                    "shape": stored[f"{name}_shape"],
                    "sha256": stored[name],
                }

            a3_evidence["array_metadata"]["a1_failure_risk"] = {
                direction: _a1_array_metadata(direction, "failure_risk")
                for direction in directions
            }
            a3_evidence["array_metadata"]["a1_vacuity"] = {
                direction: _a1_array_metadata(direction, "vacuity")
                for direction in directions
            }
            a1_sidecar = getattr(model, "retrieval_risk_sidecar", None)
            if a1_sidecar is not None:
                a3_evidence["a1_head_config"] = {
                    "feature_dim": int(a1_sidecar.network[0].in_features),
                    "hidden_dim": int(a1_sidecar.network[0].out_features),
                    "temperature": float(a1_sidecar.temperature),
                }
        retrieval_score_arrays = {
            "t2v": np.asarray(metric_t2v_matrix),
            "v2t": np.asarray(metric_v2t_matrix),
        }
        if artifact_array_store is None:
            a3_evidence["retrieval_scores"] = {
                direction: retrieval_score_arrays[direction].tolist()
                for direction in directions
            }
            a3_evidence["array_metadata"]["retrieval_scores"] = {
                direction: {
                    "dtype": str(retrieval_score_arrays[direction].dtype),
                    "shape": list(retrieval_score_arrays[direction].shape),
                    "sha256": array_sha256(retrieval_score_arrays[direction]),
                }
                for direction in directions
            }
        else:
            references = {
                direction: artifact_array_store.write(
                    f"retrieval_scores.{direction}", retrieval_score_arrays[direction]
                ).as_dict()
                for direction in directions
            }
            a3_evidence["retrieval_scores"] = references
            a3_evidence["array_metadata"]["retrieval_scores"] = references
        a3_evidence["retrieval_config"] = {
            "directions": list(directions),
            "cut_off_points": (
                [int(value) for value in cut_off_points_]
                if multi_sentence_
                else "one_to_one"
            ),
            "independent_directions": not multi_sentence_,
            "dsl": bool(getattr(args, "DSL", False)),
            "dsl_temperature": float(getattr(args, "dsl_temperature", 1.0)),
            "score_path": "post_dsl_directional_scores",
        }
        for direction in directions:
            contextual_metrics = result["contextual_risk"][direction]
            logger.info(
                "  %s contextual risk | rho=%+.4f AUROC=%.4f AURC=%.4f",
                direction.upper(),
                contextual_metrics["spearman_risk_vs_rank"],
                contextual_metrics["failure_auroc"]["value"],
                contextual_metrics["aurc"],
            )
        if artifact_array_store is not None:
            validate_artifact_json_payload(a3_evidence)
    if report_csd_logvar:
        assert intrinsic is not None
        result["csd_logvar"] = _compute_csd_logvar_metrics(
            metric_t2v_matrix=metric_t2v_matrix,
            metric_v2t_matrix=metric_v2t_matrix,
            positive_mask=positive_mask,
            intrinsic=intrinsic,
            bootstrap_samples=getattr(args, "csd_bootstrap_samples", 1000),
            logvar_min=getattr(args, "csd_logvar_min", -12.0),
            logvar_max=getattr(args, "csd_logvar_max", 4.0),
        )
        for direction in ("t2v", "v2t"):
            intrinsic_metrics = result["csd_logvar"][direction]
            logger.info(
                "  %s intrinsic | rho=%+.4f AUROC=%.4f AURC=%.4f",
                direction.upper(),
                intrinsic_metrics["spearman_uncertainty_vs_rank"],
                intrinsic_metrics["failure_auroc"]["value"],
                intrinsic_metrics["risk_coverage"]["aurc"],
            )
    return result


def eval_epoch(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    directions=("t2v", "v2t"),
    report_csd_logvar=None,
    artifact_array_store=None,
):
    """Evaluate with exact FP32 backend flags held through A3 scoring."""

    with _a3_exact_fp32_backend(args):
        return _eval_epoch_impl(
            args,
            model,
            eval_dataloader,
            device,
            n_gpu,
            directions=directions,
            report_csd_logvar=report_csd_logvar,
            artifact_array_store=artifact_array_store,
        )


def _candidate_clamp_ok(metrics):
    return all(
        metrics[name]["clamp_low_fraction"]
        + metrics[name]["clamp_high_fraction"]
        < 0.05
        for name in ("text_distribution", "video_distribution")
    )


def _contextual_evidence_array(evidence, name, direction=None):
    """Reconstruct one registered evidence array and verify its typed digest."""

    values = evidence[name]
    metadata = evidence["array_metadata"][name]
    legacy_sha = evidence.get("sha256", {}).get(name)
    if direction is not None:
        values = values[direction]
        metadata = metadata[direction]
        if isinstance(legacy_sha, dict):
            legacy_sha = legacy_sha.get(direction)
    expected_dtypes = {
        "raw_wti": np.dtype("float32"),
        "text_means": np.dtype("float32"),
        "video_means": np.dtype("float32"),
        "positive_mask": np.dtype("bool"),
        "failure_risk": np.dtype("float32"),
        "a1_failure_risk": np.dtype("float32"),
        "a1_vacuity": np.dtype("float32"),
        "retrieval_scores": np.dtype("float32"),
    }
    expected_dtype = expected_dtypes[name]
    if isinstance(values, dict) and "path" in values:
        artifact_root = evidence.get("artifact_array_root")
        if not isinstance(artifact_root, str):
            raise ValueError("A3 external array evidence is missing its artifact root")
        array = verify_array_reference(artifact_root, values)
        if values != metadata:
            raise ValueError("A3 external array metadata differs from its reference")
        if array.dtype != expected_dtype:
            raise ValueError(
                f"A3 evidence {name} dtype must be {expected_dtype}, got {array.dtype}"
            )
        legacy_sha = evidence.get("sha256", {}).get(name)
        if direction is not None and isinstance(legacy_sha, dict):
            legacy_sha = legacy_sha.get(direction)
        if legacy_sha is not None and legacy_sha != array_sha256(array):
            raise ValueError(f"A3 evidence {name} legacy SHA-256 mismatch")
        return array
    dtype = np.dtype(metadata["dtype"])
    if dtype != expected_dtype:
        raise ValueError(
            f"A3 evidence {name} dtype must be {expected_dtype}, got {dtype}"
        )
    array = np.asarray(values, dtype=dtype)
    if list(array.shape) != list(metadata["shape"]):
        raise ValueError(f"A3 evidence {name} shape mismatch")
    digest = array_sha256(array)
    if digest != metadata["sha256"]:
        raise ValueError(f"A3 evidence {name} SHA-256 mismatch")
    if legacy_sha is not None and legacy_sha != digest:
        raise ValueError(f"A3 evidence {name} legacy SHA-256 mismatch")
    return array


def _registered_retrieval_config(positive_mask, *, dsl, dsl_temperature):
    """Derive the complete immutable retrieval configuration from inputs."""

    mask = np.asarray(positive_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("A3 positive mask must be two-dimensional")
    one_to_one = (
        mask.shape[0] == mask.shape[1]
        and np.array_equal(mask, np.eye(mask.shape[0], dtype=bool))
    )
    if one_to_one:
        cut_off_points = "one_to_one"
    else:
        if not (mask.sum(axis=1) == 1).all():
            raise ValueError("A3 multi-sentence positive mask is not canonical")
        video_indices = mask.argmax(axis=1)
        expected_indices = np.repeat(
            np.arange(mask.shape[1]), mask.sum(axis=0)
        )
        if not np.array_equal(video_indices, expected_indices):
            raise ValueError("A3 multi-sentence positive groups are not contiguous")
        cut_off_points = np.cumsum(mask.sum(axis=0)).astype(int).tolist()
    return {
        "directions": ["t2v", "v2t"],
        "cut_off_points": cut_off_points,
        "independent_directions": bool(one_to_one),
        "dsl": bool(dsl),
        "dsl_temperature": float(dsl_temperature),
        "score_path": "post_dsl_directional_scores",
    }


def _registered_a3_head_configs(args, evidence):
    """Return immutable sidecar construction settings from registered args."""

    text_means = _contextual_evidence_array(evidence, "text_means")
    if text_means.ndim != 2:
        raise ValueError("A3 text means must be two-dimensional")
    return {
        "contextual": {
            "embed_dim": int(text_means.shape[1]),
            "projection_dim": int(
                getattr(args, "contextual_risk_projection_dim", 128)
            ),
            "hidden_dim": int(
                getattr(args, "contextual_risk_hidden_dim", 128)
            ),
            "topk": int(getattr(args, "contextual_risk_topk", 16)),
            "temperature": float(
                getattr(args, "contextual_risk_temperature", 1.0)
            ),
            "dropout": 0.1,
        },
        "a1": {
            "feature_dim": 8,
            "hidden_dim": int(
                getattr(args, "retrieval_risk_hidden_dim", 32)
            ),
            "temperature": float(
                getattr(args, "retrieval_risk_temperature", 1.0)
            ),
        },
    }


def _checkpoint_sidecar_state(checkpoint_path, prefix):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("A3 checkpoint state must be a mapping")
    sidecar_state = {
        name[len(prefix):]: tensor.detach().cpu().float()
        for name, tensor in state.items()
        if name.startswith(prefix)
    }
    if not sidecar_state:
        raise ValueError(f"A3 checkpoint is missing {prefix} state")
    return sidecar_state


def _verify_contextual_risk_checkpoint_outputs(
    checkpoint_path,
    evidence,
    *,
    expected_head_configs,
    verify_contextual=True,
):
    """Bind stored A3/A1 output arrays to the exact checkpoint head states."""

    raw_wti = _contextual_evidence_array(evidence, "raw_wti")
    text_means = _contextual_evidence_array(evidence, "text_means")
    video_means = _contextual_evidence_array(evidence, "video_means")
    contextual_config = evidence.get("contextual_head_config")
    a1_config = evidence.get("a1_head_config")
    if not isinstance(a1_config, dict) or (
        verify_contextual and not isinstance(contextual_config, dict)
    ):
        raise ValueError("A3 checkpoint head configuration evidence is incomplete")
    if a1_config != expected_head_configs["a1"] or (
        verify_contextual
        and contextual_config != expected_head_configs["contextual"]
    ):
        raise ValueError("A3 checkpoint head configuration differs from manifest")

    contextual = None
    if verify_contextual:
        contextual = ContextualRetrievalRiskSidecar(
            **expected_head_configs["contextual"]
        ).cpu().eval()
        contextual.load_state_dict(
            _checkpoint_sidecar_state(
                checkpoint_path, "contextual_risk_sidecar."
            ),
            strict=True,
        )
    retrieval = RetrievalRiskSidecar(**a1_config).cpu().eval()
    retrieval.load_state_dict(
        _checkpoint_sidecar_state(checkpoint_path, "retrieval_risk_sidecar."),
        strict=True,
    )
    raw_tensor = torch.from_numpy(np.asarray(raw_wti, dtype=np.float32))
    text_tensor = torch.from_numpy(np.asarray(text_means, dtype=np.float32))
    video_tensor = torch.from_numpy(np.asarray(video_means, dtype=np.float32))
    with _a3_exact_fp32_backend(
        argparse.Namespace(contextual_risk_sidecar_only=True)
    ), torch.inference_mode():
        for direction in ("t2v", "v2t"):
            if verify_contextual:
                contextual_output = contextual(
                    raw_tensor,
                    text_tensor if direction == "t2v" else video_tensor,
                    video_tensor if direction == "t2v" else text_tensor,
                    direction,
                )["failure_risk"].numpy()
                stored_contextual = _contextual_evidence_array(
                    evidence, "failure_risk", direction
                )
                if not np.array_equal(contextual_output, stored_contextual):
                    raise ValueError(
                        "A3 checkpoint-derived contextual output differs from evidence"
                    )
            a1_output = retrieval(raw_tensor, direction)
            for evidence_name, output_name in (
                ("a1_failure_risk", "failure_risk"),
                ("a1_vacuity", "vacuity"),
            ):
                stored_a1 = _contextual_evidence_array(
                    evidence, evidence_name, direction
                )
                derived_a1 = a1_output[output_name].numpy()
                if not np.array_equal(derived_a1, stored_a1):
                    raise ValueError(
                        "A3 checkpoint-derived A1 output differs from evidence"
                    )


def _recompute_contextual_risk_metrics(
    evidence,
    *,
    bootstrap_samples,
    expected_retrieval_config=None,
):
    """Rebuild registered A3 summaries solely from complete typed evidence."""

    config = evidence["metric_config"]
    expected_seeds = {"t2v": 0, "v2t": 1}
    if int(config["bootstrap_samples"]) != int(bootstrap_samples):
        raise ValueError("A3 evidence bootstrap configuration mismatch")
    if config.get("seed_by_direction") != expected_seeds:
        raise ValueError("A3 evidence seed configuration mismatch")

    raw_wti = _contextual_evidence_array(evidence, "raw_wti")
    positive_mask = _contextual_evidence_array(evidence, "positive_mask")
    _contextual_evidence_array(evidence, "text_means")
    _contextual_evidence_array(evidence, "video_means")
    if raw_wti.shape != positive_mask.shape:
        raise ValueError("A3 raw WTI and positive-mask shapes differ")

    metrics = {}
    digests = {
        "raw_wti_sha256": array_sha256(raw_wti),
        "text_means_sha256": array_sha256(
            _contextual_evidence_array(evidence, "text_means")
        ),
        "video_means_sha256": array_sha256(
            _contextual_evidence_array(evidence, "video_means")
        ),
        "a1_failure_risk_sha256": {},
        "a1_vacuity_sha256": {},
    }
    for direction in ("t2v", "v2t"):
        risk = _contextual_evidence_array(
            evidence, "failure_risk", direction
        )
        a1_risk = _contextual_evidence_array(
            evidence, "a1_failure_risk", direction
        )
        a1_vacuity = _contextual_evidence_array(
            evidence, "a1_vacuity", direction
        )
        digests["a1_failure_risk_sha256"][direction] = array_sha256(a1_risk)
        digests["a1_vacuity_sha256"][direction] = array_sha256(a1_vacuity)
        scores = raw_wti if direction == "t2v" else raw_wti.T
        mask = positive_mask if direction == "t2v" else positive_mask.T
        metrics[direction] = evaluate_retrieval_risk(
            scores,
            mask,
            risk,
            bootstrap_samples=int(bootstrap_samples),
            seed=expected_seeds[direction],
        )
    retrieval_config = evidence["retrieval_config"]
    if (
        expected_retrieval_config is not None
        and retrieval_config != expected_retrieval_config
    ):
        raise ValueError("A3 retrieval configuration mismatch")
    retrieval_directions = tuple(retrieval_config["directions"])
    if retrieval_directions != ("t2v", "v2t"):
        raise ValueError("A3 retrieval evidence must contain both directions")
    if retrieval_config.get("score_path") != "post_dsl_directional_scores":
        raise ValueError("A3 retrieval evidence score path mismatch")
    retrieval_t2v = _contextual_evidence_array(
        evidence, "retrieval_scores", "t2v"
    )
    retrieval_v2t = _contextual_evidence_array(
        evidence, "retrieval_scores", "v2t"
    )
    if retrieval_t2v.shape != raw_wti.shape or retrieval_v2t.shape != raw_wti.shape:
        raise ValueError("A3 retrieval evidence matrix shape mismatch")
    dsl_enabled = retrieval_config.get("dsl")
    if not isinstance(dsl_enabled, bool):
        raise ValueError("A3 retrieval evidence DSL flag must be boolean")
    dsl_temperature = float(retrieval_config.get("dsl_temperature"))
    if not math.isfinite(dsl_temperature) or dsl_temperature <= 0:
        raise ValueError("A3 retrieval evidence DSL temperature is invalid")
    if dsl_enabled:
        expected_t2v, expected_v2t = _apply_dual_softmax(
            raw_wti, dsl_temperature
        )
    else:
        expected_t2v, expected_v2t = raw_wti, raw_wti
    if not np.array_equal(retrieval_t2v, expected_t2v):
        raise ValueError("A3 retrieval t2v score evidence differs from raw WTI/DSL")
    if not np.array_equal(retrieval_v2t, expected_v2t):
        raise ValueError("A3 retrieval v2t score evidence differs from raw WTI/DSL")
    recorded_cut_off_points = retrieval_config.get("cut_off_points")
    cut_off_points = (
        None
        if recorded_cut_off_points == "one_to_one"
        else recorded_cut_off_points
    )
    independent_directions = bool(
        retrieval_config.get("independent_directions", True)
    )
    directional_metrics = _compute_directional_metrics(
        retrieval_t2v,
        retrieval_v2t,
        cut_off_points=cut_off_points,
        independent_directions=independent_directions,
        directions=retrieval_directions,
    )
    digests["retrieval"] = {
        direction: _serialize_retrieval_metrics(directional_metrics[direction])
        for direction in retrieval_directions
    }
    return metrics, digests


def _build_contextual_risk_candidate(
    *,
    epoch,
    checkpoint_path,
    optimizer_checkpoint_path,
    validation_metrics,
    paired_baseline,
    reference_checkpoint,
    bootstrap_samples,
    expected_retrieval_config,
    expected_head_configs,
):
    """Build one complete, validation-only A3 candidate record."""

    model_path = Path(checkpoint_path).resolve()
    optimizer_path = Path(optimizer_checkpoint_path).resolve()
    contextual = validation_metrics["contextual_risk"]
    paired_digest = validation_metrics["paired_digest"]
    paired_invariance = compare_uncertainty_paired_invariance(
        paired_baseline,
        paired_digest,
        A3_REQUIRED_PAIRED_DIGEST_KEYS,
    )
    mismatches = non_sidecar_state_dict_mismatches(
        reference_checkpoint,
        model_path,
        ("contextual_risk_sidecar.",),
    )
    evidence = validation_metrics["contextual_risk_evidence"]
    _verify_contextual_risk_checkpoint_outputs(
        model_path,
        evidence,
        expected_head_configs=expected_head_configs,
    )
    recomputed_contextual, evidence_digests = _recompute_contextual_risk_metrics(
        evidence,
        bootstrap_samples=bootstrap_samples,
        expected_retrieval_config=expected_retrieval_config,
    )
    if contextual != recomputed_contextual:
        raise ValueError("A3 validation metric summary differs from evidence")
    for digest_name, digest_value in evidence_digests.items():
        if paired_digest.get(digest_name) != digest_value:
            raise ValueError(
                f"A3 paired digest {digest_name} differs from complete evidence"
            )
    finite = numeric_metrics_are_finite(
        {
            "contextual_risk": contextual,
            "contextual_risk_evidence": evidence,
            "paired_digest": paired_digest,
        }
    )
    gate_metrics = {
        "t2v": contextual["t2v"],
        "v2t": contextual["v2t"],
        "paired_invariance": paired_invariance,
        "non_a3_mismatches": mismatches,
    }
    candidate = {
        "epoch": int(epoch),
        "path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "optimizer_path": str(optimizer_path),
        "optimizer_sha256": file_sha256(optimizer_path),
        "valid": bool(finite and not mismatches),
        "finite": finite,
        "runtime_abort": False,
        "paired_digest": paired_digest,
        "paired_invariance": paired_invariance,
        "non_a3_mismatches": mismatches,
        "rho": [
            contextual[direction]["spearman_risk_vs_rank"]
            for direction in ("t2v", "v2t")
        ],
        "auroc": [
            contextual[direction]["failure_auroc"]["value"]
            for direction in ("t2v", "v2t")
        ],
        "aurc": [
            contextual[direction]["aurc"] for direction in ("t2v", "v2t")
        ],
        "evidence": evidence,
        "metrics": gate_metrics,
    }
    candidate["acceptance"] = contextual_risk_acceptance(gate_metrics)
    candidate["valid"] = bool(
        candidate["valid"] and contextual_risk_candidate_is_valid(candidate)
    )
    return candidate


def _contextual_risk_selection_payload(candidates, winner):
    return {
        "selection_split": "val",
        "selection_metric": (
            "max(min_directional_rho,mean_directional_auroc,"
            "-mean_directional_aurc,-epoch)"
        ),
        "tie_break": "earlier_epoch",
        "candidates": candidates,
        "winner": winner,
    }


def _contextual_risk_should_continue(candidates):
    """Apply the registered epoch-2 versus epoch-1 continuation decision."""

    by_epoch = {int(candidate["epoch"]): candidate for candidate in candidates}
    if sorted(by_epoch) != [0, 1, 2] or len(candidates) != 3:
        raise ValueError(
            "A3 continuation decision requires exactly epoch indices 0, 1, and 2"
        )
    return contextual_risk_selection_tuple(by_epoch[2]) > (
        contextual_risk_selection_tuple(by_epoch[1])
    )


def _contextual_risk_continuation_eligible(candidates, *, output_dir):
    """Return the single registered artifact-valid A3 continuation decision."""

    validate_candidate_artifacts(
        candidates, output_dir=output_dir, method_name="A3"
    )
    by_epoch = {int(candidate["epoch"]): candidate for candidate in candidates}
    if sorted(by_epoch) != [0, 1, 2] or len(candidates) != 3:
        raise ValueError(
            "A3 continuation decision requires exactly epoch indices 0, 1, and 2"
        )
    return bool(
        contextual_risk_candidate_is_valid(by_epoch[2])
        and contextual_risk_selection_tuple(by_epoch[2])
        > contextual_risk_selection_tuple(by_epoch[1])
    )


def _reaudit_contextual_risk_candidate(
    stored,
    *,
    paired_baseline,
    reference_checkpoint,
    bootstrap_samples,
    expected_retrieval_config,
    expected_head_configs,
):
    """Recompute every derivable A3 audit field without mutating its record."""

    candidate = dict(stored)
    contextual = candidate["metrics"]
    paired_digest = candidate["paired_digest"]
    recomputed_contextual, evidence_digests = _recompute_contextual_risk_metrics(
        candidate["evidence"],
        bootstrap_samples=bootstrap_samples,
        expected_retrieval_config=expected_retrieval_config,
    )
    if {
        "t2v": contextual["t2v"], "v2t": contextual["v2t"]
    } != recomputed_contextual:
        raise ValueError("A3 immutable candidate metric summary mismatch")
    for digest_name, digest_value in evidence_digests.items():
        if paired_digest.get(digest_name) != digest_value:
            raise ValueError(
                f"A3 immutable candidate {digest_name} evidence mismatch"
            )
    paired_invariance = compare_uncertainty_paired_invariance(
        paired_baseline,
        paired_digest,
        A3_REQUIRED_PAIRED_DIGEST_KEYS,
    )
    if paired_invariance["passed"]:
        _verify_contextual_risk_checkpoint_outputs(
            candidate["path"],
            candidate["evidence"],
            expected_head_configs=expected_head_configs,
        )
    mismatches = non_sidecar_state_dict_mismatches(
        reference_checkpoint,
        candidate["path"],
        ("contextual_risk_sidecar.",),
    )
    finite = numeric_metrics_are_finite({
        "contextual_risk": recomputed_contextual,
        "evidence": candidate["evidence"],
        "paired_digest": paired_digest,
    })
    gate_metrics = {
        "t2v": recomputed_contextual["t2v"],
        "v2t": recomputed_contextual["v2t"],
        "paired_invariance": paired_invariance,
        "non_a3_mismatches": mismatches,
    }
    audited = {
        **candidate,
        "finite": finite,
        "paired_invariance": paired_invariance,
        "non_a3_mismatches": mismatches,
        "rho": [
            contextual[direction]["spearman_risk_vs_rank"]
            for direction in ("t2v", "v2t")
        ],
        "auroc": [
            contextual[direction]["failure_auroc"]["value"]
            for direction in ("t2v", "v2t")
        ],
        "aurc": [
            contextual[direction]["aurc"] for direction in ("t2v", "v2t")
        ],
        "metrics": gate_metrics,
        "acceptance": contextual_risk_acceptance(gate_metrics),
    }
    audited["valid"] = bool(
        finite
        and not mismatches
        and contextual_risk_candidate_is_valid(
            {**audited, "valid": True}
        )
    )
    immutable_audits = (
        "finite",
        "paired_invariance",
        "non_a3_mismatches",
        "rho",
        "auroc",
        "aurc",
        "metrics",
        "acceptance",
        "valid",
    )
    changed = [name for name in immutable_audits if candidate.get(name) != audited[name]]
    if changed:
        raise ValueError(
            "A3 immutable candidate audit mismatch: " + ", ".join(changed)
        )
    return audited


def _canonical_contextual_baseline_projection(payload):
    """Select only canonical A1-derived fields from one baseline payload."""

    evidence = payload["evidence"]
    canonical_array_names = (
        "raw_wti",
        "text_means",
        "video_means",
        "positive_mask",
        "a1_failure_risk",
        "a1_vacuity",
        "retrieval_scores",
    )
    return {
        "paired_digest": payload["paired_digest"],
        "evidence": {
            **{name: evidence[name] for name in canonical_array_names},
            "array_metadata": {
                name: evidence["array_metadata"][name]
                for name in canonical_array_names
            },
            "retrieval_config": evidence["retrieval_config"],
            "a1_head_config": evidence["a1_head_config"],
        },
    }


def _compare_regenerated_contextual_baseline(registration, regenerated):
    """Reject any persisted canonical baseline differing from fresh A1 eval."""

    if _canonical_contextual_baseline_projection(registration) != (
        _canonical_contextual_baseline_projection(regenerated)
    ):
        raise ValueError(
            "A3 persisted baseline differs from regenerated canonical baseline"
        )


def _regenerate_contextual_risk_baseline(
    args,
    eval_dataloader,
    device,
    n_gpu,
    reference_checkpoint,
):
    """Re-evaluate trusted validation from the exact canonical A1 checkpoint."""

    canonical_model = load_model(
        -1,
        args,
        n_gpu,
        device,
        model_file=reference_checkpoint,
        registered_a3_fp32=True,
    )
    if canonical_model is None:
        raise FileNotFoundError("canonical A1 checkpoint could not be loaded")
    baseline_eval = eval_epoch(
        args,
        canonical_model,
        eval_dataloader,
        device,
        n_gpu,
        report_csd_logvar=False,
        artifact_array_store=getattr(args, "artifact_array_store", None),
    )
    return {
        "reference_checkpoint": str(Path(reference_checkpoint).resolve()),
        "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
        "paired_digest": baseline_eval["paired_digest"],
        "evidence": baseline_eval["contextual_risk_evidence"],
    }


def _validate_a3_candidate_publication(candidate, *, output_dir):
    """Bind a persisted A3 candidate to its committed checkpoint pair."""

    marker_value = candidate.get("publication_marker")
    marker_sha = candidate.get("publication_marker_sha256")
    if not isinstance(marker_value, str) or not marker_value or not marker_sha:
        raise ValueError("A3 candidate publication marker evidence is missing")
    expected = _checkpoint_publication_marker_path(
        int(candidate["epoch"]), argparse.Namespace(output_dir=str(output_dir))
    ).resolve()
    marker_path = Path(marker_value).resolve()
    if marker_path != expected:
        raise ValueError("A3 candidate publication marker path mismatch")
    if file_sha256(marker_path) != marker_sha:
        raise ValueError("A3 candidate publication marker SHA-256 mismatch")
    return _validate_checkpoint_publication(
        marker_path,
        epoch=int(candidate["epoch"]),
        model_path=candidate["path"],
        optimizer_path=candidate["optimizer_path"],
    )


def _validate_a3_publication_directory(output_dir):
    """Reject retained staging or an uncommitted checkpoint pair at startup."""

    output_path = Path(output_dir)
    staging = sorted(output_path.glob(".pytorch_*.staging"))
    if staging:
        raise ValueError(
            "A3 incomplete staged checkpoint retained for diagnosis: "
            + ", ".join(str(path) for path in staging)
        )
    epochs = set()
    for pattern, prefix in (
        ("pytorch_model.bin.*", "pytorch_model.bin."),
        ("pytorch_opt.bin.*", "pytorch_opt.bin."),
    ):
        for path in output_path.glob(pattern):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                epochs.add(int(suffix))
    for epoch in sorted(epochs):
        model_path = output_path / f"pytorch_model.bin.{epoch}"
        optimizer_path = output_path / f"pytorch_opt.bin.{epoch}"
        marker_path = _checkpoint_publication_marker_path(
            epoch, argparse.Namespace(output_dir=str(output_path))
        )
        try:
            _validate_checkpoint_publication(
                marker_path,
                epoch=epoch,
                model_path=model_path,
                optimizer_path=optimizer_path,
            )
        except ValueError as exc:
            raise ValueError(
                f"A3 incomplete unpublished checkpoint pair at epoch {epoch}: {exc}"
            ) from exc


def _initialize_contextual_risk_protocol(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    resume_payload=None,
):
    """Register a fresh canonical A1 baseline or validate one A3 continuation."""

    output_dir = Path(args.output_dir)
    _validate_a3_publication_directory(output_dir)
    baseline_path = output_dir / "a3_paired_baseline.json"
    candidate_path = output_dir / "contextual_risk_candidates.json"
    selection_path = output_dir / "best_contextual_risk_checkpoint.json"
    continuing = bool(getattr(args, "resume_model", ""))
    protocol_paths = (baseline_path, candidate_path, selection_path)
    if continuing:
        missing = [path.name for path in protocol_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "A3 continuation requires original protocol artifacts: "
                + ", ".join(missing)
            )
    elif any(path.exists() for path in protocol_paths):
        raise FileExistsError(
            "fresh A3 run requires an output directory without existing "
            "A3 protocol lineage artifacts"
        )

    if continuing:
        registration = json.loads(baseline_path.read_text())
        reference_checkpoint = str(
            Path(registration["reference_checkpoint"]).resolve()
        )
        reference_sha256 = registration.get("reference_checkpoint_sha256")
        if reference_sha256 != CONTEXTUAL_RISK_CANONICAL_A1_SHA256:
            raise ValueError("A3 registered reference SHA-256 is not canonical A1")
        if file_sha256(reference_checkpoint) != reference_sha256:
            raise ValueError("A3 reference checkpoint SHA-256 mismatch")
        regenerated_baseline = _regenerate_contextual_risk_baseline(
            args,
            eval_dataloader,
            device,
            n_gpu,
            reference_checkpoint,
        )
        _compare_regenerated_contextual_baseline(
            registration, regenerated_baseline
        )
        paired_baseline = registration["paired_digest"]
        baseline_evidence = registration.get("evidence")
        if not isinstance(baseline_evidence, dict):
            raise ValueError("A3 continuation baseline complete evidence is missing")
        expected_retrieval_config = _registered_retrieval_config(
            _contextual_evidence_array(baseline_evidence, "positive_mask"),
            dsl=bool(getattr(args, "DSL", False)),
            dsl_temperature=float(getattr(args, "dsl_temperature", 1.0)),
        )
        expected_head_configs = _registered_a3_head_configs(
            args, baseline_evidence
        )
        _verify_contextual_risk_checkpoint_outputs(
            reference_checkpoint,
            baseline_evidence,
            expected_head_configs=expected_head_configs,
            verify_contextual=False,
        )
        _, baseline_digests = _recompute_contextual_risk_metrics(
            baseline_evidence,
            bootstrap_samples=getattr(
                args, "contextual_risk_bootstrap_samples", 1000
            ),
            expected_retrieval_config=expected_retrieval_config,
        )
        for digest_name, digest_value in baseline_digests.items():
            if paired_baseline.get(digest_name) != digest_value:
                raise ValueError(
                    f"A3 baseline {digest_name} differs from complete evidence"
                )
        candidates = json.loads(candidate_path.read_text())
        selection = json.loads(selection_path.read_text())
        if selection.get("candidates") != candidates:
            raise ValueError("A3 candidate and selection artifacts disagree")
        validate_candidate_artifacts(
            candidates, output_dir=output_dir, method_name="A3"
        )
        for persisted_candidate in candidates:
            _validate_a3_candidate_publication(
                persisted_candidate, output_dir=output_dir
            )
        candidates = [
            _reaudit_contextual_risk_candidate(
                candidate,
                paired_baseline=paired_baseline,
                reference_checkpoint=reference_checkpoint,
                bootstrap_samples=getattr(
                    args, "contextual_risk_bootstrap_samples", 1000
                ),
                expected_retrieval_config=expected_retrieval_config,
                expected_head_configs=expected_head_configs,
            )
            for candidate in candidates
        ]
        expected_winner = choose_contextual_risk_checkpoint(
            candidates, output_dir=output_dir
        )
        if selection.get("winner") != expected_winner:
            raise ValueError("A3 persisted winner does not match validation candidates")
        if not _contextual_risk_continuation_eligible(
            candidates, output_dir=output_dir
        ):
            raise ValueError(
                "A3 continuation is not eligible because epoch index 2 did not "
                "strictly improve over epoch index 1"
            )
        if int(getattr(args, "epochs", 5)) != 5:
            raise ValueError("A3 continuation must target five total epochs")
        if resume_payload is None:
            raise ValueError("A3 continuation requires an optimizer payload")
        resume_candidate = validate_resume_lineage(
            candidates,
            output_dir=output_dir,
            model_checkpoint=args.init_model,
            optimizer_checkpoint=args.resume_model,
            optimizer_payload=resume_payload,
            sidecar_prefix="contextual_risk_sidecar.",
            method_name="A3",
        )
        if int(resume_candidate["epoch"]) != 2:
            raise ValueError("A3 continuation must resume epoch index 2")
    else:
        reference_checkpoint = str(Path(args.init_model).resolve())
        if Path(reference_checkpoint) != CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT:
            raise ValueError(
                "fresh A3 run must initialize from the canonical A1 checkpoint"
            )
        reference_sha256 = file_sha256(reference_checkpoint)
        if reference_sha256 != CONTEXTUAL_RISK_CANONICAL_A1_SHA256:
            raise ValueError("fresh A3 reference checkpoint SHA-256 mismatch")
        if int(getattr(args, "epochs", 3)) != 3:
            raise ValueError("fresh A3 run must contain exactly three epochs")
        regenerated_baseline = _regenerate_contextual_risk_baseline(
            args,
            eval_dataloader,
            device,
            n_gpu,
            reference_checkpoint,
        )
        paired_baseline = regenerated_baseline["paired_digest"]
        baseline_evidence = regenerated_baseline["evidence"]
        expected_retrieval_config = _registered_retrieval_config(
            _contextual_evidence_array(baseline_evidence, "positive_mask"),
            dsl=bool(getattr(args, "DSL", False)),
            dsl_temperature=float(getattr(args, "dsl_temperature", 1.0)),
        )
        expected_head_configs = _registered_a3_head_configs(
            args, baseline_evidence
        )
        _verify_contextual_risk_checkpoint_outputs(
            reference_checkpoint,
            baseline_evidence,
            expected_head_configs=expected_head_configs,
            verify_contextual=False,
        )
        _recompute_contextual_risk_metrics(
            baseline_evidence,
            bootstrap_samples=getattr(
                args, "contextual_risk_bootstrap_samples", 1000
            ),
            expected_retrieval_config=expected_retrieval_config,
        )
        baseline_payload = {
            "reference_checkpoint": reference_checkpoint,
            "reference_checkpoint_sha256": reference_sha256,
            "paired_digest": paired_baseline,
            "evidence": baseline_evidence,
        }
        if "artifact_array_root" in baseline_evidence:
            validate_artifact_json_payload(baseline_payload)
        atomic_write_json(baseline_path, baseline_payload)
        candidates = []
        resume_candidate = None
    return {
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "selection_path": selection_path,
        "paired_baseline": paired_baseline,
        "reference_checkpoint": reference_checkpoint,
        "candidates": candidates,
        "resume_candidate": resume_candidate,
    }


def _record_contextual_risk_selection(
    *,
    epoch,
    checkpoint_path,
    optimizer_checkpoint_path,
    validation_metrics,
    paired_baseline,
    reference_checkpoint,
    candidate_path,
    selection_path,
    bootstrap_samples,
    expected_retrieval_config,
    expected_head_configs,
):
    """Audit and append one A3 validation candidate, then persist its winner."""

    output_dir = Path(selection_path).parent
    marker_path = _checkpoint_publication_marker_path(
        epoch, argparse.Namespace(output_dir=str(output_dir))
    )
    _validate_checkpoint_publication(
        marker_path,
        epoch=epoch,
        model_path=checkpoint_path,
        optimizer_path=optimizer_checkpoint_path,
    )
    candidate = _build_contextual_risk_candidate(
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        optimizer_checkpoint_path=optimizer_checkpoint_path,
        validation_metrics=validation_metrics,
        paired_baseline=paired_baseline,
        reference_checkpoint=reference_checkpoint,
        bootstrap_samples=bootstrap_samples,
        expected_retrieval_config=expected_retrieval_config,
        expected_head_configs=expected_head_configs,
    )
    candidate["publication_marker"] = str(marker_path.resolve())
    candidate["publication_marker_sha256"] = file_sha256(marker_path)
    if "artifact_array_root" in candidate["evidence"]:
        validate_artifact_json_payload(candidate)
    candidates = append_candidate_atomic(candidate_path, candidate)
    validate_candidate_artifacts(
        candidates,
        output_dir=Path(selection_path).parent,
        method_name="A3",
    )
    try:
        winner = choose_contextual_risk_checkpoint(
            candidates, output_dir=Path(selection_path).parent
        )
    except ValueError as exc:
        if "no valid contextual risk checkpoint" not in str(exc):
            raise
        winner = None
    selection_payload = _contextual_risk_selection_payload(candidates, winner)
    if "artifact_array_root" in candidate["evidence"]:
        validate_artifact_json_payload(selection_payload)
    atomic_write_json(selection_path, selection_payload)
    if candidate["non_a3_mismatches"]:
        raise RuntimeError(
            "terminal A3 invariance violation recorded; training is aborted"
        )
    return winner


def _maybe_run_contextual_risk_final_test(
    args, selection_payload, run_selected_jsfusion_once
):
    """Run the held-out test once, and only after the selected validation gate."""

    if not bool(getattr(args, "run_final_test", False)):
        return None
    candidates = selection_payload.get("candidates", [])
    if (
        isinstance(candidates, list)
        and len(candidates) == 3
        and {int(item.get("epoch", -1)) for item in candidates} == {0, 1, 2}
        and _contextual_risk_continuation_eligible(
            candidates, output_dir=args.output_dir
        )
    ):
        logger.info(
            "A3 continuation is eligible; JSFUSION final test is deferred "
            "until the five-epoch lineage is complete"
        )
        return None
    winner = selection_payload.get("winner")
    if winner is None:
        logger.warning(
            "A3 validation gate failed; JSFUSION test is intentionally skipped"
        )
        return None
    acceptance = contextual_risk_acceptance(winner["metrics"])
    if not acceptance["passed"]:
        logger.warning(
            "A3 validation gate failed; JSFUSION test is intentionally skipped"
        )
        return None
    selected = validate_selected_checkpoint(
        selection_payload,
        output_dir=args.output_dir,
        method_name="A3",
    )
    claim_path = Path(args.output_dir) / "a3_final_test_claim.json"
    final_path = Path(args.output_dir) / "final_test.json"
    if claim_path.exists():
        existing_claim = json.loads(claim_path.read_text())
        status = existing_claim.get("status")
        if status == "completed":
            recorded_result = Path(existing_claim.get("result_path", "")).resolve()
            if (
                recorded_result != final_path.resolve()
                or not final_path.exists()
                or file_sha256(final_path) != existing_claim.get("result_sha256")
            ):
                raise ValueError(
                    "completed A3 final-test claim has a missing or corrupt result"
                )
        if status == "started":
            raise RuntimeError(
                "A3 final-test claim is started and non-retryable; inspect crash window"
            )
        raise RuntimeError(
            f"A3 JSFUSION final test claim is {status!r} and cannot be retried"
        )
    if final_path.exists():
        raise ValueError("A3 final-test result exists without its durable claim")
    claim = {
        "status": "started",
        "selected_epoch": int(selected["epoch"]),
        "selected_path": selected["path"],
        "selected_model_sha256": selected["model_sha256"],
    }
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "A3 JSFUSION final test has already been claimed for this lineage"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        result = run_selected_jsfusion_once(selected)
        if "artifact_array_root" in selected.get("evidence", {}):
            validate_artifact_json_payload(result)
        atomic_write_json(final_path, result)
    except Exception as exc:
        atomic_write_json(
            claim_path,
            {**claim, "status": "failed", "error_type": type(exc).__name__},
        )
        raise
    atomic_write_json(
        claim_path,
        {
            **claim,
            "status": "completed",
            "result_path": str(final_path.resolve()),
            "result_sha256": file_sha256(final_path),
        },
    )
    return result


def _build_csd_logvar_candidate(
    *,
    epoch,
    checkpoint_path,
    optimizer_checkpoint_path,
    validation_metrics,
    paired_baseline,
    reference_checkpoint,
):
    """Build one fully audited bidirectional validation candidate."""

    model_path = Path(checkpoint_path).resolve()
    optimizer_path = Path(optimizer_checkpoint_path).resolve()
    intrinsic = validation_metrics["csd_logvar"]
    paired_invariance = compare_paired_invariance(
        paired_baseline,
        validation_metrics["paired_digest"],
    )
    mismatches = non_csd_state_dict_mismatches(
        reference_checkpoint,
        model_path,
    )
    finite = numeric_metrics_are_finite(validation_metrics)
    clamp_ok = _candidate_clamp_ok(intrinsic)
    return {
        "epoch": int(epoch),
        "path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "optimizer_path": str(optimizer_path),
        "optimizer_sha256": file_sha256(optimizer_path),
        "valid": bool(
            finite and paired_invariance["passed"] and not mismatches and clamp_ok
        ),
        "finite": finite,
        "paired_invariance": paired_invariance,
        "non_a2_mismatches": mismatches,
        "clamp_ok": clamp_ok,
        "rho": [
            intrinsic[direction]["spearman_uncertainty_vs_rank"]
            for direction in ("t2v", "v2t")
        ],
        "auroc": [
            intrinsic[direction]["failure_auroc"]["value"]
            for direction in ("t2v", "v2t")
        ],
        "acceptance": csd_logvar_acceptance(intrinsic),
    }


def _reaudit_csd_logvar_candidates(candidates, reference_checkpoint):
    """Recheck every persisted candidate against the original A1 tensors."""

    audited = []
    for stored in candidates:
        candidate = dict(stored)
        mismatches = non_csd_state_dict_mismatches(
            reference_checkpoint,
            candidate["path"],
        )
        candidate["non_a2_mismatches"] = mismatches
        candidate["valid"] = bool(
            candidate["finite"]
            and candidate["paired_invariance"]["passed"]
            and not mismatches
            and candidate["clamp_ok"]
        )
        audited.append(candidate)
    return audited


def _csd_logvar_selection_payload(candidates, winner):
    return {
        "selection_split": "val",
        "selection_metric": (
            "max(min(t2v_rho,v2t_rho),mean(t2v_auroc,v2t_auroc))"
        ),
        "tie_break": "earlier_epoch",
        "candidates": candidates,
        "winner": winner,
    }


def _initialize_csd_logvar_protocol(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    resume_payload=None,
):
    """Register fresh A1 evidence or restore one continuation lineage."""

    baseline_path = Path(args.output_dir) / "a1_paired_baseline.json"
    selection_path = Path(args.output_dir) / "best_csd_logvar_checkpoint.json"
    continuing = bool(args.resume_model)
    if continuing and not baseline_path.exists():
        raise FileNotFoundError(
            "A2 continuation requires the original paired baseline registration"
        )
    if continuing and not selection_path.exists():
        raise FileNotFoundError(
            "A2 continuation requires the original candidate selection list"
        )
    if not continuing and (baseline_path.exists() or selection_path.exists()):
        raise FileExistsError(
            "fresh A2 run requires an output directory without existing "
            "protocol lineage artifacts"
        )

    if continuing:
        paired_registration = json.loads(baseline_path.read_text())
        paired_baseline = paired_registration["paired_digest"]
        reference_checkpoint = paired_registration["reference_checkpoint"]
        registered_reference_sha256 = paired_registration.get(
            "reference_checkpoint_sha256"
        )
        if not registered_reference_sha256:
            raise ValueError(
                "A2 continuation baseline registration is missing the "
                "reference checkpoint SHA-256"
            )
        actual_reference_sha256 = file_sha256(reference_checkpoint)
        if actual_reference_sha256 != registered_reference_sha256:
            raise ValueError("A2 reference checkpoint SHA-256 mismatch")
        if registered_reference_sha256 != CSD_LOGVAR_CANONICAL_A1_SHA256:
            raise ValueError(
                "A2 registered reference SHA-256 is not the canonical A1"
            )
    else:
        reference_checkpoint = args.init_model
        reference_checkpoint_sha256 = file_sha256(reference_checkpoint)
        if reference_checkpoint_sha256 != CSD_LOGVAR_CANONICAL_A1_SHA256:
            raise ValueError("fresh A2 reference checkpoint SHA-256 mismatch")
        baseline_eval = eval_epoch(
            args,
            model,
            eval_dataloader,
            device,
            n_gpu,
            report_csd_logvar=False,
        )
        paired_baseline = baseline_eval["paired_digest"]
        atomic_write_json(
            baseline_path,
            {
                "reference_checkpoint": reference_checkpoint,
                "reference_checkpoint_sha256": reference_checkpoint_sha256,
                "paired_digest": paired_baseline,
            },
        )

    candidates = []
    resume_candidate = None
    if continuing:
        if resume_payload is None:
            raise ValueError("A2 continuation requires an optimizer payload")
        candidates = json.loads(selection_path.read_text())["candidates"]
        validate_csd_logvar_candidate_artifacts(
            candidates,
            output_dir=args.output_dir,
        )
        candidates = _reaudit_csd_logvar_candidates(
            candidates,
            reference_checkpoint,
        )
        resume_candidate = validate_csd_logvar_resume_lineage(
            candidates,
            output_dir=args.output_dir,
            model_checkpoint=args.init_model,
            optimizer_checkpoint=args.resume_model,
            optimizer_payload=resume_payload,
            artifacts_verified=True,
        )
        try:
            winner = choose_csd_logvar_checkpoint(candidates)
        except ValueError:
            winner = None
        atomic_write_json(
            selection_path,
            _csd_logvar_selection_payload(candidates, winner),
        )
    return {
        "baseline_path": baseline_path,
        "selection_path": selection_path,
        "paired_baseline": paired_baseline,
        "reference_checkpoint": reference_checkpoint,
        "candidates": candidates,
        "resume_candidate": resume_candidate,
    }


def _record_csd_logvar_selection(
    *,
    epoch,
    checkpoint_path,
    optimizer_checkpoint_path,
    validation_metrics,
    paired_baseline,
    reference_checkpoint,
    candidates,
    selection_path,
):
    """Append, audit, select, and atomically persist one A2 candidate."""

    epochs = [int(candidate["epoch"]) for candidate in candidates]
    if epochs and int(epoch) <= max(epochs):
        raise ValueError(
            "A2 candidate append-only epoch must exceed every recorded epoch"
        )
    if len(epochs) != len(set(epochs)):
        raise ValueError("A2 candidate epochs must be unique")

    candidate = _build_csd_logvar_candidate(
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        optimizer_checkpoint_path=optimizer_checkpoint_path,
        validation_metrics=validation_metrics,
        paired_baseline=paired_baseline,
        reference_checkpoint=reference_checkpoint,
    )
    candidates.append(candidate)
    validate_csd_logvar_candidate_artifacts(
        candidates,
        output_dir=Path(selection_path).parent,
    )
    try:
        winner = choose_csd_logvar_checkpoint(candidates)
    except ValueError:
        atomic_write_json(
            selection_path,
            _csd_logvar_selection_payload(candidates, None),
        )
        return None
    atomic_write_json(
        selection_path,
        _csd_logvar_selection_payload(candidates, winner),
    )
    return winner


def _evaluate_selected_test_checkpoints(
    args,
    model,
    test_dataloader,
    device,
    n_gpu,
    selection_payload,
    *,
    csd_logvar_sidecar_only,
    contextual_risk_sidecar_only=False,
):
    """Evaluate validation-selected checkpoints without test-time selection."""

    if csd_logvar_sidecar_only or contextual_risk_sidecar_only:
        if csd_logvar_sidecar_only:
            selected = validate_csd_logvar_selected_checkpoint(
                selection_payload,
                output_dir=args.output_dir,
            )
            method_name = "A2"
        else:
            selected = validate_selected_checkpoint(
                selection_payload,
                output_dir=args.output_dir,
                method_name="A3",
            )
            method_name = "A3"
        checkpoint_path = selected["path"]
        logger.info(
            "***** Running final bidirectional test from %s-selected "
            "checkpoint: %s *****",
            method_name,
            checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to(device)
        test_metrics = eval_epoch(
            args,
            model,
            test_dataloader,
            device,
            n_gpu,
            directions=("t2v", "v2t"),
        )
        return {
            "test_split": "test",
            "selection": selection_payload,
            "test_metrics": test_metrics,
        }

    test_metrics_by_checkpoint = {}
    final_test_payload = {}
    for direction in ("t2v", "v2t"):
        selection = selection_payload[direction]
        checkpoint_path = selection["checkpoint"]
        cache_key = (checkpoint_path, direction)
        if cache_key not in test_metrics_by_checkpoint:
            logger.info(
                "***** Running final test from %s-selected checkpoint: %s *****",
                direction.upper(),
                checkpoint_path,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint)
            model.to(device)
            test_metrics_by_checkpoint[cache_key] = eval_epoch(
                args,
                model,
                test_dataloader,
                device,
                n_gpu,
                directions=(direction,),
            )
        final_test_payload[direction] = {
            **selection,
            "test_metrics": test_metrics_by_checkpoint[cache_key],
        }
    return {
        "test_split": "test",
        "selections": final_test_payload,
    }


def main():
    global logger
    import torch.distributed as dist

    args = get_args()
    _assert_registered_a3_initial_environment(args)
    _preflight_registered_a3_clip_asset(args, verify_load=False)
    initial_git_state = collect_git_state(Path(__file__).resolve().parent)
    if bool(getattr(args, "contextual_risk_sidecar_only", False)) and bool(
        initial_git_state.get("dirty", False)
    ):
        raise ValueError(
            "registered A3 requires clean tracked source state: "
            + ", ".join(initial_git_state.get("modified_paths", []))
        )

    # Validate the committed trusted-v1 manifest against the canonical source
    # files before constructing a tokenizer, model, or dataloader.  Task10 will
    # migrate the standard shell defaults to generated train/val CSVs; this
    # gate intentionally keeps the source-of-truth check here.
    split_summary = None
    if args.datatype == "msrvtt":
        manifest = load_trusted_manifest(args.split_manifest)
        split_summary = validate_trusted_manifest(
            manifest,
            args.source_train_csv,
            args.data_path,
            args.test_csv,
        )
        if bool(getattr(args, "contextual_risk_sidecar_only", False)):
            generated_csv = validate_generated_split_files(
                manifest,
                args.data_path,
                args.train_csv,
                args.val_csv,
            )
            split_summary = {**split_summary, "generated_csv": generated_csv}

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0  # 默认单卡

    torch.cuda.set_device(args.local_rank)

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
        pass  # process group initialized

    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    args.pin_memory = torch.cuda.is_available()
    _assert_registered_a3_contract(args)

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            if not _should_keep_clip_parameter_trainable(name, args):
                param.requires_grad = False
    configure_retrieval_risk_sidecar_training(model, args)
    csd_trainable_count = configure_csd_logvar_sidecar_training(model, args)
    if csd_trainable_count is not None and is_global_rank_zero(args):
        logger.info("[CSD] trainable parameters: %d scalars", csd_trainable_count)
    contextual_trainable_count = configure_contextual_risk_sidecar_training(
        model, args
    )
    if contextual_trainable_count is not None and is_global_rank_zero(args):
        logger.info(
            "[A3] trainable parameters: %d scalars", contextual_trainable_count
        )

    ## ####################################
    # dataloader loading
    ## ####################################
    train_bundle, eval_dataloader, eval_length, eval_split = (
        prepare_requested_dataloaders(args, tokenizer)
    )

    batch_semantics = None
    if args.do_train:
        train_dataloader = train_bundle[0]
        scheduler_horizon_epochs = (
            REGISTERED_A3_SCHEDULER_HORIZON_EPOCHS
            if bool(getattr(args, "contextual_risk_sidecar_only", False))
            else args.epochs
        )
        batch_semantics = compute_batch_semantics(
            requested_effective_batch=args.requested_effective_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            world_size=args.world_size,
            dataloader_steps=len(train_dataloader),
            epochs=scheduler_horizon_epochs,
        )
        _assert_registered_a3_contract(args, batch_semantics)
        git_state = initial_git_state
        tracking_payload = build_experiment_manifest(
            args,
            split_summary=split_summary,
            batch_semantics=batch_semantics,
            git_state=git_state,
        )
        manifest_path = Path(args.output_dir) / "experiment_manifest.json"
        _publish_experiment_manifest(
            args,
            tracking_payload=tracking_payload,
            manifest_path=manifest_path,
            batch_semantics=batch_semantics,
        )

    if args.local_rank == 0:
        logger.info("***** Running %s evaluation *****", eval_split)
        frame_order_name = {0: "sequential", 1: "reverse", 2: "shuffle"}.get(
            args.eval_frame_order, "unknown"
        )
        logger.info(
            "  Split: %s | Samples: %d | Steps: %d | Batch: %d | Frame order: %s",
            eval_split,
            eval_length,
            len(eval_dataloader),
            args.batch_size_val,
            frame_order_name,
        )

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = train_bundle
        num_train_optimization_steps = batch_semantics["total_optimizer_steps"]

        coef_lr = args.coef_lr
        # distribute model by torch.nn.distributing
        optimizer, scheduler, model = prep_optimizer(
            args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr
        )

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info(
                "  Samples: %d | Batch: %d x %d GPUs x %d accum = %d eff | Steps/epoch: %d | Total opt steps: %d",
                train_length,
                args.batch_size // args.n_gpu,
                args.n_gpu,
                args.gradient_accumulation_steps,
                batch_semantics["optimizer_effective_batch"],
                batch_semantics["forward_steps_per_epoch"],
                num_train_optimization_steps,
            )

        best_t2v_score = float("-inf")
        best_t2v_checkpoint = "None"
        best_v2t_score = float("-inf")
        best_v2t_checkpoint = "None"
        a2_only = bool(getattr(args, "csd_logvar_sidecar_only", False))
        a3_only = bool(getattr(args, "contextual_risk_sidecar_only", False))
        lifecycle_owner = (
            _contextual_risk_lifecycle_owner(args)
            if a3_only
            else args.local_rank == 0
        )
        csd_protocol = None
        contextual_protocol = None

        # CRF-dev checkpoint selection (plan Task 9 wiring). Populated per epoch
        # on rank 0; the winner is written atomically to best_crf_checkpoint.json
        # after every epoch so a crash mid-training still leaves a usable pick.
        crf_dev_candidates: list[dict] = []
        crf_dev_video_ids: list[str] = []
        if bool(getattr(args, "aleatoric_crf_training", False)):
            _crf_split_path = Path("dataloaders/splits/crf_dev_audit_seed0.json")
            crf_dev_video_ids = json.loads(_crf_split_path.read_text())[
                "crf_dev_video_ids"
            ]
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        global_step = 0
        resume_checkpoint = None
        if args.resume_model:
            resume_checkpoint = torch.load(args.resume_model, map_location="cpu")

        if a2_only and args.local_rank == 0:
            csd_protocol = _initialize_csd_logvar_protocol(
                args,
                model,
                eval_dataloader,
                device,
                n_gpu,
                resume_payload=resume_checkpoint,
            )
        if a3_only and lifecycle_owner:
            contextual_protocol = _initialize_contextual_risk_protocol(
                args,
                model,
                eval_dataloader,
                device,
                n_gpu,
                resume_payload=resume_checkpoint,
            )
        if (a2_only or a3_only) and dist.is_available() and dist.is_initialized():
            dist.barrier()

        if resume_checkpoint is not None:
            resumed_epoch, global_step = _restore_training_checkpoint(
                resume_checkpoint,
                optimizer,
                scheduler,
                require_complete=a2_only or a3_only,
                expected_scheduler_horizon_epochs=(
                    REGISTERED_A3_SCHEDULER_HORIZON_EPOCHS
                    if a3_only
                    else None
                ),
                expected_scheduler_total_steps=(
                    batch_semantics["total_optimizer_steps"]
                    if a3_only
                    else None
                ),
                expected_optimizer_steps_per_epoch=(
                    batch_semantics["optimizer_steps_per_epoch"]
                    if a3_only
                    else None
                ),
                model=model if a3_only else None,
            )

        if args.local_rank == 0:
            if resumed_epoch > 0:
                logger.info("Resuming training from epoch %d", resumed_epoch + 1)
            logger.info("=" * 60)

        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(
                epoch,
                args,
                model,
                train_dataloader,
                device,
                n_gpu,
                optimizer,
                scheduler,
                global_step,
                local_rank=args.local_rank,
            )

            checkpoint_rng_states = (
                _gather_distributed_rng_states() if a3_only else None
            )
            contextual_terminal_error = None

            if lifecycle_owner:
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(device) / 1024**3
                    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
                    reserved_memory = torch.cuda.memory_reserved(device) / 1024**3
                    logger.info(
                        "  GPU Memory | Cur: %.2f GB, Peak: %.2f GB, Reserved: %.2f GB",
                        current_memory,
                        max_memory,
                        reserved_memory,
                    )
                    torch.cuda.reset_peak_memory_stats(device)

                output_model_file = save_model(
                    epoch,
                    args,
                    model,
                    optimizer,
                    tr_loss,
                    type_name="",
                    scheduler=scheduler,
                    global_step=global_step,
                    scheduler_horizon_epochs=(
                        REGISTERED_A3_SCHEDULER_HORIZON_EPOCHS
                        if a3_only
                        else None
                    ),
                    scheduler_total_steps=(
                        batch_semantics["total_optimizer_steps"]
                        if a3_only
                        else None
                    ),
                    optimizer_steps_per_epoch=(
                        batch_semantics["optimizer_steps_per_epoch"]
                        if a3_only
                        else None
                    ),
                    rng_state_by_rank=checkpoint_rng_states,
                )
                output_optimizer_file = _optimizer_checkpoint_path(epoch, args)
                if bool(getattr(args, "aleatoric_crf_training", False)):
                    from scripts.evaluate_crf_ladder import (
                        choose_checkpoint,
                        write_dev_selection,
                    )
                    crf_dev_acc = _crf_dev_evaluate(
                        model, args, device, crf_dev_video_ids
                    )
                    crf_dev_candidates.append({
                        "epoch": epoch,
                        "accuracy": crf_dev_acc,
                        "path": str(output_model_file),
                    })
                    winner = choose_checkpoint(crf_dev_candidates)
                    write_dev_selection(
                        Path(args.output_dir) / "best_crf_checkpoint.json",
                        winner,
                    )
                    logger.info(
                        "  [CRF-dev] epoch=%d acc=%.4f | best=epoch %d acc=%.4f (%s)",
                        epoch, crf_dev_acc, winner["epoch"], winner["accuracy"],
                        winner["path"],
                    )
                if os.environ.get("SMOKE_STEPS"):
                    logger.info(
                        "[SMOKE_STEPS] skipping validation eval — smoke run "
                        "only verifies the training-step pipeline."
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    return
                validation_metrics = eval_epoch(
                    args, model, eval_dataloader, device, n_gpu
                )
                if a2_only:
                    winner = _record_csd_logvar_selection(
                        epoch=epoch,
                        checkpoint_path=output_model_file,
                        optimizer_checkpoint_path=output_optimizer_file,
                        validation_metrics=validation_metrics,
                        paired_baseline=csd_protocol["paired_baseline"],
                        reference_checkpoint=csd_protocol[
                            "reference_checkpoint"
                        ],
                        candidates=csd_protocol["candidates"],
                        selection_path=csd_protocol["selection_path"],
                    )
                    if winner is None:
                        logger.warning(
                            "  No valid %s A2 checkpoint through epoch %d",
                            eval_split,
                            epoch,
                        )
                    else:
                        logger.info(
                            "  Best %s A2 | epoch=%d min-rho=%.4f "
                            "mean-AUROC=%.4f | %s",
                            eval_split,
                            winner["epoch"],
                            min(winner["rho"]),
                            sum(winner["auroc"]) / 2.0,
                            winner["path"],
                        )
                elif a3_only:
                    try:
                        winner = _record_contextual_risk_selection(
                            epoch=epoch,
                            checkpoint_path=output_model_file,
                            optimizer_checkpoint_path=output_optimizer_file,
                            validation_metrics=validation_metrics,
                            paired_baseline=contextual_protocol["paired_baseline"],
                            reference_checkpoint=contextual_protocol[
                                "reference_checkpoint"
                            ],
                            candidate_path=contextual_protocol["candidate_path"],
                            selection_path=contextual_protocol["selection_path"],
                            bootstrap_samples=args.contextual_risk_bootstrap_samples,
                            expected_retrieval_config=_registered_retrieval_config(
                                np.asarray(
                                    validation_metrics["contextual_risk_evidence"][
                                        "positive_mask"
                                    ],
                                    dtype=bool,
                                ),
                                dsl=bool(args.DSL),
                                dsl_temperature=float(args.dsl_temperature),
                            ),
                            expected_head_configs=_registered_a3_head_configs(
                                args,
                                validation_metrics["contextual_risk_evidence"],
                            ),
                        )
                    except RuntimeError as exc:
                        if "terminal A3 invariance violation" not in str(exc):
                            raise
                        winner = None
                        contextual_terminal_error = str(exc)
                    contextual_protocol["candidates"] = json.loads(
                        contextual_protocol["candidate_path"].read_text()
                    )
                    if winner is None:
                        logger.warning(
                            "  No valid %s A3 checkpoint through epoch %d",
                            eval_split,
                            epoch,
                        )
                    else:
                        logger.info(
                            "  Best %s A3 | epoch=%d min-rho=%.4f "
                            "mean-AUROC=%.4f mean-AURC=%.4f | %s",
                            eval_split,
                            winner["epoch"],
                            min(winner["rho"]),
                            sum(winner["auroc"]) / 2.0,
                            sum(winner["aurc"]) / 2.0,
                            winner["path"],
                        )
                else:
                    best_t2v_score, best_t2v_checkpoint = select_best_checkpoint(
                        best_t2v_score,
                        best_t2v_checkpoint,
                        validation_metrics["t2v"]["R1"],
                        output_model_file,
                    )
                    best_v2t_score, best_v2t_checkpoint = select_best_checkpoint(
                        best_v2t_score,
                        best_v2t_checkpoint,
                        validation_metrics["v2t"]["R1"],
                        output_model_file,
                    )
                    logger.info(
                        "  Best %s T2V R@1: %.1f | %s",
                        eval_split,
                        best_t2v_score,
                        best_t2v_checkpoint,
                    )
                    logger.info(
                        "  Best %s V2T R@1: %.1f | %s",
                        eval_split,
                        best_v2t_score,
                        best_v2t_checkpoint,
                    )
                logger.info("=" * 60)

            if a3_only and dist.is_available() and dist.is_initialized():
                terminal_payload = [contextual_terminal_error]
                dist.broadcast_object_list(terminal_payload, src=0)
                contextual_terminal_error = terminal_payload[0]
            if contextual_terminal_error is not None:
                raise RuntimeError(contextual_terminal_error)

            # Barrier so ranks 1..N-1 don't get too far ahead of rank 0's eval
            # into the next epoch's first forward — the CRF-dev evaluator adds
            # ~30-60s of rank-0-only work that would otherwise sit inside
            # allreduce wait time.
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if a3_only:
                _restore_rng_state_for_current_rank(checkpoint_rng_states)

        if lifecycle_owner:
            if a2_only:
                winner = choose_csd_logvar_checkpoint(
                    csd_protocol["candidates"],
                    output_dir=args.output_dir,
                )
                selection_payload = json.loads(
                    csd_protocol["selection_path"].read_text()
                )
                if selection_payload["winner"] != winner:
                    raise RuntimeError("persisted A2 winner does not match candidates")
            elif a3_only:
                candidates = json.loads(
                    contextual_protocol["candidate_path"].read_text()
                )
                selection_payload = json.loads(
                    contextual_protocol["selection_path"].read_text()
                )
                if selection_payload["candidates"] != candidates:
                    raise RuntimeError(
                        "persisted A3 candidates do not match append-only artifact"
                    )
                try:
                    winner = choose_contextual_risk_checkpoint(
                        candidates,
                        output_dir=args.output_dir,
                    )
                except ValueError as exc:
                    if "no valid contextual risk checkpoint" not in str(exc):
                        raise
                    winner = None
                if selection_payload["winner"] != winner:
                    raise RuntimeError("persisted A3 winner does not match candidates")
            else:
                if (
                    best_t2v_checkpoint == "None"
                    or best_v2t_checkpoint == "None"
                ):
                    raise RuntimeError(
                        "training completed without a validation checkpoint"
                    )
                selection_payload = {
                    "selection_split": "val",
                    "tie_break": "later_epoch",
                    "t2v": {
                        "selection_metric": "t2v_r1",
                        "selection_score": best_t2v_score,
                        "checkpoint": best_t2v_checkpoint,
                    },
                    "v2t": {
                        "selection_metric": "v2t_r1",
                        "selection_score": best_v2t_score,
                        "checkpoint": best_v2t_checkpoint,
                    },
                }
                atomic_write_json(
                    Path(args.output_dir) / "best_validation_checkpoints.json",
                    selection_payload,
                )

            if args.run_final_test and a3_only:
                model_to_evaluate = model.module if hasattr(model, "module") else model

                def run_selected_jsfusion_once(_selected):
                    test_factory = DATALOADER_DICT[args.datatype].get("test")
                    if test_factory is None:
                        raise ValueError(f"{args.datatype} has no test dataloader")
                    test_dataloader, test_length = test_factory(
                        args, tokenizer, subset="test"
                    )
                    logger.info(
                        "  Split: test | Samples: %d | Steps: %d | Batch: %d",
                        test_length,
                        len(test_dataloader),
                        args.batch_size_val,
                    )
                    previous_eval_split = args.eval_split
                    args.eval_split = "test"
                    try:
                        return _evaluate_selected_test_checkpoints(
                            args,
                            model_to_evaluate,
                            test_dataloader,
                            device,
                            n_gpu,
                            selection_payload,
                            csd_logvar_sidecar_only=False,
                            contextual_risk_sidecar_only=True,
                        )
                    finally:
                        args.eval_split = previous_eval_split

                final_test_payload = _maybe_run_contextual_risk_final_test(
                    args, selection_payload, run_selected_jsfusion_once
                )
            elif args.run_final_test:
                model_to_evaluate = model.module if hasattr(model, "module") else model
                test_factory = DATALOADER_DICT[args.datatype].get("test")
                if test_factory is None:
                    raise ValueError(f"{args.datatype} has no test dataloader")
                test_dataloader, test_length = test_factory(args, tokenizer, subset="test")

                logger.info(
                    "  Split: test | Samples: %d | Steps: %d | Batch: %d",
                    test_length,
                    len(test_dataloader),
                    args.batch_size_val,
                )
                previous_eval_split = args.eval_split
                args.eval_split = "test"
                try:
                    final_test_payload = _evaluate_selected_test_checkpoints(
                        args,
                        model_to_evaluate,
                        test_dataloader,
                        device,
                        n_gpu,
                        selection_payload,
                        csd_logvar_sidecar_only=a2_only,
                    )
                finally:
                    args.eval_split = previous_eval_split

                atomic_write_json(
                    Path(args.output_dir) / "final_test.json",
                    final_test_payload,
                )

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, eval_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
