from __future__ import division, print_function, unicode_literals

import argparse
import datetime
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.data_dataloaders import DATALOADER_DICT
from dataloaders.msrvtt_protocol import load_trusted_manifest, validate_trusted_manifest
from experiment_tracking import (
    CSD_LOGVAR_CANONICAL_A1_CHECKPOINT,
    CSD_LOGVAR_CANONICAL_A1_SHA256,
    atomic_write_json,
    build_experiment_manifest,
    collect_git_state,
    compute_batch_semantics,
    is_global_rank_zero,
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
    parser.add_argument("--contextual_risk_topk", type=int, default=16)
    parser.add_argument("--contextual_risk_projection_dim", type=int, default=128)
    parser.add_argument("--contextual_risk_hidden_dim", type=int, default=128)
    parser.add_argument("--contextual_risk_temperature", type=float, default=1.0)
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


def init_model(args, device, n_gpu, local_rank):
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
    # Ensure all trainable params are fp32 so fused AdamW gets uniform dtype.
    # (CLIP's convert_weights may leave visual conv/proj weights in fp16.)
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
    warmup_frac = args.warmup_proportion
    t_total = max(1, num_train_optimization_steps)

    def _warmup_cosine_lambda(step):
        x = step / t_total
        if warmup_frac > 0 and x < warmup_frac:
            return x / warmup_frac
        return 0.5 * (1.0 + math.cos(math.pi * x))

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup_cosine_lambda
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
):
    payload = {
        "epoch": int(epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": tr_loss,
        "global_step": int(global_step),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _restore_training_checkpoint(
    checkpoint,
    optimizer,
    scheduler,
    *,
    require_complete,
):
    if require_complete:
        missing = [
            name
            for name in ("scheduler_state_dict", "global_step")
            if name not in checkpoint
        ]
        if missing:
            raise ValueError(
                "A2 resume checkpoint missing " + ", ".join(missing)
            )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        if scheduler is None:
            raise ValueError("cannot restore scheduler state without a scheduler")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0))


def _optimizer_checkpoint_path(epoch, args, type_name=""):
    return os.path.join(
        args.output_dir,
        "pytorch_opt.bin.{}{}".format(
            "" if type_name == "" else type_name + ".",
            epoch,
        ),
    )


def save_model(
    epoch,
    args,
    model,
    optimizer,
    tr_loss,
    type_name="",
    scheduler=None,
    global_step=0,
):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch)
    )
    optimizer_state_file = _optimizer_checkpoint_path(epoch, args, type_name)
    if bool(getattr(args, "csd_logvar_sidecar_only", False)):
        existing = [
            path
            for path in (output_model_file, optimizer_state_file)
            if Path(path).exists()
        ]
        if existing:
            raise FileExistsError(
                "A2 append-only save refuses existing checkpoint(s): "
                + ", ".join(existing)
            )
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(
        _build_training_checkpoint_payload(
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            tr_loss=tr_loss,
            global_step=global_step,
        ),
        optimizer_state_file,
    )
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):  # for evaluation and test
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


def _time_per_dataloader_step_since_log(
    elapsed_seconds, current_dataloader_step, last_logged_dataloader_step
):
    return elapsed_seconds / (
        current_dataloader_step - last_logged_dataloader_step
    )


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    set_retrieval_risk_training_mode(model)
    set_csd_logvar_training_mode(model)
    _set_contextual_risk_training_mode(model)
    log_step = args.n_display
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            else:
                for name, tensor in loss_terms.items():
                    crf_window[name] = crf_window.get(name, 0.0) + float(tensor)
                crf_window_count += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:
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

            if (
                _should_log_training_step(global_step, log_step, smoke_cap, args)
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
                logits, aux = model.get_similarity_logits(
                    sequence_output,
                    text_token,
                    visual_output,
                    visual_hidden,
                    input_mask,
                    video_mask,
                    loose_type=model.loose_type,
                    return_csd_aux=collect_csd,
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
            if collect_csd and chunk_index == 0:
                text_payload.append(
                    {
                        name: aux[name].float().cpu()
                        for name in ("text_mu", "text_logvar")
                    }
                )
            if collect_csd and idx1 == 0:
                video_payload.append(
                    {
                        name: aux[name].float().cpu()
                        for name in ("video_mu", "video_logvar")
                    }
                )

        # bf16 张量不能直接 .numpy()，必须先转 fp32。
        b1_all_v_logits = torch.cat(row_logits, dim=1).float().cpu()
        sim_matrix.append(b1_all_v_logits.detach().numpy())

    if not collect_csd:
        return sim_matrix
    return {
        "similarity_rows": sim_matrix,
        "text_mu": torch.cat(
            [payload["text_mu"] for payload in text_payload], dim=0
        ),
        "text_logvar": torch.cat(
            [payload["text_logvar"] for payload in text_payload], dim=0
        ),
        "video_mu": torch.cat(
            [payload["video_mu"] for payload in video_payload], dim=0
        ),
        "video_logvar": torch.cat(
            [payload["video_logvar"] for payload in video_payload], dim=0
        ),
    }


def _merge_csd_eval_outputs(outputs):
    """Merge per-worker CSD payloads along retrieval matrix axes."""

    similarity_rows = [
        row for output in outputs for row in output["similarity_rows"]
    ]
    matrix = np.concatenate(tuple(similarity_rows), axis=0)
    text_mu = torch.cat([output["text_mu"] for output in outputs], dim=0)
    text_logvar = torch.cat(
        [output["text_logvar"] for output in outputs], dim=0
    )
    video_mu = outputs[0]["video_mu"]
    video_logvar = outputs[0]["video_logvar"]
    if (
        text_mu.size(0) != matrix.shape[0]
        or text_logvar.shape != text_mu.shape
        or video_mu.size(0) != matrix.shape[1]
        or video_logvar.shape != video_mu.shape
    ):
        raise ValueError(
            "intrinsic uncertainty axes do not align with retrieval matrix"
        )
    return matrix, {
        "text_mu": text_mu,
        "text_logvar": text_logvar,
        "video_mu": video_mu,
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
):
    """Score the complete candidate matrix without mutating retrieval inputs."""

    directions = _normalize_eval_directions(directions)
    raw_logits = torch.from_numpy(
        np.asarray(risk_similarity_matrix, dtype=np.float32)
    ).to(device)
    payload = {}
    array_digests = {}
    with torch.no_grad():
        for offset, direction in enumerate(directions):
            output = model.score_retrieval_risk(raw_logits, direction)
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
    if return_array_digests:
        return payload, array_digests
    return payload


def _build_paired_digest(
    result,
    risk_similarity_matrix,
    risk_array_digests,
    directions,
):
    """Build exact A1-output evidence for an A2 paired evaluation."""

    return {
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
    """Scope exact FP32 and deterministic cuDNN to paired A2 evaluation."""

    a2_only = bool(getattr(args, "csd_logvar_sidecar_only", False))
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_deterministic = torch.backends.cudnn.deterministic
    if a2_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=not a2_only,
        ):
            yield
    finally:
        if a2_only:
            torch.backends.cudnn.benchmark = prior_benchmark
            torch.backends.cudnn.deterministic = prior_deterministic


def eval_epoch(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    directions=("t2v", "v2t"),
    report_csd_logvar=None,
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
    positive_mask = None
    if report_retrieval_risk or report_csd_logvar:
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
        )
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


def _candidate_clamp_ok(metrics):
    return all(
        metrics[name]["clamp_low_fraction"]
        + metrics[name]["clamp_high_fraction"]
        < 0.05
        for name in ("text_distribution", "video_distribution")
    )


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
):
    """Evaluate validation-selected checkpoints without test-time selection."""

    if csd_logvar_sidecar_only:
        selected = validate_csd_logvar_selected_checkpoint(
            selection_payload,
            output_dir=args.output_dir,
        )
        checkpoint_path = selected["path"]
        logger.info(
            "***** Running final bidirectional test from A2-selected "
            "checkpoint: %s *****",
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
        batch_semantics = compute_batch_semantics(
            requested_effective_batch=args.requested_effective_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            world_size=args.world_size,
            dataloader_steps=len(train_dataloader),
            epochs=args.epochs,
        )
        tracking_payload = build_experiment_manifest(
            args,
            split_summary=split_summary,
            batch_semantics=batch_semantics,
            git_state=collect_git_state(Path(__file__).resolve().parent),
        )
        if is_global_rank_zero(args):
            atomic_write_json(
                Path(args.output_dir) / "experiment_manifest.json",
                tracking_payload,
            )
            _write_training_args_json(args, Path(args.output_dir) / "training_args.json")
            for key, value in batch_semantics.items():
                logger.info("  Batch semantics | %s=%s", key, value)

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
        csd_protocol = None

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
        if a2_only and dist.is_available() and dist.is_initialized():
            dist.barrier()

        if resume_checkpoint is not None:
            resumed_epoch, global_step = _restore_training_checkpoint(
                resume_checkpoint,
                optimizer,
                scheduler,
                require_complete=a2_only,
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

            if args.local_rank == 0:
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

            # Barrier so ranks 1..N-1 don't get too far ahead of rank 0's eval
            # into the next epoch's first forward — the CRF-dev evaluator adds
            # ~30-60s of rank-0-only work that would otherwise sit inside
            # allreduce wait time.
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if args.local_rank == 0:
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

            if args.run_final_test:
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
