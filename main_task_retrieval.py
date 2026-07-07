from __future__ import division, print_function, unicode_literals

import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.data_dataloaders import DATALOADER_DICT
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_mulit import UATVR
from modules.optimization import BertAdam
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from util import get_logger, parallel_apply

global logger


def get_args(description="CLIP4Clip on Retrieval Task"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--train_csv", type=str, default="data/.train.csv", help="")
    parser.add_argument("--val_csv", type=str, default="data/.val.csv", help="")
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
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--batch_size_val", type=int, default=3500, help="batch size eval")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay")
    parser.add_argument("--n_display", type=int, default=100, help="Information display frequence")
    # =========================
    # Debug/Logging: Query gate_scores
    # =========================
    parser.add_argument(
        "--gate_log_interval",
        type=int,
        default=None,
        help="Log interval (in optimizer steps) for MoE weights. If None, defaults to --n_display.",
    )
    parser.add_argument(
        "--gate_log_dir",
        type=str,
        default="logs/gate_scores",
        help="Root directory to save gate_scores logs (will create date subfolders).",
    )
    parser.add_argument(
        "--log_mus_scores",
        action="store_true",
        help="若设置，在每次评估时将每条查询的 MUS（映射不确定性）写入 logs/mus_scores/ 下的 TSV 文件。",
    )
    # =========================
    # Debug/Logging: Uncertainty-driven MoE fusion weights (base/query)
    # =========================
    parser.add_argument(
        "--log_moe_weights",
        action="store_true",
        help="If set, periodically log uncertainty-driven MoE fusion weights (base/query) to a TSV file under logs/.",
    )
    parser.add_argument(
        "--moe_log_dir",
        type=str,
        default="logs/moe_weights",
        help="Root directory to save MoE fusion weight logs (will create date subfolders).",
    )
    parser.add_argument("--video_dim", type=int, default=1024, help="video feature dimension")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--max_frames", type=int, default=100, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument("--hard_negative_rate", type=float, default=0.5, help="rate of intra negative sample")
    parser.add_argument("--negative_weighting", type=int, default=1, help="Weight the loss for intra negative")
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
    parser.add_argument("--use_mil", action="store_true", help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument("--sampled_use_mil", action="store_true", help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument("--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text.")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument("--cross_num_hidden_layers", type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument("--loose_type", action="store_true", help="Default using tight type for retrieval.")
    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")
    parser.add_argument(
        "--use_hard_negative_packing",
        action="store_true",
        help="Pack query-mined hard negative samples into the same global training batch.",
    )
    parser.add_argument(
        "--use_explicit_hard_negative_loss",
        action="store_true",
        help="Encode mapped hard-negative videos and add an explicit query-video hard-negative loss.",
    )
    parser.add_argument(
        "--hard_negative_path",
        # Raw pre-audit map kept for traceability:
        # cache_dir/hard_negatives/msrvtt_train_hardneg.json
        default="cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        type=str,
        help=(
            "Path to MSRVTT hard negative mapping JSON. Defaults to the audited clean map; "
            "the raw pre-audit map is cache_dir/hard_negatives/msrvtt_train_hardneg.json."
        ),
    )
    parser.add_argument(
        "--hard_negative_pack_seed",
        default=42,
        type=int,
        help="Seed for hard-negative batch packing shuffle.",
    )

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
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="choice a similarity header.",
    )

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--strategy", default=1, type=int, help="Sampling strategies.")
    parser.add_argument("--extra_video_cls_num", default=2, type=int, help="extra video class aggregation token")
    parser.add_argument("--extra_text_cls_num", default=2, type=int, help="extra sentence class aggregation token")
    parser.add_argument(
        "--n_video_embeddings", default=7, type=int, help="number of sampling video probabilistic embeddings"
    )
    parser.add_argument(
        "--n_text_embeddings", default=7, type=int, help="number of sampling text probabilistic embeddings"
    )
    parser.add_argument("--DSL", default=False, type=bool, help="whether using dual softmax in post testing")
    parser.add_argument(
        "--eval_vid_chunk_size",
        default=128,
        type=int,
        help="Number of videos per chunk during evaluation to limit peak GPU memory. "
             "Smaller value reduces memory at the cost of slightly slower eval.",
    )

    # Mamba-specific learning rate control
    parser.add_argument(
        "--mamba_lr_ratio",
        default=0.1,
        type=float,
        help="Learning rate ratio for Mamba modules relative to base lr (default: 0.1, i.e., 10x smaller)",
    )

    # =========================
    # Query / Uncertainty knobs
    # =========================
    parser.add_argument(
        "--uncertainty_text_head",
        default="image",
        type=str,
        choices=["image", "text", "mamba"],
        help="Text-side uncertainty head type. 'image' keeps current lightweight head; "
        "'text' enables GRU-based head; 'mamba' enables Mamba-based head (requires mamba_ssm).",
    )
    parser.add_argument(
        "--log_sigma_min",
        default=None,
        type=float,
        help="Optional clamp min for log-variance (logsigma/logvar). If None, no clamp is applied.",
    )
    parser.add_argument(
        "--log_sigma_max",
        default=None,
        type=float,
        help="Optional clamp max for log-variance (logsigma/logvar). If None, no clamp is applied.",
    )

    # Ablations & Diagnostics
    parser.add_argument(
        "--rope_mode", default="none", choices=["none", "2d", "3d"],
        help="SpatialEnhancer RoPE mode: none=disabled, 2d=per-frame, 3d=cross-frame.",
    )
    parser.add_argument(
        "--disable_spatial_enhancer", action="store_true", help="(deprecated) use --rope_mode none instead."
    )
    parser.add_argument(
        "--num_expansion_tokens", default=0, type=int,
        help="Number of learnable expansion tokens to concatenate with SAP anchors (Video-ColBERT style). 0=disabled.",
    )
    parser.add_argument(
        "--use_ada_norm", action="store_true",
        help="Enable uncertainty-aware adaptive LayerNorm before L2 normalize in probabilistic heads.",
    )

    parser.add_argument(
        "--eval_branch_mode",
        default="default",
        type=str,
        choices=["default", "base_only", "query_only", "fixed_avg"],
        help="Inference-time ablation: override fusion weights. "
        "default=dynamic(original); base_only=[1,0]; query_only=[0,1]; fixed_avg=[0.5,0.5].",
    )

    parser.add_argument(
        "--disable_query_gate_in_retrieval",
        action="store_true",
        help="If set, disable QueryRefinementModule gating during retrieval scoring (evaluation only).",
    )

    # -------------------------
    # Stage-1/2: Fusion & Gate
    # -------------------------
    parser.add_argument(
        "--fusion_mode",
        default="prob_mos",
        type=str,
        choices=["prob_mos", "logits_linear"],
        help="Fusion mode for base/query experts. "
        "prob_mos=calibrated mixture-of-softmax (scale-robust, recommended); "
        "logits_linear=legacy linear mixing in logit space (scale-sensitive).",
    )
    # Loss weights
    parser.add_argument("--w_mil", default=1e-2, type=float, help="Weight for MIL (probabilistic) loss.")
    parser.add_argument("--w_evidential", default=1e-2, type=float, help="Weight for Evidential NLL loss.")
    parser.add_argument("--w_neg_reg", default=1e-2, type=float, help="Weight for negative evidence regularization.")
    parser.add_argument("--w_hard_negative", default=5e-2, type=float, help="Weight for explicit hard-negative loss.")
    parser.add_argument(
        "--final_score_mode",
        default="wti",
        type=str,
        choices=["wti", "wti_prob_mu", "wti_anchor_wti", "wti_qc_sap"],
        help="Final retrieval score used by both training and evaluation. "
        "wti: use WTI logits only. "
        "wti_prob_mu: add lambda_prob * cosine(probabilistic text mean, SAP video mean). "
        "wti_anchor_wti: add lambda_anchor * WTI(text tokens, SAP anchors). "
        "wti_qc_sap: add lambda_qc_sap * query-conditioned SAP logits.",
    )
    parser.add_argument(
        "--lambda_prob",
        default=0.0,
        type=float,
        help="Weight for the probabilistic mean score when --final_score_mode=wti_prob_mu.",
    )
    parser.add_argument(
        "--lambda_anchor",
        default=0.0,
        type=float,
        help="Weight for the SAP AnchorWTI score when --final_score_mode=wti_anchor_wti.",
    )
    parser.add_argument(
        "--lambda_qc_sap",
        default=0.0,
        type=float,
        help="Weight for query-conditioned SAP score when --final_score_mode=wti_qc_sap.",
    )
    parser.add_argument(
        "--qc_sap_temperature",
        default=0.1,
        type=float,
        help="Softmax temperature for query-conditioned SAP anchor gate.",
    )
    parser.add_argument(
        "--w_uncertainty_reg",
        default=1e-3,
        type=float,
        help="Weight for DUQ-style evidential uncertainty regularization on retrieval logits.",
    )
    parser.add_argument("--w_orth", default=0.1, type=float, help="Weight for anchor orthogonality loss.")
    parser.add_argument(
        "--w_query_sim",
        default=1e-2,
        type=float,
        help="Weight for query-branch independent contrastive loss.",
    )
    parser.add_argument(
        "--use_uacl_intra_alignment",
        action="store_true",
        help="Enable UACL-style intra-modal contrastive alignment using Gaussian sampled views.",
    )
    parser.add_argument(
        "--w_uacl_intra",
        default=1e-2,
        type=float,
        help="Weight for UACL-style intra-modal alignment loss.",
    )
    parser.add_argument(
        "--w_uacl_kl",
        default=1e-4,
        type=float,
        help="Weight for lightweight Gaussian log-variance KL regularization in UACL alignment.",
    )
    parser.add_argument(
        "--uacl_temperature",
        default=0.07,
        type=float,
        help="Temperature for UACL-style intra-modal contrastive losses.",
    )
    parser.add_argument(
        "--uacl_sample_strategy",
        default="closest",
        choices=["closest", "random"],
        help="Gaussian sample selection strategy for UACL intra-modal positives.",
    )
    # 退火系数：默认关闭；需要时可对 evidential / neg_reg loss 做线性 warmup
    parser.add_argument(
        "--anneal_warmup_epochs",
        default=0,
        type=int,
        help="Number of warmup epochs for annealing evidential/neg_reg losses. "
             "Set <= 0 to disable annealing. During warmup, loss weight = epoch / warmup_epochs. "
             "After warmup, weight = 1.0.",
    )
    # 不确定性置信度 warmup 步数（方案 A）
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Number of warmup steps for uncertainty-weighted retrieval (Plan A). "
             "Set <= 0 to disable warmup and immediately use full uncertainty weighting. "
             "During warmup: confidence = 1 - α + α / (1 + epistemic), α = step / warmup_steps.",
    )
    # 不确定性训练模式：evidential / nig_mil(deprecated) / none
    parser.add_argument(
        "--uncertainty_mode",
        default="none",
        type=str,
        choices=["evidential", "nig_mil", "none"],
        help="How to train uncertainty parameters. "
             "evidential: enable current Dirichlet/evidential regularizers. "
             "nig_mil: deprecated compatibility mode for old NIG-MIL experiments. "
             "none: disable evidential/neg_reg losses, keep SAP architecture (baseline).",
    )
    # MoE fusion weight control
    parser.add_argument(
        "--fusion_temperature",
        default=1.5,
        type=float,
        help="Temperature for MoE fusion weight softmax. Higher = smoother weight distribution.",
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
        help="Experiment profile. default preserves the historical mainline; "
             "hygiene disables auxiliary losses for clean WTI-only attribution.",
    )
    args = parser.parse_args()

    if args.experiment_profile == "hygiene":
        args.w_mil = 0.0
        args.w_evidential = 0.0
        args.w_neg_reg = 0.0
        args.w_orth = 0.0
        args.w_hard_negative = 0.0
        args.w_uacl_intra = 0.0
        args.w_uacl_kl = 0.0
        args.uncertainty_mode = "none"
        args.use_hard_negative_packing = False
        args.use_explicit_hard_negative_loss = False
        args.use_uacl_intra_alignment = False

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps)
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        # 按类别分组打印关键参数，其余用紧凑格式
        key_params = {
            "Training": ["epochs", "batch_size", "lr", "coef_lr", "gradient_accumulation_steps", "fp16",
                         "max_frames", "max_words", "seed"],
            "Model": ["pretrained_clip_name", "sim_header", "linear_patch", "fusion_mode",
                       "extra_video_cls_num", "extra_text_cls_num", "n_video_embeddings", "n_text_embeddings",
                       "uncertainty_text_head", "log_sigma_min", "log_sigma_max"],
            "HardNeg": [
                "use_hard_negative_packing", "use_explicit_hard_negative_loss",
                "hard_negative_path", "hard_negative_pack_seed", "w_hard_negative",
            ],
            "Query": ["w_query_sim", "w_uncertainty_reg", "w_evidential", "w_neg_reg",
                      "fusion_temperature", "num_queries", "num_expansion_tokens"],
            "UACL": [
                "use_uacl_intra_alignment", "w_uacl_intra", "w_uacl_kl",
                "uacl_temperature", "uacl_sample_strategy",
            ],
            "Annealing": ["anneal_warmup_epochs", "warmup_steps"],
            "Uncertainty": ["uncertainty_mode", "experiment_profile"],
            "Scoring": [
                "final_score_mode", "lambda_prob", "lambda_anchor",
                "lambda_qc_sap", "qc_sap_temperature",
            ],
        }
        printed_keys = set()
        for group, keys in key_params.items():
            vals = []
            for k in keys:
                if k in args.__dict__:
                    vals.append(f"{k}={args.__dict__[k]}")
                    printed_keys.add(k)
            if vals:
                logger.info("  [%s] %s", group, " | ".join(vals))
        # 剩余参数紧凑打印
        rest = {k: v for k, v in sorted(args.__dict__.items()) if k not in printed_keys}
        if rest:
            rest_str = " | ".join(f"{k}={v}" for k, v in rest.items())
            logger.info("  [Other] %s", rest_str)

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
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")

    model = UATVR.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)

    return model


def _trainable_named_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.0):
    if hasattr(model, "module"):
        model = model.module

    param_optimizer = _trainable_named_parameters(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # 识别Mamba相关的参数（包括Mamba编码器、cross_scale_mamba、position_embeddings等）
    mamba_keywords = ["mamba", "ms_scale_position_embeddings", "ms_mamba_encoders", "cross_scale_mamba"]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    # 进一步细分：CLIP、Mamba、其他
    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_mamba_param_tp = [
        (n, p) for n, p in decay_param_tp if any(mk in n for mk in mamba_keywords) and "clip." not in n
    ]
    decay_other_param_tp = [
        (n, p) for n, p in decay_param_tp if "clip." not in n and not any(mk in n for mk in mamba_keywords)
    ]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_mamba_param_tp = [
        (n, p) for n, p in no_decay_param_tp if any(mk in n for mk in mamba_keywords) and "clip." not in n
    ]
    no_decay_other_param_tp = [
        (n, p) for n, p in no_decay_param_tp if "clip." not in n and not any(mk in n for mk in mamba_keywords)
    ]

    weight_decay = 0.2
    mamba_lr = args.lr * args.mamba_lr_ratio  # Mamba使用独立的较小学习率

    optimizer_grouped_parameters = [
        # CLIP parameters with coef_lr
        {"params": [p for n, p in decay_clip_param_tp], "weight_decay": weight_decay, "lr": args.lr * coef_lr},
        {"params": [p for n, p in no_decay_clip_param_tp], "weight_decay": 0.0, "lr": args.lr * coef_lr},
        # Mamba parameters with smaller lr
        {"params": [p for n, p in decay_mamba_param_tp], "weight_decay": weight_decay, "lr": mamba_lr},
        {"params": [p for n, p in no_decay_mamba_param_tp], "weight_decay": 0.0, "lr": mamba_lr},
        # Other parameters (default lr)
        {"params": [p for n, p in decay_other_param_tp], "weight_decay": weight_decay},
        {"params": [p for n, p in no_decay_other_param_tp], "weight_decay": 0.0},
    ]

    # 打印参数分组信息
    if local_rank == 0:
        n_clip = len(decay_clip_param_tp) + len(no_decay_clip_param_tp)
        n_mamba = len(decay_mamba_param_tp) + len(no_decay_mamba_param_tp)
        n_other = len(decay_other_param_tp) + len(no_decay_other_param_tp)
        logger.info(
            "[Optimizer] CLIP: %d params (lr=%.2e) | Mamba: %d params (lr=%.2e, ratio=%.2f) | Other: %d params (lr=%.2e)",
            n_clip, args.lr * coef_lr, n_mamba, mamba_lr, args.mamba_lr_ratio, n_other, args.lr,
        )

    scheduler = None
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=args.warmup_proportion,
        schedule="warmup_cosine",
        b1=0.9,
        b2=0.98,
        e=1e-6,
        t_total=num_train_optimization_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )

    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch)
    )
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch)
    )
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": tr_loss,
        },
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
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = (
            args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
        )
        model = UATVR.from_pretrained(
            args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args
        )
        model.to(device)
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


def _unpack_train_batch(batch, args):
    use_attributes = bool(getattr(args, "use_attributes", False))
    use_explicit_hn = bool(getattr(args, "use_explicit_hard_negative_loss", False))

    unpacked = {
        "input_ids": None,
        "input_mask": None,
        "segment_ids": None,
        "video": None,
        "video_mask": None,
        "sample_index": None,
        "hard_video": None,
        "hard_video_mask": None,
        "hard_valid": None,
    }

    if len(batch) == 5:
        unpacked["input_ids"], unpacked["input_mask"], unpacked["segment_ids"], unpacked["video"], unpacked["video_mask"] = batch
        return unpacked

    if len(batch) == 8:
        (
            unpacked["input_ids"],
            unpacked["input_mask"],
            unpacked["segment_ids"],
            _input_ids_a,
            _input_mask_a,
            _segment_ids_a,
            unpacked["video"],
            unpacked["video_mask"],
        ) = batch
        return unpacked

    if len(batch) == 6 and not use_attributes:
        (
            unpacked["input_ids"],
            unpacked["input_mask"],
            unpacked["segment_ids"],
            unpacked["video"],
            unpacked["video_mask"],
            unpacked["sample_index"],
        ) = batch
        return unpacked

    if len(batch) == 9 and use_explicit_hn and not use_attributes:
        (
            unpacked["input_ids"],
            unpacked["input_mask"],
            unpacked["segment_ids"],
            unpacked["video"],
            unpacked["video_mask"],
            unpacked["sample_index"],
            unpacked["hard_video"],
            unpacked["hard_video_mask"],
            unpacked["hard_valid"],
        ) = batch
        return unpacked

    if len(batch) == 9 and use_attributes and not use_explicit_hn:
        (
            unpacked["input_ids"],
            unpacked["input_mask"],
            unpacked["segment_ids"],
            _input_ids_a,
            _input_mask_a,
            _segment_ids_a,
            unpacked["video"],
            unpacked["video_mask"],
            unpacked["sample_index"],
        ) = batch
        return unpacked

    if len(batch) == 12 and use_attributes and use_explicit_hn:
        (
            unpacked["input_ids"],
            unpacked["input_mask"],
            unpacked["segment_ids"],
            _input_ids_a,
            _input_mask_a,
            _segment_ids_a,
            unpacked["video"],
            unpacked["video_mask"],
            unpacked["sample_index"],
            unpacked["hard_video"],
            unpacked["hard_video_mask"],
            unpacked["hard_valid"],
        ) = batch
        return unpacked

    raise ValueError(f"Unexpected batch size={len(batch)} from dataloader.")


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    gate_log_step = args.gate_log_interval if args.gate_log_interval is not None else args.n_display

    # 更新模型的当前 epoch，用于退火系数计算
    core = model.module if hasattr(model, "module") else model
    core._current_epoch = epoch
    start_time = time.time()
    epoch_start_time = time.time()
    total_loss = 0
    num_steps = len(train_dataloader)
    # 因果链累积器（epoch 级别平均）
    _causal_acc = {"diag": {}, "sap": {}, "prob": {}, "aux": {}, "n": 0}
    # 每个 optimizer step 对应 gradient_accumulation_steps 个 dataloader step
    total_opt_steps = num_steps // args.gradient_accumulation_steps

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self, not consider
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        batch_inputs = _unpack_train_batch(batch, args)
        loss_dict = model(
            batch_inputs["input_ids"],
            batch_inputs["segment_ids"],
            batch_inputs["input_mask"],
            batch_inputs["video"],
            batch_inputs["video_mask"],
            sample_index=batch_inputs["sample_index"],
            hard_video=batch_inputs["hard_video"],
            hard_video_mask=batch_inputs["hard_video_mask"],
            hard_valid=batch_inputs["hard_valid"],
        )
        loss = loss_dict["total"]

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            loss_dict = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)

        # 提取各个子损失的值（用于打印）
        # 仅保留标量/可转 float 的项，避免把大矩阵（例如 retrieve_logits）塞进日志
        loss_details = {}
        for k, v in loss_dict.items():
            if k == "total":
                continue
            if isinstance(v, torch.Tensor) and v.numel() != 1:
                continue
            try:
                loss_details[k] = float(v)
            except Exception:
                continue
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪从 1.0 降到 0.5，防止 NIG 层训练中期梯度崩塌
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, "module"):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            # ── 采集因果链诊断（每 opt_step 收集，rank 不限） ─────────────
            core = model.module if hasattr(model, "module") else model
            for _attr, _key in [
                ("_diag_chain", "diag"),
                ("_prob_chain", "prob"), ("_aux_chain", "aux"),
            ]:
                _d = getattr(core, _attr, None)
                if _d is not None:
                    acc = _causal_acc[_key]
                    for k, v in _d.items():
                        if isinstance(v, (int, float)):
                            acc[k] = acc.get(k, 0.0) + v
            _causal_acc["n"] += 1

            if global_step % log_step == 0 and local_rank == 0:
                # --- 计算进度 & ETA ---
                elapsed = time.time() - epoch_start_time
                opt_step = global_step % total_opt_steps or total_opt_steps
                progress = opt_step / total_opt_steps * 100
                time_per_step = (time.time() - start_time) / (log_step * args.gradient_accumulation_steps)
                remaining_steps = num_steps - (step + 1)
                eta = time_per_step * remaining_steps
                start_time = time.time()

                # --- 学习率 (param_groups: [0]CLIP-decay [1]CLIP-nodecay [2]Mamba-decay
                #                           [3]Mamba-nodecay [4]Other-decay [5]Other-nodecay) ---
                lr_clip = optimizer.param_groups[0]["lr"]
                lr_new  = optimizer.param_groups[4]["lr"] if len(optimizer.param_groups) > 4 else lr_clip

                # --- 损失字符串：过滤零值，精度 4 位 ---
                loss_parts = [f"{k}={v:.4f}" for k, v in loss_details.items() if abs(v) > 1e-8]
                loss_str = " | ".join(loss_parts) if loss_parts else "-"

                logger.info(
                    "[Epoch %d/%d] Step %d/%d (%.0f%%) | Loss: %.4f [%s] | LR clip=%.2e new=%.2e | %.2fs/step | ETA: %s",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    num_steps,
                    progress,
                    float(loss),
                    loss_str,
                    lr_clip,
                    lr_new,
                    time_per_step,
                    _fmt_time(eta),
                )

                # --- 因果链简报（每 log_step 打印一次当前值） ---
                _dc = getattr(core, "_diag_chain", None)
                _pc = getattr(core, "_prob_chain", None)
                _ac = getattr(core, "_aux_chain", None)
                if _dc:
                    logger.info(
                        "  [Chain-Ret]  pos=%.3f  neg=%.3f  gap=%.3f  pos_std=%.3f",
                        _dc["pos_mean"], _dc["neg_mean"], _dc["gap"], _dc["pos_std"],
                    )
                if _pc:
                    logger.info(
                        "  [Chain-Prob] u_mode=%.4f±%.4f  epi_v=%.4f  var_t=%.4f  kl_t=%.4f",
                        _pc.get("u_mode_mean", 0), _pc.get("u_mode_std", 0),
                        _pc.get("epistemic_v_mean", 0),
                        _pc.get("var_text_mean", 0),
                        _pc.get("kl_text_mean", 0),
                    )
                if _ac:
                    logger.info(
                        "  [Chain-Aux]  evid_loss=%.4f  neg_reg=%.4f  evid_unc=%.4f  "
                        "logsig_v=%.4f  clamp_v=%.2f/%.2f  clamp_t=%.2f/%.2f  anneal=%.3f",
                        _ac.get("evidential_loss_val", 0),
                        _ac.get("neg_reg_loss_val", 0),
                        _ac.get("evidential_uncertainty", 0),
                        _ac["logsigma_v_mean"],
                        _ac.get("logsigma_v_min_ratio", 0),
                        _ac.get("logsigma_v_max_ratio", 0),
                        _ac.get("logsigma_t_min_ratio", 0),
                        _ac.get("logsigma_t_max_ratio", 0),
                        _ac.get("anneal_factor", 1.0),
                    )
                    logger.info(
                        "  [Chain-Hygiene] score=%s profile=%s lambda_prob=%.3f lambda_anchor=%.3f "
                        "lambda_qc_sap=%.3f qc_temp=%.3f "
                        "mil=%d evid=%d neg=%d orth=%d hn=%d uacl=%d",
                        _ac.get("score_source", "wti_logits"),
                        _ac.get("experiment_profile", "default"),
                        _ac.get("lambda_prob", 0),
                        _ac.get("lambda_anchor", 0),
                        _ac.get("lambda_qc_sap", 0),
                        _ac.get("qc_sap_temperature", 0),
                        int(_ac.get("active_mil", 0)),
                        int(_ac.get("active_evidential", 0)),
                        int(_ac.get("active_neg_reg", 0)),
                        int(_ac.get("active_orth", 0)),
                        int(_ac.get("active_hard_negative", 0)),
                        int(_ac.get("active_uacl", 0)),
                    )
                    logger.info(
                        "  [Chain-Score] prob_mu=%.3f/%.3f/%.3f  anchor_wti=%.3f/%.3f/%.3f  "
                        "qc_sap=%.3f/%.3f/%.3f std=%.3f gate_ent=%.3f/%.3f gate_top1=%.3f/%.3f",
                        _ac.get("prob_mu_diag", 0),
                        _ac.get("prob_mu_off", 0),
                        _ac.get("prob_mu_gap", 0),
                        _ac.get("anchor_wti_diag", 0),
                        _ac.get("anchor_wti_off", 0),
                        _ac.get("anchor_wti_gap", 0),
                        _ac.get("qc_sap_diag", 0),
                        _ac.get("qc_sap_off", 0),
                        _ac.get("qc_sap_gap", 0),
                        _ac.get("qc_sap_std", 0),
                        _ac.get("qc_gate_entropy_pos", 0),
                        _ac.get("qc_gate_entropy_neg", 0),
                        _ac.get("qc_gate_top1_mass_pos", 0),
                        _ac.get("qc_gate_top1_mass_neg", 0),
                    )
                    if args.use_uacl_intra_alignment:
                        logger.info(
                            "  [Chain-UACL] intra=%.4f  text=%.4f  video=%.4f  kl=%.4f",
                            _ac.get("uacl_intra_loss_val", 0),
                            _ac.get("uacl_text_loss_val", 0),
                            _ac.get("uacl_video_loss_val", 0),
                            _ac.get("uacl_kl_loss_val", 0),
                        )

            if (
                args.log_moe_weights
                and (global_step % gate_log_step == 0)
                and local_rank == 0
            ):
                _log_moe_weights_tsv(args, model, epoch=epoch, step=step, global_step=global_step)

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
        # ── epoch 级因果链汇总行 ──────────────────────────────────────────
        _n = max(_causal_acc["n"], 1)
        _avg = lambda d: {k: v / _n for k, v in d.items()}
        dc = _avg(_causal_acc["diag"])
        sc = _avg(_causal_acc["sap"])
        pc = _avg(_causal_acc["prob"])
        ac = _avg(_causal_acc.get("aux", {}))
        if dc and pc:
            logger.info(
                "[Epoch %d Causal Summary]\n"
                "  Ret  : pos=%.3f  neg=%.3f  gap=%.3f  pos_std=%.3f\n"
                "  Evid : u_mode=%.4f±%.4f  epistemic_v=%.4f\n"
                "  Text : var_t=%.4f  kl_t=%.4f\n"
                "  Aux  : evid_loss=%.4f  neg_reg=%.4f  evid_unc=%.4f  logsig_v=%.4f  "
                "clamp_v=%.2f/%.2f  clamp_t=%.2f/%.2f\n"
                "  Hygiene : score=%s  profile=%s  lambda_prob=%.3f  lambda_anchor=%.3f  "
                "lambda_qc_sap=%.3f  qc_temp=%.3f  "
                "mil=%d  evid=%d  neg=%d  orth=%d  hn=%d  uacl=%d\n"
                "  Score : prob_mu=%.3f/%.3f/%.3f  anchor_wti=%.3f/%.3f/%.3f  "
                "qc_sap=%.3f/%.3f/%.3f std=%.3f gate_ent=%.3f/%.3f gate_top1=%.3f/%.3f\n"
                "  UACL : intra=%.4f  text=%.4f  video=%.4f  kl=%.4f",
                epoch + 1,
                dc.get("pos_mean", 0), dc.get("neg_mean", 0),
                dc.get("gap", 0), dc.get("pos_std", 0),
                pc.get("u_mode_mean", 0), pc.get("u_mode_std", 0),
                pc.get("epistemic_v_mean", 0),
                pc.get("var_text_mean", 0),
                pc.get("kl_text_mean", 0),
                ac.get("evidential_loss_val", 0), ac.get("neg_reg_loss_val", 0),
                ac.get("evidential_uncertainty", 0), ac.get("logsigma_v_mean", 0),
                ac.get("logsigma_v_min_ratio", 0), ac.get("logsigma_v_max_ratio", 0),
                ac.get("logsigma_t_min_ratio", 0), ac.get("logsigma_t_max_ratio", 0),
                getattr(args, "final_score_mode", "wti"),
                getattr(args, "experiment_profile", "default"),
                getattr(args, "lambda_prob", 0),
                getattr(args, "lambda_anchor", 0),
                getattr(args, "lambda_qc_sap", 0),
                getattr(args, "qc_sap_temperature", 0),
                int(round(ac.get("active_mil", 0))),
                int(round(ac.get("active_evidential", 0))),
                int(round(ac.get("active_neg_reg", 0))),
                int(round(ac.get("active_orth", 0))),
                int(round(ac.get("active_hard_negative", 0))),
                int(round(ac.get("active_uacl", 0))),
                ac.get("prob_mu_diag", 0), ac.get("prob_mu_off", 0), ac.get("prob_mu_gap", 0),
                ac.get("anchor_wti_diag", 0), ac.get("anchor_wti_off", 0), ac.get("anchor_wti_gap", 0),
                ac.get("qc_sap_diag", 0), ac.get("qc_sap_off", 0), ac.get("qc_sap_gap", 0),
                ac.get("qc_sap_std", 0),
                ac.get("qc_gate_entropy_pos", 0), ac.get("qc_gate_entropy_neg", 0),
                ac.get("qc_gate_top1_mass_pos", 0), ac.get("qc_gate_top1_mass_neg", 0),
                ac.get("uacl_intra_loss_val", 0), ac.get("uacl_text_loss_val", 0),
                ac.get("uacl_video_loss_val", 0), ac.get("uacl_kl_loss_val", 0),
            )
            _log_causal_summary_tsv(args, epoch, dc, pc, ac)
    return total_loss, global_step


def _log_moe_weights_tsv(args, model, epoch: int, step: int, global_step: int):
    """
    Write a single TSV line with uncertainty-driven MoE fusion weights summary statistics.
    This is rank0-only and is designed to be lightweight.

    We log both:
      - video-side fusion weights v_fusion_w: [B, 2] (base/query)
      - text-side fusion weights  t_fusion_w: [B, 2] (base/query)
    """
    import datetime as _dt

    # unwrap DDP
    core = model.module if hasattr(model, "module") else model
    v_w = getattr(core, "_last_moe_v_fusion_w", None)
    t_w = getattr(core, "_last_moe_t_fusion_w", None)
    if v_w is None and t_w is None:
        return

    def _stats_2(w):
        # w: [B, 2], each row sums to 1
        w = w.detach()
        # base/query components
        w0 = w[:, 0]
        w1 = w[:, 1]
        # entropy and effective number of experts
        entropy = -(w * (w + 1e-9).log()).sum(dim=-1)  # [B]
        eff_n = torch.exp(entropy)  # [B], in [1, 2]
        top1 = w.max(dim=-1).values  # [B], in [0.5, 1]
        # how often query dominates / collapses
        frac_q_lt_005 = float((w1 < 0.05).float().mean().item())
        frac_q_gt_095 = float((w1 > 0.95).float().mean().item())
        return {
            "w0_mean": float(w0.mean().item()),
            "w0_std": float(w0.std(unbiased=False).item()),
            "w1_mean": float(w1.mean().item()),
            "w1_std": float(w1.std(unbiased=False).item()),
            "ent_mean": float(entropy.mean().item()),
            "ent_std": float(entropy.std(unbiased=False).item()),
            "effn_mean": float(eff_n.mean().item()),
            "effn_std": float(eff_n.std(unbiased=False).item()),
            "top1_mean": float(top1.mean().item()),
            "top1_std": float(top1.std(unbiased=False).item()),
            "frac_q_lt_005": frac_q_lt_005,
            "frac_q_gt_095": frac_q_gt_095,
        }

    with torch.no_grad():
        v = _stats_2(v_w) if v_w is not None else None
        t = _stats_2(t_w) if t_w is not None else None

    date_str = _dt.datetime.now().strftime("%Y%m%d")
    run_id = os.path.basename(os.path.normpath(args.output_dir))
    out_dir = os.path.join(args.moe_log_dir, date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{run_id}.tsv")

    header = (
        "time\toutput_dir\trun_id\tepoch\tstep\tglobal_step\t"
        "v_w_base_mean\tv_w_base_std\tv_w_query_mean\tv_w_query_std\t"
        "v_entropy_mean\tv_entropy_std\tv_effn_mean\tv_effn_std\tv_top1_mean\tv_top1_std\t"
        "v_frac_q_lt_0.05\tv_frac_q_gt_0.95\t"
        "t_w_base_mean\tt_w_base_std\tt_w_query_mean\tt_w_query_std\t"
        "t_entropy_mean\tt_entropy_std\tt_effn_mean\tt_effn_std\tt_top1_mean\tt_top1_std\t"
        "t_frac_q_lt_0.05\tt_frac_q_gt_0.95\n"
    )

    def _fmt(x):
        return f"{x:.6f}" if isinstance(x, float) else "nan"

    line = (
        f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{args.output_dir}\t{run_id}\t"
        f"{epoch + 1}\t{step + 1}\t{global_step}\t"
        f"{_fmt(v['w0_mean']) if v else 'nan'}\t{_fmt(v['w0_std']) if v else 'nan'}\t"
        f"{_fmt(v['w1_mean']) if v else 'nan'}\t{_fmt(v['w1_std']) if v else 'nan'}\t"
        f"{_fmt(v['ent_mean']) if v else 'nan'}\t{_fmt(v['ent_std']) if v else 'nan'}\t"
        f"{_fmt(v['effn_mean']) if v else 'nan'}\t{_fmt(v['effn_std']) if v else 'nan'}\t"
        f"{_fmt(v['top1_mean']) if v else 'nan'}\t{_fmt(v['top1_std']) if v else 'nan'}\t"
        f"{_fmt(v['frac_q_lt_005']) if v else 'nan'}\t{_fmt(v['frac_q_gt_095']) if v else 'nan'}\t"
        f"{_fmt(t['w0_mean']) if t else 'nan'}\t{_fmt(t['w0_std']) if t else 'nan'}\t"
        f"{_fmt(t['w1_mean']) if t else 'nan'}\t{_fmt(t['w1_std']) if t else 'nan'}\t"
        f"{_fmt(t['ent_mean']) if t else 'nan'}\t{_fmt(t['ent_std']) if t else 'nan'}\t"
        f"{_fmt(t['effn_mean']) if t else 'nan'}\t{_fmt(t['effn_std']) if t else 'nan'}\t"
        f"{_fmt(t['top1_mean']) if t else 'nan'}\t{_fmt(t['top1_std']) if t else 'nan'}\t"
        f"{_fmt(t['frac_q_lt_005']) if t else 'nan'}\t{_fmt(t['frac_q_gt_095']) if t else 'nan'}\n"
    )

    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header)
        f.write(line)


def _log_causal_summary_tsv(args, epoch: int, dc: dict, pc: dict, ac: dict = None):
    """Write epoch-level causal-chain summary to TSV. rank0 only."""
    import datetime as _dt

    if ac is None:
        ac = {}
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    run_id = os.path.basename(os.path.normpath(args.output_dir))
    out_dir = os.path.join("logs", "causal_chain", date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{run_id}.tsv")

    header = (
        "time\trun_id\tepoch\t"
        "ret_pos_mean\tret_neg_mean\tret_gap\tret_pos_std\t"
        "u_mode_mean\tu_mode_std\tepistemic_v_mean\t"
        "var_text_mean\tkl_text_mean\t"
        "evidential_loss\tneg_reg_loss\tevidential_unc\tlogsigma_v\t"
        "logsigma_v_min_ratio\tlogsigma_v_max_ratio\tlogsigma_t_min_ratio\tlogsigma_t_max_ratio\t"
        "score_source\texperiment_profile\tfinal_score_mode\tlambda_prob\tlambda_anchor\t"
        "lambda_qc_sap\tqc_sap_temperature\t"
        "prob_mu_diag\tprob_mu_off\tprob_mu_gap\tanchor_wti_diag\tanchor_wti_off\tanchor_wti_gap\t"
        "qc_sap_diag\tqc_sap_off\tqc_sap_gap\tqc_sap_std\t"
        "qc_gate_entropy_pos\tqc_gate_entropy_neg\tqc_gate_top1_mass_pos\tqc_gate_top1_mass_neg\t"
        "active_mil\tactive_evidential\tactive_neg_reg\tactive_orth\tactive_hard_negative\tactive_uacl\t"
        "uacl_intra_loss\tuacl_text_loss\tuacl_video_loss\tuacl_kl_loss\n"
    )
    def _f(x): return f"{x:.6f}" if isinstance(x, float) else "nan"
    final_score_mode = getattr(args, "final_score_mode", "wti")
    score_source = "wti_logits" if final_score_mode == "wti" else final_score_mode
    line = (
        f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{run_id}\t{epoch + 1}\t"
        f"{_f(dc.get('pos_mean',0))}\t{_f(dc.get('neg_mean',0))}\t"
        f"{_f(dc.get('gap',0))}\t{_f(dc.get('pos_std',0))}\t"
        f"{_f(pc.get('u_mode_mean',0))}\t{_f(pc.get('u_mode_std',0))}\t"
        f"{_f(pc.get('epistemic_v_mean',0))}\t"
        f"{_f(pc.get('var_text_mean',0))}\t{_f(pc.get('kl_text_mean',0))}\t"
        f"{_f(ac.get('evidential_loss_val',0))}\t{_f(ac.get('neg_reg_loss_val',0))}\t"
        f"{_f(ac.get('evidential_uncertainty',0))}\t{_f(ac.get('logsigma_v_mean',0))}\t"
        f"{_f(ac.get('logsigma_v_min_ratio',0))}\t{_f(ac.get('logsigma_v_max_ratio',0))}\t"
        f"{_f(ac.get('logsigma_t_min_ratio',0))}\t{_f(ac.get('logsigma_t_max_ratio',0))}\t"
        f"{score_source}\t{getattr(args, 'experiment_profile', 'default')}\t"
        f"{final_score_mode}\t{_f(getattr(args, 'lambda_prob', 0.0))}\t"
        f"{_f(getattr(args, 'lambda_anchor', 0.0))}\t"
        f"{_f(getattr(args, 'lambda_qc_sap', 0.0))}\t"
        f"{_f(getattr(args, 'qc_sap_temperature', 0.0))}\t"
        f"{_f(ac.get('prob_mu_diag',0))}\t{_f(ac.get('prob_mu_off',0))}\t{_f(ac.get('prob_mu_gap',0))}\t"
        f"{_f(ac.get('anchor_wti_diag',0))}\t{_f(ac.get('anchor_wti_off',0))}\t"
        f"{_f(ac.get('anchor_wti_gap',0))}\t"
        f"{_f(ac.get('qc_sap_diag',0))}\t{_f(ac.get('qc_sap_off',0))}\t"
        f"{_f(ac.get('qc_sap_gap',0))}\t{_f(ac.get('qc_sap_std',0))}\t"
        f"{_f(ac.get('qc_gate_entropy_pos',0))}\t{_f(ac.get('qc_gate_entropy_neg',0))}\t"
        f"{_f(ac.get('qc_gate_top1_mass_pos',0))}\t{_f(ac.get('qc_gate_top1_mass_neg',0))}\t"
        f"{int(round(ac.get('active_mil',0)))}\t{int(round(ac.get('active_evidential',0)))}\t"
        f"{int(round(ac.get('active_neg_reg',0)))}\t{int(round(ac.get('active_orth',0)))}\t"
        f"{int(round(ac.get('active_hard_negative',0)))}\t{int(round(ac.get('active_uacl',0)))}\t"
        f"{_f(ac.get('uacl_intra_loss_val',0))}\t{_f(ac.get('uacl_text_loss_val',0))}\t"
        f"{_f(ac.get('uacl_video_loss_val',0))}\t{_f(ac.get('uacl_kl_loss_val',0))}\n"
    )
    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header)
        f.write(line)


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


def _log_eval_stats_tsv(args, model, step):
    """
    Write logits stats to TSV during evaluation. rank0 only.
    """
    import datetime as _dt

    # unwrap DDP
    core = model.module if hasattr(model, "module") else model
    stats = getattr(core, "_last_eval_stats", None)
    if stats is None:
        return

    # log path: logs/logits_stats/YYYYMMDD/<run_id>.tsv
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    run_id = os.path.basename(os.path.normpath(args.output_dir))
    # If output_dir is not set or default, use 'eval' as fallback
    if not run_id:
        run_id = "eval_debug"

    out_dir = os.path.join("logs", "logits_stats", date_str)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{run_id}.tsv")

    header = "time\tstep\tbase_mean\tbase_std\tquery_mean\tquery_std\tv_w_base\tv_w_query\tt_w_base\tt_w_query\n"

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (float, int)) else "nan"

    line = (
        f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{step}\t"
        f"{_fmt(stats.get('base_mean'))}\t{_fmt(stats.get('base_std'))}\t"
        f"{_fmt(stats.get('query_mean'))}\t{_fmt(stats.get('query_std'))}\t"
        f"{_fmt(stats.get('v_w_base'))}\t{_fmt(stats.get('v_w_query'))}\t"
        f"{_fmt(stats.get('t_w_base'))}\t{_fmt(stats.get('t_w_query'))}\n"
    )

    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header)
        f.write(line)


def _run_on_single_gpu(
    args,
    model,
    batch_list_t,
    batch_list_v,
    batch_sequence_output_list,
    batch_visual_output_list,
):
    # Cat on CPU — features were already moved to CPU in eval_epoch to reduce GPU pressure
    visual_output_cls_all = torch.cat([v[0] for v in batch_visual_output_list], dim=0)
    visual_output_all_full = torch.cat([v[1] for v in batch_visual_output_list], dim=0)
    video_mask_all = torch.cat([v[0] for v in batch_list_v], dim=0)

    device = next(model.parameters()).device
    chunk_size = getattr(args, "eval_vid_chunk_size", 128)
    n_vid = visual_output_cls_all.size(0)

    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output, text_token = batch_sequence_output_list[idx1]

        seq_dev  = sequence_output.to(device)
        tok_dev  = text_token.to(device)
        mask_dev = input_mask.to(device)

        row_logits = []
        for v_start in range(0, n_vid, chunk_size):
            v_end       = min(v_start + chunk_size, n_vid)
            cls_chunk   = visual_output_cls_all[v_start:v_end].to(device)
            full_chunk  = visual_output_all_full[v_start:v_end].to(device)
            vmask_chunk = video_mask_all[v_start:v_end].to(device)

            chunk_logits, *_tmp2 = model.get_similarity_logits(
                seq_dev,
                tok_dev,
                cls_chunk,
                full_chunk,
                mask_dev,
                vmask_chunk,
                loose_type=model.loose_type,
            )
            row_logits.append(chunk_logits.cpu())

        b1_all_v_logits = torch.cat(row_logits, dim=1)  # [B_t, N_v]
        _log_eval_stats_tsv(args, model, idx1)
        sim_matrix.append(b1_all_v_logits.detach().numpy())

    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu):
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
        hasattr(test_dataloader.dataset, "multi_sentence_per_video")
        and test_dataloader.dataset.multi_sentence_per_video
    ):
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        progress_bar = tqdm(
            test_dataloader, desc="Evaluating", leave=False, disable=getattr(args, "local_rank", 0) != 0
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
                sequence_output, text_token = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(
                    (
                        sequence_output,
                        text_token,
                    )
                )
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    )
                )

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, text_token, visual_output, visual_output_all = model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, video, video_mask
                )

                # Move to CPU immediately to reduce peak GPU memory during similarity computation
                batch_sequence_output_list.append((sequence_output.cpu(), text_token.cpu()))
                batch_list_t.append((input_mask.cpu(), segment_ids.cpu()))
                batch_visual_output_list.append((visual_output.cpu(), visual_output_all.cpu()))
                batch_list_v.append((video_mask.cpu(),))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        torch.cuda.empty_cache()
        if n_gpu > 1 and "LOCAL_RANK" not in os.environ:
            device_ids = list(range(n_gpu))
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

                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_visual_output_list]
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
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(
                args,
                model,
                batch_list_t,
                batch_list_v,
                batch_sequence_output_list,
                batch_visual_output_list,
            )
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # MUS 诊断日志：在相似度矩阵整合完成后、指标计算前输出
    if getattr(args, "log_mus_scores", False):
        _log_mus_scores_tsv(args, sim_matrix)

    if args.DSL:  # using dual softmax for test
        logger.info("\t Using Dual Softmax testing.")
        sim_matrix = torch.from_numpy(sim_matrix)
        v2t_matrix = sim_matrix.T

        sim_matrix = sim_matrix * F.softmax(sim_matrix / 1, dim=0) * len(sim_matrix)
        sim_matrix = sim_matrix.detach().numpy()

        v2t_matrix = v2t_matrix * F.softmax(v2t_matrix / 1, dim=0) * len(v2t_matrix)
        v2t_matrix = v2t_matrix.detach().numpy()

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(
                np.concatenate(
                    (sim_matrix[s_:e_], np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)), axis=0
                )
            )
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]
            )
        )

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("Retrieval Evaluation | #Text: %d, #Video: %d", sim_matrix.shape[0], sim_matrix.shape[1])
        tv_metrics = compute_metrics(sim_matrix)
        if not args.DSL:
            vt_metrics = compute_metrics(sim_matrix.T)
        else:
            vt_metrics = compute_metrics(v2t_matrix)

    logger.info(
        "  T2V | R@1: %5.1f  R@5: %5.1f  R@10: %5.1f  MdR: %5.1f  MnR: %5.1f",
        tv_metrics["R1"], tv_metrics["R5"], tv_metrics["R10"], tv_metrics["MR"], tv_metrics["MeanR"],
    )
    logger.info(
        "  V2T | R@1: %5.1f  R@5: %5.1f  R@10: %5.1f  MdR: %5.1f  MnR: %5.1f",
        vt_metrics["R1"], vt_metrics["R5"], vt_metrics["R10"], vt_metrics["MR"], vt_metrics["MeanR"],
    )

    R1 = tv_metrics["R1"]
    return R1


def main():
    global logger
    import torch.distributed as dist

    args = get_args()

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

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if (
                name.find("ln_final.") == 0
                or name.find("text_projection") == 0
                or name.find("logit_scale") == 0
                or name.find("visual.ln_post.") == 0
                or name.find("visual.proj") == 0
            ):
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:  # false pass
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](
            args, tokenizer, subset="val"
        )  # through this
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:  # false pass
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        frame_order_name = {0: "sequential", 1: "reverse", 2: "shuffle"}.get(args.train_frame_order, "unknown")
        logger.info("  Test: %d samples, %d steps (bs=%d) | Val: %d samples | Frame order: %s",
                     test_length, len(test_dataloader), args.batch_size_val, val_length, frame_order_name)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (
            int(len(train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps
        ) * args.epochs

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
                args.batch_size * args.gradient_accumulation_steps,
                len(train_dataloader),
                num_train_optimization_steps,
            )

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location="cpu")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resumed_epoch = checkpoint["epoch"] + 1
            # resumed_loss = checkpoint['loss']  # unused (kept for potential debugging)

        global_step = 0

        if args.local_rank == 0:
            if resumed_epoch > 0:
                logger.info("Resuming training from epoch %d", resumed_epoch + 1)
            logger.info("=" * 60)

        state = True  # everything is correct

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

            try:
                if args.local_rank == 0:
                    # 打印显存占用信息
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
                        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
                        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3  # GB
                        logger.info(
                            "  GPU Memory | Cur: %.2f GB, Peak: %.2f GB, Reserved: %.2f GB",
                            current_memory,
                            max_memory,
                            reserved_memory,
                        )
                        # 重置最大显存统计，以便下一轮重新统计
                        torch.cuda.reset_peak_memory_stats(device)

                    output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                    # Run on val dataset, this process is *TIME-consuming*.
                    # logger.info("Eval on val dataset")
                    # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                    R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                    if best_score <= R1:
                        best_score = R1
                        best_output_model_file = output_model_file
                    logger.info("  Best so far | R@1: %.1f | %s", best_score, best_output_model_file)
                    logger.info("=" * 60)
                    state = True
            except Exception as e:
                logger.error("Error occurred during evaluation: %s", str(e))
                logger.error("Error type: %s", type(e).__name__)
                import traceback

                logger.error("Traceback:\n%s", traceback.format_exc())
                logger.info("Skipping evaluation for this epoch. Testing model at the end!")
                state = False
                continue

        ## 训练中每轮评估已记录最佳分数，最终重新加载评估是冗余的。
        ## 如需在最佳 checkpoint 上做最终测试，取消下面注释：
        # if args.local_rank == 0 and state:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            # 复用主脚本评估，使用传入的 --init_model 作为权重路径
            if args.init_model is None:
                raise ValueError("--do_eval 需要指定 --init_model=<checkpoint_path>")
            model = load_model(-1, args, n_gpu, device, model_file=args.init_model)
            eval_epoch(args, model, test_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
