from __future__ import division, print_function, unicode_literals

import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.data_dataloaders import DATALOADER_DICT
from dataloaders.msrvtt_protocol import load_trusted_manifest, validate_trusted_manifest
from experiment_tracking import (
    atomic_write_json,
    build_experiment_manifest,
    collect_git_state,
    compute_batch_semantics,
    is_global_rank_zero,
)
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UATVR
from modules.optimization import BertAdam
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
        default="dataloaders/splits/msrvtt_trusted_v1_seed42.json",
        help="Versioned trusted-v1 MSRVTT split manifest.",
    )
    parser.add_argument(
        "--eval_split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split. Training always evaluates internal val.",
    )
    parser.add_argument("--data_path", type=str, default="data/caption.pickle", help="data pickle file path")
    parser.add_argument("--features_path", type=str, default="data/videos_feature.pickle", help="feature path")
    parser.add_argument(
        "--tqfs_cache_dir",
        type=str,
        default="",
        help=(
            "Optional shared cache of preprocessed TQFS frames keyed by video_id. "
            "Cache misses are populated atomically."
        ),
    )

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
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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
    args = parser.parse_args()

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
        train_factory = loaders.get("train")
        if train_factory is None:
            raise ValueError(f"{args.datatype} has no train dataloader")
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
                "tqfs_cache_dir",
                "experiment_profile",
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
        # Prepare model
        cache_dir = (
            args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
        )
        model = UATVR.from_pretrained(
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


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    epoch_start_time = time.time()
    total_loss = 0
    num_steps = len(train_dataloader)
    # 每个 optimizer step 对应 gradient_accumulation_steps 个 dataloader step
    total_opt_steps = num_steps // args.gradient_accumulation_steps

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, group_ids = (
            _unpack_train_batch(batch)
        )
        warmup_epochs = max(getattr(args, "rspr_warmup_epochs", 0.0), 0.0)
        progress_epoch = epoch + step / max(num_steps, 1)
        rspr_warmup_scale = (
            1.0
            if warmup_epochs == 0
            else min(1.0, progress_epoch / warmup_epochs)
        )
        loss = model(
            input_ids,
            segment_ids,
            input_mask,
            video,
            video_mask,
            group_ids=group_ids,
            rspr_rank_scale=rspr_warmup_scale,
            rspr_anchor_scale=rspr_warmup_scale,
        )

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            if hasattr(optimizer, "get_group_lrs"):
                scheduled_group_lrs = optimizer.get_group_lrs()
            else:
                scheduled_group_lrs = [
                    group["lr"] for group in optimizer.param_groups
                ]
            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, "module"):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1

            if global_step % log_step == 0 and local_rank == 0:
                # --- 计算进度 & ETA ---
                opt_step = global_step % total_opt_steps or total_opt_steps
                progress = opt_step / total_opt_steps * 100
                time_per_step = (time.time() - start_time) / (log_step * args.gradient_accumulation_steps)
                remaining_steps = num_steps - (step + 1)
                eta = time_per_step * remaining_steps
                start_time = time.time()

                lr_clip = scheduled_group_lrs[0]
                lr_new = (
                    scheduled_group_lrs[2]
                    if len(scheduled_group_lrs) > 2
                    else lr_clip
                )

                logger.info(
                    "[Epoch %d/%d] Step %d/%d (%.0f%%) | Loss: %.4f | LR clip=%.2e new=%.2e | %.2fs/step | ETA: %s",
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
    visual_output_all = torch.cat(batch_visual_output_list, dim=0)
    video_mask_all = torch.cat([v[0] for v in batch_list_v], dim=0)

    device = next(model.parameters()).device
    chunk_size = getattr(args, "eval_vid_chunk_size", 128)
    n_vid = visual_output_all.size(0)
    video_chunks = []
    for v_start in range(0, n_vid, chunk_size):
        v_end = min(v_start + chunk_size, n_vid)
        video_chunks.append(
            (
                visual_output_all[v_start:v_end].to(device),
                video_mask_all[v_start:v_end].to(device),
            )
        )
    del visual_output_all, video_mask_all

    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, _segment_ids, *_tmp = b1
        sequence_output, text_token = batch_sequence_output_list[idx1]
        sequence_output = sequence_output.to(device)
        text_token = text_token.to(device)
        input_mask = input_mask.to(device)

        row_logits = []
        for visual_output, video_mask in video_chunks:
            logits, *_tmp = model.get_similarity_logits(
                sequence_output,
                text_token,
                visual_output,
                input_mask,
                video_mask,
                loose_type=model.loose_type,
            )
            row_logits.append(logits)

        b1_all_v_logits = torch.cat(row_logits, dim=1).cpu()
        sim_matrix.append(b1_all_v_logits.detach().numpy())

    return sim_matrix


def select_best_checkpoint(
    best_score, best_path, candidate_score, candidate_path
):
    """Select by validation T2V R@1; ties deliberately prefer the later epoch."""
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


def evaluate_training_checkpoint(
    args,
    model,
    eval_dataloader,
    device,
    n_gpu,
    best_score,
    best_path,
    candidate_path,
):
    """Evaluate one candidate without suppressing validation failures."""
    candidate_score = eval_epoch(args, model, eval_dataloader, device, n_gpu)
    return select_best_checkpoint(
        best_score, best_path, candidate_score, candidate_path
    )


def eval_epoch(args, model, eval_dataloader, device, n_gpu):
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
    with torch.no_grad():
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
                    batch_visual_output_list.append(visual_output.cpu())
                    batch_list_v.append((video_mask.cpu(),))
                total_video_num += b
            else:
                sequence_output, text_token, visual_output = (
                    model.get_sequence_visual_output(
                        input_ids,
                        segment_ids,
                        input_mask,
                        video,
                        video_mask,
                    )
                )

                # Move to CPU immediately to reduce peak GPU memory during similarity computation
                batch_sequence_output_list.append(
                    (sequence_output.cpu(), text_token.cpu())
                )
                batch_list_t.append((input_mask.cpu(), segment_ids.cpu()))
                batch_visual_output_list.append(visual_output.cpu())
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

                    devc_batch_list = [
                        tuple(tensor.to(devc) for tensor in output)
                        for output in batch_sequence_output_list[s_:e_]
                    ]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [tensor.to(devc) for tensor in batch_visual_output_list]
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
                model,
                args,
                batch_list_t,
                batch_list_v,
                batch_sequence_output_list,
                batch_visual_output_list,
            )
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # MUS 诊断日志：在相似度矩阵整合完成后、指标计算前输出
    if getattr(args, "log_mus_scores", False):
        _log_mus_scores_tsv(args, sim_matrix)

    if args.DSL:  # using dual softmax during evaluation
        logger.info("\t Using Dual Softmax evaluation.")
        sim_matrix = torch.from_numpy(sim_matrix)
        v2t_matrix = sim_matrix.T

        sim_matrix = sim_matrix * F.softmax(sim_matrix / 1, dim=0) * len(sim_matrix)
        sim_matrix = sim_matrix.detach().numpy()

        v2t_matrix = v2t_matrix * F.softmax(v2t_matrix / 1, dim=0) * len(v2t_matrix)
        v2t_matrix = v2t_matrix.detach().numpy()

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        max_length = max(
            e_ - s_
            for s_, e_ in zip([0] + cut_off_points_[:-1], cut_off_points_)
        )
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points_[:-1], cut_off_points_):
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

        best_score = float("-inf")
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
                    epoch, args, model, optimizer, tr_loss, type_name=""
                )
                best_score, best_output_model_file = evaluate_training_checkpoint(
                    args=args,
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    n_gpu=n_gpu,
                    best_score=best_score,
                    best_path=best_output_model_file,
                    candidate_path=output_model_file,
                )
                logger.info(
                    "  Best %s T2V R@1: %.1f | %s",
                    eval_split,
                    best_score,
                    best_output_model_file,
                )
                logger.info("=" * 60)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, eval_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
