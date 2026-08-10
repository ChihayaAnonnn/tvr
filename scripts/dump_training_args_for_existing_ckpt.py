"""One-shot: reconstruct training_args.json for an existing CRF ladder ckpt.

The audit CLI (Task 9 completion) reads ``<ckpt_dir>/training_args.json`` to
rebuild the exact UATVR architecture. Runs finished BEFORE
``_write_training_args_json`` was wired do not carry that file. This helper
reconstructs it by replaying the runner's fixed argv through ``get_args``.

Usage:
    python scripts/dump_training_args_for_existing_ckpt.py \\
        --output-dir ckpts/ckpt_msrvtt_crf_20260810_025752_crf_ladder_full \\
        --features-path /data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/ \\
        --data-path /data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json \\
        --train-csv .../train.csv --val-csv .../val.csv \\
        --source-train-csv .../MSRVTT_train.9k.csv \\
        --test-csv .../MSRVTT_JSFUSION_test.csv \\
        --split-manifest .../msrvtt_trusted_v1_seed0.json \\
        --crf-ladder-root data/generated/msrvtt_crf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--features-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--source-train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--crf-ladder-root", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--freeze-layer-num", type=int, default=0)
    p.add_argument("--clip-layer-norm-precision", default="fp16")
    p.add_argument("--clip-visual-checkpoint-layers", type=int, default=12)
    return p.parse_args()


def build_argv(a: argparse.Namespace) -> list[str]:
    argv = [
        "--do_train",
        "--num_thread_reader", "12",
        "--prefetch_factor", "4",
        f"--epochs={a.epochs}",
        "--batch_size", str(a.batch_size),
        "--gradient_accumulation_steps", str(a.gradient_accumulation_steps),
        "--n_display=20",
        "--train_csv", a.train_csv,
        "--val_csv", a.val_csv,
        "--source_train_csv", a.source_train_csv,
        "--test_csv", a.test_csv,
        "--split_manifest", a.split_manifest,
        "--eval_split", "val",
        "--data_path", a.data_path,
        "--features_path", a.features_path,
        "--output_dir", a.output_dir,
        "--lr", "1e-4",
        "--max_words", str(a.max_words),
        "--max_frames", str(a.max_frames),
        "--batch_size_val", "16",
        "--datatype", "msrvtt",
        "--expand_msrvtt_sentences",
        "--feature_framerate", "1",
        "--coef_lr", "1e-3",
        "--freeze_layer_num", str(a.freeze_layer_num),
        "--slice_framepos", "0",
        "--linear_patch", "2d",
        "--sim_header", "seqTransf",
        "--pretrained_clip_name", "ViT-B/16",
        "--clip_layer_norm_precision", a.clip_layer_norm_precision,
        "--clip_gradient_checkpointing",
        "--clip_visual_checkpoint_layers", str(a.clip_visual_checkpoint_layers),
        "--extra_video_cls_num", "2",
        "--extra_text_cls_num", "2",
        "--experiment_profile", "default",
        "--experiment_desc", "crf-degradation-ladder",
        "--aleatoric_crf_training",
        "--crf_ladder_root", a.crf_ladder_root,
        "--aleatoric_prior_weight", "0",
        "--aleatoric_pooling_temperature", "0",
        "--aleatoric_ranking_weight", "1.0",
        "--aleatoric_base_margin", "0.05",
        "--aleatoric_lambda_cf", "1.0",
        "--aleatoric_lambda_cv", "1.0",
        "--aleatoric_lambda_of", "1.0",
        "--aleatoric_lambda_ov", "1.0",
        "--crf_order_fraction", "0.20",
    ]
    return argv


def main() -> int:
    cli = parse_args()
    # Import late so `--help` doesn't trigger CLIP init.
    from main_task_retrieval import _write_training_args_json, get_args

    sys.argv = ["main_task_retrieval.py"] + build_argv(cli)
    args = get_args()
    args.local_rank = 0
    args.world_size = 1
    args.n_gpu = 1

    target = Path(cli.output_dir) / "training_args.json"
    _write_training_args_json(args, target)
    body = json.loads(target.read_text())
    print(f"wrote {target} ({len(body)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
