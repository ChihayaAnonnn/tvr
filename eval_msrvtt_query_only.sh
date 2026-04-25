#!/usr/bin/env bash
set -euo pipefail

# MSRVTT query-only evaluation script (arguments are fixed in this file).
# Modify MODEL_PATH / DATA_PATH here when needed.

DATA_PATH=/data2/hxj/data/MSRVTT

# Default to the best checkpoint from logs/20260111/153216_train_msrvtt.log
MODEL_PATH=ckpts/ckpt_msrvtt_20260111_153216/pytorch_model.bin.3

CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 --master_addr=127.0.0.9 --master_port=29519 \
    main_task_retrieval.py \
    --do_eval \
    --output_dir ckpts/eval_msrvtt_query_only \
    --datatype msrvtt \
    --val_csv ${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/annotation/MSRVTT_v2.json \
    --features_path ${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/ \
    --pretrained_clip_name ViT-B/16 \
    --linear_patch 2d \
    --sim_header seqTransf \
    --strategy 2 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --max_words 32 \
    --max_frames 12 \
    --feature_framerate 1 \
    --batch_size_val 8 \
    --loose_type \
    --slice_framepos 2 \
    --uncertainty_text_head text \
    --log_sigma_min -6 \
    --log_sigma_max 6 \
    --eval_branch_mode query_only \
    --init_model ${MODEL_PATH}

