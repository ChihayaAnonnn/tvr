#!/usr/bin/env bash
set -euo pipefail

# 抑制 DDP 多卡重复警告（Grad strides do not match 等）
export TORCH_WARN_ONCE=1

DATA_PATH=/data2/hxj/data/MSVD

# Auto-run id to avoid overwriting checkpoints/logs across runs
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
COEF_LR=${COEF_LR:-1e-3}
OUTPUT_DIR=${OUTPUT_DIR:-ckpts/ckpt_msvd_${RUN_ID}}
# Query-branch training knobs
# Query head is fixed to token_wti in codebase.
W_QUERY_SIM="0.1"            # increase to make query branch stand-alone

# Baseline replication: same three weights zero as MSRVTT baseline (frame-only WTI for retrieval logits).

CUDA_VISIBLE_DEVICES=1,2,3,4 \
    torchrun --nproc_per_node=4 --master_addr=127.0.0.9 --master_port=29509 \
    main_task_retrieval.py \
    --do_train --num_thread_reader=8 --epochs=5 --batch_size=128 --n_display=20 \
    --data_path ${DATA_PATH}/desc_files \
    --features_path "${DATA_PATH}/YouTubeClips" \
    --output_dir "${OUTPUT_DIR}" \
    --lr 1e-5 --max_words 32 --max_frames 12 --batch_size_val 8 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr "${COEF_LR}" \
    --freeze_layer_num 0 --slice_framepos 3 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --strategy 2 \
    --pretrained_clip_name ViT-B/16 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --n_video_embeddings 7 \
    --n_text_embeddings 7 \
    --mamba_lr_ratio 0.1 \
    --uncertainty_text_head text \
    --log_sigma_min -6 \
    --log_sigma_max 6 \
    --w_uncertainty_reg 1e-3 \
    --gate_log_interval 100 \
    --w_query_sim "${W_QUERY_SIM}" \
    --experiment_desc "${EXPERIMENT_DESC:-}"
