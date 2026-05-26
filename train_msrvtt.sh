#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/data2/hxj/data/MSRVTT
# Use a merged attributes map (train9k + jsfusion test1k) to ensure eval split coverage.
# The MSRVTT dataloader supports comma-separated paths and will merge them at runtime.
ATTRIBUTES_PATH=/data2/hxj/project/UATVR/deploy_qwen/attributes/msrvtt/final/msrvtt_train9k_attributes.json,/data2/hxj/project/UATVR/deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json
MAX_WORDS_ATTRS=77
ATTR_NUM_BLOCKS=4

# Auto-run id to avoid overwriting checkpoints/logs across runs
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-ckpts/ckpt_msrvtt_${RUN_ID}}

# 显存：batch 256 + accum 1，有效 batch = 256。2 卡时每卡 micro-batch 128。
# 启动前请确认所选 GPU 无其他大进程（nvidia-smi）；默认避开常被占用的 GPU 4。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#_GPUS[@]}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    torchrun --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29548 \
    main_task_retrieval.py \
    --do_train --num_thread_reader=8 --epochs=5 \
    --batch_size=256 --gradient_accumulation_steps=2 --n_display=20 \
    --train_csv "${DATA_PATH}/csv/MSRVTT_train.9k.csv" \
    --val_csv "${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" \
    --data_path "${DATA_PATH}/annotation/MSRVTT_v2.json" \
    --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
    --output_dir "${OUTPUT_DIR}" \
    --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 16 \
    --datatype msrvtt --expand_msrvtt_sentences \
    --feature_framerate 1 --coef_lr 1e-3 \
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
    --log_sigma_min -3 \
    --log_sigma_max 6 \
    --w_vib 5e-2 \
    --w_orth 0.1 \
    --w_uncertainty_reg 1e-3 \
    --use_tas_uncertainty \
    --gate_log_interval 100 \
    --log_moe_weights \
    --fusion_mode prob_mos \
    --w_query_sim 0.5 \
    --fusion_temperature 1.5 \
    \
    --rope_mode 2d \
    --use_ada_norm \
    --experiment_desc "${EXPERIMENT_DESC:-}" # --enhanced_fusion_input \
# --use_attributes \
# --msrvtt_attributes_path "${ATTRIBUTES_PATH}" \
# --max_words_attrs "${MAX_WORDS_ATTRS}" \
# --attr_num_blocks "${ATTR_NUM_BLOCKS}"
