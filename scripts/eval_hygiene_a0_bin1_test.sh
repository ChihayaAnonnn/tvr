#!/usr/bin/env bash
# One-off diagnostic: run the JSFUSION test set against hygiene_a0_seed0's
# epoch-2 checkpoint (bin.1), the val-selected T2V winner after only three
# epochs of a five-epoch run that OOM'd on epoch 4 because another user's
# vLLM job landed on GPU 0 of the shared box.
#
# Single-GPU eval on GPU 1: the two eval loops in main_task_retrieval.py
# only run on local_rank 0, and GPU 0 has 26 GB of somebody else's model on
# it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT_DIR="ckpts/ckpt_msrvtt_hygiene_a0_seed0"
INIT_MODEL="${CKPT_DIR}/pytorch_model.bin.1"
if [[ ! -f "${INIT_MODEL}" ]]; then
    echo "missing checkpoint: ${INIT_MODEL}" >&2
    exit 2
fi

DATA_PATH=/data2/hxj/data/MSRVTT
LOG_DIR="${ROOT_DIR}/logs/$(date +%Y%m%d)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/hygiene_a0_seed0_bin1_test_$(date +%H%M%S).log"

echo "[eval_hygiene_a0_bin1_test] init_model=${INIT_MODEL}"
echo "[eval_hygiene_a0_bin1_test] log=${LOG_FILE}"

CUDA_VISIBLE_DEVICES=1 \
CLIP_CACHE_DIR="${ROOT_DIR}/.cache" \
OMP_NUM_THREADS=1 \
/home/xujie/.conda/envs/tvr/bin/torchrun \
    --nproc_per_node=1 \
    --master_addr=127.0.0.9 \
    --master_port=29551 \
    "${ROOT_DIR}/main_task_retrieval.py" \
    --do_eval \
    --init_model "${INIT_MODEL}" \
    --num_thread_reader 4 \
    --prefetch_factor 2 \
    --batch_size 512 \
    --gradient_accumulation_steps 1 \
    --batch_size_val 16 \
    --train_csv "${ROOT_DIR}/data/generated/msrvtt_trusted_v1/train.csv" \
    --val_csv "${ROOT_DIR}/data/generated/msrvtt_trusted_v1/val.csv" \
    --source_train_csv "${DATA_PATH}/csv/MSRVTT_train.9k.csv" \
    --test_csv "${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" \
    --split_manifest "${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed0.json" \
    --eval_split test \
    --data_path "${DATA_PATH}/annotation/MSRVTT_v2.json" \
    --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
    --output_dir "${CKPT_DIR}" \
    --max_words 32 \
    --max_frames 12 \
    --datatype msrvtt \
    --expand_msrvtt_sentences \
    --feature_framerate 1 \
    --slice_framepos 2 \
    --linear_patch 2d \
    --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --clip_layer_norm_precision fp16 \
    --clip_gradient_checkpointing \
    --clip_visual_checkpoint_layers 12 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --experiment_profile hygiene \
    --experiment_desc "final test on bin.1 (epoch-2 val-best) after OOM" \
    --rspr_mode off \
    2>&1 | tee "${LOG_FILE}"
