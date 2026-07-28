#!/usr/bin/env bash
# Score a finished run's val-selected checkpoints on the JSFUSION test split.
#
# For a training run that completed its epochs but not its test pass --
# hygiene_a0_seed0 lost that pass to a NameError after three hours, with all
# five checkpoints already on disk. The selection is read from the run's own
# best_validation_checkpoints.json, so this produces the same final_test.json
# the in-loop path would have produced; it does not accept a checkpoint on the
# command line, because a hand-picked one is a number no validation chose.
#
#   CKPT_DIR=ckpts/ckpt_msrvtt_hygiene_a0_seed0 ./scripts/final_test_from_ckpt.sh
#
# Single GPU: eval_epoch only runs on local_rank 0 anyway, so extra ranks would
# just idle while holding memory on a shared box. GPU_ID picks which one.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# No apostrophe in the :? message -- bash re-parses quotes inside ${var:?word},
# so one there swallows the rest of the script.
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the checkpoint directory of the run}"
GPU_ID="${GPU_ID:-0}"
RSPR_MODE="${RSPR_MODE:-off}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-hygiene}"

SNAPSHOT="${CKPT_DIR}/best_validation_checkpoints.json"
if [[ ! -f "${SNAPSHOT}" ]]; then
    echo "no best_validation_checkpoints.json in ${CKPT_DIR}" >&2
    exit 2
fi

DATA_PATH=/data2/hxj/data/MSRVTT
LOG_DIR="${ROOT_DIR}/logs/$(date +%Y%m%d)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(basename "${CKPT_DIR}")_final_test_$(date +%H%M%S).log"

echo "[final_test_from_ckpt] ckpt_dir=${CKPT_DIR}"
echo "[final_test_from_ckpt] gpu=${GPU_ID} log=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
CLIP_CACHE_DIR="${ROOT_DIR}/.cache" \
OMP_NUM_THREADS=1 \
/home/xujie/.conda/envs/tvr/bin/torchrun \
    --nproc_per_node=1 \
    --master_addr=127.0.0.9 \
    --master_port="${MASTER_PORT:-29553}" \
    "${ROOT_DIR}/main_task_retrieval.py" \
    --run_final_test \
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
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --experiment_profile "${EXPERIMENT_PROFILE}" \
    --experiment_desc "final test from the val selection in ${CKPT_DIR}" \
    --rspr_mode "${RSPR_MODE}" \
    2>&1 | tee "${LOG_FILE}"
