#!/usr/bin/env bash
set -euo pipefail

# 抑制 DDP 多卡重复警告（Grad strides do not match 等）
export TORCH_WARN_ONCE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH=${DATA_PATH:-/data2/hxj/data/MSRVTT}
SOURCE_TRAIN_CSV="${DATA_PATH}/csv/MSRVTT_train.9k.csv"
TEST_CSV="${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
ANNOTATION_JSON="${DATA_PATH}/annotation/MSRVTT_v2.json"
SPLIT_MANIFEST="${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed42.json"
GENERATED_SPLIT_DIR="${ROOT_DIR}/data/generated/msrvtt_trusted_v1"
TQFS_CACHE_DIR=${TQFS_CACHE_DIR:-${ROOT_DIR}/cache_dir/tqfs/msrvtt_trusted_v1_f1_m8_r224}

python3 "${ROOT_DIR}/scripts/build_msrvtt_trusted_split.py" \
    --train-csv "${SOURCE_TRAIN_CSV}" \
    --annotation-json "${ANNOTATION_JSON}" \
    --test-csv "${TEST_CSV}" \
    --manifest "${SPLIT_MANIFEST}" \
    --output-dir "${GENERATED_SPLIT_DIR}"

# Auto-run id to avoid overwriting checkpoints/logs across runs
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-ckpts/ckpt_msrvtt_${RUN_ID}}
EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-hygiene}
BACKBONE_TYPE=${BACKBONE_TYPE:-openai_clip}
BACKBONE_NAME=${BACKBONE_NAME:-EVA02-CLIP-B-16}
BACKBONE_PATH=${BACKBONE_PATH:-${ROOT_DIR}/research_refs/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt}
EVA_CLIP_ROOT=${EVA_CLIP_ROOT:-${ROOT_DIR}/research_refs/EVA/EVA-CLIP/rei}
EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN:-0}
CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION:-fp16}
CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING:-1}
CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS:-4}
if [[ "${EXPERIMENT_PROFILE}" != "default" && "${EXPERIMENT_PROFILE}" != "hygiene" ]]; then
    echo "Unsupported EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}; expected default or hygiene" >&2
    exit 2
fi
if [[ "${EVA_CLIP_USE_XATTN}" != "0" && "${EVA_CLIP_USE_XATTN}" != "1" ]]; then
    echo "Unsupported EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN}; expected 0 or 1" >&2
    exit 2
fi
if [[ "${CLIP_LAYER_NORM_PRECISION}" != "fp16" && "${CLIP_LAYER_NORM_PRECISION}" != "fp32" ]]; then
    echo "Unsupported CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION}; expected fp16 or fp32" >&2
    exit 2
fi
if [[ "${CLIP_GRADIENT_CHECKPOINTING}" != "0" && "${CLIP_GRADIENT_CHECKPOINTING}" != "1" ]]; then
    echo "Unsupported CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING}; expected 0 or 1" >&2
    exit 2
fi
if ! [[ "${CLIP_VISUAL_CHECKPOINT_LAYERS}" =~ ^[0-9]+$ ]]; then
    echo "Unsupported CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS}; expected a non-negative integer" >&2
    exit 2
fi
EXTRA_BACKBONE_ARGS=()
if [[ "${EVA_CLIP_USE_XATTN}" == "1" ]]; then
    EXTRA_BACKBONE_ARGS+=(--eva_clip_use_xattn)
fi
if [[ "${CLIP_GRADIENT_CHECKPOINTING}" == "1" ]]; then
    EXTRA_BACKBONE_ARGS+=(
        --clip_gradient_checkpointing
        --clip_visual_checkpoint_layers "${CLIP_VISUAL_CHECKPOINT_LAYERS}"
    )
fi

# 显存：batch 256 + accum 1，有效 batch = 256。
# 2 卡时每卡 micro-batch 128；4 卡时每卡 micro-batch 64。
# 启动前请确认所选 GPU 无其他大进程（nvidia-smi）。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#_GPUS[@]}}"

echo "[train_msrvtt.sh] BACKBONE_TYPE=${BACKBONE_TYPE} BACKBONE_NAME=${BACKBONE_NAME} BACKBONE_PATH=${BACKBONE_PATH} EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN} CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION} CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING} CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    torchrun --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29547 \
    "${ROOT_DIR}/main_task_retrieval.py" \
    --do_train --num_thread_reader=8 --epochs=5 \
    --batch_size=256 --gradient_accumulation_steps=1 --n_display=20 \
    --train_csv "${GENERATED_SPLIT_DIR}/train.csv" \
    --val_csv "${GENERATED_SPLIT_DIR}/val.csv" \
    --source_train_csv "${SOURCE_TRAIN_CSV}" \
    --test_csv "${TEST_CSV}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --eval_split val \
    --data_path "${ANNOTATION_JSON}" \
    --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
    --tqfs_cache_dir "${TQFS_CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 16 \
    --datatype msrvtt --expand_msrvtt_sentences \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 3 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --backbone_type "${BACKBONE_TYPE}" \
    --clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}" \
    --backbone_name "${BACKBONE_NAME}" \
    --backbone_path "${BACKBONE_PATH}" \
    --eva_clip_root "${EVA_CLIP_ROOT}" \
    "${EXTRA_BACKBONE_ARGS[@]}" \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --experiment_profile "${EXPERIMENT_PROFILE}" \
    --experiment_desc "${EXPERIMENT_DESC:-}" \
    "$@"
