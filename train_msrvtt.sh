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
TQFS_CACHE_DIR=${TQFS_CACHE_DIR:-/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224}

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
A800_THROUGHPUT_COMPARISON=${A800_THROUGHPUT_COMPARISON:-0}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
if [[ "${EXPERIMENT_PROFILE}" != "default" && "${EXPERIMENT_PROFILE}" != "hygiene" && "${EXPERIMENT_PROFILE}" != "pair_evidence_refiner" ]]; then
    echo "Unsupported EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}; expected default, hygiene, or pair_evidence_refiner" >&2
    exit 2
fi
if [[ "${EXPERIMENT_PROFILE}" == "hygiene" ]]; then
    _PROTECTED_P0_OPTIONS=(
        --batch_size
        --gradient_accumulation_steps
        --experiment_profile
        --eval_split
        --datatype
        --expand_msrvtt_sentences
        --backbone_type
        --pretrained_clip_name
        --clip_layer_norm_precision
        --clip_gradient_checkpointing
        --clip_visual_checkpoint_layers
        --tqfs_cache_dir
        --train_csv
        --val_csv
        --source_train_csv
        --test_csv
        --split_manifest
        --data_path
        --features_path
    )
    for _ARG in "$@"; do
        _FLAG="${_ARG%%=*}"
        for _PROTECTED in "${_PROTECTED_P0_OPTIONS[@]}"; do
            if [[ "${_FLAG}" == "${_PROTECTED}" || "${_PROTECTED}" == "${_FLAG}"* ]]; then
                echo "hygiene P0 cannot override protected P0 option ${_FLAG} via trailing arguments" >&2
                exit 2
            fi
        done
    done
fi
if [[ "${EXPERIMENT_PROFILE}" == "pair_evidence_refiner" ]]; then
    if [[ "${TRAIN_BATCH_SIZE}" != "256" ]]; then
        echo "pair_evidence_refiner requires TRAIN_BATCH_SIZE=256; got ${TRAIN_BATCH_SIZE}" >&2
        exit 2
    fi
    if [[ "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
        echo "pair_evidence_refiner requires TRAIN_GRADIENT_ACCUMULATION_STEPS=1; got ${TRAIN_GRADIENT_ACCUMULATION_STEPS}" >&2
        exit 2
    fi
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
if [[ "${A800_THROUGHPUT_COMPARISON}" != "0" && "${A800_THROUGHPUT_COMPARISON}" != "1" ]]; then
    echo "Unsupported A800_THROUGHPUT_COMPARISON=${A800_THROUGHPUT_COMPARISON}; expected 0 or 1" >&2
    exit 2
fi
if ! [[ "${CLIP_VISUAL_CHECKPOINT_LAYERS}" =~ ^[0-9]+$ ]]; then
    echo "Unsupported CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS}; expected a non-negative integer" >&2
    exit 2
fi
if ! [[ "${TRAIN_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "Unsupported TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS}; expected a non-negative integer" >&2
    exit 2
fi
if ! [[ "${TRAIN_PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unsupported TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR}; expected a positive integer" >&2
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

# P0 固定 batch 256 + accum 1，有效 batch = 256；4 卡时每卡 micro-batch 64。
# 0/1 位于 NUMA 0，2/4 位于 NUMA 1，且 2/4 之间为 NV8。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
if ! [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "malformed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; expected comma-separated integer GPU IDs" >&2
    exit 2
fi
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#_GPUS[@]}}"

if [[ "${EXPERIMENT_PROFILE}" == "hygiene" ]]; then
    if [[ "${#_GPUS[@]}" -ne 4 ]]; then
        echo "hygiene P0 requires exactly 4 GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
    declare -A _SEEN_GPUS=()
    for _GPU in "${_GPUS[@]}"; do
        if [[ -n "${_SEEN_GPUS[${_GPU}]:-}" ]]; then
            echo "hygiene P0 rejects duplicate GPU IDs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
            exit 2
        fi
        _SEEN_GPUS["${_GPU}"]=1
    done
    if [[ "${NPROC}" != "${#_GPUS[@]}" ]]; then
        echo "NPROC=${NPROC} does not match ${#_GPUS[@]} visible GPUs" >&2
        exit 2
    fi
    if [[ "${TRAIN_BATCH_SIZE}" != "256" ]]; then
        echo "hygiene P0 requires TRAIN_BATCH_SIZE=256; got ${TRAIN_BATCH_SIZE}" >&2
        exit 2
    fi
    if [[ "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
        echo "hygiene P0 requires TRAIN_GRADIENT_ACCUMULATION_STEPS=1; got ${TRAIN_GRADIENT_ACCUMULATION_STEPS}" >&2
        exit 2
    fi
fi

if [[ "${CLIP_GRADIENT_CHECKPOINTING}" == "0" ]]; then
    if [[ "${A800_THROUGHPUT_COMPARISON}" != "1" ]]; then
        echo "checkpoint-off requires A800_THROUGHPUT_COMPARISON=1" >&2
        exit 2
    fi
    if [[ "${RUN_ID}" != a800_no_ckpt_* ]]; then
        echo "checkpoint-off RUN_ID must start with a800_no_ckpt_; got ${RUN_ID}" >&2
        exit 2
    fi
    if [[ "${OUTPUT_DIR}" != *"${RUN_ID}"* ]]; then
        echo "checkpoint-off OUTPUT_DIR must contain RUN_ID=${RUN_ID}; got ${OUTPUT_DIR}" >&2
        exit 2
    fi
    echo "[train_msrvtt.sh] A800 throughput comparison: activation checkpointing disabled"
fi
echo "[train_msrvtt.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC} TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS} TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS} TQFS_CACHE_DIR=${TQFS_CACHE_DIR}"
echo "[train_msrvtt.sh] BACKBONE_TYPE=${BACKBONE_TYPE} BACKBONE_NAME=${BACKBONE_NAME} BACKBONE_PATH=${BACKBONE_PATH} EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN} CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION} CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING} CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    torchrun --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29547 \
    "${ROOT_DIR}/main_task_retrieval.py" \
    --do_train --num_thread_reader "${TRAIN_NUM_WORKERS}" \
    --prefetch_factor "${TRAIN_PREFETCH_FACTOR}" --epochs=5 \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
    --n_display=20 \
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
    --pair_refiner_num_views 4 \
    --pair_refiner_lambda_max 0.1 \
    --pair_refiner_query_block_size 16 \
    --pair_refiner_candidate_block_size 32 \
    --pair_refiner_alignment_temperature 0.07 \
    --experiment_desc "${EXPERIMENT_DESC:-}" \
    "$@"
