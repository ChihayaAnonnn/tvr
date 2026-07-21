#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_train_msrvtt_bg.sh"
cd "${ROOT_DIR}"

run_controller() {
    mkdir -p logs

    RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
    RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
    RUN_TAG="${RUN_TAG:-}"
    if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsupported RUN_TAG=${RUN_TAG}; use letters, digits, dot, underscore, or hyphen" >&2
        return 2
    fi
    RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
    RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_SUFFIX}}"
    LOG_DIR="logs/${RUN_DATE}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${RUN_SUFFIX}_train_msrvtt.log"
    TRAIN_PID_FILE="${TRAIN_PID_FILE:-}"

    echo "[run_train_msrvtt_bg] RUN_DATE=${RUN_DATE} RUN_TIME=${RUN_TIME} RUN_TAG=${RUN_TAG}"
    echo "[run_train_msrvtt_bg] LOG_FILE=${LOG_FILE}"
    echo "[run_train_msrvtt_bg] Starting internal training worker (completely detached)"

    setsid env \
        RUN_ID="${RUN_ID}" \
        RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1 \
        bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 &

    TRAIN_PID=$!
    if [[ -n "${TRAIN_PID_FILE}" ]]; then
        echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
    fi
    echo "[run_train_msrvtt_bg] PID=${TRAIN_PID}"
    echo "[run_train_msrvtt_bg] MSRVTT 训练已在后台启动。你可以安全关闭 Cursor。"
    echo "[run_train_msrvtt_bg] 随时可以运行以下命令查看日志："
    echo "tail -f ${LOG_FILE}"

    tail -n 50 -F "${LOG_FILE}"
}


run_worker() {
    unset RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER

    # 抑制 DDP 多卡重复警告（Grad strides do not match 等）
    export TORCH_WARN_ONCE=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

    DATA_PATH=${DATA_PATH:-/data2/hxj/data/MSRVTT}
    SOURCE_TRAIN_CSV="${DATA_PATH}/csv/MSRVTT_train.9k.csv"
    TEST_CSV="${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
    ANNOTATION_JSON="${DATA_PATH}/annotation/MSRVTT_v2.json"
    SPLIT_MANIFEST="${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    GENERATED_SPLIT_DIR="${ROOT_DIR}/data/generated/msrvtt_trusted_v1"
    TQFS_CACHE_DIR=${TQFS_CACHE_DIR:-/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224}
    export CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-${ROOT_DIR}/.cache}

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
    CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION:-fp16}
    CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING:-1}
    CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS:-4}
    A800_THROUGHPUT_COMPARISON=${A800_THROUGHPUT_COMPARISON:-0}
    TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
    TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-2}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
    TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
    FREEZE_LAYER_NUM=${FREEZE_LAYER_NUM:-0}
    RSPR_MODE=${RSPR_MODE:-legacy}
    RSPR_SAMPLE_COUNT=${RSPR_SAMPLE_COUNT:-4}
    RSPR_EVAL_SAMPLE_COUNT=${RSPR_EVAL_SAMPLE_COUNT:-8}
    RSPR_MATCH_MODE=${RSPR_MATCH_MODE:-soft}
    RSPR_DETACH_SAMPLES=${RSPR_DETACH_SAMPLES:-0}
    RSPR_MATCH_TEMPERATURE=${RSPR_MATCH_TEMPERATURE:-0.07}
    RSPR_PROB_TEMPERATURE=${RSPR_PROB_TEMPERATURE:-0.07}
    RSPR_RANK_TEMPERATURE=${RSPR_RANK_TEMPERATURE:-0.07}
    RSPR_HARD_NEGATIVES=${RSPR_HARD_NEGATIVES:-8}
    RSPR_PRIOR_STD=${RSPR_PRIOR_STD:-0.1}
    RSPR_PROB_WEIGHT=${RSPR_PROB_WEIGHT:-0.1}
    RSPR_RANK_WEIGHT=${RSPR_RANK_WEIGHT:-0.1}
    RSPR_ANCHOR_WEIGHT=${RSPR_ANCHOR_WEIGHT:-1e-4}
    RSPR_WARMUP_EPOCHS=${RSPR_WARMUP_EPOCHS:-1.0}
    RSPR_EVAL_SEED=${RSPR_EVAL_SEED:-0}
    RSPR_TOP_R=${RSPR_TOP_R:-100}
    RSPR_DET_TEMPERATURE=${RSPR_DET_TEMPERATURE:-1.0}
    RSPR_RERANK_TEMPERATURE=${RSPR_RERANK_TEMPERATURE:-1.0}
    RSPR_RERANK_WEIGHT=${RSPR_RERANK_WEIGHT:-0.1}
    RSPR_PAIR_CHUNK_SIZE=${RSPR_PAIR_CHUNK_SIZE:-4096}
    RSPR_FREEZE_CLIP=${RSPR_FREEZE_CLIP:-0}
    RSPR_FREEZE_DSA=${RSPR_FREEZE_DSA:-0}
    if [[ "${EXPERIMENT_PROFILE}" != "default" && "${EXPERIMENT_PROFILE}" != "hygiene" ]]; then
        echo "Unsupported EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}; expected default or hygiene" >&2
        exit 2
    fi
    if [[ "${EXPERIMENT_PROFILE}" == "hygiene" ]]; then
        _PROTECTED_HYGIENE_OPTIONS=(
            --batch_size
            --gradient_accumulation_steps
            --experiment_profile
            --eval_split
            --datatype
            --expand_msrvtt_sentences
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
            for _PROTECTED in "${_PROTECTED_HYGIENE_OPTIONS[@]}"; do
                if [[ "${_FLAG}" == "${_PROTECTED}" || "${_PROTECTED}" == "${_FLAG}"* ]]; then
                    echo "hygiene cannot override protected baseline option ${_FLAG} via trailing arguments" >&2
                    exit 2
                fi
            done
        done
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
    if [[ "${RSPR_MODE}" != "legacy" && "${RSPR_MODE}" != "off" && "${RSPR_MODE}" != "mean" && "${RSPR_MODE}" != "stochastic" ]]; then
        echo "Unsupported RSPR_MODE=${RSPR_MODE}; expected legacy, off, mean, or stochastic" >&2
        exit 2
    fi
    if [[ "${RSPR_MATCH_MODE}" != "soft" && "${RSPR_MATCH_MODE}" != "hard" ]]; then
        echo "Unsupported RSPR_MATCH_MODE=${RSPR_MATCH_MODE}; expected soft or hard" >&2
        exit 2
    fi
    for _RSPR_BOOLEAN in RSPR_DETACH_SAMPLES RSPR_FREEZE_CLIP RSPR_FREEZE_DSA; do
        if [[ "${!_RSPR_BOOLEAN}" != "0" && "${!_RSPR_BOOLEAN}" != "1" ]]; then
            echo "Unsupported ${_RSPR_BOOLEAN}=${!_RSPR_BOOLEAN}; expected 0 or 1" >&2
            exit 2
        fi
    done
    if [[ "${RSPR_MODE}" == "mean" && "${RSPR_SAMPLE_COUNT}" != "1" ]]; then
        echo "RSPR_MODE=mean requires RSPR_SAMPLE_COUNT=1" >&2
        exit 2
    fi
    if [[ "${RSPR_MODE}" == "stochastic" ]]; then
        for _RSPR_K in RSPR_SAMPLE_COUNT RSPR_EVAL_SAMPLE_COUNT; do
            if ! [[ "${!_RSPR_K}" =~ ^[1-9][0-9]*$ ]] || (( ${!_RSPR_K} % 2 )); then
                echo "Unsupported ${_RSPR_K}=${!_RSPR_K}; stochastic mode requires a positive even integer" >&2
                exit 2
            fi
        done
    fi
    for _RSPR_TEMPERATURE in RSPR_MATCH_TEMPERATURE RSPR_PROB_TEMPERATURE RSPR_RANK_TEMPERATURE RSPR_DET_TEMPERATURE RSPR_RERANK_TEMPERATURE; do
        if ! [[ "${!_RSPR_TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || ! awk -v value="${!_RSPR_TEMPERATURE}" 'BEGIN { exit !(value > 0 && value < 1e308) }'; then
            echo "Unsupported ${_RSPR_TEMPERATURE}=${!_RSPR_TEMPERATURE}; expected a positive finite number" >&2
            exit 2
        fi
    done
    if ! [[ "${RSPR_TOP_R}" =~ ^[0-9]+$ ]]; then
        echo "Unsupported RSPR_TOP_R=${RSPR_TOP_R}; expected a non-negative integer" >&2
        exit 2
    fi
    EXTRA_CLIP_ARGS=()
    if [[ "${CLIP_GRADIENT_CHECKPOINTING}" == "1" ]]; then
        EXTRA_CLIP_ARGS+=(
            --clip_gradient_checkpointing
            --clip_visual_checkpoint_layers "${CLIP_VISUAL_CHECKPOINT_LAYERS}"
        )
    fi
    RSPR_OPTIONAL_ARGS=()
    if [[ "${RSPR_DETACH_SAMPLES}" == "1" ]]; then
        RSPR_OPTIONAL_ARGS+=(--rspr_detach_samples)
    fi
    if [[ "${RSPR_FREEZE_CLIP}" == "1" ]]; then
        RSPR_OPTIONAL_ARGS+=(--rspr_freeze_clip)
    fi
    if [[ "${RSPR_FREEZE_DSA}" == "1" ]]; then
        RSPR_OPTIONAL_ARGS+=(--rspr_freeze_dsa)
    fi

    # hygiene baseline 固定 batch 256 + accum 1，有效 batch = 256；4 卡时每卡 micro-batch 64。
    # 0/1 位于 NUMA 0，2/4 位于 NUMA 1，且 2/4 之间为 NV8。
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
    if ! [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "malformed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; expected comma-separated integer GPU IDs" >&2
        exit 2
    fi
    IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${NPROC:-${#_GPUS[@]}}"

    if [[ "${EXPERIMENT_PROFILE}" == "hygiene" ]]; then
        _BATCH_PROFILE_LABEL="hygiene baseline"
        if [[ "${#_GPUS[@]}" -ne 4 ]]; then
            echo "${_BATCH_PROFILE_LABEL} requires exactly 4 GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
            exit 2
        fi
        declare -A _SEEN_GPUS=()
        for _GPU in "${_GPUS[@]}"; do
            if [[ -n "${_SEEN_GPUS[${_GPU}]:-}" ]]; then
                echo "${_BATCH_PROFILE_LABEL} rejects duplicate GPU IDs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
                exit 2
            fi
            _SEEN_GPUS["${_GPU}"]=1
        done
        if [[ "${NPROC}" != "${#_GPUS[@]}" ]]; then
            echo "NPROC=${NPROC} does not match ${#_GPUS[@]} visible GPUs" >&2
            exit 2
        fi
        if [[ "${TRAIN_BATCH_SIZE}" != "256" ]]; then
            echo "${_BATCH_PROFILE_LABEL} requires TRAIN_BATCH_SIZE=256; got ${TRAIN_BATCH_SIZE}" >&2
            exit 2
        fi
        if [[ "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
            echo "${_BATCH_PROFILE_LABEL} requires TRAIN_GRADIENT_ACCUMULATION_STEPS=1; got ${TRAIN_GRADIENT_ACCUMULATION_STEPS}" >&2
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
        echo "[run_train_msrvtt_bg:worker] A800 throughput comparison: activation checkpointing disabled"
    fi
    echo "[run_train_msrvtt_bg:worker] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC} TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS} TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS} TQFS_CACHE_DIR=${TQFS_CACHE_DIR} CLIP_CACHE_DIR=${CLIP_CACHE_DIR}"
    echo "[run_train_msrvtt_bg:worker] PRETRAINED_CLIP_NAME=ViT-B/16 CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION} CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING} CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS}"
    echo "[run_train_msrvtt_bg:worker] RSPR_MODE=${RSPR_MODE} RSPR_K=${RSPR_SAMPLE_COUNT} RSPR_EVAL_K=${RSPR_EVAL_SAMPLE_COUNT} RSPR_PROB_WEIGHT=${RSPR_PROB_WEIGHT} RSPR_RANK_WEIGHT=${RSPR_RANK_WEIGHT} RSPR_ANCHOR_WEIGHT=${RSPR_ANCHOR_WEIGHT} RSPR_FREEZE_CLIP=${RSPR_FREEZE_CLIP} RSPR_FREEZE_DSA=${RSPR_FREEZE_DSA} RSPR_TOP_R=${RSPR_TOP_R} RSPR_EVAL_SEED=${RSPR_EVAL_SEED}"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        torchrun --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29547 \
        "${ROOT_DIR}/main_task_retrieval.py" \
        --do_train --run_final_test --num_thread_reader "${TRAIN_NUM_WORKERS}" \
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
        --freeze_layer_num "${FREEZE_LAYER_NUM}" --slice_framepos 3 \
        --linear_patch 2d --sim_header seqTransf \
        --pretrained_clip_name ViT-B/16 \
        --clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}" \
        "${EXTRA_CLIP_ARGS[@]}" \
        --extra_video_cls_num 2 \
        --extra_text_cls_num 2 \
        --experiment_profile "${EXPERIMENT_PROFILE}" \
        --experiment_desc "${EXPERIMENT_DESC:-}" \
        --rspr_mode "${RSPR_MODE}" \
        --rspr_sample_count "${RSPR_SAMPLE_COUNT}" \
        --rspr_eval_sample_count "${RSPR_EVAL_SAMPLE_COUNT}" \
        --rspr_match_mode "${RSPR_MATCH_MODE}" \
        --rspr_match_temperature "${RSPR_MATCH_TEMPERATURE}" \
        --rspr_prob_temperature "${RSPR_PROB_TEMPERATURE}" \
        --rspr_rank_temperature "${RSPR_RANK_TEMPERATURE}" \
        --rspr_hard_negatives "${RSPR_HARD_NEGATIVES}" \
        --rspr_prior_std "${RSPR_PRIOR_STD}" \
        --rspr_prob_weight "${RSPR_PROB_WEIGHT}" \
        --rspr_rank_weight "${RSPR_RANK_WEIGHT}" \
        --rspr_anchor_weight "${RSPR_ANCHOR_WEIGHT}" \
        --rspr_warmup_epochs "${RSPR_WARMUP_EPOCHS}" \
        --rspr_eval_seed "${RSPR_EVAL_SEED}" \
        --rspr_top_r "${RSPR_TOP_R}" \
        --rspr_det_temperature "${RSPR_DET_TEMPERATURE}" \
        --rspr_rerank_temperature "${RSPR_RERANK_TEMPERATURE}" \
        --rspr_rerank_weight "${RSPR_RERANK_WEIGHT}" \
        --rspr_pair_chunk_size "${RSPR_PAIR_CHUNK_SIZE}" \
        "${RSPR_OPTIONAL_ARGS[@]}" \
        "$@"
}


if [[ "${RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER:-0}" == "1" ]]; then
    run_worker "$@"
else
    run_controller "$@"
fi
