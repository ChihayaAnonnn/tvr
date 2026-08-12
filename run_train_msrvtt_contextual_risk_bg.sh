#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "${ROOT_DIR}" rev-parse --path-format=absolute --git-common-dir)"
COMMON_REPO_ROOT="$(cd "${COMMON_GIT_DIR}/.." && pwd)"
BASE_CKPT="${COMMON_REPO_ROOT}/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing canonical A1 checkpoint BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
RUN_TAG="${RUN_TAG:-contextual_risk_a3}"
RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ckpts/ckpt_msrvtt_${RUN_ID}}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "A3 output directory already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi

RUN_DATE="${RUN_DATE}" \
RUN_TIME="${RUN_TIME}" \
RUN_TAG="${RUN_TAG}" \
RUN_ID="${RUN_ID}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}" \
NPROC=4 \
TRAIN_BATCH_SIZE=256 \
TRAIN_GRADIENT_ACCUMULATION_STEPS=1 \
CLIP_LAYER_NORM_PRECISION=fp32 \
EXPERIMENT_PROFILE=hygiene \
EXPERIMENT_DESC="${EXPERIMENT_DESC:-A3 contextual retrieval-risk sidecar}" \
bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" \
    --use_uatvr_full \
    --use_ada_norm \
    --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar \
    --w_retrieval_risk 0.0 \
    --use_contextual_risk_sidecar \
    --contextual_risk_sidecar_only \
    --contextual_risk_topk 16 \
    --contextual_risk_projection_dim 128 \
    --contextual_risk_hidden_dim 128 \
    --contextual_risk_temperature 1.0 \
    --w_contextual_risk_fit 0.25 \
    --w_contextual_risk_rank 0.25 \
    --seed 0 \
    --epochs 3 \
    --run_final_test \
    "$@"
