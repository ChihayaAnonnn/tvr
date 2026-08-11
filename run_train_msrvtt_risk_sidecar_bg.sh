#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CKPT="${BASE_CKPT:-${ROOT_DIR}/ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4}"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi

RUN_TAG="${RUN_TAG:-risk_sidecar_a1}" \
EXPERIMENT_DESC="${EXPERIMENT_DESC:-A1 frozen June retrieval-risk sidecar}" \
bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" \
    --use_uatvr_full \
    --use_ada_norm \
    --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar \
    --retrieval_risk_sidecar_only \
    --w_retrieval_risk 0.05 \
    --retrieval_risk_temperature 1.0 \
    --retrieval_risk_hidden_dim 32 \
    --retrieval_risk_bootstrap_samples 1000 \
    --DSL True \
    --dsl_temperature 2.0 \
    "$@"
