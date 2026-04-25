#!/usr/bin/env bash
set -euo pipefail

# Run MSRVTT eval under 3 branch modes:
#   default / base_only / query_only
#
# Usage:
#   INIT_MODEL=ckpts/ckpt_msrvtt_xxx/pytorch_model.bin.1 \
#   USE_ATTRIBUTES=1 \
#   ATTR_PATH="/abs/path/train9k.json,/abs/path/test1k.json" \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash eval_msrvtt_ablation.sh
#
# Notes:
# - ATTR_PATH supports comma-separated files (merged by dataloader).
# - Each mode will use a different RUN_ID and MASTER_PORT to avoid conflicts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATATYPE=${DATATYPE:-msrvtt}
BASE_RUN_ID=${RUN_ID:-"msrvtt_abl_$(date +%Y%m%d_%H%M%S)"}

MODES=("default" "base_only" "query_only")

for i in "${!MODES[@]}"; do
  MODE="${MODES[$i]}"
  # Use different port per run (avoid collision if previous process lingers)
  MASTER_PORT=${MASTER_PORT:-29520}
  PORT=$((MASTER_PORT + i))

  echo "[eval_msrvtt_ablation] MODE=${MODE} RUN_ID=${BASE_RUN_ID}_${MODE} MASTER_PORT=${PORT}"

  RUN_ID="${BASE_RUN_ID}_${MODE}" \
  DATATYPE="${DATATYPE}" \
  EVAL_BRANCH_MODE="${MODE}" \
  MASTER_PORT="${PORT}" \
  USE_ATTRIBUTES="${USE_ATTRIBUTES:-1}" \
  ATTR_PATH="${ATTR_PATH:-/data2/hxj/project/UATVR/deploy_qwen/attributes/msrvtt/final/msrvtt_train9k_attributes.json,/data2/hxj/project/UATVR/deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json}" \
  MAX_WORDS_ATTRS="${MAX_WORDS_ATTRS:-77}" \
  ENHANCED_FUSION="${ENHANCED_FUSION:-1}" \
  USE_ATTR_ADAPTER="${USE_ATTR_ADAPTER:-0}" \
  ATTR_ADAPTER_RANK="${ATTR_ADAPTER_RANK:-64}" \
  bash eval.sh
done

