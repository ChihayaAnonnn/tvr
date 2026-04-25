#!/usr/bin/env bash
set -euo pipefail

# One-click MSRVTT ablation evals (most informative set):
# - caption-only: default, base_only, query_only
# - attributes: base_only, query_only, default
#
# Logs:
# - eval.sh already writes logs to logs/eval/<date>/*.log (tee).
#
# Usage:
#   bash scripts/eval_msrvtt_ablation.sh ckpts/ckpt_msrvtt_xxx/pytorch_model.bin.0
#
# Optional env overrides:
# - RUN_ID_BASE: customize run id prefix
# - CUDA_VISIBLE_DEVICES: which GPU to use for eval (default: 0)
# - MASTER_PORT: torchrun master port (default: 29519)
# - USE_ATTR: set 1 to enable attributes runs (default: 1)
# - USE_CAP: set 1 to enable caption-only runs (default: 1)
#
# Notes:
# - For MSRVTT + USE_ATTRIBUTES=1, eval.sh will auto-pick:
#   deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json(.jsonl)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT="${1:-${INIT_MODEL:-}}"
if [[ -z "${CKPT}" ]]; then
  echo "用法: bash scripts/eval_msrvtt_ablation.sh <checkpoint_path>" >&2
  echo "例如: bash scripts/eval_msrvtt_ablation.sh ckpts/ckpt_msrvtt_20260118_232754/pytorch_model.bin.0" >&2
  exit 2
fi

USE_ATTR="${USE_ATTR:-1}"
USE_CAP="${USE_CAP:-1}"

RUN_ID_BASE="${RUN_ID_BASE:-msrvtt_abl_$(date +%Y%m%d_%H%M%S)}"

echo "[eval_msrvtt_ablation] ROOT_DIR=${ROOT_DIR}"
echo "[eval_msrvtt_ablation] CKPT=${CKPT}"
echo "[eval_msrvtt_ablation] RUN_ID_BASE=${RUN_ID_BASE}"
echo "[eval_msrvtt_ablation] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MASTER_PORT=${MASTER_PORT:-29519}"

run_one () {
  local suffix="$1"
  local use_attrs="$2"
  local mode="$3"
  echo
  echo "============================================================"
  echo "[eval_msrvtt_ablation] RUN: ${RUN_ID_BASE}_${suffix}"
  echo "  USE_ATTRIBUTES=${use_attrs}  EVAL_BRANCH_MODE=${mode}"
  echo "============================================================"
  RUN_ID="${RUN_ID_BASE}_${suffix}" \
    DATATYPE=msrvtt \
    INIT_MODEL="${CKPT}" \
    USE_ATTRIBUTES="${use_attrs}" \
    EVAL_BRANCH_MODE="${mode}" \
    bash eval.sh
}

if [[ "${USE_CAP}" == "1" ]]; then
  run_one "cap_default" 0 default
  run_one "cap_baseonly" 0 base_only
fi

if [[ "${USE_ATTR}" == "1" ]]; then
  run_one "attr_baseonly" 1 base_only
  run_one "attr_queryonly" 1 query_only
  run_one "attr_default" 1 default
fi

if [[ "${USE_CAP}" == "1" ]]; then
  run_one "cap_queryonly" 0 query_only
fi

echo
echo "[eval_msrvtt_ablation] Done."
echo "[eval_msrvtt_ablation] 你可以在 logs/eval/<date>/ 下查看每次运行的日志文件。"

