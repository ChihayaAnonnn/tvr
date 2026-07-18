#!/usr/bin/env bash
set -euo pipefail

# Run MSRVTT training in background, log to logs/, then tail the log.
# Ctrl-C will stop tailing, but training will continue in background.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
RUN_TAG="${RUN_TAG:-}"
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsupported RUN_TAG=${RUN_TAG}; use letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi
RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_SUFFIX}}"
LOG_DIR="logs/${RUN_DATE}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_SUFFIX}_train_msrvtt.log"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-}"

echo "[run_train_msrvtt_bg] RUN_DATE=${RUN_DATE} RUN_TIME=${RUN_TIME} RUN_TAG=${RUN_TAG}"
echo "[run_train_msrvtt_bg] LOG_FILE=${LOG_FILE}"
echo "[run_train_msrvtt_bg] Starting: bash train_msrvtt.sh (completely detached)"

# 透传受支持参数给 train_msrvtt.sh；退役参数由 argparse 明确拒绝。
setsid env RUN_ID="${RUN_ID}" bash train_msrvtt.sh "$@" >"${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
if [[ -n "${TRAIN_PID_FILE}" ]]; then
  echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
fi
echo "[run_train_msrvtt_bg] PID=${TRAIN_PID}"
echo "[run_train_msrvtt_bg] MSRVTT 训练已在后台启动。你可以安全关闭 Cursor。"
echo "[run_train_msrvtt_bg] 随时可以运行以下命令查看日志："
echo "tail -f ${LOG_FILE}"

# 启动后立即查看前 50 行日志确认启动成功
tail -n 50 -F "${LOG_FILE}"
