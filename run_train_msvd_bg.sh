#!/usr/bin/env bash
set -euo pipefail

# Run MSVD training in background, log to logs/, then tail the log.
# Ctrl-C will stop tailing, but training will continue in background.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_TIME}}"
LOG_DIR="logs/${RUN_DATE}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_TIME}_train_msvd.log"

echo "[run_train_msvd_bg] RUN_DATE=${RUN_DATE} RUN_TIME=${RUN_TIME}"
echo "[run_train_msvd_bg] LOG_FILE=${LOG_FILE}"
echo "[run_train_msvd_bg] Starting: bash train_msvd.sh (completely detached)"

# 使用 setsid 开启新会话，确保不受终端关闭影响，并指定执行 train_msvd.sh
setsid env RUN_ID="${RUN_ID}" bash train_msvd.sh >"${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "[run_train_msvd_bg] PID=${TRAIN_PID}"
echo "[run_train_msvd_bg] MSVD 训练已在后台启动。你可以安全关闭 Cursor。"
echo "[run_train_msvd_bg] 随时可以运行以下命令查看日志："
echo "tail -f ${LOG_FILE}"

# 启动后立即查看前 50 行日志确认启动成功
tail -n 50 -F "${LOG_FILE}"
