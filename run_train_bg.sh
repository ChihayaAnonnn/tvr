#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="logs/train_${RUN_ID}.log"
echo "[run_train_bg] RUN_ID=${RUN_ID}"
echo "[run_train_bg] LOG_FILE=${LOG_FILE}"
echo "[DEPRECATED] run_train_bg.sh 已弃用，请改用: bash run_train_msrvtt_bg.sh"
echo "[run_train_bg] Starting: bash run_train_msrvtt_bg.sh (completely detached)"

# Deprecated wrapper: forward to the new canonical script.
exec bash run_train_msrvtt_bg.sh
