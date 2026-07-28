#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# A0 baseline: RSPR_MODE=off (纯 DSA)，与 A3-fixed 严格可比的确定性对照。
# seed0 协议、hygiene profile、官方配方、5 epochs 全部与 A3-fixed 对齐。
RUN_ID="rspr_a0_seed0"
MONITOR_DIR="${ROOT_DIR}/logs/monitor"
MONITOR_LOG="${MONITOR_DIR}/${RUN_ID}.log"
LOCK_FILE="${MONITOR_DIR}/${RUN_ID}.lock"
OUTPUT_DIR="${ROOT_DIR}/ckpts/ckpt_msrvtt_${RUN_ID}"
POLL_SECONDS=60

mkdir -p "${MONITOR_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] watcher already running for ${RUN_ID}" >>"${MONITOR_LOG}"
    exit 0
fi

log() {
    echo "[$(date '+%F %T')] $*" >>"${MONITOR_LOG}"
}

gpu_compute_processes() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

log "watcher started; waiting for all GPU compute processes to exit"
while [[ -n "$(gpu_compute_processes)" ]]; do
    log "GPU busy: $(gpu_compute_processes | tr '\n' ';')"
    sleep "${POLL_SECONDS}"
done

log "all GPUs are free; launching ${RUN_ID}"
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=off \
RUN_ID="${RUN_ID}" \
./run_train_msrvtt_bg.sh >>"${MONITOR_LOG}" 2>&1 &
CONTROLLER_PID=$!
log "controller started: pid=${CONTROLLER_PID}"

TRAIN_STARTED=0
STARTUP_CHECKS=0
while kill -0 "${CONTROLLER_PID}" 2>/dev/null; do
    if pgrep -f "${ROOT_DIR}/main_task_retrieval.py.*--do_train" >/dev/null; then
        if [[ "${TRAIN_STARTED}" == "0" ]]; then
            TRAIN_STARTED=1
            log "training process detected"
        fi
    elif [[ "${TRAIN_STARTED}" == "1" ]]; then
        break
    else
        STARTUP_CHECKS=$((STARTUP_CHECKS + 1))
        if (( STARTUP_CHECKS >= 30 )); then
            log "training process was not detected within 30 minutes"
            break
        fi
    fi
    sleep "${POLL_SECONDS}"
done

if kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
    kill "${CONTROLLER_PID}" 2>/dev/null || true
fi

if [[ -f "${OUTPUT_DIR}/final_test.json" ]]; then
    log "A0 completed successfully: ${OUTPUT_DIR}/final_test.json"
    exit 0
fi

LATEST_LOG="$(find "${ROOT_DIR}/logs" -maxdepth 2 -type f \
    -name "${RUN_ID}_*_train_msrvtt.log" -print0 |
    xargs -0 -r ls -1t | head -n 1)"
log "A0 stopped without final_test.json; inspect ${LATEST_LOG:-no training log found}"
exit 1
