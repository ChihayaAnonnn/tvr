#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# A1 baseline: RSPR_MODE=mean, K=1 (概率均值-only，无随机噪声、无 rank loss)。
# 用于分离"参数量增加"与"随机采样+BCE"两种收益来源。
# 其余超参与 A3-fixed 严格对齐 (λ_a=1e-2, prob_loss=soft_bce, seed0, hygiene)。
# 该 watcher 等 A0 完成后再启动，避免 GPU 争用。
RUN_ID="rspr_a1_seed0"
MONITOR_DIR="${ROOT_DIR}/logs/monitor"
MONITOR_LOG="${MONITOR_DIR}/${RUN_ID}.log"
LOCK_FILE="${MONITOR_DIR}/${RUN_ID}.lock"
OUTPUT_DIR="${ROOT_DIR}/ckpts/ckpt_msrvtt_${RUN_ID}"
A0_FINAL="${ROOT_DIR}/ckpts/ckpt_msrvtt_rspr_a0_seed0/final_test.json"
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

log "watcher started; blocking on A0 completion (${A0_FINAL})"
while [[ ! -f "${A0_FINAL}" ]]; do
    sleep "${POLL_SECONDS}"
done
log "A0 completed; waiting for GPU to be idle"

while [[ -n "$(gpu_compute_processes)" ]]; do
    log "GPU busy: $(gpu_compute_processes | tr '\n' ';')"
    sleep "${POLL_SECONDS}"
done

log "all GPUs are free; launching ${RUN_ID}"
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=mean \
RSPR_SAMPLE_COUNT=1 \
RSPR_PROB_LOSS=soft_bce \
RSPR_ANCHOR_WEIGHT=1e-2 \
RSPR_GRAD_DIAGNOSTICS=1 \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
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
    log "A1 completed successfully: ${OUTPUT_DIR}/final_test.json"
    exit 0
fi

LATEST_LOG="$(find "${ROOT_DIR}/logs" -maxdepth 2 -type f \
    -name "${RUN_ID}_*_train_msrvtt.log" -print0 |
    xargs -0 -r ls -1t | head -n 1)"
log "A1 stopped without final_test.json; inspect ${LATEST_LOG:-no training log found}"
exit 1
