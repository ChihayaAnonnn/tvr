#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# A3fix-v3: 只改 λ (x3),COEF_LR 保持默认 1e-3,以拆开 v2 里的混淆。
# v2 同时动了 λ 和 coef_lr,结果无法归因:
#   - A0 vs A0-v2 证明 coef_lr 1e-2 单独就让验证 T2V R@1 每个 epoch 掉约 2.5,
#     这份损害与 RSPR 无关。
#   - ρ = ‖∇L_RSPR‖/‖∇L_DSA‖ 是梯度范数之比,与学习率无关。coef_lr 不改变 RSPR
#     的梯度份额,只改变 trunk 每步走多远(对两个损失同等)。所以 v2 观察到的
#     +0.4 (T2V) / +1.4 (V2T) 既可能来自 λx3,也可能来自 trunk 走得更远。
# 本 run 填上 2x2 里缺的那格 (λx3, coef_lr=1e-3),对照基线是已有的 rspr_a0_seed0。
# 其余(seed0、hygiene、FREEZE_LAYER_NUM=8、5 epochs、warmup=1)与 a3fix/a0 对齐。
RUN_ID="rspr_a3fixv3_seed0"
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
RSPR_MODE=stochastic \
RSPR_PROB_LOSS=soft_bce \
RSPR_PROB_WEIGHT=0.3 \
RSPR_RANK_WEIGHT=0.3 \
RSPR_ANCHOR_WEIGHT=3e-2 \
COEF_LR=1e-3 \
RSPR_GRAD_DIAGNOSTICS=1 \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
FREEZE_LAYER_NUM=8 \
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
    log "A3fix-v3 completed successfully: ${OUTPUT_DIR}/final_test.json"
    exit 0
fi

LATEST_LOG="$(find "${ROOT_DIR}/logs" -maxdepth 2 -type f \
    -name "${RUN_ID}_*_train_msrvtt.log" -print0 |
    xargs -0 -r ls -1t | head -n 1)"
log "A3fix-v3 stopped without final_test.json; inspect ${LATEST_LOG:-no training log found}"
exit 1
