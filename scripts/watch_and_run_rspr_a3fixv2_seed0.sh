#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# A3fix-v2: 在 A3-fixed 基础上叠加三处修正,与 rspr_a0v2_seed0 严格匹配。
#   1. COEF_LR 1e-3 -> 1e-2: trunk 学习率 1e-7 -> 1e-6。A3-fixed 实测 RSPR 已占
#      trunk 梯度的 2.9%(epoch1) ~ 15%(epoch5),不是"接不上",而是 trunk 本身
#      几乎不动;这一项才是让 RSPR 的梯度份额真正转化为表征变化的杠杆。
#   2. λ 整体 x3 (λ_p=λ_r=0.3, λ_a=0.03): 保持 §7 标定出的 RSPR 三项内部平衡
#      (A3-fixed 下 σ 未塌缩),只提高相对 L_DSA 的份额到 epoch1-3 的 9%~27%。
#      不用更大的倍数: ρ 随 L_DSA 收敛单调上升,x140 会在 epoch5 让 RSPR 压过
#      DSA 21 倍。
#   3. 推理侧 rerank 走新默认(确定性召回 + logit_scale 对齐),由 CLI 默认值提供。
# 梯度诊断全程开启,用 grad_{prob,rank,anchor}_logvar 复核 λ_a。
RUN_ID="rspr_a3fixv2_seed0"
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
COEF_LR=1e-2 \
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
    log "A3fix-v2 completed successfully: ${OUTPUT_DIR}/final_test.json"
    exit 0
fi

LATEST_LOG="$(find "${ROOT_DIR}/logs" -maxdepth 2 -type f \
    -name "${RUN_ID}_*_train_msrvtt.log" -print0 |
    xargs -0 -r ls -1t | head -n 1)"
log "A3fix-v2 stopped without final_test.json; inspect ${LATEST_LOG:-no training log found}"
exit 1
