#!/usr/bin/env bash
# 自动化 smoke test：起训练 -> 等 N 秒 -> 检查日志关键指标 -> kill 收尾
# 用途：验证 NVLink 生效、无 OOM、能跑通前若干 iter
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SMOKE_SECONDS=${SMOKE_SECONDS:-240}   # 观察多少秒后收尾（默认 4 分钟）
PID_FILE="$(mktemp -t uatvr_smoke_pid.XXXXXX)"
trap 'rm -f "${PID_FILE}"' EXIT

echo "[smoke] Launching training with NCCL_DEBUG=INFO, will observe for ${SMOKE_SECONDS}s"

# 用 hygiene profile 允许的方式：epochs=1 是非受保护参数
EXPERIMENT_DESC=${EXPERIMENT_DESC:-smoke_nvlink_unfreeze} \
NCCL_DEBUG=INFO \
TRAIN_PID_FILE="${PID_FILE}" \
bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" --epochs=1 &

CONTROLLER_PID=$!
# 等 controller 把 worker PID 写入文件（最多 10 秒）
for _ in $(seq 1 20); do
    if [[ -s "${PID_FILE}" ]]; then break; fi
    sleep 0.5
done
if [[ ! -s "${PID_FILE}" ]]; then
    echo "[smoke] FAILED: worker PID not captured within 10s"
    kill "${CONTROLLER_PID}" 2>/dev/null || true
    exit 2
fi
WORKER_PID=$(cat "${PID_FILE}")
echo "[smoke] worker PID=${WORKER_PID}"

# 找最新日志文件
sleep 3
LOG_FILE=$(ls -1t logs/*/*_train_msrvtt.log 2>/dev/null | head -n1 || true)
if [[ -z "${LOG_FILE}" ]]; then
    echo "[smoke] FAILED: no log file appeared under logs/"
    kill -TERM -"${WORKER_PID}" 2>/dev/null || kill "${WORKER_PID}" 2>/dev/null || true
    exit 2
fi
echo "[smoke] LOG_FILE=${LOG_FILE}"

# 观察窗口内 tail 一下日志末尾（后台）
tail -n 0 -F "${LOG_FILE}" &
TAIL_PID=$!

REMAIN=${SMOKE_SECONDS}
STEP=10
while (( REMAIN > 0 )); do
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
        echo ""
        echo "[smoke] worker exited early — see log for details"
        break
    fi
    sleep "${STEP}"
    REMAIN=$(( REMAIN - STEP ))
done

# 收尾：先温柔杀掉整个进程组，兜底 SIGKILL
echo ""
echo "[smoke] stopping training (worker=${WORKER_PID})"
kill "${TAIL_PID}" 2>/dev/null || true
if kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill -TERM -"${WORKER_PID}" 2>/dev/null || kill -TERM "${WORKER_PID}" 2>/dev/null || true
    sleep 5
    if kill -0 "${WORKER_PID}" 2>/dev/null; then
        kill -KILL -"${WORKER_PID}" 2>/dev/null || kill -KILL "${WORKER_PID}" 2>/dev/null || true
    fi
fi
# 顺带 kill 残留的 torchrun/python worker（同一 log 里的 CUDA 进程）
pkill -TERM -f "main_task_retrieval.py" 2>/dev/null || true
sleep 3
pkill -KILL -f "main_task_retrieval.py" 2>/dev/null || true

echo ""
echo "===================== SMOKE REPORT ====================="
echo "log: ${LOG_FILE}"
echo ""

# 1) 致命错误（排除本脚本主动 kill 造成的 SIGTERM traceback）
echo "--- [1] fatal errors (OOM / CUDA / traceback) ---"
FATAL_HITS=$(grep -nE "CUDA out of memory|OutOfMemoryError|NCCL error|assertion.*failed" "${LOG_FILE}" || true)
# 排除仅由 smoke 自己 SIGTERM 引发的 SignalException 15
OTHER_TRACEBACK=$(grep -nE "RuntimeError|Traceback" "${LOG_FILE}" \
    | grep -vE "SignalException.*sigval=15|got signal: 15|SIGTERM" || true)
if [[ -n "${FATAL_HITS}" || -n "${OTHER_TRACEBACK}" ]]; then
    echo "${FATAL_HITS}"
    echo "${OTHER_TRACEBACK}"
    FATAL=1
else
    echo "  (none — SIGTERM traceback from smoke shutdown is ignored)"
    FATAL=0
fi
echo ""

# 2) NVLink / P2P 通道确认
echo "--- [2] NCCL channels (want to see P2P/NVLink between GPU2 & GPU4) ---"
grep -nE "NCCL INFO Channel.*(P2P|NVL|via.*P2P)" "${LOG_FILE}" | head -n 12 || echo "  (no NCCL channel lines found — check NCCL_DEBUG)"
echo ""
echo "--- [2b] any NCCL ring/topo summary ---"
grep -nE "NCCL INFO (Ring|Trees|Topology|NVLink)" "${LOG_FILE}" | head -n 10 || echo "  (none)"
echo ""

# 3) 有没有真正跑到训练迭代
echo "--- [3] training progress (loss / step lines) ---"
grep -nE "loss|Epoch|Step|iter" "${LOG_FILE}" | tail -n 15 || echo "  (no training log lines yet — window may be too short)"
echo ""

# 4) 显存峰值（如果日志有打）
echo "--- [4] memory hints ---"
grep -nE "memory|MB|GiB|GB" "${LOG_FILE}" | grep -iE "mem|alloc|reserv" | head -n 10 || echo "  (no memory lines)"
echo ""

echo "===================== VERDICT ====================="
if (( FATAL == 1 )); then
    echo "FAIL: fatal errors detected — inspect log above"
    exit 1
fi
if ! grep -qE "loss|Epoch|Step" "${LOG_FILE}"; then
    echo "INCONCLUSIVE: training didn't reach loss lines in ${SMOKE_SECONDS}s — bump SMOKE_SECONDS and retry"
    exit 3
fi
echo "PASS: no fatal errors and training was progressing"
