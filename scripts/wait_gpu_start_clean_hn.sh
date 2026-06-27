#!/usr/bin/env bash
set -euo pipefail

# Watch GPUs and launch the clean hard-negative MSRVTT run once they are idle.
# The raw command requested for launch is kept below; the script wraps it with
# logging, a lock, and a started marker to avoid duplicate runs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
MAX_USED_MB="${MAX_USED_MB:-2000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-5}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
CHECK_ONCE="${CHECK_ONCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
TAIL_AFTER_START="${TAIL_AFTER_START:-1}"
QUIET="${QUIET:-0}"

RUN_DATE_VALUE="${RUN_DATE_VALUE:-20260622}"
RUN_TIME_VALUE="${RUN_TIME_VALUE:-hn_pack_clean_wmil0_repeat1_4gpu_b64}"
EXPERIMENT_DESC_VALUE="${EXPERIMENT_DESC_VALUE:-Clean HN packing, 4GPU global batch=64, w_mil=0}"

LOG_DIR="logs/${RUN_DATE_VALUE}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/wait_gpu_${RUN_TIME_VALUE}.log}"
TRAIN_LOG="${LOG_DIR}/${RUN_TIME_VALUE}_train_msrvtt.log"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${LOG_DIR}/${RUN_TIME_VALUE}.train.pid}"
STARTED_FILE="${STARTED_FILE:-${LOG_DIR}/${RUN_TIME_VALUE}.watcher_started}"
LOCK_DIR="${LOCK_DIR:-${LOG_DIR}/.${RUN_TIME_VALUE}.watcher.lock}"

mkdir -p "${LOG_DIR}"

log() {
  local msg="$*"
  local line
  line="[$(date '+%F %T')] ${msg}"
  printf '%s\n' "${line}" >> "${WATCH_LOG}"
  if [[ "${QUIET}" != "1" ]]; then
    printf '%s\n' "${line}" || true
  fi
}

trim_number() {
  local value="$1"
  value="${value//[[:space:]]/}"
  printf '%s' "${value}"
}

gpu_count() {
  awk -v ids="${GPU_IDS}" 'BEGIN { print split(ids, parts, ",") }'
}

cleanup_lock() {
  rm -f "${LOCK_DIR}/watcher.pid" 2>/dev/null || true
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  if [[ -f "${LOCK_DIR}/watcher.pid" ]]; then
    old_pid="$(cat "${LOCK_DIR}/watcher.pid")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${old_pid}" 2>/dev/null; then
      log "Removing stale watcher lock for dead PID ${old_pid}: ${LOCK_DIR}"
      rm -f "${LOCK_DIR}/watcher.pid"
      rmdir "${LOCK_DIR}" 2>/dev/null || true
      mkdir "${LOCK_DIR}"
    else
      log "Another watcher appears to be running with PID ${old_pid}."
      exit 1
    fi
  else
    log "Another watcher lock already exists: ${LOCK_DIR}"
    exit 1
  fi
fi
printf '%s\n' "$$" > "${LOCK_DIR}/watcher.pid"
trap cleanup_lock EXIT

LAST_SNAPSHOT=""

all_gpus_idle() {
  local snapshot
  if ! snapshot="$(nvidia-smi -i "${GPU_IDS}" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
    log "nvidia-smi failed: ${snapshot}"
    return 1
  fi

  LAST_SNAPSHOT="${snapshot}"
  local expected count ok
  expected="$(gpu_count)"
  count=0
  ok=1

  while IFS=',' read -r index used total util; do
    [[ -z "${index:-}" ]] && continue
    index="$(trim_number "${index}")"
    used="$(trim_number "${used}")"
    total="$(trim_number "${total}")"
    util="$(trim_number "${util}")"
    count=$((count + 1))
    if (( used > MAX_USED_MB )); then
      ok=0
    fi
    if (( util > MAX_UTIL_PCT )); then
      ok=0
    fi
  done <<< "${snapshot}"

  if (( count != expected )); then
    log "Expected ${expected} GPUs from GPU_IDS=${GPU_IDS}, but nvidia-smi returned ${count}."
    return 1
  fi

  (( ok == 1 ))
}

archive_existing_train_log() {
  if [[ -s "${TRAIN_LOG}" ]]; then
    local archive_path
    archive_path="${TRAIN_LOG}.before_watcher_$(date '+%Y%m%d_%H%M%S')"
    mv "${TRAIN_LOG}" "${archive_path}"
    log "Archived existing train log to ${archive_path}"
  fi
}

launch_training() {
  if [[ -e "${STARTED_FILE}" ]]; then
    log "Started marker already exists, not launching again: ${STARTED_FILE}"
    exit 0
  fi

  log "GPU idle condition met. Snapshot:"
  while IFS= read -r line; do
    log "  ${line}"
  done <<< "${LAST_SNAPSHOT}"

  log "Launch command:"
  log "RUN_DATE=${RUN_DATE_VALUE} RUN_TIME=${RUN_TIME_VALUE} CUDA_VISIBLE_DEVICES=${GPU_IDS} EXPERIMENT_DESC=\"${EXPERIMENT_DESC_VALUE}\" bash run_train_msrvtt_bg.sh --use_hard_negative_packing --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json --w_mil 0 --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1, not launching training."
    exit 0
  fi

  archive_existing_train_log
  {
    printf 'started_at=%s\n' "$(date '+%F %T')"
    printf 'gpu_ids=%s\n' "${GPU_IDS}"
    printf 'max_used_mb=%s\n' "${MAX_USED_MB}"
    printf 'max_util_pct=%s\n' "${MAX_UTIL_PCT}"
    printf 'train_log=%s\n' "${TRAIN_LOG}"
    printf 'train_pid_file=%s\n' "${TRAIN_PID_FILE}"
  } > "${STARTED_FILE}"

  if [[ "${TAIL_AFTER_START}" == "0" ]]; then
    export NO_TAIL=1
    log "TAIL_AFTER_START=0, exporting NO_TAIL=1 for the launcher."
  fi

  RUN_DATE="${RUN_DATE_VALUE}" \
  RUN_TIME="${RUN_TIME_VALUE}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  EXPERIMENT_DESC="${EXPERIMENT_DESC_VALUE}" \
  TRAIN_PID_FILE="${TRAIN_PID_FILE}" \
  bash run_train_msrvtt_bg.sh \
    --use_hard_negative_packing \
    --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json \
    --w_mil 0 \
    --w_evidential 0 \
    --w_neg_reg 0 \
    --warmup_steps 500 \
    --batch_size 64 \
    --gradient_accumulation_steps 1
}

log "Watcher started. GPU_IDS=${GPU_IDS} MAX_USED_MB=${MAX_USED_MB} MAX_UTIL_PCT=${MAX_UTIL_PCT} CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS}"
log "Watcher log: ${WATCH_LOG}"
log "Train log: ${TRAIN_LOG}"

while true; do
  if all_gpus_idle; then
    launch_training
    exit 0
  fi

  log "GPUs are not idle yet. Current snapshot:"
  while IFS= read -r line; do
    log "  ${line}"
  done <<< "${LAST_SNAPSHOT}"

  if [[ "${CHECK_ONCE}" == "1" ]]; then
    log "CHECK_ONCE=1, exiting without launch."
    exit 2
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
