#!/usr/bin/env bash
set -uo pipefail

# Run the two v2 seed-0 runs strictly back to back.
#
# Each watcher already gates on "no GPU compute process anywhere", so the only
# correct way to queue them is sequentially in one process.  Backgrounding both
# lets the second watcher's gate fire before the first run has claimed the
# GPUs, and the two runs then collide (observed 2026-07-28: an unrelated vLLM
# server held 29.5 GB on GPU 0 and rank 0 OOM'd three seconds into step 1).
#
# A3fix-v2 only starts if A0-v2 produced final_test.json: without the matched
# baseline the comparison is meaningless, so a failed control should stop the
# queue rather than silently spend five epochs of GPU time.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHAIN_LOG="${ROOT_DIR}/logs/monitor/chain_v2_seed0.log"
mkdir -p "$(dirname "${CHAIN_LOG}")"

log() {
    echo "[$(date '+%F %T')] $*" >>"${CHAIN_LOG}"
}

run_stage() {
    local name="$1" script="$2" output_dir="$3"
    log "stage ${name}: starting ${script}"
    "${script}" >>"${CHAIN_LOG}" 2>&1
    local status=$?
    if [[ -f "${output_dir}/final_test.json" ]]; then
        log "stage ${name}: completed (final_test.json present)"
        return 0
    fi
    log "stage ${name}: FAILED (exit=${status}, no final_test.json)"
    return 1
}

if ! run_stage a0v2 "${ROOT_DIR}/scripts/watch_and_run_rspr_a0v2_seed0.sh" \
    "${ROOT_DIR}/ckpts/ckpt_msrvtt_rspr_a0v2_seed0"; then
    log "chain aborted: A3fix-v2 not started because its matched baseline failed"
    exit 1
fi

run_stage a3fixv2 "${ROOT_DIR}/scripts/watch_and_run_rspr_a3fixv2_seed0.sh" \
    "${ROOT_DIR}/ckpts/ckpt_msrvtt_rspr_a3fixv2_seed0"
