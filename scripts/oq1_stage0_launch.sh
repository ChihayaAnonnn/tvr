#!/usr/bin/env bash
# Stage 0 archive-qualification launcher for study oq1-kl-variance-hubness-v1.
#
# Registered command target of experiments/oq1-kl-variance-hubness-v1/phases/
# stage0-archive-qualification.json. Enforces the study's fail-closed
# device_selection policy at launch time, lets the manage CLI act as the single
# authority over the committed manifest, then hands off to the production
# controller in run_train_msrvtt_bg.sh (ARCHIVE_QUALIFICATION_SCENARIO branch).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/xujie/.conda/envs/tvr/bin/python
REGISTRY="experiments/oq1-kl-variance-hubness-v1/registry.jsonl"
MIN_FREE_MIB=70000

SCENARIO="" RUN_ID="" RUN_DIR="" DEVICES="" PORT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for v in SCENARIO RUN_ID RUN_DIR DEVICES PORT; do
    [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }
done
cd "${ROOT_DIR}"

# Fail-closed re-probe of the reserve-time device pin (study.yaml
# hardware_policy.device_selection). Refuse launch if any pinned GPU has
# dropped below the free-memory floor; archive the probe either way.
PROBE_FILE="${RUN_DIR}/provenance/gpu-probe-launch.json"
PROBE_OK=1
PROBE_ROWS=""
IFS=',' read -ra _DEV <<< "${DEVICES}"
if [[ "${#_DEV[@]}" -ne 4 ]]; then
    echo "device pin must name exactly 4 GPUs, got ${DEVICES}" >&2
    exit 3
fi
for idx in "${_DEV[@]}"; do
    line=$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader -i "${idx}")
    free=$(echo "${line}" | awk -F', ' '{print $3}' | awk '{print $1}')
    uuid=$(echo "${line}" | awk -F', ' '{print $2}')
    PROBE_ROWS+="{\"index\": ${idx}, \"uuid\": \"${uuid}\", \"free_mib\": ${free}},"
    if (( free < MIN_FREE_MIB )); then
        PROBE_OK=0
    fi
done
printf '{"probed_at_utc": "%s", "min_free_mib": %s, "pinned_devices": "%s", "sufficient": %s, "gpus": [%s]}\n' \
    "$(date -u +%Y%m%dT%H%M%SZ)" "${MIN_FREE_MIB}" "${DEVICES}" \
    "$([[ ${PROBE_OK} == 1 ]] && echo true || echo false)" "${PROBE_ROWS%,}" \
    > "${PROBE_FILE}"
if [[ "${PROBE_OK}" != "1" ]]; then
    echo "refusing launch: pinned GPUs below ${MIN_FREE_MIB} MiB free; evidence in ${PROBE_FILE}" >&2
    exit 3
fi

MANIFEST_SHA=$("${PY}" - "${RUN_DIR}/manifest.json" <<'PYEOF'
import sys
from experiment_artifacts import manifest_sha256, strict_json_load
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(manifest_sha256(strict_json_load(handle)))
PYEOF
)

PYTHONPATH=. "${PY}" scripts/manage_experiment_run.py verify-run \
    --repo-root "${ROOT_DIR}" \
    --registry "${REGISTRY}" \
    --run-dir "${RUN_DIR}" \
    --run-id "${RUN_ID}" \
    --manifest-sha256 "${MANIFEST_SHA}"

exec env \
    ARCHIVE_QUALIFICATION_SCENARIO="${SCENARIO}" \
    CUDA_VISIBLE_DEVICES="${DEVICES}" \
    NPROC=4 \
    MASTER_ADDR=127.0.0.9 \
    MASTER_PORT="${PORT}" \
    OUTPUT_DIR="${RUN_DIR}/checkpoints" \
    LOG_FILE="${RUN_DIR}/logs/controller.log" \
    bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" \
    --artifact_run_dir "${RUN_DIR}" \
    --artifact_run_id "${RUN_ID}" \
    --artifact_manifest_sha256 "${MANIFEST_SHA}"
