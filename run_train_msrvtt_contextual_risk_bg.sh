#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "${ROOT_DIR}" rev-parse --path-format=absolute --git-common-dir)"
COMMON_REPO_ROOT="$(cd "${COMMON_GIT_DIR}/.." && pwd)"
BASE_CKPT="${COMMON_REPO_ROOT}/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
CONTROLLER_SCRIPT="${A3_BG_CONTROLLER:-${ROOT_DIR}/run_train_msrvtt_bg.sh}"
MANAGE_RUN_SCRIPT="${A3_TEST_MANAGE_CLI:-${ROOT_DIR}/scripts/manage_experiment_run.py}"

validate_test_seams() {
    if [[ -v LEGACY_A3_LAYOUT && "${A3_LAUNCHER_TEST_MODE:-}" != "1" ]]; then
        echo "LEGACY_A3_LAYOUT requires A3_LAUNCHER_TEST_MODE=1" >&2
        exit 2
    fi
    if [[ "${LEGACY_A3_LAYOUT:-0}" != "0" && "${LEGACY_A3_LAYOUT:-0}" != "1" ]]; then
        echo "LEGACY_A3_LAYOUT must be 0 or 1" >&2
        exit 2
    fi
    if [[ -n "${A3_BG_CONTROLLER:-}" || -n "${A3_TEST_OUTPUT_DIR:-}" ]]; then
        if [[ "${A3_LAUNCHER_TEST_MODE:-}" != "1" ]]; then
            echo "A3 test seams require A3_LAUNCHER_TEST_MODE=1" >&2
            exit 2
        fi
    fi
    if [[ -n "${A3_TEST_MANAGE_CLI:-}" || -n "${A3_TEST_ARTIFACT_ROOT:-}" || -n "${A3_TEST_REPO_ROOT:-}" ]]; then
        if [[ "${A3_LAUNCHER_TEST_MODE:-}" != "1" ]]; then
            echo "A3 artifact test seams require A3_LAUNCHER_TEST_MODE=1" >&2
            exit 2
        fi
    fi
    if [[ -n "${A3_TEST_OWNERSHIP_TIMEOUT_SECONDS:-}" && "${A3_LAUNCHER_TEST_MODE:-}" != "1" ]]; then
        echo "A3 test timeout requires A3_LAUNCHER_TEST_MODE=1" >&2
        exit 2
    fi
}

reject_registered_environment_overrides() {
    local name
    for name in \
        OUTPUT_DIR RUN_ID RUN_DATE RUN_TIME RUN_TAG LOG_FILE ARTIFACT_RUN_DIR \
        CUDA_VISIBLE_DEVICES NPROC TRAIN_BATCH_SIZE \
        TRAIN_GRADIENT_ACCUMULATION_STEPS CLIP_LAYER_NORM_PRECISION \
        EXPERIMENT_PROFILE TRAIN_PID_FILE RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER \
        TVR_PYTHON TVR_TORCHRUN A800_THROUGHPUT_COMPARISON \
        CLIP_GRADIENT_CHECKPOINTING CLIP_VISUAL_CHECKPOINT_LAYERS \
        FREEZE_LAYER_NUM DATA_PATH CLIP_CACHE_DIR TRAIN_NUM_WORKERS \
        TRAIN_PREFETCH_FACTOR MASTER_PORT MASTER_ADDR WORLD_SIZE RANK \
        LOCAL_RANK GROUP_RANK ROLE_RANK OMP_NUM_THREADS MKL_NUM_THREADS \
        OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS NCCL_P2P_LEVEL \
        NCCL_IB_DISABLE NCCL_DEBUG TORCH_WARN_ONCE PYTHONHASHSEED \
        CUBLAS_WORKSPACE_CONFIG SMOKE_STEPS EXPERIMENT_DESC \
        REGISTERED_CLIP_ASSET_PATH REGISTERED_CLIP_ASSET_SHA256 \
        ARTIFACT_MANIFEST_SHA256 WORKER_GROUP_DRAIN_TIMEOUT_SECONDS \
        WORKER_GROUP_TERM_TIMEOUT_SECONDS WORKER_GROUP_KILL_TIMEOUT_SECONDS; do
        if [[ -v "${name}" ]]; then
            echo "registered A3 environment ${name} cannot be overridden" >&2
            exit 2
        fi
    done
}

reject_registered_accelerator_prefix_overrides() {
    local name
    while IFS= read -r name; do
        case "${name}" in
            NCCL_*|TORCH_NCCL_*|CUDA_*|CUBLAS_*|NVIDIA_TF32_OVERRIDE|TORCH_ALLOW_TF32_CUBLAS_OVERRIDE)
                echo "registered A3 environment ${name} cannot be inherited" >&2
                exit 2
                ;;
        esac
    done < <(compgen -e)
}

reject_registered_argument_overrides() {
    if [[ "$#" -ne 0 ]]; then
        echo "registered A3 launcher does not accept trailing arguments" >&2
        exit 2
    fi
}

validate_test_seams
reject_registered_accelerator_prefix_overrides
reject_registered_environment_overrides
reject_registered_argument_overrides "$@"

CLIP_CACHE_DIR="${COMMON_REPO_ROOT}/.cache"
CLIP_ASSET="${CLIP_CACHE_DIR}/ViT-B-16.pt"
CLIP_ASSET_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"

if [[ "${A3_LAUNCHER_TEST_MODE:-0}" != "1" ]]; then
    if ! git -C "${ROOT_DIR}" diff --quiet -- || \
       ! git -C "${ROOT_DIR}" diff --cached --quiet --; then
        echo "registered A3 requires clean tracked source before launch" >&2
        exit 2
    fi
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing canonical A1 checkpoint BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${CLIP_ASSET}" ]]; then
    echo "Missing registered CLIP asset CLIP_ASSET=${CLIP_ASSET}" >&2
    exit 2
fi
ACTUAL_CLIP_ASSET_SHA256="$(sha256sum -- "${CLIP_ASSET}" | awk '{print $1}')"
if [[ "${ACTUAL_CLIP_ASSET_SHA256}" != "${CLIP_ASSET_SHA256}" ]]; then
    echo "Registered CLIP asset SHA-256 mismatch: ${CLIP_ASSET}" >&2
    exit 2
fi

RUN_DATE="$(date +%Y%m%d)"
RUN_TIME="$(date +%H%M%S)"
RUN_TAG="contextual_risk_a3"
RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
RUN_ID="${RUN_DATE}_${RUN_SUFFIX}"
ARTIFACT_ARGS=()
ARTIFACT_MODE=1
if [[ "${LEGACY_A3_LAYOUT:-0}" == "1" ]]; then
    ARTIFACT_MODE=0
fi
if [[ "${ARTIFACT_MODE}" == "0" && -n "${A3_TEST_OUTPUT_DIR:-}" ]]; then
    if [[ -z "${A3_BG_CONTROLLER:-}" ]]; then
        echo "A3_TEST_OUTPUT_DIR requires A3_BG_CONTROLLER" >&2
        exit 2
    fi
    OUTPUT_DIR="${A3_TEST_OUTPUT_DIR}"
elif [[ "${ARTIFACT_MODE}" == "0" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/ckpts/ckpt_msrvtt_${RUN_ID}"
else
    if [[ -z "${ARTIFACT_RUN_ID:-}" ]]; then
        echo "registered A3 requires ARTIFACT_RUN_ID" >&2
        exit 2
    fi
    ARTIFACT_REPO_ROOT="${A3_TEST_REPO_ROOT:-${ROOT_DIR}}"
    ARTIFACT_ROOT="${A3_TEST_ARTIFACT_ROOT:-${COMMON_REPO_ROOT}/artifacts}"
    REGISTRY_PATH="experiments/contextual-risk-v2/registry.jsonl"
    EXPECTED_RUN_DIR="$(
        /home/xujie/.conda/envs/tvr/bin/python - "${ARTIFACT_ROOT}" "${ARTIFACT_RUN_ID}" <<'PY'
import sys
from experiment_artifacts import make_run_paths
print(make_run_paths(sys.argv[1], "contextual-risk-v2", sys.argv[2]).root)
PY
    )" || exit 2
    # The CLI is the single authority for the committed manifest and clean
    # source check.  Parse only its small machine-readable identity response.
    VERIFY_JSON="$(
        /home/xujie/.conda/envs/tvr/bin/python "${MANAGE_RUN_SCRIPT}" verify-run \
            --repo-root "${ARTIFACT_REPO_ROOT}" \
            --registry "${REGISTRY_PATH}" \
            --run-dir "${EXPECTED_RUN_DIR}" \
            --run-id "${ARTIFACT_RUN_ID}" \
            --manifest-sha256 "$(
                /home/xujie/.conda/envs/tvr/bin/python - "${EXPECTED_RUN_DIR}/manifest.json" <<'PY'
import json, sys
from experiment_artifacts import manifest_sha256
with open(sys.argv[1], encoding="utf-8") as handle:
    print(manifest_sha256(json.load(handle)))
PY
            )"
    )" || exit 2
    mapfile -t ARTIFACT_IDENTITY < <(
        /home/xujie/.conda/envs/tvr/bin/python - "${VERIFY_JSON}" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload["run_dir"])
print(payload["manifest_sha256"])
print(payload["state"])
print("1" if payload["test_authorized"] else "0")
print(payload["stage"])
print(json.dumps(payload["command_argv"], separators=(",", ":")))
PY
    )
    ARTIFACT_RUN_DIR="${ARTIFACT_IDENTITY[0]}"
    ARTIFACT_MANIFEST_SHA256="${ARTIFACT_IDENTITY[1]}"
    if ! /home/xujie/.conda/envs/tvr/bin/python - "${ARTIFACT_IDENTITY[5]}" "${ARTIFACT_RUN_ID}" <<'PY'
import json, sys
actual = json.loads(sys.argv[1])
expected = [
    "env", f"ARTIFACT_RUN_ID={sys.argv[2]}", "bash",
    "run_train_msrvtt_contextual_risk_bg.sh",
]
if actual != expected:
    raise SystemExit("sealed command is not this exact registered A3 launcher invocation")
PY
    then
        exit 2
    fi
    if [[ "${ARTIFACT_IDENTITY[2]}" != "reserved" ]]; then
        echo "registered A3 run must be in reserved state" >&2
        exit 2
    fi
    OUTPUT_DIR="${ARTIFACT_RUN_DIR}/checkpoints"
    LOG_FILE="${ARTIFACT_RUN_DIR}/logs/train.log"
    RUN_ID="${ARTIFACT_RUN_ID}"
    ARTIFACT_ARGS=(
        --artifact_run_dir "${ARTIFACT_RUN_DIR}"
        --artifact_run_id "${ARTIFACT_RUN_ID}"
        --artifact_manifest_sha256 "${ARTIFACT_MANIFEST_SHA256}"
    )
    FINAL_TEST_ARGS=()
    if [[ "${ARTIFACT_IDENTITY[3]}" == "1" ]]; then
        if [[ "${ARTIFACT_IDENTITY[4]}" != "final" ]]; then
            echo "test authorization requires a final-stage reservation" >&2
            exit 2
        fi
        FINAL_TEST_ARGS=(--run_final_test)
    fi
fi
if [[ "${ARTIFACT_MODE}" == "0" ]]; then
    FINAL_TEST_ARGS=(--run_final_test)
fi

if [[ "${ARTIFACT_MODE}" == "0" && -e "${OUTPUT_DIR}" ]]; then
    echo "A3 output directory already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi

RESERVATION_DIR="${OUTPUT_DIR}.reserve"
if [[ "${ARTIFACT_MODE}" == "0" ]] && ! mkdir "${RESERVATION_DIR}"; then
    echo "A3 output reservation already exists or cannot be created: ${RESERVATION_DIR}" >&2
    exit 2
fi
RESERVATION_HELD=$((1 - ARTIFACT_MODE))
CONTROLLER_MAY_HAVE_SPAWNED=0
if [[ "${ARTIFACT_MODE}" == "1" ]]; then
    HANDOFF_PID_FILE="$(mktemp "${ARTIFACT_RUN_DIR}/provenance/controller-pid.XXXXXX")"
else
    HANDOFF_PID_FILE="$(mktemp "${RESERVATION_DIR}/controller-pid.XXXXXX")"
fi

retain_reservation_failure() {
    echo "$1" >&2
    if [[ "${ARTIFACT_MODE}" == "1" && "${CONTROLLER_MAY_HAVE_SPAWNED}" == "1" ]]; then
        kill -TERM "${CONTROLLER_PID}" 2>/dev/null || true
        wait "${CONTROLLER_PID}" 2>/dev/null || true
        unlink "${HANDOFF_PID_FILE}" 2>/dev/null || true
        if ! publish_artifact_terminal 2; then
            echo "A3 handoff failure terminal publication failed; retained run evidence: ${ARTIFACT_RUN_DIR}" >&2
        fi
    fi
    RESERVATION_HELD=0
    trap - EXIT
    exit 2
}

cleanup_reservation() {
    local status=$?
    if [[ "${RESERVATION_HELD}" == "1" ]]; then
        unlink "${HANDOFF_PID_FILE}" 2>/dev/null || true
        rmdir "${RESERVATION_DIR}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup_reservation EXIT

publish_artifact_terminal() {
    local controller_status="$1"
    [[ "${ARTIFACT_MODE}" == "1" ]] || return 0
    if [[ -f "${LOG_FILE}.process-group-drain.txt" ]] \
       && grep -q 'UNRESOLVED:live-process-group' \
            "${LOG_FILE}.process-group-drain.txt"; then
        echo "A3 refuses terminal hashing with an unresolved worker process group" >&2
        return 2
    fi
    local verify_json source_state terminal_state
    if [[ "${controller_status}" == "0" ]]; then
        source_state="validating"
    else
        verify_json="$(
            /home/xujie/.conda/envs/tvr/bin/python "${MANAGE_RUN_SCRIPT}" verify-run \
                --repo-root "${ARTIFACT_REPO_ROOT}" \
                --registry "${REGISTRY_PATH}" \
                --run-dir "${ARTIFACT_RUN_DIR}" \
                --run-id "${ARTIFACT_RUN_ID}" \
                --manifest-sha256 "${ARTIFACT_MANIFEST_SHA256}"
        )" || return 2
        source_state="$(
            /home/xujie/.conda/envs/tvr/bin/python - "${verify_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["state"])
PY
        )" || return 2
    fi
    if [[ "${controller_status}" == "0" && "${source_state}" == "validating" ]]; then
        terminal_state="completed"
    elif [[ "${controller_status}" != "0" && "${source_state}" == "validating" ]]; then
        terminal_state="failed"
    elif [[ "${controller_status}" != "0" && ( "${source_state}" == "running" || "${source_state}" == "reserved" ) ]]; then
        terminal_state="cancelled"
    else
        echo "A3 cannot terminalize status ${controller_status} from ${source_state}" >&2
        return 2
    fi
    /home/xujie/.conda/envs/tvr/bin/python "${MANAGE_RUN_SCRIPT}" transition-run \
        --repo-root "${ARTIFACT_REPO_ROOT}" \
        --registry "${REGISTRY_PATH}" \
        --run-dir "${ARTIFACT_RUN_DIR}" \
        --run-id "${ARTIFACT_RUN_ID}" \
        --manifest-sha256 "${ARTIFACT_MANIFEST_SHA256}" \
        --expected "${source_state}" \
        --next "${terminal_state}" \
        --exit-code "${controller_status}"
}

retain_reservation_on_signal() {
    local signal_name="$1"
    local exit_status="$2"
    trap - INT TERM
    if [[ "${CONTROLLER_MAY_HAVE_SPAWNED}" == "1" ]]; then
        kill -TERM "${CONTROLLER_PID}" 2>/dev/null || true
        wait "${CONTROLLER_PID}" 2>/dev/null || true
        unlink "${HANDOFF_PID_FILE}" 2>/dev/null || true
        if [[ "${ARTIFACT_MODE}" == "1" ]] && ! publish_artifact_terminal "${exit_status}"; then
            echo "A3 signal terminal publication failed; retained run evidence: ${ARTIFACT_RUN_DIR}" >&2
            exit 2
        fi
        if [[ "${ARTIFACT_MODE}" == "1" ]]; then
            echo "A3 ${signal_name} terminalized after closing writers; retained run evidence: ${ARTIFACT_RUN_DIR}" >&2
        else
            echo "A3 ${signal_name} received after controller start; retaining reservation: ${RESERVATION_DIR}" >&2
        fi
        RESERVATION_HELD=0
        trap - EXIT
    fi
    exit "${exit_status}"
}
trap 'retain_reservation_on_signal SIGINT 130' INT
trap 'retain_reservation_on_signal SIGTERM 143' TERM

CONTROLLER_MAY_HAVE_SPAWNED=1
RUN_DATE="${RUN_DATE}" \
RUN_TIME="${RUN_TIME}" \
RUN_TAG="${RUN_TAG}" \
RUN_ID="${RUN_ID}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOG_FILE="${LOG_FILE:-}" \
TRAIN_PID_FILE="${HANDOFF_PID_FILE}" \
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
CUDA_VISIBLE_DEVICES="0,2,3,4" \
CUDA_MODULE_LOADING=LAZY \
NPROC=4 \
TRAIN_BATCH_SIZE=256 \
TRAIN_GRADIENT_ACCUMULATION_STEPS=1 \
TRAIN_NUM_WORKERS=12 \
TRAIN_PREFETCH_FACTOR=4 \
CLIP_LAYER_NORM_PRECISION=fp32 \
CLIP_GRADIENT_CHECKPOINTING=1 \
CLIP_VISUAL_CHECKPOINT_LAYERS=4 \
CLIP_CACHE_DIR="${COMMON_REPO_ROOT}/.cache" \
REGISTERED_CLIP_ASSET_PATH="${CLIP_ASSET}" \
REGISTERED_CLIP_ASSET_SHA256="${CLIP_ASSET_SHA256}" \
FREEZE_LAYER_NUM=-1 \
A800_THROUGHPUT_COMPARISON=0 \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NCCL_P2P_LEVEL=NVL \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=WARN \
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_WARN_ONCE=1 \
WORKER_GROUP_DRAIN_TIMEOUT_SECONDS=30 \
WORKER_GROUP_TERM_TIMEOUT_SECONDS=10 \
WORKER_GROUP_KILL_TIMEOUT_SECONDS=5 \
MASTER_ADDR=127.0.0.9 \
MASTER_PORT=29547 \
DATA_PATH=/data2/hxj/data/MSRVTT \
PYTHONHASHSEED=0 \
EXPERIMENT_PROFILE=hygiene \
EXPERIMENT_DESC="A3 contextual retrieval-risk sidecar" \
bash "${CONTROLLER_SCRIPT}" \
    --use_uatvr_full \
    --use_ada_norm \
    --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar \
    --w_retrieval_risk 0.0 \
    --use_contextual_risk_sidecar \
    --contextual_risk_sidecar_only \
    --registered_contextual_risk_a3 \
    --registered_clip_asset_path "${CLIP_ASSET}" \
    --registered_clip_asset_sha256 "${CLIP_ASSET_SHA256}" \
    --contextual_risk_topk 16 \
    --contextual_risk_projection_dim 128 \
    --contextual_risk_hidden_dim 128 \
    --contextual_risk_temperature 1.0 \
    --w_contextual_risk_fit 0.25 \
    --w_contextual_risk_rank 0.25 \
    --seed 0 \
    --epochs 3 \
    "${FINAL_TEST_ARGS[@]}" \
    "${ARTIFACT_ARGS[@]}" \
    "$@" &
CONTROLLER_PID=$!

while [[ ! -s "${HANDOFF_PID_FILE}" ]]; do
    if ! kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
        set +e
        wait "${CONTROLLER_PID}"
        EARLY_CONTROLLER_STATUS=$?
        set -e
        if [[ "${EARLY_CONTROLLER_STATUS}" == "0" ]]; then
            EARLY_CONTROLLER_STATUS=2
        fi
        if [[ "${ARTIFACT_MODE}" == "1" ]]; then
            unlink "${HANDOFF_PID_FILE}" 2>/dev/null || true
            RESERVATION_HELD=0
            publish_artifact_terminal "${EARLY_CONTROLLER_STATUS}" || true
        fi
        echo "A3 controller exited before confirming worker handoff" >&2
        exit "${EARLY_CONTROLLER_STATUS}"
    fi
    sleep 0.05
done

read -r WORKER_PID < "${HANDOFF_PID_FILE}"
if ! [[ "${WORKER_PID}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    retain_reservation_failure "A3 worker handoff is not a live PID; leaving reservation: ${RESERVATION_DIR}"
fi

OWNERSHIP_TIMEOUT_SECONDS=30
if [[ -n "${A3_TEST_OWNERSHIP_TIMEOUT_SECONDS:-}" ]]; then
    OWNERSHIP_TIMEOUT_SECONDS="${A3_TEST_OWNERSHIP_TIMEOUT_SECONDS}"
fi
if ! [[ "${OWNERSHIP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    retain_reservation_failure "A3 ownership timeout must be a positive integer"
fi
OWNERSHIP_START_SECONDS=${SECONDS}
while [[ ! -d "${OUTPUT_DIR}" ]]; do
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
        retain_reservation_failure "A3 worker exited before creating output; leaving reservation: ${RESERVATION_DIR}"
    fi
    if (( SECONDS - OWNERSHIP_START_SECONDS >= OWNERSHIP_TIMEOUT_SECONDS )); then
        retain_reservation_failure "A3 output ownership timed out; leaving reservation: ${RESERVATION_DIR}"
    fi
    sleep 0.05
done
if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    retain_reservation_failure "A3 worker exited before durable output ownership; leaving reservation: ${RESERVATION_DIR}"
fi

unlink "${HANDOFF_PID_FILE}"
if [[ "${ARTIFACT_MODE}" == "0" ]]; then
    rmdir "${RESERVATION_DIR}"
fi
RESERVATION_HELD=0
set +e
wait "${CONTROLLER_PID}"
CONTROLLER_STATUS=$?
set -e

if [[ "${ARTIFACT_MODE}" == "1" ]]; then
    if ! publish_artifact_terminal "${CONTROLLER_STATUS}"; then
        echo "A3 terminal publication failed; retained run evidence: ${ARTIFACT_RUN_DIR}" >&2
        exit 2
    fi
fi
exit "${CONTROLLER_STATUS}"
