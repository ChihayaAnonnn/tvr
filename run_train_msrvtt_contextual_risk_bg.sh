#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "${ROOT_DIR}" rev-parse --path-format=absolute --git-common-dir)"
COMMON_REPO_ROOT="$(cd "${COMMON_GIT_DIR}/.." && pwd)"
BASE_CKPT="${COMMON_REPO_ROOT}/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
CONTROLLER_SCRIPT="${A3_BG_CONTROLLER:-${ROOT_DIR}/run_train_msrvtt_bg.sh}"

reject_registered_environment_overrides() {
    local name
    for name in \
        OUTPUT_DIR RUN_ID RUN_DATE RUN_TIME RUN_TAG \
        CUDA_VISIBLE_DEVICES NPROC TRAIN_BATCH_SIZE \
        TRAIN_GRADIENT_ACCUMULATION_STEPS CLIP_LAYER_NORM_PRECISION \
        EXPERIMENT_PROFILE TRAIN_PID_FILE; do
        if [[ -v "${name}" ]]; then
            echo "registered A3 environment ${name} cannot be overridden" >&2
            exit 2
        fi
    done
}

reject_registered_argument_overrides() {
    local argument option
    for argument in "$@"; do
        option="${argument%%=*}"
        case "${option}" in
            --init_model|--resume_model|--use_uatvr_full|--use_ada_norm|\
            --use_retrieval_risk_sidecar|--retrieval_risk_sidecar_only|\
            --w_retrieval_risk|--use_contextual_risk_sidecar|\
            --contextual_risk_sidecar_only|--contextual_risk_topk|\
            --contextual_risk_projection_dim|--contextual_risk_hidden_dim|\
            --contextual_risk_temperature|--contextual_risk_bootstrap_samples|\
            --w_contextual_risk_fit|--w_contextual_risk_rank|\
            --clip_layer_norm_precision|--fp16|--seed|--nproc|\
            --nproc_per_node|--cuda_visible_devices|--batch_size|\
            --train_batch_size|--gradient_accumulation_steps|--epochs|\
            --experiment_profile|--eval_split|--datatype|\
            --expand_msrvtt_sentences|--train_csv|--val_csv|\
            --source_train_csv|--test_csv|--split_manifest|--data_path|\
            --features_path|--output_dir|--run_id|--run_date|--run_time|\
            --run_tag|--run_final_test|--use_csd_logvar_sidecar|\
            --csd_logvar_sidecar_only|--intrinsic_concentration_only)
                echo "registered A3 option ${option} cannot be overridden" >&2
                exit 2
                ;;
        esac
    done
}

reject_registered_environment_overrides
reject_registered_argument_overrides "$@"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing canonical A1 checkpoint BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi

RUN_DATE="$(date +%Y%m%d)"
RUN_TIME="$(date +%H%M%S)"
RUN_TAG="contextual_risk_a3"
RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
RUN_ID="${RUN_DATE}_${RUN_SUFFIX}"
if [[ -n "${A3_TEST_OUTPUT_DIR:-}" ]]; then
    if [[ -z "${A3_BG_CONTROLLER:-}" ]]; then
        echo "A3_TEST_OUTPUT_DIR requires A3_BG_CONTROLLER" >&2
        exit 2
    fi
    OUTPUT_DIR="${A3_TEST_OUTPUT_DIR}"
else
    OUTPUT_DIR="${ROOT_DIR}/ckpts/ckpt_msrvtt_${RUN_ID}"
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "A3 output directory already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi

RESERVATION_DIR="${OUTPUT_DIR}.reserve"
if ! mkdir "${RESERVATION_DIR}"; then
    echo "A3 output reservation already exists or cannot be created: ${RESERVATION_DIR}" >&2
    exit 2
fi
RESERVATION_HELD=1
HANDOFF_PID_FILE="$(mktemp "${RESERVATION_DIR}/controller-pid.XXXXXX")"

cleanup_reservation() {
    local status=$?
    if [[ "${RESERVATION_HELD}" == "1" ]]; then
        unlink "${HANDOFF_PID_FILE}" 2>/dev/null || true
        rmdir "${RESERVATION_DIR}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup_reservation EXIT

RUN_DATE="${RUN_DATE}" \
RUN_TIME="${RUN_TIME}" \
RUN_TAG="${RUN_TAG}" \
RUN_ID="${RUN_ID}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_PID_FILE="${HANDOFF_PID_FILE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}" \
NPROC=4 \
TRAIN_BATCH_SIZE=256 \
TRAIN_GRADIENT_ACCUMULATION_STEPS=1 \
CLIP_LAYER_NORM_PRECISION=fp32 \
EXPERIMENT_PROFILE=hygiene \
EXPERIMENT_DESC="${EXPERIMENT_DESC:-A3 contextual retrieval-risk sidecar}" \
bash "${CONTROLLER_SCRIPT}" \
    --use_uatvr_full \
    --use_ada_norm \
    --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar \
    --w_retrieval_risk 0.0 \
    --use_contextual_risk_sidecar \
    --contextual_risk_sidecar_only \
    --contextual_risk_topk 16 \
    --contextual_risk_projection_dim 128 \
    --contextual_risk_hidden_dim 128 \
    --contextual_risk_temperature 1.0 \
    --w_contextual_risk_fit 0.25 \
    --w_contextual_risk_rank 0.25 \
    --seed 0 \
    --epochs 3 \
    --run_final_test \
    "$@" &
CONTROLLER_PID=$!

while [[ ! -s "${HANDOFF_PID_FILE}" ]]; do
    if ! kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
        if wait "${CONTROLLER_PID}"; then
            echo "A3 controller exited before confirming worker handoff" >&2
            exit 2
        else
            exit $?
        fi
    fi
    sleep 0.05
done

unlink "${HANDOFF_PID_FILE}"
rmdir "${RESERVATION_DIR}"
RESERVATION_HELD=0
wait "${CONTROLLER_PID}"
