#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CKPT="${BASE_CKPT:-${ROOT_DIR}/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4}"
CSD_VARIANT="${CSD_VARIANT:-CG}"
CSD_EPOCHS="${CSD_EPOCHS:-3}"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi

case "${CSD_VARIANT}" in
    "CG") CSD_WEIGHT=1.0; GROUNDING_WEIGHT=0.05 ;;
    "C") CSD_WEIGHT=1.0; GROUNDING_WEIGHT=0.0 ;;
    "G") CSD_WEIGHT=0.0; GROUNDING_WEIGHT=0.05 ;;
    *)
        echo "Unsupported CSD_VARIANT=${CSD_VARIANT}; expected CG, C, or G" >&2
        exit 2
        ;;
esac

RUN_TAG="${RUN_TAG:-csd_logvar_a2_${CSD_VARIANT,,}}" \
EXPERIMENT_DESC="${EXPERIMENT_DESC:-A2-${CSD_VARIANT} independent CSD logvar sidecar}" \
bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" \
    --use_uatvr_full --use_ada_norm --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar --w_retrieval_risk 0.0 \
    --retrieval_risk_temperature 1.0 --retrieval_risk_hidden_dim 32 \
    --retrieval_risk_bootstrap_samples 1000 \
    --use_csd_logvar_sidecar --csd_logvar_sidecar_only \
    --w_csd "${CSD_WEIGHT}" --w_csd_grounding "${GROUNDING_WEIGHT}" --w_csd_vib 0.0001 \
    --csd_logvar_hidden_dim 128 --csd_logvar_min -12 --csd_logvar_max 4 \
    --csd_logvar_init -8 --csd_initial_scale 5 --csd_initial_shift 5 \
    --csd_hardness_temperature 1 --csd_bootstrap_samples 1000 \
    --epochs "${CSD_EPOCHS}" --DSL True --dsl_temperature 2.0 --run_final_test "$@"
