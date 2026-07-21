#!/usr/bin/env bash
# Source-only helpers for deriving one validated RSPR configuration per entrypoint.

rspr_load_effective_config() {
    RSPR_MODE=${RSPR_MODE:-legacy}
    RSPR_SAMPLE_COUNT=${RSPR_SAMPLE_COUNT:-4}
    RSPR_EVAL_SAMPLE_COUNT=${RSPR_EVAL_SAMPLE_COUNT:-8}
    RSPR_MATCH_MODE=${RSPR_MATCH_MODE:-soft}
    RSPR_DETACH_SAMPLES=${RSPR_DETACH_SAMPLES:-0}
    RSPR_MATCH_TEMPERATURE=${RSPR_MATCH_TEMPERATURE:-0.07}
    RSPR_PROB_TEMPERATURE=${RSPR_PROB_TEMPERATURE:-0.07}
    RSPR_RANK_TEMPERATURE=${RSPR_RANK_TEMPERATURE:-0.07}
    RSPR_HARD_NEGATIVES=${RSPR_HARD_NEGATIVES:-8}
    RSPR_PRIOR_STD=${RSPR_PRIOR_STD:-0.1}
    RSPR_PROB_WEIGHT=${RSPR_PROB_WEIGHT:-0.1}
    RSPR_RANK_WEIGHT=${RSPR_RANK_WEIGHT:-0.1}
    RSPR_ANCHOR_WEIGHT=${RSPR_ANCHOR_WEIGHT:-1e-4}
    RSPR_WARMUP_EPOCHS=${RSPR_WARMUP_EPOCHS:-1.0}
    RSPR_EVAL_SEED=${RSPR_EVAL_SEED:-0}
    RSPR_TOP_R=${RSPR_TOP_R:-100}
    RSPR_DET_TEMPERATURE=${RSPR_DET_TEMPERATURE:-1.0}
    RSPR_RERANK_TEMPERATURE=${RSPR_RERANK_TEMPERATURE:-1.0}
    RSPR_RERANK_WEIGHT=${RSPR_RERANK_WEIGHT:-0.1}
    RSPR_PAIR_CHUNK_SIZE=${RSPR_PAIR_CHUNK_SIZE:-4096}
    RSPR_FREEZE_CLIP=${RSPR_FREEZE_CLIP:-0}
    RSPR_FREEZE_DSA=${RSPR_FREEZE_DSA:-0}

    RSPR_TRAILING_ARGS=()
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --rspr_detach_samples)
                RSPR_DETACH_SAMPLES=1
                shift
                ;;
            --rspr_freeze_clip)
                RSPR_FREEZE_CLIP=1
                shift
                ;;
            --rspr_freeze_dsa)
                RSPR_FREEZE_DSA=1
                shift
                ;;
            --rspr_mode|--rspr_sample_count|--rspr_eval_sample_count|--rspr_match_mode|--rspr_match_temperature|--rspr_prob_temperature|--rspr_rank_temperature|--rspr_hard_negatives|--rspr_prior_std|--rspr_prob_weight|--rspr_rank_weight|--rspr_anchor_weight|--rspr_warmup_epochs|--rspr_eval_seed|--rspr_top_r|--rspr_det_temperature|--rspr_rerank_temperature|--rspr_rerank_weight|--rspr_pair_chunk_size)
                if [[ "$#" -lt 2 ]]; then
                    echo "Missing value for $1" >&2
                    return 2
                fi
                rspr_assign_cli_value "$1" "$2"
                shift 2
                ;;
            --rspr_*=*)
                rspr_assign_cli_value "${1%%=*}" "${1#*=}" || return $?
                shift
                ;;
            --rspr_*)
                echo "Unsupported RSPR CLI option $1" >&2
                return 2
                ;;
            *)
                RSPR_TRAILING_ARGS+=("$1")
                shift
                ;;
        esac
    done

    rspr_validate_effective_config || return $?
    rspr_build_cli_args
}


rspr_assign_cli_value() {
    case "$1" in
        --rspr_mode) RSPR_MODE="$2" ;;
        --rspr_sample_count) RSPR_SAMPLE_COUNT="$2" ;;
        --rspr_eval_sample_count) RSPR_EVAL_SAMPLE_COUNT="$2" ;;
        --rspr_match_mode) RSPR_MATCH_MODE="$2" ;;
        --rspr_match_temperature) RSPR_MATCH_TEMPERATURE="$2" ;;
        --rspr_prob_temperature) RSPR_PROB_TEMPERATURE="$2" ;;
        --rspr_rank_temperature) RSPR_RANK_TEMPERATURE="$2" ;;
        --rspr_hard_negatives) RSPR_HARD_NEGATIVES="$2" ;;
        --rspr_prior_std) RSPR_PRIOR_STD="$2" ;;
        --rspr_prob_weight) RSPR_PROB_WEIGHT="$2" ;;
        --rspr_rank_weight) RSPR_RANK_WEIGHT="$2" ;;
        --rspr_anchor_weight) RSPR_ANCHOR_WEIGHT="$2" ;;
        --rspr_warmup_epochs) RSPR_WARMUP_EPOCHS="$2" ;;
        --rspr_eval_seed) RSPR_EVAL_SEED="$2" ;;
        --rspr_top_r) RSPR_TOP_R="$2" ;;
        --rspr_det_temperature) RSPR_DET_TEMPERATURE="$2" ;;
        --rspr_rerank_temperature) RSPR_RERANK_TEMPERATURE="$2" ;;
        --rspr_rerank_weight) RSPR_RERANK_WEIGHT="$2" ;;
        --rspr_pair_chunk_size) RSPR_PAIR_CHUNK_SIZE="$2" ;;
        *)
            echo "Unsupported RSPR CLI option $1" >&2
            return 2
            ;;
    esac
}


rspr_validate_effective_config() {
    if [[ "${RSPR_MODE}" != "legacy" && "${RSPR_MODE}" != "off" && "${RSPR_MODE}" != "mean" && "${RSPR_MODE}" != "stochastic" ]]; then
        echo "Unsupported RSPR_MODE=${RSPR_MODE}; expected legacy, off, mean, or stochastic" >&2
        return 2
    fi
    if [[ "${RSPR_MATCH_MODE}" != "soft" && "${RSPR_MATCH_MODE}" != "hard" ]]; then
        echo "Unsupported RSPR_MATCH_MODE=${RSPR_MATCH_MODE}; expected soft or hard" >&2
        return 2
    fi
    for _RSPR_BOOLEAN in RSPR_DETACH_SAMPLES RSPR_FREEZE_CLIP RSPR_FREEZE_DSA; do
        if [[ "${!_RSPR_BOOLEAN}" != "0" && "${!_RSPR_BOOLEAN}" != "1" ]]; then
            echo "Unsupported ${_RSPR_BOOLEAN}=${!_RSPR_BOOLEAN}; expected 0 or 1" >&2
            return 2
        fi
    done
    if [[ "${RSPR_MODE}" != "mean" && "${RSPR_MODE}" != "stochastic" ]] && (( RSPR_FREEZE_CLIP || RSPR_FREEZE_DSA )); then
        echo "RSPR freeze flags require mean or stochastic mode" >&2
        return 2
    fi
    for _RSPR_INTEGER in RSPR_SAMPLE_COUNT RSPR_EVAL_SAMPLE_COUNT RSPR_HARD_NEGATIVES RSPR_EVAL_SEED RSPR_TOP_R RSPR_PAIR_CHUNK_SIZE; do
        if ! [[ "${!_RSPR_INTEGER}" =~ ^-?[0-9]+$ ]]; then
            echo "Unsupported ${_RSPR_INTEGER}=${!_RSPR_INTEGER}; expected an integer" >&2
            return 2
        fi
    done
    for _RSPR_TEMPERATURE in RSPR_MATCH_TEMPERATURE RSPR_PROB_TEMPERATURE RSPR_RANK_TEMPERATURE RSPR_DET_TEMPERATURE RSPR_RERANK_TEMPERATURE; do
        if ! rspr_is_positive_finite "${!_RSPR_TEMPERATURE}"; then
            echo "Unsupported ${_RSPR_TEMPERATURE}=${!_RSPR_TEMPERATURE}; expected a positive finite number" >&2
            return 2
        fi
    done
    if [[ "${RSPR_MODE}" == "legacy" ]]; then
        return 0
    fi
    if [[ "${RSPR_MODE}" == "mean" && "${RSPR_SAMPLE_COUNT}" != "1" ]]; then
        echo "RSPR_MODE=mean requires RSPR_SAMPLE_COUNT=1" >&2
        return 2
    fi
    if [[ "${RSPR_MODE}" == "stochastic" ]]; then
        for _RSPR_K in RSPR_SAMPLE_COUNT RSPR_EVAL_SAMPLE_COUNT; do
            if ! [[ "${!_RSPR_K}" =~ ^[1-9][0-9]*$ ]] || (( ${!_RSPR_K} % 2 )); then
                echo "Unsupported ${_RSPR_K}=${!_RSPR_K}; stochastic mode requires a positive even integer" >&2
                return 2
            fi
        done
    fi
    if [[ "${RSPR_MODE}" == "mean" || "${RSPR_MODE}" == "stochastic" ]] && (( RSPR_HARD_NEGATIVES <= 0 )); then
        echo "Unsupported RSPR_HARD_NEGATIVES=${RSPR_HARD_NEGATIVES}; expected a positive integer" >&2
        return 2
    fi
    for _RSPR_POSITIVE in RSPR_PRIOR_STD; do
        if ! rspr_is_positive_finite "${!_RSPR_POSITIVE}"; then
            echo "Unsupported ${_RSPR_POSITIVE}=${!_RSPR_POSITIVE}; expected a positive finite number" >&2
            return 2
        fi
    done
    if (( RSPR_PAIR_CHUNK_SIZE <= 0 )); then
        echo "Unsupported RSPR_PAIR_CHUNK_SIZE=${RSPR_PAIR_CHUNK_SIZE}; expected a positive integer" >&2
        return 2
    fi
    for _RSPR_NONNEGATIVE in RSPR_PROB_WEIGHT RSPR_RANK_WEIGHT RSPR_ANCHOR_WEIGHT RSPR_RERANK_WEIGHT RSPR_WARMUP_EPOCHS; do
        if ! rspr_is_nonnegative_finite "${!_RSPR_NONNEGATIVE}"; then
            echo "Unsupported ${_RSPR_NONNEGATIVE}=${!_RSPR_NONNEGATIVE}; expected a nonnegative finite number" >&2
            return 2
        fi
    done
    if (( RSPR_TOP_R < 0 )); then
        echo "Unsupported RSPR_TOP_R=${RSPR_TOP_R}; expected a non-negative integer" >&2
        return 2
    fi
}


rspr_is_positive_finite() {
    [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] && awk -v value="$1" 'BEGIN { exit !(value > 0 && value < 1e308) }'
}


rspr_is_nonnegative_finite() {
    [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] && awk -v value="$1" 'BEGIN { exit !(value >= 0 && value < 1e308) }'
}


rspr_build_cli_args() {
    RSPR_CLI_ARGS=(
        --rspr_mode "${RSPR_MODE}"
        --rspr_sample_count "${RSPR_SAMPLE_COUNT}"
        --rspr_eval_sample_count "${RSPR_EVAL_SAMPLE_COUNT}"
        --rspr_match_mode "${RSPR_MATCH_MODE}"
        --rspr_match_temperature "${RSPR_MATCH_TEMPERATURE}"
        --rspr_prob_temperature "${RSPR_PROB_TEMPERATURE}"
        --rspr_rank_temperature "${RSPR_RANK_TEMPERATURE}"
        --rspr_hard_negatives "${RSPR_HARD_NEGATIVES}"
        --rspr_prior_std "${RSPR_PRIOR_STD}"
        --rspr_prob_weight "${RSPR_PROB_WEIGHT}"
        --rspr_rank_weight "${RSPR_RANK_WEIGHT}"
        --rspr_anchor_weight "${RSPR_ANCHOR_WEIGHT}"
        --rspr_warmup_epochs "${RSPR_WARMUP_EPOCHS}"
        --rspr_eval_seed "${RSPR_EVAL_SEED}"
        --rspr_top_r "${RSPR_TOP_R}"
        --rspr_det_temperature "${RSPR_DET_TEMPERATURE}"
        --rspr_rerank_temperature "${RSPR_RERANK_TEMPERATURE}"
        --rspr_rerank_weight "${RSPR_RERANK_WEIGHT}"
        --rspr_pair_chunk_size "${RSPR_PAIR_CHUNK_SIZE}"
    )
    if [[ "${RSPR_DETACH_SAMPLES}" == "1" ]]; then
        RSPR_CLI_ARGS+=(--rspr_detach_samples)
    fi
    if [[ "${RSPR_FREEZE_CLIP}" == "1" ]]; then
        RSPR_CLI_ARGS+=(--rspr_freeze_clip)
    fi
    if [[ "${RSPR_FREEZE_DSA}" == "1" ]]; then
        RSPR_CLI_ARGS+=(--rspr_freeze_dsa)
    fi
}


rspr_log_effective_config() {
    local prefix="$1"
    echo "[${prefix}] RSPR_MODE=${RSPR_MODE} RSPR_K=${RSPR_SAMPLE_COUNT} RSPR_EVAL_K=${RSPR_EVAL_SAMPLE_COUNT} RSPR_PROB_WEIGHT=${RSPR_PROB_WEIGHT} RSPR_RANK_WEIGHT=${RSPR_RANK_WEIGHT} RSPR_ANCHOR_WEIGHT=${RSPR_ANCHOR_WEIGHT} RSPR_FREEZE_CLIP=${RSPR_FREEZE_CLIP} RSPR_FREEZE_DSA=${RSPR_FREEZE_DSA} RSPR_TOP_R=${RSPR_TOP_R} RSPR_EVAL_SEED=${RSPR_EVAL_SEED}"
}
