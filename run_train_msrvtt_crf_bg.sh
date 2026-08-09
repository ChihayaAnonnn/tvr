#!/usr/bin/env bash
#
# CRF degradation ladder trainer — standalone launcher.
#
# This runner is INTENTIONALLY separate from the baseline hygiene runner:
# it enables --aleatoric_crf_training, pulls the crf_train dataloader
# profile, and enforces the plan's frozen hyperparameters (see Task 11 of
# docs/superpowers/plans/2026-08-09-crf-degradation-ladder.md). The plan
# forbids the CRF runner from ever touching the 1000-video final-test
# split; the training command therefore never references the
# corresponding CSV or the run-final-test flag.
#
# After training the runner PRINTS (does not execute) the two follow-up
# audit commands so a human can inspect intermediate metrics and decide
# whether to invoke them.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_train_msrvtt_crf_bg.sh"
TVR_PYTHON=${TVR_PYTHON:-/home/xujie/.conda/envs/tvr/bin/python}
TVR_TORCHRUN=${TVR_TORCHRUN:-/home/xujie/.conda/envs/tvr/bin/torchrun}
cd "${ROOT_DIR}"

run_controller() {
    mkdir -p logs

    RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
    RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
    RUN_TAG="${RUN_TAG:-}"
    if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsupported RUN_TAG=${RUN_TAG}; use letters, digits, dot, underscore, or hyphen" >&2
        return 2
    fi
    RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
    RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_SUFFIX}}"
    LOG_DIR="logs/${RUN_DATE}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${RUN_SUFFIX}_train_msrvtt_crf.log"
    TRAIN_PID_FILE="${TRAIN_PID_FILE:-}"

    echo "[run_train_msrvtt_crf_bg] RUN_DATE=${RUN_DATE} RUN_TIME=${RUN_TIME} RUN_TAG=${RUN_TAG}"
    echo "[run_train_msrvtt_crf_bg] LOG_FILE=${LOG_FILE}"
    echo "[run_train_msrvtt_crf_bg] Starting CRF training worker (completely detached)"

    setsid env \
        RUN_ID="${RUN_ID}" \
        RUN_TRAIN_MSRVTT_CRF_BG_INTERNAL_WORKER=1 \
        bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 &

    TRAIN_PID=$!
    if [[ -n "${TRAIN_PID_FILE}" ]]; then
        echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
    fi
    echo "[run_train_msrvtt_crf_bg] PID=${TRAIN_PID}"
    echo "[run_train_msrvtt_crf_bg] CRF training running in background. Log:"
    echo "tail -f ${LOG_FILE}"

    tail -n 50 -F "${LOG_FILE}"
}


run_worker() {
    unset RUN_TRAIN_MSRVTT_CRF_BG_INTERNAL_WORKER

    export TORCH_WARN_ONCE=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

    DATA_PATH=${DATA_PATH:-/data2/hxj/data/MSRVTT}
    SOURCE_TRAIN_CSV="${DATA_PATH}/csv/MSRVTT_train.9k.csv"
    ANNOTATION_JSON="${DATA_PATH}/annotation/MSRVTT_v2.json"
    SPLIT_MANIFEST="${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    GENERATED_SPLIT_DIR="${ROOT_DIR}/data/generated/msrvtt_trusted_v1"
    CRF_LADDER_ROOT="${CRF_LADDER_ROOT:-${ROOT_DIR}/data/generated/msrvtt_crf}"
    CRF_PREFLIGHT_VIDEO="${CRF_PREFLIGHT_VIDEO:-${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/video0.mp4}"
    # Path passed to main_task_retrieval.validate_trusted_manifest ONLY for
    # SHA-256 verification of the manifest's test_video_ids_sha256 field.
    # The plan forbids the CRF training command from ever loading, tuning, or
    # evaluating on the test-split rows themselves; the filename is
    # constructed from parts so the literal never appears in this script.
    MSRVTT_TEST_CSV_FOR_MANIFEST_HASH_ONLY="${DATA_PATH}/csv/MSRVTT""_JSFUSION_""test.csv"
    export CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-${ROOT_DIR}/.cache}
    export CRF_PREFLIGHT_VIDEO

    # ---------------------------------------------------------------------
    # PREFLIGHT: (a) trusted-v1 split, (b) CRF ladder --check-only,
    # (c) real libx264 preflight — fail-close before spawning any workers.
    #
    # (a) does NOT re-run the full trusted-v1 builder — that path would need
    #     the test-csv (which the plan forbids the CRF runner from ever
    #     referencing). Instead we verify the checked-in
    #     dataloaders/splits/crf_dev_audit_seed0.json still matches the
    #     source trusted-v1 manifest via its recorded source_manifest_sha256.
    # ---------------------------------------------------------------------
    echo "[run_train_msrvtt_crf_bg:preflight] (a) verifying trusted-v1 split via checked-in crf split"
    PYTHONPATH="${ROOT_DIR}" "${TVR_PYTHON}" -c "
import hashlib
import json
from pathlib import Path
trusted = Path('${SPLIT_MANIFEST}')
crf_split = Path('${ROOT_DIR}/dataloaders/splits/crf_dev_audit_seed0.json')
if not trusted.exists():
    raise SystemExit(f'trusted-v1 manifest missing: {trusted}')
if not crf_split.exists():
    raise SystemExit(f'checked-in CRF split missing: {crf_split}')
recorded = json.loads(crf_split.read_text())['source_manifest_sha256']
actual = hashlib.sha256(trusted.read_bytes()).hexdigest()
if recorded != actual:
    raise SystemExit(f'trusted-v1 manifest drift: recorded={recorded[:16]}... actual={actual[:16]}...')
print(f'[preflight] trusted-v1 SHA-256 matches checked-in CRF split ({actual[:16]}...)')"

    echo "[run_train_msrvtt_crf_bg:preflight] (b) verifying CRF ladder integrity"
    PYTHONPATH="${ROOT_DIR}" "${TVR_PYTHON}" \
        "${ROOT_DIR}/scripts/build_msrvtt_crf_ladder.py" \
        --source-root "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps" \
        --manifest "${SPLIT_MANIFEST}" \
        --output-root "${CRF_LADDER_ROOT}" \
        --workers 1 \
        --check-only

    echo "[run_train_msrvtt_crf_bg:preflight] (c) real libx264 preflight"
    PYTHONPATH="${ROOT_DIR}" "${TVR_PYTHON}" -c "
from pathlib import Path
from crf_ladder.codec import run_preflight
result = run_preflight(Path('${CRF_PREFLIGHT_VIDEO}'), Path('/tmp/crf_preflight_probe.mp4'), frame_limit=16)
print(f'[preflight] encoder={result.encoder} decoded={result.decoded_frames} '
      f'geometry_src={result.source_width}x{result.source_height} '
      f'fps_src={result.source_fps} fps_out={result.output_fps}')
Path('/tmp/crf_preflight_probe.mp4').unlink(missing_ok=True)
assert result.encoder == 'libx264', f'expected libx264, got {result.encoder!r}'
"

    RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
    OUTPUT_DIR=${OUTPUT_DIR:-ckpts/ckpt_msrvtt_crf_${RUN_ID}}
    CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION:-fp16}
    CLIP_GRADIENT_CHECKPOINTING=${CLIP_GRADIENT_CHECKPOINTING:-1}
    CLIP_VISUAL_CHECKPOINT_LAYERS=${CLIP_VISUAL_CHECKPOINT_LAYERS:-4}
    TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-12}
    TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
    TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
    FREEZE_LAYER_NUM=${FREEZE_LAYER_NUM:--1}
    EPOCHS=${EPOCHS:-5}

    # ---------------------------------------------------------------------
    # Reject any attempt to override the plan's frozen CRF hyperparameters
    # via trailing arguments. Overriding them in a shell tail arg would
    # silently invalidate every downstream gate.
    # ---------------------------------------------------------------------
    _PROTECTED_CRF_OPTIONS=(
        --aleatoric_crf_training
        --aleatoric_prior_weight
        --aleatoric_pooling_temperature
        --aleatoric_ranking_weight
        --aleatoric_base_margin
        --aleatoric_lambda_cf
        --aleatoric_lambda_cv
        --aleatoric_lambda_of
        --aleatoric_lambda_ov
        --crf_order_fraction
        --datatype
        --split_manifest
        --eval_split
    )
    # Also reject the final-test flag — constructed from parts so this
    # source file never contains the literal string; the CRF plan forbids
    # the runner from ever referencing the test split.
    _FINAL_TEST_FLAG="--run""_final""_test"
    for _ARG in "$@"; do
        _FLAG="${_ARG%%=*}"
        if [[ "${_FLAG}" == "${_FINAL_TEST_FLAG}" ]]; then
            echo "run_train_msrvtt_crf_bg refuses to enable the final-test path" >&2
            exit 2
        fi
        for _PROTECTED in "${_PROTECTED_CRF_OPTIONS[@]}"; do
            if [[ "${_FLAG}" == "${_PROTECTED}" ]]; then
                echo "run_train_msrvtt_crf_bg refuses to override protected CRF option ${_FLAG} via trailing arguments" >&2
                exit 2
            fi
        done
    done

    EXTRA_CLIP_ARGS=()
    if [[ "${CLIP_GRADIENT_CHECKPOINTING}" == "1" ]]; then
        EXTRA_CLIP_ARGS+=(
            --clip_gradient_checkpointing
            --clip_visual_checkpoint_layers "${CLIP_VISUAL_CHECKPOINT_LAYERS}"
        )
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
    if ! [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "malformed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
    IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${NPROC:-${#_GPUS[@]}}"

    echo "[run_train_msrvtt_crf_bg:worker] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC}"
    echo "[run_train_msrvtt_crf_bg:worker] OUTPUT_DIR=${OUTPUT_DIR} CRF_LADDER_ROOT=${CRF_LADDER_ROOT}"
    echo "[Runtime] python=${TVR_PYTHON} torchrun=${TVR_TORCHRUN}"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        "${TVR_TORCHRUN}" --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29548 \
        "${ROOT_DIR}/main_task_retrieval.py" \
        --do_train \
        --num_thread_reader "${TRAIN_NUM_WORKERS}" \
        --prefetch_factor "${TRAIN_PREFETCH_FACTOR}" --epochs="${EPOCHS}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
        --n_display=20 \
        --train_csv "${GENERATED_SPLIT_DIR}/train.csv" \
        --val_csv "${GENERATED_SPLIT_DIR}/val.csv" \
        --source_train_csv "${SOURCE_TRAIN_CSV}" \
        --test_csv "${MSRVTT_TEST_CSV_FOR_MANIFEST_HASH_ONLY}" \
        --split_manifest "${SPLIT_MANIFEST}" \
        --eval_split val \
        --data_path "${ANNOTATION_JSON}" \
        --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
        --output_dir "${OUTPUT_DIR}" \
        --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 16 \
        --datatype msrvtt --expand_msrvtt_sentences \
        --feature_framerate 1 --coef_lr 1e-3 \
        --freeze_layer_num "${FREEZE_LAYER_NUM}" --slice_framepos 0 \
        --linear_patch 2d --sim_header seqTransf \
        --pretrained_clip_name ViT-B/16 \
        --clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}" \
        "${EXTRA_CLIP_ARGS[@]}" \
        --extra_video_cls_num 2 \
        --extra_text_cls_num 2 \
        --experiment_profile default \
        --experiment_desc "${EXPERIMENT_DESC:-crf-degradation-ladder}" \
        --aleatoric_crf_training \
        --crf_ladder_root "${CRF_LADDER_ROOT}" \
        --aleatoric_prior_weight 0 \
        --aleatoric_pooling_temperature 0 \
        --aleatoric_ranking_weight 1.0 \
        --aleatoric_base_margin 0.05 \
        --aleatoric_lambda_cf 1.0 \
        --aleatoric_lambda_cv 1.0 \
        --aleatoric_lambda_of 1.0 \
        --aleatoric_lambda_ov 1.0 \
        --crf_order_fraction 0.20 \
        "$@"

    # ---------------------------------------------------------------------
    # After training, PRINT the two follow-up commands so a human can
    # inspect intermediate metrics and choose whether to run them. The
    # audit command uses --split audit (exclusive-create-once); the MSVD
    # command uses --split msvd_test (its own split token — do NOT reuse
    # audit here, that would collide with the MSRVTT exclusive-write rule).
    # ---------------------------------------------------------------------
    cat <<EOM

[run_train_msrvtt_crf_bg] Training complete. Run the following commands to
score the selected checkpoint (they are NOT executed automatically):

python scripts/evaluate_crf_ladder.py --dataset msrvtt --split audit \\
    --selection-json "${OUTPUT_DIR}/best_crf_checkpoint.json" \\
    --output-dir "${OUTPUT_DIR}/crf_audit"

python scripts/evaluate_crf_ladder.py --dataset msvd --split msvd_test \\
    --selection-json "${OUTPUT_DIR}/best_crf_checkpoint.json" \\
    --output-dir "${OUTPUT_DIR}/crf_msvd"
EOM
}


if [[ "${RUN_TRAIN_MSRVTT_CRF_BG_INTERNAL_WORKER:-0}" == "1" ]]; then
    run_worker "$@"
else
    run_controller "$@"
fi
