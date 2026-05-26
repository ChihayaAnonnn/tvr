#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   # MSRVTT (caption only)
#   INIT_MODEL=ckpts/ckpt_msrvtt_20260111_153216/pytorch_model.bin.3 bash eval.sh
#
#   # MSRVTT (caption+attrs, query_only)
#   INIT_MODEL=ckpts/ckpt_msrvtt_20260111_153216/pytorch_model.bin.3 \
#   USE_ATTRIBUTES=1 \
#   ATTR_PATH=/data2/hxj/project/UATVR/deploy_qwen/attributes/msrvtt/final/msrvtt_train9k_attributes.json \
#   EVAL_BRANCH_MODE=query_only \
#   bash eval.sh
#
#   # MSVD (caption+attrs, default fusion)
#   DATATYPE=msvd \
#   INIT_MODEL=ckpts/ckpt_msvd_20260112_021453/pytorch_model.bin.1 \
#   USE_ATTRIBUTES=1 \
#   ATTR_PATH=/data2/hxj/project/UATVR/deploy_qwen/attributes/msvd/final/msvd_test_attributes.json \
#   EVAL_BRANCH_MODE=default \
#   bash eval.sh

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
DATATYPE=${DATATYPE:-msrvtt}                 # msrvtt | msvd
EVAL_BRANCH_MODE=${EVAL_BRANCH_MODE:-default} # base_only | query_only | default
USE_ATTRIBUTES=${USE_ATTRIBUTES:-0}          # 0 | 1
ATTR_PATH=${ATTR_PATH:-}
: "${INIT_MODEL:?请设置 INIT_MODEL=<checkpoint_path>，例如 ckpts/ckpt_msrvtt_20260111_153216/pytorch_model.bin.3}"

MSRVTT_DATA_PATH=${MSRVTT_DATA_PATH:-/data2/hxj/data/MSRVTT}
MSVD_DATA_PATH=${MSVD_DATA_PATH:-/data2/hxj/data/MSVD}

OUTPUT_DIR=${OUTPUT_DIR:-ckpts/eval_${DATATYPE}_${EVAL_BRANCH_MODE}_${RUN_ID}}

# Always log eval outputs (both console + file).
LOG_DATE=${LOG_DATE:-$(date +%Y%m%d)}
LOG_DIR=${LOG_DIR:-logs/eval/${LOG_DATE}}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_${DATATYPE}_${EVAL_BRANCH_MODE}_ua${USE_ATTRIBUTES}.log}

if [[ "${DATATYPE}" == "msrvtt" ]]; then
  VAL_CSV="${MSRVTT_DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
  DATA_PATH_ARG="${MSRVTT_DATA_PATH}/annotation/MSRVTT_v2.json"
  FEATURES_PATH_ARG="${MSRVTT_DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/"
elif [[ "${DATATYPE}" == "msvd" ]]; then
  # MSVD dataloader mainly uses --data_path (desc_files folder) + --features_path (YouTubeClips).
  # --val_csv is kept for argparse compatibility; it is not the primary signal for MSVD splits here.
  VAL_CSV=${VAL_CSV:-data/.val.csv}
  DATA_PATH_ARG="${MSVD_DATA_PATH}/desc_files"
  FEATURES_PATH_ARG="${MSVD_DATA_PATH}/YouTubeClips"
else
  echo "未知 DATATYPE: ${DATATYPE} (仅支持 msrvtt/msvd)" >&2
  exit 2
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(--eval_branch_mode "${EVAL_BRANCH_MODE}")
if [[ "${USE_ATTRIBUTES}" == "1" ]]; then
  # Auto-pick MSRVTT JSFUSION test attributes if not explicitly provided.
  if [[ -z "${ATTR_PATH}" && "${DATATYPE}" == "msrvtt" ]]; then
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DEFAULT_ATTR_JSON="${ROOT_DIR}/deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json"
    DEFAULT_ATTR_JSONL="${ROOT_DIR}/deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.jsonl"
    if [[ -f "${DEFAULT_ATTR_JSON}" ]]; then
      ATTR_PATH="${DEFAULT_ATTR_JSON}"
    elif [[ -f "${DEFAULT_ATTR_JSONL}" ]]; then
      ATTR_PATH="${DEFAULT_ATTR_JSONL}"
    fi
  fi
  if [[ -z "${ATTR_PATH}" ]]; then
    echo "USE_ATTRIBUTES=1 但未设置 ATTR_PATH，且未找到默认的 attributes 文件。" >&2
    echo "请设置：ATTR_PATH=/abs/path/to/attributes.json(or .jsonl)" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--use_attributes)
  if [[ "${DATATYPE}" == "msrvtt" ]]; then
    EXTRA_ARGS+=(--msrvtt_attributes_path "${ATTR_PATH}")
  elif [[ "${DATATYPE}" == "msvd" ]]; then
    EXTRA_ARGS+=(--msvd_attributes_path "${ATTR_PATH}")
  fi
  # max_words_attrs must match training config to avoid token length mismatch
  if [[ -n "${MAX_WORDS_ATTRS}" ]]; then
    EXTRA_ARGS+=(--max_words_attrs "${MAX_WORDS_ATTRS}")
  fi
fi

# 单卡评测（用 torchrun 注入分布式环境变量）。注意：此脚本不传 --DSL，确保 DSL 关闭。
echo "[eval.sh] RUN_ID=${RUN_ID}"
echo "[eval.sh] DATATYPE=${DATATYPE} EVAL_BRANCH_MODE=${EVAL_BRANCH_MODE} USE_ATTRIBUTES=${USE_ATTRIBUTES}"
echo "[eval.sh] INIT_MODEL=${INIT_MODEL}"
echo "[eval.sh] OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${USE_ATTRIBUTES}" == "1" ]]; then
  echo "[eval.sh] ATTR_PATH=${ATTR_PATH}"
fi
echo "[eval.sh] LOG_FILE=${LOG_FILE}"

set -o pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
  torchrun --nproc_per_node=1 --master_addr=127.0.0.9 --master_port="${MASTER_PORT:-29520}" \
  main_task_retrieval.py \
  --do_eval \
  --log_mus_scores \
  --init_model "${INIT_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --datatype "${DATATYPE}" \
  --val_csv "${VAL_CSV}" \
  --data_path "${DATA_PATH_ARG}" \
  --features_path "${FEATURES_PATH_ARG}" \
  --pretrained_clip_name ViT-B/16 \
  --linear_patch 2d \
  --sim_header seqTransf \
  --strategy 2 \
  --extra_video_cls_num 2 \
  --extra_text_cls_num 2 \
  --max_words 32 \
  --max_frames 12 \
  --feature_framerate 1 \
  --batch_size_val 8 \
  --loose_type \
  --slice_framepos 3 \
  --uncertainty_text_head text \
  --log_sigma_min -3 \
  --log_sigma_max 6 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
