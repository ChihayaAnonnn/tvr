#!/usr/bin/env bash
set -euo pipefail

: "${EVAL_SPLIT:?请显式设置 EVAL_SPLIT=val 或 EVAL_SPLIT=test}"
if [[ "${EVAL_SPLIT}" != "val" && "${EVAL_SPLIT}" != "test" ]]; then
  echo "EVAL_SPLIT 只能是 val 或 test" >&2
  exit 2
fi

# 用法示例：
#   EVAL_SPLIT=val INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> bash eval.sh
#   EVAL_SPLIT=test INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> bash eval.sh
#   # 可选 attributes 输入
#   EVAL_SPLIT=val INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
#   USE_ATTRIBUTES=1 ATTR_PATH=/abs/path/to/attributes.json bash eval.sh
#   # MSVD
#   DATATYPE=msvd \
#   EVAL_SPLIT=val INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
#   bash eval.sh

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATATYPE=${DATATYPE:-msrvtt}        # msrvtt | msvd
USE_ATTRIBUTES=${USE_ATTRIBUTES:-0} # 0 | 1
ATTR_PATH=${ATTR_PATH:-}
MAX_WORDS_ATTRS=${MAX_WORDS_ATTRS:-77}
: "${INIT_MODEL:?请设置 INIT_MODEL=<checkpoint_path>}"

# 模型结构参数（需与训练配置一致）
EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-hygiene}   # default | hygiene
CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION:-fp16} # fp16 | fp32
if [[ "${EXPERIMENT_PROFILE}" != "default" && "${EXPERIMENT_PROFILE}" != "hygiene" ]]; then
  echo "Unsupported EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}; expected default or hygiene" >&2
  exit 2
fi
if [[ "${CLIP_LAYER_NORM_PRECISION}" != "fp16" && "${CLIP_LAYER_NORM_PRECISION}" != "fp32" ]]; then
  echo "Unsupported CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION}; expected fp16 or fp32" >&2
  exit 2
fi

DATA_PATH=${DATA_PATH:-/data2/hxj/data/MSRVTT}
MSRVTT_DATA_PATH=${MSRVTT_DATA_PATH:-${DATA_PATH}}
MSVD_DATA_PATH=${MSVD_DATA_PATH:-/data2/hxj/data/MSVD}

OUTPUT_DIR=${OUTPUT_DIR:-ckpts/eval_${DATATYPE}_${RUN_ID}}

# Always log eval outputs (both console + file).
LOG_DATE=${LOG_DATE:-$(date +%Y%m%d)}
LOG_DIR=${LOG_DIR:-logs/eval/${LOG_DATE}}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_${DATATYPE}_ua${USE_ATTRIBUTES}.log}

if [[ "${DATATYPE}" == "msrvtt" ]]; then
  SOURCE_TRAIN_CSV="${MSRVTT_DATA_PATH}/csv/MSRVTT_train.9k.csv"
  TEST_CSV="${MSRVTT_DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
  ANNOTATION_JSON="${MSRVTT_DATA_PATH}/annotation/MSRVTT_v2.json"
  SPLIT_MANIFEST="${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed42.json"
  GENERATED_SPLIT_DIR="${ROOT_DIR}/data/generated/msrvtt_trusted_v1"
  python3 "${ROOT_DIR}/scripts/build_msrvtt_trusted_split.py" \
    --train-csv "${SOURCE_TRAIN_CSV}" \
    --annotation-json "${ANNOTATION_JSON}" \
    --test-csv "${TEST_CSV}" \
    --manifest "${SPLIT_MANIFEST}" \
    --output-dir "${GENERATED_SPLIT_DIR}"
  VAL_CSV="${GENERATED_SPLIT_DIR}/val.csv"
  DATA_PATH_ARG="${ANNOTATION_JSON}"
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
EXTRA_ARGS+=(--experiment_profile "${EXPERIMENT_PROFILE}")
EXTRA_ARGS+=(--eval_split "${EVAL_SPLIT}")
EXTRA_ARGS+=(--clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}")
if [[ "${DATATYPE}" == "msrvtt" ]]; then
  EXTRA_ARGS+=(--source_train_csv "${SOURCE_TRAIN_CSV}")
  EXTRA_ARGS+=(--test_csv "${TEST_CSV}")
  EXTRA_ARGS+=(--split_manifest "${SPLIT_MANIFEST}")
fi
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
echo "[eval.sh] DATATYPE=${DATATYPE} EVAL_SPLIT=${EVAL_SPLIT} USE_ATTRIBUTES=${USE_ATTRIBUTES} EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}"
echo "[eval.sh] PRETRAINED_CLIP_NAME=ViT-B/16 CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION}"
echo "[eval.sh] INIT_MODEL=${INIT_MODEL}"
echo "[eval.sh] OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${USE_ATTRIBUTES}" == "1" ]]; then
  echo "[eval.sh] ATTR_PATH=${ATTR_PATH}"
fi
echo "[eval.sh] LOG_FILE=${LOG_FILE}"

set -o pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
  torchrun --nproc_per_node=1 --master_addr=127.0.0.9 --master_port="${MASTER_PORT:-29520}" \
  "${ROOT_DIR}/main_task_retrieval.py" \
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
  --extra_video_cls_num 2 \
  --extra_text_cls_num 2 \
  --max_words 32 \
  --max_frames 8 \
  --feature_framerate 1 \
  --batch_size_val 8 \
  --slice_framepos 3 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
