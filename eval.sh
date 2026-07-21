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
if [[ "${EXPERIMENT_PROFILE}" != "default" && "${EXPERIMENT_PROFILE}" != "hygiene" ]]; then
  echo "Unsupported EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}; expected default or hygiene" >&2
  exit 2
fi
if [[ "${CLIP_LAYER_NORM_PRECISION}" != "fp16" && "${CLIP_LAYER_NORM_PRECISION}" != "fp32" ]]; then
  echo "Unsupported CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION}; expected fp16 or fp32" >&2
  exit 2
fi
if [[ "${RSPR_MODE}" != "legacy" && "${RSPR_MODE}" != "off" && "${RSPR_MODE}" != "mean" && "${RSPR_MODE}" != "stochastic" ]]; then
  echo "Unsupported RSPR_MODE=${RSPR_MODE}; expected legacy, off, mean, or stochastic" >&2
  exit 2
fi
if [[ "${RSPR_MATCH_MODE}" != "soft" && "${RSPR_MATCH_MODE}" != "hard" ]]; then
  echo "Unsupported RSPR_MATCH_MODE=${RSPR_MATCH_MODE}; expected soft or hard" >&2
  exit 2
fi
for _RSPR_BOOLEAN in RSPR_DETACH_SAMPLES RSPR_FREEZE_CLIP RSPR_FREEZE_DSA; do
  if [[ "${!_RSPR_BOOLEAN}" != "0" && "${!_RSPR_BOOLEAN}" != "1" ]]; then
    echo "Unsupported ${_RSPR_BOOLEAN}=${!_RSPR_BOOLEAN}; expected 0 or 1" >&2
    exit 2
  fi
done
if [[ "${RSPR_MODE}" == "mean" && "${RSPR_SAMPLE_COUNT}" != "1" ]]; then
  echo "RSPR_MODE=mean requires RSPR_SAMPLE_COUNT=1" >&2
  exit 2
fi
if [[ "${RSPR_MODE}" == "stochastic" ]]; then
  for _RSPR_K in RSPR_SAMPLE_COUNT RSPR_EVAL_SAMPLE_COUNT; do
    if ! [[ "${!_RSPR_K}" =~ ^[1-9][0-9]*$ ]] || (( ${!_RSPR_K} % 2 )); then
      echo "Unsupported ${_RSPR_K}=${!_RSPR_K}; stochastic mode requires a positive even integer" >&2
      exit 2
    fi
  done
fi
for _RSPR_TEMPERATURE in RSPR_MATCH_TEMPERATURE RSPR_PROB_TEMPERATURE RSPR_RANK_TEMPERATURE RSPR_DET_TEMPERATURE RSPR_RERANK_TEMPERATURE; do
  if ! [[ "${!_RSPR_TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || ! awk -v value="${!_RSPR_TEMPERATURE}" 'BEGIN { exit !(value > 0 && value < 1e308) }'; then
    echo "Unsupported ${_RSPR_TEMPERATURE}=${!_RSPR_TEMPERATURE}; expected a positive finite number" >&2
    exit 2
  fi
done
if ! [[ "${RSPR_TOP_R}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported RSPR_TOP_R=${RSPR_TOP_R}; expected a non-negative integer" >&2
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
EXTRA_ARGS+=(--rspr_mode "${RSPR_MODE}")
EXTRA_ARGS+=(--rspr_sample_count "${RSPR_SAMPLE_COUNT}")
EXTRA_ARGS+=(--rspr_eval_sample_count "${RSPR_EVAL_SAMPLE_COUNT}")
EXTRA_ARGS+=(--rspr_match_mode "${RSPR_MATCH_MODE}")
EXTRA_ARGS+=(--rspr_match_temperature "${RSPR_MATCH_TEMPERATURE}")
EXTRA_ARGS+=(--rspr_prob_temperature "${RSPR_PROB_TEMPERATURE}")
EXTRA_ARGS+=(--rspr_rank_temperature "${RSPR_RANK_TEMPERATURE}")
EXTRA_ARGS+=(--rspr_hard_negatives "${RSPR_HARD_NEGATIVES}")
EXTRA_ARGS+=(--rspr_prior_std "${RSPR_PRIOR_STD}")
EXTRA_ARGS+=(--rspr_prob_weight "${RSPR_PROB_WEIGHT}")
EXTRA_ARGS+=(--rspr_rank_weight "${RSPR_RANK_WEIGHT}")
EXTRA_ARGS+=(--rspr_anchor_weight "${RSPR_ANCHOR_WEIGHT}")
EXTRA_ARGS+=(--rspr_warmup_epochs "${RSPR_WARMUP_EPOCHS}")
EXTRA_ARGS+=(--rspr_eval_seed "${RSPR_EVAL_SEED}")
EXTRA_ARGS+=(--rspr_top_r "${RSPR_TOP_R}")
EXTRA_ARGS+=(--rspr_det_temperature "${RSPR_DET_TEMPERATURE}")
EXTRA_ARGS+=(--rspr_rerank_temperature "${RSPR_RERANK_TEMPERATURE}")
EXTRA_ARGS+=(--rspr_rerank_weight "${RSPR_RERANK_WEIGHT}")
EXTRA_ARGS+=(--rspr_pair_chunk_size "${RSPR_PAIR_CHUNK_SIZE}")
if [[ "${RSPR_DETACH_SAMPLES}" == "1" ]]; then
  EXTRA_ARGS+=(--rspr_detach_samples)
fi
if [[ "${RSPR_FREEZE_CLIP}" == "1" ]]; then
  EXTRA_ARGS+=(--rspr_freeze_clip)
fi
if [[ "${RSPR_FREEZE_DSA}" == "1" ]]; then
  EXTRA_ARGS+=(--rspr_freeze_dsa)
fi
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
echo "[eval.sh] RSPR_MODE=${RSPR_MODE} RSPR_K=${RSPR_SAMPLE_COUNT} RSPR_EVAL_K=${RSPR_EVAL_SAMPLE_COUNT} RSPR_PROB_WEIGHT=${RSPR_PROB_WEIGHT} RSPR_RANK_WEIGHT=${RSPR_RANK_WEIGHT} RSPR_ANCHOR_WEIGHT=${RSPR_ANCHOR_WEIGHT} RSPR_FREEZE_CLIP=${RSPR_FREEZE_CLIP} RSPR_FREEZE_DSA=${RSPR_FREEZE_DSA} RSPR_TOP_R=${RSPR_TOP_R} RSPR_EVAL_SEED=${RSPR_EVAL_SEED}"
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
