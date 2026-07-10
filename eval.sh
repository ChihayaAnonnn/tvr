#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   # MSRVTT (caption only)
#   INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> bash eval.sh
#
#   # MSRVTT (caption+attrs, query_only)
#   INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
#   USE_ATTRIBUTES=1 \
#   EVAL_BRANCH_MODE=query_only \
#   bash eval.sh
#
#   # MSVD (caption only)
#   DATATYPE=msvd \
#   INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
#   bash eval.sh
#
#   # Deprecated NIG-MIL compatibility mode
#   INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
#   UNCERTAINTY_MODE=nig_mil \
#   bash eval.sh

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
DATATYPE=${DATATYPE:-msrvtt}                 # msrvtt | msvd
EVAL_BRANCH_MODE=${EVAL_BRANCH_MODE:-default} # base_only | query_only | default
USE_ATTRIBUTES=${USE_ATTRIBUTES:-0}          # 0 | 1
ATTR_PATH=${ATTR_PATH:-}
: "${INIT_MODEL:?请设置 INIT_MODEL=<checkpoint_path>}"

# 模型结构参数（需与训练配置一致）
FUSION_MODE=${FUSION_MODE:-prob_mos}          # prob_mos | logits_linear
ROPE_MODE=${ROPE_MODE:-2d}                    # none | 2d | 3d
USE_ADA_NORM=${USE_ADA_NORM:-1}              # 0 | 1
UNCERTAINTY_MODE=${UNCERTAINTY_MODE:-evidential}    # evidential | none | nig_mil
EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-default}   # default | hygiene
BACKBONE_TYPE=${BACKBONE_TYPE:-openai_clip}          # openai_clip | eva_clip
BACKBONE_NAME=${BACKBONE_NAME:-EVA02-CLIP-B-16}
BACKBONE_PATH=${BACKBONE_PATH:-ref/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt}
EVA_CLIP_ROOT=${EVA_CLIP_ROOT:-ref/EVA/EVA-CLIP/rei}
EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN:-0}              # 0 | 1
if [[ "${EVA_CLIP_USE_XATTN}" != "0" && "${EVA_CLIP_USE_XATTN}" != "1" ]]; then
  echo "Unsupported EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN}; expected 0 or 1" >&2
  exit 2
fi

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
EXTRA_ARGS+=(--fusion_mode "${FUSION_MODE}")
EXTRA_ARGS+=(--rope_mode "${ROPE_MODE}")
EXTRA_ARGS+=(--uncertainty_mode "${UNCERTAINTY_MODE}")
EXTRA_ARGS+=(--experiment_profile "${EXPERIMENT_PROFILE}")
EXTRA_ARGS+=(--backbone_type "${BACKBONE_TYPE}")
EXTRA_ARGS+=(--backbone_name "${BACKBONE_NAME}")
EXTRA_ARGS+=(--backbone_path "${BACKBONE_PATH}")
EXTRA_ARGS+=(--eva_clip_root "${EVA_CLIP_ROOT}")
if [[ "${EVA_CLIP_USE_XATTN}" == "1" ]]; then
  EXTRA_ARGS+=(--eva_clip_use_xattn)
fi
EXTRA_ARGS+=(--final_score_mode "${FINAL_SCORE_MODE:-wti}")
EXTRA_ARGS+=(--lambda_prob "${LAMBDA_PROB:-0.0}")
EXTRA_ARGS+=(--lambda_anchor "${LAMBDA_ANCHOR:-0.0}")
EXTRA_ARGS+=(--lambda_qc_sap "${LAMBDA_QC_SAP:-0.0}")
EXTRA_ARGS+=(--qc_sap_temperature "${QC_SAP_TEMPERATURE:-0.1}")
if [[ "${USE_ADA_NORM}" == "1" ]]; then
  EXTRA_ARGS+=(--use_ada_norm)
fi
if [[ "${DATATYPE}" == "msrvtt" ]]; then
  EXTRA_ARGS+=(--expand_msrvtt_sentences)
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
echo "[eval.sh] DATATYPE=${DATATYPE} EVAL_BRANCH_MODE=${EVAL_BRANCH_MODE} USE_ATTRIBUTES=${USE_ATTRIBUTES}"
echo "[eval.sh] FUSION_MODE=${FUSION_MODE} ROPE_MODE=${ROPE_MODE} USE_ADA_NORM=${USE_ADA_NORM} UNCERTAINTY_MODE=${UNCERTAINTY_MODE} EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE}"
echo "[eval.sh] BACKBONE_TYPE=${BACKBONE_TYPE} BACKBONE_NAME=${BACKBONE_NAME} BACKBONE_PATH=${BACKBONE_PATH} EVA_CLIP_USE_XATTN=${EVA_CLIP_USE_XATTN}"
echo "[eval.sh] FINAL_SCORE_MODE=${FINAL_SCORE_MODE:-wti} LAMBDA_PROB=${LAMBDA_PROB:-0.0} LAMBDA_ANCHOR=${LAMBDA_ANCHOR:-0.0}"
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
  --max_frames 8 \
  --feature_framerate 1 \
  --batch_size_val 8 \
  --loose_type \
  --slice_framepos 3 \
  --n_video_embeddings 7 \
  --n_text_embeddings 7 \
  --uncertainty_text_head text \
  --log_sigma_min -1.5 \
  --log_sigma_max 4 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
