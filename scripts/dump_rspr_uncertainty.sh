#!/usr/bin/env bash
# Dump per-query RSPR uncertainty for one checkpoint on the test split.
#
# The eval path writes similarity matrices (--dump_sim_matrix) but nothing
# about the distribution heads, so the stratifier experiment had no sigma to
# work with. This runs scripts/probe_rspr_uncertainty.py, which already encodes
# the split and computes sigma^2 and U_pair, and adds --dump_uncertainty so the
# per-query values land in an npz next to the similarity matrices.
#
# The model flags below must match the ones the checkpoint was trained with;
# they are copied from final_test_from_ckpt.sh, which is what produced the
# matrices these values have to align with. RSPR_MODE has no default on
# purpose -- a checkpoint trained with legacy heads scored under `off` would
# load a model with no distribution heads and fail late.
#
#   CKPT=ckpts/ckpt_msrvtt_parity_a4_seed0/pytorch_model.bin.2 \
#   OUT=.scratch/unc/parity_a4.npz RSPR_MODE=legacy ./scripts/dump_rspr_uncertainty.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT="${CKPT:?set CKPT to the checkpoint file to probe}"
OUT="${OUT:?set OUT to the .npz path to write}"
RSPR_MODE="${RSPR_MODE:?set RSPR_MODE to the mode the checkpoint was trained with}"
GPU_ID="${GPU_ID:-0}"

DATA_PATH=/data2/hxj/data/MSRVTT
SCRATCH_OUT="$(mktemp -d)"
mkdir -p "$(dirname "${OUT}")"

echo "[dump_rspr_uncertainty] ckpt=${CKPT} mode=${RSPR_MODE} gpu=${GPU_ID} -> ${OUT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
CLIP_CACHE_DIR="${ROOT_DIR}/.cache" \
OMP_NUM_THREADS=1 \
MASTER_ADDR=127.0.0.11 \
MASTER_PORT="${MASTER_PORT:-29573}" \
RANK=0 WORLD_SIZE=1 \
/home/xujie/.conda/envs/tvr/bin/python \
    "${ROOT_DIR}/scripts/probe_rspr_uncertainty.py" \
    --init_model "${CKPT}" \
    --dump_uncertainty "${OUT}" \
    --do_eval \
    --num_thread_reader 4 \
    --prefetch_factor 2 \
    --batch_size 512 \
    --batch_size_val 16 \
    --train_csv "${ROOT_DIR}/data/generated/msrvtt_trusted_v1/train.csv" \
    --val_csv "${ROOT_DIR}/data/generated/msrvtt_trusted_v1/val.csv" \
    --source_train_csv "${DATA_PATH}/csv/MSRVTT_train.9k.csv" \
    --test_csv "${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" \
    --split_manifest "${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed0.json" \
    --eval_split test \
    --data_path "${DATA_PATH}/annotation/MSRVTT_v2.json" \
    --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
    --output_dir "${SCRATCH_OUT}" \
    --max_words 32 \
    --max_frames 12 \
    --datatype msrvtt \
    --expand_msrvtt_sentences \
    --feature_framerate 1 \
    --slice_framepos 2 \
    --linear_patch 2d \
    --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --clip_layer_norm_precision fp16 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --rspr_mode "${RSPR_MODE}"

rm -rf "${SCRATCH_OUT}"
