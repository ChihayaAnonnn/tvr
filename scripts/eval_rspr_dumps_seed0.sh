#!/usr/bin/env bash
# Re-evaluate the seed-0 RSPR checkpoints on the test split under the *fixed*
# rerank defaults (deterministic recall + logit_scale alignment) and keep the
# score matrices.
#
# Two reasons this exists as one script rather than three ad-hoc commands:
#   * a1/a3fix's recorded final_test.json predate the rerank fix, so their
#     numbers are not comparable to a3fixv3's until they are recomputed here.
#   * The dumped matrices answer the uncertainty-AUC and permutation-control
#     questions offline, so they must come from the same eval configuration.
#
# Runs strictly sequentially on one GPU: a training job owns the other cards.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUMP_DIR="${ROOT_DIR}/logs/rspr_dumps"
mkdir -p "${DUMP_DIR}"

run_eval() {
    local tag="$1" checkpoint="$2" mode="$3" samples="$4"
    echo "=== ${tag}: ${checkpoint} (mode=${mode} K=${samples})"
    EVAL_SPLIT=test \
    RUN_ID="dump_${tag}" \
    INIT_MODEL="${ROOT_DIR}/${checkpoint}" \
    OUTPUT_DIR="${ROOT_DIR}/ckpts/eval_dump_${tag}" \
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}" \
    MASTER_PORT="${MASTER_PORT:-29561}" \
    RSPR_MODE="${mode}" \
    RSPR_SAMPLE_COUNT="${samples}" \
    RSPR_DUMP_SCORES="${DUMP_DIR}/${tag}_test.npz" \
    bash "${ROOT_DIR}/eval.sh"
}

run_eval a3fixv3 ckpts/ckpt_msrvtt_rspr_a3fixv3_seed0/pytorch_model.bin.1 stochastic 4
run_eval a3fix ckpts/ckpt_msrvtt_rspr_a3fix_seed0/pytorch_model.bin.2 stochastic 4
run_eval a1 ckpts/ckpt_msrvtt_rspr_a1_seed0/pytorch_model.bin.1 mean 1
