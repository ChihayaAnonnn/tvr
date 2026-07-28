#!/usr/bin/env bash
# The baseline every RSPR arm must beat: pure TI+DSA, official recipe,
# our own data protocol.
#
# The parity run reached 48.9 T2V R@1 against a published 49.6, which closed
# the recipe question but produced a number selected on the reported test set
# -- comparable to the literature and to nothing of ours. This run uses the
# same optimizer settings on the trusted-v1 split: 8500 train videos, the
# held-out 500 selecting the checkpoint, test touched once at the end.
#
# Its number replaces the 46.4 that every prior RSPR arm was compared against.
# Nothing measured against that old baseline carries over.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-hygiene_a0_seed0}"

TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
EXPERIMENT_PROFILE=hygiene \
RSPR_MODE=off \
RUN_ID="${RUN_ID}" \
EXPERIMENT_DESC="official-recipe TI+DSA baseline on the trusted split" \
exec ./run_train_msrvtt_bg.sh
