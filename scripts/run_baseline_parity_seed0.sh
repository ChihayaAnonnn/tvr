#!/usr/bin/env bash
# Baseline repair: pure TI+DSA under the official UATVR recipe.
#
# Target: 49.6 T2V R@1 on MSR-VTT JSFUSION test, the published TI+DSA
# ablation number. The local hygiene baseline reaches 46.4 with the same
# model, so this run measures the optimization recipe alone -- RSPR is off.
#
# Nothing downstream is interpretable until this lands: the entire
# probabilistic apparatus is worth about +1.2 R@1 in the original paper,
# and 1 sigma on the 1000-query test split is 1.6pp.
#
# Note the parity protocol selects checkpoints on the reported test set,
# exactly as the official script does. Comparable to the literature; not a
# clean number for our own claims. Those keep using the trusted split.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-parity_a0_seed0}"

TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
EXPERIMENT_PROFILE=parity \
RSPR_MODE=off \
RUN_ID="${RUN_ID}" \
EXPERIMENT_DESC="official-parity TI+DSA baseline, target 49.6 T2V R@1" \
exec ./run_train_msrvtt_bg.sh
