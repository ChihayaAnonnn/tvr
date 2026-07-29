#!/usr/bin/env bash
# A4 under the parity protocol: UATVR's own DUA, trained and selected exactly
# the way the paper does it.
#
# This completes a 2x2. The other three cells are measured:
#
#                   rspr_mode=off   rspr_mode=legacy
#   hygiene              46.3             46.5        (+0.2, McNemar p=0.885)
#   parity               48.9              ?
#
# hygiene trains on 8500 videos and picks the checkpoint on a held-out 500.
# parity trains on all 9000 and picks the checkpoint on JSFUSION test -- the
# reported set -- because that is what the official script does. So the row
# difference is the protocol and the column difference is the DUA, and this
# cell is the interaction.
#
# Under hygiene the DUA is not merely small, it is absent: 23 queries flip one
# way and 25 the other, the two models give the ground truth an identical rank
# on 651/1000 queries. The paper credits the same apparatus with +1.2. If
# parity recovers something near +1.2, the difference between the two rows is
# where the published gain lives, and the honest reading is that selecting on
# the test split harvests it. If parity is also flat, the component is inert
# and the whole DUA family -- theirs and ours -- comes out of the story.
#
# Either answer is worth having, which is why this is the only run queued.
# Parity numbers compare to the literature; they are not clean for our own
# claims, and nothing here should become a selection for anything else.
#
# See docs/experiments/baseline-gap-diagnosis.md.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-parity_a4_seed0}"

TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
EXPERIMENT_PROFILE=parity \
RSPR_MODE=legacy \
RUN_ID="${RUN_ID}" \
EXPERIMENT_DESC="A4 under parity: UATVR legacy DUA with the paper's own training scope and test-split selection, to locate the published +1.2" \
exec ./run_train_msrvtt_bg.sh
