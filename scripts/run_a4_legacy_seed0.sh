#!/usr/bin/env bash
# A4: UATVR's own DUA (--rspr_mode legacy), on the trusted split, official recipe.
#
# ckpts/ holds seven RSPR replacements for this component and zero measurements
# of the component itself. Every arm has been scored against A0 (rspr_mode=off,
# TI+DSA only); none against the thing RSPR is supposed to improve on.
#
# What this run decides is the ruler, not the arm. A0 vs A0v2 -- the same
# configuration twice -- landed 1.1 R@1 apart on test, which is wider than any
# effect in the arm table. The paper credits its entire probabilistic apparatus
# with +1.2 (TI+DSA 49.6 -> full UATVR 50.8). So:
#
#   A4 - A0 lands cleanly near +0.5..+1.2  => the pipeline can resolve effects
#                                             at this scale; RSPR's null result
#                                             is about RSPR's design.
#   A4 - A0 lands inside +-1               => the pipeline cannot resolve effects
#                                             at this scale at all, and no amount
#                                             of arm-tuning was ever going to show
#                                             one. That is a conclusion about the
#                                             experimental design, and it is
#                                             publishable as such.
#
# Everything except --rspr_mode matches scripts/run_baseline_hygiene_seed0.sh, so
# the difference is the DUA heads and their two losses (MILNCELoss_BoF at 1e-2,
# KLdivergence at 1e-4, added to dsa_loss) and nothing else. The sample counts
# n_video_embeddings / n_text_embeddings default to 7, as in the official
# train.sh. (--strategy is dead here -- argparse accepts it, no code reads it.)
#
# See docs/experiments/baseline-gap-diagnosis.md.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-hygiene_a4_seed0}"

TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
EXPERIMENT_PROFILE=hygiene \
RSPR_MODE=legacy \
RUN_ID="${RUN_ID}" \
EXPERIMENT_DESC="A4: UATVR legacy DUA under the trusted-v1 protocol, to calibrate whether the pipeline resolves a +0.5..+1.2 effect" \
exec ./run_train_msrvtt_bg.sh
