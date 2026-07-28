# Baseline gap diagnosis: why local A0 sat 3.2 R@1 below the paper

Date: 2026-07-28. Commits `df36226`, `4fe418c`, `8da91e3`.

## The problem

Every RSPR arm was being compared against a local baseline that was itself
about three R@1 below the published number it was supposed to reproduce.

| run | MSR-VTT T2V R@1 |
| --- | --- |
| published TI | 48.4 |
| **published TI+DSA** | **49.6** |
| published TI+DSA+DUA† | 50.1 |
| published UATVR (full) | 50.8 |
| local A0 (hygiene profile) | 46.4 |
| local A1v3 | 46.8 |

Nothing downstream was interpretable. The entire probabilistic apparatus is
worth about +1.2 R@1 in the original paper, and 1σ on the 1000-query test
split is 1.6pp — so a 3.2-point deficit is wider than the effect being
measured, and any arm could beat or lose to the baseline for reasons that
have nothing to do with RSPR.

## Root causes

Three defects, found in this order. Each was hiding the next.

### 1. `freeze_layer_num` silently changed from 0 to 8

Commit `a9454b2` (2026-07-22) changed the launcher default
`FREEZE_LAYER_NUM=${FREEZE_LAYER_NUM:-0}` to `:-8`. With 8, resblocks 0–7 of
*both* CLIP towers stop training; only the top four move, and they move at
the trunk rate `lr * coef_lr`. The official recipe freezes nothing.

This is the term that dominates the gap. It went unnoticed for six days
because the run log printed 24 RSPR knobs and no optimization settings, and
the experiment manifest recorded the same. A diff of two manifests could not
show it.

The local recipe had drifted on four other axes too: batch 256 vs 512,
`max_frames` 8 vs 12, `slice_framepos` 3 (TQFS) vs 2 (uniform), `lr` 1e-4 vs
5e-5.

### 2. `--clip_gradient_checkpointing` had been a no-op for ten days

The 2026-07-18 cleanup commit `22d32b9` deleted
`configure_clip_gradient_checkpointing`, the only code that set
`grad_checkpointing` on the visual transformer. The flag was still parsed,
range-validated, logged, and written to the experiment manifest — the
forward pass just ignored it.

No run noticed, because defect 1 made every run cheap enough to fit anyway:
eight frozen resblocks, eight frames, 64 clips per rank. The parity recipe
(nothing frozen, 12 frames, 128 clips per rank) OOMed on all four ranks with
the non-checkpointed forward visible in the traceback.

This is the more instructive of the two. A validated, logged, manifest-
recorded flag that does nothing is *worse* than a missing flag: the
provenance record positively asserts a property the run did not have.

### 3. Two protocol guards blocked the parity data recipe

Not defects — the trusted-v1 protocol correctly refusing an unfamiliar
request — but both had to be answered before parity could run at all.

- `MSRVTT_TrainDataLoader` requires the train CSV to equal the manifest's
  8500 `train_video_ids`. Parity trains on all 9000.
- `dataloader_msrvtt_val` requires 20 contiguous captions per video. Parity
  selects on JSFUSION test, which has one.

## Fixes

| defect | fix | commit |
| --- | --- | --- |
| recipe drift invisible | `optimization` section in the experiment manifest; `freeze_layer_num`/`max_frames`/`max_words`/`slice_framepos` in the run log's `[Training]` line | `b5550af`, `53d8e56` |
| recipe drift itself | `parity` profile pinning the official recipe, enforced in both the launcher and `validate_trusted_cli` | `53d8e56` |
| 8500 vs 9000 train videos | `--fold_val_into_train`, parity-only in both directions | `df36226` |
| val loader shape | val loader takes test-set semantics when there is no held-out val | `4fe418c` |
| dead checkpointing flag | restored the wiring; the layer count is now logged | `8da91e3` |

## The parity profile

`scripts/run_baseline_parity_seed0.sh`. Reproduces
`research_refs/UATVR_official/train.sh`: `lr 5e-5`, `coef_lr 1e-3`,
`freeze_layer_num 0`, `max_frames 12`, `max_words 32`, `slice_framepos 2`,
batch 512, no accumulation, 5 epochs, ViT-B/16, full 9k train split,
JSFUSION test for selection.

Two deliberate deviations from the official script, both forced by hardware:

- 4×A800 at 128 clips per rank instead of 8 GPUs at 64. The contrastive
  batch is still 512.
- All 12 visual layers gradient-checkpointed to fit 40 GB. Compute cost, not
  a semantic change: 26 GB per rank, ~47 min per epoch.

`gradient_accumulation_steps` is pinned to 1 and the contract rejects
anything else. Accumulation restores the *optimizer* batch, not the
*contrastive* one — 256×2 trains a 256-way InfoNCE and is not parity.

## What a parity run is and is not

Parity selects checkpoints on the reported test set, exactly as the official
script does. That makes the number comparable to the literature and unusable
for our own claims. Every RSPR claim keeps using the trusted split with its
held-out 500 videos. `--fold_val_into_train` is rejected outside the parity
profile for this reason, and the manifest records which of the two a run was.

## Standing lesson

Three of the four problems here were provenance failures, not modelling
failures: a default that changed without appearing in any log, a flag that
was recorded as on while being off, and a comparison against a published
number that had never actually been reproduced locally.

The rule this suggests: a setting that is worth validating is worth
asserting the *effect* of. `--clip_gradient_checkpointing` was validated for
range and recorded in the manifest; what was missing was any check that it
changed the model. The restored helper returns its layer count and the run
log prints it, so the assertion is now about the outcome rather than the
input.

## Status

Parity baseline `parity_a0_seed0` launched 2026-07-28 13:34, RSPR off,
5 epochs, ~4 h. Target 49.6 T2V R@1.

Open: if parity lands materially short of 49.6, the remaining gap is not the
optimization recipe and the next suspect is the TI/DSA implementation
itself. If it lands at or near 49.6, re-run the RSPR arms on top of it —
every existing RSPR result was measured against the crippled baseline and
none of them carry over.
