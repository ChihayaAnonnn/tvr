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

## The recipe, and what a profile now chooses

The recipe is `research_refs/UATVR_official/train.sh` and it is the same in
every profile: `lr 5e-5`, `coef_lr 1e-3`, `freeze_layer_num 0`,
`max_frames 12`, `max_words 32`, `slice_framepos 2`, batch 512, no
accumulation, 5 epochs, ViT-B/16, all 12 visual layers checkpointed.

A profile chooses the *data protocol*, nothing else (commit `7a93b9c`):

| | train videos | selects on | comparable to |
| --- | --- | --- | --- |
| `parity` | 9000 (`--fold_val_into_train`) | JSFUSION test | the literature |
| `hygiene` / `default` | 8500 | held-out 500 | our own arms |

Keeping a separate local recipe was what allowed the drift in the first
place, and it had no defender once measured. Both the launcher and
`validate_trusted_cli` pin the five-tuple for `hygiene` and `parity` alike,
so an RSPR arm and its baseline can differ only in RSPR. The eight arm
launchers under `scripts/` had `FREEZE_LAYER_NUM=8` pinned inline, which
would have overridden the shared recipe silently; those pins are gone.

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

## Result

`parity_a0_seed0`, RSPR off, 5 epochs, 35 min/epoch, 21 GB peak per rank.
Log `logs/20260728/parity_a0_seed0_133440_train_msrvtt.log`.

Per-epoch T2V R@1 on JSFUSION test:

| epoch | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| T2V R@1 | 46.6 | 47.7 | **48.9** | 47.3 | 47.3 |

Selected checkpoint (epoch 3):

| | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- |
| T2V | 48.9 | 74.4 | 82.2 | 2.0 | 13.5 |
| V2T | 48.3 | 75.1 | 84.6 | 2.0 | 8.9 |

**48.9 vs the published 49.6: a 0.7 gap, down from 3.2, and inside 1σ
(1.6pp).** The TI/DSA implementation does not need to be the next suspect.

> **Correction, later the same night.** The sentence that stood here said
> "the optimization recipe was the problem." That was wrong, and it was
> wrong in the way this document warns about: it credited the recipe for a
> change that also moved the protocol. The controlled comparison is below
> under *What the recipe was actually worth* — the recipe is worth −0.1
> R@1. The gap was the protocol.

### Read this number carefully

It is comparable to the literature and to nothing else. Both 48.9 and the
published 49.6 are a max over five epochs *on the reported test set*, so
they are like-for-like — but the local A0/A1 numbers (46.4, 46.8) were
selected on a held-out val set and evaluated on test once, which is the
honest protocol and a strictly harder one.

The premium test-set selection buys is visible in the epoch table: 48.9 at
the best epoch against 47.3 at the last, i.e. up to 1.6 points. So
"46.4 → 48.9" overstates what changed. The defensible claims are:

- against the literature, like-for-like: gap 3.2 → 0.7.
- against our own prior baseline: the recipe is worth something, but how
  much is not yet measured, because no run has used the fixed recipe under
  val selection.

Single seed. Two hardware deviations as described above.

## The own-protocol baseline

`scripts/run_baseline_hygiene_seed0.sh`, run id `hygiene_a0_seed0`. Same
recipe, trusted split, val selection, test touched once. Launched
2026-07-28 18:27; 332 steps per epoch, 26 GB per rank, ~33 min per epoch.

Per-epoch validation (500 held-out videos × 20 captions — not comparable to
any 1000×1 test number on this page):

| epoch | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| val T2V R@1 | 57.8 | **59.0** | 58.0 | 57.8 | 57.5 |
| val V2T R@1 | 86.6 | 87.0 | **87.8** | 86.2 | 84.6 |

Test, from the val-selected checkpoints (`final_test.json`):

| selected by | ckpt | T2V R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- |
| **val T2V (59.0)** | bin.1 | **46.3** | 75.4 | 84.0 | 2.0 | 13.1 |
| val V2T (87.8) | bin.2 | 46.9 | 74.2 | 83.8 | 2.0 | 14.3 |

| selected by | ckpt | V2T R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- |
| val T2V (59.0) | bin.1 | 47.0 | 75.7 | 84.4 | 2.0 | 8.4 |
| **val V2T (87.8)** | bin.2 | **47.3** | 74.6 | 83.5 | 2.0 | 9.2 |

**46.3 T2V R@1. This number, not 48.9, is what every RSPR arm must beat.**

### Provenance

Two caveats, neither affecting the number but both worth stating.

The run was interrupted after epoch 3 by a CUDA OOM that was not ours —
another user's vLLM job (26 GB) landed on a rank of the shared box while our
steady state was 26 GB of 40. Epochs 4–5 were recovered with `--resume_from`
(commit `db46433`) rather than retrained. Resume restores weights, optimizer
state, and the best-val trackers; it does not restore RNG or sampler state,
so this is a paused run rather than a bit-for-bit reproduction of an
uninterrupted one.

`best_validation_checkpoints.json` for epochs 1–3 was reconstructed by hand
from the log, because the code that writes it per epoch did not exist when
those epochs ran. Epochs 4–5 wrote it themselves. The reconstructed values
match the log lines they came from, and the selection they encode (bin.1 at
59.0) survived epochs 4–5 on its own merits — both scored lower.

The final test pass itself was lost to a `NameError` (`selection_payload`,
fixed in `841f254`) after all five checkpoints were already on disk, and was
recovered with `scripts/final_test_from_ckpt.sh` (commit `fd55ba3`). That
script reproduced 46.3 for bin.1 exactly, matching an earlier one-off eval of
the same checkpoint through a different code path.

## What the recipe was actually worth

The parity section above left this open: "the recipe is worth something, but
how much is not yet measured, because no run has used the fixed recipe under
val selection." It is measured now, and the answer is that it is worth
nothing.

`rspr_a0_seed0` and `hygiene_a0_seed0` are the same protocol — hygiene
profile, seed 0, RSPR off, 8500 train videos, val selection, test touched
once. The only difference between them is the recipe.

| | recipe | val T2V R@1 | **test T2V R@1** |
| --- | --- | --- | --- |
| `rspr_a0_seed0` | drifted (batch 256, freeze 8, 8 frames, lr 1e-4, TQFS) | 59.6 | **46.4** |
| `hygiene_a0_seed0` | official (batch 512, freeze 0, 12 frames, lr 5e-5, uniform) | 59.0 | **46.3** |

**−0.1 R@1.** The official recipe is not better than the drifted one on our
protocol; it is indistinguishable from it, and its val score is marginally
lower.

So the 3.2-point gap decomposes the other way round from what the parity run
suggested:

| | train videos | selects on | T2V R@1 |
| --- | --- | --- | --- |
| `hygiene_a0_seed0` | 8500 | held-out val | 46.3 |
| `parity_a0_seed0` | 9000 | JSFUSION test | 48.9 |
| published TI+DSA | 9000 | JSFUSION test | 49.6 |

Everything between 46.3 and 48.9 is protocol: 500 more training videos, and
a max over five epochs on the test set instead of one shot from a val-chosen
checkpoint. The parity epoch table bounds the second term at up to 1.6
points on its own (48.9 best vs 47.3 last). Splitting the 2.6 between the
two would cost five test evals of the existing hygiene checkpoints; it has
not been done, and doing it spends test-set information on a diagnostic.

### What this does to the prior RSPR results

The standing decision was that every RSPR number was void because the
baseline was crippled. That was over-broad.

All seven arms and their 46.4 baseline share `requested_effective_batch: 256`
in their manifests, and the launcher default plus the inline
`FREEZE_LAYER_NUM=8` pins (removed in `7a93b9c`) put the same freeze on all
of them. The arms were compared against a baseline trained the same wrong
way, so those comparisons were internally consistent. What the drift cost
was comparability to the literature — not internal validity.

Read that way, the arms already have their answer:

| arm | test T2V R@1 | vs 46.4 |
| --- | --- | --- |
| A1 | 47.0 | +0.6 |
| A3fixv3 | 47.0 | +0.6 |
| A1v3 | 46.8 | +0.4 |
| **A0 (baseline)** | **46.4** | — |
| A3fix | 46.0 | −0.4 |
| A3fixv2 | 45.8 | −0.6 |
| A0v2 | 45.3 | −1.1 |

Every one of these is inside 1σ (1.6pp) of the baseline. The spread is
consistent with seed noise around zero effect. Re-running them on the
official recipe will make them comparable to the literature, but the recipe
moved the baseline by 0.1, so there is no reason to expect it to move an arm
by more — and no reason to expect a re-run alone to turn +0.6 into a result.

## 2026-07-29: reading the arm manifests killed the table twice over

Two facts turned up in the manifests that the table above cannot survive.

### A0 and A0v2 are the same configuration, 1.1 R@1 apart

Diffing `ckpts/ckpt_msrvtt_rspr_a0_seed0/experiment_manifest.json` against
`ckpts/ckpt_msrvtt_rspr_a0v2_seed0/experiment_manifest.json` field by field
returns exactly two differences: the list of uncommitted paths, and
`rspr_recall_source` / `rspr_rerank_scale`. Both of those are rerank knobs,
and both runs are `rspr_mode: off`, where `rspr_eval` is False and the rerank
never executes. Same seed, same batch 256, same freeze 8, same lr.

| run | val T2V R@1 | test T2V R@1 |
| --- | --- | --- |
| `rspr_a0_seed0` | 59.6 | 46.4 |
| `rspr_a0v2_seed0` | 57.2 | 45.3 |

**Same-configuration reproduction error: 1.1 on test, 2.4 on val.** Not seed
variation — nondeterminism at fixed seed. (Caveat: both runs had dirty trees
including `modules/modeling.py`, so bit-identical code is not provable. The
RSPR files in those trees cannot matter — `off` sets `self.rspr = None` and
returns `dsa_loss` alone.)

Set that against what is being measured. The entire probabilistic apparatus
in the UATVR paper is worth +1.2 (TI+DSA 49.6 → full 50.8). The ruler's
graduation is the size of the object. Every entry in the arm table, largest
effect +0.6, is smaller than the pipeline's own reproduction error at n=1.

Resolving +0.6 to 2σ at σ≈1.6 needs ~28 seeds per arm. That is not a budget
question to be negotiated; it is outside the machine.

### Every non-`off` arm's test number went through the reranker

`--rspr_top_r` defaulted to 100, and `rspr_eval` is True for `mean` and
`stochastic`, so `_run_on_single_gpu` returned reranked matrices. No arm ever
set it to 0 — including `a1_seed0`, whose canonical definition in
`scripts/rspr_ablation_matrix.py` said `--rspr_top_r 0` while its manifest
says 100.

| arm | mode | `rspr_rerank_scale` | eval path |
| --- | --- | --- | --- |
| a0, a0v2 | off | — | clean DSA |
| a1, a3, a3fix | mean/stoch | absent | **defective rerank** |
| a1v3, a3fixv2, a3fixv3 | mean/stoch | `logit_scale` | fixed rerank |

So the arms do not measure the training-side auxiliary loss. They measure it
plus a rerank term, and the first generation's rerank added bare cosines to
logit-scale logits (see `rspr-rerank-scale-defect`). The two generations are
not comparable to each other, and neither is comparable to A0.

The one question the ablation was built to answer has never been asked.

### What was deleted, and why deletion was the fix

Reranking is the only path by which RSPR reached the inference ranking:
`forward()` returns `None` at eval, so the two distribution heads, the
matcher, `SoftContrastiveMatchLoss`, `StochasticRankLoss` and `anchor_kl` are
otherwise an auxiliary loss on the trunk and nothing else. That path is
measured and null — `u_pair` residual AUC below random, reranked score never
beating the λ=0 control — and it was contaminating every arm by default.

`70df3a4` removes it: `modules/rspr_rerank.py`,
`modules/rspr_score_analysis.py`, `scripts/analyze_rspr_scores.py`, the
eval-side plumbing in `main_task_retrieval.py`, and eight flags.
`c467b8d` removes the unimported `prob_models/probemb.py` (and the ruff
exemption that existed only to keep its dead branch off the F821 gate),
`prob_models/screening_utils.py`, and nine superseded per-arm launchers.
3434 lines net; suite 372 → 298 with the deleted tests.

`get_rspr_{text,video}_distribution` and `scripts/probe_rspr_uncertainty.py`
stay. Whether the variance channel can be made to carry information is still
open, and that script is how it gets measured.

## Next

1. **Run A4 (`--rspr_mode legacy`) once.** `ckpts/` contains no legacy run:
   UATVR's own DUA — `PIENet` + `UncertaintyModuleImage` + `MILNCE_BoF` +
   `KLdivergence`, the published +0.5-to-+1.2 — has never been measured under
   this protocol, while seven replacements for it have. It is the calibration
   the whole ablation was missing. If DUA also lands within ±1, the finding is
   that the apparatus cannot resolve effects of this size, which is a result
   about the experimental design and can be written as one. If it clears +0.5
   cleanly, the apparatus is fine and the RSPR design is what needs to change.
   One training run, ~3 hours, and it decides whether the core four
   components stay.
2. **Do not re-run the seven arms.** They buy comparability, not
   significance, and post-deletion they would be measuring something the
   originals did not measure anyway.
3. If the 46.3 → 48.9 decomposition matters for the writeup, five test evals
   of the existing `hygiene_a0_seed0` checkpoints settle it. Diagnostic
   only; it must not become a selection.
