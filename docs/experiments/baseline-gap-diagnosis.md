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

## 2026-07-29: A4 ran, and UATVR's own DUA is a null here too

`hygiene_a4_seed0`, `--rspr_mode legacy`, seed 0, five epochs, 02:21–06:08.
Manifest diff against `hygiene_a0_seed0` is `rspr_mode: off → legacy` and
nothing else behavioral. Both runs select epoch 2 on held-out val.

| test, T2V-selected ckpt | A0 (`off`) | A4 (`legacy`) | Δ | McNemar p |
| --- | --- | --- | --- | --- |
| T2V R@1 | 46.3 | 46.5 | +0.2 | 0.885 |
| T2V R@5 | 75.4 | 75.7 | +0.3 | 0.749 |
| T2V R@10 | 84.0 | 83.7 | −0.3 | 0.678 |
| T2V MnR | 13.12 | 13.27 | −0.15 | — |
| V2T R@1 | 47.0 | 47.9 | +0.9 | 0.306 |
| V2T R@5 | 75.7 | 75.9 | +0.2 | 0.871 |
| V2T R@10 | 84.4 | 84.8 | +0.4 | 0.572 |

Per-epoch val T2V R@1, A0 vs A4: 57.8/58.0, 59.0/58.8, 58.0/58.4, 57.8/57.8,
57.5/57.7. The two curves track within 0.4 at every epoch.

### The ruler is finer than the ±1.1 floor suggested, and the effect is still zero

Two checks change how the +0.2 should be read.

**Eval is deterministic, so the 1.1 floor is entirely a training-side
quantity.** Re-scoring `hygiene_a0_seed0`'s saved checkpoints under the
current post-deletion code reproduces all eight test numbers to the decimal
(46.3/75.4/84.0/13.1, 47.0/75.7/84.4/8.4, and both V2T-selected numbers).
That also proves the A+B deletions are eval-neutral, so A0 and A4 are
comparable despite running on different commits.

**A0 and A4 are the same two models on the same 1000 queries, so the
comparison is paired and does not need the cross-run floor.** Exact McNemar
on T2V R@1: 23 queries A0 gets and A4 misses, 25 the other way, p = 0.885.
Not one of the six paired tests reaches p < 0.3. The two models assign the
ground-truth video an *identical* rank on 651/1000 queries; mean |Δrank| is
2.52. This is not an effect the instrument failed to resolve — it is the
absence of an effect, measured tightly.

### The heads did train

Not a dead branch. A4's checkpoint carries 24 tensors A0's does not
(`pie_net_{text,video}`, `uncertain_net_{text,video}`), the optimizer's head
group goes 58 → 82 params at lr 5e-5, and over epochs 2–5 the uncertainty
nets drift 30–42 % in relative weight norm with biases moving ~2×. Training
loss sits ~0.6–0.8 above A0's throughout (2.68→1.36 vs 1.89→0.78), which is
the MILNCE + KL terms being optimized. They are optimized; they just do not
move retrieval.

### What this settles

The launcher's decision rule offered two outcomes: DUA clears +0.5 (ruler is
fine, RSPR's design is at fault) or DUA lands inside ±1 (ruler cannot resolve
this scale). The answer is a third one. The ruler is fine — better than
assumed, once you stop comparing across retrainings — and **the published
component is a null under this protocol.**

So the seven RSPR arms were not failing to reproduce a working baseline
component. There is no working baseline component here to reproduce. Their
null results were correct measurements of a real null, and the thing that now
needs explaining is the paper's +1.2, not our +0.2.

## 2026-07-29: the 2×2 closes — the DUA is inert under the paper's protocol too

`parity_a4_seed0`, 09:13–12:16, five epochs, no resume and no OOM. Manifest
diff against `parity_a0_seed0` is `rspr_mode: off → legacy` and nothing else
behavioral. Both select epoch 3.

**Test T2V R@1, MSR-VTT JSFUSION 1000×1000:**

| | `off` | `legacy` (DUA) | Δ | McNemar p |
| --- | --- | --- | --- | --- |
| **hygiene** — 8500 train, held-out-500 selection | 46.3 | 46.5 | +0.2 | 0.885 |
| **parity** — 9000 train, JSFUSION-test selection | 48.9 | 49.3 | +0.4 | 0.712 |

Parity does not recover the published +1.2. Of the six paired tests in the
parity row, none reaches p < 0.32 and three are exactly p = 1.000 (T2V R@5
26 vs 27 discordant, T2V R@10 16 vs 16, V2T R@1 41 vs 40). The two models
give the ground truth an identical rank on 605/1000 queries.

So the answer is the second branch: **the component is inert under both
protocols.** We reproduce UATVR's baseline — 48.9 against the published
TI+DSA 49.6 — and then fail to reproduce its probabilistic gain, +0.4 where
the paper reports +1.2.

### What selection-on-test is worth, measured directly

Parity scores every epoch on the reported set, so the premium is visible:

| | epoch-by-epoch test T2V R@1 | mean | reported (max) | premium |
| --- | --- | --- | --- | --- |
| parity A0 | 46.6 47.7 **48.9** 47.3 47.3 | 47.56 | 48.9 | +1.34 |
| parity A4 | 46.6 46.6 **49.3** 48.3 47.9 | 47.74 | 49.3 | +1.56 |

Taking the best of five epochs on the test set is worth about +1.4 — more
than the +1.2 the paper credits to its entire probabilistic apparatus. And
the DUA's own effect shrinks from +0.40 (max-over-epochs, as reported) to
+0.18 (epoch mean), which is the hygiene number to two decimals.

State this carefully: max-minus-mean overstates the premium, because the max
of five draws exceeds their mean by construction even with no selection
effect, and the epochs are a learning curve rather than iid draws. It is not
a clean estimate of "how much cheating buys you." What it does establish is
that epoch-to-epoch spread on the reported set is of the same order as the
published effect — so a protocol that picks the best epoch on that set cannot
separate the two. The paired A0-vs-A4 comparison is unaffected by any of
this, and it is flat.

### Where this leaves the story

Four cells, two protocols, one component, and the largest effect anywhere is
+0.4 at p = 0.71. Combined with the seven RSPR arms, every version of "add a
probabilistic head to TI+DSA" measured in this repository is a null. That is
now a finding with a real denominator behind it rather than a failure to get
a method working.

## 2026-07-30: FIRE's judgments land on our exact test grid, and the labels are worse than the noise

FIRE (Rodriguez et al., EACL 2023, arXiv:2210.05038) had humans judge
caption-video pairs pooled from CLIP4CLIP, SSB and CE, and released them. The
release is not linked from the arXiv landing page; the address is in a first-page
footnote, `pedro.ai/multimodal-retrieval-evaluation`, pointing at the archived
`facebookresearch/mm-retrieval-evaluation` (Git LFS, CC BY-NC-SA 4.0 in the repo
LICENSE but CC BY-NC 2.0 in the JSON metadata — the two disagree).

The MSR-VTT half lands on **exactly the JSFUSION 1K test grid we score on**:
999/1000 videos, 995/995 captions, zero judgments outside the grid. No feature
re-extraction, no split surgery, no retraining. 24,167 judgments, mean 24.5
videos judged per caption.

What the judgments say about the benchmark:

| | |
| --- | --- |
| genuinely new positives | 2,147 (2,855 relevant judgments, 708 merely confirm the original pair) |
| captions with ≥1 extra correct video | 887 / 995 |
| mean positives per caption | 1.00 → 3.18 |
| **original ground-truth pairs judged irrelevant** | **219 of the 927 judged = 23.6%** |

That last row is not FIRE's headline and is worth separating from it. The
false-negative problem is that the benchmark calls correct answers wrong. This
is the converse: on roughly a quarter of captions an annotator says the video
the benchmark *demands* does not match. We keep those relevant anyway — deleting
ground truth would change what the benchmark is rather than how it is scored —
so none of the correction below is driven by it.

### Re-scoring our four checkpoints

`--dump_sim_matrix` was added to `eval_epoch`, writing the same matrix that is
handed to `compute_metrics`, so an offline recompute has to reproduce the logged
number. All four did, to the decimal: 48.9 / 49.3 / 46.3 / 46.5. Scored with
`scripts/fire_corrected_metrics.py`:

| run | orig R@1 | FIRE R@1 | condensed R@1 | nDCG@10 | mAP | bpref | judged@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parity A0 | 48.9 | 70.1 | 74.5 | 70.9 | 65.0 | 66.3 | 55.9% |
| parity A4 | 49.3 | 69.7 | 74.8 | 70.8 | 64.9 | 65.9 | 56.0% |
| hygiene A0 | 46.3 | 67.3 | 71.8 | 70.8 | 64.5 | 64.7 | 57.4% |
| hygiene A4 | 46.5 | 68.2 | 72.6 | 71.0 | 64.6 | 64.5 | 57.1% |

**The correction is +21 R@1.** FIRE reported up to +25 for 2022-era models;
ours sits just under that, which is what a stronger model on the same pool
should look like. FIRE R@10 is 94–95 and condensed R@10 is 99.4–99.7: at rank 10
this benchmark is saturated, exactly as FIRE said in 2023.

### The pooling caveat, measured rather than asserted

FIRE judged ~24 of 1000 videos per caption, pooled from three 2022 models. Our
backbone is stronger than all three, so pairs we rank highly that they missed
are unjudged, not known-irrelevant, and scoring unjudged as irrelevant biases
against us. Depth of coverage over *our* ranking:

    judged@1  90–92%     judged@5  71–73%     judged@10  56–57%

So the pool is trustworthy at rank 1 and half-blind by rank 10. FIRE R@10 is a
lower bound, condensed R@10 an upper bound, and the truth is between. Since R@1
is what the literature competes on, this is usable where it matters.

### The DUA null survives the correction

The obvious hypothesis was that bad labels were hiding the effect: with 89% of
captions having an extra correct answer, a model putting an unlabelled-correct
video first is scored wrong. Paired McNemar on per-query hits:

| | original R@1 | FIRE R@1 | condensed R@1 |
| --- | --- | --- | --- |
| parity A0→A4 | +0.40 (p=0.712) | **−0.40** (p=0.744) | +0.30 (p=0.795) |
| hygiene A0→A4 | +0.20 (p=0.885) | +0.90 (p=0.306) | +0.80 (p=0.322) |

Nothing reaches p<0.3 under any labelling, and the two protocols disagree in
sign once the labels are corrected. The DUA was not being masked by bad ground
truth. It is absent.

### What this changes

Three error sources are now measured on the same test split, and each is larger
than the effects the literature reports (+0.5 to +1.5 R@1):

| source | size |
| --- | --- |
| label error (original vs FIRE R@1) | **21 points** |
| max-over-epochs selection on the reported set | ~1.4 |
| same-config retraining floor | 1.1 |

The first is 14–40× the effect sizes being competed over. Switching to another
TVR benchmark does not help: MSVD, VATEX, DiDeMo and ActivityNet are all
repurposed captioning datasets with the identical one-positive construction, and
none of them has human relevance judgments. MSR-VTT + FIRE is now the most
trustworthy TVR test set available, precisely because somebody did the
annotation. The move is to stay on this grid and change the labels and the
metrics, not the dataset.

## 2026-07-31: one conformal threshold does not fit every query, and the free margin cannot tell which

Motivated by DAB (arXiv:2607.20984, ECCV 2026). Reading its §4.4 corrects
something recorded earlier here: DAB *does* do selective prediction. It reports
a risk–coverage curve and AURC 0.429 → 0.256 for R@1, using the top1–top2
**KL margin** as the confidence signal. Two things about that are worth
testing rather than asserting.

First, the confidence signal is a margin over the ranked list, not the learned
variance — the paper says so itself ("emerges from the bridge-induced
distributional gap rather than from text variance alone"). Any deterministic
model has a top1–top2 margin for free. Second, the comparison is against
*random ordering*, which any non-zero signal beats, and the risk is defined
against MSR-VTT's single positive, which we now know is wrong 21 points of the
time.

`scripts/conformal_coverage_probe.py`, run on the four already-dumped
similarity matrices with FIRE's judgments. Split conformal, α=0.1, 200 random
calibration/test halves. Two set constructions: an **adaptive margin set**
`{v : s_top1 − s_v ≤ λ}`, whose size varies per query, and a **fixed top-k
set**, whose size does not. Two coverage targets: `any` (the set holds at least
one relevant video) and `all` (it holds every judged relevant video); they fail
in opposite directions, so a claim that only holds for one is an artefact of
the target.

**The adaptive margin set is worth nothing over a constant k.** At a matched
90% target:

| run | adaptive mean size | fixed-k size | adaptive vs fixed-k |
| --- | --- | --- | --- |
| parity A0 | 5.4 | 5.4 | +0.8% |
| parity A4 | 5.4 | 5.5 | +2.2% |
| hygiene A0 | 5.0 | 5.0 | +0.5% |
| hygiene A4 | 5.1 | 4.8 | −5.5% |

Three of four are the same size or *larger* than always returning the same
number of videos. The comparison is if anything generous to the adaptive set:
rank discreteness makes fixed-k land at 90.6–91.1% coverage against the
adaptive set's 89.6–89.8%, so fixed-k is buying that size at a full point more
coverage. The free margin carries some global ranking signal (see AURC below)
but not enough per-query information to size a set.

(Checkpoint note: the hygiene runs selected different epochs for the two
directions, bin.1 for t2v and bin.2 for v2t, so everything here reads the
t2v-selected bin.1. Parity selected bin.2 for both.)

**One global threshold gives 90% coverage on average and misses badly
everywhere.** Coverage by the true number of relevant videos (parity A0;
all four runs agree to ~1pt):

| |Rel| | 1 | 2 | 3 | 4–5 | 6+ | spread |
| --- | --- | --- | --- | --- | --- | --- |
| `any` coverage | 83.3% | 88.4% | 92.8% | 95.5% | 98.3% | **15.0 pt** |
| `all` coverage | 98.6% | 83.7% | 81.4% | 84.4% | 88.9% | **17.2 pt** |
| `any`, oracle Mondrian | 90.1% | 90.3% | 90.2% | 90.4% | 90.7% | 0.7 pt |

Marginal validity holds exactly as the theory says (89.8% at α=0.1) and is
uninformative: the queries with one correct answer — the ones a user is most
likely to have a specific target for — are covered 83% of the time, seven
points under the advertised rate, while the ambiguous ones are over-covered to
98%. `all` breaks the other way, which rules out the target being the cause.

**The problem is fixable in principle and not fixable with what is free.**
Calibrating one threshold per *oracle* ambiguity stratum collapses the spread
to 0.4–1.2 pt across all four runs, at no cost in set size. But the oracle
conditions on the label it is supposed to be robust to.
The deployable version buckets by a label-free ambiguity score — quintiles of
(top1–top2 gap rank + caption-length rank), boundaries from the calibration
half only:

| run | global spread | predicted-stratum Mondrian | oracle Mondrian |
| --- | --- | --- | --- |
| parity A0 | 15.0 pt | 13.3 pt | 0.7 pt |
| parity A4 | 15.1 pt | 13.2 pt | 0.4 pt |
| hygiene A0 | 16.0 pt | 14.1 pt | 1.2 pt |
| hygiene A4 | 15.7 pt | 14.8 pt | 1.2 pt |

Ambiguity *is* predictable in the weak statistical sense — Spearman ρ against
|Rel| is −0.41 for the top1–top2 gap, −0.38 for caption length, −0.22 for the
top-1 score, all far past significance, all pointing the same way (a flat,
low-scoring, short query is the ambiguous one). It is nowhere near predictable
enough to matter: ρ=0.41 closes 6–13% of the gap. **The distance between
13.3 pt and 0.7 pt is the open problem**, and it is the first place in this
project where a learned uncertainty head has a target that is measurable, has
ground truth, and is not R@1.

**The label correction moves the absolute risk a lot and DAB's claim not at
all.** Recomputing DAB's own curve with our free margin:

| labels | AURC | random ordering | reduction |
| --- | --- | --- | --- |
| original | 0.279 | 0.511 | 45.4% |
| FIRE | 0.162 | 0.299 | 45.7% |

Absolute risk nearly halves, the relative reduction does not budge. So the
hypothesis that label error corrupts risk–coverage conclusions is **not**
supported — worth recording, because it was the third motivation and it failed.
What survives is the comparison DAB skipped: a raw cosine top1–top2 gap, with
no probabilistic machinery anywhere, cuts AURC 45–49% against random, versus
the 40% DAB reports for its bridge-induced KL margin. Different backbones and
different R@1 (52.6 vs our 48.9), so this is not a like-for-like win — but it
does mean DAB has not shown its distributional apparatus beats the free signal,
because it never ran that arm.

## 2026-07-31: σ knows something about ambiguity, and it does not help

The gate above left one hypothesis alive and it was the only one that could
have redeemed the DUA work: σ failed as a *re-ranker*, but stratification is a
much weaker requirement, so maybe σ works as a *stratifier*. A4 was trained
with UATVR's own DUA, whose only per-query uncertainty is the two probabilistic
heads' log-variance, and nothing exposed it — `_loose_similarity` computes it
on every eval call and throws it away. `Model.get_legacy_logsigma` plus
`scripts/dump_rspr_uncertainty.sh` now dump it per query.

Alignment with the similarity matrices is proven rather than assumed: the
probe's own deterministic matrix matches the eval dump to 1.3e-5 with argmax
and diagonal-rank agreement 1.0, and the resulting Top-1 reproduces the logged
49.3 / 46.5 exactly.

**σ is the single strongest ambiguity feature, and its sign is backwards.**

| feature | ρ vs \|Rel\|, parity A4 | ρ, hygiene A4 |
| --- | --- | --- |
| text σ² | **−0.506** | **−0.485** |
| top1–top2 gap | −0.407 | −0.391 |
| caption length | −0.376 | −0.376 |
| top-1 score | −0.191 | −0.202 |
| video σ² | −0.010 (p=0.75) | +0.019 (p=0.56) |

Every free feature says "flat, low-scoring, short query → many relevant
videos". Text σ² says the opposite: **larger variance goes with *fewer* correct
answers**. Whatever the head learned, it is not the ambiguity the word
"uncertainty" is meant to name. Video σ² is pure noise. So the head that this
project spent months on has one informative output out of two, pointed the
wrong way.

Because a rank-sum assumes every feature agrees on which direction is
"confident", it cancels: `sigma only` 16.0 pt and `gap+len+sigma` 15.5 pt,
both worse than `gap+len`'s 13.2 pt and worse than doing nothing. The fair
test is to let the data set the weights and the signs, which is what the
method would do anyway — it already assumes FIRE labels on a calibration
portion. A least-squares fit of the features onto |Rel| on the calibration
half of each of the 200 splits, bucketed by quintiles of the prediction:

| predictor (parity A4) | ρ vs \|Rel\| | spread | mean size | Δ spread vs `sup gap+len` |
| --- | --- | --- | --- | --- |
| `sup gap+len` | −0.487 | **13.0 pt** | 4.9 | — |
| `sup sigma only` | −0.505 | 15.0 pt | 6.1 | +1.72 ± 0.16 |
| `sup gap+len+sigma` | **−0.553** | 13.8 pt | 5.0 | +0.70 ± 0.12 |

hygiene A4 agrees: 14.8 / 15.4 / 15.2 pt, with σ costing +0.36 ± 0.14 pt.
The Δ column is paired — every predictor sees the same 200 splits — so ±0.12
is the real error bar and this is one of the few effects in this project that
clears its noise floor. It clears it in the wrong direction.

**The dissociation is the finding.** Adding σ makes the predictor measurably
*better* at predicting how many videos are relevant (ρ −0.487 → −0.553, both
runs) and measurably *worse* at equalising coverage. Predicting |Rel| is
therefore not the right objective for a stratifier. Refitting on the conformal
score itself instead of on |Rel| tests that directly and does not rescue it
either: `mgn gap+len` 13.9 / 14.8 pt, `mgn gap+len+sigma` 13.6 / 15.2 pt.

**σ is dead as a stratifier too.** That closes the last route by which the
learned-variance work could have been salvaged. Every arm of it is now
measured: null as a re-ranker, anti-informative as an error predictor
(AUROC 0.32, below chance), and negative as a stratifier.

### How good would an ambiguity predictor have to be?

Every real predictor lands at 13–15 pt while the oracle sits at 0.3–1.3 pt,
and neither adding σ nor changing the regression target moves it. That makes
the useful question quantitative. Corrupting the true |Rel| with increasing
noise traces spread against predictor quality (5 noise draws per level, all
four runs):

| ρ vs \|Rel\| | parity A0 | parity A4 | hygiene A0 | hygiene A4 |
| --- | --- | --- | --- | --- |
| 1.000 | 0.7 pt | 0.3 pt | 1.3 pt | 1.1 pt |
| 0.965 | 1.5 | 1.6 | 4.1 | 1.5 |
| 0.940 | 3.5 | 3.5 | 4.4 | 4.5 |
| 0.825 | 5.9 | 6.1 | 7.2 | 6.7 |
| 0.578 | 10.6 | 10.1 | 10.3 | 10.8 |
| 0.330 | 13.4 | 13.3 | 14.1 | 14.0 |

The curve is steep and the useful region starts late: **ρ ≈ 0.83 to halve the
gap, ρ ≈ 0.94 to close it to a few points.** The best signal available inside
a trained TVR model, including its learned variance, reaches ρ = 0.55. This
is the target number for the direction, and it is a target no published TVR
uncertainty method is anywhere near.

One caveat that cuts against the real predictors: `sup gap+len+sigma` reaches
ρ = 0.553 but gives 13.8 pt, while a noised oracle at ρ = 0.578 gives 10.1 pt.
A real predictor is *worse* than random noise of the same rank correlation,
so its errors are structured — it is wrong about the same queries the
threshold is already wrong about. Rank correlation therefore overstates how
useful a predictor will be, and any future ambiguity model has to be scored
on spread directly, not on ρ.

## 2026-07-31: the objective was wrong, but fixing it buys 2.5 points, not 7

Two things changed at once against the bucketed baseline, and the 2×2 says
which one paid. The objective: instead of predicting |Rel| and cutting the
prediction into quintile buckets, regress the conditional quantile of the
conformal score directly and conformalize the residual (CQR — Romano,
Patterson & Candès, NeurIPS 2019), which needs no |Rel| label at all. The
features: instead of the top of the ranking (gap12, caption length, σ²), read
the whole score row — gaps from rank 1 to ranks 2/3/5/10/50/100/500, near-tie
counts within a quarter/half/one standard deviation of the top, softmax
entropy at two gallery-scaled temperatures, row mean and standard deviation,
and the top-1 z-score. Seventeen columns, nineteen with σ².
`scripts/conditional_conformal_probe.py`, 200 splits, α = 0.1.

A note on which "spread" is being quoted, because two different numbers have
gone by that name. Averaging the five per-stratum coverages over splits and
subtracting afterwards gives the number the gate experiment reports (13.8,
oracle 0.7). Taking max-minus-min inside each split and averaging that gives
something far larger, because a max of five noisy estimates is biased upward —
the oracle's true 0.7 comes out near 9. That bias is a floor both methods sit
on, so it compresses differences: the real 13.1-point baseline-to-oracle gap
reads as 4.1. The within-split version was used earlier only because it had an
obvious standard error. The table below puts the error bar on the right scale
instead, by resampling the 200 splits with replacement using the same
resampled indices for every method.

parity_a0 (the other three runs agree; full table in `.scratch/cqr_probe.txt`):

| method | 1 | 2 | 3 | 4-5 | 6+ | spread | size | med | p90 | Δ spread |
|---|---|---|---|---|---|---|---|---|---|---|
| global | 83.3 | 88.4 | 92.8 | 95.5 | 98.3 | 15.0 | 5.4 | 2.9 | 12.1 | +1.20 ± 0.12 |
| bucket gap+len | 83.7 | 89.4 | 94.1 | 94.8 | 97.5 | 13.8 | 5.5 | 3.0 | 12.5 | — |
| bucket rich | 83.9 | 89.4 | 93.6 | 94.1 | 96.8 | 12.9 | 4.8 | 3.2 | 10.6 | −0.88 ± 0.14 |
| CQR gap+len | 83.0 | 89.3 | 93.5 | 95.6 | 98.2 | 15.2 | 6.1 | 3.0 | 14.2 | +1.37 ± 0.18 |
| CQR rich | 83.8 | 90.7 | 95.3 | 95.1 | 93.8 | 11.5 | 7.9 | 3.2 | 17.3 | −2.26 ± 0.26 |
| CQR gbr rich | 83.1 | 88.7 | 94.2 | 95.1 | 98.3 | 15.1 | 7.2 | 2.8 | 15.7 | +1.35 ± 0.16 |
| oracle \|Rel\| | 90.1 | 90.3 | 90.2 | 90.4 | 90.7 | 0.7 | 5.5 | 2.6 | 12.7 | −12.85 ± 0.32 |

Δ spread across the four runs — CQR rich: −2.26, −2.34, −2.65, −2.71. bucket
rich: −0.88, −1.01, −2.03, −1.51. CQR gap+len: +1.37, +1.43, +0.75, +1.17.
Marginal coverage is 89.8–90.1% for every method, as it must be.

Reading the 2×2: the new objective **on its own makes things worse**. A
two-feature linear quantile regression is beaten by five buckets, which can
express a monotone nonlinearity that two coefficients cannot. The new features
on their own help a little. Together they are more than the sum, which is the
signature of a linear fit that only becomes worth having once it has enough
columns to work with. Gradient boosting at the same objective is bad
everywhere: 250 fitting rows is not enough for 100 trees, and it degenerates
to roughly the global threshold.

The pre-registered criterion was ≤ −2.0 pt paired on at least three of four
runs, with a rider that a spread reduction bought at more than 1.5× the mean
set size does not count. CQR rich meets the first on 4/4 and fails the second
on 3/4: mean size ratios 1.44, 1.51, 1.51, 1.56. The p90 confirms this is not
a handful of pathological queries — the whole upper tail is 40–50% fatter
(17.3 vs 12.5 on parity_a0). The median barely moves (3.2 vs 3.0), so a
typical user would not notice, but the average over a corpus is the honest
statistic and it went up. **Verdict: the hypothesis is half right and the
direction is not rescued.** Halving the 13-point gap needs ≈7 pt; the best
label-free method buys 2.5 and charges for it.

The one method that is strictly better than the baseline on both axes is
`bucket rich`: −0.88 to −2.03 spread and *smaller* sets everywhere (mean 4.7–4.8
vs 5.3–5.7, p90 10.0–10.7 vs 11.5–13.1). It is the new baseline to beat.

### What the strata actually ask for, and why it is backwards

Fitting one threshold per true stratum on all 1000 queries of parity_a0:

| \|Rel\| | n | oracle λ | mean set | median set |
|---|---|---|---|---|
| 1 | 363 | 4.035 | 8.2 | 3 |
| 2 | 206 | 2.585 | 4.5 | 2 |
| 3 | 124 | 1.716 | 2.9 | 2 |
| 4-5 | 126 | 1.521 | 3.4 | 3 |
| 6+ | 181 | 0.943 | 3.5 | 3 |

The required threshold falls monotonically and steeply as |Rel| rises, and the
sets that conditional coverage demands are **largest for the most specific
queries**. This is not a quirk; it follows from the target. `any` coverage asks
for one relevant video in the set, and a query with six right answers gets that
from the top few almost for free, while a query with exactly one right answer
needs a wide net. So the intuitive story — "return more results when the query
is vague" — is the wrong way round under `any`. It is the right way round under
`all` (cover every judged relevant video), which fails in the opposite
direction. Any writeup has to pick a target and say which, or report both and
own the tension; describing the method as "bigger sets for ambiguous queries"
without saying which target is simply false half the time.

This also explains why every deployable method above is stuck. All of them
improve the 6+ end (97.5 → 93.8) and none of them move the |Rel|=1 end (83.7 →
83.8, against the oracle's 90.1). They are trimming over-coverage where it is
cheap, not fixing under-coverage where it hurts.

### There is signal in the bottom stratum; nothing is using it

Spearman of each feature against the conformal score, over all 1000 queries
versus inside the 363 |Rel|=1 queries only (parity_a0):

| feature | all | \|Rel\|=1 |
|---|---|---|
| entropy T=0.25 | +0.293 | +0.476 |
| gap 1-100 | −0.359 | −0.476 |
| gap 1-50 | −0.355 | −0.476 |
| gap 1-10 | −0.313 | −0.475 |
| top1 z | −0.346 | −0.462 |
| n within 1.0sd | +0.273 | +0.445 |
| gap 1-2 | −0.249 | −0.441 |
| caption words | −0.008 | −0.146 |

Every feature is *more* informative about the conformal score inside the
|Rel|=1 stratum than it is overall — the free signals are strongest exactly
where all the unclosed gap lives. What they are predicting there is not
ambiguity, since |Rel| is fixed at 1 across the whole subset; it is whether
this particular model is about to fail on this particular query. That is
selective prediction, and a single global monotone λ(x) cannot serve it and
ambiguity at the same time, because the two effects want different things from
the same features.

Note also that `margin_any` has a large point mass at zero — it is zero for
every query whose top-1 is already relevant — which is why the median set size
sits near 3 while the mean is 5 to 8. Any regression on this target is fitting
a spike plus a tail, and a linear quantile regression is a poor shape for that.

## 2026-07-31: the same feature needs opposite signs within and between strata

The paragraph above ends on an inference — that the two effects "want different
things from the same features" — drawn from two correlations. It is the only
justification for fitting the difficulty head separately inside each group
rather than pooling everything into one regression, so it is worth measuring
rather than asserting. `scripts/within_between_probe.py` measures it at the
quantile that actually sets the threshold.

For each feature, standardised over all 1000 queries, three slopes of the
τ = 0.9 quantile of the conformal score:

- **β_within** — a quantile regression fitted inside each stratum, averaged
  over the five with stratum sizes as weights.
- **β_between** — the five per-stratum oracle thresholds (4.04 / 2.58 / 1.72 /
  1.52 / 0.94) regressed on the five per-stratum mean feature values. Using the
  thresholds rather than mean scores keeps both slopes on λ's scale.
- **β_pool** — one quantile regression over all 1000 queries, no strata.

parity_a0, bootstrap over queries, 95% CI in brackets:

| feature | β_within | β_between | β_pool | per-stratum β |
|---|---|---|---|---|
| entropy T=0.25 | **+0.592** [+0.05, +1.10] | **−2.261** [−2.87, −1.44] | −0.070 | +0.75 +0.03 +1.02 +0.93 +0.39 |
| gap 1-100 | **−0.880** [−0.99, −0.60] | **+4.109** [+2.44, +5.45] | −0.747 | −1.20 −1.01 −0.78 −0.64 −0.32 |
| top1 z | **−0.798** [−0.92, −0.54] | **+3.784** [+2.35, +5.24] | −0.634 | −1.13 −0.87 −0.71 −0.65 −0.34 |

The sign flips for **every feature in every one of the four runs**, with the
within CI clear of zero each time, and the per-stratum betas agree with each
other rather than one stratum dragging the average.

The two slopes sit on different variance scales — the stratum means are much
less spread than individual queries, sd(x̄_s) / sd_within = 0.55 / 0.30 / 0.33 —
so the raw magnitudes are not comparable. Converted to the λ shift per one SD of
the relevant variation: entropy −1.09 between against +0.52 within, gap 1-100
+1.16 against −0.84, top1 z +1.18 against −0.77. **Same order of magnitude,
opposite direction.** Neither effect is a rounding error on the other, which is
what makes a single monotone λ(x) genuinely unable to serve both.

The mechanism is the one the CQR section guessed at. Between strata, a flat
score row means many near-ties, which means |Rel| is large, which means the
threshold should be *small*. Within a stratum |Rel| is pinned, so a flat row can
only mean this query is hard and the threshold should be *large*.

**The pre-registered attenuation rider asked the wrong question for two of the
three features.** It required |β_pool| < 0.5·|β_within|, on the theory that the
two effects cancel in the pooled fit. Entropy behaves that way in 3 of 4 runs
(+0.59 within → −0.07 pooled). gap 1-100 and top1 z fail it in all four,
because their stratum means barely differ (sd(x̄_s) ≈ 0.3) so the pooled fit is
dominated by within-stratum variation and simply recovers β_within. That is
worse than cancellation, not better: the pooled model **learns the within-stratum
slope and then applies it between strata, where the required sign is opposite**.
It does not merely fail to help across strata, it moves λ the wrong way.

### The oracle is a ceiling on one axis only

Part B pins |Rel| = 1 and asks whether anything is left. Difficulty is a
least-squares prediction of the conformal score fitted on the calibration half
only, so the quintiles are out of sample; `random` replaces it with noise and is
the floor for a spread read off ~36-query cells.

parity_a0, 400 splits inside the 363 |Rel|=1 queries:

| threshold | easy | 2 | 3 | 4 | hard | spread | size | marginal |
|---|---|---|---|---|---|---|---|---|
| single | 96.2% | 93.9% | 89.9% | 86.4% | 83.1% | **13.1 pt** | 8.0 | 89.8% |
| random | 89.7% | 89.8% | 89.5% | 90.1% | 89.9% | 0.6 pt | 8.0 | 89.8% |
| CQR within | 87.1% | 91.3% | 90.2% | 90.2% | 91.9% | 4.8 pt | 32.9 | 90.1% |

Paired single − random across the four runs: **+12.28 / +13.86 / +8.18 /
+10.17 ± 0.4 pt**, against a pre-registered bar of ≥ 5.0 pt in ≥ 3 of 4.

So there is a **second conditional-coverage gap of the same size as the first,
orthogonal to it**. Oracle Mondrian closes the |Rel| axis to 0.7 pt and leaves
this one untouched — inside the single stratum it is still 96.2% against 83.1%.
Every earlier statement of the form "the oracle closes the gap to 0.7, so 13
points is what is on the table" was measuring one axis and calling it the total.
The problem is two-dimensional: how many answers exist, and whether this model
will find one.

Third reading, and a constraint on the design: **CQR inside the stratum buys its
uniformity with set size again, and worse than before** — 13.1 → 4.8 pt at four
times the set (8.0 → 32.9 videos), against 1.5× for the global version. 90
fitting rows against 17 columns overfits, and the conformal correction can only
compensate by enlarging every set. A difficulty head is justified by Part B, but
not this recipe: it needs far fewer features, or a much stronger signal than the
score row can supply.

## 2026-07-31: the two axes have opposite diagnoses

Everything so far has hand-rolled its calibration layer — Mondrian buckets, then
CQR. Both are pre-2020 and both sit inside a single family. Gibbs, Cherian &
Candès (arXiv 2305.12616, number unverified — no network from this machine)
give the general form: for a finite-dimensional class F = span{φ_1 … φ_d},
fitting the pinball loss at τ = 1 − α has as its first-order condition

    E[ φ(X) ( 1{S ≤ φ(X)ᵀβ} − (1 − α) ) ] = 0,

so coverage holds along every direction in F. A global threshold is φ = 1;
Mondrian is φ = group indicators. The design question is therefore not which
algorithm but which F, and the two-axis finding above says what to put in it:

    φ(x) = [ 1{b(x)=k} ]ₖ ⊕ [ 1{b(x)=k} · d(x) ]ₖ

with b the predicted-|Rel| quintile and d a single within-bucket-centred
difficulty scalar. `scripts/gcc_conditional_probe.py` implements the pinball LP
directly so the +∞-imputation variant — which only perturbs the objective by
−τ·φ(x) — reuses the same constraint block.

Both axes are read off every method: `sprd_rel` over the true |Rel| strata,
`sprd_dif` over difficulty quintiles taken *within* each true stratum, so all
500 test queries contribute to the second reading rather than only the 363 with
|Rel| = 1.

parity_a0, 200 splits, paired bootstrap against `bucket rich`:

| method | sprd_rel | sprd_dif | size | marg | Δ rel | Δ dif |
|---|---|---|---|---|---|---|
| global | 15.0 | 15.0 | 5.4 | 89.8% | +2.42 ± 0.27 | +1.03 ± 0.48 |
| bucket rich | 12.6 | 14.0 | 5.1 | 90.2% | — | — |
| GCC buckets | 13.1 | 15.7 | 4.8 | 89.0% | +0.55 ± 0.37 | +1.71 ± 0.53 |
| **GCC +diff** | 11.6 | **5.1** | 6.1 | 88.9% | −0.94 ± 0.39 | **−8.86 ± 0.51** |
| GCC +inter | 12.8 | 5.9 | 6.6 | 88.1% | +0.19 ± 0.37 | −8.02 ± 0.48 |
| GCC +inter (imp, 20 splits) | 11.7 | 4.8 | 12.6 | 90.8% | −0.95 | −9.10 |
| oracle \|Rel\| | **0.7** | 15.2 | 5.5 | 90.3% | −11.63 ± 0.37 | +1.22 ± 0.46 |
| oracle 2-axis | 2.4 | 2.5 | 12.7 | 91.7% | −10.13 ± 0.39 | −11.29 ± 0.49 |

`GCC buckets` lands within 1 pt of `bucket rich` on `sprd_rel` in all four runs
(+0.55 / +0.75 / +0.00 / +0.78), which was the pre-registered correctness check:
quantile regression on group indicators is Mondrian, so anything else would have
meant the LP was wrong. The small positive sign is the handicap built into the
comparison — GCC fits its basis on one calibration quarter and calibrates on the
other, so it sees half the data `bucket rich` does.

### Axis two is a calibration-layer problem, and one basis function closes it

Adding a single centred difficulty scalar takes `sprd_dif` from 14.0 to 5.1
(and to 4.5 / 3.3 / 3.2 in the other three runs) at 1.2× the set size, using
only free score-row statistics. The two-axis oracle reaches 2.5, so most of the
available range is gone. The 13-point second gap found in step-002 was largely a
consequence of nobody having put difficulty in the conditioning set.

### Axis one is a signal problem, and the calibration layer is exhausted

Every calibration variant sits at 11.4–14.6 on `sprd_rel` while the oracle needs
0.4–0.8. Enriching F does not help, and it cannot: no choice of directions
manufactures |Rel| information that b(x) does not contain. The pre-registered
diagnostic threshold was ≤ 5.0 pt for "calibration layer" and > 8.0 pt for
"signal layer"; the observed 11.6–14.6 is unambiguous. The half-data handicap is
0.5–0.8 pt and does not move this.

### Perfecting axis one transfers nothing to axis two

`oracle |Rel|` closes its own axis to 0.7 and leaves `sprd_dif` at 15.2 — where
the plain global threshold left it (15.0), and slightly worse than the
deployable bucketing (14.0). The paired delta is positive in all four runs
(+1.22 / +2.36 / +2.34 / +2.67). A perfect ambiguity estimator would not improve
difficulty-conditional coverage at all. "Estimate |Rel| better and the problem
goes away" is false.

### The interaction was over-extrapolated from step-002

The pre-registered expectation was that per-bucket slopes are required, since
step-002 showed the required slope differs in sign between strata. They are not:
`GCC +inter` is worse than the shared-slope `GCC +diff` on **both** axes in
**all four** runs (Δ rel 1.1–1.7 pt worse, Δ dif 0.9–1.1 pt worse) and costs
8–14% more set size. 250 calibration rows split five ways is 50 points per
bucket, which at the 90th percentile is about five points of tail each.

The step-002 finding is not overturned — what it demanded was that difficulty
enter as a *within*-bucket slope, and that is exactly what `+diff` does via the
centring, and it is worth 10 points on axis two against `GCC buckets`. What
fails is the stronger reading, that each bucket needs its own slope. **The
effect is real at n = 1000 and not estimable at n = 250.** Worth stating
explicitly rather than quietly dropping the interaction.

### Two costs to record

Finite-sample conditional validity is expensive here: the +∞ imputation improves
both spreads slightly but runs 2.3–2.8× the set size and over-covers to
90.7–91.0%. The asymptotic version belongs in the main results and this in an
appendix. It was run on 20 splits rather than 200 — that is a cost cap, not a
silent one.

And the two-axis oracle does not reach 0.7 on either axis (2.0–2.5 / 2.5–3.1) at
2.5× the set size, because 25 cells over a 500-query calibration half is about
20 points each. **The true two-axis ceiling is not measurable on a 1000-query
grid.** That is a sample-size limit, not a method limit, and belongs in the
limitations.

## 2026-08-01: an external judge moves axis one, by two points

step-003 left axis one with a specific, testable diagnosis: the score row does
not carry |Rel|, so no calibration layer can recover it, and the only remaining
move is a model that is *asked* whether a retrieved video is correct rather than
how highly it scores. A dual encoder cannot be asked that — it emits one scalar
per pair and the question "is this one right" is never posed. A generative VLM
can be, once per pair, and the count of yes answers estimates |Rel|.

Three numbers were computed before the falsification criterion was written, to
establish that the pipeline is well-posed rather than to preview the answer.
Spearman(|judged pool|, |Rel|) is 0.059, so a judge with a constant yes-rate
cannot manufacture a correlation by tracking pool size. 96% of FIRE's relevant
videos sit inside our top-50, and a perfect judge restricted to top-50 scores
0.972. And a pure dual-encoder margin count over top-50 scores 0.225 with its
threshold tuned on the whole test set; the fitted 17-feature predictor scores
0.55, which is the number a judge has to beat.

Qwen2.5-VL-7B-Instruct judged 64,433 (caption, video) pairs — 995 distinct
captions against top-50 ∪ judged-pool, eight frames each, one forward pass per
pair reading P(yes) off the first answer position rather than generating.

### The judge estimates |Rel| better than free features and worse than required

| estimator | rho vs \|Rel\| |
|---|---|
| soft count sum P(yes), K=50 | 0.653 |
| soft count, K=30 (best K; 10–50 spans only 0.65–0.69) | 0.670 |
| hard count at 0.5 | 0.585 |
| cross-fitted calibration + fusion with the retrieval score | 0.713 |
| same fusion, judge removed | 0.549 |
| perfect judge on the same pool | 0.972 |

The pre-registered gate was 0.75, chosen loose: halving the coverage gap needs
0.83 on the noised-oracle curve. **It fails.** The gate's stated reasoning was
that with a pool ceiling of 0.972 the shortfall could only be blamed on the
judge, and three measurements now confirm exactly that rather than merely allow
it. Per-pair agreement with FIRE over 24,237 judged pairs is AUC 0.924 —
the judge *ranks* well — but TPR at the 0.5 threshold is 0.505, so it misses
half of what humans call relevant. The decomposition criterion authorised one
calibration round on the strength of that AUC; the round was run, and isotonic
plus logistic fusion bought 0.670 → 0.713 and stopped. And replacing the judge
with the human label on the 29% of top-30 that FIRE happened to judge, leaving
the VLM on the rest, lifts rho from 0.690 to **0.878** [0.834, 0.910]. The pool
is not the constraint, the coverage of the pool is not the constraint, and the
calibration layer is not the constraint. **Judge capacity is.**

### On the metric that matters it is worth two points

The project's standing rule is to score any ambiguity predictor by spread and
never by rho, because a real predictor at rho = 0.55 gave 13.8 pt where a noised
oracle at rho = 0.58 gave 10.1. So the judge's count was appended to
`fit_axes` as an eighteenth feature — nothing else changed, so the delta between
each arm and its twin is the judge and only the judge.

| arm | sprd_rel (4 seeds) | sprd_dif | size |
|---|---|---|---|
| global | 15.0 / 15.1 / 14.7 / 14.6 | 15.0 | 5.4 |
| bucket rich | 12.6 / 12.8 / 12.1 / 12.0 | 14.0 | 5.1 |
| GCC +diff | 11.6 / 11.9 / 11.7 / 11.0 | 5.1 | 6.1 |
| **GCC +diff VLM** | **9.6 / 9.8 / 9.7 / 9.4** | 5.6 | 6.3 |
| oracle \|Rel\| | 0.7 / 1.3 / 0.8 / 1.0 | 15.2 | 5.5 |

Paired bootstrap against `GCC +diff`: Δsprd_rel −2.07 / −2.06 / −1.99 / −1.64
± 0.42, significant in all four seeds, with Δsprd_dif between +0.11 and +0.58
and a size ratio of 0.99–1.03. **The judge buys two points on axis one and
costs nothing in set size.** A Mondrian-only variant reaches the same 9.5–10.5
but gives back all of axis two, so the function-class arm is the one to keep.

Two points is real and small. The residual gap is 11.6 against an oracle at 0.7,
so this closes 19% of it. Against the noised-oracle curve the real judge at
rho = 0.67 gives 9.6 where interpolation predicts about 9.0 — **a real predictor
under-delivers relative to a noised oracle at matched rho for the second time**,
which is the strongest evidence yet that rho is a screen and not a forecast.

### The cost is the problem, not the effect

Those two points cost fifty 7B forward passes per query. As a *test-time*
module that is not a defensible system, and the paper should say so before a
reviewer does. Escalating judge capacity to 32B or 72B is the variable the
attribution points at, but it makes the cost objection worse, not better, even
if it works.

The alternative the attribution also permits is to move the judge to training
time: label the training set offline, distil a cheap ambiguity head from the
pseudo-labels, and pay nothing at test time. That path now has a measured
ceiling rather than a hope — a perfect distillation of this judge is rho 0.67,
which is 9.6 pt.

## Next

1. **Drop the DUA family from the contribution.** Theirs and ours. Eight
   arms across two protocols with nothing above +0.4 is not a tuning
   problem, and the core four components (two distribution heads, the
   matcher, the two losses) have no measured value to defend.
2. **Do not re-run the seven arms.** They buy comparability, not
   significance, and post-deletion they would be measuring something the
   originals did not measure anyway.
3. Report paired McNemar, not cross-run deltas, for any two arms sharing a
   test split. The per-query ranks are in each run's `final_test.json` under
   `selections.t2v.test_metrics.{t2v,v2t}.cols`.
4. The remaining live direction is the one that does not need a cross-run
   delta at all: **selective retrieval / risk–coverage**, where uncertainty
   is scored by whether it can rank a single model's own predictions by
   reliability. That comparison lives inside one checkpoint, so neither the
   1.1 retraining floor nor the epoch-selection premium touches it. Note the
   prior result that `u_pair`'s residual AUC came in below chance — that has
   to be re-derived before anything is built on it.
5. **Score everything under FIRE labels from now on, alongside the original.**
   The judgments are on this exact grid and cost nothing to apply. Report
   original R@k for comparability with the literature, FIRE R@k and
   condensed R@k as the bracket the true value lies in, and judged@k so the
   pool depth is visible rather than assumed. Under corrected labels R@1 also
   stops being the natural statistic — with 3.18 positives per caption,
   nDCG@10 and mAP use the whole ranking instead of one threshold.
6. Two things to check before building on FIRE. Whether the 23.6%
   original-pair rejection rate is annotator noise or real caption error —
   the release carries only 16 disagreement records, so it is mostly
   single-annotator and cannot be settled from the file alone. And whether
   the MSVD half (158MB, the bulk of the 683K) covers a split we can use as a
   second test bed.
7. **The live direction is now item 4 made concrete: predict query ambiguity
   well enough to restore conditional coverage.** The gate passed — the
   conditional coverage failure is 15 points, an oracle fixes it to 0.7, and
   the free margin only reaches 13.3. Steps (a) and (b) are done and are
   recorded in the 2026-07-31 σ section: σ is negative as a stratifier, and
   the noised-oracle sweep sets the bar at ρ ≈ 0.83 to halve the gap against
   the ρ = 0.55 the free features plus σ reach. What remains: (c) a *learned*
   ambiguity predictor — a text encoder fine-tuned on |Rel| — since the linear
   fit on four scalar features is not a serious attempt and the target is
   labelled; (d) report average set size at fixed *conditional* coverage as
   the headline, with R@1 alongside for comparability; (e) score any predictor
   on spread directly, never on ρ, because a real predictor at ρ = 0.55 gives
   13.8 pt where a noised oracle at ρ = 0.58 gives 10.1 pt. Note the pool
   caveat carries into this: `any` coverage is a lower bound because an
   unjudged relevant video in the set is not counted.

   The 2026-07-31 CQR section revises this. (e) is now measured properly: the
   paired bootstrap over splits, not the within-split max-minus-min, which
   compresses a 13.1-point difference to 4.1. The best label-free method
   (CQR on 17 gallery features) buys 2.5 pt and pays 1.5× mean set size; the
   best method that is free on both axes is quintile bucketing on those same
   features. Neither is close to the ≈7 pt that halves the gap.

   Three things now come before (c). **First, pick the coverage target and
   say so.** Under `any`, conditional coverage demands the *largest* sets for
   the *most specific* queries (oracle λ 4.04 at |Rel|=1 against 0.94 at 6+),
   which is the opposite of the story the proposal has been telling. Under
   `all` it runs the intuitive way. **Second, the unclosed gap is entirely at
   |Rel|=1** — every deployable method trims over-coverage at 6+ and none of
   them moves 83.7% at the bottom, against an oracle 90.1%. **Third, that
   bottom stratum is not an ambiguity problem at all**: with |Rel| held at 1,
   the free features still correlate 0.44–0.48 with the conformal score, so
   what is unexploited there is failure prediction, not ambiguity prediction.
   A single monotone λ(x) has to serve both and cannot. The natural next
   design is two signals rather than one — a |Rel| estimate for the stratum
   and a failure estimate within it — before spending GPU on (c).
8. Venue note. This has no R@1 SOTA table and its headline statistic is set
   size at guaranteed coverage, which reads as a non-contribution to a CVPR or
   ECCV reviewer. SIGIR, EMNLP or TMLR fit the claim better. Also check
   novelty against the near neighbours before committing: CLARA (2606.18992)
   uses conformal set size as an ambiguity measure for composed image
   retrieval, SAFEVPR (2605.28048) uses Mondrian conformal for visual place
   recognition, and VQPP (2602.17814, code released) is already the first
   text-to-video query performance prediction benchmark. What is not taken is
   conditional coverage under ambiguity heterogeneity, with multi-positive
   human judgments to define it against.
