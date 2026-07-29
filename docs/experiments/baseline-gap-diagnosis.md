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
