# Stage 1 baselines — HALT

**Gate decision:** [stage1-gate.json](stage1-gate.json)
**Payload:** [stage1-gate-payload.json](stage1-gate-payload.json)
**Run:** `msrvtt-baseline-validation_baselines-seed000-20260818T040000Z-1fdd78f384d1`
**Winner checkpoint:** `checkpoints/pytorch_model.bin.2` (epoch 2, model_sha256 `e503946f0c7d861a2ea6e3e49eecb30e8f51ecb67a83e994fd1d8a26938d565f`)

## Decision

**Stage 1 gate: FAIL.** `next_authorized_stage = null`. The campaign
halts at Stage 1 per the frozen `evaluate_stage1_gate` contract.

Two failed checks:
1. `a3_improves_strongest`: A3 did not improve the strongest scalar
   baseline on either mean AUROC or mean AURC.
   - `mean_auroc_delta` = -0.0287 (A3 loses ~2.87 AUROC points)
   - `mean_aurc_delta`  = +0.0019 (A3 loses on selective risk too)
2. `directional_rho_gte_0_20`: V2T Spearman ρ = 0.19735 < 0.20 (misses
   by 0.003). T2V ρ = 0.5661 (well above threshold).

## Numbers (MSRVTT trusted-v1 val, 500 videos × 20 captions = 10 000 pairs)

Failures per direction: T2V = 2 348 / 10 000; V2T = 35 / 500.

Direction-wise metrics:

| Scorer                | T2V AUROC | T2V ρ  | V2T AUROC | V2T ρ  | mean AUROC | mean AURC |
|-----------------------|----------:|-------:|----------:|-------:|-----------:|----------:|
| **entropy** (strongest) | 0.8717 | 0.5591 | **0.7883** | **0.2549** | **0.8300** | **0.0441** |
| one_minus_top1        |    0.8819 | 0.5709 |    0.7581 | 0.2278 |     0.8200 |    0.0442 |
| **A3 (winner ep 2)**  |    0.8796 | 0.5661 |    0.7230 | 0.1974 |     0.8013 |    0.0460 |
| negative_margin       |    0.8742 | 0.5509 |    0.5919 | 0.0815 |     0.7330 |    0.0535 |
| negative_energy       |    0.6923 | 0.2920 |    0.7287 | 0.2017 |     0.7105 |    0.0775 |
| score_variance        |    0.4685 | -0.0437|    0.4137 | -0.0786|     0.4411 |    0.1733 |

Retrieval sanity: `t2v R@1 = 76.52`, `v2t R@1 = 93.0` (matches the
frozen retriever's expected values under fp32 pinning; ranks are
preserved with respect to the pre-sidecar backbone).

## Interpretation

- **A3's t2v arm is strong**: AUROC 0.880, ρ 0.566, comparable to the
  best scalar baselines. Not the story-breaker.
- **A3's v2t arm is the failure mode.** V2T has only 500 queries with
  35 failures; that regime is high-variance for AUROC and for Spearman
  ρ. A3's v2t AUROC 0.723 is materially below entropy's 0.788.
- Under a fixed-config, single-seed, 3-epoch training on the trusted
  split, adding a 128-dim MLP over top-K candidate context does not
  beat a zero-parameter entropy scorer computed on the same frozen
  retriever's val scores.
- **The Frozen-retriever guarantee holds**: `retrieval` check in
  `paired_invariance` = True; `raw_wti` differs only because the run's
  evaluator autocast path differs from the reference (see the
  paired_invariance.checks{a1_failure_risk, a1_vacuity, raw_wti,
  text_means, video_means} = False set, with `retrieval` True — R@k
  and ranks are byte-identical though pre-DSL sim values drift on
  bf16 vs fp32). Consistent with
  `scripts/verify_frozen_retrieval_equivalence.py` findings.

## Protocol next steps

`evaluate_stage1_gate` returned `next_authorized_stage = null`, so:

- **Do not** prepare Stage 2. The controller's
  `assert_phase_authorized` would refuse anyway (predecessor gate
  requires `passed=true` and `next_authorized_stage=2`).
- **Do not** silently re-run Stage 1 with different hyperparameters.
  That would violate the frozen `study.yaml` and constitutes
  post-hoc selection. Any redesign of A3 requires a new
  pre-registered study.
- **Do finalize** Stage 1 as stopped so the archive summary and
  registry are consistent. The commit sequence for finalization is
  `git add summaries/stage1-baselines*.{json,md} && git commit`,
  followed by `run_reliability_campaign.py finalize --phase … --summary
  summaries/stage1-baselines.json`.
- **Do restore** the frozen `{0,2,3,4}` GPU topology after Stage 1
  cleanup: revert commits `05f57fa`, `62069e0`, `3fe60cf` (the
  {0,1,2,3} pivot) so the launcher/main/tests are back to their
  qualified state. Cite this Stage 1 summary as the "no-work-pending"
  justification for the revert.

## Scientific implication for PAPER.md

This is a documentable **negative result** on trusted-v1 val:
"A candidate-conditioned A3 sidecar (128-dim MLP over query mean,
top-16 candidate context, and A1's 8 score-geometry features) trained
for 3 epochs from the A1 checkpoint fails to improve on a
zero-parameter entropy scorer under a frozen retriever on the trusted
MSRVTT val split." It should be reported alongside the A2-CG negative
result in `findings.md §2.3`, with the caveat that A3 was previously
reported as passing on JSFUSION test — the trusted split's smaller
v2t sample (500 queries, 35 failures) drives the v2t AUROC/ρ gap and
motivates either a larger v2t evaluation set or a v2t-specific
redesign in a future study.
