# RSPR core Stage 1: MSR-VTT protocol

Status: A3 GPU smoke completed on commit `6fde108`; the approved minimal
screening schedule is pending, beginning with A0 training seed 0. No complete
Stage 1 training result exists yet.

## Fixed end-to-end training

The canonical A3 run starts from OpenAI CLIP `ViT-B/16` weights and uses one
continuous five-epoch optimizer schedule:

```bash
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=stochastic \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
FREEZE_LAYER_NUM=8 \
RUN_ID=rspr_a3_seed0 \
./run_train_msrvtt_bg.sh
```

CLIP blocks 8–11, DSA, WTI, and RSPR are trainable from the first optimizer
step. DSA and probability losses use their full weights immediately; rank and
anchor weights warm up during the first epoch. `--init_model` is not a stage
transition requirement. A0–A8 use the same CLIP start, trusted split, five
epochs, and optimizer schedule; only the ablation arguments differ. A4 keeps
the legacy branch/loss semantics but does not require a historical UATVR
checkpoint.

For a canonical ablation command fragment, use `/home/xujie/.conda/envs/tvr/bin/python scripts/rspr_ablation_matrix.py --ablation A3 --print-shell-args`; it only prints arguments and never launches training.

## Record sheet

### First round: seed-0 screening

| order | ablation | training seed | purpose | status |
| ---: | --- | ---: | --- | --- |
| 1 | A0 | 0 | deterministic DSA/WTI baseline | pending |
| 2 | A1 | 0 | mean-only capacity control | pending |
| 3 | A2 | 0 | detached-sample reparameterization control | pending |
| 4 | A3 | 0 | complete RSPR | pending |
| 5 | A7 | 0 | A3 without stochastic rank loss | pending |
| 6 | A8 | 0 | A3 without anchor KL | pending |

### Second round: gated confirmation

| configuration | additional training seeds | unlock condition | status |
| --- | --- | --- | --- |
| A0 | 1, 2 | A3 passes the first-round gate | locked |
| A1 | 1, 2 | A3 passes the first-round gate | locked |
| minimal RSPR winner (A3/A7/A8) | 1, 2 | selected by the simplicity rule | locked |

The current Stage 1 budget is at most 12 full jobs: six seed-0 screening jobs
and, only after promotion, six confirmation jobs. If A3 fails the gate, the
budget stops at six.

For every executed run, record the data protocol hash, Git commit, ablation,
training seed, K, parameter count, T2V/V2T R@1/R@5/R@10, MdR, MnR, peak GPU
memory, throughput, Top-R latency, logvar min/mean/p50/p95/max, U_pair error
AUROC, and repeated-evaluation rank agreement.

## Minimal screening gate

- A3 must exceed `max(A0 T2V R@1, A1 T2V R@1)` by at least 0.5 percentage
  points on internal val.
- A3 may trail `max(A0 V2T R@1, A1 V2T R@1)` by at most 0.5 percentage
  points.
- All existing numerical stop conditions remain mandatory; OOM also blocks
  promotion.
- If the A3-over-A2 T2V R@1 gain is less than 0.5 percentage points, do not
  claim an independent benefit from reparameterization gradients.
- If A7 or A8 is within 0.5 T2V R@1 points of A3, satisfies the same V2T guard,
  and remains numerically stable, prefer the simpler configuration.
- If both A7 and A8 qualify, choose higher T2V R@1, then higher V2T R@1.
- If A3 fails the gate, stop after the six first-round jobs.
- If A3 passes, add seeds 1 and 2 only for A0, A1, and the selected winner;
  report all three per-seed results plus mean and standard deviation.

The 0.5-percentage-point cutoff is a practical screening threshold, not a
claim of statistical significance.

[Approved minimal-screening design](../superpowers/specs/2026-07-23-rspr-minimal-screening-design.md)

## Stop conditions

Stop the main experiment and repair the implementation before continuing if any loss is NaN or Inf; `logvar` in every dimension remains at either -8 or 2 for a continuous epoch; A3 fails to produce finite gradients for DSA, mean, or logvar; or fixed-noise repeated evaluation has inconsistent rankings.

## Runtime acceptance

The canonical four-GPU A3 smoke used
`RUN_ID=rspr_smoke_20260722_105328`. At optimizer step 20 it reported total
loss `3.0812`, DSA `2.5358`, probability `5.4342`, rank `0.6918`, anchor
`0.0000`, pair uncertainty `0.0003`, text/video variance means `0.0100/0.0100`,
CLIP/other learning rates `5.72e-09/5.72e-06`, and `0.93s` per step. All values
were finite, with no OOM, NaN, Inf, or training traceback before intentional
termination. Polling allowed the run to reach step 80 before its isolated
worker process group was stopped; the final torchrun `SignalException` records
that requested `SIGTERM`, not a training failure.

Smoke log:
`logs/20260722/105328_rspr_smoke_train_msrvtt.log`. Smoke output:
`ckpts/ckpt_msrvtt_rspr_smoke_20260722_105328/`. No checkpoint was written
because the run stopped before the first epoch completed.

Step 7 uses the approved minimal schedule. Run A0, A1, A2, A3, A7, and A8
once with training seed 0. A6 is not run because the canonical A3 configuration
already uses the soft matcher. A4 and A5 remain conditional one-seed follow-ups
outside the current budget. The first pending job is A0 seed 0.

Step 8 (a separate DiDeMo data-protocol plan) remains gated and has not been
created. It may begin only after the selected MSR-VTT winner has a directionally
consistent three-seed result. The current repository has no DiDeMo/VATEX
loader; do not assemble a temporary loader inside Stage 1.
