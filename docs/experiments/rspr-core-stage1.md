# RSPR core Stage 1: MSR-VTT protocol

Status: engineering protocol prepared on the current commit. No GPU smoke run, A0–A8 training, or second-dataset work has been executed; all run-time fields below remain pending.

## Fixed training stages

Stage A is the one-epoch initialization command:

```bash
RSPR_MODE=stochastic \
RSPR_RANK_WEIGHT=0 \
RSPR_WARMUP_EPOCHS=0 \
RSPR_FREEZE_CLIP=1 \
RSPR_FREEZE_DSA=0 \
RUN_ID=rspr_stage_a_seed0 \
./run_train_msrvtt_bg.sh --epochs 1
```

Stage B starts from the Stage A checkpoint:

```bash
RSPR_MODE=stochastic \
RSPR_RANK_WEIGHT=0.1 \
RSPR_WARMUP_EPOCHS=1 \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
FREEZE_LAYER_NUM=8 \
RUN_ID=rspr_stage_b_seed0 \
./run_train_msrvtt_bg.sh --init_model ckpts/ckpt_msrvtt_rspr_stage_a_seed0/pytorch_model.bin.0
```

For a canonical ablation command fragment, use `python3 scripts/rspr_ablation_matrix.py --ablation A3 --print-shell-args`; it only prints arguments and never launches training.

## Record sheet

Fill one row for every ablation and seed, using the same trusted split manifest. `pending` is intentional until the corresponding experiment is run.

| data protocol hash | git commit | ablation (A0–A8) | seed | K | parameter count | R@1 | R@5 | R@10 | MdR | MnR | peak GPU memory | throughput | Top-R latency | logvar min | logvar mean | logvar p50 | logvar p95 | logvar max | U_pair error AUROC | repeated-evaluation rank agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pending | pending | A0 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A1 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A2 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A3 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A4 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A5 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A6 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A7 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| pending | pending | A8 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Stop conditions

Stop the main experiment and repair the implementation before continuing if any loss is NaN or Inf; `logvar` in every dimension remains at either -8 or 2 for a continuous epoch; A3 fails to produce finite gradients for DSA, mean, or logvar; or fixed-noise repeated evaluation has inconsistent rankings.

## Pending run-time acceptance

Step 6 (a 20-optimizer-step single-GPU smoke run) is pending. It must verify finite total and four unweighted losses, finite gradients for mean/logvar/DSA, near-unit sample norms, non-monotonic memory growth, and exact fixed-noise checkpoint reload reproducibility.

Step 7 (MSR-VTT A0–A8) is pending. Run A0, A1, A3, A6, A7, and A8 first, then A2, A4, and A5; each needs at least three seeds on the same trusted split manifest. Do not create or implement Beta evidence until all records and cost diagnostics are complete.

Step 8 (a separate DiDeMo data-protocol plan) is pending and has not been created. The current repository has no DiDeMo/VATEX loader. Any paper claim about the core module requires both MSR-VTT and DiDeMo validation with the same model commit and A0/A1/A3/A6/A7/A8 subset.
