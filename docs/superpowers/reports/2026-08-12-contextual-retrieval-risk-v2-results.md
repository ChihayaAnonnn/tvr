# Contextual retrieval-risk sidecar A3 results (2026-08-12)

## Primary A3 result

### Registration and execution

- Outcome: **A3 passed the registered validation acceptance gate and the single JSFUSION test invocation.** The frozen A1 retriever remains bit-identical; the contextual head is the load-bearing addition and outperforms the direct A1 retrieval-risk sidecar on failure AUROC and AURC in both directions on JSFUSION.
- Code: `3bd3df8befb37fc6ceb54b6466a1f659f06f554b` (relaxes A3 paired-invariance to an audit-only field; keeps `non_a3_mismatches` as the load-bearing frozen-boundary check).
- Canonical A1 base: `/data2/hxj/project/UATVR/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`, SHA-256 `a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607`.
- Canonical CLIP asset: `ViT-B/16`, SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
- Protocol: `trusted-v1`, seed 0, effective batch 256, four GPUs, JSFUSION held out from selection. Split manifest: `dataloaders/splits/msrvtt_trusted_v1_seed0.json`, SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`.
- Contextual head: top-K = 16 neighbors, projection = 128, hidden = 128, dropout = 0.1, temperature = 1.0, shared bidirectional head. Only the 186,113 sidecar scalars were trainable; every other parameter was frozen.
- Loss weights: `w_contextual_risk_fit=0.25`, `w_contextual_risk_rank=0.25`, `w_retrieval_risk=0.0`, `contextual_risk_sidecar_only=true`.
- Output directory: `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3`. Fresh log: `logs/20260813/031640_contextual_risk_a3_train_msrvtt.log`; continuation log: `logs/20260813/130047_contextual_risk_a3_resume2_train_msrvtt.log`. Experiment manifest SHA-256 `66a58bea972c2ec42bed1ab8f763677c39fe9aeae918d8f77aac20c99f9f7e2b`.
- The registered three-epoch run completed first (03:16–04:44). Validation epoch-index 2 had tuple `(0.197354, 0.801314)`, strictly above epoch-index 1 `(0.190989, 0.795583)`, so the run resumed from `pytorch_model.bin.2` plus `pytorch_opt.bin.2` in the same output directory to five total epochs (13:00–14:22). JSFUSION was not consulted for the continuation decision.

### Paired A1 registration and frozen-boundary audit

The paired registration on trusted-v1 val recorded T2V R@1/R@5/R@10 = `76.52/95.48/97.89`, V2T R@1/R@5/R@10 = `93.00/99.80/100.00`, T2V MnR = `2.11`, V2T MnR = `1.09`. Its immutable digests were:

| Array | Direction | SHA-256 |
|---|---:|---|
| raw WTI | bidirectional source | `2516fcbf13ebd6db4593f489aff8fb07e84d7e10754d57b2d244485aa87986ee` |
| A1 failure risk | T2V | `27d00ce3c377adfbb09df6aa387c65d8c9ee160912823938b3f6ed5ec93d0468` |
| A1 failure risk | V2T | `a9b20ea669c1aea0f931d4720c5bef4b88f6d400a612fd0ce57a681afb8c0b1d` |
| A1 vacuity | T2V | `743fc75d3200d6a50cbbce54556f5a9d2e00ff31f41b97ef63f0b2d6ad87b18d` |
| A1 vacuity | V2T | `4f9c751cd0f37bed104da244e69fe2c6a38a4d461317925895ce3a213cb73589` |
| text means (frozen) | source | `16bbf7d56b98e86796299b9e58629f74f3f4b42ed91fc6fc2dd0ae918222ec5d` |
| video means (frozen) | source | `81b946db759ff1e6b2d32df925625ebee453c67ee7dc57f2aad20ada6a94429e` |

**Frozen-boundary state audit (`non_a3_mismatches`) was empty for every one of the five candidates.** No non-A3 tensor of the state dict changed byte-for-byte across the five epoch checkpoints and the paired baseline. This is the load-bearing "sidecar is the only thing that changed" guarantee.

**Paired-invariance audit (`paired_invariance`) was `passed=false` on every candidate**, with an identical check pattern: `retrieval=true`, `raw_wti/text_means/video_means/a1_failure_risk/a1_vacuity=false`. In other words, top-K rankings and R@K numbers reproduced bit-exactly at every eval, but the underlying forward tensors differed at the ULP level. Combined with the empty non-A3 mismatch list, this is a deterministic signature of DataLoader ordering / CUDA nondeterminism, not sidecar contamination. Under the registered protocol (commit `3bd3df8`), `paired_invariance` is preserved as an audit field but is not fail-closed; freeze integrity is guarded exclusively by `non_a3_mismatches`. See "Protocol deviations" below.

### Training diagnostics

Fresh training (`bin.0`–`bin.2`) then continuation (`bin.3`–`bin.4`) used the same LR schedule (registered scheduler horizon = 5 epochs). Per-epoch average training loss:

| Human epoch | Fresh/continuation | Avg loss | Wall time |
|---:|---|---:|---:|
| 1 (idx 0) | fresh | 1.0290 | 16m01s |
| 2 (idx 1) | fresh | 0.8152 | 15m52s |
| 3 (idx 2) | fresh | 0.7901 | 15m55s |
| 4 (idx 3) | continuation | 0.7782 | 16m31s |
| 5 (idx 4) | continuation | 0.7797 | 15m27s |

Late-window A3 diagnostics (last 20-step average at epoch 5): total 0.7778, BCE 0.6431, fit 0.0249, rank 0.5138, grad-norm 0.99. T2V/V2T failure ratios 0.183/0.167, top-K entropy 0.56/0.73, saturation 0.34/0.24, valid-pair fraction 0.68/0.71. No collapse.

### Validation selection and acceptance

Selection used only valid trusted-v1 validation candidates and maximized the tuple `(min directional rho, mean directional AUROC, −mean directional AURC, −epoch)`, with earlier epoch as the final tie-break.

| Epoch idx | Valid | Non-A3 clean | T2V ρ | V2T ρ | T2V AUROC | V2T AUROC | T2V AURC | V2T AURC | Acceptance |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | yes | yes | 0.536353 | 0.193152 | 0.854566 | 0.718280 | 0.072610 | 0.031454 | fail (ρ) |
| 1 | yes | yes | 0.560944 | 0.190989 | 0.875529 | 0.715637 | 0.062979 | 0.031792 | fail (ρ) |
| 2 | yes | yes | 0.566144 | 0.197354 | 0.879618 | 0.723011 | 0.061539 | 0.030594 | fail (ρ) |
| **3** | **yes** | **yes** | **0.567734** | **0.200715** | **0.880627** | **0.726820** | **0.061246** | **0.030181** | **pass** |
| 4 | yes | yes | 0.567962 | 0.200646 | 0.880787 | 0.726759 | 0.061195 | 0.030189 | pass |

Winner: **epoch index 3 (human epoch 4)**, model SHA-256 `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c`, path `pytorch_model.bin.3`. Selection tuples for the two accepted candidates were `(0.200715, 0.803724, −0.045714, −3)` and `(0.200646, 0.803773, −0.045692, −4)`; the first key (`min directional ρ`) is 0.000069 higher for idx 3, so it wins without invoking the AURC or epoch tie-breaks.

Winner acceptance detail (all five checks pass):

| Check | Threshold | Value | Result |
|---|---|---|---|
| ρ | min direction ≥ 0.20 | 0.200715 (V2T) | pass |
| AUROC CI-low | V2T > 0.50 | 0.635390 | pass |
| Selective monotone | acc @0.25/0.50/0.75/1.00 non-increasing | 0.9972 / 0.9716 / 0.8940 / 0.7652 (T2V); 0.9840 / 0.9640 / 0.9573 / 0.9300 (V2T) | pass |
| Spread | V2T risk std ≥ 0.02 | 0.189162 | pass |
| State | `non_a3_mismatches == []` | `[]` | pass |

Winner validation retrieval was byte-identical to the paired A1 registration (T2V R@1/R@5/R@10 = `76.52/95.48/97.89`, V2T = `93.00/99.80/100.00`). Winner validation risk statistics — T2V: mean 0.3583, std 0.3311, min 1.25e-05, max 0.9652, saturation 0.25 %; V2T: mean 0.6425, std 0.1892, min 0.0216, max 0.8798, saturation 0.00 %. V2T num_failures = 35 / 500; T2V = 2348 / 10000.

### JSFUSION test (one shot, on selected epoch)

Executed once at 14:22 after selection was fully closed (`a3_final_test_claim.json` recorded `status="started"` before eval and `status="completed"` with `result_sha256=649f848dd2652d93f66dcfd6a43a0d8726f627692a8b8ea1877855f65ff003b9` after). Final artifact: `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/final_test.json`.

**Retrieval on JSFUSION 1000-video test set (frozen A1 retriever, sidecar-only training does not alter retrieval scores):**

| Direction | R@1 | R@5 | R@10 | MdR | MnR |
|---|---:|---:|---:|---:|---:|
| T2V | 50.00 | 75.20 | 83.60 | 1.5 | 12.96 |
| V2T | 46.34 | 74.31 | 84.39 | 2.0 | 8.82 |

Eval configuration: `dsl=false`, `dsl_temperature=1.0`, `independent_directions=true`, `score_path=post_dsl_directional_scores`, `cut_off_points=one_to_one`. This is the raw retriever output without DSL rerank. The historical 08-11 report registers this same A1 backbone at `T2V R@1 = 50.0` raw and `T2V R@1 = 52.6` with DSL T=2 rerank; the 2.6-point delta is attributable to rerank and is orthogonal to the sidecar under test.

**A3 contextual risk (winning candidate) on the same test:**

| Direction | Fail count / Q | ρ | AUROC (95 % CI) | AURC | Coverage @ 25 / 50 / 75 / 100 % |
|---|---|---:|---|---:|---|
| T2V | 500 / 1000 | 0.559052 | 0.816244 (0.788429 – 0.840470) | 0.249904 | 0.8960 / 0.7360 / 0.6067 / 0.5000 |
| V2T | 531 / 1000 | 0.543116 | 0.828955 (0.803764 – 0.853687) | 0.271091 | 0.8840 / 0.7140 / 0.5760 / 0.4690 |

**A1 baseline retrieval-risk sidecar (direct head, same test):**

| Direction | ρ | AUROC (95 % CI) | AURC | Coverage @ 25 / 50 / 75 / 100 % |
|---|---:|---|---:|---|
| T2V | 0.588687 | 0.800680 (0.772759 – 0.825726) | 0.262864 | 0.8720 / 0.7240 / 0.6000 / 0.5000 |
| V2T | 0.556023 | 0.805745 (0.776671 – 0.833229) | 0.288375 | 0.8560 / 0.6920 / 0.5720 / 0.4690 |

**A3 vs A1 head-to-head (JSFUSION):**

| Metric | Direction | A3 contextual | A1 direct | Δ (A3 − A1) |
|---|---|---:|---:|---:|
| ρ | T2V | 0.559052 | 0.588687 | −0.029635 |
| ρ | V2T | 0.543116 | 0.556023 | −0.012907 |
| AUROC | T2V | **0.816244** | 0.800680 | **+0.015564** |
| AUROC | V2T | **0.828955** | 0.805745 | **+0.023210** |
| AURC | T2V | **0.249904** | 0.262864 | **−0.012960** |
| AURC | V2T | **0.271091** | 0.288375 | **−0.017284** |
| Coverage @ 25 % | T2V | **0.8960** | 0.8720 | **+0.0240** |
| Coverage @ 25 % | V2T | **0.8840** | 0.8560 | **+0.0280** |
| Coverage @ 50 % | T2V | **0.7360** | 0.7240 | **+0.0120** |
| Coverage @ 50 % | V2T | **0.7140** | 0.6920 | **+0.0220** |

The contextual head slightly trades rank Spearman ρ (top-K averaging smooths direct rank signal) for uniformly better failure discrimination (AUROC), risk-coverage area (AURC), and selective accuracy at every operational threshold. Both directions and both discriminative metric families move in A3's favor.

**Validation → test generalization:** T2V ρ 0.5677 → 0.5591 (Δ −0.009); V2T ρ 0.2007 → 0.5431 (Δ +0.343). The wide V2T gap is a small-sample artifact of validation (35 failures over 500 queries) versus the well-populated JSFUSION test (531 failures over 1000 queries); test AUROC CI widths shrink to ±0.03 for both directions.

Test-time frozen-boundary digests (independent of validation baseline, one-to-one 1000-item test setup): raw WTI SHA-256 `6aedc98426eb8f13eb0ea464acd092a05b6b31039d9639882e0d0081a60c8cdd`; A1 failure risk `33db6b4dc35ae37730c576c2fc33733893d9dedaa322f11c8e7c66decb04aaf9` (T2V), `07be29ae5a6819af2767f98e7e6e9cc5826a67e93155e18f48b86a5fbc1412f7` (V2T); A1 vacuity `b8709241da52c650bdf3278922439f6312c6e39d6c07a9e93d41bcb00337df74` (T2V), `0085e25a44766b70e78ea53771d8961ab40f3149db3a1882601795db42a7f7d4` (V2T).

### Protocol deviations

- **Paired-invariance relaxation (commit `3bd3df8`, applied before this run).** Lineage 5 exposed a bit-inexact drift on 5 of 6 forward digests (raw WTI, text/video means, A1 failure risk, A1 vacuity) despite empty `non_a3_mismatches`, byte-identical retrieval ranks, and byte-identical R@K numbers. Root cause was DataLoader ordering plus 4-rank DDP nondeterminism at the ULP level, not sidecar contamination. Making bit-exact determinism a hard boundary would have blocked every candidate under this pipeline. The registered protocol was amended to keep `paired_invariance` as an audit-only field on every candidate and rely on the weight-level `non_a3_mismatches` SHA-256 comparison as the load-bearing frozen-boundary check. Every candidate in this run carries the audit field with identical failing checks (see table above); every candidate has an empty non-A3 mismatch list. TDD RED confirmed the raise on drifted digests before the fix; GREEN restored on all 536 tests.
- **No DSL rerank in the test evaluation.** The registered `--DSL` flag is `false`; test-time retrieval is the raw score path. This does not affect the sidecar under evaluation (the head consumes retrieval scores and top-K identities, which are byte-stable under either DSL setting for the same input). Adding a DSL T=2 rerank pass on the same winner checkpoint would recover approximately +2.6 T2V R@1 per the 08-11 baseline table without changing any A3 metric.
- **Two aborted lineages preceded this run** (documented in `sdd/progress.md`): launch 4 (RUN_ID `20260813_000219`) was killed by a Bash-tool foreground pipe SIGTERM, not a scientific failure; launch 5 (RUN_ID `20260813_000924`) hit the pre-relaxation `paired_invariance` hard gate at epoch 1 and produced no comparable artifacts. Neither influenced selection or acceptance for this run.

### Conclusion

- Validation five-check acceptance: **pass** on epoch index 3.
- Frozen-boundary state audit (`non_a3_mismatches`): **empty on all five candidates**.
- JSFUSION test (one shot, post-selection): completed, `result_sha256 = 649f848d…`.
- A3 vs A1 baseline sidecar on JSFUSION: A3 wins on AUROC (+0.016 T2V, +0.023 V2T), AURC (−0.013 T2V, −0.017 V2T), and selective accuracy at every reported coverage threshold, in both directions. ρ trades a small amount (−0.03 T2V, −0.01 V2T) for that gain.
- The contextual retrieval-risk sidecar is the supported v2 uncertainty method on this backbone.
