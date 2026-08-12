# CSD log-variance A2 results (2026-08-12)

## Primary A2-CG result

### Registration and execution

- Outcome: **A2-CG failed the registered validation acceptance gate.** The A1 retrieval-risk sidecar remains the supported uncertainty method, and the conditional A2-C/A2-G ablations must not run.
- Code: `e7c1580c56e558b643260394523cbdf347d5a805` (the manifest recorded the pre-existing untracked experiment logs as a dirty worktree; no production source was modified for this run).
- Canonical A1 base: `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`, SHA-256 `a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607`.
- Protocol: `trusted-v1`, seed 0, effective batch 256, four GPUs, JSFUSION held out from selection. Split manifest: `dataloaders/splits/msrvtt_trusted_v1_seed0.json`, SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`.
- A2-CG weights: CSD `1.0`, grounding `0.05`, VIB `0.0001`; log-variance bounds `[-12, 4]`; only the 263,426 A2 sidecar scalars were trainable.
- Output: `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg`. Initial log: `logs/20260812/115504_csd_logvar_a2_cg_train_msrvtt.log`; continuation log: `logs/20260812/124342_csd_logvar_a2_cg_resume_train_msrvtt.log`.
- The registered three-epoch run completed first. Validation epoch-index 2 had tuple `(-0.147238, 0.509348)`, strictly above epoch-index 1 `(-0.153254, 0.411331)`, so the run resumed from `pytorch_model.bin.2` plus `pytorch_opt.bin.2` in the same output directory to five total epochs. No test metric entered this decision.

### Paired A1 registration and frozen-boundary audit

The paired registration recorded validation retrieval T2V R@1/R@5/R@10 = `75.88/94.78/97.26` and V2T = `93.00/99.80/100.00`. Its immutable digests were:

| Array | Direction | SHA-256 |
|---|---:|---|
| raw WTI | bidirectional source | `2ca359feb192241741ec2b2c270b457b3732cbf667c498650d377b6d74cead29` |
| A1 failure risk | T2V | `4c2c4a58e67b067032283afd297dde670489ea518c31deadada76c3fbd634ae2` |
| A1 failure risk | V2T | `517ab872d5d22eb2b083f91987d4197fe7347d0623c08ab048dfe910102f188b` |
| A1 vacuity | T2V | `e07e07dec325cf427c34ddbdc9a43015642b74735c3083007e9b6818056095c3` |
| A1 vacuity | V2T | `35471aaecf9bd4bbb15c6f3e329e4264aabf2eb37096d1887dd83744db76f48f` |

The selected epoch-index 2 candidate passed exact raw-WTI, retrieval, A1-risk, and A1-vacuity comparisons. Its non-A2 mismatch list was empty, and its checkpoint SHA-256 was `e539108aa8296eaf1e3c71d968117b5f153cd8d7c444282c8b39e9998c8635be`. Thus **A1 remained unchanged in the selected A2 checkpoint**. Continuation candidates 3 and 4 failed paired-invariance and were invalid, despite finite metrics, zero non-A2 state-dict mismatches, and acceptable clamps; they were not eligible for selection.

### Training diagnostics

The table averages the logged 20-step CSD windows within each epoch. `Loss` is the epoch-average total training loss.

| Epoch | Loss | BCE + | BCE - | Ground fit | Ground rank | VIB | Mean dist. | Var. dist. | Scale | Shift | Text clamp L/H | Video clamp L/H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0722 | 6.924897 | 0.001015 | 0.166800 | 0.693100 | 3.510152 | 2.068664 | 0.337536 | 4.955312 | 4.999888 | 0/0 | 0/0 |
| 2 | 0.0709 | 6.455121 | 0.001609 | 0.166852 | 0.693100 | 3.593164 | 2.068652 | 0.286400 | 4.862979 | 4.999279 | 0/0 | 0/0 |
| 3 | 0.0703 | 6.198494 | 0.002070 | 0.166900 | 0.693100 | 3.660836 | 2.068652 | 0.250403 | 4.827533 | 4.999003 | 0/0 | 0/0 |
| 4 | 0.0701 | 6.059503 | 0.002391 | 0.166900 | 0.693100 | 3.704142 | 2.068691 | 0.230367 | 4.808961 | 4.998879 | 0/0 | 0/0 |
| 5 | 0.0699 | 5.939279 | 0.002691 | 0.166900 | 0.693100 | 3.746279 | 2.068644 | 0.212159 | 4.794556 | 4.998785 | 0/0 | 0/0 |

### Validation selection and acceptance

Selection used only valid validation candidates and maximized `(min directional rho, mean directional AUROC)`, with earlier epoch as the final tie-break. The winner was **epoch index 2 (human epoch 3)**.

| Epoch index | Valid | Paired invariant | T2V rho | V2T rho | T2V AUROC | V2T AUROC | Acceptance |
|---:|:---:|:---:|---:|---:|---:|---:|:---:|
| 0 | yes | yes | -0.208021 | -0.008704 | 0.367918 | 0.482862 | fail |
| 1 | yes | yes | -0.153254 | -0.043909 | 0.402586 | 0.420076 | fail |
| **2** | **yes** | **yes** | **-0.147238** | **0.061364** | **0.406392** | **0.612305** | **fail** |
| 3 | no | no | -0.135875 | 0.030122 | 0.413804 | 0.555363 | fail |
| 4 | no | no | -0.134939 | -0.027107 | 0.414401 | 0.451429 | fail |

For the selected checkpoint, validation acceptance checks were: rho **fail**, AUROC **fail**, monotone selective accuracy **fail**, non-zero variance **pass**, clamp rate **pass**. Finiteness, clamp validity, paired invariance, and the empty non-A2 mismatch audit all passed. Overall A2-CG acceptance was **false**.

### One-shot JSFUSION test (descriptive only)

These test results describe the validation-selected epoch-index 2 checkpoint; they did not affect continuation, selection, or acceptance.

| Direction | Retrieval R@1/R@5/R@10 | Intrinsic rho | Intrinsic AUROC (95% CI) | AURC | Accuracy @ 25/50/75/100% coverage |
|---|---|---:|---:|---:|---|
| T2V | 52.90/77.30/85.30 | -0.164629 | 0.427073 (0.390808–0.461048) | 0.533536 | 0.444/0.476/0.505333/0.529 |
| V2T | 52.27/77.87/85.57 | 0.028813 | 0.513301 (0.477634–0.547934) | 0.461412 | 0.544/0.548/0.529333/0.529 |

For reference, the unchanged A1 sidecar on this same JSFUSION pass retained strong uncertainty ranking: T2V rho/AUROC/AURC `0.562229/0.789520/0.245835`; V2T `0.492443/0.765230/0.256541`.

Selected-checkpoint intrinsic distributions and CSD distance components on JSFUSION were:

| Quantity | Min | Max | Mean | Std | Clamp low/high |
|---|---:|---:|---:|---:|---:|
| Text logvar | -8.272092 | -8.217414 | -8.253934 | 0.009380 | 0/0 |
| Text variance | 0.000255550 | 0.000269912 | 0.000260244 | 0.000002450 | 0/0 |
| Video logvar | -8.417719 | -8.417622 | -8.417668 | 0.000014648 | 0/0 |
| Video variance | 0.000220918 | 0.000220940 | 0.000220929 | 0.00000000324 | 0/0 |
| CSD mean distance | 1.877068 | 2.230935 | 2.069375 | 0.041264 | n/a |
| CSD variance distance | 0.243952 | 0.251316 | 0.246361 | 0.001255 | n/a |

### Registered conclusion

**A2-CG did not pass.** It missed both bidirectional correlation and bidirectional AUROC gates and failed the monotone selective-accuracy check, even though variance was non-zero and no clamping occurred. The selected checkpoint preserved A1 exactly, so **A1 remained unchanged and is retained**. Per the registered stop rule, A2-C and A2-G are not launched; intrinsic Gaussian uncertainty is unsupported by this run.
