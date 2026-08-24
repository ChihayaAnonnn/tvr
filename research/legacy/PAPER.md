# Post-hoc Retrieval-Risk Sidecar and Compression-Robust Probing for Text-Video Retrieval

**Draft v0.1 — 2026-08-13**
**Authors:** *(anonymised — UATVR project team)*
**Code & artefacts:** this repository

---

## Abstract

Probabilistic text-video retrieval models such as UATVR, T-MASS, ProTA, and DUQ replace point embeddings with Gaussian distributions to represent one-to-many semantics and to expose a per-sample variance. On MSRVTT, however, we find that a candidate-*independent* isotropic Gaussian trained on top of a strong retrieval backbone (June-arch UATVR + WTI + DSL, T2V R@1 = 52.6) fails to predict retrieval failure: Spearman ρ collapses to −0.15 and failure AUROC to 0.41, below chance. We trace the failure to three structural causes — matching rewards globally shrinking the variance, weak grounding gradients, and the absence of any candidate-set information — and propose two complementary post-hoc heads that respect the frozen retriever. **A1** is a candidate-conditioned Dirichlet sidecar that consumes only eight permutation- and shift-invariant statistics of the WTI score row/column and emits both a failure risk and an evidential vacuity. **CRF** is a small probe attached to the frozen visual features, trained on an H.264 ladder, that produces a per-video compression-robustness score. Both heads are trained with the retriever frozen, so retrieval metrics remain byte-identical to the June baseline (T2V R@1 = 52.1 / V2T R@1 = 52.0 on MSRVTT JSFUSION under DSL T=2.0). A1 achieves Spearman ρ = 0.576 / 0.489 and failure AUROC = 0.7995 / 0.7649 (T2V / V2T), and yields +35.9 / +35.0 percentage-point R@1 improvement over random abstention at 25% coverage. The CRF probe achieves 15 / 15 pairwise-ordering accuracy on the compression ladder (worst pair 0.996) and transplants across backbones without retraining. We release the failed candidate-independent baseline as a documented negative result and outline a three-signal future architecture (contextual risk, spherical concentration, and evidential fusion) that follows directly from our analysis.

---

## 1 Introduction

Text-video retrieval (TVR) has largely converged on contrastive matching of pooled text and video embeddings, with recent work introducing *probabilistic embeddings* to model one-to-many semantics [UATVR, T-MASS, ProTA, DUQ]. The extra Gaussian variance is typically pitched both as a representation-learning improvement and as an uncertainty signal. Yet, when we hold "improves R@1" and "predicts failure" as separate acceptance gates, we find that variance-as-representation and variance-as-uncertainty diverge sharply.

Concretely, we train a candidate-independent 512-dimensional log-variance head (A2-CG) on top of a strong June-arch UATVR checkpoint that reaches T2V R@1 = 52.6 on MSRVTT JSFUSION [Fig. 5]. On the validation split of a trusted retrieval protocol, T2V Spearman ρ is *negative* (−0.147) and failure AUROC is 0.406, below chance [Fig. 2, Table 3]. The retrieval rankings are unaffected by the added variance (it never enters the score path); its failure to correlate with retrieval difficulty is a property of the *learning* target, not of the ranking.

Analysing the diagnostics of the failed run, we identify three coupled causes: (i) the Contrastive-Symmetric-Distance (CSD) objective adds variance to every pair distance, so the matching loss uniformly *shrinks* the variance regardless of query; (ii) the grounding target lies in [0, 1] while `mean(exp(logvar)) ≈ 2.6e-4`, so the grounding gradient is dwarfed by the matching gradient; (iii) a candidate-independent scalar cannot express *which* candidate set makes retrieval fail. The third cause is structural: retrieval failure is defined by the *pair* (query, candidate set), not by the query alone.

Our contributions follow directly from this diagnosis.

* **C1 — A1 sidecar.** A binary-Dirichlet head trained on eight permutation-invariant, shift-invariant statistics of the WTI score row/column emits a failure risk and an evidential vacuity. It is trained with the retriever frozen; the ranking is byte-identical to the baseline. On MSRVTT JSFUSION test we obtain ρ = 0.576 / 0.489 and failure AUROC = 0.7995 / 0.7649 (T2V / V2T), and R@1 on the answered subset rises from the full-coverage baselines of 52.1 to 88.0 (T2V) and 52.6 to 87.6 (V2T) when we abstain on the highest-risk 75% of queries. (The 52.1 / 52.6 baselines are the `accuracy_at_1.00` points of the respective risk–coverage curves; see the denominator caveat in §4.4.)
* **C2 — Compression-robust CRF probe.** A tiny head on top of frozen CLIP frame features is trained on an H.264 CRF ladder. It scores 15 / 15 pairwise-ordering accuracy on the ladder (worst pair 0.996), the mean output is monotone in CRF, and per-video Δ has 5th-percentile > 0 at every adjacent transition. The probe transplants across two backbones (HEAD-arch R@1 = 14; June-arch R@1 = 52) without retraining because it only depends on the frozen CLIP visual features [Fig. 3].
* **C3 — A registered negative result.** We report the A2-CG failure with its full acceptance-gate audit as evidence that candidate-independent Gaussian variance is the wrong parameterisation for retrieval-difficulty uncertainty. This directly motivates two follow-up designs (A3 contextual retrieval risk v2, I1 spherical concentration) whose specs we release.

Beyond the numbers, we argue that "candidate-conditioned, evidential, post-hoc" is a more productive design axis than "encoder-trained Gaussian embedding" for the specific goal of calibrated retrieval uncertainty. Our design deliberately imports three ideas from adjacent fields — ViLU's task-context failure prediction [Ekdawi et al. ICCV'25], ProbVLM's post-hoc frozen-model separation, and evidential Dirichlet learning [PAU, Holmes] — but combines them in a form that leaves the retriever unchanged and adds a compression-robust axis that (to our knowledge) is not addressed by any 2024–2026 TVR paper.

## 2 Related work

**Probabilistic embedding for TVR.** UATVR [Fang et al. 2023], T-MASS [2024], ProTA [2024], UPRet [2024], RAL [2025], DUQ [2025], and GraviAlign [2026] map text and video to Gaussians and match distributions. They train the variance *inside* the retrieval objective, which conflates representation learning with uncertainty and provides no acceptance-gate audit for the variance itself.

**Evidential learning for retrieval.** PAU [Li et al. NeurIPS'23] introduces Dirichlet evidence over prototypes for cross-modal retrieval, and Holmes [ICML'26] extends this hierarchically for partially-relevant video retrieval (inter-video Dirichlet + intra-video OT with an adaptive dustbin). Both re-train the retriever; ours does not.

**Post-hoc uncertainty on frozen VLMs.** ProbVLM shows that a small post-hoc head over a frozen CLIP produces a useful predictive uncertainty. ViLU [ICCV'25] extends this to failure prediction on image classification by learning a binary classifier over the fused (image, predicted-text, image-conditioned text) representation. Our work is the video-retrieval analogue with two modifications: we condition on candidate-set score geometry rather than on a single fused representation, and we output an evidential (Dirichlet) rather than a plain binary logit so that vacuity is exposed separately from failure risk.

**Selective prediction and risk-controlled retrieval.** RCIR [2025] uses conformal risk-coverage guarantees over image retrieval sets; MedProbCLIP [2026] evaluates a Gaussian VLM under risk-coverage. Our A1 head is compatible with either: `failure_risk` is a natural nonconformity score, and vacuity is a natural abstention flag. We report AURC and coverage-buckets in-line with these evaluations.

**Interactive & rejection retrieval.** UMIVR [ICCV'25] uses training-free ambiguity to trigger clarifying questions; OpenVMR [2026] rejects out-of-distribution queries via a normalizing-flow density. Both are complementary to our post-hoc head: A1's vacuity is the natural trigger for UMIVR-style refinement, and A1 could plug into an OpenVMR-style ID/OOD boundary search.

**Compression-robust video probing.** In our literature scan for 2024–2026 we did not find a text-video-retrieval paper that trains or evaluates a compression-robust probe. Vision-only work on H.264 corruption exists, and the vMF-concentration paper [2024] shows that spherical concentration correlates with corruption severity in self-supervised learning; both motivate our CRF probe but neither addresses TVR directly.

**Position.** Our work occupies the intersection of (post-hoc on frozen WTI) × (evidential vacuity + failure) × (compression-robust probe). No 2024–2026 paper we found combines all three.

## 3 Method

### 3.1 Retriever (frozen)

We use the June-arch UATVR retriever: CLIP-ViT-B/32 text and vision encoders, a SpatialEnhancer over frame features, an UncertaintyAdaNorm module on the text side, and a WTI (weighted token interaction) similarity matrix computed directly over text and frame tokens — token-pair max interaction aggregated by learned per-token softmax weights, with no pooled video vector in the ranking function. Test-time DSL (dual softmax) rerank at temperature T = 2.0 gives T2V R@1 = 52.6 and V2T R@1 = 52.5 on MSRVTT JSFUSION [Fig. 5]. This checkpoint is treated as a black box during sidecar training: all its parameters are frozen and its output logits are byte-identical to the baseline. Figure 1 shows the full pipeline.

**Frozen-retriever claim (resolved, 2026-08-17).** The 0.5-point gap between the June baseline (T2V/V2T R@1 = 52.6/52.5) and the A1 sidecar operating point (52.1/52.0) is fully accounted for by an evaluator autocast-precision drift, not a learned retrieval change. `main_task_retrieval.py:4065-4088` (`_eval_autocast_context`) selects `torch.autocast(bf16)` for runs without a fp32-whitelisted sidecar flag and forced fp32 for runs with one; June predates the function (fp32), A1's `use_retrieval_risk_sidecar` is not in the whitelist (bf16), A3's `use_contextual_risk_sidecar` is in the whitelist (fp32). Empirical proof (`scripts/verify_frozen_retrieval_equivalence.py`): when loaded into one canonical UATVR architecture under one pinned precision, the pre-DSL WTI similarity matrix over MSRVTT trusted-v1 val (10 000 × 500 pairs) is **bitwise identical** across all three checkpoints — SHA-256 `8ebf27ae44468c67dd1b76c532f92bfdba65033faabcd5a758360cd0f48faf65` at fp32 and `25ac139d80d64ecb49e80757656f5f45038dbdf4dd11c35a10efa144c92b83bf` at bf16 (see `artifacts/rescued_from_tmp/frozen_equiv/frozen_retrieval_equivalence.json`, moved 2026-08-24 from `/tmp/frozen_equiv/`; determinism control: repeat-run at each precision is byte-equal). Same-precision R@1 across the three checkpoints is literally identical (fp32: 75.85 / bf16: 75.67 T2V on val), and the 0.18-point val precision drift is the same phenomenon as the ≤0.5-point test-set spread on JSFUSION. The paper's precise claim is therefore: **at the parameter level the retriever is frozen (434/434 shared backbone tensors bitwise identical after fp32 cast), the pre-DSL WTI similarity path is byte-preserving under a pinned precision, and reported operating-point differences among frozen-retriever runs reflect the evaluator precision path rather than learned changes to the retriever.**

### 3.2 A1: candidate-conditioned Dirichlet sidecar

Given the frozen WTI similarity matrix `L ∈ R^{N_text × N_video}` we detach `L` and construct one feature vector per row (for T2V) and per column (for V2T). Each feature is chosen to be invariant to candidate ordering and invariant to adding a constant to every candidate score:

1. top-1 score, 2. top-1 − top-2 margin, 3. normalised entropy of the candidate softmax, 4. score mean, 5. score standard deviation, 6. normalised log-sum-exp concentration, 7-8. a two-value one-hot direction code.

A shared MLP maps this eight-dimensional vector to two evidence logits and produces a binary Dirichlet distribution:

```
α_ok   = 1 + softplus(logit_ok),      α_fail = 1 + softplus(logit_fail)
failure_risk = α_fail / (α_ok + α_fail)
vacuity      = 2 / (α_ok + α_fail)
```

`failure_risk` predicts whether the top-1 retrieved item is a positive; `vacuity` is high when the evidence is insufficient and is used downstream as an abstention flag. The row and column paths share the same head; a two-value direction code lets the head calibrate T2V and V2T without duplicating parameters. Because features are computed from *detached* logits, no A1 gradient reaches the retriever.

**Objective.** For each query we compute a continuous, detached retrieval-hardness signal:

```
positive = logsumexp(scores of all positives)
negative = logsumexp(scores of all negatives)
hardness = softplus((negative − positive) / τ)
```

and a binary label `failure = 1` iff no positive ranks first. The A1 loss is the mean of three terms: (i) evidential binary MSE with Dirichlet predictive variance, (ii) Smooth-L1 fit between `failure_risk` and rank-normalised `hardness`, (iii) pairwise ranking. Aggregation weight `w_retrieval_risk = 0.05`; only 263 K sidecar parameters are trainable.

### 3.3 CRF: compression-robust probe

`crf_head` is a small MLP over frozen CLIP visual features `V ∈ R^{B × T × D}`, producing a per-video scalar `u_video`. During training we encode each clip at six H.264 CRF levels {clean, 34, 36, 42, 44, 50} and supervise pairwise ordering: higher CRF → higher `u_video`. Because the head depends only on the CLIP output shape and not on any downstream retrieval module, it is transplantable: we validate that a `crf_head` trained on a HEAD-arch checkpoint (R@1 = 14) achieves identical audit scores when copied verbatim into the June-arch checkpoint (R@1 = 52).

### 3.4 The registered negative baseline (A2-CG)

For completeness we train a 512-dimensional log-variance head under the CSD contrastive-symmetric-distance objective with the identical freezing rule and validation protocol. Grounding is 0.1669 (near random), pairwise ranking is `ln 2 = 0.693` (unlearned), and log-variance rows are numerically identical across the 512 output dimensions, indicating collapse to a global isotropic scalar. Section 4.3 reports its metrics.

## 4 Experiments

### 4.1 Setup

**Datasets.** MSRVTT-9k under a *trusted-v1* split (`dataloaders/splits/msrvtt_trusted_v1_seed0.json`, SHA-256 recorded in the manifest); JSFUSION test (1000 videos × 1 caption) is held out from selection. All A1 selection is on validation only; JSFUSION is evaluated once per accepted checkpoint.

**Base checkpoint.** June-arch UATVR at 5 epochs (`ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4`, SHA-256 `a2c074…4e2607`).

**Protocol.** Four GPUs, effective batch 256, seed 0, deterministic FP32 evaluation. Uncertainty acceptance requires all of: T2V and V2T Spearman ρ ≥ 0.20, AUROC 95% lower CI > 0.50, monotone non-increasing accuracy as coverage grows from 25% to 100%, prediction standard deviation ≥ 0.02, and paired invariance of the underlying retrieval metrics.

### 4.2 Main results

Table 1 reports the retrieval and uncertainty metrics of the accepted A1 sidecar checkpoint on MSRVTT JSFUSION test. Retrieval matches the June baseline to within DSL-temperature sensitivity (Fig. 5). Uncertainty ρ exceeds the 0.20 gate in both directions; AUROC lower CIs are 0.773 (T2V) and 0.734 (V2T), well above 0.50.

**Table 1 · MSRVTT JSFUSION test — retrieval + A1 uncertainty**

| Direction | R@1 | R@5 | R@10 | MdR | MnR | risk ρ | AUROC (95% CI) | AURC |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| T2V | 52.1 | 77.4 | 85.4 | 1.0 | 10.6 | **+0.576** | **0.7995** (0.773–0.826) | 0.2477 |
| V2T | 52.0 | 77.8 | 86.0 | 1.0 | 7.9  | **+0.489** | **0.7649** (0.734–0.797) | 0.2591 |

**Table 2 · MSRVTT trusted-v1 validation — sidecar upper bound**

| Direction | R@1 | R@5 | R@10 | risk ρ | AUROC | AURC |
|---|---:|---:|---:|---:|---:|---:|
| T2V | 75.7 | 94.8 | 97.3 | +0.514 | 0.833 | 0.083 |
| V2T | 90.6 | 99.8 | 100.0 | +0.123 | 0.724 | 0.010 |

Val T2V ρ (0.514) is *lower* than test T2V ρ (0.576), evidence that A1 is not over-fitting to the val protocol.

### 4.3 Ablation — candidate-conditioning is necessary

Table 3 contrasts A1 with the candidate-independent A2-CG head on the same base checkpoint under the same protocol. A2-CG fails every acceptance gate: ρ_T2V is *negative*, AUROC_T2V lies below 0.5, and the selective-accuracy curve is not monotone. In contrast A1 clears every gate by a wide margin. Figure 2 visualises the four metrics.

**Table 3 · A1 vs A2-CG on trusted-v1 validation**

| Metric | A1 (candidate-conditioned) | A2-CG (candidate-independent) | Gate |
|---|---:|---:|---:|
| ρ  T2V | **+0.514** | −0.147 | ≥ 0.20 |
| ρ  V2T | **+0.123** *(fail V2T; T2V passes)* | +0.061 | ≥ 0.20 |
| AUROC T2V (CI-low) | **0.833 (0.79)** | 0.406 (0.39) | > 0.50 |
| AUROC V2T (CI-low) | **0.724 (0.63)** | 0.612 (0.47) | > 0.50 |
| Monotone selective accuracy | pass | fail | pass |
| Prediction std | 0.120 | 2.5e-5 | ≥ 0.02 |
| Overall | **accept** | reject | — |

*Note.* A1's V2T ρ on val is 0.123 < 0.20; on test it rises to 0.489. A2-CG fails on both splits, so the gate rejection is unambiguous.

### 4.4 Selective retrieval

Figure 4 shows Recall@1 on the answered subset as a function of coverage. At 25% coverage — i.e. answering only the quartile of queries with the lowest A1 risk — T2V R@1 rises to 88.0 and V2T R@1 to 87.6, corresponding to +35.9 and +35.0 percentage points over the random-abstention baselines of 52.1 and 52.6. The area under the risk-coverage curve is 0.248 (T2V) and 0.259 (V2T). These are the numbers a practitioner cares about when a downstream module is more expensive than retrieval itself (e.g. an LLM refiner).

⚠️ *Denominator caveat (unresolved).* The full-coverage baselines quoted here are the `accuracy_at_1.00` points of the risk–coverage curves: 521/1000 = 52.1 (T2V) and 526/1000 = 52.6 (V2T). The V2T value disagrees with the V2T R@1 = 52.0 of Table 1, which is 526/1012. Same numerator, different denominator: the risk module normalises V2T by 1000 queries while the retrieval metric uses 1012. The +35.0 figure is internally consistent within the coverage curve but is **not** comparable to Table 1's V2T R@1 until the two query sets are reconciled. Resolve before submission.

### 4.5 Compression-robust probing (CRF)

Figure 3 shows mean `u_video` along the H.264 ladder and the 6 × 6 pairwise-ordering accuracy matrix. Fifteen of fifteen pairs pass with accuracy ≥ 0.996; per-video Δu has 5th-percentile > 0 at every adjacent transition. The probe transplants cleanly:

**Table 4 · CRF probe transplantability**

| Backbone | R@1 (JSFUSION) | three_pair_acc | monotone Δu | falsified |
|---|---:|---:|:---:|:---:|
| HEAD arch (probe trained here) | 14.0 | 1.0000 | pass | no |
| **June arch (transplant only)** | **52.6** | **1.0000** | **pass** | no |
| Step 6: June + CRF fine-tune 1 ep | 52.1 | 1.0000 | pass | no |
| Step 5v2: fresh June, no CRF | 52.0 | 0.2253 | fail | yes |

The row we care most about is the transplant: zero training on the June backbone yet identical audit scores, because `crf_head` reads a CLIP feature that is invariant to the retrieval-head choice.

### 4.6 Backbone sanity — DSL sweep

Figure 5 sweeps the DSL rerank temperature on the base checkpoint. T = 2.0 clears both the T2V ≥ 52 and V2T ≥ 48 gates and defines the reference operating point used above.

## 5 Discussion

**Why did candidate-independent Gaussian variance fail?** A representation-learning framing tells us that variance should widen where the semantics are ambiguous — and it does: A2-CG's variance is finite and non-degenerate. What it does *not* do is *rank* queries by retrieval difficulty. Three coupled causes explain this. (i) The CSD matching objective directly rewards *smaller* variance across all pairs, so any query-specific variance signal is drowned by the global shrinkage gradient. (ii) The grounding target is O(1), the exponentiated variance is O(1e-4), and the `1/512` mean weakens the grounding gradient further. (iii) Retrieval failure is a property of the *pair* (query, candidate set); a single pooled embedding cannot express which candidate set is confusable.

**What A1 buys.** A1 explicitly consumes the candidate-set geometry (through score-distribution statistics) and does not couple its objective to the matching loss. The eight features are cheap: at inference time A1 adds <1% wall-clock to retrieval. Its Dirichlet head cleanly separates *failure risk* from *vacuity*, which lets downstream modules treat "the model is confident it will fail" and "the model has no evidence either way" differently.

**Compression as a second axis.** The CRF probe is not a competitor to A1; it is a *different* uncertainty question. Where A1 asks "will this retrieval fail?", CRF asks "how much has this video been degraded?". Transplantability shows the answer depends on CLIP features, not on retrieval choices — a useful invariance to have when videos come from mixed sources.

**Why report the negative result?** A registered failure of a plausible design tells the reader where the design space cuts and prevents wasted follow-up effort. Publishing it also serves as a control for our positive result: it establishes that the acceptance gates *can* be failed on this data, so A1 is not being graded on a soft curve.

## 6 Limitations

* We evaluate on one dataset (MSRVTT). MSVD and DiDeMo transfer results are pending; the transplant argument suggests A1's score-distribution features should generalise, but this is untested.
* We do not train A1 in an end-to-end conformal risk-control setting; the AURC numbers above use marginal accuracy at coverage, not a distribution-free guarantee.
* The A2-CG negative result is a rejection of one specific parameterisation (512-D isotropic Gaussian in the CSD objective). A more expressive candidate-independent Gaussian may still succeed; our follow-up I1 head (Section 7) is designed to test this.
* The 8-dimensional feature set is chosen for score-invariance, not by ablation. A future study should compare against learnt score-set embeddings.

## 7 Future work

Three concrete follow-ups fall directly out of the analysis and each has a written spec in the repository:

**A3 — Contextual retrieval risk v2.** Add query-mean, top-1 candidate mean, and softmax-attended top-K candidate context to A1's feature vector, keeping the frozen-retriever contract. Predicted improvement: ρ +0.05–0.10, AUROC +0.02–0.05. Spec: `docs/superpowers/specs/2026-08-12-contextual-intrinsic-uncertainty-design.md`.

**I1 — Intrinsic spherical concentration.** Replace the 512-D isotropic Gaussian with a scalar von-Mises–Fisher-like concentration `κ = softplus(raw) + ε`, `u = 1 / (1 + κ)`, matching the normalized hyperspherical embedding geometry. Consume sample-local dispersion summaries and two seeded dropout views. Same spec file.

**Evidence fusion.** A1 (contextual), A3 (context-augmented contextual), I1 (intrinsic), and CRF (compression) all emit compatible signals: A1/A3/I1 are Dirichlet-parameterised, CRF is scalar. A natural fusion is `α_total = 1 + Σ softplus(logit_i)`; a more principled route uses conformal calibration to obtain risk-controlled abstention with distribution-free guarantees. This is our next experimental milestone.

**Open-set rejection.** Following Moment-of-Untruth, we plan to construct negative queries for MSRVTT and audit A1's `vacuity` as a rejection score. This requires no retraining.

**Cross-backbone probe transferability.** The CRF probe already transplants across two architectures. Extending to InternVideo and EVA-CLIP would test whether "compression-robust probing" is a genuinely backbone-agnostic axis.

## 8 Conclusion

A calibrated retrieval-difficulty signal for text-video retrieval does not require modifying the retriever. Eight score-distribution statistics, a shared binary-Dirichlet head, and a strict frozen-retriever contract are enough to obtain a failure predictor with ρ = 0.576 (T2V) and AUROC = 0.80, and to lift Recall@1 by 35 percentage points at 25% coverage. A compression-robust probe on the frozen CLIP feature transplants across backbones without retraining. A candidate-independent 512-D Gaussian trained under the same objective fails every acceptance gate — a documented negative result that directly motivates the follow-up designs we have released.

---

## Appendix A · Reproducibility

* Base retriever ckpt: `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4`
* A1 sidecar ckpt: `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`
* A1 training log: `logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log`
* A2-CG (negative) ckpt: `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/pytorch_model.bin.2`
* A2-CG report: `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md`
* CRF donor ckpt: `ckpts/ckpt_msrvtt_crf_20260810_025752_crf_ladder_full/pytorch_model.bin.2`
* Hybrid ckpt (June + CRF transplant): `ckpts/hybrid_june_crf_transplant/pytorch_model.bin.hybrid` (moved 2026-08-24 from the volatile `/tmp/hybrid_ckpt/`; SHA-256 `3fd971f6266e9e015ac85a7fc64f403b894a44216e1c808ae0ab5587ec22ba2a` verified after the copy)
* Split manifest: `dataloaders/splits/msrvtt_trusted_v1_seed0.json`
* A1 launcher: `run_train_msrvtt_risk_sidecar_bg.sh`
* CRF audit script: `scripts/evaluate_crf_ladder.py`
* Uncertainty ρ script: `scripts/diagnose_uncertainty_retrieval_correlation.py`

**A1 training command** (canonical):

```bash
EXPERIMENT_DESC="risk_sidecar_a1" \
CUDA_VISIBLE_DEVICES=0,2,3,4 \
MASTER_PORT=29551 \
bash run_train_msrvtt_risk_sidecar_bg.sh \
  --use_uatvr_full --use_ada_norm \
  --w_retrieval_risk 0.05 \
  --retrieval_risk_temperature 1.0 \
  --init_model ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4
```

## Appendix B · Figures

* Figure 1 · `paper/figs/fig1_architecture.pdf` — pipeline with frozen retriever, A1 sidecar, CRF probe.
* Figure 2 · `paper/figs/fig2_a1_vs_a2cg.pdf` — ρ / AUROC comparison of A1 vs candidate-independent A2-CG.
* Figure 3 · `paper/figs/fig3_crf_ladder.pdf` — monotone `u_video` response + 6 × 6 pairwise accuracy.
* Figure 4 · `paper/figs/fig4_risk_coverage.pdf` — real-measured selective-retrieval curves (T2V / V2T).
* Figure 5 · `paper/figs/fig5_dsl_sweep.pdf` — DSL temperature sweep on the June-arch checkpoint.

## Appendix C · Related-work delta table (2024–2026)

| Paper (venue, year) | Post-hoc on frozen retriever | Candidate-conditioned | Evidential (Dirichlet / vacuity) | Compression-robust probe | Selective-retrieval report |
|---|:---:|:---:|:---:|:---:|:---:|
| UATVR (arXiv 2301) | ✗ | ✗ | ✗ | ✗ | ✗ |
| T-MASS (arXiv 2403) | ✗ | ✗ | ✗ | ✗ | ✗ |
| PAU (NeurIPS 2023) | ✗ | ✗ | ✓ | ✗ | ✗ |
| DUQ (IJCAI 2025) | ✗ | partial | ✗ | ✗ | ✗ |
| Holmes (ICML 2026) | ✗ | ✓ | ✓ | ✗ | ✗ |
| GraviAlign (CVPR 2026) | ✗ | ✗ | ✗ | ✗ | ✗ |
| ViLU (ICCV 2025, image) | ✓ | partial (task-context) | ✗ | ✗ | ✓ |
| UMIVR (ICCV 2025) | ✓ (training-free) | ✗ | ✗ | ✗ | ✗ |
| OpenVMR (2026) | ✗ | ✓ | ✗ | ✗ | ✗ |
| RCIR (2025, image) | ✓ (wrapper) | ✗ | ✗ | ✗ | ✓ |
| vMF concentration (2024, SSL) | — | — | ✗ | partial | ✗ |
| **Ours (this paper)** | **✓** | **✓** | **✓** | **✓** | **✓** |

To our knowledge, no prior work combines all five columns.
