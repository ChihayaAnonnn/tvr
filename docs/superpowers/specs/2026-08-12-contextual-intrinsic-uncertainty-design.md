# Contextual and Intrinsic Retrieval Uncertainty Design

**Date:** 2026-08-12

## 1. Goal

Replace the failed requirement that one candidate-independent Gaussian `logsigma`
must explain candidate-dependent retrieval failure. The new design keeps the
retriever and the accepted A1 sidecar frozen, and adds two separately trained and
reported uncertainty signals:

1. **Contextual retrieval risk v2 (A3):** predicts top-1 failure from a query,
   its competing candidates, and the frozen WTI score distribution.
2. **Intrinsic concentration (I1):** predicts candidate-independent input
   ambiguity from a frozen representation using a scalar spherical
   concentration rather than a redundant 512-dimensional diagonal variance.

The implementation must preserve the selected retrieval checkpoint's raw WTI,
DSL retrieval metrics, A1 failure risk, and A1 vacuity exactly under the paired
deterministic FP32 evaluation protocol.

## 2. Evidence motivating the design

The registered CSD A2-CG run was numerically stable but did not learn a risk
ordering. Its grounding fit stayed near `0.1669`, its pairwise ranking term stayed
near `0.6931 = ln(2)`, and variance distance decreased from `0.3375` to `0.2122`.
All 512 rows of each final log-variance projection remained exactly identical.
Thus the optimizer learned a low, nearly constant isotropic variance.

The failure has three structural causes:

- CSD adds variance to every pair distance, so positive matching rewards globally
  reducing variance.
- `mean(exp(logvar))` was about `2.6e-4` while the grounding target was in
  `[0, 1]`; the exponential and the `1/512` mean weakened the grounding gradient.
- retrieval failure depends on the candidate set, while A2 only observed one
  pooled mean embedding.

The design follows the task-context principle in ViLU, the post-hoc frozen-model
separation in ProbVLM, and the spherical concentration treatment in
probabilistic contrastive learning. PCME++ remains a reference for full joint
probabilistic embedding training, not a justification for training variance alone
with frozen means.

## 3. Scope and non-goals

### In scope

- Add A3 and I1 as new opt-in modules with independent flags, checkpoints,
  metrics, manifests, and launchers.
- Freeze every pre-existing parameter during A3 or I1 training.
- Reuse the trusted-v1 split, deterministic FP32 paired audit, append-only
  checkpoint selection, and validation-only continuation rules.
- Run A3 first. Run I1 only after its unit/integration tests pass; A3 acceptance
  does not depend on I1.
- Report A1, A3, and I1 separately. A combined score is descriptive only in this
  iteration.

### Out of scope

- Fine-tuning the June/UATVR retrieval backbone.
- Replacing WTI or DSL with a probabilistic distance.
- Claiming I1 is a retrieval-failure probability.
- Re-running the failed CSD-C/G ablations.
- Repeated JSFUSION checkpoint selection or test-time tuning.
- A full PCME++ reproduction with jointly trainable mean encoders.

## 4. A3: contextual retrieval risk v2

### 4.1 Inputs and architecture

A3 is a shared directional sidecar. For T2V it consumes a text query and video
candidates; for V2T it consumes a video query and text candidates. It receives
detached tensors only.

For each query, construct:

- normalized query mean `q`;
- top-1 candidate mean `c1`;
- softmax-attended top-K candidate context `c_ctx`;
- the existing eight A1 score-distribution features;
- top-K normalized scores and gaps, padded deterministically when necessary;
- a two-dimensional direction code.

The candidate context is:

```text
indices = topk(scores, K)
weights = softmax(scores[indices] / temperature)
c_ctx = sum(weights * candidate_means[indices])
```

Modality-specific linear projections map query/candidate means to a shared
128-dimensional space. The concatenation
`[q_proj, c1_proj, c_ctx_proj, score_features, topk_features, direction]`
passes through a small MLP and produces one failure logit. Default `K=16`.

This is deliberately candidate-conditioned. It is named contextual risk, never
intrinsic variance or `logsigma`.

### 4.2 Training objective

Training uses the existing four-GPU global in-batch WTI matrix and multi-positive
mask. For each direction:

- `failure = 1` when the top-1 candidate is not positive;
- weighted BCE directly supervises failure, with a bounded batch-adaptive
  positive weight;
- Smooth-L1 fits `sigmoid(logit)` to robust-rank-normalized detached retrieval
  hardness;
- pairwise ranking orders predictions by detached hardness.

The loss is the mean of T2V and V2T losses:

```text
L_A3 = L_weighted_BCE + 0.25 * L_fit + 0.25 * L_rank
```

All targets are computed from detached scores. No A3 gradient may reach WTI,
the backbone, A1, CSD-A2, or I1.

### 4.3 Evaluation and acceptance

A3 is evaluated against full validation candidate matrices. Report for both
directions:

- Spearman rho against positive rank;
- failure AUROC with bootstrap 95% CI;
- AURC;
- accuracy at 25/50/75/100% coverage;
- output mean/std/min/max and saturation rate;
- paired retrieval and A1 digests.

A3 passes only if all conditions hold on the selected trusted-v1 validation
checkpoint:

- both directional rho values are at least `0.20`;
- both AUROC confidence-interval lower bounds exceed `0.50`;
- accuracy is monotonically non-increasing as coverage grows from 25% to 100%;
- prediction standard deviation is at least `0.02` in both directions;
- all paired invariance and non-A3 state-dict audits pass.

Selection maximizes `(min directional rho, mean directional AUROC, -mean AURC)`
over valid candidates with earlier epoch as the final tie-break.

## 5. I1: intrinsic spherical concentration

### 5.1 Representation

I1 predicts one scalar raw concentration for text and one for video:

```text
kappa = softplus(raw_kappa) + 1e-4
u_intrinsic = 1 / (1 + kappa)
```

This matches the normalized hyperspherical embedding geometry and removes the
unsupported claim of a learned 512-dimensional anisotropic covariance. Each
modality uses an independent two-layer MLP with nonzero Xavier initialization.
The head consumes only the sample's detached pooled mean plus sample-local
dispersion summaries. It never consumes retrieval candidates, WTI logits, ranks,
A1, or A3.

Sample-local summaries are computed from already available normalized token/frame
features and masks: mean cosine disagreement with the pooled mean, standard
deviation of that disagreement, valid-token/frame fraction, and norm statistics.

### 5.2 Intrinsic targets and loss

Two stochastic semantic-preserving pooled views are generated from the already
computed token/frame features, avoiding a second encoder forward:

- text: independently drop valid non-special tokens and re-pool;
- video: independently drop valid frames and re-pool;
- each view must retain at least one valid element;
- sampling is seeded and reproducible per rank/step.

The detached perturbation disagreement is:

```text
d_view = 1 - cosine(view_1, view_2)
target = robust_rank_normalize(d_view)
```

The training loss is:

```text
L_I1 = SmoothL1(u_intrinsic, target)
       + pairwise_ranking(u_intrinsic, d_view)
       + 0.01 * concentration_prior
```

The prior is a weak penalty on extreme `log(kappa)` values. There is no CSD term,
no pairwise matching BCE, and no VIB term. This prevents the old global
variance-shrink shortcut.

### 5.3 Evaluation and acceptance

I1 has a different scientific claim from A3. On trusted-v1 validation report:

- correlation with held-out stochastic-view disagreement;
- monotonic response to registered text-token and video-frame dropout severity;
- output distribution and boundary/saturation rates;
- retrieval-rank rho/AUROC as secondary descriptive metrics only;
- paired retrieval, A1, and A3 invariance where applicable.

I1 passes its intrinsic claim only if:

- held-out disagreement Spearman rho is at least `0.40` for both modalities;
- mean uncertainty is monotonic over dropout severities `0, 0.1, 0.2, 0.3`;
- endpoint increase from severity 0 to 0.3 is at least `0.05` after `[0,1]`
  parameterization;
- uncertainty standard deviation is at least `0.02` for both modalities;
- saturation below `1e-4` or above `1-1e-4` is under `5%`;
- all paired invariance and non-I1 state-dict audits pass.

Failure to predict retrieval rank does not invalidate the intrinsic claim, but it
must be disclosed. I1 must not be described as a failure predictor unless it
independently meets the A3 retrieval acceptance gate.

## 6. Training and checkpoint protocol

### A3 registered run

- Initialize from canonical A1
  `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`.
- Train only A3 for three epochs, effective batch 256, seed 0, four GPUs, FP32.
- Continue to five total epochs only when epoch index 2 is strictly better than
  epoch index 1 under the registered A3 selection tuple.
- Do not evaluate JSFUSION unless the selected validation checkpoint passes A3.

### I1 registered run

- Initialize from the same canonical A1, not from A3, so intrinsic evidence is
  independently attributable.
- Train only I1 for three epochs with the same continuation rule using its
  intrinsic selection tuple.
- Do not evaluate JSFUSION unless the intrinsic validation gate passes.

### Test hygiene

Each passing method receives exactly one JSFUSION evaluation from its
validation-selected checkpoint. A failed validation run produces no JSFUSION
result. Test metrics never influence continuation, selection, calibration, or
acceptance.

## 7. Interfaces and files

New focused modules:

- `prob_models/contextual_retrieval_risk.py`: A3 features, network, and loss.
- `prob_models/intrinsic_concentration.py`: I1 network, local summaries, seeded
  stochastic views, and loss.
- `prob_models/uncertainty_protocol.py`: shared append-only artifact audit,
  paired invariance, selection, acceptance, and distribution validation.

Existing integration points:

- `modules/modeling_mulit.py`: opt-in construction, detached training calls, and
  evaluation auxiliary outputs.
- `main_task_retrieval.py`: CLI, freeze rules, deterministic evaluation,
  directional metrics, manifests, selection, continuation, and final test gate.
- `experiment_tracking.py`: canonical reference hash and manifest sections.

Launchers:

- `run_train_msrvtt_contextual_risk_bg.sh`
- `run_train_msrvtt_intrinsic_concentration_bg.sh`

Old A1/CSD flags and checkpoint loading remain backward compatible. New
checkpoints must never silently activate A3 or I1 without their explicit flags.

## 8. Diagnostics and failure handling

Every 20 steps log:

- total and component losses;
- target and prediction mean/std;
- BCE class balance and adaptive weight;
- ranking valid-pair fraction;
- gradient norm of each trainable sidecar;
- output saturation;
- A3 top-K entropy or I1 view disagreement statistics.

Abort a run on non-finite values, a zero trainable gradient for two consecutive
windows, missing positives/negatives, artifact overwrite, hash mismatch, or any
non-sidecar state change. Constant outputs are an acceptance failure, not a
runtime retry condition.

## 9. Tests

Unit tests cover:

- directional top-K feature construction and padding;
- failure labels with multi-positive masks;
- weighted BCE and ranking direction;
- scalar concentration bounds and non-identical initialization;
- seeded view generation, minimum retained elements, and monotonic synthetic
  disagreement;
- rejection of invalid weights, temperatures, K, and masks.

Integration tests cover:

- exact trainable parameter sets for A3-only and I1-only modes;
- frozen modules staying in eval mode;
- no gradient reaching WTI, backbone, A1, or the other new sidecar;
- four-GPU gather shape semantics;
- checkpoint round-trip and legacy checkpoint compatibility;
- paired digests and non-sidecar state-dict equality.

Protocol tests cover append-only artifacts, resume lineage, deterministic
selection, continuation, acceptance gates, and the no-test-on-failure rule.

## 10. Experiment decision table

| Outcome | Action |
|---|---|
| A3 passes | One JSFUSION test; retain A3 alongside A1 |
| A3 fails with nonconstant outputs | Stop; candidate context/objective is insufficient |
| A3 fails with constant outputs | Diagnose scale/optimization before any new architecture |
| I1 passes intrinsic gate | One corruption/JSFUSION descriptive evaluation; support intrinsic claim |
| I1 fails intrinsic gate | Stop scalar intrinsic branch; do not return to CSD weight tuning |
| Both pass | Report separately; optional combined-score study requires a new registration |

## 11. References and code

- ViLU: Learning Vision-Language Uncertainties for Failure Prediction, ICCV 2025:
  <https://arxiv.org/abs/2507.07620>, <https://github.com/ykrmm/ViLU>
- ProbVLM: Probabilistic Adapter for Frozen Vision-Language Models, ICCV 2023:
  <https://arxiv.org/abs/2307.00398>, <https://github.com/ExplainableML/ProbVLM>
- Probabilistic Contrastive Learning Recovers the Correct Aleatoric Uncertainty
  of Ambiguous Inputs, ICML 2023: <https://arxiv.org/abs/2302.02865>,
  <https://github.com/mkirchhof/Probabilistic_Contrastive_Learning>
- Improved Probabilistic Image-Text Representations (PCME++), ICLR 2024:
  <https://arxiv.org/abs/2305.18171>, <https://github.com/naver-ai/pcmepp>
- Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment
  Retrieval (DEMR), AAAI 2026: <https://arxiv.org/abs/2512.00953>,
  <https://github.com/KaijingOfficial/DEMR>
- Post-hoc Probabilistic Vision-Language Models (BayesVLM):
  <https://arxiv.org/abs/2412.06014>, <https://github.com/AaltoML/BayesVLM>
