# Independent CSD Log-Variance Sidecar Design

**Date:** 2026-08-12
**Status:** Pending written-spec review

## Decision

Extend the successful frozen retrieval-risk experiment with an independent,
interpretable Gaussian uncertainty branch. The new branch learns explicit text and
video log-variance from frozen representations using a PCME++ closed-form sampled
distance (CSD) objective and detached retrieval-hardness grounding.

The existing WTI retrieval path and the trained A1 retrieval-risk sidecar remain
frozen. CSD is an auxiliary training objective and is not used to rerank candidates
in this phase. The experiment must therefore answer a narrow question: can an
intrinsic text/video Gaussian variance become aligned with retrieval difficulty
without changing retrieval or the already successful query-conditioned sidecar?

## Motivation

The A1 retrieval-risk sidecar succeeded on the MSRVTT JSFUSION test:

- T2V Spearman `rho=0.5760`, failure AUROC `0.7995`;
- V2T Spearman `rho=0.4892`, failure AUROC `0.7649`;
- low-risk 25% accuracy was `88.0%/87.6%` for T2V/V2T.

However, that sidecar describes the geometry of a complete candidate-score
distribution. It does not establish that the model's Gaussian `logsigma` is an
interpretable, modality-intrinsic uncertainty signal. Earlier SAP `logsigma`
experiments collapsed because their supervision was weak and their log-standard-
deviation/log-variance semantics were inconsistent.

PCME++ defines the exact expected squared distance between independent diagonal
Gaussian embeddings and trains it with pairwise matching BCE. It supplies a direct
gradient to variance without Monte Carlo sampling. Retrieval-hardness grounding
then aligns the learned intrinsic uncertainty with the operational quantity of
interest: actual retrieval difficulty.

Primary references:

- PCME++ paper: https://proceedings.iclr.cc/paper_files/paper/2024/file/ad9d6ab10446114cf5482d5e1f971a84-Paper-Conference.pdf
- PCME++ official implementation: https://github.com/naver-ai/pcmepp/blob/main/pcmepp/criterions/pcmepp.py
- DEMR official repository: https://github.com/KaijingOfficial/DEMR

## Goals

- Learn independent per-dimension `text_logvar` and `video_logvar` signals.
- Use explicit `log(sigma^2)` semantics throughout the new branch.
- Ground text variance in T2V hardness and video variance in V2T hardness.
- Preserve A1 WTI logits, retrieval metrics, risk, and vacuity exactly under a
  paired evaluation path.
- Report Gaussian uncertainty as its own signal rather than feeding it into the A1
  sidecar.
- Attribute improvements through CSD-only and hardness-only ablations after the
  combined experiment passes.

## Non-Goals

- Do not use CSD distance for retrieval or DSL reranking.
- Do not feed log-variance into the existing retrieval-risk sidecar.
- Do not unfreeze CLIP, WTI, SAP, the A1 sidecar, or any existing Gaussian module.
- Do not transplant the full DEMR temporal-grounding architecture, RFF block, or
  query-reconstruction task.
- Do not add PCME++ pseudo-positive mining or mixed-sample data augmentation in the
  first experiment.
- Do not reinterpret legacy `logsigma` checkpoint fields; the new branch has
  separate, explicit names and state-dict keys.

## Initialization and Frozen Boundary

Initialize A2 from the fully evaluated A1 checkpoint:

```text
ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4
```

Before A2 training, run a paired FP32 evaluation of that checkpoint and register:

- T2V/V2T retrieval metrics;
- A1 sidecar Spearman, AUROC, and AURC;
- a deterministic digest of the raw WTI similarity matrix.

All existing parameters remain frozen. Only these new parameters are trainable:

```text
csd_logvar_sidecar.text_head.*
csd_logvar_sidecar.video_head.*
csd_logvar_sidecar.raw_scale
csd_logvar_sidecar.shift
```

The training setup must assert the exact allowlist. Evaluation of a trained A2
checkpoint must also verify that every non-A2 state-dict tensor is numerically
identical to the A1 initialization checkpoint.

## Architecture

### Inputs

Use the normalized frozen modality means already available in the June UATVR
similarity path:

- `mu_text`: masked mean-pooled normalized text-token representation;
- `mu_video`: normalized SAP aggregate mean.

Both inputs are detached before entering the new branch. CSD and grounding losses
therefore cannot update the representations that produce WTI scores.

### Log-Variance Heads

Create a standalone `CSDLogVarSidecar` with symmetric text and video heads:

```text
Linear(512, 128) -> SiLU -> Linear(128, 512)
```

Each output is an explicit diagonal `log(sigma^2)`. Initialize the final weight to
zero and final bias to `-8.0`, and clamp outputs to `[-12.0, 4.0]`. This gives a
small but non-zero initial variance contribution while avoiding the legacy
`logsigma=-1.5` scale, whose summed 512-dimensional variance would dominate the
normalized mean distance at initialization.

The two heads contain 263,424 trainable scalar parameters. Including CSD scale and
shift gives 263,426 trainable scalars.

Define scalar intrinsic uncertainties for evaluation and grounding:

```text
u_text  = mean(exp(text_logvar), dim=-1)
u_video = mean(exp(video_logvar), dim=-1)
```

### Closed-Form Sampled Distance

For text `i` and video `j`, use the PCME++ CSD definition:

```text
d_ij = sum((mu_text_i - mu_video_j)^2)
     + sum(exp(text_logvar_i) + exp(video_logvar_j))
```

Convert distance to a matching logit:

```text
match_logit_ij = -softplus(raw_scale) * d_ij + shift
```

Initialize the effective positive scale to `5.0` and shift to `5.0`. Constraining
the scale through softplus prevents a sign flip that would make larger distances
look like better matches.

Known positives come from the DDP-gathered `group_ids` multi-positive mask. Use
unweighted all-pair `BCEWithLogitsLoss`, matching the first-stage PCME++ objective.
Known duplicate captions/videos must never be treated as negatives.

## Losses

The combined A2-CG objective is:

```text
L_A2 = 1.0 * L_CSD + 0.05 * L_ground + 1e-4 * L_VIB
```

### CSD Matching Loss

`L_CSD` is pairwise BCE between `match_logit` and the multi-positive mask. Log its
positive and negative components separately, along with mean-distance,
variance-distance, effective scale, and shift.

### Retrieval-Hardness Grounding

Reuse the existing detached multi-positive hardness computed from raw WTI logits.
Map directions explicitly:

```text
u_text  <-> T2V hardness
u_video <-> V2T hardness
```

For each direction, combine:

- Smooth-L1 between intrinsic uncertainty and robust rank-normalized hardness;
- pairwise ranking loss requiring harder queries to have greater uncertainty.

Do not add another hardness loss to the A1 sidecar; it already contains this
supervision and is frozen in A2.

### Variational Information Bottleneck

Use the diagonal-Gaussian KL to `N(0, I)` with explicit log-variance semantics:

```text
KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
```

The means are detached, so only the variance terms and new heads receive
gradients. `L_VIB` prevents uncontrolled variance growth and complements the CSD
positive-pair pressure toward zero variance.

## Training Modes and Compatibility

Add opt-in configuration with disabled defaults. The exact flag names will be
fixed in the implementation plan, but they must express:

- enable CSD log-variance sidecar;
- CSD-sidecar-only training;
- CSD, grounding, and VIB weights;
- hidden dimension, logvar bounds, initial scale, and initial shift.

When A2-only mode is active:

- the root UATVR module remains in training mode so its loss path executes;
- every frozen child remains in evaluation mode;
- only the new heads and CSD scalars remain in training mode;
- raw WTI logits and group IDs are gathered in the same rank order;
- legacy commands and checkpoints behave identically when A2 is disabled.

Checkpoints without A2 keys may initialize the new branch only when the feature is
explicitly enabled. A2 checkpoints must round-trip the heads, scale, shift, and
configuration without changing A1 outputs.

## Evaluation

### Paired Invariance

Use the same FP32 checkpoint-loading and evaluation path for both A1 and A2. Require:

- exact equality of raw WTI matrices or their deterministic digest;
- exact equality of A1 sidecar risk and vacuity arrays;
- exact equality of retrieval metrics;
- numerical identity of all non-A2 state-dict tensors.

This paired baseline replaces comparison against the earlier mixed-precision
52.6/52.5 sweep when judging invariance.

### Intrinsic-Uncertainty Metrics

Collect each intrinsic uncertainty once per query item, not once per candidate
chunk. Report:

- `rho(u_text, T2V rank)` and `rho(u_video, V2T rank)`;
- failure AUROC with deterministic stratified bootstrap confidence interval;
- risk-coverage AURC and accuracy at 25%, 50%, 75%, and 100% coverage;
- logvar and variance min/max/mean/std;
- fraction of values at each clamp boundary;
- CSD mean-distance and variance-distance statistics.

For multi-caption validation, text uncertainty has one value per caption and video
uncertainty has one value per unique video. Positive masks must use the existing
cutoff/group mapping rather than an identity assumption.

Continue reporting the existing A1 sidecar metrics separately. Do not combine or
rename the two uncertainty sources.

### Checkpoint Selection

Reject checkpoints with non-finite values, failed paired invariance, or clamp rate
at or above 5%. Among valid checkpoints:

1. maximize `min(T2V rho, V2T rho)`;
2. break ties using mean directional failure AUROC;
3. break remaining ties in favor of the earlier epoch.

Only the selected validation checkpoint is evaluated on JSFUSION test.

## Experiment Sequence

All A2 runs start independently from the same A1 `bin.4`, use seed 0, trusted-v1,
effective batch 256, and the same retrieval/evaluation configuration.

| Experiment | CSD | Hardness grounding | Purpose |
|------------|-----|--------------------|---------|
| A1 | no | existing sidecar only | registered baseline |
| A2-CG | yes | yes | primary combined experiment |
| A2-C | yes | no | isolate CSD contribution |
| A2-G | no | yes | isolate grounding contribution |

Run A2-CG for three epochs first. If its registered validation score is still
improving at epoch 3, resume to epoch 5. Run A2-C and A2-G only after A2-CG passes,
so failed combined training does not spend compute on premature ablations.

## Acceptance and Stop Rules

A2-CG passes only if:

- paired raw WTI, retrieval metrics, and A1 sidecar outputs are unchanged;
- both validation correlations reach `rho >= 0.20`;
- both failure AUROCs exceed `0.5` with reported confidence intervals;
- removing high-uncertainty queries monotonically improves retained accuracy at
  the registered coverage points;
- intrinsic uncertainty has non-zero cross-sample variance;
- fewer than 5% of logvar elements hit either clamp boundary.

Interpret ablations as follows:

- If A2-CG and A2-G pass but A2-C fails, retain hardness-grounded logvar and do not
  claim a CSD contribution.
- If A2-C passes and A2-G fails, describe logvar as probabilistic matching
  uncertainty rather than retrieval-hardness uncertainty.
- If only A2-CG passes, claim complementarity and retain both losses.
- If no A2 variant reaches the bidirectional correlation gate, retain A1 as the
  final uncertainty method and report intrinsic Gaussian uncertainty as unsupported.

Variance collapse or saturation is a loss-scaling failure, not permission to
unfreeze the retrieval model. Inspect CSD positive/negative balance, scale/shift,
VIB magnitude, and clamp statistics before changing architecture.

## Verification

Implementation verification must include:

- unit tests for CSD against hand-computed Gaussian distances;
- tests distinguishing log-variance from legacy log-standard-deviation sampling;
- multi-positive pairwise BCE tests;
- detached mean and hardness-target tests;
- gradient allowlist tests proving only A2 parameters update;
- exact WTI and A1 sidecar invariance tests;
- checkpoint save/load round-trip tests;
- multi-caption text/video uncertainty alignment tests;
- explicit failure tests for missing positives, non-finite inputs, invalid bounds,
  and non-positive scales;
- launcher syntax and experiment-manifest tests;
- one optimizer-step smoke test before any full multi-GPU experiment.

## Deliverables

- Standalone CSD/log-variance module and losses.
- UATVR A2-only integration with frozen-boundary assertions.
- Directionally correct intrinsic-uncertainty evaluation and checkpoint selection.
- Reproducible A2-CG launcher and manifest fields.
- Tests proving retrieval and A1 sidecar invariance.
- A result report comparing A1, A2-CG, A2-C, and A2-G when the registered sequence
  reaches those stages.
