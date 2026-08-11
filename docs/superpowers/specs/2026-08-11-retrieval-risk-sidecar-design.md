# Retrieval-Risk Sidecar Design

**Date:** 2026-08-11

## Decision

Route 1+3 is not a viable retrieval architecture: its selected checkpoint reached
only T2V R@1 28.2 and V2T R@1 32.2 on the final MSRVTT test. The next experiment
must therefore stop using SAP uncertainty or confidence to rescale retrieval
similarities.

Implement uncertainty as a removable sidecar on top of the strongest deterministic
WTI retrieval checkpoint. The sidecar predicts retrieval failure and evidential
vacuity, while the original similarity matrix remains the sole input to ranking.

## Goals and Non-Goals

### Goals

- Preserve the deterministic checkpoint's rankings exactly when the sidecar is
  enabled.
- Produce query-conditioned T2V and V2T failure risk from candidate-score geometry.
- Distinguish predicted failure probability from insufficient evidence (vacuity).
- Train against retrieval difficulty without allowing the target to be gamed by
  changing the retrieval scores.
- Evaluate matching-direction Spearman correlation, failure AUROC, and risk-coverage
  AURC in addition to normal retrieval metrics.

### Non-Goals

- Do not use sidecar risk to rerank candidates in this phase.
- Do not continue route 1+3 confidence multiplication.
- Do not claim anchor entropy, Gaussian variance, or CRF `u_video` is retrieval
  uncertainty.
- Do not implement K-way Dirichlet evidence over the full candidate set in the
  first version.
- Do not jointly fine-tune the retrieval encoder until the frozen-sidecar experiment
  has been evaluated.

## Architecture

Given the unmodified WTI similarity matrix `logits` with shape `[N_text, N_video]`,
construct one inference-safe feature vector for every row and column. The feature
extractor uses only the score distribution, never the ground-truth positive index:

- top-1 score;
- top-1 minus top-2 margin;
- normalized entropy of the candidate softmax;
- score mean and population standard deviation;
- normalized log-sum-exp concentration.

Each feature is invariant to candidate ordering. Margin, entropy, standard
deviation, and centered concentration are invariant to adding a constant to every
candidate score. Candidate-count-dependent statistics are normalized so training
and evaluation sets can have different candidate counts.

A shared `RetrievalRiskSidecar` MLP maps the feature vector to two logits. Softplus
evidence gives a binary Dirichlet distribution:

```text
alpha_ok   = 1 + softplus(evidence_logit_ok)
alpha_fail = 1 + softplus(evidence_logit_fail)
failure_risk = alpha_fail / (alpha_ok + alpha_fail)
vacuity       = 2 / (alpha_ok + alpha_fail)
```

The same head is applied to rows for T2V and columns for V2T. Direction identity is
included as a two-value one-hot feature so a shared head can calibrate the two
directions without duplicating parameters.

The retrieval path and sidecar path are strictly separated:

```text
WTI logits -> existing retrieval loss and ranking (unchanged)
          \-> detach -> score features -> risk sidecar -> sidecar loss
```

Detaching before feature extraction guarantees that sidecar loss cannot change the
retrieval encoder, SAP, or WTI logits. Only sidecar parameters receive gradients.

## Supervision and Loss

Training positives must be represented by a Boolean `positive_mask` matching the
similarity matrix. Do not assume an identity matrix when multiple captions can map
to the same video. The integration must use sample/video identity when it is
available and reject rows or columns without a positive.

For each direction, compute detached continuous retrieval hardness:

```text
positive = logsumexp(scores of all positives)
negative = logsumexp(scores of all negatives)
hardness = softplus((negative - positive) / temperature)
```

The primary binary label is `failure = 1` when no positive is ranked first and zero
otherwise. The sidecar objective contains:

1. binary evidential MSE plus Dirichlet predictive variance;
2. Smooth-L1 fit between `failure_risk` and robust rank-normalized hardness;
3. pairwise ranking loss enforcing that harder queries receive greater risk.

All targets are detached. Empty directions, a candidate dimension below two, and
non-finite logits produce an explicit validation error rather than silent NaNs.

Default weights are zero for checkpoint and command-line compatibility. The first
experiment uses a single aggregate `w_retrieval_risk=0.05`; the individual loss
components remain separately logged for diagnosis.

## Training and Checkpoint Compatibility

- Load a strong deterministic checkpoint, not the failed route 1+3 checkpoint.
- Instantiate the sidecar only when `--use_retrieval_risk_sidecar` is supplied.
- Freeze all non-sidecar parameters for the A1 run and assert this condition in the
  training setup.
- Save sidecar parameters in the normal model state dictionary.
- Loading older checkpoints must tolerate the absent sidecar keys only when the
  sidecar is newly enabled; unrelated missing or unexpected keys remain errors.
- Existing runs with the flag disabled must follow the byte-for-byte-equivalent
  scoring path and retain zero additional losses.

## Evaluation

Evaluation first computes the complete, unmodified similarity matrix. Risk features
must be calculated from the complete candidate dimension, not independently inside
candidate chunks. Report:

- existing T2V/V2T R@1, R@5, R@10, median rank, and mean rank;
- Spearman correlation between T2V row risk and T2V ground-truth rank;
- Spearman correlation between V2T column risk and V2T ground-truth rank;
- AUROC for rank greater than zero;
- risk-coverage AURC;
- min, max, mean, and standard deviation for risk and vacuity.

The evaluator must support MSRVTT's multi-sentence structure by accepting explicit
positive masks or query/candidate video IDs. It must not reuse
`score_video_risk()`, which currently measures CRF uncertainty.

## Verification and Acceptance

Unit tests cover feature invariance, candidate-count normalization, clear-versus-
ambiguous ordering after training, multi-positive masks, detached targets, gradient
isolation, old-checkpoint defaults, and invalid inputs. An integration test compares
retrieval logits and rankings with the sidecar disabled and enabled and requires
exact equality.

The A1 experiment is accepted only when:

- deterministic retrieval metrics are unchanged from the selected baseline;
- matching-direction Spearman `rho >= 0.20` in at least the registered target
  direction;
- failure AUROC is greater than 0.5 with a reported confidence interval;
- removing high-risk queries improves risk-coverage accuracy monotonically at the
  registered coverage points.

If A1 misses the uncertainty gate, inspect calibration and supervision before any
encoder unfreezing. If no sidecar variant passes without degrading retrieval, retain
the deterministic model and document retrieval uncertainty as unsupported.

