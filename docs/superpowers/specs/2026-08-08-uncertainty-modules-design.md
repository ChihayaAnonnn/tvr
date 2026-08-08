# Standalone Uncertainty Modules Design

## Goal

Implement standalone, uncertainty-aware representation modules in PyTorch for
video-text retrieval. The first delivery contains an aleatoric module followed
by an epistemic module. It does not inspect, import, modify, or integrate with
the existing retrieval implementation.

The modules must help distinguish three questions:

1. Which token or frame is semantically relevant?
2. Which observed evidence is unreliable?
3. When is the model representation outside its knowledge?

## Scope

The delivery is a new top-level `uncertainty_modules` Python package with unit
tests and usage documentation. It consumes precomputed feature tensors and has
no encoder, dataset, degradation pipeline, retrieval loss, training loop, or
main-flow integration.

Video and text use the same modality-agnostic aleatoric class through separate
instances, so their parameters are independent while their APIs are identical.

## Package Layout

```text
uncertainty_modules/
├── __init__.py
├── types.py
├── aleatoric.py
├── epistemic.py
├── losses.py
├── README.md
└── tests/
    ├── test_aleatoric.py
    ├── test_epistemic.py
    └── test_losses.py
```

The package depends only on PyTorch and the Python standard library. Tests use
the repository's available test runner but exercise only this package.

## Shared Result Types

`ProbabilisticEmbedding` is an immutable result container with:

- `mean`: `[B, D]` semantic embedding.
- `variance`: `[B, D]` positive diagonal variance.
- `uncertainty_score`: `[B]` scalar summary of the variance.

`AleatoricOutput` contains the probabilistic embedding plus:

- `local_uncertainty`: `[B, N]` token/frame uncertainty.
- `global_uncertainty`: `[B]` sample uncertainty.
- `relevance_weights`: `[B, N]` semantic-only normalized weights.
- `aggregation_weights`: `[B, N]` relevance adjusted by reliability.

`EpistemicOutput` contains:

- `uncertainty_score`: `[B]` final or single-source knowledge uncertainty.
- named component scores and diagonal disagreement variances when applicable.
- nearest prototype distance and index when prototype estimation is used.
- optional OOD decisions only when an explicit calibrated threshold is given.

## Aleatoric Module

### Interface

`AleatoricUncertaintyModule(input_dim, hidden_dim, min_variance,
max_variance)` accepts `features: [B, N, D]` and an optional boolean
`mask: [B, N]`, where `True` marks a valid element.

The constructor rejects nonpositive dimensions and invalid variance ranges.
The forward method rejects incompatible tensor ranks/shapes and any batch item
whose mask contains no valid element.

### Separate Branches

Two independent MLPs operate on the input features:

- The relevance branch emits scalar token logits. Masked softmax produces
  `relevance_weights` that express semantic importance only.
- The uncertainty branch emits a diagonal token variance `[B, N, D]`. A
  positive parameterization and configured bounds keep the variance finite and
  stable. Averaging over `D` produces `local_uncertainty`.

Reliability is `exp(-local_uncertainty)`. Aggregation weights are the normalized
product of semantic relevance and reliability. Invalid elements have exactly
zero weight.

Reliability is detached before it enters aggregation. Consequently, semantic
retrieval gradients train the relevance branch but cannot turn the uncertainty
branch into a second attention head. Dedicated uncertainty objectives train the
uncertainty branch.

### Probabilistic Aggregation

The mean embedding is the aggregation-weighted sum of the original input
features. Its diagonal variance combines:

1. token variance propagated as `sum(weights**2 * token_variance)`, and
2. a global diagonal variance predicted from the sample-level aggregated
   representation.

The final variance is constrained to the configured finite interval. The
global uncertainty is the feature-wise mean of the global variance. The
embedding uncertainty score is the feature-wise mean of the final variance.

### Training Losses

The standalone package provides three optional loss functions:

- `uncertainty_ranking_loss(clean, degraded, margin)` penalizes cases where a
  degraded representation is not more uncertain than its clean counterpart.
- `semantic_consistency_loss(clean_mean, degraded_mean)` preserves semantic
  representations under controlled degradation.
- `variance_prior_loss(variance, target_log_variance)` weakly regularizes log
  variance away from persistent lower- or upper-bound collapse.

The variance floor provides numerical safety; ranking supervision and the weak
prior provide behavioral anti-collapse pressure. The module does not claim to
prevent collapse without an appropriate training objective.

## Epistemic Module

`EpistemicUncertaintyModule` is encoder-independent and supports three sources.

### Ensemble Disagreement

`from_ensemble(samples: [M, B, D])` requires at least two model samples. It
computes population variance (`unbiased=False`) across `M`, preserves the
diagonal variance `[B, D]`, and averages over `D` for a sample score `[B]`.

### MC Dropout Disagreement

`from_mc_dropout(samples: [M, B, D])` uses the same stable population-variance
calculation and validation. The separate method preserves the semantic origin
of the estimate and permits later extension without coupling the module to a
model's dropout execution.

### Prototype Distance

`from_prototypes(features: [B, D], prototypes: [K, D])` requires at least one
prototype and matching dimensions. The default metric is nearest cosine
distance after numerically safe normalization. Squared Euclidean distance is
also supported. The result includes each sample's minimum distance and nearest
prototype index.

### Combination and OOD Decisions

`combine` accepts available named component scores, explicit nonnegative
weights, and validation-derived calibration statistics. Each score is
z-normalized using its supplied mean and standard deviation before weighted
aggregation. No weights, scales, or thresholds are silently learned or
hard-coded.

An OOD boolean is returned only when the caller supplies a calibrated threshold;
the decision is `combined_score > threshold`. Invalid weights, missing
statistics, zero standard deviations, and inconsistent batch shapes raise clear
exceptions.

## Numerical and Error Behavior

- Masked softmax is implemented without allowing invalid positions to
  contribute to normalization.
- Positive variance uses a stable smooth transform before finite bounding.
- Cosine distance uses an epsilon-safe normalization.
- Ensemble and MC variance use population variance to avoid small-sample NaNs.
- Public methods validate rank, batch, feature, mask, sample-count, and
  prototype-count assumptions close to the API boundary.
- Outputs preserve the input device and floating-point dtype.

## Test Strategy

Implementation follows red-green-refactor and proceeds in this order:

1. shared result containers and aleatoric validation,
2. aleatoric relevance, uncertainty, masking, aggregation, and gradients,
3. aleatoric losses,
4. ensemble and MC-dropout epistemic estimates,
5. prototype estimates,
6. calibrated combination and OOD decisions,
7. documentation and a complete package-level verification run.

Aleatoric tests cover output shapes, finite positive bounded variances, masked
zero weights, normalization over valid elements, all-empty-mask errors,
single-token and variable-length inputs, dtype/device preservation, and
autograd. A branch-isolation test verifies that a mean-only semantic loss sends
gradient to relevance parameters but not through reliability to uncertainty
parameters.

Loss tests verify ordering behavior, zero semantic loss for identical inputs,
and minimization of the variance prior at its target.

Epistemic tests verify zero disagreement for identical samples, increasing
scores under disagreement, correct nearest prototypes under both metrics,
calibrated weighted combination, explicit-threshold OOD decisions, and all
documented validation failures.

## Acceptance Criteria

- The new package can be imported without importing project retrieval code.
- Aleatoric and epistemic APIs return the documented shapes and finite values.
- Relevance and uncertainty parameters are structurally separate, and semantic
  mean gradients do not train uncertainty through reliability pooling.
- Both video and text features are supported by separate instances of the same
  aleatoric class.
- Ensemble, MC dropout, and prototype epistemic estimates are all implemented.
- All package-local tests and the autograd smoke test pass.
- No existing project code or main training/retrieval flow is changed.

## Deferred Work

Integration with encoders, controlled degradation, loss weighting, calibration
dataset selection, OOD threshold fitting, retrieval reranking, selective
retrieval, and retrieval metrics are deferred until the standalone module is
reviewed and approved.
