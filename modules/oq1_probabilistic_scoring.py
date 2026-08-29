"""Frozen scoring implementation for study oq1-kl-variance-hubness-v1.

Implements the registered arm family (study.yaml `variants.arms`,
PREREGISTRATION.md §4) over diagonal Gaussians:

    cost(t, v) = lam_A * (-1/2 log sigma_v^2)          (A) candidate-only
               + lam_B * (+1/2 log sigma_t^2)          (B) query-only
               + lam_C * (sigma_v^2 / 2 sigma_t^2)     (C) bilinear
               + lam_D * (||mu_t - mu_v||^2 / 2 sigma_t^2)  (D) Mahalanobis

plus the two alternative-cost control arms (symmetric KL, ELK). Retrieval
logits are ``-cost``; the caller applies the CLIP logit scale.

Implementation constants frozen here (NOT in the frozen study.yaml, which
does not pin them; recorded in the study control-plane README):

- ``logsigma`` follows the repo convention of prob_models.tensor_utils
  (``sample = mu + exp(logsigma) * eps``), i.e. logsigma = log sigma, so
  log sigma^2 = 2 * logsigma.
- log sigma is clamped to [OQ1_LOG_SIGMA_MIN, OQ1_LOG_SIGMA_MAX] before any
  cost term, identically for every arm and both modalities.
- Every per-dimension sum is normalised by the embedding dimension (mean
  over dims). This is a single monotone rescaling shared by all pairs, so
  rankings are unchanged; it only keeps the softmax temperature of
  ``logit_scale * (-cost)`` in a workable range.
- lam weights are fixed {0,1} constants (lam_weights_are_fixed_not_tuned).
  Zero-weighted terms are still multiplied into the graph (0.0 * term) so
  the corresponding head parameters always participate in autograd and DDP
  with find_unused_parameters=False stays valid; their gradients are
  exactly zero.
- arm4 freezes sigma_t == 1 by ``logsigma_t * 0.0`` (again keeping the
  text-uncertainty head connected with exactly-zero gradients, preserving
  head_parameter_count_identical_for arms 1/2/4/5/6).
"""

import torch

OQ1_LOG_SIGMA_MIN = -3.0
OQ1_LOG_SIGMA_MAX = 3.0

# arm name -> spec. kind: akl | symmetric_kl | elk. lam order: (A, B, C, D).
OQ1_ARM_SPECS = {
    "arm1_full_akl": {"kind": "akl", "lam": (1.0, 1.0, 1.0, 1.0), "freeze_sigma_t": False},
    "arm2_no_A": {"kind": "akl", "lam": (0.0, 1.0, 1.0, 1.0), "freeze_sigma_t": False},
    "arm3_D_only": {"kind": "akl", "lam": (0.0, 0.0, 0.0, 1.0), "freeze_sigma_t": False},
    "arm4_unit_sigma_t": {"kind": "akl", "lam": (1.0, 0.0, 1.0, 1.0), "freeze_sigma_t": True},
    "arm5_symmetric_kl": {"kind": "symmetric_kl"},
    "arm6_elk": {"kind": "elk"},
}

OQ1_PROBABILISTIC_ARMS = tuple(OQ1_ARM_SPECS)
OQ1_ALL_ARMS = ("arm0_deterministic",) + OQ1_PROBABILISTIC_ARMS

# ELK needs a [Bt, Bv, D] broadcast; chunk the text axis to bound activations.
_ELK_TEXT_CHUNK = 128


def clamp_logsigma(logsigma):
    return torch.clamp(logsigma, min=OQ1_LOG_SIGMA_MIN, max=OQ1_LOG_SIGMA_MAX)


def _akl_cost(mu_t, s2_t, mu_v, s2_v, lam, dim):
    lam_a, lam_b, lam_c, lam_d = lam
    inv_var_t = torch.exp(-s2_t)  # [Bt, D]
    var_v = torch.exp(s2_v)       # [Bv, D]

    term_a = -0.5 * s2_v.mean(dim=-1)  # [Bv]
    term_b = 0.5 * s2_t.mean(dim=-1)   # [Bt]
    term_c = (0.5 / dim) * (inv_var_t @ var_v.t())  # [Bt, Bv]
    # sum_d (mu_t - mu_v)^2 / (2 var_t), expanded to avoid a 3-D tensor
    term_d = (0.5 / dim) * (
        (mu_t.square() * inv_var_t).sum(dim=-1, keepdim=True)
        - 2.0 * ((mu_t * inv_var_t) @ mu_v.t())
        + inv_var_t @ mu_v.square().t()
    )
    return (
        lam_a * term_a.unsqueeze(0)
        + lam_b * term_b.unsqueeze(1)
        + lam_c * term_c
        + lam_d * term_d
    )


def _symmetric_kl_cost(mu_t, s2_t, mu_v, s2_v, dim):
    # 1/2 [KL(t||v) + KL(v||t)]; the log-variance terms cancel and the
    # additive constant -1/2 per dim is dropped (rank-irrelevant).
    var_t, inv_var_t = torch.exp(s2_t), torch.exp(-s2_t)
    var_v, inv_var_v = torch.exp(s2_v), torch.exp(-s2_v)
    cross = (0.25 / dim) * (var_t @ inv_var_v.t() + inv_var_t @ var_v.t())
    inv_sum_t = inv_var_t  # for Delta^2 / (4 var_t)
    inv_sum_v = inv_var_v  # for Delta^2 / (4 var_v)

    def _pairwise_weighted_sqdist(weight_t_side, on_text_axis):
        # sum_d (mu_t - mu_v)^2 * w_d with w attached to one side's rows
        if on_text_axis:
            w = weight_t_side  # [Bt, D]
            return (
                (mu_t.square() * w).sum(dim=-1, keepdim=True)
                - 2.0 * ((mu_t * w) @ mu_v.t())
                + w @ mu_v.square().t()
            )
        w = weight_t_side  # [Bv, D]
        return (
            mu_t.square() @ w.t()
            - 2.0 * (mu_t @ (mu_v * w).t())
            + (mu_v.square() * w).sum(dim=-1).unsqueeze(0)
        )

    delta_over_var_t = _pairwise_weighted_sqdist(inv_sum_t, on_text_axis=True)
    delta_over_var_v = _pairwise_weighted_sqdist(inv_sum_v, on_text_axis=False)
    return cross + (0.25 / dim) * (delta_over_var_t + delta_over_var_v)


def _elk_cost(mu_t, s2_t, mu_v, s2_v, dim):
    # cost = -log ELK = mean_d [ 1/2 log(var_t + var_v) + Delta^2 / (2 (var_t + var_v)) ]
    # (2*pi constant dropped). No separable candidate-only term exists here.
    var_t, var_v = torch.exp(s2_t), torch.exp(s2_v)
    chunks = []
    for start in range(0, mu_t.size(0), _ELK_TEXT_CHUNK):
        end = start + _ELK_TEXT_CHUNK
        var_sum = var_t[start:end].unsqueeze(1) + var_v.unsqueeze(0)  # [bt, Bv, D]
        delta_sq = (mu_t[start:end].unsqueeze(1) - mu_v.unsqueeze(0)).square()
        chunk = 0.5 * torch.log(var_sum) + delta_sq / (2.0 * var_sum)
        chunks.append(chunk.mean(dim=-1))
    return torch.cat(chunks, dim=0)


def oq1_cost_matrix(arm, mu_t, logsigma_t, mu_v, logsigma_v):
    """Pairwise cost matrix [Bt, Bv] for one registered probabilistic arm.

    mu_* are the l2-normalised means, logsigma_* the raw head outputs
    (log sigma, repo convention). Clamping and the log sigma -> log sigma^2
    conversion happen here so every caller scores identically.
    """
    spec = OQ1_ARM_SPECS[arm]
    dim = mu_t.size(-1)
    s2_t = 2.0 * clamp_logsigma(logsigma_t)
    s2_v = 2.0 * clamp_logsigma(logsigma_v)
    if spec.get("freeze_sigma_t"):
        s2_t = s2_t * 0.0
    if spec["kind"] == "akl":
        return _akl_cost(mu_t, s2_t, mu_v, s2_v, spec["lam"], dim)
    if spec["kind"] == "symmetric_kl":
        return _symmetric_kl_cost(mu_t, s2_t, mu_v, s2_v, dim)
    if spec["kind"] == "elk":
        return _elk_cost(mu_t, s2_t, mu_v, s2_v, dim)
    raise ValueError(f"unknown oq1 arm kind: {spec['kind']}")
