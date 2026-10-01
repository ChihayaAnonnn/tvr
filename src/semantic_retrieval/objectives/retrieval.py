"""Known-label-only objectives with sufficient counts for distributed means."""

import math

import torch
from torch.nn import functional as F

from semantic_retrieval.models.pair_matcher import validate_offsets, validate_pairs

LOSS_NAMES = ("coarse", "match", "pair", "exist", "align", "group")


def contrastive_rows(logits, labels):
    losses = []
    for row, lab in zip(logits, labels):
        known = lab >= 0
        pos = lab == 1
        if pos.any() and (lab == 0).any():
            losses.append(torch.logsumexp(row[known], 0) - row[pos].mean())
    zero = logits.sum() * 0
    return (torch.stack(losses).mean() if losses else zero), len(losses)


def compute_objective(
    coarse_logits,
    pair_scores,
    presence_logits,
    targets,
    pair_indices,
    query_offsets,
    *,
    weights,
    coarse_relevance=None,
    auxiliary_losses=None,
):
    validate_pairs(pair_indices, *targets.relevance.shape)
    validate_offsets(query_offsets, len(pair_scores.logits))
    q, v = targets.relevance.shape
    if len(query_offsets) != q + 1 or presence_logits.shape != (q,) or pair_scores.logits.shape != (len(pair_indices),):
        raise ValueError("objective shapes disagree")
    for i, (a, b) in enumerate(zip(query_offsets[:-1], query_offsets[1:])):
        if not (pair_indices[a:b, 0] == i).all():
            raise ValueError("packed pairs must belong to offset query")
    aux = auxiliary_losses or {}
    if (
        set(weights) - set(LOSS_NAMES)
        or set(aux) - {"align", "group"}
        or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights.values())
    ):
        raise ValueError("invalid objective weights or auxiliary keys")
    cr = targets.relevance if coarse_relevance is None else coarse_relevance
    if cr.shape != coarse_logits.shape or cr.ndim != 2 or not ((cr == -1) | (cr == 0) | (cr == 1)).all():
        raise ValueError("coarse labels must be audited and match scores")
    if not all(torch.isfinite(x).all() for x in (coarse_logits, pair_scores.logits, presence_logits)):
        raise ValueError("nonfinite scores")
    zero = coarse_logits.sum() * 0 + pair_scores.logits.sum() * 0 + presence_logits.sum() * 0
    terms = {k: zero for k in LOSS_NAMES}
    counts = {k: 0 for k in LOSS_NAMES}
    terms["coarse"], counts["coarse"] = contrastive_rows(coarse_logits, cr)
    lab = targets.relevance[pair_indices[:, 0], pair_indices[:, 1]]
    ml = []
    for a, b in zip(query_offsets[:-1].tolist(), query_offsets[1:].tolist()):
        value, count = contrastive_rows(pair_scores.logits[a:b][None], lab[a:b][None])
        if count:
            ml.append(value)
    if ml:
        terms["match"] = torch.stack(ml).mean()
        counts["match"] = len(ml)
    for key, logits, labels in [
        ("pair", pair_scores.logits, lab),
        ("exist", presence_logits, targets.candidate_presence),
    ]:
        mask = labels >= 0
        counts[key] = int(mask.sum())
        if mask.any():
            terms[key] = F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
    for key, value in aux.items():
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError("auxiliary loss must be finite scalar")
        terms[key] = value
    terms["total"] = zero + sum(weights.get(k, 0) * terms[k] for k in LOSS_NAMES)
    terms.update({"valid_" + k + "_count": torch.tensor(n, device=coarse_logits.device) for k, n in counts.items()})
    return terms
