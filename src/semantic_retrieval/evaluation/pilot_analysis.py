"""Frozen, benchmark-annotation-only validation analysis for the semantic pilot.

Numeric observations live in hash-bound NPZ inputs and run-local NPY outputs.
No routine here infers semantic irrelevance from an unannotated video.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from semantic_retrieval.evaluation.binary import binary_metrics

KS = (4, 8, 16, 32, 64, 128)
SCALARS = ("top_score", "margin", "negative_entropy", "energy")
GRID = np.linspace(0, 1, 21)


def rate(numerator, denominator):
    return dict(
        numerator=float(numerator),
        denominator=int(denominator),
        value=float(numerator / denominator) if denominator else None,
    )


def risk_curve(confidence, errors):
    confidence, errors = np.asarray(confidence, float), np.asarray(errors)
    if confidence.ndim != 1 or confidence.shape != errors.shape or not np.isfinite(confidence).all():
        raise ValueError("invalid RC arrays")
    if not np.isin(errors, (0, 1)).all():
        raise ValueError("RC requires binary benchmark errors")
    if not len(errors):
        return dict(coverage=np.array([]), risk=np.array([]), aurc=None, denominator=0)
    order = np.argsort(-confidence)
    end = np.r_[np.flatnonzero(np.diff(confidence[order]) != 0), len(order) - 1]
    counts = end + 1
    coverage = counts / len(order)
    risk = np.cumsum(errors[order])[end] / counts
    return dict(
        coverage=coverage, risk=risk, aurc=float(np.sum(np.diff(np.r_[0.0, coverage]) * risk)), denominator=len(errors)
    )


def official_ranking(scores, official_relevance, gallery):
    scores = np.asarray(scores, float)
    canonical = gallery["canonical_ids"]
    aliases = gallery["aliases"]
    official = [i for c in canonical for i in aliases[c]]
    if (
        len(set(canonical)) != len(canonical)
        or len(set(official)) != len(official)
        or any(not aliases[c] for c in canonical)
    ):
        raise ValueError("gallery must have unique canonical and official identities")
    relevance = np.asarray(official_relevance)
    if scores.ndim != 2 or scores.shape[1] != len(canonical) or relevance.shape != (len(scores), len(official)):
        raise ValueError("official ranking shape mismatch")
    if not np.isfinite(scores).all() or not np.isin(relevance, (-1, 0, 1)).all():
        raise ValueError("invalid ranking labels/scores")
    expanded = np.repeat(scores, [len(aliases[c]) for c in canonical], axis=1)
    ids = np.asarray(official)
    rows = []
    for s, y in zip(expanded, relevance):
        if (y < 0).any() or not (y == 1).any():
            continue
        ordered = y[np.lexsort((ids, -s))] == 1
        positive = np.flatnonzero(ordered)
        first = int(positive[0]) + 1
        ap = float(np.mean(np.arange(1, len(positive) + 1) / (positive + 1)))
        dcg = float(np.sum(ordered[:10] / np.log2(np.arange(2, min(10, len(ordered)) + 2))))
        idcg = float(np.sum(1 / np.log2(np.arange(2, min(10, len(positive)) + 2))))
        rows.append((first, ap, dcg / idcg))
    n = len(rows)
    rank = np.asarray([r[0] for r in rows], float)
    result = {f"R{k}": rate(np.sum(rank <= k), n) for k in (1, 5, 10)}
    result.update(
        MRR=rate(np.sum(1 / rank), n),
        mAP=rate(sum(r[1] for r in rows), n),
        nDCG10=rate(sum(r[2] for r in rows), n),
        MeanR=rate(np.sum(rank), n),
        MedianR=dict(value=float(np.median(rank)) if n else None, denominator=n),
        excluded_count=len(scores) - n,
        single_positive_AP_equals_RR=bool(n and all(np.sum(y == 1) == 1 for y in relevance)),
    )
    return result


def policy_metrics(returned, truth, candidates, *, initial_candidates=None):
    returned, truth, candidates = (np.asarray(a) for a in (returned, truth, candidates))
    if returned.ndim != 2 or truth.shape != returned.shape or candidates.shape != returned.shape:
        raise ValueError("policy array shape mismatch")
    if any(not np.isin(a, (0, 1)).all() for a in (returned, truth, candidates)) or np.any(
        returned & ~candidates.astype(bool)
    ):
        raise ValueError("invalid benchmark sets")
    returned, truth, candidates = (a.astype(bool) for a in (returned, truth, candidates))
    eligible = truth.any(axis=1)
    excluded = int((~eligible).sum())
    returned, truth, candidates = (a[eligible] for a in (returned, truth, candidates))
    n = len(truth)
    did_return = returned.any(axis=1)
    present = (truth & candidates).any(axis=1)
    tp = (returned & truth).sum(axis=1)
    answer_count, positive_count = returned.sum(axis=1), truth.sum(axis=1)
    precision = np.divide(tp, answer_count, out=np.zeros(n, float), where=answer_count > 0)
    recall = tp / positive_count if n else np.zeros(0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n, float), where=(precision + recall) > 0)
    wrong = did_return & (tp == 0)
    result = dict(
        coverage=rate(did_return.sum(), n),
        present_coverage=rate((did_return & present).sum(), present.sum()),
        absent_false_accept=rate((did_return & ~present).sum(), (~present).sum()),
        false_abstain=rate((~did_return & present).sum(), present.sum()),
        wrong_return=rate(wrong.sum(), did_return.sum()),
        wrong_return_all=rate(wrong.sum(), n),
        set_micro_precision=rate(tp.sum(), answer_count.sum()),
        set_micro_recall=rate(tp.sum(), positive_count.sum()),
        macro_precision=rate(precision.sum(), n),
        macro_recall=rate(recall.sum(), n),
        macro_f1=rate(f1.sum(), n),
        abstain_rate=rate((~did_return).sum(), n),
        clarify_rate=rate(0, n),
        excluded_count=excluded,
        positive_histogram={str(k): int(np.sum(positive_count == k)) for k in np.unique(positive_count)},
        multi_answer_status="available" if np.any(positive_count > 1) else "N/A",
    )
    if initial_candidates is not None:
        initial = np.asarray(initial_candidates, bool)
        if initial.shape != eligible.shape + (truth.shape[1],):
            raise ValueError("initial candidates shape mismatch")
        miss = ~(initial[eligible] & truth).any(axis=1)
        result["expansion_candidate_recovery"] = rate((miss & present).sum(), miss.sum())
        result["expansion_return_recovery"] = rate((miss & (tp > 0)).sum(), miss.sum())
    return result


def scalar_confidences(scores, candidates):
    scores, candidates = np.asarray(scores, float), np.asarray(candidates, bool)
    if scores.shape != candidates.shape or not candidates.any(axis=1).all():
        raise ValueError("each panel needs candidates")
    values = {name: [] for name in SCALARS}
    for s, mask in zip(scores, candidates):
        s = np.sort(s[mask])[::-1]
        p = np.exp(s - logsumexp(s))
        values["top_score"].append(s[0])
        values["margin"].append(s[0] - s[1] if len(s) > 1 else 0.0)
        values["negative_entropy"].append(float(np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)))))
        values["energy"].append(float(logsumexp(s)))
    return {k: np.array(v) for k, v in values.items()}


def apply_operating_point(
    scores, candidates, operating_point, *, presence_logits=None, scalar=None, top1=False, canonical_ids=None
):
    scores, candidates = np.asarray(scores, float), np.asarray(candidates, bool)
    if operating_point is None:
        return None
    support = candidates & (expit(scores) >= operating_point["pair_threshold"])
    gate = np.ones(len(scores), bool)
    if operating_point["accept_kind"] == "presence":
        if presence_logits is None:
            raise ValueError("trained presence logits required")
        gate = expit(presence_logits) >= operating_point["accept_threshold"]
    elif operating_point["accept_kind"] == "scalar":
        if scalar is None:
            raise ValueError("scalar confidence required")
        gate = np.asarray(scalar) >= operating_point["accept_threshold"]
    support &= gate[:, None]
    if top1:
        names = np.asarray(canonical_ids if canonical_ids is not None else [str(i) for i in range(scores.shape[1])])
        first = np.array(
            [np.lexsort((names, -np.where(mask, values, -np.inf)))[0] for values, mask in zip(scores, candidates)]
        )
        keep = np.zeros_like(support)
        keep[np.arange(len(scores)), first] = support[np.arange(len(scores)), first]
        support = keep
    return support


def evaluate_threshold_grid(scores, truth, candidates, *, presence_logits=None, scalar=None, points=None):
    """Describe an already declared grid; this function never selects an action rule."""
    kind = "presence" if presence_logits is not None else ("scalar" if scalar is not None else "pair_only")
    if points is None:
        thresholds = GRID if kind == "presence" else (np.quantile(scalar, GRID) if scalar is not None else [0.0])
        points = [
            dict(pair_threshold=float(pair), accept_threshold=float(accept), accept_kind=kind)
            for pair in GRID
            for accept in thresholds
        ]
    rows = []
    for point in points:
        answers = apply_operating_point(scores, candidates, point, presence_logits=presence_logits, scalar=scalar)
        metrics = policy_metrics(answers, truth, candidates)
        values = {
            k: metrics[k]["value"]
            for k in ("present_coverage", "absent_false_accept", "set_micro_precision", "macro_recall")
        }
        rows.append(dict(**point, **values, returned_items=int(answers.sum())))
    return rows


def select_operating_point(scores, truth, candidates, *, presence_logits=None, scalar=None):
    grid = evaluate_threshold_grid(scores, truth, candidates, presence_logits=presence_logits, scalar=scalar)
    best, key = None, None
    for point in grid:
        values = {
            k: point[k] for k in ("present_coverage", "absent_false_accept", "set_micro_precision", "macro_recall")
        }
        if (
            any(v is None for v in values.values())
            or values["present_coverage"] < 0.5
            or values["absent_false_accept"] > 0.1
            or values["set_micro_precision"] < 0.9
        ):
            continue
        score = (values["macro_recall"], -point["returned_items"], point["pair_threshold"], point["accept_threshold"])
        if key is None or score > key:
            best = {k: point[k] for k in ("pair_threshold", "accept_threshold", "accept_kind")}
            key = score
    return dict(status="feasible" if best is not None else "infeasible", operating_point=best, grid=grid)


def probability_metrics(labels, probabilities):
    labels, p = np.asarray(labels), np.asarray(probabilities, float)
    if labels.shape != p.shape or labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("invalid probability observations")
    if not len(labels):
        return dict(
            auroc=None,
            ap=None,
            brier=None,
            ece=None,
            nll=None,
            num_positive=0,
            num_negative=0,
            reliability=np.empty((0, 4)),
        )
    result = binary_metrics(labels, p)
    order = np.argsort(-p)
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    tp = np.cumsum(labels[order])[ends]
    result["ap"] = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / labels.sum()) if labels.sum() else None
    reliability = []
    bins = np.minimum((p * 15).astype(int), 14)
    for b in range(15):
        mask = bins == b
        if mask.any():
            reliability.append([b, mask.sum(), p[mask].mean(), labels[mask].mean()])
    result["reliability"] = np.asarray(reliability, float).reshape(-1, 4)
    return result


def _validate_calibration_identity(identity):
    required = {
        "model_sha256",
        "preprocessing_sha256",
        "policy_sha256",
        "label_protocol",
        "relevance_space",
        "fit_gallery_version",
        "target_gallery_version",
        "fit_gallery_manifest_sha256",
        "target_gallery_manifest_sha256",
        "split_sha256",
        "track",
        "candidate_mechanism",
        "k_weights",
    }
    if (
        set(identity) != required
        or identity["relevance_space"] != "benchmark_annotation"
        or not isinstance(identity["label_protocol"], dict)
        or not identity["label_protocol"]
    ):
        raise ValueError("complete benchmark calibration identity required")
    if any(
        not isinstance(identity[k], str) or not re.fullmatch("[a-f0-9]{64}", identity[k])
        for k in required
        if k.endswith("sha256")
    ):
        raise ValueError("invalid calibration manifest hash")
    if identity["track"] not in ("natural", "controlled"):
        raise ValueError("calibration track must be natural or controlled")
    if (
        identity["fit_gallery_version"] == identity["target_gallery_version"]
        or identity["fit_gallery_manifest_sha256"] == identity["target_gallery_manifest_sha256"]
    ):
        raise ValueError("fit and target galleries must be distinct")
    if identity["k_weights"] != {str(k): 1 / 6 for k in KS}:
        raise ValueError("six K calibration mixture must be equally weighted")
    if (
        not identity["candidate_mechanism"]
        or not identity["fit_gallery_version"]
        or not identity["target_gallery_version"]
    ):
        raise ValueError("empty calibration protocol identity")


def _fit_weighted(logits, labels, weights):
    x, y, weights = (np.asarray(v, float) for v in (logits, labels, weights))
    if (
        x.ndim != 1
        or x.shape != y.shape
        or x.shape != weights.shape
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("invalid calibration observations")
    counts = dict(known_count=len(y), positive_count=int(y.sum()), negative_count=int(len(y) - y.sum()))
    if not len(y) or not y.any() or y.all():
        return dict(status="unavailable", reason="both known classes required", temperature=None, bias=None, **counts)
    weights = weights / weights.sum()

    def objective(theta):
        z = x * np.exp(-theta[0]) + theta[1]
        loss = np.sum(weights * (np.logaddexp(0.0, z) - y * z))
        residual = weights * (expit(z) - y)
        gradient = np.array([np.sum(residual * (-x * np.exp(-theta[0]))), residual.sum()])
        return loss, gradient

    fitted = minimize(objective, [0.0, 0.0], jac=True, method="L-BFGS-B", bounds=[(-8.0, 8.0), (-40.0, 40.0)])
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError("NLL calibration solver failed")
    return dict(
        status="available",
        temperature=float(np.exp(fitted.x[0])),
        bias=float(fitted.x[1]),
        nll=float(fitted.fun),
        **counts,
    )


def fit_pilot_calibration(rows, identity, *, split, has_presence=True):
    if split != "val_calibration":
        raise ValueError("fit only on val_calibration; test is sealed")
    _validate_calibration_identity(identity)
    if sorted(r["K"] for r in rows) != list(KS):
        raise ValueError("calibration requires exactly all six K panels")
    pair, px, py, pw = {}, [], [], []
    presence_x, presence_y, presence_w = [], [], []
    for r in rows:
        x, y, mask = np.asarray(r["pair_logits"]), np.asarray(r["pair_labels"]), np.asarray(r["candidate_mask"], bool)
        if x.shape != y.shape or mask.shape != x.shape or x.shape != (len(r["query_ids"]), len(r["canonical_ids"])):
            raise ValueError("calibration pair identity/shape mismatch")
        if not np.isin(y, (-1, 0, 1)).all():
            raise ValueError("invalid pair benchmark labels")
        for i, q in enumerate(r["query_ids"]):
            for j in np.flatnonzero(mask[i] & (y[i] >= 0)):
                key, observation = (str(q), r["canonical_ids"][j]), (float(x[i, j]), int(y[i, j]))
                if key in pair and (pair[key][1] != observation[1] or abs(pair[key][0] - observation[0]) > 1e-6):
                    raise ValueError("same query-canonical relationship changed across calibration K")
                pair[key] = observation
        if has_presence:
            event = np.asarray(r["presence_labels"])
            logits = np.asarray(r["presence_logits"])
            if event.shape != (len(x),) or logits.shape != event.shape or not np.isin(event, (-1, 0, 1)).all():
                raise ValueError("invalid presence calibration arrays")
            known = event >= 0
            if not known.any():
                raise ValueError("each calibration K needs known labels")
            presence_x.extend(logits[known])
            presence_y.extend(event[known])
            presence_w.extend(np.full(known.sum(), 1 / (6 * known.sum())))
    for x, y in pair.values():
        px.append(x)
        py.append(y)
        pw.append(1.0)
    pair_state = _fit_weighted(px, py, pw)
    presence_state = (
        _fit_weighted(presence_x, presence_y, presence_w)
        if has_presence
        else dict(
            status="unavailable",
            reason="arm has no trained presence head",
            temperature=None,
            bias=None,
            known_count=0,
            positive_count=0,
            negative_count=0,
        )
    )
    presence_state["positive_prevalence"] = float(np.average(presence_y, weights=presence_w)) if presence_y else None
    return dict(identity=dict(identity), pair=pair_state, presence=presence_state)


def map_threshold(raw_threshold, state):
    if not 0 <= raw_threshold <= 1 or state["status"] != "available":
        raise ValueError("threshold mapping requires available monotonic calibration")
    if raw_threshold in (0.0, 1.0):
        return float(raw_threshold)
    logit = math.log(raw_threshold / (1 - raw_threshold))
    return float(expit(logit / state["temperature"] + state["bias"]))


def calibrated_probabilities(logits, state, head, *, track, target_gallery_manifest_sha256):
    if state["identity"]["track"] != track:
        raise ValueError("calibration track cannot be mixed")
    if state["identity"]["target_gallery_manifest_sha256"] != target_gallery_manifest_sha256:
        raise ValueError("target gallery manifest identity mismatch")
    fitted = state[head]
    if fitted["status"] != "available":
        return None
    return expit(np.asarray(logits) / fitted["temperature"] + fitted["bias"])


def cluster_bootstrap(seed_rows, metric_delta, *, samples=10000, seed=20261001, confidence=0.95):
    if not seed_rows or type(samples) is not int or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid cluster bootstrap configuration")
    components = sorted(set(str(x) for r in seed_rows for x in r["component_ids"]))
    if not components:
        raise ValueError("bootstrap needs source components")
    mapping = {}
    for r in seed_rows:
        if len(r["query_ids"]) != len(r["component_ids"]):
            raise ValueError("query/component length mismatch")
        for q, c in zip(r["query_ids"], r["component_ids"]):
            if str(q) in mapping and mapping[str(q)] != str(c):
                raise ValueError("query moved between source components")
            mapping[str(q)] = str(c)
        if set(map(str, r["component_ids"])) != set(components):
            raise ValueError("all seeds must use the same source components")
    positions = [
        {c: np.flatnonzero(np.asarray(r["component_ids"]).astype(str) == c) for c in components} for r in seed_rows
    ]
    point_seeds = [metric_delta(r, np.arange(len(r["component_ids"]))) for r in seed_rows]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = rng.choice(components, len(components), replace=True)
        deltas = [metric_delta(r, np.concatenate([pos[c] for c in selected])) for r, pos in zip(seed_rows, positions)]
        if all(v is not None and np.isfinite(v) for v in deltas):
            values.append(float(np.mean(deltas)))
    alpha = (1 - confidence) / 2
    available = len(values) >= math.ceil(0.95 * samples) and all(v is not None and np.isfinite(v) for v in point_seeds)
    low, high = np.quantile(values, [alpha, 1 - alpha]) if available else (None, None)
    return dict(
        status="available" if available else "unavailable",
        point_delta=float(np.mean(point_seeds)) if all(v is not None for v in point_seeds) else None,
        ci_low=float(low) if low is not None else None,
        ci_high=float(high) if high is not None else None,
        confidence=confidence,
        samples=samples,
        valid_replicates=len(values),
        seed=seed,
        component_count=len(components),
        seed_values=point_seeds,
        method="paired source-component bootstrap; per-seed recomputation then seed mean",
    ), np.array(values)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pilot_rows(
    path,
    *,
    query_ids,
    component_ids,
    pair_logits,
    relevance,
    candidate_mask,
    official_relevance,
    presence_logits=None,
    initial_candidate_mask=None,
    **observations,
):
    """Inference-output exporter; caller supplies original source components and official GT.

    Additional numeric observations (latency_ms, token_count, initial_gt_rank,
    unique_pair_count, alias_count, permutation, score_error, answer_consistent,
    action_consistent, attempted_k_consistent, losses/gradient probes) are consumed
    by the sensitivity/diagnostic pipeline, never used to choose locked thresholds.
    """
    path = Path(path).resolve()
    arrays = dict(
        query_ids=np.asarray(query_ids, str),
        component_ids=np.asarray(component_ids, str),
        pair_logits=np.asarray(pair_logits, np.float64),
        relevance=np.asarray(relevance, np.int8),
        candidate_mask=np.asarray(candidate_mask, bool),
        official_relevance=np.asarray(official_relevance, np.int8),
    )
    if presence_logits is not None:
        arrays["presence_logits"] = np.asarray(presence_logits, float)
    if initial_candidate_mask is not None:
        arrays["initial_candidate_mask"] = np.asarray(initial_candidate_mask, bool)
    for key, value in observations.items():
        if not re.fullmatch("[a-z][a-z0-9_]*", key) or key in arrays:
            raise ValueError("unsafe/duplicate exported observation name")
        arrays[key] = np.asarray(value)
    n = len(arrays["query_ids"])
    if arrays["component_ids"].shape != (n,) or arrays["pair_logits"].ndim != 2 or arrays["pair_logits"].shape[0] != n:
        raise ValueError("export query/component/score shape mismatch")
    if any(a.dtype.kind == "O" or (a.dtype.kind in "fci" and not np.isfinite(a).all()) for a in arrays.values()):
        raise ValueError("export arrays must be finite and pickle-free")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(path), sha256=file_sha256(path), format="npz")


def _read_hashed(reference):
    path = Path(reference["path"])
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("input must be an absolute canonical regular file")
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("input SHA-256 mismatch")
    return path


def _load_group(group, bundle):
    if (
        group["arm"] not in ("A0", "P1", "A1", "A2", "A3", "A4")
        or group["track"] not in ("natural", "controlled")
        or group["mechanism"] not in ("fixed", "system", "full")
        or group["K"] not in KS
    ):
        raise ValueError("unknown benchmark arm/track/mechanism/K")
    if group["split"] not in ("val_development", "val_calibration", "val_locked"):
        raise ValueError("pilot analysis uses validation only; test is sealed")
    path = _read_hashed(group["arrays"])
    with np.load(path, allow_pickle=False) as stored:
        row = {name: stored[name] for name in stored.files}
    gallery = bundle["galleries"][group["gallery"]]
    required = {"query_ids", "component_ids", "pair_logits", "relevance", "candidate_mask", "official_relevance"}
    if not required <= set(row):
        raise ValueError("score output missing benchmark annotations/source components")
    n, m = row["pair_logits"].shape
    if (
        n == 0
        or m != len(gallery["canonical_ids"])
        or row["query_ids"].shape != (n,)
        or row["component_ids"].shape != (n,)
    ):
        raise ValueError("score output gallery/query shape mismatch")
    if (
        row["query_ids"].dtype.kind not in "US"
        or row["component_ids"].dtype.kind not in "US"
        or any(not str(q) for q in row["query_ids"])
    ):
        raise ValueError("source query/component identity must be strings")
    for key in ("relevance", "candidate_mask"):
        if row[key].shape != (n, m) or not np.isin(row[key], (0, 1)).all():
            raise ValueError("complete benchmark relationship matrix required")
    if not row["candidate_mask"].any(axis=1).all() or np.any(row["candidate_mask"].sum(axis=1) > group["K"]):
        raise ValueError("empty or over-budget candidate panel")
    if group["arm"] in ("A2", "A3", "A4") and ("presence_logits" not in row or row["presence_logits"].shape != (n,)):
        raise ValueError("trained presence arm requires presence observations")
    if not np.isfinite(row["pair_logits"]).all() or (
        "presence_logits" in row and not np.isfinite(row["presence_logits"]).all()
    ):
        raise ValueError("nonfinite model output")
    # The official and canonical annotations must agree after canonicalization.
    official_ranking(row["pair_logits"], row["official_relevance"], gallery)
    off = 0
    for j, c in enumerate(gallery["canonical_ids"]):
        end = off + len(gallery["aliases"][c])
        expected = (row["official_relevance"][:, off:end] == 1).any(axis=1)
        if not np.array_equal(expected, row["relevance"][:, j] == 1):
            raise ValueError("canonical/official benchmark GT identity mismatch")
        off = end
    for key in (
        "latency_ms",
        "token_count",
        "initial_gt_rank",
        "unique_pair_count",
        "score_error",
        "answer_consistent",
        "action_consistent",
        "attempted_k_consistent",
        "alias_count",
        "permutation",
    ):
        if key in row and (row[key].shape != (n,) or not np.isfinite(row[key]).all()):
            raise ValueError("diagnostic observations must align with query rows")
    row.update(group=group, gallery=gallery)
    return row


def _validate_bundle(bundle):
    if (
        bundle.get("schema_version") != 1
        or bundle.get("relevance_space") != "benchmark_annotation"
        or not isinstance(bundle.get("label_protocol"), dict)
        or not bundle["label_protocol"]
    ):
        raise ValueError("benchmark label/protocol identity required")
    for field in ("preprocessing_sha256", "policy_sha256"):
        if not re.fullmatch("[a-f0-9]{64}", bundle.get(field, "")):
            raise ValueError("invalid frozen preprocessing/policy identity")
    seen = set()
    for group in bundle["groups"]:
        key = (group["arm"], group["seed"], group["split"], group["track"], group["mechanism"], group["K"])
        if key in seen:
            raise ValueError("duplicate analysis group")
        seen.add(key)
        if type(group["seed"]) is not int or group["seed"] not in (0, 1, 2):
            raise ValueError("pilot requires predeclared seeds 0/1/2")
        for name in ("model_sha256", "split_sha256"):
            if not re.fullmatch("[a-f0-9]{64}", group[name]):
                raise ValueError("invalid group frozen identity")
    for g in bundle["galleries"].values():
        if not re.fullmatch("[a-f0-9]{64}", g["manifest_sha256"]) or not g["version"]:
            raise ValueError("invalid gallery manifest identity")


def _key(row):
    g = row["group"]
    return f"{g['arm']}_seed{g['seed']}_{g['track']}_{g['mechanism']}_K{g['K']}"


def _model_key(row):
    g = row["group"]
    return f"{g['arm']}_seed{g['seed']}"


def _top1(row):
    scores, candidates = row["pair_logits"], row["candidate_mask"].astype(bool)
    # Canonical ID tie resolution is invariant to storage permutation.
    ids = np.asarray(row["gallery"]["canonical_ids"])
    index = np.array([np.lexsort((ids, -np.where(mask, s, -np.inf)))[0] for s, mask in zip(scores, candidates)])
    error = ~row["relevance"][np.arange(len(index)), index].astype(bool)
    return index, error


def _raw_curve(row, scalar=None):
    index, error = _top1(row)
    pair = expit(row["pair_logits"][np.arange(len(index)), index])
    confidence = scalar if scalar is not None else pair
    if scalar is None and row["group"]["arm"] in ("A2", "A3", "A4"):
        confidence = np.minimum(pair, expit(row["presence_logits"]))
    return risk_curve(confidence, error)


def _store(context, key, values, references):
    reference = context.array_store.write(key, np.asarray(values)).as_dict()
    references.append(reference)
    return reference


def _write_json(path, payload):
    from experiment_artifacts import artifact_json_bytes

    with Path(path).open("xb") as stream:
        stream.write(artifact_json_bytes(payload))


def _freeze(context, bundle, rows, references):
    if any(r["group"]["split"] == "val_locked" for r in rows):
        raise ValueError("development freeze must not read locked observations")
    development = [
        r
        for r in rows
        if r["group"]["split"] == "val_development" and r["group"]["mechanism"] == "fixed" and r["group"]["K"] == 32
    ]
    if not development:
        raise ValueError("development K32 fixed panels required")
    policies, scalar_policies, h1_scores, h2_scores = {}, {}, {s: [] for s in SCALARS}, {s: [] for s in SCALARS}
    for r in development:
        g = r["group"]
        key = _key(r)
        presence = r.get("presence_logits") if g["arm"] in ("A2", "A3", "A4") else None
        selected = select_operating_point(
            r["pair_logits"], r["relevance"], r["candidate_mask"], presence_logits=presence
        )
        grid = selected.pop("grid")
        selected["grid_reference"] = _store(
            context,
            key + "_grid",
            [
                [
                    p["pair_threshold"],
                    p["accept_threshold"],
                    *[
                        p[k] if p[k] is not None else -1.0
                        for k in ("present_coverage", "absent_false_accept", "set_micro_precision", "macro_recall")
                    ],
                ]
                for p in grid
            ],
            references,
        )
        policies[key] = selected
        if g["arm"] == "A1":
            confidences = scalar_confidences(r["pair_logits"], r["candidate_mask"])
            for scalar, confidence in confidences.items():
                if g["track"] == "controlled":
                    chosen = select_operating_point(
                        r["pair_logits"], r["relevance"], r["candidate_mask"], scalar=confidence
                    )
                    chosen.pop("grid")
                    scalar_policies[f"{key}_{scalar}"] = chosen
                    if chosen["operating_point"] is not None:
                        answers = apply_operating_point(
                            r["pair_logits"], r["candidate_mask"], chosen["operating_point"], scalar=confidence
                        )
                        h1_scores[scalar].append(
                            policy_metrics(answers, r["relevance"], r["candidate_mask"])["macro_recall"]["value"]
                        )
                    else:
                        h1_scores[scalar].append(None)
                else:
                    h2_scores[scalar].append(_raw_curve(r, confidence)["aurc"])
    h1 = [s for s in SCALARS if h1_scores[s] and all(v is not None for v in h1_scores[s])]
    h2 = [s for s in SCALARS if h2_scores[s]]
    strongest = {
        "B-H1": max(h1, key=lambda s: np.mean(h1_scores[s])) if h1 else None,
        "B-H2": min(h2, key=lambda s: np.mean(h2_scores[s])) if h2 else None,
    }
    identities = {}
    for r in development:
        key = _model_key(r)
        if key in identities and identities[key] != r["group"]["model_sha256"]:
            raise ValueError("development arm changed model identity across tracks")
        identities[key] = r["group"]["model_sha256"]
    calibration = {}
    for r in development:
        arm, seed, track = (r["group"][k] for k in ("arm", "seed", "track"))
        fit = [
            f
            for f in rows
            if f["group"]["split"] == "val_calibration"
            and f["group"]["arm"] == arm
            and f["group"]["seed"] == seed
            and f["group"]["track"] == track
            and f["group"]["mechanism"] == "fixed"
        ]
        state_key = f"{arm}_seed{seed}_{track}"
        if not fit:
            calibration[state_key] = dict(status="unavailable", reason="no calibration observations")
            continue
        if len(set(f["group"]["split_sha256"] for f in fit)) != 1:
            raise ValueError("calibration K panels changed source split identity")
        fit_gallery = fit[0]["gallery"]
        if any(f["gallery"] != fit_gallery for f in fit) or any(
            f["group"]["model_sha256"] != r["group"]["model_sha256"] for f in fit
        ):
            raise ValueError("calibration changed gallery/model identity")
        target = bundle["galleries"]["locked"]
        identity = dict(
            model_sha256=r["group"]["model_sha256"],
            preprocessing_sha256=bundle["preprocessing_sha256"],
            policy_sha256=bundle["policy_sha256"],
            label_protocol=bundle["label_protocol"],
            relevance_space=bundle["relevance_space"],
            fit_gallery_version=fit_gallery["version"],
            target_gallery_version=target["version"],
            fit_gallery_manifest_sha256=fit_gallery["manifest_sha256"],
            target_gallery_manifest_sha256=target["manifest_sha256"],
            split_sha256=fit[0]["group"]["split_sha256"],
            track=track,
            candidate_mechanism="fixed",
            k_weights={str(k): 1 / 6 for k in KS},
        )
        fit_rows = [
            dict(
                K=f["group"]["K"],
                query_ids=f["query_ids"],
                canonical_ids=f["gallery"]["canonical_ids"],
                pair_logits=f["pair_logits"],
                pair_labels=f["relevance"],
                candidate_mask=f["candidate_mask"],
                presence_logits=f.get("presence_logits"),
                presence_labels=(f["relevance"].astype(bool) & f["candidate_mask"].astype(bool))
                .any(axis=1)
                .astype(int),
            )
            for f in fit
        ]
        calibration[state_key] = fit_pilot_calibration(
            fit_rows, identity, split="val_calibration", has_presence=arm in ("A2", "A3", "A4")
        )
    frozen = dict(
        schema_version=1,
        relevance_space=bundle["relevance_space"],
        label_protocol=bundle["label_protocol"],
        preprocessing_sha256=bundle["preprocessing_sha256"],
        policy_sha256=bundle["policy_sha256"],
        source_sha=context.manifest["git_sha"],
        analysis_sha256=file_sha256(__file__),
        target_gallery=bundle["galleries"]["locked"],
        strongest_scalar=strongest,
        policies=policies,
        scalar_policies=scalar_policies,
        calibration=calibration,
        model_identities={_model_key(r): r["group"]["model_sha256"] for r in development},
        development_components=sorted(set(str(c) for r in development for c in r["component_ids"])),
        calibration_components=sorted(
            set(str(c) for r in rows if r["group"]["split"] == "val_calibration" for c in r["component_ids"])
        ),
        array_references=references,
    )
    if set(frozen["development_components"]) & set(frozen["calibration_components"]):
        raise ValueError("development and calibration source components overlap")
    _write_json(context.paths.provenance / "pilot_frozen.json", frozen)
    return frozen


def _evaluate_group(context, bundle, frozen, row, references, *, raw_rc):
    g, key = row["group"], _key(row)
    state = frozen["calibration"].get(f"{g['arm']}_seed{g['seed']}_{g['track']}")
    calibrated = state is not None and "identity" in state and g["mechanism"] == "fixed"
    pair_p = (
        calibrated_probabilities(
            row["pair_logits"],
            state,
            "pair",
            track=g["track"],
            target_gallery_manifest_sha256=row["gallery"]["manifest_sha256"],
        )
        if calibrated
        else None
    )
    presence_p = (
        calibrated_probabilities(
            row["presence_logits"],
            state,
            "presence",
            track=g["track"],
            target_gallery_manifest_sha256=row["gallery"]["manifest_sha256"],
        )
        if calibrated and g["arm"] in ("A2", "A3", "A4")
        else None
    )
    _, error = _top1(row)
    first, _ = _top1(row)
    top_pair = (
        expit(row["pair_logits"][np.arange(len(first)), first]) if raw_rc else pair_p[np.arange(len(first)), first]
    )
    confidence = top_pair.copy()
    if g["arm"] in ("A2", "A3", "A4"):
        confidence = np.minimum(top_pair, expit(row["presence_logits"]) if raw_rc else presence_p)
    curve, pair_curve = risk_curve(confidence, error), risk_curve(top_pair, error)
    full_row = dict(row, candidate_mask=np.ones_like(row["candidate_mask"], bool))
    full_first, full_error = _top1(full_row)
    full_pair = expit(row["pair_logits"][np.arange(len(full_first)), full_first])
    full_pair_curve = risk_curve(full_pair, full_error)
    full_curve = full_pair_curve
    if g["arm"] in ("A2", "A3", "A4"):
        full_curve = (
            risk_curve(np.minimum(full_pair, expit(row["full_presence_logits"])), full_error)
            if "full_presence_logits" in row
            else None
        )
    point_key = f"{g['arm']}_seed{g['seed']}_{g['track']}_fixed_K32"
    point = frozen["policies"].get(point_key, {}).get("operating_point")
    answers = apply_operating_point(
        row["pair_logits"],
        row["candidate_mask"],
        point,
        presence_logits=row.get("presence_logits") if g["arm"] in ("A2", "A3", "A4") else None,
    )
    metrics = (
        policy_metrics(
            answers, row["relevance"], row["candidate_mask"], initial_candidates=row.get("initial_candidate_mask")
        )
        if answers is not None
        else dict(status="unavailable", reason="development operating point infeasible")
    )
    top1_answers = apply_operating_point(
        row["pair_logits"],
        row["candidate_mask"],
        point,
        presence_logits=row.get("presence_logits") if g["arm"] in ("A2", "A3", "A4") else None,
        top1=True,
        canonical_ids=row["gallery"]["canonical_ids"],
    )
    scalars = scalar_confidences(row["pair_logits"], row["candidate_mask"])
    scalar_results = {}
    scalar_answers = {}
    if g["arm"] == "A1":
        for scalar, values in scalars.items():
            scalar_point = (
                frozen["scalar_policies"]
                .get(f"A1_seed{g['seed']}_controlled_fixed_K32_{scalar}", {})
                .get("operating_point")
            )
            scalar_answer = apply_operating_point(
                row["pair_logits"], row["candidate_mask"], scalar_point, scalar=values
            )
            scalar_answers[scalar] = scalar_answer
            c = risk_curve(values, error)
            scalar_results[scalar] = dict(
                aurc=c["aurc"],
                rc_reference=_store(
                    context, key + "_" + scalar + "_rc", np.column_stack([c["coverage"], c["risk"]]), references
                ),
                policy=policy_metrics(scalar_answer, row["relevance"], row["candidate_mask"])
                if scalar_answer is not None
                else dict(status="unavailable"),
            )
    candidate = row["candidate_mask"].astype(bool)
    event = (row["relevance"].astype(bool) & candidate).any(axis=1).astype(int)
    probabilities = {}
    for head, labels, p in [
        ("pair", row["relevance"][candidate], pair_p[candidate] if pair_p is not None else None),
        ("presence", event, presence_p),
    ]:
        if p is None:
            probabilities[head] = dict(
                status="unavailable", reason="matching track calibration unavailable or head untrained"
            )
            continue
        pm = probability_metrics(labels, p)
        reliability = pm.pop("reliability")
        pm["reliability_reference"] = _store(context, key + "_" + head + "_reliability", reliability, references)
        pm["status"] = "available"
        if head == "presence":
            absence = probability_metrics(1 - labels, 1 - p)
            absence.pop("reliability")
            pm["candidate_absence_detection"] = absence
        probabilities[head] = pm
    descriptive_grid = evaluate_threshold_grid(
        row["pair_logits"],
        row["relevance"],
        candidate,
        presence_logits=row.get("presence_logits") if g["arm"] in ("A2", "A3", "A4") else None,
    )
    grid_values = [
        [
            p["pair_threshold"],
            p["accept_threshold"],
            *[
                p[k] if p[k] is not None else -1.0
                for k in ("present_coverage", "absent_false_accept", "set_micro_precision", "macro_recall")
            ],
        ]
        for p in descriptive_grid
    ]
    # Locked grid values are descriptive and never change the development-selected policy.
    grid_reference = _store(context, key + "_locked_descriptive_grid", grid_values, references)
    cost = dict(
        unique_pair_scores=rate(np.sum(row.get("unique_pair_count", candidate.sum(axis=1))), len(candidate)),
        latency_p50_ms=float(np.median(row["latency_ms"])) if "latency_ms" in row else None,
        latency_p95_ms=float(np.quantile(row["latency_ms"], 0.95)) if "latency_ms" in row else None,
        attempted_k=g["K"],
    )
    invariance = dict(status="unavailable")
    checks = ("score_error", "answer_consistent", "action_consistent", "attempted_k_consistent")
    if all(k in row for k in checks):
        invariance = dict(
            status="available",
            max_fp32_score_error=float(np.max(row["score_error"])),
            answer_consistent=bool(np.all(row["answer_consistent"] == 1)),
            action_consistent=bool(np.all(row["action_consistent"] == 1)),
            attempted_k_consistent=bool(np.all(row["attempted_k_consistent"] == 1)),
            passed=bool(np.max(row["score_error"]) <= 1e-6 and all(np.all(row[k] == 1) for k in checks[1:])),
        )
    # Freeze raw policy actions. Map thresholds for deployment; NLL fitting never re-selects policy.
    mapped = None
    if point is not None and pair_p is not None and (point["accept_kind"] != "presence" or presence_p is not None):
        transformed_actions = apply_calibrated_operating_point(
            row["pair_logits"], candidate, point, state, presence_logits=row.get("presence_logits")
        )
        if not np.array_equal(transformed_actions, answers):
            raise ValueError("monotonic calibration changed frozen actions")
        mapped = dict(
            action_equivalence_verified=True,
            pair_threshold=map_threshold(point["pair_threshold"], state["pair"]),
            accept_threshold=map_threshold(point["accept_threshold"], state["presence"])
            if point["accept_kind"] == "presence" and presence_p is not None
            else point["accept_threshold"],
            accept_kind=point["accept_kind"],
            action_rule="compare transformed logits at mapped raw-logit boundary to avoid sigmoid saturation",
        )
    summary = dict(
        **g,
        gallery_manifest_sha256=row["gallery"]["manifest_sha256"],
        gallery_size=len(row["gallery"]["canonical_ids"]),
        label_protocol=bundle["label_protocol"],
        relevance_space="benchmark_annotation",
        official_ranking=official_ranking(row["pair_logits"], row["official_relevance"], row["gallery"]),
        policy=metrics,
        top1_policy=policy_metrics(top1_answers, row["relevance"], candidate)
        if top1_answers is not None
        else dict(status="unavailable"),
        aurc=curve["aurc"],
        pair_only_aurc=pair_curve["aurc"],
        full_gallery_aurc=full_curve["aurc"] if full_curve is not None else None,
        full_gallery_pair_only_aurc=full_pair_curve["aurc"],
        full_gallery_rc_rule="raw sigmoid; full-gallery presence only; finite-panel calibration cannot migrate silently",
        full_gallery_rc_reference=_store(
            context, key + "_full_rc", np.column_stack([full_curve["coverage"], full_curve["risk"]]), references
        )
        if full_curve is not None
        else None,
        full_gallery_pair_rc_reference=_store(
            context,
            key + "_full_pair_rc",
            np.column_stack([full_pair_curve["coverage"], full_pair_curve["risk"]]),
            references,
        ),
        top1_accuracy=rate((~error).sum(), len(error)),
        candidate_gt_recall=rate(event.sum(), len(event)),
        probabilities=probabilities,
        scalars=scalar_results,
        cost=cost,
        invariance=invariance,
        operating_point=point,
        mapped_operating_point=mapped,
        descriptive_grid_reference=grid_reference,
        rc_probability_rule="raw sigmoid fallback for all related arms"
        if raw_rc
        else "track-specific calibrated min; selection score, not correctness probability",
        rc_reference=_store(context, key + "_rc", np.column_stack([curve["coverage"], curve["risk"]]), references),
        pair_only_rc_reference=_store(
            context, key + "_pair_rc", np.column_stack([pair_curve["coverage"], pair_curve["risk"]]), references
        ),
    )
    for name in (
        "pair_logits",
        "presence_logits",
        "full_presence_logits",
        "candidate_mask",
        "relevance",
        "official_relevance",
        "initial_candidate_mask",
        "latency_ms",
        "token_count",
        "initial_gt_rank",
        "unique_pair_count",
        "score_error",
        "alias_count",
        "permutation",
        "loss",
        "match_gradient_norm",
        "presence_gradient_norm",
        "gradient_cosine",
        "update_norm",
        "known_label_count",
    ):
        if name in row:
            _store(context, key + "_" + name, row[name], references)
    if answers is not None:
        _store(context, key + "_returned", answers, references)
    return dict(
        row=row,
        summary=summary,
        answers=answers,
        scalar_answers=scalar_answers,
        error=error,
        confidence=confidence,
        scalars=scalars,
        curve=curve,
        pair_curve=pair_curve,
        full_curve=full_curve,
        full_pair_curve=full_pair_curve,
    )


def _effect(
    context, evaluated, first_arm, second_arm, *, track, metric, scalar=None, samples=10000, confidence=0.95, references
):
    seed_rows = []
    for seed in (0, 1, 2):

        def select(arm):
            return [
                e
                for e in evaluated
                if e["row"]["group"]["arm"] == arm
                and e["row"]["group"]["seed"] == seed
                and e["row"]["group"]["track"] == track
                and e["row"]["group"]["mechanism"] == "fixed"
                and e["row"]["group"]["K"] == 32
            ]

        a, b = select(first_arm), select(second_arm)
        if not a or not b:
            continue
        a, b = a[0], b[0]
        # Identical query/panel order is part of paired evaluation identity, not guessed from metric arrays.
        for field in ("query_ids", "component_ids", "relevance", "candidate_mask"):
            if not np.array_equal(a["row"][field], b["row"][field]):
                raise ValueError("paired arms changed query/panel/source-component identity")
        seed_rows.append(
            dict(query_ids=a["row"]["query_ids"], component_ids=a["row"]["component_ids"], first=a, second=b)
        )
    if len(seed_rows) != 3:
        return dict(status="unavailable", reason="paired effects require all three predeclared seeds")

    def value(e, index, is_second):
        if metric == "aurc":
            conf = e["scalars"][scalar] if is_second and scalar else e["confidence"]
            return risk_curve(conf[index], e["error"][index])["aurc"]
        if metric == "top1_accuracy":
            return float(np.mean(~e["error"][index]))
        answers = e["scalar_answers"].get(scalar) if is_second and scalar else e["answers"]
        if answers is None:
            return None
        r = e["row"]
        return policy_metrics(answers[index], r["relevance"][index], r["candidate_mask"][index])[metric]["value"]

    def delta(r, index):
        a, b = value(r["first"], index, False), value(r["second"], index, True)
        return a - b if a is not None and b is not None else None

    result, replicates = cluster_bootstrap(seed_rows, delta, samples=samples, confidence=confidence)
    result.update(
        first=first_arm,
        second=second_arm,
        track=track,
        metric=metric,
        scalar=scalar,
        samples_reference=_store(
            context, f"effect_{first_arm}_{second_arm}_{track}_{metric}_{scalar or 'pair'}", replicates, references
        ),
    )
    return result


def _hypotheses(effects, gates):
    tests = {}
    for hypothesis, names in [
        ("B-H1", ("absent_false_accept", "present_coverage")),
        ("B-H2", ("aurc", "top1_accuracy")),
    ]:
        selected = [e for e in effects if e.get("hypothesis") == hypothesis]
        if len(selected) != 4 or any(e["status"] != "available" for e in selected):
            tests[hypothesis] = dict(
                status="inconclusive", reason="unavailable frozen comparator, operating point or three-seed interval"
            )
            continue
        comparisons = []
        for comparator in ("pair_only", "strongest_scalar"):
            by_metric = {e["metric"]: e for e in selected if e["comparator"] == comparator}
            primary, safety = by_metric[names[0]], by_metric[names[1]]
            minimum = 0.05 if hypothesis == "B-H1" else 0.01
            tolerance = 0.05 if hypothesis == "B-H1" else 0.01
            passed = (
                primary["point_delta"] <= -minimum
                and primary["ci_high"] < 0
                and safety["point_delta"] >= -tolerance
                and safety["ci_low"] >= -tolerance
            )
            failed = primary["ci_low"] > -minimum or safety["ci_high"] < -tolerance
            comparisons.append(
                dict(comparator=comparator, status="PASS" if passed else ("FAIL" if failed else "inconclusive"))
            )
        tests[hypothesis] = dict(
            status="PASS"
            if all(c["status"] == "PASS" for c in comparisons)
            else ("FAIL" if any(c["status"] == "FAIL" for c in comparisons) else "inconclusive"),
            comparisons=comparisons,
        )
    qualified = all(
        gates.get(k) is True
        for k in ("data", "four_gpu_qualification", "gradients", "stability", "invariance", "power")
    )
    if not qualified:
        return tests, "inconclusive"
    if all(t["status"] == "PASS" for t in tests.values()):
        return tests, "PASS-benchmark-pilot"
    if any(t["status"] == "FAIL" for t in tests.values()):
        return tests, "FAIL"
    return tests, "inconclusive"


def _strata(row, answers):
    truth, candidates = row["relevance"], row["candidate_mask"].astype(bool)
    event = (truth.astype(bool) & candidates).any(axis=1)
    masks = {"candidate_present": event, "candidate_absent": ~event}
    if "token_count" in row:
        for lo, hi in ((1, 8), (9, 16), (17, 32)):
            masks[f"tokens_{lo}_{hi}"] = (row["token_count"] >= lo) & (row["token_count"] <= hi)
    if "initial_gt_rank" in row:
        for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)):
            masks[f"initial_GT_rank_{lo}_{hi}"] = (row["initial_gt_rank"] >= lo) & (row["initial_gt_rank"] <= hi)
        masks["initial_GT_rank_above128"] = row["initial_gt_rank"] > 128
    return {
        name: policy_metrics(answers[mask], truth[mask], candidates[mask])
        if answers is not None
        else dict(status="unavailable", denominator=int(mask.sum()))
        for name, mask in masks.items()
    }


def _cases(bundle, evaluated):
    categories = {
        k: []
        for k in (
            "present_GT_rejected",
            "candidate_absent_accepted",
            "expansion_recovered_or_exhausted",
            "pair_presence_conflict",
        )
    }
    for e in evaluated:
        r = e["row"]
        answer = e["answers"]
        if answer is None:
            continue
        truth = r["relevance"].astype(bool)
        candidate = r["candidate_mask"].astype(bool)
        present = (truth & candidate).any(axis=1)
        returned = answer.any(axis=1)
        hit = (truth & answer).any(axis=1)
        pair_top = expit(np.max(np.where(candidate, r["pair_logits"], -np.inf), axis=1))
        presence = expit(r["presence_logits"]) if r["group"]["arm"] in ("A2", "A3", "A4") else None
        initial = r.get("initial_candidate_mask", candidate)
        miss = ~(truth & initial).any(axis=1)
        events = {
            "present_GT_rejected": present & ~returned,
            "candidate_absent_accepted": ~present & returned,
            "expansion_recovered_or_exhausted": miss,
            "pair_presence_conflict": (pair_top >= 0.5) != (presence >= 0.5)
            if presence is not None
            else np.zeros(len(truth), bool),
        }
        for name, mask in events.items():
            for i in np.flatnonzero(mask):
                q = str(r["query_ids"][i])
                detail = bundle.get("case_metadata", {}).get(q, {})
                categories[name].append(
                    dict(
                        query_id=q,
                        arm=r["group"]["arm"],
                        seed=r["group"]["seed"],
                        track=r["group"]["track"],
                        K=r["group"]["K"],
                        source_component=str(r["component_ids"][i]),
                        query=detail.get("query_text"),
                        frames=detail.get("frame_references", []),
                        official_GT=[
                            v
                            for j, c in enumerate(r["gallery"]["canonical_ids"])
                            for v, positive in zip(
                                r["gallery"]["aliases"][c],
                                r["official_relevance"][
                                    i,
                                    sum(
                                        len(r["gallery"]["aliases"][x]) for x in r["gallery"]["canonical_ids"][:j]
                                    ) : sum(
                                        len(r["gallery"]["aliases"][x]) for x in r["gallery"]["canonical_ids"][: j + 1]
                                    ),
                                ],
                            )
                            if positive == 1
                        ],
                        answer_ids=[r["gallery"]["canonical_ids"][j] for j in np.flatnonzero(answer[i])],
                        pair_top_probability=float(pair_top[i]),
                        presence_probability=float(presence[i]) if presence is not None else None,
                        action="return" if returned[i] else "abstain",
                        expansion_hit=bool(hit[i]),
                        label_scope="benchmark_annotation",
                    )
                )
    # Query ID hash chooses cases before viewing text/images; duplicate panels do not get extra quota.
    for name, values in categories.items():
        values.sort(
            key=lambda v: (hashlib.sha256(v["query_id"].encode()).hexdigest(), v["arm"], v["seed"], v["track"], v["K"])
        )
        picked = []
        seen = set()
        for value in values:
            if value["query_id"] not in seen:
                picked.append(value)
                seen.add(value["query_id"])
            if len(picked) == 3:
                break
        categories[name] = picked
    return categories


def _plots(context, bundle, evaluated, effects, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    gallery = bundle["galleries"]["locked"]
    caption = f"val_locked | benchmark_annotation | natural/controlled, fixed/system | seeds={sorted(set(e['row']['group']['seed'] for e in evaluated))}\nG={len(gallery['canonical_ids'])} n={sorted(set(len(e['error']) for e in evaluated))}; SHA={gallery['manifest_sha256']}\npaired source-component bootstrap: 97.5% primary / 95% auxiliary; seed scatter separate"

    def save(name, fig, axes):
        fig.suptitle(caption, fontsize=8)
        for ax in np.asarray(axes, dtype=object).reshape(-1):
            ax.grid(alpha=0.2)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        target = context.paths.plots / (name + ".png")
        # Analysis outputs are immutable; fail on pre-existing scientific figure.
        if target.exists():
            raise ValueError("analysis figure already exists; use a new reserved run")
        fig.savefig(target, dpi=120)
        plt.close(fig)
        paths[name] = dict(path=target.relative_to(context.paths.root).as_posix(), sha256=file_sha256(target))

    fig, rc_axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = rc_axes[1]
    rc_axes[0].set(title="full gallery", xlabel="coverage", ylabel="benchmark top-1 error risk")
    ax.set_title("finite-budget service")
    for e in evaluated:
        g = e["row"]["group"]
        if g["track"] == "natural" and g["K"] == 32:
            label = f"{g['arm']} s{g['seed']} {g['mechanism']} n={len(e['error'])} G={len(e['row']['gallery']['canonical_ids'])} SHA={e['row']['gallery']['manifest_sha256'][:8]}"
            if e["full_curve"] is not None:
                rc_axes[0].step(e["full_curve"]["coverage"], e["full_curve"]["risk"], where="pre", label=label)
            rc_axes[0].plot(e["full_pair_curve"]["coverage"], e["full_pair_curve"]["risk"], "--", alpha=0.5)
            ax.step(e["curve"]["coverage"], e["curve"]["risk"], where="pre", label=label)
            ax.plot(e["pair_curve"]["coverage"], e["pair_curve"]["risk"], "--", alpha=0.5)
            p = e["summary"]["policy"]
            if "coverage" in p and p["wrong_return"]["value"] is not None:
                ax.scatter(p["coverage"]["value"], p["wrong_return"]["value"], s=20)
    ax.set(xlabel="top-1 coverage (ties grouped)", ylabel="benchmark top-1 error risk")
    if ax.lines:
        ax.legend(fontsize=6)
    save("rc", fig, rc_axes)
    for name, xmetric, ymetric, xlabel, ylabel in [
        (
            "false_accept_coverage",
            "present_coverage",
            "absent_false_accept",
            "present return coverage",
            "candidate-absent false accept",
        ),
        ("set_precision_recall", "macro_recall", "set_micro_precision", "macro set recall", "micro set precision"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, track in zip(axes, ("controlled", "natural")):
            ax.set(title=track, xlabel=xlabel, ylabel=ylabel)
            for e in evaluated:
                if e["row"]["group"]["track"] != track:
                    continue
                p = e["summary"]["policy"]
                grid_ref = e["summary"]["descriptive_grid_reference"]
                grid = np.load(context.paths.root / grid_ref["path"], allow_pickle=False)
                xcol, ycol = (2, 3) if name == "false_accept_coverage" else (5, 4)
                known = (grid[:, xcol] >= 0) & (grid[:, ycol] >= 0)
                if known.any():
                    ax.plot(grid[known, xcol], grid[known, ycol], ".", alpha=0.08, markersize=1)
                if xmetric in p and p[xmetric]["value"] is not None and p[ymetric]["value"] is not None:
                    ax.scatter(p[xmetric]["value"], p[ymetric]["value"], label=_key(e["row"]))
            if ax.collections:
                ax.legend(fontsize=5)
        save(name, fig, axes)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for e in evaluated:
        track = e["row"]["group"]["track"]
        for head in ("pair", "presence"):
            p = e["summary"]["probabilities"][head]
            ax = axes[0 if track == "natural" else 1, 0 if head == "pair" else 1]
            ax.set(title=f"{track} {head}", xlabel="mean predicted probability", ylabel="benchmark event frequency")
            if p["status"] == "available":
                ref = p["reliability_reference"]
                values = np.load(context.paths.root / ref["path"], allow_pickle=False)
                ax.plot(values[:, 2], values[:, 3], marker=".", label=_key(e["row"]))
    for ax in axes.reshape(-1):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if len(ax.lines) > 1:
            ax.legend(fontsize=4)
        else:
            ax.text(0.1, 0.8, "Calibration unavailable")
    save("reliability", fig, axes)
    fig, ax = plt.subplots(figsize=(8, 4))
    available = [e for e in effects if e["status"] == "available"]
    for i, e in enumerate(available):
        ax.errorbar(
            e["point_delta"], i, xerr=[[e["point_delta"] - e["ci_low"]], [e["ci_high"] - e["point_delta"]]], fmt="o"
        )
    ax.set_yticks(
        range(len(available)),
        [f"{e['first']}-{e['second']} {e['metric']} {e.get('scalar') or 'pair'}" for e in available],
        fontsize=6,
    )
    ax.axvline(0, color="gray")
    ax.set_xlabel("paired effect with query-component CI; seed scatter in summary")
    if not available:
        ax.text(0.1, 0.5, "Three-seed paired effects unavailable", transform=ax.transAxes)
    save("ablation_effects", fig, [ax])
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for e in evaluated:
        g = e["row"]["group"]
        s = e["summary"]
        label = f"{g['arm']} seed{g['seed']} {g['track']} {g['mechanism']}"
        for ax, value, title in zip(
            axes.reshape(-1),
            [
                s["candidate_gt_recall"]["value"],
                s["policy"].get("wrong_return", {}).get("value"),
                s["probabilities"]["presence"].get("brier"),
                s["cost"]["latency_p95_ms"],
            ],
            ["GT enters candidates", "wrong return / all returns", "presence Brier", "latency P95 ms"],
        ):
            ax.set(xlabel="K (static and system separately)", ylabel=title)
            if value is not None:
                ax.scatter(g["K"], value, label=label)
    for ax in axes.reshape(-1):
        if ax.collections:
            ax.legend(fontsize=4)
    save("k_cost", fig, axes)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    found = False
    for e in evaluated:
        r = e["row"]
        if "score_error" not in r:
            continue
        found = True
        for ax, key in zip(axes, ("alias_count", "permutation")):
            if key in r:
                ax.scatter(r[key], r["score_error"], label=_key(r))
    for ax, key in zip(axes, ("alias count 1/2/4/8", "five permutations")):
        ax.set(xlabel=key, ylabel="FP32 absolute score error")
        ax.axhline(1e-6, color="red", linestyle="--")
        if not found:
            ax.text(0.1, 0.5, "Measured invariance probes unavailable", transform=ax.transAxes)
    save("alias_permutation", fig, axes)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    found = False
    for e in evaluated:
        r = e["row"]
        for name in (
            "loss",
            "match_gradient_norm",
            "presence_gradient_norm",
            "gradient_cosine",
            "update_norm",
            "known_label_count",
        ):
            if name in r:
                found = True
                v = r[name].reshape(-1)
                ax = axes[0] if name == "loss" else axes[1]
                ax.plot(np.arange(len(v)) * 20, v, label=_key(r) + " " + name)
    for ax in axes:
        ax.set(xlabel="optimizer update (probes every 20 updates)", ylabel="loss / measured gradient diagnostics")
        if found:
            ax.legend(fontsize=4)
        else:
            ax.text(0.1, 0.5, "Measured training probes unavailable", transform=ax.transAxes)
    save("loss_gradient", fig, axes)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    lines = []
    for name, cases in report["cases"].items():
        lines.append(name)
        for case in cases:
            lines.append(
                f"  {case['query_id']} {case['arm']} s{case['seed']} {case['track']} K={case['K']} GT={','.join(case['official_GT'])} action={case['action']}"
            )
    ax.text(0, 1, "\n".join(lines) or "No eligible cases", va="top", fontsize=7)
    save("case_chains", fig, [ax])
    return paths


def run_pilot_analysis(context, bundle, *, mode, bootstrap_samples=10000):
    """Connected CPU pipeline. Freeze must precede one-shot locked analysis.

    This API operates inside an already verified running Scheme A context;
    scripts/analyze_semantic_pilot.py binds all config/bundle/array inputs to the
    committed manifest and publishes via the shared lifecycle.
    """
    _validate_bundle(bundle)
    references = []
    if mode == "freeze":
        if any(g["split"] == "val_locked" for g in bundle["groups"]):
            raise ValueError("freeze must not consume locked arrays")
        rows = [_load_group(g, bundle) for g in bundle["groups"]]
        return _freeze(context, bundle, rows, references)
    if mode != "locked" or any(g["split"] != "val_locked" for g in bundle["groups"]):
        raise ValueError("locked analysis only accepts frozen validation locked observations")
    frozen = json.loads(_read_hashed(bundle["frozen"]).read_text())
    for key in ("relevance_space", "label_protocol", "preprocessing_sha256", "policy_sha256"):
        if frozen[key] != bundle[key]:
            raise ValueError("frozen benchmark/model/policy identity mismatch")
    if frozen["analysis_sha256"] != file_sha256(__file__) or frozen["target_gallery"] != bundle["galleries"]["locked"]:
        raise ValueError("frozen analysis source or target gallery identity mismatch")
    rows = [_load_group(g, bundle) for g in bundle["groups"]]
    for r in rows:
        if (
            frozen["model_identities"].get(_model_key(r)) != r["group"]["model_sha256"]
            or r["gallery"] != frozen["target_gallery"]
        ):
            raise ValueError("frozen model/target identity mismatch")
        if set(map(str, r["component_ids"])) & set(frozen["development_components"] + frozen["calibration_components"]):
            raise ValueError("locked source components overlap fit/development partitions")
    # A missing natural calibration forces the same raw RC rule on every related arm.
    natural = [r for r in rows if r["group"]["track"] == "natural" and r["group"]["mechanism"] == "fixed"]
    raw_rc = False
    for r in natural:
        g = r["group"]
        state = frozen["calibration"].get(f"{g['arm']}_seed{g['seed']}_natural", {})
        if state.get("pair", {}).get("status") != "available" or (
            g["arm"] in ("A2", "A3", "A4") and state.get("presence", {}).get("status") != "available"
        ):
            raw_rc = True
    evaluated = [
        _evaluate_group(
            context,
            bundle,
            frozen,
            r,
            references,
            raw_rc=raw_rc or r["group"]["mechanism"] != "fixed" or r["group"]["track"] != "natural",
        )
        for r in rows
    ]
    effects = []
    for hypothesis, track, metrics, scalar in [
        ("B-H1", "controlled", ("absent_false_accept", "present_coverage"), frozen["strongest_scalar"]["B-H1"]),
        ("B-H2", "natural", ("aurc", "top1_accuracy"), frozen["strongest_scalar"]["B-H2"]),
    ]:
        for comparator in ("pair_only", "strongest_scalar"):
            for metric in metrics:
                if comparator == "strongest_scalar" and scalar is None:
                    effect = dict(status="unavailable", reason="development strongest scalar infeasible", metric=metric)
                else:
                    effect = _effect(
                        context,
                        evaluated,
                        "A3",
                        "A1",
                        track=track,
                        metric=metric,
                        scalar=scalar if comparator == "strongest_scalar" else None,
                        samples=bootstrap_samples,
                        confidence=0.975,
                        references=references,
                    )
                effect.update(hypothesis=hypothesis, comparator=comparator)
                effects.append(effect)
    for second in ("A2", "A4"):
        for track, metric in [
            ("controlled", "absent_false_accept"),
            ("controlled", "present_coverage"),
            ("natural", "aurc"),
        ]:
            effects.append(
                _effect(
                    context,
                    evaluated,
                    "A3",
                    second,
                    track=track,
                    metric=metric,
                    samples=bootstrap_samples,
                    references=references,
                )
            )
    expansion_effects = _expansion_effects(context, evaluated, references, bootstrap_samples)
    gates = {}
    if "qualification" in bundle:
        evidence = json.loads(_read_hashed(bundle["qualification"]).read_text())
        if (
            evidence.get("relevance_space") != "benchmark_annotation"
            or evidence.get("target_gallery_manifest_sha256") != frozen["target_gallery"]["manifest_sha256"]
        ):
            raise ValueError("qualification gate evidence identity mismatch")
        gates = evidence.get("gates", {})
    if any(e["summary"]["invariance"].get("passed") is False for e in evaluated):
        raise ValueError("software invariance violation is a stop condition, not a method FAIL")
    hypotheses, verdict = _hypotheses(effects, gates)
    if bootstrap_samples != 10000:
        verdict = "inconclusive"  # reduced replicates are fixture-only, never registered confirmation
    report = dict(
        schema_version=1,
        split="val_locked",
        relevance_space="benchmark_annotation",
        label_protocol=bundle["label_protocol"],
        source_sha=context.manifest["git_sha"],
        analysis_sha256=file_sha256(__file__),
        frozen_reference=bundle["frozen"],
        strongest_scalar=frozen["strongest_scalar"],
        groups=[e["summary"] for e in evaluated],
        effects=effects,
        expansion_effects=expansion_effects,
        hypotheses=hypotheses,
        verdict=verdict,
        qualification_gates=gates,
        test_authorized=False,
        array_references=references,
        evaluation_gpu_hours=bundle.get("evaluation_gpu_hours"),
        training_gpu_hours=bundle.get("training_gpu_hours"),
        cost_scope="per-panel matcher/presence latency; offline full-gallery bank separately counted in evaluation GPU hours",
        strata={_key(e["row"]): _strata(e["row"], e["answers"]) for e in evaluated},
        cases=_cases(bundle, evaluated),
        limitations=[
            "Official benchmark annotations do not certify semantic no-answer or unannotated-video irrelevance.",
            "Single-positive mAP equals MRR; multi-answer strata are N/A when all canonical positive counts equal one.",
            "Missing measured latency/GPU hours/training/invariance probes remain unavailable.",
        ],
    )
    report["plots"] = _plots(context, bundle, evaluated, effects, report)
    compact = dict(report)
    compact["groups"] = []
    compact["strata"] = {}
    compact["array_reference_documents"] = []
    compact.pop("array_references")
    for offset in range(0, len(references), 400):
        array_path = context.paths.provenance / f"array_references_{offset // 400:03d}.json"
        _write_json(array_path, dict(array_references=references[offset : offset + 400]))
        compact["array_reference_documents"].append(
            dict(path=array_path.relative_to(context.paths.root).as_posix(), sha256=file_sha256(array_path))
        )
    for e in evaluated:
        group_path = context.paths.provenance / (_key(e["row"]) + "_summary.json")
        _write_json(group_path, dict(e["summary"], strata=report["strata"][_key(e["row"])]))
        compact["groups"].append(
            dict(
                arm=e["row"]["group"]["arm"],
                seed=e["row"]["group"]["seed"],
                track=e["row"]["group"]["track"],
                mechanism=e["row"]["group"]["mechanism"],
                K=e["row"]["group"]["K"],
                reference=dict(
                    path=group_path.relative_to(context.paths.root).as_posix(), sha256=file_sha256(group_path)
                ),
            )
        )
    _write_json(context.paths.provenance / "pilot_analysis.json", compact)
    markdown = [
        "# Benchmark pilot validation analysis",
        f"Verdict: **{verdict}**. Test remains sealed.",
        f"Label scope: benchmark_annotation. Frozen rule: `{bundle['frozen']['sha256']}`.",
    ]
    markdown += [f"![{name}](../{ref['path']})" for name, ref in report["plots"].items()]
    markdown += [
        "Full denominators, gallery/model/protocol identities and component CIs: [pilot_analysis.json](pilot_analysis.json)."
    ]
    with (context.paths.provenance / "pilot_analysis.md").open("x") as stream:
        stream.write("\n\n".join(markdown) + "\n")
    return report


def export_model_scores(
    context,
    model,
    query_inputs,
    video,
    frame_mask,
    *,
    query_ids,
    component_ids,
    canonical_relevance,
    official_relevance,
    gallery,
    panels,
    arm,
    seed,
    split,
    split_sha256,
    label_protocol,
    preprocessing_sha256,
    policy_sha256,
    pair_batch_size=64,
):
    """Actual model → score fragment consumer used by the reserved inference worker.

    E0 supplies immutable fixed-CLIP panel masks. Full-gallery matching scores
    provide official un-abstained ranking; each finite panel is scored separately
    to measure its service matching/presence latency and unique-pair cost.
    """
    import time

    import torch

    from semantic_retrieval.runtime.checkpoint import model_sha256
    from semantic_retrieval.types import EncodedFeatures, PairScores

    if split not in ("val_development", "val_calibration", "val_locked") or arm not in (
        "A0",
        "P1",
        "A1",
        "A2",
        "A3",
        "A4",
    ):
        raise ValueError("benchmark inference requires explicit validation split/arm")
    if len(set(query_ids)) != len(query_ids) or len(component_ids) != len(query_ids) or pair_batch_size <= 0:
        raise ValueError("unique source queries and positive inference batch size required")
    device = next(model.parameters()).device
    ids = {str(q): i for i, q in enumerate(query_ids)}
    n, m = len(query_ids), len(gallery["canonical_ids"])
    if np.asarray(canonical_relevance).shape != (n, m):
        raise ValueError("source benchmark truth must match full gallery")
    parsed = []
    for p in panels:
        qids = np.asarray(p["query_ids"]).astype(str)
        positions = np.array([ids[q] for q in qids])
        mask = np.asarray(p["candidate_mask"], bool)
        if mask.shape != (len(qids), m) or not mask.any(axis=1).all() or np.any(mask.sum(axis=1) > p["K"]):
            raise ValueError("panel query/candidate budget identity mismatch")
        parsed.append(
            dict(p, query_ids=qids, positions=positions, candidate_mask=mask, presence_logits=[], latency_ms=[])
        )
    model.eval().requires_grad_(False)
    started = time.perf_counter()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        q = model.encode_query(query_inputs)
        stream = (
            ((video[j : j + 8], frame_mask[j : j + 8]) for j in range(0, m, 8)) if torch.is_tensor(video) else video
        )
        cached = []
        for chunk_video, chunk_mask in stream:
            encoded = model.encode_gallery(chunk_video.to(device), chunk_mask.to(device))
            cached.append((encoded.frame_tokens.cpu(), encoded.pooled_video.cpu(), encoded.frame_mask.cpu()))
        vt = torch.cat([v[0] for v in cached])
        vp = torch.cat([v[1] for v in cached])
        vm = torch.cat([v[2] for v in cached])
        if len(vt) != m:
            raise ValueError("streamed gallery size differs from manifest")

        def score_candidates(qi, candidates):
            selected = torch.as_tensor(candidates, dtype=torch.long)
            features = EncodedFeatures(
                q.text_tokens,
                q.pooled_text,
                vt[selected].to(device),
                vp[selected].to(device),
                q.text_mask,
                vm[selected].to(device),
            )
            indices = torch.tensor([[qi, j] for j in range(len(selected))], device=device, dtype=torch.long)
            return model.score_pairs(features, indices)

        scores = np.empty((n, m), float)
        full_presence = np.empty(n, float) if arm in ("A2", "A3", "A4") else None
        for qi in range(n):
            batches = []
            for start in range(0, m, pair_batch_size):
                batches.append(score_candidates(qi, np.arange(start, min(start + pair_batch_size, m))))
            full_logits = torch.cat([b.logits for b in batches])
            scores[qi] = full_logits.cpu().numpy()
            if full_presence is not None:
                all_pairs = PairScores(full_logits, torch.cat([b.features for b in batches]), None)
                full_presence[qi] = float(model.score_presence(all_pairs, torch.tensor([0, m], device=device)).cpu()[0])
            for p in parsed:
                for pi in np.flatnonzero(p["positions"] == qi):
                    candidates = np.flatnonzero(p["candidate_mask"][pi])
                    synchronize()
                    begin = time.perf_counter()
                    finite = score_candidates(qi, candidates)
                    presence = (
                        model.score_presence(finite, torch.tensor([0, len(candidates)], device=device))
                        if arm in ("A2", "A3", "A4")
                        else None
                    )
                    synchronize()
                    elapsed = 1000 * (time.perf_counter() - begin)
                    p["presence_logits"].append((int(pi), float(presence.cpu()[0]) if presence is not None else None))
                    p["latency_ms"].append((int(pi), elapsed))
        synchronize()
    groups = []
    canonical = np.asarray(canonical_relevance)
    official = np.asarray(official_relevance)
    for p in parsed:
        positions = p["positions"]
        qids = p["query_ids"]
        tag = f"{arm}_seed{seed}_{split}_{p['track']}_K{p['K']}"
        latency = np.array([v for _, v in sorted(p["latency_ms"])])
        pres = np.array([v for _, v in sorted(p["presence_logits"])]) if arm in ("A2", "A3", "A4") else None
        observations = dict(
            latency_ms=latency,
            unique_pair_count=p["candidate_mask"].sum(axis=1),
            token_count=query_inputs.text_mask.sum(axis=1).cpu().numpy()[positions],
        )
        if full_presence is not None:
            observations["full_presence_logits"] = full_presence[positions]
        initial = p.get("initial_candidate_mask")
        ref = write_pilot_rows(
            context.paths.arrays / (tag + ".npz"),
            query_ids=qids,
            component_ids=np.asarray(component_ids)[positions],
            pair_logits=scores[positions],
            relevance=canonical[positions],
            candidate_mask=p["candidate_mask"],
            presence_logits=pres,
            official_relevance=official[positions],
            initial_candidate_mask=initial,
            **observations,
        )
        groups.append(
            dict(
                arm=arm,
                seed=seed,
                split=split,
                track=p["track"],
                mechanism=p.get("mechanism", "fixed"),
                K=p["K"],
                gallery=gallery["version"],
                model_sha256=model_sha256(model),
                split_sha256=split_sha256,
                arrays=ref,
            )
        )
    elapsed = time.perf_counter() - started
    fragment = dict(
        schema_version=1,
        relevance_space="benchmark_annotation",
        label_protocol=label_protocol,
        preprocessing_sha256=preprocessing_sha256,
        policy_sha256=policy_sha256,
        galleries={gallery["version"]: gallery},
        groups=groups,
        source_sha=context.manifest["git_sha"],
        inference_wall_seconds=elapsed,
        evaluation_gpu_hours=elapsed / 3600 if device.type == "cuda" else 0.0,
        latency_scope="finite-panel matcher and presence only; excludes offline full-gallery bank/encoding/coarse lookup",
    )
    _write_json(context.paths.provenance / "pilot_score_fragment.json", fragment)
    return fragment


def build_pilot_bundle(fragment_references, *, locked_gallery, frozen=None, qualification=None):
    """Assemble only hash-bound inference worker fragments, preserving every track/K identity."""
    fragments = [json.loads(_read_hashed(ref).read_text()) for ref in fragment_references]
    if not fragments:
        raise ValueError("score fragments required")
    first = fragments[0]
    identity = ("relevance_space", "label_protocol", "preprocessing_sha256", "policy_sha256")
    if any(any(f[k] != first[k] for k in identity) for f in fragments):
        raise ValueError("score fragments changed benchmark/preprocessing/policy identity")
    galleries = {"locked": locked_gallery}
    groups = []
    for f in fragments:
        for name, gallery in f["galleries"].items():
            if name in galleries and galleries[name] != gallery:
                raise ValueError("conflicting score gallery manifests")
            galleries[name] = gallery
        groups.extend(f["groups"])
    bundle = dict(
        schema_version=1,
        **{k: first[k] for k in identity},
        galleries=galleries,
        groups=groups,
        fragment_references=fragment_references,
        evaluation_gpu_hours=sum(f["evaluation_gpu_hours"] for f in fragments),
        inference_wall_seconds=sum(f["inference_wall_seconds"] for f in fragments),
    )
    if frozen is not None:
        bundle["frozen"] = frozen
    if qualification is not None:
        bundle["qualification"] = qualification
    _validate_bundle(bundle)
    return bundle


def apply_calibrated_operating_point(scores, candidates, point, state, *, presence_logits=None, scalar=None):
    """Preserve raw actions using the monotonic logit map, including sigmoid saturation."""

    def supported(logits, threshold, fitted):
        if fitted["status"] != "available":
            raise ValueError("calibrated actions require available state")
        if threshold in (0.0, 1.0):
            return expit(logits) >= threshold
        boundary = math.log(threshold / (1 - threshold)) / fitted["temperature"] + fitted["bias"]
        return np.asarray(logits) / fitted["temperature"] + fitted["bias"] >= boundary

    answer = np.asarray(candidates, bool) & supported(scores, point["pair_threshold"], state["pair"])
    if point["accept_kind"] == "presence":
        answer &= supported(presence_logits, point["accept_threshold"], state["presence"])[:, None]
    elif point["accept_kind"] == "scalar":
        answer &= (np.asarray(scalar) >= point["accept_threshold"])[:, None]
    return answer


def stream_pilot_gallery(path, *, gallery_version):
    """Read immutable full-gallery metadata and lazily decode bounded video batches."""
    from types import SimpleNamespace

    import torch

    from semantic_retrieval.data.episodes import EpisodePreprocessing
    from semantic_retrieval.data.video import decode_video, sample_frame_indices

    path = Path(path)
    raw = json.loads(path.read_text())
    if (
        set(raw) != {"schema_version", "gallery_version", "preprocessing", "videos"}
        or raw["schema_version"] != 1
        or raw["gallery_version"] != gallery_version
    ):
        raise ValueError("full gallery manifest schema/version mismatch")
    pre = EpisodePreprocessing(**raw["preprocessing"])
    rows = {}
    aliases = {}
    seen = set()
    for row in raw["videos"]:
        if (
            set(row) != {"canonical_id", "alias_id", "video_path", "content_sha256", "frame_count"}
            or row["alias_id"] in seen
            or type(row["frame_count"]) is not int
            or row["frame_count"] <= 0
        ):
            raise ValueError("invalid full gallery row")
        file = Path(row["video_path"])
        file = (file if file.is_absolute() else path.parent / file).resolve()
        if file_sha256(file) != row["content_sha256"]:
            raise ValueError("gallery video SHA mismatch")
        canonical = row["canonical_id"]
        value = (file, row["content_sha256"], row["frame_count"])
        if canonical in rows and rows[canonical][1:] != value[1:]:
            raise ValueError("canonical gallery conflict")
        rows.setdefault(canonical, value)
        aliases.setdefault(canonical, []).append(row["alias_id"])
        seen.add(row["alias_id"])
    ids = tuple(sorted(rows))
    if not ids:
        raise ValueError("empty full gallery")

    def batches():
        for start in range(0, len(ids), 8):
            subset = ids[start : start + 8]
            video = torch.zeros(len(subset), pre.max_frames, 3, pre.image_size, pre.image_size)
            mask = torch.zeros(len(subset), pre.max_frames, dtype=torch.bool)
            for j, canonical in enumerate(subset):
                file, digest, count = rows[canonical]
                if file_sha256(file) != digest:
                    raise ValueError("gallery input changed before decoding")
                frames, valid = decode_video(
                    file, frame_indices=sample_frame_indices(count, pre.max_frames), image_size=pre.image_size
                )
                video[j, : len(frames)] = frames
                mask[j, : len(frames)] = valid
            yield video, mask

    return SimpleNamespace(
        canonical_ids=ids,
        aliases={c: tuple(sorted(v)) for c, v in aliases.items()},
        preprocessing=pre,
        content_sha256={c: rows[c][1] for c in ids},
        frame_counts={c: rows[c][2] for c in ids},
        batches=batches,
    )


def _expansion_effects(context, evaluated, references, samples):
    """Compare actual expanding trajectories with initial and static same-budget outputs."""
    effects = []
    for arm in ("A1", "A2", "A3", "A4"):
        for track in ("natural", "controlled"):
            for static_k in (32, 64, 128):
                rows = []
                for seed in (0, 1, 2):
                    matching = [
                        e
                        for e in evaluated
                        if e["row"]["group"]["arm"] == arm
                        and e["row"]["group"]["seed"] == seed
                        and e["row"]["group"]["track"] == track
                    ]
                    systems = [
                        e
                        for e in matching
                        if e["row"]["group"]["mechanism"] == "system" and e["row"]["group"]["K"] == 128
                    ]
                    statics = [
                        e
                        for e in matching
                        if e["row"]["group"]["mechanism"] == "fixed" and e["row"]["group"]["K"] == static_k
                    ]
                    initial = [
                        e
                        for e in matching
                        if e["row"]["group"]["mechanism"] == "fixed" and e["row"]["group"]["K"] == 32
                    ]
                    if not systems or not statics or not initial:
                        continue
                    first, second, base = systems[0], statics[0], initial[0]
                    for e in (second, base):
                        for field in ("query_ids", "component_ids", "relevance"):
                            if not np.array_equal(first["row"][field], e["row"][field]):
                                raise ValueError("expansion comparisons changed source-query identity")
                    miss = ~(base["row"]["candidate_mask"].astype(bool) & base["row"]["relevance"].astype(bool)).any(
                        axis=1
                    )
                    if not miss.any():
                        continue
                    rows.append(
                        dict(
                            query_ids=first["row"]["query_ids"][miss],
                            component_ids=first["row"]["component_ids"][miss],
                            first=first["answers"][miss] if first["answers"] is not None else None,
                            second=second["answers"][miss] if second["answers"] is not None else None,
                            truth=first["row"]["relevance"][miss].astype(bool),
                        )
                    )
                if len(rows) != 3 or any(r["first"] is None or r["second"] is None for r in rows):
                    effects.append(
                        dict(
                            status="unavailable",
                            arm=arm,
                            track=track,
                            static_k=static_k,
                            reason="requires three-seed actual expansion and paired static outputs",
                        )
                    )
                    continue

                def delta(r, index):
                    first = (r["first"][index] & r["truth"][index]).any(axis=1).mean()
                    second = (r["second"][index] & r["truth"][index]).any(axis=1).mean()
                    return float(first - second)

                result, values = cluster_bootstrap(rows, delta, samples=samples)
                result.update(
                    arm=arm,
                    track=track,
                    static_k=static_k,
                    metric="initial-miss correct-return recovery delta",
                    samples_reference=_store(context, f"expansion_{arm}_{track}_staticK{static_k}", values, references),
                )
                if track == "natural" and result["component_count"] < 30:
                    result.update(status="descriptive", reason="fewer than 30 natural miss source components")
                effects.append(result)
    return effects
