#!/usr/bin/env python
"""Conditional conformal for ambiguity-heterogeneous text-video retrieval.

The gate experiment (scripts/conformal_coverage_probe.py) established the
problem: one global conformal threshold covers 83% of unambiguous queries and
98% of highly ambiguous ones, a 15-point spread, while an oracle that knows
|Rel| closes it to under 1.3. The obvious fix -- predict |Rel| from free
features, bucket on the prediction, one threshold per bucket -- reaches 13.0.

That attempt also produced a dissociation worth taking seriously: adding the
model's own log-variance improved the |Rel| prediction (Spearman 0.487 ->
0.553) while making coverage *less* uniform (+0.70 +- 0.12 pt). Predicting how
many videos are relevant is therefore not the same objective as making coverage
uniform, and optimising the former can hurt the latter.

This script drops the |Rel| detour. What a per-query threshold should be is the
conditional quantile of the conformal score,

    lambda(x)  s.t.  P(margin <= lambda(x) | X = x) = 1 - alpha,

which conformalized quantile regression (Romano, Patterson & Candes, NeurIPS
2019) estimates directly: fit a quantile regression on part of the calibration
half, then correct its residuals conformally on the rest. The correction keeps
the finite-sample marginal guarantee no matter how bad the regression is, so
the only thing at risk is set size, never validity. No |Rel| label is used.

The second change is the feature set. gap12 and caption length describe the top
of the ranking; |Rel| is a property of the query *relative to the whole
gallery*, so the features here read the whole score row -- where it falls off,
how heavy its head is, how many videos sit within a fraction of a standard
deviation of the best one.

Every method below is evaluated on identical splits so spreads can be compared
paired, which at these effect sizes is the only comparison that means anything.

Usage:
    python scripts/conditional_conformal_probe.py \
        --fire .scratch/mm-retrieval-evaluation/data/fire_msrvtt_dataset.json \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --run parity_a0=.scratch/simdump/parity_a0/sim_test_pytorch_model.bin.2.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_coverage_probe import (  # noqa: E402
    STRATA, STRATA_NAMES, build_scores, conformal_quantile, stratum_of,
)
from fire_corrected_metrics import load_fire, load_grid  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def gallery_features(sim, qlen, sigma=None):
    """Per-query descriptors of the whole score row, all label-free.

    The four scalars used so far (top1, gap12, caption length, sigma^2) all
    describe either the query alone or the top two entries. How many videos are
    relevant is a statement about the query against this particular gallery, so
    what the row looks like below rank 2 -- where it drops off, how many
    near-ties sit behind the winner -- is the part that was never measured.

    Everything is either an absolute score or a gap; the caller standardises,
    so no attempt is made to keep the columns commensurate here.
    """
    srt = np.sort(sim, axis=1)[:, ::-1]
    top1 = srt[:, 0]
    mu, sd = sim.mean(axis=1), sim.std(axis=1)
    cols, names = [], []

    def add(nm, v):
        names.append(nm)
        cols.append(np.asarray(v, dtype=np.float64))

    add("top1", top1)
    add("row mean", mu)
    add("row std", sd)
    # How far the winner sticks out of its own gallery, in gallery units. This
    # is the scale-free version of top1 and the one that survives a run whose
    # scores are globally shifted.
    add("top1 z", (top1 - mu) / sd)
    for k in (1, 2, 4, 9, 49, 99, 499):
        add(f"gap 1-{k + 1}", top1 - srt[:, k])
    # Near-tie counts at two widths. A query with fifty videos within a quarter
    # of a standard deviation of the best one is ambiguous in a way no single
    # gap can express.
    for frac in (0.25, 0.5, 1.0):
        add(f"n within {frac}sd", (sim >= (top1 - frac * sd)[:, None]).sum(axis=1))
    # Softmax entropy at two temperatures, again in gallery units so the
    # temperature means the same thing across runs. exp(entropy) is the
    # effective number of candidates the score row is spreading mass over.
    for t in (0.25, 1.0):
        z = (sim - top1[:, None]) / (t * sd[:, None])
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        add(f"entropy T={t}", -(p * np.log(p + 1e-12)).sum(axis=1))
    add("caption words", qlen.astype(np.float64))
    if sigma is not None:
        add("text sigma^2", sigma[0])
        add("video sigma^2", sigma[1])
    return np.column_stack(cols), names


def set_sizes_var(sim, lam):
    """|{v : s_top1 - s_v <= lam_i}| with a different lam_i per query."""
    top1 = sim.max(axis=1, keepdims=True)
    return (top1 - sim <= np.asarray(lam)[:, None]).sum(axis=1)


def _standardise(fit, *rest):
    mu, sd = fit.mean(axis=0), fit.std(axis=0)
    sd[sd == 0] = 1.0
    return [(x - mu) / sd for x in (fit, *rest)]


def cqr(sc, X, cal, tst, alpha, kind):
    """Conformalized quantile regression: a per-query threshold.

    The calibration half is split again -- one part fits the quantile
    regression, the other conformalises its residuals. That second split is not
    optional: reusing the fitting queries to size the correction would make the
    residuals optimistically small and break the guarantee. It also means CQR
    sees half the calibration data the bucketed baseline does, which is a real
    handicap and is left in place rather than papered over.
    """
    half = len(cal) // 2
    fit, conf = cal[:half], cal[half:]
    Xf, Xc, Xt = _standardise(X[fit], X[conf], X[tst])
    y = sc["margin_any"][fit]

    if kind == "linear":
        # A small L1 penalty: 17 columns against 250 rows, several of them
        # near-collinear by construction (gap 1-2 and gap 1-3 differ by one
        # order statistic), so the unpenalised fit is unstable across splits.
        m = QuantileRegressor(quantile=1 - alpha, alpha=1e-3, solver="highs")
    else:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=1 - alpha, n_estimators=100,
            max_depth=2, learning_rate=0.1, random_state=0,
        )
    m.fit(Xf, y)

    # One-sided score: the set only fails by being too small, so only the upper
    # side of the interval exists and the correction is a plain quantile of the
    # signed residual rather than the two-sided max CQR uses for intervals.
    e = sc["margin_any"][conf] - m.predict(Xc)
    return m.predict(Xt) + conformal_quantile(e, alpha)


def bucketed(sc, X, cal, tst, alpha, n_rel, lam_global):
    """The current best deployable method, reproduced on the same splits.

    Least squares from the named columns onto |Rel| on the calibration half,
    quintile buckets of the prediction, one conformal threshold per bucket.
    """
    Xc, Xa = _standardise(X[cal], X)
    A = np.column_stack([Xc, np.ones(len(cal))])
    w = np.linalg.lstsq(A, n_rel[cal].astype(float), rcond=None)[0]
    amb = -(np.column_stack([Xa, np.ones(len(X))]) @ w)

    edges = np.quantile(amb[cal], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(amb, edges)
    lam = np.full(len(tst), lam_global)
    for k in range(5):
        mc, mt = cal[b[cal] == k], np.flatnonzero(b[tst] == k)
        if len(mc) >= 10 and len(mt):
            lam[mt] = conformal_quantile(sc["margin_any"][mc], alpha)
    return lam


def oracle(sc, cal, tst, alpha, strat, lam_global):
    """One threshold per *true* |Rel| stratum -- the ceiling, not a method."""
    lam = np.full(len(tst), lam_global)
    for s in range(len(STRATA)):
        mc, mt = cal[strat[cal] == s], np.flatnonzero(strat[tst] == s)
        if len(mc) >= 10 and len(mt):
            lam[mt] = conformal_quantile(sc["margin_any"][mc], alpha)
    return lam


def evaluate(sc, X, names, n_rel, alpha, n_rep, seed):
    """Run every method on identical splits and accumulate per-split results."""
    n = len(n_rel)
    strat = np.array([stratum_of(r) for r in n_rel])
    half = n // 2
    gl = [names.index(c) for c in ("gap 1-2", "caption words")]
    rich = list(range(X.shape[1]))

    # A 2x2 over what changed: the objective (bucket the predicted |Rel| vs
    # regress the conformal quantile directly) and the features (the two
    # scalars used so far vs the whole score row). Without both off-diagonal
    # cells a win cannot be attributed to either.
    methods = ["global", "bucket gap+len", "bucket rich",
               "CQR gap+len", "CQR linear", "CQR gbr", "oracle |Rel|"]
    acc = {m: {"cov": [[] for _ in STRATA], "spread": [], "size": [], "med": [],
               "p90": [], "marg": []}
           for m in methods}
    rng = np.random.RandomState(seed)

    for _ in range(n_rep):
        perm = rng.permutation(n)
        cal, tst = perm[:half], perm[half:]
        lam_g = conformal_quantile(sc["margin_any"][cal], alpha)

        lams = {
            "global": np.full(len(tst), lam_g),
            "bucket gap+len": bucketed(sc, X[:, gl], cal, tst, alpha, n_rel, lam_g),
            "bucket rich": bucketed(sc, X[:, rich], cal, tst, alpha, n_rel, lam_g),
            "CQR gap+len": cqr(sc, X[:, gl], cal, tst, alpha, "linear"),
            "CQR linear": cqr(sc, X, cal, tst, alpha, "linear"),
            "CQR gbr": cqr(sc, X, cal, tst, alpha, "gbr"),
            "oracle |Rel|": oracle(sc, cal, tst, alpha, strat, lam_g),
        }
        for m, lam in lams.items():
            cov = sc["margin_any"][tst] <= lam
            a = acc[m]
            a["marg"].append(cov.mean())
            sz = set_sizes_var(sc["sim"][tst], lam)
            a["size"].append(sz.mean())
            # The mean is dragged by a handful of queries whose threshold
            # admits hundreds of videos; the median says what a typical user
            # would actually be handed.
            a["med"].append(np.median(sz))
            a["p90"].append(np.percentile(sz, 90))
            per_s = []
            for s in range(len(STRATA)):
                sel = strat[tst] == s
                # NaN rather than a short column: the bootstrap below resamples
                # whole splits, so every method needs the same rectangular
                # n_rep x 5 table with the same holes in it.
                a["cov"][s].append(cov[sel].mean() if sel.any() else np.nan)
                if sel.any():
                    per_s.append(cov[sel].mean())
            # Spread within the split, so two methods differ on the same
            # queries and the difference can carry a paired standard error.
            a["spread"].append(max(per_s) - min(per_s))
    return methods, acc


def boot_spreads(acc, methods, seed, n_boot=2000):
    """Paired bootstrap over splits for the spread of the averaged columns.

    Two different numbers have been called "spread" here and they are not
    interchangeable. Averaging the five per-stratum coverages over splits first
    and subtracting afterwards gives the quantity the earlier probe reports
    (13.8, oracle 0.7). Taking max-minus-min inside each split and averaging
    that gives something much larger, because a max of five noisy estimates is
    biased upward -- the oracle's true 0.7 shows up as roughly 9. Worse, that
    bias is a floor both methods sit on, so it compresses differences: a real
    13.1-point gap between baseline and oracle reads as 4.1.

    So the within-split version is the wrong effect-size scale, but it was the
    only one with an obvious standard error. Resampling the 200 splits with
    replacement -- the same resampled indices for every method, which keeps the
    comparison paired -- puts an error bar on the right scale instead.
    """
    rng = np.random.RandomState(seed + 7)
    n_rep = len(acc[methods[0]]["spread"])
    idx = rng.randint(0, n_rep, (n_boot, n_rep))
    out = {}
    for m in methods:
        C = np.array(acc[m]["cov"]).T           # (n_rep, 5)
        means = np.nanmean(C[idx], axis=1)      # (n_boot, 5)
        out[m] = (means.max(axis=1) - means.min(axis=1)) * 100
    return out


def report(methods, acc, seed, base="bucket gap+len"):
    print("      method          " + "".join(f"{nm:>8s}" for nm in STRATA_NAMES)
          + "     spread   size   med   p90   marg    d(spread) vs " + base)
    bs = boot_spreads(acc, methods, seed)
    for m in methods:
        a = acc[m]
        v = np.array([np.nanmean(c) for c in a["cov"]]) * 100
        d = bs[m] - bs[base]
        delta = "          -" if m == base else f"  {d.mean():+6.2f} +- {d.std():.2f}"
        print(f"      {m:<15s} " + "".join(f"{x:7.1f}%" for x in v)
              + f"  {v.max() - v.min():6.1f} pt {np.mean(a['size']):6.1f}"
              + f" {np.mean(a['med']):5.1f} {np.mean(a['p90']):5.1f}"
              + f"  {np.mean(a['marg']) * 100:5.1f}%" + delta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fire", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--run", action="append", required=True, help="name=path.npz")
    p.add_argument("--unc", action="append", default=[],
                   help="name=path.npz from scripts/dump_rspr_uncertainty.sh")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    unc = {}
    for spec in a.unc:
        nm, path = spec.split("=", 1)
        z = np.load(path)
        unc[nm] = (z["text_sigma2"], z["video_sigma2"])

    queries, videos = load_grid(a.test_csv)
    relevant, judged, outside, gt_bad = load_fire(a.fire, queries, videos)
    n_rel = np.array([len(r) for r in relevant])
    qlen = np.array([len(q.split()) for q in queries])

    print(f"grid {len(queries)}x{len(videos)} | mean positives {n_rel.mean():.2f} "
          f"| alpha={a.alpha} reps={a.reps}\n")

    for spec in a.run:
        name, path = spec.split("=", 1)
        sim = np.load(path)["t2v"].astype(np.float64)
        assert sim.shape == (len(queries), len(videos)), f"{name}: {sim.shape}"
        sc = build_scores(sim, relevant)
        X, names = gallery_features(sim, qlen, unc.get(name))

        print(f"=== {name} ===  {X.shape[1]} features"
              + (" (incl. sigma^2)" if name in unc else ""))
        methods, acc = evaluate(sc, X, names, n_rel, a.alpha, a.reps, a.seed)
        report(methods, acc, a.seed)
        print()


if __name__ == "__main__":
    main()
