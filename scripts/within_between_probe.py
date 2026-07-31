#!/usr/bin/env python
"""Is the within-stratum effect opposite in sign to the between-stratum one?

The CQR probe left a loose end that decides the whole module design. Free
features correlate 0.29-0.36 with the conformal score over all 1000 queries but
0.44-0.48 inside the 363 queries with |Rel| = 1. A within-group correlation that
*exceeds* the pooled one is the signature of suppression: two effects pushing
the same feature in opposite directions and cancelling when the groups are
merged.

The mechanism that would produce it is specific. Between strata, high entropy
means many near-tied videos, which means |Rel| is large, which means the oracle
threshold is *small* (0.94 at |Rel| >= 6). Within a stratum |Rel| is held fixed,
so high entropy can only mean this particular query is hard, the relevant video
sits further down, and the threshold has to be *large*. Same feature, opposite
required response.

If that holds, no monotone lambda(x) can serve both and the 2.5 points CQR
bought are a ceiling on that model form, not on the features. It is also the
only justification for fitting the difficulty head separately inside each group
rather than pooling everything into one regression. So it is worth testing
directly instead of inferring it from two correlations.

Part A measures it at the quantile that actually sets the threshold. For each
feature: a tau = 1 - alpha quantile-regression slope inside each stratum, the
size-weighted average of those (beta_within), the slope of the per-stratum
oracle thresholds against the per-stratum mean feature (beta_between), and the
pooled slope that ignores strata (beta_pool).

Part B asks the other half of the question: with |Rel| pinned, is there enough
within-stratum heterogeneity left to be worth a second head? A single threshold
is applied inside |Rel| = 1 and coverage is read off by out-of-sample difficulty
quintile. Cells of ~36 queries are noisy enough that a spread has to be compared
against random quintiles, which is what the `random` row is for.

Usage:
    python scripts/within_between_probe.py \
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conditional_conformal_probe import gallery_features, set_sizes_var  # noqa: E402
from conformal_coverage_probe import (  # noqa: E402
    STRATA, build_scores, conformal_quantile, stratum_of,
)
from fire_corrected_metrics import load_fire, load_grid  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# The three features that carried the within-|Rel|=1 signal in the CQR probe
# (Spearman +0.476 / -0.476 / -0.462 there against 0.29-0.36 pooled). Naming
# them here rather than sweeping all 17 keeps this a test of a stated claim.
FEATURES = ["entropy T=0.25", "gap 1-100", "top1 z"]
QUINTILE_NAMES = ["easy", "2", "3", "4", "hard"]


def qslope(x, y, tau):
    """Slope of the tau-quantile of y on a single standardised feature x."""
    m = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    m.fit(x.reshape(-1, 1), y)
    return float(m.coef_[0])


def part_a_once(x, y, strat, alpha):
    """beta_within / beta_between / beta_pool for one feature on one sample.

    beta_within is the size-weighted mean of the per-stratum slopes rather than
    a pooled fit with stratum dummies. Both estimate the same thing when the
    slope is common; the weighted mean also survives the slope differing by
    stratum, which is exactly what is in question.

    beta_between regresses the five *oracle thresholds* on the five mean feature
    values. Using the thresholds rather than mean scores keeps both sides on the
    scale lambda actually lives on, so the two betas are comparable.
    """
    tau = 1.0 - alpha
    betas, ns, lam_s, xbar_s = [], [], [], []
    for s in range(len(STRATA)):
        m = strat == s
        if m.sum() < 30:
            continue
        betas.append(qslope(x[m], y[m], tau))
        ns.append(m.sum())
        lam_s.append(conformal_quantile(y[m], alpha))
        xbar_s.append(x[m].mean())

    w = np.asarray(ns, dtype=np.float64)
    beta_within = float(np.average(betas, weights=w))

    # Five points, weighted by stratum size. A slope, not a correlation: the
    # question is sign and magnitude on lambda's scale.
    xb, lb = np.asarray(xbar_s), np.asarray(lam_s)
    xc = xb - np.average(xb, weights=w)
    beta_between = float(np.sum(w * xc * lb) / np.sum(w * xc * xc))

    return beta_within, beta_between, qslope(x, y, tau), np.asarray(betas)


def part_a(sc, X, names, strat, alpha, n_boot, seed):
    y = sc["margin_any"]
    rng = np.random.RandomState(seed)
    n = len(y)
    out = {}
    for f in FEATURES:
        x = X[:, names.index(f)]
        x = (x - x.mean()) / x.std()
        point = part_a_once(x, y, strat, alpha)

        # Bootstrap over queries, not splits: these are population quantities
        # of the 1000-query grid, with no calibration/test split involved.
        bw, bb, bp = [], [], []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            try:
                a, b, c, _ = part_a_once(x[idx], y[idx], strat[idx], alpha)
            except (ValueError, ZeroDivisionError):
                continue
            bw.append(a); bb.append(b); bp.append(c)
        out[f] = (point, np.array(bw), np.array(bb), np.array(bp))
    return out


def part_b(sc, X, strat, alpha, n_rep, seed, target=0):
    """Inside one stratum: does a single threshold cover unevenly by difficulty?

    Difficulty is a leave-out prediction -- least squares of the conformal score
    on the standardised features fitted on the calibration half only -- so the
    quintiles are not fitted on the queries they are scored on. The `random` row
    replaces that prediction with noise and is the noise floor for a spread read
    off ~36-query cells.
    """
    idx = np.flatnonzero(strat == target)
    y_all, sim_all = sc["margin_any"], sc["sim"]
    half = len(idx) // 2
    rng = np.random.RandomState(seed)

    rows = ["single", "random", "CQR within"]
    acc = {r: {"cov": [[] for _ in QUINTILE_NAMES], "size": [], "marg": []}
           for r in rows}

    for _ in range(n_rep):
        perm = rng.permutation(idx)
        cal, tst = perm[:half], perm[half:]

        mu, sd = X[cal].mean(axis=0), X[cal].std(axis=0)
        sd[sd == 0] = 1.0
        Xc, Xt = (X[cal] - mu) / sd, (X[tst] - mu) / sd
        A = np.column_stack([Xc, np.ones(len(cal))])
        w = np.linalg.lstsq(A, y_all[cal], rcond=None)[0]
        pred_c = A @ w
        pred_t = np.column_stack([Xt, np.ones(len(tst))]) @ w

        edges = np.quantile(pred_c, [0.2, 0.4, 0.6, 0.8])
        q = {"single": np.digitize(pred_t, edges),
             "random": rng.randint(0, 5, len(tst))}
        q["CQR within"] = q["single"]

        lam_s = conformal_quantile(y_all[cal], alpha)
        # CQR inside the stratum: fit on one calibration quarter, conformalise
        # the residuals on the other. 90 rows against 17 columns is thin, and
        # that thinness is part of what is being measured.
        h = len(cal) // 2
        fit, conf = cal[:h], cal[h:]
        Xf = (X[fit] - mu) / sd
        m = QuantileRegressor(quantile=1 - alpha, alpha=1e-3, solver="highs")
        m.fit(Xf, y_all[fit])
        e = y_all[conf] - m.predict((X[conf] - mu) / sd)
        lam_cqr = m.predict(Xt) + conformal_quantile(e, alpha)

        lams = {"single": np.full(len(tst), lam_s),
                "random": np.full(len(tst), lam_s),
                "CQR within": lam_cqr}

        for r in rows:
            cov = y_all[tst] <= lams[r]
            a = acc[r]
            a["marg"].append(cov.mean())
            a["size"].append(set_sizes_var(sim_all[tst], lams[r]).mean())
            for k in range(5):
                sel = q[r] == k
                a["cov"][k].append(cov[sel].mean() if sel.any() else np.nan)
    return rows, acc, len(idx)


def boot_spreads(acc, rows, seed, n_boot=2000):
    """Paired bootstrap over splits, on the averaged-column spread.

    Same convention as conditional_conformal_probe.boot_spreads: average the
    five per-cell coverages over splits first, then subtract. Taking max-min
    inside a split is biased upward by the max of five noisy estimates, which
    at 36 queries per cell would swamp the effect being measured.
    """
    rng = np.random.RandomState(seed + 11)
    n_rep = len(acc[rows[0]]["marg"])
    idx = rng.randint(0, n_rep, (n_boot, n_rep))
    out = {}
    for r in rows:
        C = np.array(acc[r]["cov"]).T
        means = np.nanmean(C[idx], axis=1)
        out[r] = (means.max(axis=1) - means.min(axis=1)) * 100
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fire", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--run", action="append", required=True, help="name=path.npz")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    queries, videos = load_grid(a.test_csv)
    relevant, judged, outside, gt_bad = load_fire(a.fire, queries, videos)
    n_rel = np.array([len(r) for r in relevant])
    qlen = np.array([len(q.split()) for q in queries])
    strat = np.array([stratum_of(r) for r in n_rel])

    print(f"grid {len(queries)}x{len(videos)} | mean positives {n_rel.mean():.2f} "
          f"| alpha={a.alpha} | part A boot={a.boot} | part B reps={a.reps}")
    print("stratum sizes: " + ", ".join(
        f"{nm}={int((strat == s).sum())}"
        for s, nm in enumerate(["1", "2", "3", "4-5", "6+"])) + "\n")

    for spec in a.run:
        name, path = spec.split("=", 1)
        sim = np.load(path)["t2v"].astype(np.float64)
        sc = build_scores(sim, relevant)
        X, names = gallery_features(sim, qlen)

        print(f"=== {name} ===")
        print("  [A] tau=0.9 slope of the conformal score on one feature")
        print(f"      {'feature':<16s}{'b_within':>10s}{'b_between':>11s}"
              f"{'b_pool':>9s}   {'flip':>5s} {'atten':>6s}   per-stratum betas")
        res = part_a(sc, X, names, strat, a.alpha, a.boot, a.seed)
        for f in FEATURES:
            (bw, bb, bp, per_s), sw, sb, sp = res[f]
            lo, hi = np.percentile(sw, [2.5, 97.5])
            flip = "yes" if bw * bb < 0 and lo * hi > 0 else "no"
            atten = "yes" if abs(bp) < 0.5 * abs(bw) else "no"
            print(f"      {f:<16s}{bw:+10.3f}{bb:+11.3f}{bp:+9.3f}   {flip:>5s} "
                  f"{atten:>6s}   " + " ".join(f"{b:+.2f}" for b in per_s))
            print(f"      {'':<16s}[{lo:+.3f},{hi:+.3f}]"
                  f"  [{np.percentile(sb, 2.5):+.2f},{np.percentile(sb, 97.5):+.2f}]"
                  f"  [{np.percentile(sp, 2.5):+.2f},{np.percentile(sp, 97.5):+.2f}]")

        rows, accb, n_t = part_b(sc, X, strat, a.alpha, a.reps, a.seed)
        bs = boot_spreads(accb, rows, a.seed)
        d = bs["single"] - bs["random"]
        print(f"\n  [B] inside |Rel|=1 ({n_t} queries), coverage by difficulty quintile")
        print("      " + f"{'threshold':<12s}"
              + "".join(f"{nm:>8s}" for nm in QUINTILE_NAMES)
              + f"{'spread':>9s}{'size':>7s}{'marg':>8s}")
        for r in rows:
            v = np.array([np.nanmean(c) for c in accb[r]["cov"]]) * 100
            print(f"      {r:<12s}" + "".join(f"{x:7.1f}%" for x in v)
                  + f"{v.max() - v.min():7.1f}pt"
                  + f"{np.mean(accb[r]['size']):7.1f}"
                  + f"{np.mean(accb[r]['marg']) * 100:7.1f}%")
        print(f"      single - random spread: {d.mean():+.2f} +- {d.std():.2f} pt")
        print()


if __name__ == "__main__":
    main()
