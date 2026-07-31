#!/usr/bin/env python
"""Conditional conformal over a function class, on both coverage axes.

step-002 established the structure this script is built against. Conditional
coverage fails on two orthogonal axes -- how many videos are relevant (15 pt)
and, with that pinned, whether this model is about to fail (13 pt) -- and the
same feature requires opposite-signed threshold responses on the two. Mondrian
bucketing addresses only the first; a pooled quantile regression addresses
neither, because it fits one slope to two effects that cancel.

Gibbs, Cherian & Candes (arXiv 2305.12616, number unverified -- no network
here) target exactly this. Given a finite-dimensional class F = span{phi_1 ..
phi_d}, fitting the pinball loss at tau = 1 - alpha,

    beta_hat = argmin_beta  sum_i rho_tau(S_i - phi(x_i)^T beta),

has as its first-order condition

    E[ phi(X) ( 1{S <= phi(X)^T beta} - (1 - alpha) ) ] = 0,

i.e. coverage holds along every direction in F, not just on average. Mondrian
is the special case where phi is a set of group indicators, and the global
threshold is the case phi = 1. So the whole family sits on one axis and the
design question becomes which F to pick.

Ours writes both axes into F:

    phi(x) = [ 1{b(x)=k} ]_{k=0..4}  +  [ 1{b(x)=k} * d(x) ]_{k=0..4}

with b the predicted-|Rel| quintile (axis one) and d a single within-bucket
centred difficulty scalar (axis two). The interaction is the point: step-002
showed the required d-slope differs in sign between strata, so a shared slope
cannot serve both. d is one scalar rather than seventeen features because
step-002 also showed a 17-column within-stratum fit inflates sets fourfold.

Two versions are run. The asymptotic one fits beta once on the calibration
half. The finite-sample one re-solves per test point with the test score
imputed at +infinity, which adds a linear term -tau * phi(x) to the LP and
nothing else, so A_eq is built once per split and only the objective moves.
That version is far slower and is run on fewer splits; the reduced count is
printed rather than hidden.

Usage:
    python scripts/gcc_conditional_probe.py \
        --fire .scratch/mm-retrieval-evaluation/data/fire_msrvtt_dataset.json \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --run parity_a0=.scratch/simdump/parity_a0/sim_test_pytorch_model.bin.2.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conditional_conformal_probe import gallery_features, set_sizes_var  # noqa: E402
from conformal_coverage_probe import (  # noqa: E402
    STRATA, STRATA_NAMES, build_scores, conformal_quantile, stratum_of,
)
from fire_corrected_metrics import load_fire, load_grid  # noqa: E402

warnings.filterwarnings("ignore")

N_BUCKET = 5
QUINT_NAMES = ["easy", "2", "3", "4", "hard"]


# --------------------------------------------------------------------------
# pinball LP
# --------------------------------------------------------------------------

def pinball_setup(Phi):
    """The equality block of the quantile-regression LP, reused across solves.

    Variables are [beta (free), u+ (>=0), u- (>=0)] and the constraint is
    Phi beta + u+ - u- = y, so at the optimum u+ and u- split the residual into
    its positive and negative parts and the objective tau*sum(u+) +
    (1-tau)*sum(u-) is the pinball loss. Imputing a test point at +infinity
    only adds a linear term in beta, so this block never changes within a split.
    """
    n, d = Phi.shape
    A = sparse.hstack(
        [sparse.csc_matrix(Phi), sparse.identity(n, format="csc"),
         -sparse.identity(n, format="csc")], format="csc")
    bounds = [(None, None)] * d + [(0, None)] * (2 * n)
    return A, bounds, n, d


def pinball_solve(A, bounds, n, d, y, tau, lin=None):
    c = np.concatenate([np.zeros(d) if lin is None else lin,
                        np.full(n, tau), np.full(n, 1.0 - tau)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if not r.success:
        return None
    return r.x[:d]


# --------------------------------------------------------------------------
# the two axes
# --------------------------------------------------------------------------

def vlm_counts(cache_dir, sim, topk):
    """Soft count sum_j P(yes | caption, video j) over the top-k retrieved.

    No fitting, so this column carries no leakage: it is a raw model output.
    Queries the judge never saw come back as nan.
    """
    cache = {}
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("pyes_") and f.endswith(".json"):
            cache.update(json.load(open(os.path.join(cache_dir, f))))
    order = np.argsort(-sim, axis=1)
    out = np.full(len(sim), np.nan)
    for i in range(len(sim)):
        ps = [cache.get(f"{i}:{j}") for j in order[i, :topk]]
        if all(p is not None for p in ps):
            out[i] = float(np.sum(ps))
    return out


def fit_axes(X, y, n_rel, fit_idx):
    """Learn b(.) and d(.) on one calibration quarter only.

    Both are least squares -- |Rel| for the bucketing, the conformal score for
    difficulty. Neither needs to be good: the quantile regression downstream
    guarantees coverage along whatever directions these define, and their
    quality only shows up in set size and in how well the predicted buckets
    track the true strata.
    """
    mu, sd = X[fit_idx].mean(axis=0), X[fit_idx].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.column_stack([Z, np.ones(len(Z))])

    w_amb = np.linalg.lstsq(A[fit_idx], n_rel[fit_idx].astype(float), rcond=None)[0]
    amb = -(A @ w_amb)                       # larger = fewer relevant videos
    edges = np.quantile(amb[fit_idx], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(amb, edges)

    w_dif = np.linalg.lstsq(A[fit_idx], y[fit_idx], rcond=None)[0]
    d = A @ w_dif
    d = (d - d[fit_idx].mean()) / (d[fit_idx].std() + 1e-12)
    # Centre within predicted bucket so the interaction columns carry the
    # within-bucket variation and the indicator columns carry the level. Without
    # this the two blocks are correlated and the fitted slopes are not the
    # within-bucket slopes step-002 measured.
    for k in range(N_BUCKET):
        m = b == k
        mf = m & np.isin(np.arange(len(b)), fit_idx)
        if mf.sum() >= 2:
            d[m] -= d[mf].mean()
    return b, d


def basis(b, d, kind):
    ind = np.column_stack([(b == k).astype(float) for k in range(N_BUCKET)])
    if kind == "buckets":
        return ind
    if kind == "diff":
        return np.column_stack([ind, d])
    return np.column_stack([ind, ind * d[:, None]])


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------

def mondrian(y, cal, tst, alpha, grp, lam_g, n_grp):
    lam = np.full(len(tst), lam_g)
    for k in range(n_grp):
        mc, mt = cal[grp[cal] == k], np.flatnonzero(grp[tst] == k)
        if len(mc) >= 10 and len(mt):
            lam[mt] = conformal_quantile(y[mc], alpha)
    return lam


def oracle_2axis(y, cal, tst, alpha, strat, quint, lam_g):
    """One threshold per (true stratum x difficulty quintile). The real ceiling.

    Twenty-five cells over a 500-query calibration half is about 20 points each,
    so each threshold is the 19th of 20 order statistics and is noisy. It is
    reported as an upper reference, not as something reachable.
    """
    lam = np.full(len(tst), lam_g)
    for s in range(len(STRATA)):
        for q in range(N_BUCKET):
            mc = cal[(strat[cal] == s) & (quint[cal] == q)]
            mt = np.flatnonzero((strat[tst] == s) & (quint[tst] == q))
            if len(mc) >= 10 and len(mt):
                lam[mt] = conformal_quantile(y[mc], alpha)
    return lam


def gcc(y, Phi, cal, tst, alpha, imputed):
    """Quantile regression over F, optionally with the +infinity imputation."""
    tau = 1.0 - alpha
    A, bnd, n, d = pinball_setup(Phi[cal])
    beta = pinball_solve(A, bnd, n, d, y[cal], tau)
    if beta is None:
        return None
    if not imputed:
        return Phi[tst] @ beta
    out = np.empty(len(tst))
    for j, i in enumerate(tst):
        bj = pinball_solve(A, bnd, n, d, y[cal], tau, lin=-tau * Phi[i])
        out[j] = Phi[i] @ (beta if bj is None else bj)
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(sc, X, n_rel, alpha, n_rep, seed, imputed_reps, Xv=None):
    """Xv is X with the external judge's count appended as one extra column.

    Everything about the two VLM arms is identical to `bucket rich` and
    `GCC +diff` except that column, so the delta between the pairs is what the
    judge buys and nothing else.
    """
    y, sim = sc["margin_any"], sc["sim"]
    n = len(n_rel)
    strat = np.array([stratum_of(r) for r in n_rel])
    half = n // 2

    methods = ["global", "bucket rich", "GCC buckets", "GCC +diff", "GCC +inter",
               "GCC +inter (imp)", "oracle |Rel|", "oracle 2-axis"]
    if Xv is not None:
        methods = methods[:2] + ["bucket VLM", "GCC +diff VLM"] + methods[2:]
    acc = {m: {"rel": [[] for _ in STRATA], "dif": [[] for _ in QUINT_NAMES],
               "size": [], "med": [], "p90": [], "marg": [], "n": 0}
           for m in methods}
    rng = np.random.RandomState(seed)

    for rep in range(n_rep):
        perm = rng.permutation(n)
        cal, tst = perm[:half], perm[half:]
        lam_g = conformal_quantile(y[cal], alpha)
        # Axis machinery is fitted on one calibration quarter and the
        # thresholds calibrated on the other, so the quantile regression never
        # sees the queries that defined its own basis. That costs GCC half the
        # calibration data relative to `bucket rich`, which is left in place.
        h = len(cal) // 2
        fit, cal2 = cal[:h], cal[h:]
        b, d = fit_axes(X, y, n_rel, fit)

        # Difficulty quintile inside the *true* stratum: the axis-two readout.
        # Edges come from the calibration half only.
        quint = np.zeros(n, dtype=int)
        for s in range(len(STRATA)):
            m = strat == s
            e = np.quantile(d[cal[strat[cal] == s]], [0.2, 0.4, 0.6, 0.8])
            quint[m] = np.digitize(d[m], e)

        P = {k: basis(b, d, k) for k in ("buckets", "diff", "inter")}
        lams = {
            "global": np.full(len(tst), lam_g),
            "bucket rich": mondrian(y, cal, tst, alpha, b, lam_g, N_BUCKET),
            "GCC buckets": gcc(y, P["buckets"], cal2, tst, alpha, False),
            "GCC +diff": gcc(y, P["diff"], cal2, tst, alpha, False),
            "GCC +inter": gcc(y, P["inter"], cal2, tst, alpha, False),
            "oracle |Rel|": mondrian(y, cal, tst, alpha, strat, lam_g, len(STRATA)),
            "oracle 2-axis": oracle_2axis(y, cal, tst, alpha, strat, quint, lam_g),
        }
        if Xv is not None:
            bv, dv = fit_axes(Xv, y, n_rel, fit)
            lams["bucket VLM"] = mondrian(y, cal, tst, alpha, bv, lam_g, N_BUCKET)
            lams["GCC +diff VLM"] = gcc(y, basis(bv, dv, "diff"), cal2, tst,
                                        alpha, False)
        if rep < imputed_reps:
            lams["GCC +inter (imp)"] = gcc(y, P["inter"], cal2, tst, alpha, True)

        for m, lam in lams.items():
            if lam is None:
                continue
            cov = y[tst] <= lam
            a = acc[m]
            a["n"] += 1
            a["marg"].append(cov.mean())
            sz = set_sizes_var(sim[tst], lam)
            a["size"].append(sz.mean())
            a["med"].append(np.median(sz))
            a["p90"].append(np.percentile(sz, 90))
            for s in range(len(STRATA)):
                sel = strat[tst] == s
                a["rel"][s].append(cov[sel].mean() if sel.any() else np.nan)
            for q in range(N_BUCKET):
                sel = quint[tst] == q
                a["dif"][q].append(cov[sel].mean() if sel.any() else np.nan)
    return methods, acc


def boot(acc, methods, key, seed, n_boot=2000):
    """Paired bootstrap over splits of the averaged-column spread.

    Methods run on different numbers of splits (the imputed one is capped)
    cannot be paired, so each is resampled over its own splits and only
    equal-length pairs get a paired delta.
    """
    rng = np.random.RandomState(seed + 23)
    out = {}
    for m in methods:
        C = np.array(acc[m][key]).T
        if not len(C):
            continue
        idx = rng.randint(0, len(C), (n_boot, len(C)))
        means = np.nanmean(C[idx], axis=1)
        out[m] = (means.max(axis=1) - means.min(axis=1)) * 100
    return out


def deltas(methods, acc, seed, base):
    br = boot(acc, methods, "rel", seed)
    bd = boot(acc, methods, "dif", seed)
    print(f"\n      paired vs `{base}`     d(sprd_rel)        d(sprd_dif)"
          f"          size ratio")
    for m in methods:
        if m == base or not acc[m]["n"]:
            continue
        sz = np.mean(acc[m]["size"]) / np.mean(acc[base]["size"])
        if len(br.get(m, [])) and acc[m]["n"] == acc[base]["n"]:
            dr, dd = br[m] - br[base], bd[m] - bd[base]
            print(f"      {m:<18s} {dr.mean():+7.2f} +- {dr.std():.2f}"
                  f"   {dd.mean():+7.2f} +- {dd.std():.2f}      {sz:5.2f}x")
        else:
            print(f"      {m:<18s} {br[m].mean() - br[base].mean():+7.2f} (unpaired)"
                  f"   {bd[m].mean() - bd[base].mean():+7.2f} (unpaired)"
                  f"      {sz:5.2f}x")


def report(methods, acc, seed, base="bucket rich"):
    br = boot(acc, methods, "rel", seed)
    bd = boot(acc, methods, "dif", seed)
    print(f"      {'method':<18s}" + "".join(f"{nm:>7s}" for nm in STRATA_NAMES)
          + f"{'sprd_rel':>10s}" + "".join(f"{nm:>7s}" for nm in QUINT_NAMES)
          + f"{'sprd_dif':>10s}{'size':>7s}{'med':>6s}{'p90':>7s}{'marg':>7s}"
          + f"{'splits':>8s}")
    for m in methods:
        a = acc[m]
        if not a["n"]:
            continue
        vr = np.array([np.nanmean(c) for c in a["rel"]]) * 100
        vd = np.array([np.nanmean(c) for c in a["dif"]]) * 100
        print(f"      {m:<18s}" + "".join(f"{x:6.1f}%" for x in vr)
              + f"{vr.max() - vr.min():8.1f}pt" + "".join(f"{x:6.1f}%" for x in vd)
              + f"{vd.max() - vd.min():8.1f}pt"
              + f"{np.mean(a['size']):7.1f}{np.mean(a['med']):6.1f}"
              + f"{np.mean(a['p90']):7.1f}{np.mean(a['marg']) * 100:6.1f}%"
              + f"{a['n']:8d}")
    print(f"\n      paired vs `{base}`     d(sprd_rel)        d(sprd_dif)"
          f"          size ratio")
    for m in methods:
        if m == base or not acc[m]["n"]:
            continue
        sz = np.mean(acc[m]["size"]) / np.mean(acc[base]["size"])
        if len(br.get(m, [])) and acc[m]["n"] == acc[base]["n"]:
            dr, dd = br[m] - br[base], bd[m] - bd[base]
            print(f"      {m:<18s} {dr.mean():+7.2f} +- {dr.std():.2f}"
                  f"   {dd.mean():+7.2f} +- {dd.std():.2f}      {sz:5.2f}x")
        else:
            print(f"      {m:<18s} {br[m].mean() - br[base].mean():+7.2f} (unpaired)"
                  f"   {bd[m].mean() - bd[base].mean():+7.2f} (unpaired)"
                  f"      {sz:5.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fire", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--run", action="append", required=True, help="name=path.npz")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--imputed-reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vlm", help="dir of scripts/vlm_ambiguity_probe.py caches; "
                                 "adds the judge's soft count as one feature")
    p.add_argument("--vlm-topk", type=int, default=30)
    a = p.parse_args()

    queries, videos = load_grid(a.test_csv)
    relevant, judged, outside, gt_bad = load_fire(a.fire, queries, videos)
    n_rel = np.array([len(r) for r in relevant])
    qlen = np.array([len(q.split()) for q in queries])

    print(f"grid {len(queries)}x{len(videos)} | mean positives {n_rel.mean():.2f} "
          f"| alpha={a.alpha} | reps={a.reps} | imputed on first "
          f"{a.imputed_reps} splits only (not a silent cap)\n")

    for spec in a.run:
        name, path = spec.split("=", 1)
        sim = np.load(path)["t2v"].astype(np.float64)
        sc = build_scores(sim, relevant)
        X, _ = gallery_features(sim, qlen)
        Xv = None
        if a.vlm:
            v = vlm_counts(a.vlm, sim, a.vlm_topk)
            print(f"judge covers {np.isfinite(v).mean():.0%} of queries, "
                  f"spearman(count, |Rel|) = "
                  f"{spearmanr(v[np.isfinite(v)], n_rel[np.isfinite(v)]).statistic:.3f}")
            Xv = np.column_stack([X, np.nan_to_num(v, nan=np.nanmean(v))])
        t0 = time.time()
        methods, acc = evaluate(sc, X, n_rel, a.alpha, a.reps, a.seed,
                                a.imputed_reps, Xv)
        print(f"=== {name} ===  ({time.time() - t0:.0f}s)")
        report(methods, acc, a.seed)
        if Xv is not None:
            # The judge only earns its cost against the best free-feature arm,
            # not against the arm it replaces one column of.
            deltas(methods, acc, a.seed, "GCC +diff")
        print()


if __name__ == "__main__":
    main()
