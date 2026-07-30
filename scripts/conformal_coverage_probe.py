#!/usr/bin/env python
"""Gate experiment for ambiguity-conditional set-valued text-video retrieval.

The question this answers is whether there is a problem worth solving. The
proposal is to stop returning a ranked list scored by R@1 and instead return a
*set* with a distribution-free coverage guarantee, where the set is small when
the query is unambiguous and large when it is not. That proposal only has a
technical core if marginal split conformal -- one global threshold for every
query -- gives visibly uneven coverage across queries of different ambiguity.
If coverage is already uniform, there is nothing to condition on and the idea
dies here.

Three things are measured, all from similarity matrices already on disk:

  1. Does marginal coverage hold conditionally? Calibrate one threshold on half
     the queries, then measure coverage separately within strata defined by how
     many videos FIRE says are actually relevant. Marginal coverage is
     guaranteed by construction; the spread across strata is not.

  2. Is the set a usable object, and does the per-query margin carry anything?
     A margin set {v : s_top1 - s_v <= lambda} adapts its size per query. A
     fixed top-k set does not. At matched coverage, if the adaptive set is no
     smaller on average, the margin carries no per-query information and the
     "free confidence signal" everyone leans on is worth nothing.

  3. How much does the choice of labels move a DAB-style risk-coverage curve?
     DAB (arXiv:2607.20984) reports AURC 0.429 -> 0.256 for its KL margin
     against random ordering, computed against MSR-VTT's single positive. Under
     FIRE's judgments the same query has 3.18 positives on average, so part of
     the "risk" being reported is label error. Recomputing the same curve under
     both labellings says how much.

Coverage targets. `any` means the set contains at least one relevant video;
more positives make this easier. `all` means it contains every *judged*
relevant video; more positives make this harder. Both are reported because they
fail in opposite directions and a conditional-coverage claim that only holds for
one of them is an artefact of the target, not a property of the model.

Pool caveat: FIRE judged ~24 of 1000 videos per caption, so a set may contain an
unjudged relevant video that is not counted. `any` coverage is therefore a lower
bound and `all` coverage an upper bound on the truth.

Usage:
    python scripts/conformal_coverage_probe.py \
        --fire .scratch/mm-retrieval-evaluation/data/fire_msrvtt_dataset.json \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --run parity_a0=.scratch/simdump/parity_a0/sim_test_pytorch_model.bin.2.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_corrected_metrics import load_fire, load_grid  # noqa: E402

STRATA = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10**9)]
STRATA_NAMES = ["1", "2", "3", "4-5", "6+"]


def stratum_of(n_rel):
    for s, (lo, hi) in enumerate(STRATA):
        if lo <= n_rel <= hi:
            return s
    raise AssertionError(n_rel)


def conformal_quantile(scores, alpha):
    """The finite-sample split-conformal threshold.

    Taking the ceil((n+1)(1-alpha))-th smallest calibration score, rather than
    the plain empirical quantile, is what makes test coverage >= 1-alpha rather
    than approximately 1-alpha. If the index runs past the end no finite
    threshold is valid and the set has to be everything.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def build_scores(sim, relevant):
    """Per-query nonconformity scores and the free confidence signals.

    margin_any is the extra slack below the top-1 score needed to admit the
    best relevant video, so thresholding it gives `any` coverage. margin_all
    needs the worst relevant video, so thresholding it gives `all` coverage.
    rank_any is the same quantity in rank space, which is what a fixed top-k
    set thresholds.
    """
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    top1 = sim[np.arange(n), order[:, 0]]
    top2 = sim[np.arange(n), order[:, 1]]

    rank_of = np.empty_like(order)
    np.put_along_axis(rank_of, order, np.tile(np.arange(n), (n, 1)), axis=1)

    margin_any = np.empty(n)
    margin_all = np.empty(n)
    rank_any = np.empty(n, dtype=int)
    for i in range(n):
        rel = np.fromiter(relevant[i], int, len(relevant[i]))
        s = sim[i, rel]
        margin_any[i] = top1[i] - s.max()
        margin_all[i] = top1[i] - s.min()
        rank_any[i] = rank_of[i, rel].min()

    return {
        "sim": sim,
        "top1": top1,
        "gap12": top1 - top2,
        "margin_any": margin_any,
        "margin_all": margin_all,
        "rank_any": rank_any,
    }


def set_sizes(sim, lam):
    """|{v : s_top1 - s_v <= lam}| for every query, without materialising sets."""
    top1 = sim.max(axis=1, keepdims=True)
    return (top1 - sim <= lam).sum(axis=1)


def run_splits(sc, n_rel, alpha, n_rep, rng, pred_feat=None, qlen=None):
    """Split conformal, repeated over random calibration/test halves.

    Everything is accumulated per repeat and averaged at the end, so the
    reported numbers are not a single lucky split.
    """
    n = len(n_rel)
    strat = np.array([stratum_of(r) for r in n_rel])
    half = n // 2

    acc = {
        "cov_any": [], "cov_all": [], "size_any": [], "size_med": [],
        "cov_topk": [], "k_topk": [],
        "cov_any_by_s": [[] for _ in STRATA], "cov_mond_by_s": [[] for _ in STRATA],
        "size_mond": [], "cov_all_by_s": [[] for _ in STRATA],
        "cov_pred_by_s": [[] for _ in STRATA], "size_pred": [],
    }

    # Label-free ambiguity score. Both features are computed from the query and
    # the ranking only, never from a judgment, so using all 1000 to form the
    # score leaks nothing; only the bucket boundaries come from calibration.
    if pred_feat is not None:
        from scipy.stats import rankdata
        amb = rankdata(-pred_feat) + rankdata(-qlen.astype(float))

    for _ in range(n_rep):
        perm = rng.permutation(n)
        cal, tst = perm[:half], perm[half:]

        # --- adaptive margin set, one global threshold (marginal conformal) ---
        lam = conformal_quantile(sc["margin_any"][cal], alpha)
        covered = sc["margin_any"][tst] <= lam
        sizes = set_sizes(sc["sim"][tst], lam)
        acc["cov_any"].append(covered.mean())
        acc["size_any"].append(sizes.mean())
        acc["size_med"].append(np.median(sizes))
        for s in range(len(STRATA)):
            m = strat[tst] == s
            if m.any():
                acc["cov_any_by_s"][s].append(covered[m].mean())

        lam_all = conformal_quantile(sc["margin_all"][cal], alpha)
        cov_all = sc["margin_all"][tst] <= lam_all
        acc["cov_all"].append(cov_all.mean())
        for s in range(len(STRATA)):
            m = strat[tst] == s
            if m.any():
                acc["cov_all_by_s"][s].append(cov_all[m].mean())

        # --- fixed top-k set, the non-adaptive reference at the same target ---
        k = conformal_quantile(sc["rank_any"][cal].astype(float), alpha) + 1
        acc["cov_topk"].append((sc["rank_any"][tst] < k).mean())
        acc["k_topk"].append(k)

        # --- Mondrian: one threshold per oracle ambiguity stratum ---
        mond_cov, mond_size = np.zeros(len(tst), bool), np.zeros(len(tst))
        for s in range(len(STRATA)):
            mc, mt = cal[strat[cal] == s], np.flatnonzero(strat[tst] == s)
            if len(mc) < 10 or len(mt) == 0:
                # Too few calibration queries in this stratum for a threshold
                # to mean anything; fall back to the global one.
                lam_s = lam
            else:
                lam_s = conformal_quantile(sc["margin_any"][mc], alpha)
            mond_cov[mt] = sc["margin_any"][tst[mt]] <= lam_s
            mond_size[mt] = set_sizes(sc["sim"][tst[mt]], lam_s)
            acc["cov_mond_by_s"][s].append(mond_cov[mt].mean() if len(mt) else np.nan)
        acc["size_mond"].append(mond_size.mean())

        # --- Mondrian on predicted ambiguity, the deployable version ---
        if pred_feat is not None:
            edges = np.quantile(amb[cal], [0.2, 0.4, 0.6, 0.8])
            pb = np.digitize(amb, edges)
            pcov, psize = np.zeros(len(tst), bool), np.zeros(len(tst))
            for b in range(5):
                mc, mt = cal[pb[cal] == b], np.flatnonzero(pb[tst] == b)
                if len(mc) < 10 or len(mt) == 0:
                    lam_b = lam
                else:
                    lam_b = conformal_quantile(sc["margin_any"][mc], alpha)
                pcov[mt] = sc["margin_any"][tst[mt]] <= lam_b
                psize[mt] = set_sizes(sc["sim"][tst[mt]], lam_b)
            acc["size_pred"].append(psize.mean())
            for s in range(len(STRATA)):
                m = strat[tst] == s
                if m.any():
                    acc["cov_pred_by_s"][s].append(pcov[m].mean())

    out = {}
    for k_, v in acc.items():
        if k_.endswith("_by_s"):
            out[k_] = [float(np.mean(x)) if x else float("nan") for x in v]
        else:
            # size_pred / cov_pred_* stay empty when no predictor was passed.
            out[k_] = float(np.mean(v)) if len(v) else float("nan")
    return out


def risk_coverage(conf, correct):
    """AURC for selective prediction: reject the least confident queries first."""
    order = np.argsort(-conf)
    err = (~correct[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean()), risk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fire", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--run", action="append", required=True, help="name=path.npz")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    queries, videos = load_grid(a.test_csv)
    relevant, judged, outside, gt_bad = load_fire(a.fire, queries, videos)
    n_rel = np.array([len(r) for r in relevant])
    qlen = np.array([len(q.split()) for q in queries])

    print(f"grid {len(queries)}x{len(videos)} | mean positives {n_rel.mean():.2f} "
          f"| {outside} judgments outside grid | alpha={a.alpha} reps={a.reps}")
    print("ambiguity strata (true |Rel|): " + "  ".join(
        f"{nm}:{int((np.array([stratum_of(r) for r in n_rel]) == s).sum())}"
        for s, nm in enumerate(STRATA_NAMES)) + "\n")

    for spec in a.run:
        name, path = spec.split("=", 1)
        sim = np.load(path)["t2v"].astype(np.float64)
        assert sim.shape == (len(queries), len(videos)), f"{name}: {sim.shape}"
        sc = build_scores(sim, relevant)
        rng = np.random.RandomState(a.seed)
        r = run_splits(sc, n_rel, a.alpha, a.reps, rng)

        print(f"=== {name} ===")
        print(f"  [2] adaptive margin set   coverage {r['cov_any']*100:5.1f}%  "
              f"mean size {r['size_any']:7.1f}  median {r['size_med']:6.1f}")
        print(f"      fixed top-k set       coverage {r['cov_topk']*100:5.1f}%  "
              f"mean size {r['k_topk']:7.1f}   <- non-adaptive reference")
        gain = (1 - r["size_any"] / r["k_topk"]) * 100
        print(f"      adaptive is {gain:+.1f}% the size of fixed-k at matched target\n")

        print("  [1] conditional coverage of ONE global threshold, by true |Rel|")
        print("      stratum        " + "".join(f"{nm:>8s}" for nm in STRATA_NAMES) + "     spread")
        for tag, key in (("any", "cov_any_by_s"), ("all", "cov_all_by_s")):
            v = np.array(r[key]) * 100
            fin = v[np.isfinite(v)]
            print(f"      {tag:<3s} coverage " + "".join(f"{x:7.1f}%" for x in v)
                  + f"   {fin.max()-fin.min():6.1f} pt")
        v = np.array(r["cov_mond_by_s"]) * 100
        fin = v[np.isfinite(v)]
        print("      any, Mondrian " + "".join(f"{x:7.1f}%" for x in v)
              + f"   {fin.max()-fin.min():6.1f} pt")
        print(f"      Mondrian mean set size {r['size_mond']:.1f} "
              f"(global {r['size_any']:.1f})\n")

        print("  [3] can ambiguity be predicted without labels? (Spearman vs |Rel|)")
        for fname, fv in (("top1-top2 gap", sc["gap12"]), ("top1 score", sc["top1"]),
                          ("caption length", qlen.astype(float))):
            rho, pv = spearmanr(fv, n_rel)
            print(f"      {fname:<16s} rho {rho:+.3f}  p={pv:.2e}")
        print("      (negative rho = more positives, so all three point the same way:"
              " a flat, low-scoring, short query is the ambiguous one)")

        # The oracle Mondrian above conditions on the label it is trying to be
        # robust to, so it is an upper bound, not a method. This is the
        # deployable version: bucket by a label-free predictor of ambiguity,
        # calibrate per bucket, then re-measure coverage in the TRUE strata.
        rng2 = np.random.RandomState(a.seed)
        pred_r = run_splits(sc, n_rel, a.alpha, a.reps, rng2,
                            pred_feat=sc["gap12"], qlen=qlen)
        v = np.array(pred_r["cov_pred_by_s"]) * 100
        fin = v[np.isfinite(v)]
        print("\n      predicted-stratum Mondrian (label-free), coverage by TRUE |Rel|")
        print("      " + "".join(f"{x:7.1f}%" for x in v)
              + f"   {fin.max()-fin.min():6.1f} pt   mean size {pred_r['size_pred']:.1f}\n")

        # DAB's own confidence signal is the top1-top2 margin, scored against
        # the single annotated positive. Under random ordering the risk at every
        # coverage is the overall error rate in expectation, so that error rate
        # is the random-ordering AURC and needs no simulation.
        top1_col = np.argmax(sim, axis=1)
        hit_orig = top1_col == np.arange(sim.shape[0])
        hit_fire = np.array([c in relevant[i] for i, c in enumerate(top1_col)])
        au_o, _ = risk_coverage(sc["gap12"], hit_orig)
        au_f, _ = risk_coverage(sc["gap12"], hit_fire)
        print(f"  [4] risk-coverage by top1-top2 margin, R@1 risk")
        print(f"      original labels  AURC {au_o:.3f}  vs random {1-hit_orig.mean():.3f}"
              f"   reduction {(1-au_o/(1-hit_orig.mean()))*100:5.1f}%")
        print(f"      FIRE labels      AURC {au_f:.3f}  vs random {1-hit_fire.mean():.3f}"
              f"   reduction {(1-au_f/(1-hit_fire.mean()))*100:5.1f}%\n")


if __name__ == "__main__":
    main()
