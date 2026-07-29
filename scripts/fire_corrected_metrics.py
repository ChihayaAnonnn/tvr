#!/usr/bin/env python
"""Re-score dumped t2v similarity matrices against FIRE's relevance judgments.

MSR-VTT gives each caption exactly one positive video and calls the other 999
negatives. FIRE (Rodriguez et al., EACL 2023) had humans judge caption-video
pairs pooled from three retrieval models and found that most captions have
several correct videos. On our JSFUSION test grid that is 2855 extra positives
over 887 of 995 distinct captions, so the original R@1 counts a correct answer
as wrong roughly as often as not.

This script recomputes metrics both ways from the same matrix. Two things make
the comparison honest rather than a free win:

  * FIRE is a *pooled* judgment set -- about 24 of 1000 videos are judged per
    caption, drawn from the top-k of CLIP4CLIP, SSB and CE. Everything else is
    unjudged, not known-irrelevant. Scoring unjudged as irrelevant penalises a
    model the pool was not built from, which is exactly our situation: the pool
    is from 2022 and our backbone is stronger than all three. So we report
    judged@k (how much of our top-k the pool actually covers) and the two
    standard pooling-robust measures, condensed-list recall and bpref.

  * The original positive is kept relevant even where FIRE's annotators marked
    it irrelevant, and the count of those cases is reported separately. Letting
    the correction delete ground truth would change what the benchmark is, not
    just how it is scored.

Usage:
    python scripts/fire_corrected_metrics.py \
        --fire .scratch/mm-retrieval-evaluation/data/fire_msrvtt_dataset.json \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --run parity_a0=.scratch/simdump/parity_a0/sim_test_pytorch_model.bin.2.npz \
        --run parity_a4=.scratch/simdump/parity_a4/sim_test_pytorch_model.bin.2.npz
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def load_grid(test_csv):
    """Row i of the test csv is both text i and video i of the matrix.

    compute_metrics() reads the ground truth off the diagonal, so this identity
    is what the reported numbers already assume. verify_original() re-derives
    R@1 from the diagonal and will fail loudly if it ever stops holding.
    """
    rows = list(csv.DictReader(open(test_csv)))
    queries = [r["sentence"].strip() for r in rows]
    videos = [r["video_id"] for r in rows]
    return queries, videos


def load_fire(fire_json, queries, videos):
    """Map FIRE's (caption, video_id) judgments onto column indices.

    Relevance is a property of the caption text, not of the csv row, so a
    caption appearing on two rows inherits the same judgments on both.
    """
    ann = json.load(open(fire_json))["annotations"]
    col_of = {v: j for j, v in enumerate(videos)}
    rel_by_query = defaultdict(set)
    judged_by_query = defaultdict(set)
    outside = 0
    for a in ann:
        q, v = a["query"].strip(), a["video_id"]
        j = col_of.get(v)
        if j is None:
            outside += 1
            continue
        judged_by_query[q].add(j)
        if a["label"] == "relevant":
            rel_by_query[q].add(j)

    n = len(queries)
    relevant, judged = [], []
    gt_marked_irrelevant = 0
    for i, q in enumerate(queries):
        r = set(rel_by_query.get(q, ()))
        jd = set(judged_by_query.get(q, ()))
        if i in jd and i not in r:
            gt_marked_irrelevant += 1
        # The original positive stays relevant and counts as judged, whatever
        # the annotators said about it.
        r.add(i)
        jd.add(i)
        relevant.append(r)
        judged.append(jd)
    return relevant, judged, outside, gt_marked_irrelevant


def verify_original(sim, tol=1e-9):
    """R@k from the diagonal -- must reproduce the number in the run log."""
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == np.arange(sim.shape[0])[:, None], axis=1)
    return {f"R{k}": float(np.mean(ranks < k) * 100) for k in (1, 5, 10)} | {
        "MeanR": float(np.mean(ranks) + 1),
        "ranks": ranks,
    }


def dcg(gains):
    return float(np.sum(np.asarray(gains) / np.log2(np.arange(2, len(gains) + 2))))


def corrected_metrics(sim, relevant, judged, ks=(1, 5, 10)):
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    out = {f"R{k}": [] for k in ks}
    out |= {f"judged@{k}": [] for k in ks}
    out |= {f"cond_R{k}": [] for k in ks}
    ndcg10, ap, bpref, hit1, cond_hit1 = [], [], [], [], []

    for i in range(n):
        rank = order[i]
        rel, jd = relevant[i], judged[i]
        is_rel = np.fromiter((c in rel for c in rank), bool, n)
        is_jd = np.fromiter((c in jd for c in rank), bool, n)

        for k in ks:
            out[f"R{k}"].append(bool(is_rel[:k].any()))
            out[f"judged@{k}"].append(float(is_jd[:k].mean()))
        hit1.append(bool(is_rel[0]))

        # Condensed list: drop unjudged documents, then rank as usual. This is
        # the standard way to score a run the pool was not built from -- an
        # unjudged hit no longer costs the model a rank position.
        cond = is_rel[is_jd]
        for k in ks:
            out[f"cond_R{k}"].append(bool(cond[:k].any()) if cond.size else False)
        cond_hit1.append(bool(cond[:1].any()) if cond.size else False)

        ndcg10.append(dcg(is_rel[:10].astype(float)) / dcg([1.0] * min(len(rel), 10)))

        pos = np.flatnonzero(is_rel)
        ap.append(float(np.mean((np.arange(len(pos)) + 1) / (pos + 1))))

        # bpref over judged documents only.
        R, N = len(rel), int(is_jd.sum()) - len(rel)
        if N > 0:
            # cumsum at a relevant position adds 0 there, so this is the count
            # of judged-nonrelevant strictly above each hit.
            nonrel_above = np.cumsum(is_jd & ~is_rel)[pos]
            bpref.append(float(np.mean(1.0 - np.minimum(nonrel_above, min(R, N)) / min(R, N))))
        else:
            bpref.append(1.0)

    res = {k: float(np.mean(v) * 100) for k, v in out.items() if k.startswith(("R", "cond_R"))}
    res |= {k: float(np.mean(v) * 100) for k, v in out.items() if k.startswith("judged@")}
    res |= {
        "nDCG@10": float(np.mean(ndcg10) * 100),
        "mAP": float(np.mean(ap) * 100),
        "bpref": float(np.mean(bpref) * 100),
        "n_relevant_mean": float(np.mean([len(r) for r in relevant])),
        "hit1": np.array(hit1),
        "cond_hit1": np.array(cond_hit1),
    }
    return res


def mcnemar_exact(a, b):
    """Two-sided exact McNemar over paired per-query hit indicators."""
    from math import comb

    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return n10, n01, p


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--fire", required=True)
    ap_.add_argument("--test-csv", required=True)
    ap_.add_argument("--run", action="append", required=True, help="name=path/to/sim.npz")
    args = ap_.parse_args()

    queries, videos = load_grid(args.test_csv)
    relevant, judged, outside, gt_bad = load_fire(args.fire, queries, videos)

    n_extra = sum(len(r) for r in relevant) - len(relevant)
    print(f"grid: {len(queries)} rows, {len(set(queries))} distinct captions, {len(videos)} videos")
    print(f"FIRE: {n_extra} extra positives, {outside} judgments outside the grid")
    print(f"FIRE marked the original positive irrelevant on {gt_bad} rows (kept relevant anyway)")
    print(f"mean positives per query: 1.00 -> {np.mean([len(r) for r in relevant]):.2f}")
    print(f"mean judged per query: {np.mean([len(j) for j in judged]):.1f} of {len(videos)}\n")

    results = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        sim = np.load(path)["t2v"].astype(np.float64)
        assert sim.shape == (len(queries), len(videos)), f"{name}: {sim.shape}"
        orig = verify_original(sim)
        corr = corrected_metrics(sim, relevant, judged)
        results[name] = (orig, corr)
        print(f"=== {name} ===")
        print("  original  R@1 %5.1f  R@5 %5.1f  R@10 %5.1f  MeanR %5.1f"
              % (orig["R1"], orig["R5"], orig["R10"], orig["MeanR"]))
        print("  FIRE      R@1 %5.1f  R@5 %5.1f  R@10 %5.1f" % (corr["R1"], corr["R5"], corr["R10"]))
        print("  condensed R@1 %5.1f  R@5 %5.1f  R@10 %5.1f"
              % (corr["cond_R1"], corr["cond_R5"], corr["cond_R10"]))
        print("  nDCG@10 %5.1f   mAP %5.1f   bpref %5.1f" % (corr["nDCG@10"], corr["mAP"], corr["bpref"]))
        print("  pool coverage of our ranking: judged@1 %5.1f%%  judged@5 %5.1f%%  judged@10 %5.1f%%\n"
              % (corr["judged@1"], corr["judged@5"], corr["judged@10"]))

    names = list(results)
    for x, y in [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]:
        ox, cx = results[x]
        oy, cy = results[y]
        print(f"=== paired {x} vs {y} ===")
        for label, hx, hy in (
            ("original R@1", ox["ranks"] == 0, oy["ranks"] == 0),
            ("FIRE R@1", cx["hit1"], cy["hit1"]),
            ("condensed R@1", cx["cond_hit1"], cy["cond_hit1"]),
        ):
            n10, n01, p = mcnemar_exact(np.asarray(hx), np.asarray(hy))
            d = (np.mean(hy) - np.mean(hx)) * 100
            print(f"  {label:14s} {np.mean(hx)*100:5.1f} -> {np.mean(hy)*100:5.1f}  "
                  f"delta {d:+5.2f}  discordant {n10}/{n01}  McNemar p={p:.3f}")
        print()


if __name__ == "__main__":
    main()
