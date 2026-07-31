#!/usr/bin/env python
"""Can an external VLM judge estimate how many videos a caption actually matches?

step-003 established that the ambiguity axis of conditional coverage is a
*signal* problem, not a calibration problem: Mondrian, CQR and conditional
conformal over a function class all sit at 11.4-14.6 pt of coverage spread
across |Rel| strata while an oracle stratifier needs only 0.4-0.8. No choice of
function class can invent |Rel| information that the score row does not carry.

A dual encoder structurally cannot supply it. It emits one scalar per pair, so
"is this one actually correct" is never asked -- only "is it higher than the
others". A generative VLM can be asked that question directly, once per pair,
and the count of yes answers is an estimate of |Rel|.

Feasibility, all computed before the criterion was written:
  * Spearman(|judged pool|, |Rel|) = 0.059, so a judge with a constant yes-rate
    cannot fake a correlation by tracking pool size.
  * 96% of FIRE's relevant videos are inside our top-50, and a perfect judge
    restricted to top-50 would score rho = 0.972. The pool is not the ceiling.
  * A pure dual-encoder margin count over top-50 scores rho = 0.225 even with
    its threshold tuned on the whole test set. The 17-feature linear predictor
    scores 0.55. That is the number to beat.

Two readouts come out of one pass, because the judged pool and the retrieved
pool are unioned before judging:

  count   -- yes-count over top-K, the deployable estimator, scored by Spearman
             against |Rel|. Soft variant sums P(yes) instead of thresholding.
  pairwise-- on the FIRE-judged pairs only, AUC of P(yes) against the human
             label. Separates "judge ranks correctly but the threshold is off"
             from "judge is wrong".

Usage:
    python scripts/vlm_ambiguity_probe.py \
        --fire .scratch/mm-retrieval-evaluation/data/fire_msrvtt_dataset.json \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --sim .scratch/simdump/parity_a0/sim_test_pytorch_model.bin.2.npz \
        --frames .scratch/grid_frames --model /data2/hxj/model/qwen25vl7b \
        --n-queries 200 --topk 50 --out .scratch/vlm_judge
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_corrected_metrics import load_fire, load_grid  # noqa: E402

PROMPT = (
    "Caption: \"{cap}\"\n"
    "The frames above are sampled in order from one short video. "
    "Would this caption be an acceptable description of that video? "
    "Answer with one word, yes or no."
)


def auc(scores, labels):
    """Rank AUC with ties averaged; returns nan if one class is absent."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Judge:
    def __init__(self, model_dir, n_frames, device="cuda"):
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.n_frames = n_frames
        tok = self.proc.tokenizer
        # Score the first generated token. Both casings are summed because the
        # chat template does not pin which one the model prefers.
        self.yes = [tok.encode(t)[0] for t in ("yes", "Yes", " yes", " Yes")]
        self.no = [tok.encode(t)[0] for t in ("no", "No", " no", " No")]

    def _one(self, cap, frames):
        msg = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": PROMPT.format(cap=cap)}]}]
        return self.proc.apply_chat_template(msg, tokenize=False,
                                             add_generation_prompt=True), frames

    @torch.no_grad()
    def batch(self, caps, framesets):
        texts, vids = zip(*[self._one(c, f) for c, f in zip(caps, framesets)])
        enc = self.proc(text=list(texts), videos=list(vids),
                        padding=True, return_tensors="pt").to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(lp[:, self.yes], dim=-1)
        n = torch.logsumexp(lp[:, self.no], dim=-1)
        # Renormalise over the two answers so the readout is P(yes | yes or no).
        return torch.sigmoid(y - n).cpu().numpy()


def load_frames(root, vid, n):
    d = os.path.join(root, vid)
    fs = sorted(os.listdir(d))[:n]
    return [Image.open(os.path.join(d, f)).convert("RGB") for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", default=".scratch/mm-retrieval-evaluation/data/"
                                      "fire_msrvtt_dataset.json")
    ap.add_argument("--test-csv",
                    default="/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv")
    ap.add_argument("--sim",
                    default=".scratch/simdump/parity_a0/"
                            "sim_test_pytorch_model.bin.2.npz")
    ap.add_argument("--frames", default=".scratch/grid_frames")
    ap.add_argument("--model", default="/data2/hxj/model/qwen25vl7b")
    ap.add_argument("--out", default=".scratch/vlm_judge")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # Judging is embarrassingly parallel over queries. Each shard owns a slice
    # of the query list and its own cache file; the readout reads every shard's
    # file back, so a run with --shard unset still sees work done by shards.
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()

    queries, videos = load_grid(a.test_csv)
    relevant, judged, _, _ = load_fire(a.fire, queries, videos)
    sim = np.load(a.sim)["t2v"]
    order = np.argsort(-sim, axis=1)

    # One row per distinct caption; a caption repeated on two rows carries the
    # same judgments, so keeping both would double-count it in the correlation.
    first = {}
    for i, q in enumerate(queries):
        first.setdefault(q, i)
    rows = np.array(sorted(first.values()))
    rng = np.random.default_rng(a.seed)
    rows = rng.permutation(rows)[: a.n_queries]

    os.makedirs(a.out, exist_ok=True)
    stem = f"pyes_seed{a.seed}_k{a.topk}"
    cache_path = os.path.join(
        a.out, stem + ("" if a.shard < 0 else f".s{a.shard}") + ".json")
    cache = {}
    for f in sorted(os.listdir(a.out)):
        if f.startswith(stem) and f.endswith(".json"):
            cache.update(json.load(open(os.path.join(a.out, f))))
    mine = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    todo = rows if a.shard < 0 else rows[a.shard::a.n_shards]
    pending = []
    for i in todo:
        pool = sorted(set(order[i, : a.topk].tolist()) | set(judged[i]))
        for j in pool:
            if f"{i}:{j}" not in cache:
                pending.append((int(i), int(j)))
    print(f"{len(rows)} queries, {len(pending)} pairs to judge "
          f"({len(cache)} cached)", flush=True)

    if pending:
        judge = Judge(a.model, a.n_frames)
        for b in range(0, len(pending), a.batch):
            chunk = pending[b: b + a.batch]
            caps = [queries[i] for i, _ in chunk]
            fr = [load_frames(a.frames, videos[j], a.n_frames) for _, j in chunk]
            p = judge.batch(caps, fr)
            for (i, j), v in zip(chunk, p):
                cache[f"{i}:{j}"] = mine[f"{i}:{j}"] = float(v)
            if (b // a.batch) % 25 == 0:
                print(f"  {b + len(chunk)}/{len(pending)}", flush=True)
                json.dump(mine, open(cache_path, "w"))
        json.dump(mine, open(cache_path, "w"))

    if a.shard >= 0:
        print("shard done; rerun without --shard for the readout")
        return

    # ---- readout 1: count over top-K vs |Rel| -----------------------------
    n_rel = np.array([len(relevant[i]) for i in rows], float)
    top = [order[i, : a.topk].tolist() for i in rows]
    p_top = [np.array([cache[f"{i}:{j}"] for j in t]) for i, t in zip(rows, top)]
    soft = np.array([p.sum() for p in p_top])
    print("\n=== estimator vs |Rel| (n=%d queries) ===" % len(rows))
    print("%-22s %8s" % ("estimator", "spearman"))
    print("%-22s %8.3f" % ("soft sum P(yes)", spearmanr(soft, n_rel).statistic))
    best = (-1.0, None)
    for thr in np.arange(0.1, 0.95, 0.05):
        hard = np.array([(p >= thr).sum() for p in p_top], float)
        r = spearmanr(hard, n_rel).statistic
        if r > best[0]:
            best = (r, thr)
        if abs(thr - 0.5) < 1e-9:
            print("%-22s %8.3f" % ("hard count @0.50", r))
    print("%-22s %8.3f   (thr=%.2f, tuned on this set -- optimistic)"
          % ("hard count @best", best[0], best[1]))
    oracle = np.array([len(set(t) & relevant[i]) for i, t in zip(rows, top)], float)
    print("%-22s %8.3f" % ("oracle on same pool", spearmanr(oracle, n_rel).statistic))

    # ---- readout 2: per-pair agreement on FIRE-judged pairs ---------------
    ps, ys = [], []
    for i in rows:
        for j in sorted(judged[i]):
            k = f"{i}:{j}"
            if k in cache:
                ps.append(cache[k])
                ys.append(1 if j in relevant[i] else 0)
    ps, ys = np.array(ps), np.array(ys)
    acc = ((ps >= 0.5).astype(int) == ys).mean()
    tpr = (ps[ys == 1] >= 0.5).mean()
    tnr = (ps[ys == 0] < 0.5).mean()
    print("\n=== per-pair agreement with FIRE (%d pairs, %.1f%% relevant) ==="
          % (len(ys), 100 * ys.mean()))
    print("AUC %.3f | acc %.3f | TPR %.3f | TNR %.3f | yes-rate %.3f"
          % (auc(ps, ys), acc, tpr, tnr, (ps >= 0.5).mean()))

    json.dump({"rows": rows.tolist(), "n_rel": n_rel.tolist(),
               "soft": soft.tolist(), "auc": auc(ps, ys),
               "rho_soft": spearmanr(soft, n_rel).statistic,
               "rho_hard_best": best[0], "thr_best": float(best[1])},
              open(os.path.join(a.out, f"summary_seed{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
