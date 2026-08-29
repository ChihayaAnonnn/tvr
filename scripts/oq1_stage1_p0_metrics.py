#!/usr/bin/env python
"""Frozen Stage-1 P0 / hubness metrics for study oq1-kl-variance-hubness-v1.

Computes, from an archived raw similarity matrix, exactly the quantities the
frozen protocol gates on (study.yaml stop_gates.stage1_baseline_regime and
metric_definitions):

  - retrieval: R@1 / R@5 / R@10 / MdR / MnR, rank = #{j : s_ij > s_i,gt}
    (identical convention to scripts/oq1_tier0_correction_envelope.py)
  - hubness: N_k(j) k-occurrence over candidates via top-k membership,
    S(N_k) = Fisher moment (population) skewness, k in {5, 10}

P0 gate (arm0 on val1k, t2v, DSL off):
  S(N10) >= 1.5  AND  MnR >= 8.0  AND  R@1 <= 60.0

Inputs (two modes):
  --run-dir  archived run directory: follows metrics/selected_validation.json
             -> best epoch's metrics/val1k_eval_NNN.json -> the externalized
             `oq1_raw_sim` array reference (digest-verified on load).
  --npy      a raw similarity NPY directly (ad-hoc / sanity checks).

Every number is emitted with its (split, dsl_setting, direction) triplet
(PC-12). Rows are text queries, cols are video candidates; ground truth is
the diagonal (val1k is 1000 x 1, structurally isomorphic to 1K-A).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiment_artifacts import strict_json_loads, verify_array_reference  # noqa: E402

P0_GATE = {  # study.yaml stop_gates.stage1_baseline_regime (frozen 20a4a1d)
    "arm0_val1k_hubness_skewness_n10_gte": 1.5,
    "arm0_val1k_mnr_gte": 8.0,
    "arm0_val1k_r1_lte": 60.0,
}


def retrieval_metrics(sim):
    """rank = #{j: s_ij > s_i,gt}; MdR/MnR are 1-based (Tier-0 convention)."""
    gt_scores = np.diag(sim)
    ranks = (sim > gt_scores[:, None]).sum(axis=1)
    return {
        "R1": float((ranks == 0).mean() * 100),
        "R5": float((ranks < 5).mean() * 100),
        "R10": float((ranks < 10).mean() * 100),
        "MdR": float(np.median(ranks) + 1),
        "MnR": float(ranks.mean() + 1),
    }


def hubness_metrics(sim, k):
    """N_k over columns (candidates); S(N_k) = population Fisher skewness."""
    topk = np.argpartition(-sim, k, axis=1)[:, :k]
    nk = np.bincount(topk.ravel(), minlength=sim.shape[1]).astype(np.float64)
    mu, sd = nk.mean(), nk.std()
    return {
        f"S_N{k}": float(((nk - mu) ** 3).mean() / sd**3) if sd > 0 else 0.0,
        f"max_N{k}": int(nk.max()),
        f"zero_N{k}": int((nk == 0).sum()),
    }


def direction_report(sim, *, split, dsl_setting, direction):
    report = {"split": split, "dsl_setting": dsl_setting, "direction": direction}
    report.update(retrieval_metrics(sim))
    for k in (5, 10):
        report.update(hubness_metrics(sim, k))
    return report


def load_from_run_dir(run_dir):
    run_dir = Path(run_dir).resolve()
    selected = strict_json_loads(
        (run_dir / "metrics/selected_validation.json").read_text(encoding="utf-8")
    )
    sequence = int(selected["summary_metrics"]["eval_sequence"])
    eval_doc = strict_json_loads(
        (run_dir / f"metrics/val1k_eval_{sequence:03d}.json").read_text(
            encoding="utf-8"
        )
    )
    block = eval_doc["oq1_raw_sim"]
    if block["dsl_setting"] != "off_raw_pre_dsl":
        raise SystemExit("archived matrix is not the raw pre-DSL matrix")
    sim = verify_array_reference(run_dir, block["reference"])
    return np.asarray(sim, dtype=np.float64), {
        "run_dir": str(run_dir),
        "best_epoch": int(selected["summary_metrics"]["best_epoch"]),
        "eval_sequence": sequence,
        "array_sha256": block["reference"]["sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="archived run directory")
    source.add_argument("--npy", help="raw similarity NPY (rows=text, cols=video)")
    parser.add_argument("--split", default="msrvtt_trusted_v1_val1k")
    parser.add_argument(
        "--gate", action="store_true",
        help="evaluate the P0 regime gate (arm0 runs only)",
    )
    parser.add_argument("--output", help="write the JSON report here as well")
    args = parser.parse_args()

    if args.run_dir:
        sim, provenance = load_from_run_dir(args.run_dir)
    else:
        sim = np.asarray(np.load(args.npy), dtype=np.float64)
        provenance = {"npy": str(Path(args.npy).resolve())}
    if sim.ndim != 2:
        raise SystemExit(f"similarity matrix must be 2-D, got {sim.shape}")

    report = {
        "study": "oq1-kl-variance-hubness-v1",
        "matrix_shape": list(sim.shape),
        "provenance": provenance,
        "t2v": direction_report(
            sim, split=args.split, dsl_setting="off", direction="t2v"
        ),
        "v2t": direction_report(
            sim.T, split=args.split, dsl_setting="off", direction="v2t"
        ),
    }
    if args.gate:
        t2v = report["t2v"]
        checks = {
            "S_N10_gte_1.5": t2v["S_N10"] >= P0_GATE["arm0_val1k_hubness_skewness_n10_gte"],
            "MnR_gte_8.0": t2v["MnR"] >= P0_GATE["arm0_val1k_mnr_gte"],
            "R1_lte_60.0": t2v["R1"] <= P0_GATE["arm0_val1k_r1_lte"],
        }
        report["p0_gate"] = {
            "thresholds": P0_GATE,
            "checks": checks,
            "pass": all(checks.values()),
            "on_failure": "halt_study_as_inconclusive (does NOT fire F-A, does NOT unseal test)",
        }

    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
