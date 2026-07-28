#!/usr/bin/env python
"""Report what RSPR reranking and pair uncertainty actually contribute.

Run against the ``--rspr_dump_scores`` output of an evaluation:

    python scripts/analyze_rspr_scores.py logs/rspr_dumps/a3fixv3_test.npz

Three questions, all answered without touching a GPU:

* ``lambda=0`` vs the recorded scores: does reranking move any decision?
* Recorded vs permuted matcher scores: is the movement information or noise?
* Uncertainty AUC: does ``u_pair`` know when a retrieval is wrong?
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_task_retrieval import (  # noqa: E402
    _build_rspr_metric_matrices,
    _compute_directional_metrics,
)
from modules.rspr_score_analysis import (  # noqa: E402
    candidate_recall_ceiling,
    count_rerank_flips,
    permute_probability_within_candidates,
    rank_auc,
    recover_rerank_coefficient,
    replay_rerank,
    top1_uncertainty_and_correctness,
)


def _metrics(dump, t2v, v2t):
    metric_t2v, metric_v2t = _build_rspr_metric_matrices(t2v, v2t, dump["recall"])
    return _compute_directional_metrics(metric_t2v, metric_v2t)


def _format(name, t2v_metrics, v2t_metrics):
    return (
        f"{name:<28}"
        f" T2V {t2v_metrics['R1']:5.1f}/{t2v_metrics['R5']:5.1f}/{t2v_metrics['R10']:5.1f}"
        f" MnR {t2v_metrics['MeanR']:5.1f}"
        f" | V2T {v2t_metrics['R1']:5.1f}/{v2t_metrics['R5']:5.1f}/{v2t_metrics['R10']:5.1f}"
        f" MnR {v2t_metrics['MeanR']:5.1f}"
    )


def _flip_report(name, baseline, reranked):
    repaired, broken = count_rerank_flips(baseline, reranked)
    return f"{name} +{repaired}/-{broken}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="Path to an --rspr_dump_scores .npz")
    parser.add_argument("--permutations", type=int, default=10)
    arguments = parser.parse_args()

    dump = {name: matrix for name, matrix in np.load(arguments.dump).items()}
    text_count, video_count = dump["recall"].shape
    coefficient = recover_rerank_coefficient(dump)
    print(f"{arguments.dump}: {text_count} texts x {video_count} videos")
    print(f"recovered rerank coefficient lambda*c/T_p = {coefficient:.4f}")

    scored = np.isfinite(dump["probability"])
    uncertainty = dump["uncertainty"][scored]
    print(
        f"u_pair over {scored.sum()} scored pairs: mean {uncertainty.mean():.3e}"
        f" std {uncertainty.std():.3e}"
        f" min {uncertainty.min():.3e} max {uncertainty.max():.3e}"
    )

    print("\n--- reranking ---")
    baseline_t2v, baseline_v2t = replay_rerank(dump, dump["probability"], 0.0)
    print(_format("lambda=0 (deterministic)", *_metrics(dump, baseline_t2v, baseline_v2t)))
    print(_format("recorded", *_metrics(dump, dump["t2v"], dump["v2t"])))
    print(
        f"{'top-1 repaired/broken':<28}"
        f" {_flip_report('T2V', baseline_t2v, dump['t2v'])}"
        f" | {_flip_report('V2T', baseline_v2t.T, dump['v2t'].T)}"
    )
    print(
        f"{'true pair inside Top-R':<28}"
        f" T2V {100 * candidate_recall_ceiling(np.isfinite(dump['t2v'])):5.1f}"
        f" | V2T {100 * candidate_recall_ceiling(np.isfinite(dump['v2t']).T):5.1f}"
        "   (perfect-rerank ceiling)"
    )

    permuted_r1 = {"t2v": [], "v2t": []}
    for seed in range(arguments.permutations):
        # Each direction ranks along its own axis, so each needs its candidate
        # lists shuffled along that axis; one shared permutation would leave the
        # other direction's lists partly intact.
        t2v, _ = replay_rerank(
            dump,
            permute_probability_within_candidates(
                dump["probability"], np.isfinite(dump["t2v"]), seed=seed, axis=1
            ),
            coefficient,
        )
        _, v2t = replay_rerank(
            dump,
            permute_probability_within_candidates(
                dump["probability"], np.isfinite(dump["v2t"]), seed=seed, axis=0
            ),
            coefficient,
        )
        t2v_metrics, v2t_metrics = _metrics(dump, t2v, v2t)
        permuted_r1["t2v"].append(t2v_metrics["R1"])
        permuted_r1["v2t"].append(v2t_metrics["R1"])
        if seed == 0:
            print(_format("permuted (seed 0)", t2v_metrics, v2t_metrics))
    print(
        f"{'permuted R@1 over ' + str(arguments.permutations) + ' seeds':<28}"
        f" T2V {np.mean(permuted_r1['t2v']):5.2f} +- {np.std(permuted_r1['t2v']):.2f}"
        f" | V2T {np.mean(permuted_r1['v2t']):5.2f} +- {np.std(permuted_r1['v2t']):.2f}"
    )

    print("\n--- does u_pair predict correctness? ---")
    metric_t2v, metric_v2t = _build_rspr_metric_matrices(
        dump["t2v"], dump["v2t"], dump["recall"]
    )
    for name, ranking, direction_uncertainty in (
        ("T2V", metric_t2v, dump["uncertainty"]),
        ("V2T", metric_v2t.T, dump["uncertainty"].T),
    ):
        values, correct = top1_uncertainty_and_correctness(ranking, direction_uncertainty)
        usable = np.isfinite(values)
        # High uncertainty should mark the failures, so score the "wrong" class.
        auc = rank_auc(values[usable], ~correct[usable])
        print(
            f"{name} top-1 u_pair -> incorrect: AUC {auc:.4f}"
            f" (correct mean {values[usable & correct].mean():.3e},"
            f" wrong mean {values[usable & ~correct].mean():.3e})"
        )

    matched = np.equal.outer(np.arange(text_count), np.arange(video_count))
    pair_auc = rank_auc(dump["uncertainty"][scored], ~matched[scored])
    print(
        f"pair-level u_pair -> mismatched: AUC {pair_auc:.4f}"
        f" over {scored.sum()} scored pairs ({matched[scored].sum()} true pairs)"
    )


if __name__ == "__main__":
    main()
