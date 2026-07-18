#!/usr/bin/env python3
"""Build a deterministic, internal-val-only audit packet for P1 errors.

The packet contrasts three matched groups from frozen checkpoint diagnostics:
stable confident errors, stable uncertain errors, and stable correct controls.
It never loads JSFusion, trains, or changes a model/checkpoint.  Heuristic
labels are review prompts, not ground-truth semantic annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT / "cache_dir/p1_diagnostics"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/val.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache_dir/tqfs/msrvtt_trusted_v1_f1_m8_r224"
DEFAULT_KEYS = (
    "pytorch_model_bin_1",
    "pytorch_model_bin_2",
    "pytorch_model_bin_3",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--diagnostic-dir", default=str(DEFAULT_DIAGNOSTIC_DIR))
    parser.add_argument("--val-csv", default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--tqfs-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-key", action="append", default=[])
    parser.add_argument("--confident-margin", type=float, default=2.0)
    parser.add_argument("--confident-entropy", type=float, default=0.1)
    parser.add_argument("--uncertain-margin", type=float, default=1.0)
    parser.add_argument("--uncertain-entropy", type=float, default=0.2)
    return parser.parse_args(argv)


def read_tsv(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    indexed = {int(row["query_index"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate query_index in {path}")
    return indexed


def read_caption_groups(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("video_id", "")).strip()
            caption = str(row.get("sentence", "")).strip()
            if not video_id or not caption:
                raise ValueError("internal-val CSV requires video_id and sentence")
            groups[video_id].append(caption)
    if not groups:
        raise ValueError(f"no caption groups in {path}")
    return dict(groups)


def _tokens(text: str) -> set[str]:
    return {token for token in str(text).lower().replace("'", " ").split() if token}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_caption_overlap(query: str, candidate_captions: Sequence[str]) -> tuple[float, str]:
    scored = [(_jaccard(query, caption), caption) for caption in candidate_captions]
    return max(scored, key=lambda pair: (pair[0], pair[1]))


def length_bucket(row: dict) -> str:
    count = int(row["caption_word_count"])
    if count <= 4:
        return "1-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17+"


def _stable_rows(rows_by_key: dict[str, dict[int, dict]]) -> list[tuple[int, list[dict]]]:
    keys = list(rows_by_key)
    common = set.intersection(*(set(rows_by_key[key]) for key in keys))
    return [(index, [rows_by_key[key][index] for key in keys]) for index in sorted(common)]


def select_groups(
    rows_by_key: dict[str, dict[int, dict]],
    confident_margin: float,
    confident_entropy: float,
    uncertain_margin: float,
    uncertain_entropy: float,
) -> tuple[
    list[tuple[int, list[dict]]],
    list[tuple[int, list[dict]]],
    list[tuple[int, list[dict]]],
]:
    """Return stable confident errors, length-matched uncertain and correct controls."""

    stable = _stable_rows(rows_by_key)
    confident = [
        pair
        for pair in stable
        if all(row["correct"] == "False" for row in pair[1])
        and len({row["pred_video_id"] for row in pair[1]}) == 1
        and all(float(row["top1_top2_margin"]) >= confident_margin for row in pair[1])
        and all(float(row["row_entropy"]) <= confident_entropy for row in pair[1])
    ]
    if not confident:
        raise ValueError("no stable confident errors satisfy the requested thresholds")

    uncertain_candidates = [
        pair
        for pair in stable
        if all(row["correct"] == "False" for row in pair[1])
        and all(float(row["top1_top2_margin"]) <= uncertain_margin for row in pair[1])
        and all(float(row["row_entropy"]) >= uncertain_entropy for row in pair[1])
    ]
    correct_candidates = [
        pair for pair in stable if all(row["correct"] == "True" for row in pair[1])
    ]
    return confident, _match_controls(confident, uncertain_candidates), _match_controls(confident, correct_candidates)


def _match_controls(
    reference: Sequence[tuple[int, list[dict]]],
    candidates: Sequence[tuple[int, list[dict]]],
) -> list[tuple[int, list[dict]]]:
    needed = Counter(length_bucket(rows[-1]) for _, rows in reference)
    by_bucket: dict[str, list[tuple[int, list[dict]]]] = defaultdict(list)
    for pair in candidates:
        by_bucket[length_bucket(pair[1][-1])].append(pair)
    selected = []
    for bucket in sorted(needed):
        pool = sorted(by_bucket[bucket], key=lambda pair: pair[0])
        if len(pool) < needed[bucket]:
            raise ValueError(
                f"insufficient {bucket} controls: need={needed[bucket]} got={len(pool)}"
            )
        selected.extend(pool[: needed[bucket]])
    return sorted(selected, key=lambda pair: pair[0])


def frame_stats(cache_dir: Path, video_id: str) -> dict[str, float | int]:
    path = cache_dir / f"{video_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing cached video for audit: {path}")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError(f"invalid cached frames at {path}: shape={frames.shape}")
    temporal = (
        float(np.abs(frames[1:] - frames[:-1]).mean()) if frames.shape[0] > 1 else 0.0
    )
    return {
        "frame_count": int(frames.shape[0]),
        "frame_spatial_std": round(float(frames.reshape(frames.shape[0], -1).std(axis=1).mean()), 6),
        "temporal_delta": round(temporal, 6),
    }


def rule_hypothesis(row: dict, lexical_overlap: float) -> str:
    if int(row["caption_word_count"]) <= 4:
        return "short_query_ambiguity"
    if lexical_overlap >= 0.3:
        return "lexical_semantic_neighbor"
    if float(row["gt_rank_mean"]) <= 3:
        return "near_miss_visual_or_instance_confusion"
    return "cross_modal_pair_mismatch_candidate"


def build_audit_rows(
    group_name: str,
    pairs: Sequence[tuple[int, list[dict]]],
    caption_groups: dict[str, list[str]],
    cache_dir: Path,
) -> list[dict]:
    rows = []
    for query_index, checkpoint_rows in pairs:
        latest = checkpoint_rows[-1]
        query_video_id = latest["query_video_id"]
        predicted_video_id = latest["pred_video_id"]
        if query_video_id not in caption_groups or predicted_video_id not in caption_groups:
            raise ValueError("audit video missing from internal-val caption groups")
        overlap, nearest_caption = best_caption_overlap(
            latest["query_caption"], caption_groups[predicted_video_id]
        )
        margins = [float(row["top1_top2_margin"]) for row in checkpoint_rows]
        entropies = [float(row["row_entropy"]) for row in checkpoint_rows]
        ranks = [int(row["gt_rank"]) for row in checkpoint_rows]
        rows.append(
            {
                "audit_group": group_name,
                "query_index": query_index,
                "query_video_id": query_video_id,
                "query_caption": latest["query_caption"],
                "caption_word_count": int(latest["caption_word_count"]),
                "length_bucket": length_bucket(latest),
                "pred_video_id": predicted_video_id,
                "pred_caption_with_max_lexical_overlap": nearest_caption,
                "pred_caption_max_jaccard": round(overlap, 6),
                "gt_rank_mean": round(float(np.mean(ranks)), 6),
                "gt_rank_min": min(ranks),
                "gt_rank_max": max(ranks),
                "gt_ranks_by_checkpoint": json.dumps(ranks),
                "margin_mean": round(float(np.mean(margins)), 6),
                "margin_min": round(float(np.min(margins)), 6),
                "entropy_mean": round(float(np.mean(entropies)), 6),
                "entropy_max": round(float(np.max(entropies)), 6),
                "gt_frame_stats": json.dumps(frame_stats(cache_dir, query_video_id), sort_keys=True),
                "pred_frame_stats": json.dumps(frame_stats(cache_dir, predicted_video_id), sort_keys=True),
                "rule_hypothesis": rule_hypothesis(
                    {**latest, "gt_rank_mean": float(np.mean(ranks))}, overlap
                ),
                "human_label": "",
                "human_notes": "",
            }
        )
    return rows


def summarize_groups(groups: dict[str, list[dict]]) -> dict:
    result = {}
    for name, rows in groups.items():
        hypotheses = Counter(row["rule_hypothesis"] for row in rows)
        result[name] = {
            "n": len(rows),
            "length_buckets": dict(sorted(Counter(row["length_bucket"] for row in rows).items())),
            "rule_hypotheses": dict(sorted(hypotheses.items())),
            "mean_pred_caption_jaccard": round(
                float(np.mean([float(row["pred_caption_max_jaccard"]) for row in rows])), 6
            ),
            "mean_gt_rank": round(float(np.mean([float(row["gt_rank_mean"]) for row in rows])), 6),
            "mean_margin": round(float(np.mean([float(row["margin_mean"]) for row in rows])), 6),
            "mean_entropy": round(float(np.mean([float(row["entropy_mean"]) for row in rows])), 6),
            "candidate_ceiling_all_checkpoints": {
                f"top_{cutoff}": round(
                    sum(
                        all(rank <= cutoff for rank in json.loads(row["gt_ranks_by_checkpoint"]))
                        for row in rows
                    )
                    / len(rows),
                    6,
                )
                for cutoff in (3, 5, 10, 20)
            },
        }
    return result


def write_tsv(path: Path, rows: Sequence[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args, summary: dict):
    lines = [
        "# P1 Stable Error Audit Packet",
        "",
        "- Scope: trusted-v1 internal-val only; no JSFusion rows are loaded.",
        f"- Diagnostics: `{args.diagnostic_dir}`",
        f"- Checkpoints: `{', '.join(args.checkpoint_key or DEFAULT_KEYS)}`",
        "- `rule_hypothesis` is a review prompt, not a ground-truth label.",
        "",
        "## Group summary",
        "",
        "| Group | N | Mean GT rank | Mean margin | Mean entropy | Mean pred-caption Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                name,
                values["n"],
                values["mean_gt_rank"],
                values["mean_margin"],
                values["mean_entropy"],
                values["mean_pred_caption_jaccard"],
            )
        )
    lines.extend(
        [
            "",
            "## Strict candidate ceiling",
            "",
            "The following ratios require the ground-truth video to be within the listed top-k for **every** frozen checkpoint.",
            "",
            "| Group | Top-3 | Top-5 | Top-10 | Top-20 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in summary.items():
        ceiling = values["candidate_ceiling_all_checkpoints"]
        lines.append(
            "| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
                name,
                ceiling["top_3"],
                ceiling["top_5"],
                ceiling["top_10"],
                ceiling["top_20"],
            )
        )
    lines.extend(
        [
            "",
            "## Required review rule",
            "",
            "- Complete `human_label` only with one of: `short_query_ambiguity`, `semantic_neighbor`, `action_or_temporal`, `visual_near_duplicate`, `frame_coverage`, `annotation_ambiguity`, or `other`.",
            "- Do not treat heuristic categories or JSFusion results as supervision or pseudo-labels.",
            "- This satisfies P1's three-checkpoint screening rule for a bounded candidate-correction hypothesis; it is not semantic supervision or a causal claim. Any P2/P3 training must be preregistered and later repeated with independent seeds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    keys = tuple(args.checkpoint_key) or DEFAULT_KEYS
    diagnostic_dir = Path(args.diagnostic_dir)
    rows_by_key = {
        key: read_tsv(diagnostic_dir / key / "t2v_rows.tsv") for key in keys
    }
    caption_groups = read_caption_groups(Path(args.val_csv))
    confident, uncertain, correct = select_groups(
        rows_by_key,
        args.confident_margin,
        args.confident_entropy,
        args.uncertain_margin,
        args.uncertain_entropy,
    )
    cache_dir = Path(args.tqfs_cache_dir)
    groups = {
        "persistent_confident_error": build_audit_rows(
            "persistent_confident_error", confident, caption_groups, cache_dir
        ),
        "stable_uncertain_error_control": build_audit_rows(
            "stable_uncertain_error_control", uncertain, caption_groups, cache_dir
        ),
        "stable_correct_control": build_audit_rows(
            "stable_correct_control", correct, caption_groups, cache_dir
        ),
    }
    summary = summarize_groups(groups)
    output_dir = Path(args.output_dir or (diagnostic_dir / "stable_error_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in groups.items():
        write_tsv(output_dir / f"{name}.tsv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "report.md", args, summary)
    print(f"[p1-audit] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
