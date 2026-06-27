#!/usr/bin/env python3
"""Audit and clean MSRVTT query hard-negative mappings.

The generated hard-negative map can contain cross-video near-duplicates.  This
script diagnoses those pairs and writes a conservative filtered map that the
existing hard-negative batch sampler can consume directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_msrvtt_hard_negatives import (  # noqa: E402
    CaptionSample,
    STOPWORDS,
    TOKEN_RE,
    load_caption_samples,
    read_train_video_ids,
    write_json,
)


@dataclass(frozen=True)
class AuditConfig:
    min_token_len: int = 2
    keep_stopwords: bool = False
    risk_jaccard: float = 0.6
    risk_overlap: float = 0.8
    risk_dense_score: float = 0.95
    weak_jaccard: float = 0.05
    weak_dense_score: float = 0.15
    clean_max_jaccard: float = 0.8
    clean_max_overlap: float = 0.9
    clean_max_dense_score: float = 0.95
    max_per_hard_index: int = 80
    max_per_hard_video: int = 120
    sample_limit_per_bucket: int = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and clean an MSRVTT hard-negative mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="cache_dir/hard_negatives/msrvtt_train_hardneg.json",
        help="Input hard-negative JSON built by build_msrvtt_hard_negatives.py.",
    )
    parser.add_argument(
        "--output_clean",
        default="cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        help="Filtered hard-negative JSON output.",
    )
    parser.add_argument(
        "--report",
        default="cache_dir/hard_negatives/msrvtt_train_hardneg_audit.md",
        help="Markdown audit report output.",
    )
    parser.add_argument(
        "--samples_csv",
        default="cache_dir/hard_negatives/msrvtt_train_hardneg_audit_samples.csv",
        help="CSV file with representative pairs for manual review.",
    )
    parser.add_argument(
        "--train_csv",
        default="/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv",
        help="MSRVTT train CSV used to reconstruct caption sample order.",
    )
    parser.add_argument(
        "--data_path",
        default="/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json",
        help="MSRVTT annotation JSON used to reconstruct caption sample order.",
    )
    parser.add_argument(
        "--unfold_sentences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the sample granularity used by the hard-negative builder.",
    )
    parser.add_argument("--min_token_len", type=int, default=2)
    parser.add_argument("--keep_stopwords", action="store_true")
    parser.add_argument("--risk_jaccard", type=float, default=0.6)
    parser.add_argument("--risk_overlap", type=float, default=0.8)
    parser.add_argument("--risk_dense_score", type=float, default=0.95)
    parser.add_argument("--weak_jaccard", type=float, default=0.05)
    parser.add_argument("--weak_dense_score", type=float, default=0.15)
    parser.add_argument("--clean_max_jaccard", type=float, default=0.8)
    parser.add_argument("--clean_max_overlap", type=float, default=0.9)
    parser.add_argument("--clean_max_dense_score", type=float, default=0.95)
    parser.add_argument("--max_per_hard_index", type=int, default=80)
    parser.add_argument("--max_per_hard_video", type=int, default=120)
    parser.add_argument("--sample_limit_per_bucket", type=int, default=100)
    parser.add_argument(
        "--include_captions_in_clean",
        action="store_true",
        help="Store anchor_caption/hard_caption in the clean JSON for inspection.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        min_token_len=args.min_token_len,
        keep_stopwords=args.keep_stopwords,
        risk_jaccard=args.risk_jaccard,
        risk_overlap=args.risk_overlap,
        risk_dense_score=args.risk_dense_score,
        weak_jaccard=args.weak_jaccard,
        weak_dense_score=args.weak_dense_score,
        clean_max_jaccard=args.clean_max_jaccard,
        clean_max_overlap=args.clean_max_overlap,
        clean_max_dense_score=args.clean_max_dense_score,
        max_per_hard_index=args.max_per_hard_index,
        max_per_hard_video=args.max_per_hard_video,
        sample_limit_per_bucket=args.sample_limit_per_bucket,
    )


def normalize_caption(text: str) -> str:
    return " ".join(m.group(0) for m in TOKEN_RE.finditer(text.lower()))


def caption_token_set(text: str, config: AuditConfig) -> set[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if config.min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= config.min_token_len]
    if not config.keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


def pair_text_metrics(anchor_caption: str, hard_caption: str, config: AuditConfig) -> dict:
    anchor_tokens = caption_token_set(anchor_caption, config)
    hard_tokens = caption_token_set(hard_caption, config)
    intersection = len(anchor_tokens & hard_tokens)
    union = len(anchor_tokens | hard_tokens)
    min_size = min(len(anchor_tokens), len(hard_tokens))
    return {
        "exact_caption": normalize_caption(anchor_caption) == normalize_caption(hard_caption),
        "jaccard": intersection / union if union else 0.0,
        "overlap": intersection / min_size if min_size else 0.0,
        "shared_tokens": intersection,
        "anchor_tokens": len(anchor_tokens),
        "hard_tokens": len(hard_tokens),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "p99": None}
    return {
        "mean": sum(vals) / len(vals),
        "p10": percentile(vals, 0.10),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p99": percentile(vals, 0.99),
    }


def as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def content_rejection_reasons(row: dict, config: AuditConfig) -> list[str]:
    reasons = []
    if row["same_video"]:
        reasons.append("same_video")
    if row["exact_caption"]:
        reasons.append("exact_caption")
    if row["jaccard"] >= config.clean_max_jaccard:
        reasons.append("high_jaccard")
    if row["overlap"] >= config.clean_max_overlap:
        reasons.append("high_overlap")
    if row["dense_score"] >= config.clean_max_dense_score:
        reasons.append("high_dense_score")
    if row["invalid_index"]:
        reasons.append("invalid_index")
    return reasons


def row_risk_bucket(row: dict, config: AuditConfig) -> str:
    if row["exact_caption"]:
        return "exact"
    if (
        row["jaccard"] >= config.risk_jaccard
        or row["overlap"] >= config.risk_overlap
        or row["dense_score"] >= config.risk_dense_score
    ):
        return "high_risk"
    if row["jaccard"] <= config.weak_jaccard and row["dense_score"] < config.weak_dense_score:
        return "weak"
    return "normal"


def build_rows(samples: list[CaptionSample], mapping: dict[str, dict], config: AuditConfig) -> list[dict]:
    rows = []
    n = len(samples)
    for key, item in mapping.items():
        anchor_idx = as_int(item.get("anchor_index", key))
        hard_idx = as_int(item.get("hard_index"))
        invalid = not (0 <= anchor_idx < n and 0 <= hard_idx < n)
        if invalid:
            anchor_caption = ""
            hard_caption = ""
            anchor_video_id = str(item.get("anchor_video_id", ""))
            hard_video_id = str(item.get("hard_video_id", ""))
            text_metrics = {
                "exact_caption": False,
                "jaccard": 0.0,
                "overlap": 0.0,
                "shared_tokens": 0,
                "anchor_tokens": 0,
                "hard_tokens": 0,
            }
        else:
            anchor = samples[anchor_idx]
            hard = samples[hard_idx]
            anchor_caption = anchor.caption
            hard_caption = hard.caption
            anchor_video_id = anchor.video_id
            hard_video_id = hard.video_id
            text_metrics = pair_text_metrics(anchor_caption, hard_caption, config)

        row = {
            "key": str(key),
            "item": item,
            "anchor_index": anchor_idx,
            "hard_index": hard_idx,
            "anchor_video_id": anchor_video_id,
            "hard_video_id": hard_video_id,
            "anchor_caption": anchor_caption,
            "hard_caption": hard_caption,
            "dense_score": as_float(item.get("dense_score")),
            "bm25_score": as_float(item.get("bm25_score")),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "same_video": anchor_video_id == hard_video_id,
            "invalid_index": invalid,
            **text_metrics,
        }
        row["risk_bucket"] = row_risk_bucket(row, config)
        row["reasons"] = content_rejection_reasons(row, config)
        rows.append(row)
    rows.sort(key=lambda row: row["anchor_index"])
    return rows


def clean_item(row: dict, include_captions: bool = False) -> dict:
    item = dict(row["item"])
    if include_captions:
        item["anchor_caption"] = row["anchor_caption"]
        item["hard_caption"] = row["hard_caption"]
    else:
        item.pop("anchor_caption", None)
        item.pop("hard_caption", None)
    return item


def audit_and_clean_mapping(
    samples: list[CaptionSample],
    mapping: dict[str, dict],
    config: AuditConfig,
    include_captions_in_clean: bool = False,
) -> tuple[dict[str, dict], dict, list[dict]]:
    rows = build_rows(samples, mapping, config)
    reason_counts: Counter[str] = Counter()
    clean_rows = []

    eligible = [row for row in rows if not row["reasons"]]
    rejected_keys = set()
    for row in rows:
        if row["reasons"]:
            rejected_keys.add(row["key"])
            reason_counts.update(row["reasons"])

    def cap_sort_key(row: dict) -> tuple[float, float, float, int]:
        return (
            row["overlap"],
            row["jaccard"],
            row["dense_score"],
            row["anchor_index"],
        )

    hard_index_counts: Counter[int] = Counter()
    hard_video_counts: Counter[str] = Counter()
    for row in sorted(eligible, key=cap_sort_key):
        if config.max_per_hard_index > 0 and hard_index_counts[row["hard_index"]] >= config.max_per_hard_index:
            row["reasons"] = ["cap_hard_index"]
            reason_counts.update(row["reasons"])
            rejected_keys.add(row["key"])
            continue
        if config.max_per_hard_video > 0 and hard_video_counts[row["hard_video_id"]] >= config.max_per_hard_video:
            row["reasons"] = ["cap_hard_video"]
            reason_counts.update(row["reasons"])
            rejected_keys.add(row["key"])
            continue
        hard_index_counts[row["hard_index"]] += 1
        hard_video_counts[row["hard_video_id"]] += 1
        clean_rows.append(row)

    clean_rows.sort(key=lambda row: row["anchor_index"])
    clean_mapping = {
        row["key"]: clean_item(row, include_captions=include_captions_in_clean)
        for row in clean_rows
    }
    kept_key_set = set(clean_mapping)

    for row in rows:
        row["bucket"] = "removed" if row["key"] in rejected_keys else row["risk_bucket"]

    original_hard_indices = Counter(row["hard_index"] for row in rows if not row["invalid_index"])
    original_hard_videos = Counter(row["hard_video_id"] for row in rows if not row["invalid_index"])

    summary = {
        "audit": {
            "num_samples": len(samples),
            "mapping_size": len(rows),
            "same_video_pairs": sum(1 for row in rows if row["same_video"]),
            "invalid_index_pairs": sum(1 for row in rows if row["invalid_index"]),
            "exact_caption_pairs": sum(1 for row in rows if row["exact_caption"]),
            "high_risk_pairs": sum(1 for row in rows if row["risk_bucket"] == "high_risk"),
            "weak_pairs": sum(1 for row in rows if row["risk_bucket"] == "weak"),
            "unique_hard_indices": len(original_hard_indices),
            "unique_hard_videos": len(original_hard_videos),
            "max_hard_index_count": max(original_hard_indices.values(), default=0),
            "max_hard_video_count": max(original_hard_videos.values(), default=0),
            "dense_score": distribution(row["dense_score"] for row in rows),
            "bm25_score": distribution(row["bm25_score"] for row in rows),
            "jaccard": distribution(row["jaccard"] for row in rows),
            "overlap": distribution(row["overlap"] for row in rows),
        },
        "clean": {
            "kept_mapping_size": len(clean_mapping),
            "removed_total": len(rows) - len(clean_mapping),
            "removal_rate": (len(rows) - len(clean_mapping)) / max(1, len(rows)),
            "removal_reasons": dict(sorted(reason_counts.items())),
            "exact_caption_pairs": sum(1 for row in rows if row["key"] in kept_key_set and row["exact_caption"]),
            "high_risk_pairs": sum(1 for row in rows if row["key"] in kept_key_set and row["risk_bucket"] == "high_risk"),
            "weak_pairs": sum(1 for row in rows if row["key"] in kept_key_set and row["risk_bucket"] == "weak"),
            "unique_hard_indices": len(hard_index_counts),
            "unique_hard_videos": len(hard_video_counts),
            "max_hard_index_count": max(hard_index_counts.values(), default=0),
            "max_hard_video_count": max(hard_video_counts.values(), default=0),
        },
        "config": config.__dict__,
    }
    return clean_mapping, summary, rows


def load_hard_negative_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("mapping"), dict):
        raise ValueError(f"Input file has no mapping object: {path}")
    return obj


def format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_report(path: str, input_obj: dict, clean_obj: dict, summary: dict, rows: list[dict]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    audit = summary["audit"]
    clean = summary["clean"]
    config = summary["config"]
    top_hard_indices = Counter(row["hard_index"] for row in rows if not row["invalid_index"]).most_common(10)
    top_hard_videos = Counter(row["hard_video_id"] for row in rows if not row["invalid_index"]).most_common(10)

    lines = [
        "# MSRVTT Hard Negative Audit",
        "",
        "## Inputs",
        "",
        f"- Input mapping: `{input_obj['meta'].get('source_path', 'unknown')}`",
        f"- Clean mapping: `{clean_obj['meta'].get('output_path', 'unknown')}`",
        f"- Clean rules: `dense < {config['clean_max_dense_score']}`, "
        f"`overlap < {config['clean_max_overlap']}`, "
        f"`jaccard < {config['clean_max_jaccard']}`, "
        f"`max_per_hard_index={config['max_per_hard_index']}`, "
        f"`max_per_hard_video={config['max_per_hard_video']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Original mapping size | {audit['mapping_size']} |",
        f"| Clean mapping size | {clean['kept_mapping_size']} |",
        f"| Removed total | {clean['removed_total']} |",
        f"| Removal rate | {clean['removal_rate']:.2%} |",
        f"| Exact caption pairs | {audit['exact_caption_pairs']} |",
        f"| High-risk pairs | {audit['high_risk_pairs']} |",
        f"| Weak pairs | {audit['weak_pairs']} |",
        f"| Clean exact caption pairs | {clean['exact_caption_pairs']} |",
        f"| Clean high-risk pairs | {clean['high_risk_pairs']} |",
        f"| Clean weak pairs | {clean['weak_pairs']} |",
        f"| Same-video pairs | {audit['same_video_pairs']} |",
        f"| Invalid-index pairs | {audit['invalid_index_pairs']} |",
        f"| Max hard-index count before / after | {audit['max_hard_index_count']} / {clean['max_hard_index_count']} |",
        f"| Max hard-video count before / after | {audit['max_hard_video_count']} / {clean['max_hard_video_count']} |",
            "",
            "## Removal Reasons",
            "",
            "Reason counts are non-exclusive because one pair can match multiple filters.",
            "",
            "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in clean["removal_reasons"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Score Distributions",
            "",
            "| Field | Mean | P10 | P50 | P90 | P99 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field in ["dense_score", "bm25_score", "jaccard", "overlap"]:
        dist = audit[field]
        lines.append(
            f"| {field} | {format_float(dist['mean'])} | {format_float(dist['p10'])} | "
            f"{format_float(dist['p50'])} | {format_float(dist['p90'])} | {format_float(dist['p99'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Reused Hard Captions",
            "",
            "| hard_index | Count | hard_video_id | hard_caption |",
            "|---:|---:|---|---|",
        ]
    )
    row_by_hard = {row["hard_index"]: row for row in rows if not row["invalid_index"]}
    for hard_idx, count in top_hard_indices:
        row = row_by_hard.get(hard_idx)
        if row:
            lines.append(
                f"| {hard_idx} | {count} | {row['hard_video_id']} | {escape_md(row['hard_caption'])} |"
            )

    lines.extend(
        [
            "",
            "## Top Reused Hard Videos",
            "",
            "| hard_video_id | Count |",
            "|---|---:|",
        ]
    )
    for video_id, count in top_hard_videos:
        lines.append(f"| {video_id} | {count} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def sample_rows(rows: list[dict], limit: int) -> list[dict]:
    buckets = ["removed", "exact", "high_risk", "weak", "normal"]
    selected = []
    seen = set()
    for bucket in buckets:
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        bucket_rows.sort(
            key=lambda row: (
                -row["overlap"],
                -row["jaccard"],
                -row["dense_score"],
                row["anchor_index"],
            )
        )
        for row in bucket_rows[:limit]:
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            selected.append(row)
    return selected


def write_samples_csv(path: str, rows: list[dict], limit: int) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "bucket",
        "reasons",
        "anchor_index",
        "hard_index",
        "anchor_video_id",
        "hard_video_id",
        "dense_score",
        "bm25_score",
        "jaccard",
        "overlap",
        "anchor_caption",
        "hard_caption",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sample_rows(rows, limit):
            writer.writerow(
                {
                    "bucket": row["bucket"],
                    "reasons": ";".join(row["reasons"]),
                    "anchor_index": row["anchor_index"],
                    "hard_index": row["hard_index"],
                    "anchor_video_id": row["anchor_video_id"],
                    "hard_video_id": row["hard_video_id"],
                    "dense_score": f"{row['dense_score']:.8f}",
                    "bm25_score": f"{row['bm25_score']:.8f}",
                    "jaccard": f"{row['jaccard']:.8f}",
                    "overlap": f"{row['overlap']:.8f}",
                    "anchor_caption": row["anchor_caption"],
                    "hard_caption": row["hard_caption"],
                }
            )


def build_clean_object(
    input_obj: dict,
    input_path: str,
    output_path: str,
    clean_mapping: dict[str, dict],
    summary: dict,
) -> dict:
    meta = dict(input_obj.get("meta", {}))
    meta.update(
        {
            "task": "msrvtt_query_hard_negative_clean",
            "cleaned_from": os.path.abspath(input_path),
            "output_path": os.path.abspath(output_path),
            "audit_config": summary["config"],
        }
    )
    stats = dict(input_obj.get("stats", {}))
    stats.update(
        {
            "clean_mapping_size": len(clean_mapping),
            "clean_removed_total": summary["clean"]["removed_total"],
            "clean_removal_rate": summary["clean"]["removal_rate"],
            "clean_removal_reasons": summary["clean"]["removal_reasons"],
            "clean_exact_caption_pairs": summary["clean"]["exact_caption_pairs"],
            "clean_high_risk_pairs": summary["clean"]["high_risk_pairs"],
            "clean_weak_pairs": summary["clean"]["weak_pairs"],
            "clean_unique_hard_indices": summary["clean"]["unique_hard_indices"],
            "clean_unique_hard_videos": summary["clean"]["unique_hard_videos"],
            "clean_max_hard_index_count": summary["clean"]["max_hard_index_count"],
            "clean_max_hard_video_count": summary["clean"]["max_hard_video_count"],
        }
    )
    return {
        "meta": meta,
        "stats": stats,
        "audit": summary,
        "mapping": clean_mapping,
    }


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    input_obj = load_hard_negative_json(args.input)
    input_obj.setdefault("meta", {})["source_path"] = os.path.abspath(args.input)
    train_video_ids = read_train_video_ids(args.train_csv)
    samples = load_caption_samples(
        args.data_path,
        train_video_ids,
        unfold_sentences=args.unfold_sentences,
        limit_samples=0,
    )
    clean_mapping, summary, rows = audit_and_clean_mapping(
        samples=samples,
        mapping=input_obj["mapping"],
        config=config,
        include_captions_in_clean=args.include_captions_in_clean,
    )
    clean_obj = build_clean_object(
        input_obj=input_obj,
        input_path=args.input,
        output_path=args.output_clean,
        clean_mapping=clean_mapping,
        summary=summary,
    )
    write_json(clean_obj, args.output_clean)
    write_report(args.report, input_obj, clean_obj, summary, rows)
    write_samples_csv(args.samples_csv, rows, config.sample_limit_per_bucket)

    audit = summary["audit"]
    clean = summary["clean"]
    print(
        "[hardneg-audit] "
        f"mapping={audit['mapping_size']} clean={clean['kept_mapping_size']} "
        f"removed={clean['removed_total']} ({clean['removal_rate']:.2%}) "
        f"exact={audit['exact_caption_pairs']} high_risk={audit['high_risk_pairs']} "
        f"max_hard_index={audit['max_hard_index_count']}->{clean['max_hard_index_count']} "
        f"max_hard_video={audit['max_hard_video_count']}->{clean['max_hard_video_count']}",
        flush=True,
    )
    print(f"[hardneg-audit] wrote clean map: {args.output_clean}", flush=True)
    print(f"[hardneg-audit] wrote report: {args.report}", flush=True)
    print(f"[hardneg-audit] wrote samples: {args.samples_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[hardneg-audit] interrupted", file=sys.stderr)
        raise
