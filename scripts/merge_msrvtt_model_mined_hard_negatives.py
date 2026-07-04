#!/usr/bin/env python3
"""Merge sharded MSRVTT model-mined hard-negative maps."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_msrvtt_model_mined_hard_negatives import (  # noqa: E402
    summarize_mapping,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded model-mined MSRVTT hard-negative JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard JSON files to merge.")
    parser.add_argument("--output", required=True, help="Merged hard-negative JSON output.")
    parser.add_argument("--report", default=None, help="Markdown report path. Defaults to '<output>.md'.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("mapping"), dict):
        raise ValueError(f"Input has no mapping object: {path}")
    return obj


def _anchor_index_for(key: str, item: dict, source_path: str) -> int:
    try:
        return int(item.get("anchor_index", key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid anchor_index in {source_path}: key={key!r}") from exc


def _num_samples_for(obj: dict) -> int:
    stats = obj.get("stats", {})
    if isinstance(stats, dict) and stats.get("num_samples") is not None:
        return int(stats["num_samples"])
    meta = obj.get("meta", {})
    if isinstance(meta, dict):
        start = int(meta.get("query_start", 0) or 0)
        end = int(meta.get("query_end", 0) or 0)
        if end > start:
            return end - start
    return len(obj.get("mapping", {}))


def _num_videos_for(objects: list[dict]) -> int:
    values = []
    for obj in objects:
        stats = obj.get("stats", {})
        if isinstance(stats, dict) and stats.get("num_train_videos") is not None:
            values.append(int(stats["num_train_videos"]))
    return max(values, default=0)


def merge_mapping_objects(objects: list[dict], source_paths: list[str]) -> dict:
    if len(objects) != len(source_paths):
        raise ValueError("objects and source_paths must have the same length")

    merged_mapping: dict[str, dict] = {}
    seen_sources: dict[int, str] = {}
    for obj, source_path in zip(objects, source_paths):
        for key, item in obj["mapping"].items():
            if not isinstance(item, dict):
                continue
            anchor_index = _anchor_index_for(str(key), item, source_path)
            if anchor_index in seen_sources:
                raise ValueError(
                    f"Duplicate anchor_index={anchor_index} in {source_path}; "
                    f"first seen in {seen_sources[anchor_index]}"
                )
            seen_sources[anchor_index] = source_path
            normalized = dict(item)
            normalized["anchor_index"] = anchor_index
            merged_mapping[str(anchor_index)] = normalized

    merged_mapping = {
        key: merged_mapping[key]
        for key in sorted(merged_mapping, key=lambda value: int(value))
    }
    num_samples = sum(_num_samples_for(obj) for obj in objects)
    num_videos = _num_videos_for(objects)
    stats = summarize_mapping(merged_mapping, num_samples=num_samples, num_videos=num_videos, elapsed_seconds=0.0)
    return {
        "meta": {
            "task": "msrvtt_model_mined_hard_negative_merged",
            "source_paths": list(source_paths),
            "source_count": len(source_paths),
            "source_metas": [obj.get("meta", {}) for obj in objects],
        },
        "stats": stats,
        "mapping": merged_mapping,
    }


def write_report(path: str | Path, merged: dict, elapsed_seconds: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = merged["stats"]
    lines = [
        "# Merged MSRVTT Model-Mined Hard Negative Map",
        "",
        f"- Source shards: {merged['meta']['source_count']}",
        f"- Mapping size: {stats['mapping_size']}",
        f"- Num samples: {stats['num_samples']}",
        f"- Unmapped: {stats['unmapped_count']} ({stats['unmapped_rate']:.2%})",
        f"- Max hard-video reuse: {stats['max_hard_video_count']}",
        f"- Elapsed: {elapsed_seconds:.1f}s",
        "",
        "## Sources",
        "",
    ]
    for source in merged["meta"]["source_paths"]:
        lines.append(f"- `{source}`")
    lines.extend(["", "## Summary", "", "| Metric | Value |", "|---|---:|"])
    for key in sorted(stats):
        lines.append(f"| {key} | {stats[key]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    start_time = time.time()
    objects = [load_json(path) for path in args.inputs]
    merged = merge_mapping_objects(objects, source_paths=list(args.inputs))
    elapsed = time.time() - start_time
    merged["stats"]["elapsed_seconds"] = round(elapsed, 3)
    merged["meta"]["output_path"] = os.path.abspath(args.output)
    write_json(merged, args.output)
    report_path = args.report or str(Path(args.output).with_suffix(".md"))
    write_report(report_path, merged, elapsed)
    print(
        f"[model-mined-merge] wrote {args.output} mapping={merged['stats']['mapping_size']} "
        f"sources={len(args.inputs)}",
        flush=True,
    )
    print(f"[model-mined-merge] wrote report {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
