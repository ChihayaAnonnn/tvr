#!/usr/bin/env python3
"""Read-only P1 diagnostics for one frozen trusted-v1 WTI baseline.

The diagnostic scores only the internal validation split.  It does not build a
test dataloader, backpropagate, or alter a checkpoint.  Outputs expose whether
WTI score, top-1/top-2 margin, entropy, caption length, and selected-frame
statistics already explain retrieval failures before any new mechanism is
considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = "/data2/hxj/data/MSRVTT"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "cache_dir/tqfs/msrvtt_trusted_v1_f1_m8_r224"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cache_dir/p1_diagnostics"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class ValidationItem:
    index: int
    video_id: str
    caption: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--train-csv",
        default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/train.csv"),
    )
    parser.add_argument(
        "--source-train-csv",
        default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_train.9k.csv",
    )
    parser.add_argument(
        "--test-csv",
        default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_JSFUSION_test.csv",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(PROJECT_ROOT / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"),
    )
    parser.add_argument(
        "--val-csv",
        default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/val.csv"),
    )
    parser.add_argument(
        "--data-path",
        default=f"{DEFAULT_DATA_ROOT}/annotation/MSRVTT_v2.json",
    )
    parser.add_argument(
        "--features-path",
        default=f"{DEFAULT_DATA_ROOT}/videos/compressed_videos/msrvtt_224_12fps/",
    )
    parser.add_argument("--tqfs-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--video-chunk-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available.")
    return parser.parse_args(argv)


def validate_inputs(args):
    """Validate trusted-v1 sources before reading internal-val rows."""

    from scripts.diagnose_msrvtt_hard_negative_runtime import (
        validate_trusted_diagnostic_inputs,
    )

    validate_trusted_diagnostic_inputs(args, scored_csv=args.val_csv)


def read_validation_items(csv_path: str) -> list[ValidationItem]:
    items = []
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("video_id", "")).strip()
            caption = str(row.get("sentence", "")).strip()
            if not video_id or not caption:
                raise ValueError("internal-val rows require non-empty video_id and sentence")
            items.append(ValidationItem(len(items), video_id, caption))
    if not items:
        raise ValueError(f"no validation items found in {csv_path}")
    return items


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text))]


def descending_rank(scores: Sequence[float], target_index: int) -> int:
    if target_index < 0 or target_index >= len(scores):
        raise IndexError(f"target_index={target_index} is out of range")
    target = float(scores[target_index])
    return 1 + sum(float(score) > target for score in scores)


def normalized_entropy(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return 0.0
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return entropy / math.log(values.size)


def _top_two(scores: Sequence[float]) -> tuple[int, int]:
    if len(scores) < 2:
        raise ValueError("at least two candidates are required")
    order = np.argsort(-np.asarray(scores), kind="stable")
    return int(order[0]), int(order[1])


def _row_statistics(scores: Sequence[float], target_index: int) -> dict[str, float | int | bool]:
    top1, top2 = _top_two(scores)
    top_k = min(10, len(scores))
    top_values = np.partition(np.asarray(scores), -top_k)[-top_k:]
    top1_score = float(scores[top1])
    gt_score = float(scores[target_index])
    return {
        "correct": top1 == target_index,
        "gt_rank": descending_rank(scores, target_index),
        "gt_logit": round(gt_score, 6),
        "top1_index": top1,
        "top1_logit": round(top1_score, 6),
        "top2_logit": round(float(scores[top2]), 6),
        "top1_top2_margin": round(top1_score - float(scores[top2]), 6),
        "gt_top1_gap": round(gt_score - top1_score, 6),
        "row_entropy": round(normalized_entropy(scores), 6),
        "top10_mean": round(float(np.mean(top_values)), 6),
    }


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return round(statistics.fmean(values), 6) if values else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = [bool(value) for value in values]
    return round(sum(values) / len(values), 6) if values else 0.0


def _caption_lookup(items: Sequence[ValidationItem]) -> dict[str, str]:
    captions = {}
    for item in items:
        captions.setdefault(item.video_id, item.caption)
    return captions


def build_t2v_rows(
    items: Sequence[ValidationItem],
    sim_matrix: np.ndarray,
    video_ids: Sequence[str],
    video_stats: dict[str, dict[str, float]],
) -> list[dict]:
    if sim_matrix.shape != (len(items), len(video_ids)):
        raise ValueError(f"unexpected T2V matrix shape={sim_matrix.shape}")
    column_by_video = {video_id: index for index, video_id in enumerate(video_ids)}
    if len(column_by_video) != len(video_ids):
        raise ValueError("candidate video IDs must be unique")
    captions = _caption_lookup(items)
    rows = []
    for item, scores in zip(items, sim_matrix):
        target_index = column_by_video[item.video_id]
        values = _row_statistics(scores, target_index)
        predicted_video_id = video_ids[int(values["top1_index"])]
        rows.append(
            {
                "query_index": item.index,
                "query_video_id": item.video_id,
                "query_caption": item.caption,
                "caption_word_count": len(tokenize(item.caption)),
                "pred_video_id": predicted_video_id,
                "pred_reference_caption": captions[predicted_video_id],
                "pred_frame_count": video_stats[predicted_video_id]["frame_count"],
                "pred_frame_spatial_std": video_stats[predicted_video_id]["frame_spatial_std"],
                "pred_temporal_delta": video_stats[predicted_video_id]["temporal_delta"],
                **values,
            }
        )
    return rows


def build_v2t_rows(
    items: Sequence[ValidationItem],
    sim_matrix: np.ndarray,
    video_ids: Sequence[str],
    video_stats: dict[str, dict[str, float]],
) -> list[dict]:
    """Aggregate each target video's captions with max, matching eval metrics."""

    groups: dict[str, list[int]] = defaultdict(list)
    for item in items:
        groups[item.video_id].append(item.index)
    target_video_ids = list(groups)
    if target_video_ids != list(video_ids):
        raise ValueError("validation row groups must follow candidate video order")
    captions = _caption_lookup(items)
    grouped_scores = np.stack(
        [sim_matrix[groups[video_id], :].max(axis=0) for video_id in target_video_ids],
        axis=0,
    )
    rows = []
    for query_index, query_video_id in enumerate(video_ids):
        values = _row_statistics(grouped_scores[:, query_index], query_index)
        predicted_video_id = target_video_ids[int(values["top1_index"])]
        rows.append(
            {
                "query_video_id": query_video_id,
                "pred_caption_video_id": predicted_video_id,
                "pred_reference_caption": captions[predicted_video_id],
                "query_frame_count": video_stats[query_video_id]["frame_count"],
                "query_frame_spatial_std": video_stats[query_video_id]["frame_spatial_std"],
                "query_temporal_delta": video_stats[query_video_id]["temporal_delta"],
                **values,
            }
        )
    return rows


def summarize_rows(rows: Sequence[dict], direction: str) -> dict[str, float | int]:
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    summary = {
        "direction": direction,
        "queries": len(rows),
        "r1": _rate(row["correct"] for row in rows) * 100,
        "error_count": len(wrong),
        "mean_gt_rank": _mean(row["gt_rank"] for row in rows),
        "correct_top1_top2_margin": _mean(row["top1_top2_margin"] for row in correct),
        "wrong_top1_top2_margin": _mean(row["top1_top2_margin"] for row in wrong),
        "correct_row_entropy": _mean(row["row_entropy"] for row in correct),
        "wrong_row_entropy": _mean(row["row_entropy"] for row in wrong),
        "correct_top10_mean": _mean(row["top10_mean"] for row in correct),
        "wrong_top10_mean": _mean(row["top10_mean"] for row in wrong),
    }
    if direction == "T2V":
        short = [row for row in rows if int(row["caption_word_count"]) <= 4]
        summary["short_caption_queries"] = len(short)
        summary["short_caption_r1"] = _rate(row["correct"] for row in short) * 100
    return summary


def build_task_args(args):
    from scripts.diagnose_msrvtt_hard_negative_runtime import build_task_args as build

    task_args = build(args, args.checkpoint)
    task_args.strategy = 1
    task_args.tqfs_cache_dir = args.tqfs_cache_dir
    return task_args


def compute_similarity_and_video_stats(args, device):
    import torch
    from torch.utils.data import DataLoader

    from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
    from main_task_retrieval import init_model
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

    task_args = build_task_args(args)
    tokenizer = ClipTokenizer()
    dataset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=task_args.max_words,
        feature_framerate=task_args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=task_args.max_frames,
        frame_order=task_args.eval_frame_order,
        slice_framepos=task_args.slice_framepos,
        tqfs_cache_dir=args.tqfs_cache_dir,
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)
    model = init_model(task_args, device, 1, 0).eval()
    if device.type == "cpu":
        model.float()

    text_batches = []
    video_outputs, video_masks, video_ids = [], [], []
    video_stats: dict[str, dict[str, float]] = {}
    seen_video_ids = set()
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, input_mask, segment_ids, video, video_mask = (
                tensor.to(device) for tensor in batch
            )
            batch_size = input_ids.size(0)
            batch_video_ids = dataset.video_ids[offset : offset + batch_size]
            text_token = model.encode_text_tokens(input_ids)
            prepared_text_mask = input_mask.view(-1, input_mask.shape[-1])
            text_token, prepared_text_mask = model.prepare_text_for_similarity(
                text_token, prepared_text_mask
            )
            text_batches.append(
                (text_token.cpu(), prepared_text_mask.cpu())
            )

            first_rows = [
                row for row, video_id in enumerate(batch_video_ids)
                if video_id not in seen_video_ids and not seen_video_ids.add(video_id)
            ]
            if first_rows:
                indices = torch.tensor(first_rows, device=device, dtype=torch.long)
                selected_video = video.index_select(0, indices)
                selected_mask = video_mask.index_select(0, indices)
                visual_output = model.encode_video_frames(
                    selected_video, selected_mask
                )
                selected_mask = selected_mask.view(
                    -1, selected_mask.shape[-1]
                )
                visual_output, selected_mask = (
                    model.prepare_video_for_similarity(
                        visual_output, selected_mask
                    )
                )
                video_outputs.append(visual_output.cpu())
                video_masks.append(selected_mask.cpu())
                for local_row, video_id in zip(first_rows, (batch_video_ids[i] for i in first_rows)):
                    valid = int(video_mask[local_row].sum().item())
                    frames = video[local_row, 0, :valid, 0].float()
                    frame_std = float(frames.flatten(1).std(dim=1).mean().item()) if valid else 0.0
                    temporal = float((frames[1:] - frames[:-1]).abs().mean().item()) if valid > 1 else 0.0
                    video_ids.append(video_id)
                    video_stats[video_id] = {
                        "frame_count": valid,
                        "frame_spatial_std": round(frame_std, 6),
                        "temporal_delta": round(temporal, 6),
                    }
            offset += batch_size

        if offset != len(dataset.video_ids):
            raise ValueError("dataloader did not cover every internal-val row")
        visual_output_all = torch.cat(video_outputs, dim=0)
        video_mask_all = torch.cat(video_masks, dim=0)
        sim_rows = []
        for text_token, input_mask in text_batches:
            chunks = []
            for start in range(0, len(video_ids), args.video_chunk_size):
                end = min(start + args.video_chunk_size, len(video_ids))
                logits, _ = model.get_similarity_logits(
                    text_token.to(device),
                    visual_output_all[start:end].to(device),
                    input_mask.to(device),
                    video_mask_all[start:end].to(device),
                    prepared=True,
                )
                chunks.append(logits.cpu())
            sim_rows.append(torch.cat(chunks, dim=1).numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(sim_rows, axis=0), list(dataset.video_ids), video_ids, video_stats


def write_tsv(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args, summaries, elapsed_seconds: float):
    lines = [
        "# P1 Frozen-Baseline Retrieval Diagnostic",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Split: trusted-v1 internal-val only (`{args.val_csv}`)",
        f"- TQFS cache: `{args.tqfs_cache_dir}`",
        f"- Elapsed: {elapsed_seconds / 60:.1f} min",
        "",
    ]
    for summary in summaries:
        lines.extend([f"## {summary['direction']}", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in summary.items() if key != "direction")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- This is read-only internal validation evidence, not a test-set result.",
            "- Do not introduce P2/P3 unless error slices are stable across checkpoints or seeds and cannot be explained by WTI score, margin, or entropy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import torch

    args = parse_args()
    validate_inputs(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.time()
    print(f"[p1] split=internal-val device={device} checkpoint={args.checkpoint}", flush=True)
    sim_matrix, row_video_ids, candidate_video_ids, video_stats = compute_similarity_and_video_stats(args, device)
    items = read_validation_items(args.val_csv)
    if row_video_ids != [item.video_id for item in items]:
        raise ValueError("dataloader and validation CSV video ordering differ")
    t2v_rows = build_t2v_rows(items, sim_matrix, candidate_video_ids, video_stats)
    v2t_rows = build_v2t_rows(items, sim_matrix, candidate_video_ids, video_stats)
    summaries = [summarize_rows(t2v_rows, "T2V"), summarize_rows(v2t_rows, "V2T")]

    checkpoint_key = Path(args.checkpoint).name.replace(".", "_")
    output_dir = Path(args.output_dir) / checkpoint_key
    write_tsv(output_dir / "t2v_rows.tsv", t2v_rows)
    write_tsv(output_dir / "v2t_rows.tsv", v2t_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "summaries": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", args, summaries, time.time() - start)
    print(f"[p1] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
