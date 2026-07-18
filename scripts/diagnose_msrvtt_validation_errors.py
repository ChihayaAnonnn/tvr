#!/usr/bin/env python3
"""Diagnose whether trusted-v1 internal-val top-1 errors are hard-negative-like.

方案 C: compare a baseline checkpoint and a hard-negative checkpoint on
MSRVTT trusted-v1 internal-val rows, then classify which top-1 errors were
fixed, regressed, or unchanged. JSFusion 1K remains an explicit blind test and
is never scored by this diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_msrvtt_hard_negative_runtime import (  # noqa: E402
    validate_trusted_diagnostic_inputs,
)

DEFAULT_OUTPUT_DIR = "cache_dir/hard_negatives/diagnostics"
DEFAULT_DATA_ROOT = "/data2/hxj/data/MSRVTT"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "with",
}
TOPIC_KEYWORDS = {
    "basketball",
    "baseball",
    "cartoon",
    "cooking",
    "dancing",
    "download",
    "football",
    "game",
    "gameplay",
    "gymnastics",
    "hockey",
    "kitchen",
    "minecraft",
    "mortal",
    "music",
    "news",
    "singing",
    "soccer",
    "talking",
    "train",
    "tutorial",
    "wrestling",
}


@dataclass(frozen=True)
class ValidationItem:
    index: int
    video_id: str
    caption: str


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in (m.group(0).lower() for m in TOKEN_RE.finditer(str(text)))
        if len(token) >= 2 and token not in STOPWORDS
    }


def text_pair_metrics(query: str, candidate: str) -> dict[str, float | bool]:
    q_tokens = tokenize(query)
    c_tokens = tokenize(candidate)
    if not q_tokens or not c_tokens:
        return {"jaccard": 0.0, "overlap": 0.0, "shared_topic": False}
    inter = q_tokens & c_tokens
    union = q_tokens | c_tokens
    jaccard = len(inter) / max(1, len(union))
    overlap = len(inter) / max(1, min(len(q_tokens), len(c_tokens)))
    shared_topic = bool((q_tokens & c_tokens) & TOPIC_KEYWORDS)
    return {
        "jaccard": round(jaccard, 6),
        "overlap": round(overlap, 6),
        "shared_topic": shared_topic,
    }


def is_hard_like_pair(
    query: str,
    candidate: str,
    min_jaccard: float = 0.30,
    min_overlap: float = 0.50,
) -> bool:
    metrics = text_pair_metrics(query, candidate)
    return (
        bool(metrics["shared_topic"])
        or float(metrics["jaccard"]) >= min_jaccard
        or float(metrics["overlap"]) >= min_overlap
    )


def rank_of_ground_truth(scores: Sequence[float], gt_index: int) -> int:
    if gt_index < 0 or gt_index >= len(scores):
        raise IndexError(f"gt_index={gt_index} out of range for {len(scores)} scores")
    gt_score = float(scores[gt_index])
    return 1 + sum(1 for score in scores if float(score) > gt_score)


def _top1_index(scores: Sequence[float]) -> int:
    if not scores:
        raise ValueError("scores must not be empty")
    return max(range(len(scores)), key=lambda idx: float(scores[idx]))


def _transition(baseline_correct: bool, target_correct: bool, baseline_top1: int, target_top1: int) -> str:
    if baseline_correct and target_correct:
        return "both_correct"
    if not baseline_correct and target_correct:
        return "fixed_by_hn"
    if baseline_correct and not target_correct:
        return "regressed_by_hn"
    if baseline_top1 == target_top1:
        return "both_wrong_same"
    return "both_wrong_changed"


def build_validation_error_rows(
    items: list[ValidationItem],
    baseline_sim: Sequence[Sequence[float]],
    target_sim: Sequence[Sequence[float]],
) -> list[dict]:
    if len(baseline_sim) != len(items) or len(target_sim) != len(items):
        raise ValueError("sim matrix row count must match validation item count")

    candidates: list[ValidationItem] = []
    video_to_column: dict[str, int] = {}
    for item in items:
        if item.video_id not in video_to_column:
            video_to_column[item.video_id] = len(candidates)
            candidates.append(item)

    rows = []
    for row_index, item in enumerate(items):
        baseline_scores = list(baseline_sim[row_index])
        target_scores = list(target_sim[row_index])
        if (
            len(baseline_scores) != len(candidates)
            or len(target_scores) != len(candidates)
        ):
            raise ValueError(
                "sim matrix column count must match unique validation videos"
            )

        baseline_top1 = _top1_index(baseline_scores)
        target_top1 = _top1_index(target_scores)
        baseline_pred = candidates[baseline_top1]
        target_pred = candidates[target_top1]
        baseline_correct = baseline_pred.video_id == item.video_id
        target_correct = target_pred.video_id == item.video_id
        baseline_metrics = text_pair_metrics(item.caption, baseline_pred.caption)
        target_metrics = text_pair_metrics(item.caption, target_pred.caption)
        baseline_hard_like = False if baseline_correct else is_hard_like_pair(item.caption, baseline_pred.caption)
        target_hard_like = False if target_correct else is_hard_like_pair(item.caption, target_pred.caption)
        gt_column = video_to_column[item.video_id]
        baseline_gt_score = float(baseline_scores[gt_column])
        target_gt_score = float(target_scores[gt_column])
        baseline_top1_score = float(baseline_scores[baseline_top1])
        target_top1_score = float(target_scores[target_top1])

        rows.append(
            {
                "query_index": item.index,
                "query_video_id": item.video_id,
                "query_caption": item.caption,
                "baseline_top1_index": baseline_top1,
                "baseline_top1_video_id": baseline_pred.video_id,
                "baseline_top1_caption": baseline_pred.caption,
                "baseline_correct": baseline_correct,
                "baseline_gt_rank": rank_of_ground_truth(
                    baseline_scores, gt_column
                ),
                "baseline_gt_logit": round(baseline_gt_score, 6),
                "baseline_top1_logit": round(baseline_top1_score, 6),
                "baseline_top1_margin": round(baseline_top1_score - baseline_gt_score, 6),
                "baseline_pred_jaccard": baseline_metrics["jaccard"],
                "baseline_pred_overlap": baseline_metrics["overlap"],
                "baseline_pred_shared_topic": baseline_metrics["shared_topic"],
                "baseline_pred_hard_like": baseline_hard_like,
                "target_top1_index": target_top1,
                "target_top1_video_id": target_pred.video_id,
                "target_top1_caption": target_pred.caption,
                "target_correct": target_correct,
                "target_gt_rank": rank_of_ground_truth(
                    target_scores, gt_column
                ),
                "target_gt_logit": round(target_gt_score, 6),
                "target_top1_logit": round(target_top1_score, 6),
                "target_top1_margin": round(target_top1_score - target_gt_score, 6),
                "target_pred_jaccard": target_metrics["jaccard"],
                "target_pred_overlap": target_metrics["overlap"],
                "target_pred_shared_topic": target_metrics["shared_topic"],
                "target_pred_hard_like": target_hard_like,
                "transition": _transition(baseline_correct, target_correct, baseline_top1, target_top1),
            }
        )
    return rows


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(sum(bool(v) for v in values) / len(values), 6) if values else 0.0


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 6) if values else 0.0


def summarize_error_rows(rows: list[dict]) -> dict[str, float | int]:
    transitions = {name: 0 for name in ("both_correct", "fixed_by_hn", "regressed_by_hn", "both_wrong_same", "both_wrong_changed")}
    for row in rows:
        transitions[row["transition"]] = transitions.get(row["transition"], 0) + 1

    baseline_errors = [row for row in rows if not row["baseline_correct"]]
    target_errors = [row for row in rows if not row["target_correct"]]
    fixed = [row for row in rows if row["transition"] == "fixed_by_hn"]
    regressed = [row for row in rows if row["transition"] == "regressed_by_hn"]
    return {
        "num_queries": len(rows),
        "baseline_error_count": len(baseline_errors),
        "target_error_count": len(target_errors),
        "baseline_top1_acc": _rate(row["baseline_correct"] for row in rows),
        "target_top1_acc": _rate(row["target_correct"] for row in rows),
        "both_correct_count": transitions["both_correct"],
        "fixed_by_hn_count": transitions["fixed_by_hn"],
        "regressed_by_hn_count": transitions["regressed_by_hn"],
        "both_wrong_same_count": transitions["both_wrong_same"],
        "both_wrong_changed_count": transitions["both_wrong_changed"],
        "fixed_to_regressed_ratio": round(len(fixed) / max(1, len(regressed)), 6),
        "baseline_error_hard_like_rate": _rate(row["baseline_pred_hard_like"] for row in baseline_errors),
        "target_error_hard_like_rate": _rate(row["target_pred_hard_like"] for row in target_errors),
        "fixed_error_hard_like_rate": _rate(row["baseline_pred_hard_like"] for row in fixed),
        "regressed_error_hard_like_rate": _rate(row["target_pred_hard_like"] for row in regressed),
        "baseline_gt_rank_mean": _mean(row["baseline_gt_rank"] for row in rows),
        "target_gt_rank_mean": _mean(row["target_gt_rank"] for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose MSRVTT validation top-1 errors for hard-negative-like patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--target_checkpoint", required=True)
    parser.add_argument("--baseline_name", default="b1only_v2")
    parser.add_argument("--target_name", default="explicit_hn_infonce_w005")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train_csv",
        default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/train.csv"),
    )
    parser.add_argument(
        "--source_train_csv",
        default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_train.9k.csv",
    )
    parser.add_argument(
        "--test_csv",
        default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_JSFUSION_test.csv",
    )
    parser.add_argument(
        "--split_manifest",
        default=str(
            PROJECT_ROOT
            / "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
        ),
    )
    parser.add_argument(
        "--val_csv",
        default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/val.csv"),
    )
    parser.add_argument("--data_path", default=f"{DEFAULT_DATA_ROOT}/annotation/MSRVTT_v2.json")
    parser.add_argument("--features_path", default=f"{DEFAULT_DATA_ROOT}/videos/compressed_videos/msrvtt_224_12fps/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--video_chunk_size", type=int, default=128)
    parser.add_argument("--max_queries", type=int, default=0, help="Debug only. 0 means all validation rows.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    return parser.parse_args()


def read_validation_items(csv_path: str, max_queries: int = 0) -> list[ValidationItem]:
    items: list[ValidationItem] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row.get("video_id", "")).strip()
            caption = str(row.get("sentence", "")).strip()
            if not video_id or not caption:
                continue
            items.append(ValidationItem(len(items), video_id, caption))
            if max_queries > 0 and len(items) >= max_queries:
                break
    if not items:
        raise ValueError(f"No validation rows loaded from {csv_path}")
    return items


def build_validation_dataloader(args: argparse.Namespace):
    from torch.utils.data import DataLoader

    from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
    from scripts.diagnose_msrvtt_hard_negative_runtime import build_task_args

    task_args = build_task_args(args, args.baseline_checkpoint)
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
        use_attributes=False,
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )
    if args.max_queries > 0:
        dataset.video_ids = dataset.video_ids[: args.max_queries]
        dataset.sentences = dataset.sentences[: args.max_queries]
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)


def compute_validation_sim_matrix(args: argparse.Namespace, checkpoint: str, device):
    import numpy as np
    import torch

    from scripts.diagnose_msrvtt_hard_negative_runtime import load_model_for_checkpoint

    dataloader = build_validation_dataloader(args)
    model, _task_args = load_model_for_checkpoint(args, checkpoint, device)
    model.eval()

    dataset_video_ids = list(dataloader.dataset.video_ids)
    text_batches = []
    video_batches = []
    seen_video_ids: set[str] = set()
    row_offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            if len(batch) != 5:
                raise ValueError(f"Unexpected validation batch len={len(batch)}")
            input_ids, input_mask, segment_ids, video, video_mask = batch
            batch_size = input_ids.size(0)
            batch_video_ids = dataset_video_ids[
                row_offset : row_offset + batch_size
            ]
            if len(batch_video_ids) != batch_size:
                raise ValueError(
                    "validation dataloader rows exceed dataset.video_ids"
                )

            text_token = model.encode_text_tokens(input_ids)
            prepared_text_mask = input_mask.view(-1, input_mask.shape[-1])
            text_token, prepared_text_mask = model.prepare_text_for_similarity(
                text_token, prepared_text_mask
            )
            text_batches.append(
                (text_token.cpu(), prepared_text_mask.cpu())
            )

            first_local_rows = []
            for local_row, video_id in enumerate(batch_video_ids):
                if video_id not in seen_video_ids:
                    seen_video_ids.add(video_id)
                    first_local_rows.append(local_row)
            if first_local_rows:
                local_indices = torch.tensor(
                    first_local_rows,
                    device=video.device,
                    dtype=torch.long,
                )
                unique_video = video.index_select(0, local_indices)
                unique_video_mask = video_mask.index_select(0, local_indices)
                visual_output = model.encode_video_frames(
                    unique_video,
                    unique_video_mask,
                )
                unique_video_mask = unique_video_mask.view(
                    -1, unique_video_mask.shape[-1]
                )
                visual_output, unique_video_mask = (
                    model.prepare_video_for_similarity(
                        visual_output, unique_video_mask
                    )
                )
                video_batches.append(
                    (visual_output.cpu(), unique_video_mask.cpu())
                )
            row_offset += batch_size

        if row_offset != len(dataset_video_ids):
            raise ValueError(
                "validation dataloader row count does not match dataset.video_ids"
            )
        if not text_batches or not video_batches:
            raise ValueError("validation dataloader must contain text and video rows")

        visual_output_all = torch.cat([batch[0] for batch in video_batches], dim=0)
        video_mask_all = torch.cat([batch[1] for batch in video_batches], dim=0)
        n_video = visual_output_all.size(0)
        sim_rows = []
        for text_token, input_mask in text_batches:
            row_chunks = []
            for start in range(0, n_video, args.video_chunk_size):
                end = min(start + args.video_chunk_size, n_video)
                logits, _ = model.get_similarity_logits(
                    text_token.to(device),
                    visual_output_all[start:end].to(device),
                    input_mask.to(device),
                    video_mask_all[start:end].to(device),
                    prepared=True,
                )
                row_chunks.append(logits.detach().cpu())
            sim_rows.append(torch.cat(row_chunks, dim=1).numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(sim_rows, axis=0).tolist()


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary: dict[str, float | int],
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MSRVTT Validation Error Diagnostic",
        "",
        f"- Baseline: `{args.baseline_name}` / `{args.baseline_checkpoint}`",
        f"- Target: `{args.target_name}` / `{args.target_checkpoint}`",
        f"- Validation CSV: `{args.val_csv}`",
        f"- Queries: {summary['num_queries']}",
        f"- Elapsed: {elapsed_seconds / 60:.1f} min",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in sorted(summary):
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `fixed_by_hn`: baseline top-1 is wrong and target top-1 is correct.",
            "- `regressed_by_hn`: baseline top-1 is correct and target top-1 is wrong.",
            "- `hard_like` marks wrong top-1 pairs with high token overlap or shared topic keywords.",
            "",
            "## Outputs",
            "",
            f"- Per-query TSV: `{path.with_suffix('.tsv')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import torch

    args = parse_args()
    validate_trusted_diagnostic_inputs(args, scored_csv=args.val_csv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start_time = time.time()
    items = read_validation_items(args.val_csv, max_queries=args.max_queries)
    print(
        f"[valdiag] queries={len(items)} device={device} baseline={args.baseline_name} target={args.target_name}",
        flush=True,
    )
    baseline_sim = compute_validation_sim_matrix(args, args.baseline_checkpoint, device)
    target_sim = compute_validation_sim_matrix(args, args.target_checkpoint, device)
    rows = build_validation_error_rows(items, baseline_sim, target_sim)
    summary = summarize_error_rows(rows)

    output_dir = Path(args.output_dir)
    stem = f"validation_errors_{args.baseline_name}_vs_{args.target_name}_n{len(items)}"
    tsv_path = output_dir / f"{stem}.tsv"
    json_path = output_dir / f"{stem}.summary.json"
    md_path = output_dir / f"{stem}.md"
    write_tsv(tsv_path, rows)
    write_json(json_path, summary)
    write_report(md_path, args, summary, time.time() - start_time)
    print(f"[valdiag] wrote {tsv_path}", flush=True)
    print(f"[valdiag] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
