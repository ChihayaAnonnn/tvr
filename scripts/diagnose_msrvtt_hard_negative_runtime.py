#!/usr/bin/env python3
"""Runtime hardness diagnostics for MSRVTT hard-negative mappings.

This script compares how two checkpoints score mapped hard-negative videos.
It is intentionally evaluation-only: it never backpropagates or starts
training.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.machinery
import importlib.util
import json
import random
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_BASELINE_CKPT = "ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4"
DEFAULT_TARGET_CKPT = "ckpts/ckpt_msrvtt_20260630_explicit_hn_infonce_w005_wmil0_4gpu_b64/pytorch_model.bin.3"
DEFAULT_HARD_NEGATIVE_PATH = "cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json"
DEFAULT_OUTPUT_DIR = "cache_dir/hard_negatives/diagnostics"
DEFAULT_DATA_ROOT = "/data2/hxj/data/MSRVTT"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The MSRVTT loader imports raw-frame helpers at module import time. Runtime
# hard-negative diagnostics use compressed videos via RawVideoExtractor, so a
# minimal cv2 placeholder is enough in environments without opencv-python.
if importlib.util.find_spec("cv2") is None:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub


@dataclass(frozen=True)
class RuntimeSample:
    sample_index: int
    anchor_video_id: str
    anchor_caption: str
    hard_index: int
    hard_video_id: str
    hard_caption: str


def compute_rank(scores: Sequence[float], target_index: int) -> int:
    """Return 1-based descending rank with optimistic tie handling."""

    if target_index < 0 or target_index >= len(scores):
        raise IndexError(f"target_index={target_index} out of range for {len(scores)} scores")
    target = float(scores[target_index])
    return 1 + sum(1 for score in scores if float(score) > target)


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 6) if values else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = [bool(v) for v in values]
    return round(sum(values) / len(values), 6) if values else 0.0


def summarize_runtime_rows(rows: list[dict], margin: float = 0.5) -> dict[str, float | int]:
    """Aggregate per-sample baseline/target hard-negative diagnostics."""

    baseline_gaps = [float(row["baseline_pos_logit"]) - float(row["baseline_hard_logit"]) for row in rows]
    target_gaps = [float(row["target_pos_logit"]) - float(row["target_hard_logit"]) for row in rows]
    gap_deltas = [target - baseline for baseline, target in zip(baseline_gaps, target_gaps)]
    baseline_ranks = [int(row["baseline_hard_rank"]) for row in rows]
    target_ranks = [int(row["target_hard_rank"]) for row in rows]

    summary: dict[str, float | int] = {
        "num_rows": len(rows),
        "baseline_gap_mean": _mean(baseline_gaps),
        "target_gap_mean": _mean(target_gaps),
        "gap_delta_mean": _mean(gap_deltas),
        "baseline_margin_fail_rate": _rate(gap < margin for gap in baseline_gaps),
        "target_margin_fail_rate": _rate(gap < margin for gap in target_gaps),
        "baseline_hard_rank_mean": _mean(baseline_ranks),
        "target_hard_rank_mean": _mean(target_ranks),
    }
    for k in (1, 5, 10, 50, 100):
        summary[f"baseline_hard_rank_top{k}_rate"] = _rate(rank <= k for rank in baseline_ranks)
        summary[f"target_hard_rank_top{k}_rate"] = _rate(rank <= k for rank in target_ranks)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs target checkpoint scores for mapped MSRVTT hard negatives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline_checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--target_checkpoint", default=DEFAULT_TARGET_CKPT)
    parser.add_argument("--baseline_name", default="b1only_v2")
    parser.add_argument("--target_name", default="explicit_hn_infonce_w005")
    parser.add_argument("--hard_negative_path", default=DEFAULT_HARD_NEGATIVE_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_csv", default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_train.9k.csv")
    parser.add_argument("--val_csv", default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_JSFUSION_test.csv")
    parser.add_argument("--data_path", default=f"{DEFAULT_DATA_ROOT}/annotation/MSRVTT_v2.json")
    parser.add_argument("--features_path", default=f"{DEFAULT_DATA_ROOT}/videos/compressed_videos/msrvtt_224_12fps/")
    parser.add_argument("--max_anchors", type=int, default=256, help="Number of valid anchor samples to diagnose.")
    parser.add_argument(
        "--max_rank_videos",
        type=int,
        default=1024,
        help="Video pool size for hard-rank computation. Set 0 to use all train videos.",
    )
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--video_batch_size", type=int, default=64)
    parser.add_argument("--video_chunk_size", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    return parser.parse_args()


@contextlib.contextmanager
def _temporary_argv(argv: list[str]):
    old = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def build_task_args(cli_args: argparse.Namespace, checkpoint: str):
    """Build a main_task_retrieval Namespace matching train_msrvtt.sh."""

    from main_task_retrieval import get_args

    argv = [
        "diagnose_msrvtt_hard_negative_runtime",
        "--do_eval",
        "--output_dir",
        str(Path(cli_args.output_dir) / "_tmp_model_args"),
        "--init_model",
        checkpoint,
        "--train_csv",
        cli_args.train_csv,
        "--val_csv",
        cli_args.val_csv,
        "--data_path",
        cli_args.data_path,
        "--features_path",
        cli_args.features_path,
        "--num_thread_reader",
        "1",
        "--batch_size",
        "64",
        "--gradient_accumulation_steps",
        "1",
        "--batch_size_val",
        "16",
        "--max_words",
        "32",
        "--max_frames",
        "8",
        "--datatype",
        "msrvtt",
        "--expand_msrvtt_sentences",
        "--feature_framerate",
        "1",
        "--coef_lr",
        "1e-3",
        "--freeze_layer_num",
        "0",
        "--slice_framepos",
        "3",
        "--loose_type",
        "--linear_patch",
        "2d",
        "--sim_header",
        "seqTransf",
        "--strategy",
        "2",
        "--pretrained_clip_name",
        "ViT-B/16",
        "--extra_video_cls_num",
        "2",
        "--extra_text_cls_num",
        "2",
        "--n_video_embeddings",
        "7",
        "--n_text_embeddings",
        "7",
        "--mamba_lr_ratio",
        "0.1",
        "--uncertainty_text_head",
        "text",
        "--log_sigma_min",
        "-1.5",
        "--log_sigma_max",
        "4",
        "--w_evidential",
        "1e-2",
        "--w_neg_reg",
        "5e-2",
        "--w_orth",
        "0.1",
        "--w_uncertainty_reg",
        "1e-3",
        "--fusion_mode",
        "prob_mos",
        "--w_query_sim",
        "0.5",
        "--fusion_temperature",
        "1.5",
        "--rope_mode",
        "2d",
        "--use_ada_norm",
        "--anneal_warmup_epochs",
        "0",
        "--uncertainty_mode",
        "none",
    ]
    with _temporary_argv(argv):
        args = get_args("MSRVTT hard-negative runtime diagnostic")
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1
    args.n_gpu = 1
    args.use_explicit_hard_negative_loss = False
    args.use_hard_negative_packing = False
    return args


def load_hard_mapping(path: str, dataset_len: int) -> list[int]:
    from dataloaders.hard_negative_mapping import load_hard_negative_index

    return load_hard_negative_index(path, dataset_len)


def build_msrvtt_train_dataset(cli_args: argparse.Namespace):
    from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

    tokenizer = ClipTokenizer()
    task_args = build_task_args(cli_args, cli_args.baseline_checkpoint)
    return MSRVTT_TrainDataLoader(
        csv_path=cli_args.train_csv,
        json_path=cli_args.data_path,
        features_path=cli_args.features_path,
        max_words=task_args.max_words,
        feature_framerate=task_args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=task_args.max_frames,
        unfold_sentences=True,
        frame_order=task_args.eval_frame_order,
        slice_framepos=task_args.slice_framepos,
        strategy=task_args.strategy,
        use_attributes=False,
    )


def select_runtime_samples(dataset, hard_index: list[int], max_anchors: int, seed: int) -> list[RuntimeSample]:
    valid_indices = [
        idx
        for idx, hard_idx in enumerate(hard_index)
        if hard_idx >= 0 and idx in dataset.sentences_dict and hard_idx in dataset.sentences_dict
    ]
    rng = random.Random(seed)
    rng.shuffle(valid_indices)
    if max_anchors > 0:
        valid_indices = valid_indices[:max_anchors]

    samples = []
    for idx in valid_indices:
        hard_idx = hard_index[idx]
        anchor_video_id, anchor_caption = dataset.sentences_dict[idx]
        hard_video_id, hard_caption = dataset.sentences_dict[hard_idx]
        samples.append(
            RuntimeSample(
                sample_index=idx,
                anchor_video_id=anchor_video_id,
                anchor_caption=anchor_caption,
                hard_index=hard_idx,
                hard_video_id=hard_video_id,
                hard_caption=hard_caption,
            )
        )
    return samples


def build_video_pool(dataset, samples: list[RuntimeSample], max_rank_videos: int, seed: int) -> list[str]:
    required = []
    seen = set()
    for sample in samples:
        for video_id in (sample.anchor_video_id, sample.hard_video_id):
            if video_id not in seen:
                required.append(video_id)
                seen.add(video_id)

    all_videos = list(dict.fromkeys(dataset.csv_video_ids))
    if max_rank_videos == 0:
        pool = required + [video_id for video_id in all_videos if video_id not in seen]
        return pool

    target_size = max(max_rank_videos, len(required))
    candidates = [video_id for video_id in all_videos if video_id not in seen]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return required + candidates[: max(0, target_size - len(required))]


def tensorize_text_batch(dataset, samples: list[RuntimeSample], start: int, end: int):
    import numpy as np
    import torch

    input_ids, input_mask, segment_ids = [], [], []
    for sample in samples[start:end]:
        ids, mask, seg, _ = dataset._get_text(sample.anchor_video_id, sample.anchor_caption, max_words=dataset.max_words)
        input_ids.append(ids)
        input_mask.append(mask)
        segment_ids.append(seg)
    return (
        torch.from_numpy(np.stack(input_ids, axis=0)).long(),
        torch.from_numpy(np.stack(input_mask, axis=0)).long(),
        torch.from_numpy(np.stack(segment_ids, axis=0)).long(),
    )


def load_video_batch(dataset, video_ids: list[str], start: int, end: int):
    import torch

    video, video_mask = dataset._get_rawvideo(video_ids[start:end])
    video = torch.from_numpy(video).unsqueeze(1).float()
    video_mask = torch.from_numpy(video_mask).unsqueeze(1).long()
    return video, video_mask


def load_model_for_checkpoint(cli_args: argparse.Namespace, checkpoint: str, device):
    from main_task_retrieval import init_model

    task_args = build_task_args(cli_args, checkpoint)
    task_args.init_model = checkpoint
    model = init_model(task_args, device, 1, 0)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model, task_args


def compute_checkpoint_scores(
    cli_args: argparse.Namespace,
    checkpoint: str,
    dataset,
    samples: list[RuntimeSample],
    video_pool: list[str],
    device,
) -> list[list[float]]:
    import numpy as np
    import torch

    model, _task_args = load_model_for_checkpoint(cli_args, checkpoint, device)
    sample_count = len(samples)
    pool_count = len(video_pool)
    score_matrix = np.empty((sample_count, pool_count), dtype=np.float32)

    with torch.no_grad():
        text_features = []
        for start in range(0, sample_count, cli_args.text_batch_size):
            end = min(start + cli_args.text_batch_size, sample_count)
            ids, mask, seg = tensorize_text_batch(dataset, samples, start, end)
            ids = ids.to(device)
            mask = mask.to(device)
            seg = seg.to(device)
            sequence_output, text_token = model.get_sequence_output(ids, seg, mask)
            text_features.append((sequence_output.cpu(), text_token.cpu(), mask.cpu()))

        col_start = 0
        for video_start in range(0, pool_count, cli_args.video_batch_size):
            video_end = min(video_start + cli_args.video_batch_size, pool_count)
            video, video_mask = load_video_batch(dataset, video_pool, video_start, video_end)
            video = video.to(device)
            video_mask = video_mask.to(device)
            visual_output = model.get_visual_output(video, video_mask, shaped=False)

            for text_batch_idx, (sequence_output, text_token, text_mask) in enumerate(text_features):
                row_start = text_batch_idx * cli_args.text_batch_size
                row_end = min(row_start + sequence_output.size(0), sample_count)
                logits, _ = model.get_similarity_logits(
                    sequence_output.to(device),
                    text_token.to(device),
                    visual_output,
                    text_mask.to(device),
                    video_mask,
                    loose_type=model.loose_type,
                )
                score_matrix[row_start:row_end, col_start:video_end] = logits.detach().cpu().numpy()

            col_start = video_end

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score_matrix.tolist()


def build_runtime_rows(
    samples: list[RuntimeSample],
    video_pool: list[str],
    baseline_scores: list[list[float]],
    target_scores: list[list[float]],
) -> list[dict]:
    video_to_pool_index = {video_id: idx for idx, video_id in enumerate(video_pool)}
    rows = []
    for idx, sample in enumerate(samples):
        pos_idx = video_to_pool_index[sample.anchor_video_id]
        hard_idx = video_to_pool_index[sample.hard_video_id]
        baseline_row = baseline_scores[idx]
        target_row = target_scores[idx]
        baseline_pos = float(baseline_row[pos_idx])
        baseline_hard = float(baseline_row[hard_idx])
        target_pos = float(target_row[pos_idx])
        target_hard = float(target_row[hard_idx])
        rows.append(
            {
                "sample_index": sample.sample_index,
                "anchor_video_id": sample.anchor_video_id,
                "hard_index": sample.hard_index,
                "hard_video_id": sample.hard_video_id,
                "anchor_caption": sample.anchor_caption,
                "hard_caption": sample.hard_caption,
                "baseline_pos_logit": round(baseline_pos, 6),
                "baseline_hard_logit": round(baseline_hard, 6),
                "baseline_gap": round(baseline_pos - baseline_hard, 6),
                "baseline_hard_rank": compute_rank(baseline_row, hard_idx),
                "target_pos_logit": round(target_pos, 6),
                "target_hard_logit": round(target_hard, 6),
                "target_gap": round(target_pos - target_hard, 6),
                "target_hard_rank": compute_rank(target_row, hard_idx),
                "gap_delta": round((target_pos - target_hard) - (baseline_pos - baseline_hard), 6),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict],
    summary: dict[str, float | int],
    video_pool_size: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MSRVTT Hard Negative Runtime Diagnostic",
        "",
        f"- Baseline: `{args.baseline_name}` / `{args.baseline_checkpoint}`",
        f"- Target: `{args.target_name}` / `{args.target_checkpoint}`",
        f"- Hard map: `{args.hard_negative_path}`",
        f"- Anchors: {summary['num_rows']}",
        f"- Video pool: {video_pool_size}",
        f"- Margin: {args.margin}",
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
            "- `gap = pos_logit - hard_logit`; larger means the checkpoint pushes the mapped hard video farther below the positive video.",
            "- `hard_rank` is the mapped hard video's 1-based rank inside this diagnostic video pool, not necessarily full-train rank unless `--max_rank_videos 0` was used.",
            "- High top-k rate means the mapped hard negatives are genuinely confusing for the model; low top-k rate means they are already easy negatives.",
            "",
            "## Outputs",
            "",
            f"- Per-sample TSV: `{path.with_suffix('.tsv')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_json(path: Path, summary: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    import torch

    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start_time = time.time()

    dataset = build_msrvtt_train_dataset(args)
    hard_index = load_hard_mapping(args.hard_negative_path, len(dataset))
    samples = select_runtime_samples(dataset, hard_index, args.max_anchors, args.seed)
    if not samples:
        raise RuntimeError("No valid hard-negative samples found.")
    video_pool = build_video_pool(dataset, samples, args.max_rank_videos, args.seed)

    print(
        f"[diagnose] anchors={len(samples)} video_pool={len(video_pool)} device={device} "
        f"baseline={args.baseline_name} target={args.target_name}",
        flush=True,
    )
    baseline_scores = compute_checkpoint_scores(args, args.baseline_checkpoint, dataset, samples, video_pool, device)
    target_scores = compute_checkpoint_scores(args, args.target_checkpoint, dataset, samples, video_pool, device)
    rows = build_runtime_rows(samples, video_pool, baseline_scores, target_scores)
    summary = summarize_runtime_rows(rows, margin=args.margin)

    output_dir = Path(args.output_dir)
    stem = f"runtime_hardness_{args.baseline_name}_vs_{args.target_name}_n{len(samples)}_v{len(video_pool)}"
    tsv_path = output_dir / f"{stem}.tsv"
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.summary.json"
    write_tsv(tsv_path, rows)
    save_summary_json(json_path, summary)
    write_report(md_path, args, rows, summary, len(video_pool), time.time() - start_time)

    print(f"[diagnose] wrote {tsv_path}", flush=True)
    print(f"[diagnose] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
