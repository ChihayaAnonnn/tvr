#!/usr/bin/env python3
"""Build model-mined MSRVTT hard-negative mappings.

The earlier hard-negative map is static and text-only.  This script instead
uses a trained retrieval checkpoint to mine videos that the current model ranks
near the top for each training caption.  Output stays compatible with
``dataloaders.hard_negative_mapping.load_hard_negative_index``.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if importlib.util.find_spec("cv2") is None:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

from scripts.build_msrvtt_hard_negatives import (  # noqa: E402
    STOPWORDS,
    TOKEN_RE,
)
from scripts.diagnose_msrvtt_hard_negative_runtime import (  # noqa: E402
    validate_trusted_diagnostic_inputs,
)

DEFAULT_DATA_ROOT = "/data2/hxj/data/MSRVTT"
DEFAULT_OUTPUT = "cache_dir/hard_negatives/msrvtt_train_hardneg_model_mined_b1_e4.json"


@dataclass(frozen=True)
class CaptionSample:
    sample_index: int
    video_id: str
    caption: str
    sen_id: int | None = None
    json_sentence_index: int | None = None


@dataclass(frozen=True)
class ModelHardNegativeConfig:
    top_k: int = 20
    min_rank: int = 2
    max_jaccard: float = 0.8
    max_overlap: float = 0.9
    min_token_len: int = 2
    keep_stopwords: bool = False


@dataclass(frozen=True)
class CandidateSelection:
    hard_index: int
    hard_video_id: str
    hard_caption: str
    model_rank: int
    candidate_count: int
    skipped_rank_window: int
    skipped_same_video: int
    skipped_text_risk: int
    exact_caption: bool
    jaccard: float
    overlap: float
    shared_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MSRVTT hard-negative map from a trained retrieval checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
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
    parser.add_argument("--top_k", type=int, default=20, help="Inspect top-K model-ranked train videos per caption.")
    parser.add_argument("--min_rank", type=int, default=2, help="Skip candidates before this 1-based model rank.")
    parser.add_argument("--max_jaccard", type=float, default=0.8)
    parser.add_argument("--max_overlap", type=float, default=0.9)
    parser.add_argument("--min_token_len", type=int, default=2)
    parser.add_argument("--keep_stopwords", action="store_true")
    parser.add_argument("--limit_queries", type=int, default=0, help="Debug only. 0 means all train captions.")
    parser.add_argument(
        "--query_start",
        type=int,
        default=0,
        help="0-based inclusive start caption index for sharded mining.",
    )
    parser.add_argument(
        "--query_end",
        type=int,
        default=0,
        help="0-based exclusive end caption index for sharded mining. 0 means dataset length.",
    )
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--video_batch_size", type=int, default=64, help="Batch size for encoding raw videos.")
    parser.add_argument("--video_chunk_size", type=int, default=256, help="Cached video feature chunk size for logits.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--include_captions", action="store_true", help="Store anchor/hard captions in output JSON.")
    parser.add_argument("--progress_interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", default=None, help="Defaults to '<output>.checkpoint.json'.")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--keep_checkpoint", action="store_true")
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    return " ".join(match.group(0).lower() for match in TOKEN_RE.finditer(str(text)))


def caption_token_set(text: str, config: ModelHardNegativeConfig) -> set[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(str(text))]
    if config.min_token_len > 1:
        tokens = [token for token in tokens if len(token) >= config.min_token_len]
    if not config.keep_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return set(tokens)


def text_pair_metrics(anchor_caption: str, hard_caption: str, config: ModelHardNegativeConfig | None = None) -> dict:
    config = config or ModelHardNegativeConfig()
    anchor_tokens = caption_token_set(anchor_caption, config)
    hard_tokens = caption_token_set(hard_caption, config)
    intersection = len(anchor_tokens & hard_tokens)
    union = len(anchor_tokens | hard_tokens)
    min_size = min(len(anchor_tokens), len(hard_tokens))
    return {
        "exact_caption": normalize_caption(anchor_caption) == normalize_caption(hard_caption),
        "jaccard": round(intersection / union, 6) if union else 0.0,
        "overlap": round(intersection / min_size, 6) if min_size else 0.0,
        "shared_tokens": intersection,
        "anchor_tokens": len(anchor_tokens),
        "hard_tokens": len(hard_tokens),
    }


def is_text_risky(metrics: dict, config: ModelHardNegativeConfig) -> bool:
    return (
        bool(metrics["exact_caption"])
        or float(metrics["jaccard"]) >= config.max_jaccard
        or float(metrics["overlap"]) >= config.max_overlap
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 6) if values else None


def build_video_to_sample_indices(samples: Sequence[CaptionSample]) -> dict[str, list[int]]:
    by_video: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        by_video[sample.video_id].append(sample.sample_index)
    return dict(by_video)


def choose_caption_for_candidate_video(
    anchor: CaptionSample,
    candidate_indices: Sequence[int],
    samples: Sequence[CaptionSample],
    config: ModelHardNegativeConfig,
) -> tuple[CaptionSample, dict] | None:
    options = []
    for sample_index in candidate_indices:
        candidate = samples[int(sample_index)]
        metrics = text_pair_metrics(anchor.caption, candidate.caption, config)
        if is_text_risky(metrics, config):
            return None
        options.append((candidate, metrics))
    if not options:
        return None
    return min(
        options,
        key=lambda item: (
            float(item[1]["overlap"]),
            float(item[1]["jaccard"]),
            int(item[1]["shared_tokens"]),
            item[0].sample_index,
        ),
    )


def select_hard_negative_candidate(
    anchor: CaptionSample,
    ranked_video_indices: Sequence[int],
    video_ids: Sequence[str],
    samples: Sequence[CaptionSample],
    video_to_sample_indices: dict[str, list[int]],
    config: ModelHardNegativeConfig,
) -> CandidateSelection | None:
    skipped_rank_window = 0
    skipped_same_video = 0
    skipped_text_risk = 0
    candidate_count = min(config.top_k, len(ranked_video_indices))

    for model_rank, video_idx in enumerate(ranked_video_indices[: config.top_k], start=1):
        if model_rank < config.min_rank:
            skipped_rank_window += 1
            continue
        video_id = video_ids[int(video_idx)]
        if video_id == anchor.video_id:
            skipped_same_video += 1
            continue
        candidate_indices = video_to_sample_indices.get(video_id, [])
        chosen = choose_caption_for_candidate_video(anchor, candidate_indices, samples, config)
        if chosen is None:
            skipped_text_risk += 1
            continue
        hard_sample, metrics = chosen
        return CandidateSelection(
            hard_index=hard_sample.sample_index,
            hard_video_id=hard_sample.video_id,
            hard_caption=hard_sample.caption,
            model_rank=model_rank,
            candidate_count=candidate_count,
            skipped_rank_window=skipped_rank_window,
            skipped_same_video=skipped_same_video,
            skipped_text_risk=skipped_text_risk,
            exact_caption=bool(metrics["exact_caption"]),
            jaccard=float(metrics["jaccard"]),
            overlap=float(metrics["overlap"]),
            shared_tokens=int(metrics["shared_tokens"]),
        )
    return None


def _ranked_video_indices(scores: Sequence[float], top_k: int) -> list[int]:
    limit = min(max(1, int(top_k)), len(scores))
    return sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))[:limit]


def _mapping_item(
    anchor: CaptionSample,
    candidate: CandidateSelection,
    positive_score: float | None,
    hard_score: float | None,
    include_captions: bool,
) -> dict:
    item = {
        "anchor_index": anchor.sample_index,
        "anchor_video_id": anchor.video_id,
        "hard_index": candidate.hard_index,
        "hard_video_id": candidate.hard_video_id,
        "model_rank": candidate.model_rank,
        "model_score": round(float(hard_score), 6) if hard_score is not None else None,
        "positive_score": round(float(positive_score), 6) if positive_score is not None else None,
        "positive_minus_hard": round(float(positive_score) - float(hard_score), 6)
        if positive_score is not None and hard_score is not None
        else None,
        "candidate_count": candidate.candidate_count,
        "skipped_rank_window": candidate.skipped_rank_window,
        "skipped_same_video": candidate.skipped_same_video,
        "skipped_text_risk": candidate.skipped_text_risk,
        "exact_caption": candidate.exact_caption,
        "jaccard": round(candidate.jaccard, 6),
        "overlap": round(candidate.overlap, 6),
        "shared_tokens": candidate.shared_tokens,
        "miner_backend": "checkpoint_topk",
    }
    if include_captions:
        item["anchor_caption"] = anchor.caption
        item["hard_caption"] = candidate.hard_caption
    return item


def summarize_mapping(mapping: dict[str, dict], num_samples: int, num_videos: int, elapsed_seconds: float = 0.0) -> dict:
    values = list(mapping.values())
    model_ranks = [float(item["model_rank"]) for item in values if item.get("model_rank") is not None]
    margins = [
        float(item["positive_minus_hard"])
        for item in values
        if item.get("positive_minus_hard") is not None
    ]
    hard_indices = Counter(int(item["hard_index"]) for item in values)
    hard_videos = Counter(str(item["hard_video_id"]) for item in values)
    skipped_rank = sum(int(item.get("skipped_rank_window", 0) or 0) for item in values)
    skipped_same = sum(int(item.get("skipped_same_video", 0) or 0) for item in values)
    skipped_text = sum(int(item.get("skipped_text_risk", 0) or 0) for item in values)
    unmapped_count = int(num_samples) - len(values)
    return {
        "num_train_videos": int(num_videos),
        "num_samples": int(num_samples),
        "mapping_size": len(values),
        "unmapped_count": unmapped_count,
        "unmapped_rate": round(unmapped_count / max(1, int(num_samples)), 6),
        "fallback_count": 0,
        "model_rank_mean": _mean(model_ranks),
        "model_rank_p50": percentile(model_ranks, 0.5),
        "model_rank_p90": percentile(model_ranks, 0.9),
        "positive_minus_hard_mean": _mean(margins),
        "positive_minus_hard_p50": percentile(margins, 0.5),
        "positive_minus_hard_p10": percentile(margins, 0.1),
        "unique_hard_indices": len(hard_indices),
        "unique_hard_videos": len(hard_videos),
        "max_hard_index_count": max(hard_indices.values(), default=0),
        "max_hard_video_count": max(hard_videos.values(), default=0),
        "skipped_rank_window_total": skipped_rank,
        "skipped_same_video_total": skipped_same,
        "skipped_text_risk_total": skipped_text,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }


def build_model_mined_mapping(
    samples: Sequence[CaptionSample],
    video_ids: Sequence[str],
    scores: Sequence[Sequence[float]],
    config: ModelHardNegativeConfig,
    include_captions: bool = False,
) -> dict:
    if len(samples) != len(scores):
        raise ValueError("scores row count must match samples")
    video_to_sample_indices = build_video_to_sample_indices(samples)
    video_to_index = {video_id: idx for idx, video_id in enumerate(video_ids)}
    mapping: dict[str, dict] = {}

    for row_idx, anchor in enumerate(samples):
        row = list(scores[row_idx])
        if len(row) != len(video_ids):
            raise ValueError("each score row must match video_ids")
        ranked = _ranked_video_indices(row, config.top_k)
        candidate = select_hard_negative_candidate(
            anchor=anchor,
            ranked_video_indices=ranked,
            video_ids=video_ids,
            samples=samples,
            video_to_sample_indices=video_to_sample_indices,
            config=config,
        )
        if candidate is None:
            continue
        positive_score = row[video_to_index[anchor.video_id]] if anchor.video_id in video_to_index else None
        hard_score = row[video_to_index[candidate.hard_video_id]] if candidate.hard_video_id in video_to_index else None
        mapping[str(anchor.sample_index)] = _mapping_item(
            anchor=anchor,
            candidate=candidate,
            positive_score=positive_score,
            hard_score=hard_score,
            include_captions=include_captions,
        )

    return {
        "meta": {
            "task": "msrvtt_model_mined_hard_negative_unit",
            "top_k": config.top_k,
            "min_rank": config.min_rank,
            "max_jaccard": config.max_jaccard,
            "max_overlap": config.max_overlap,
            "min_token_len": config.min_token_len,
            "keep_stopwords": config.keep_stopwords,
        },
        "stats": summarize_mapping(mapping, len(samples), len(video_ids)),
        "mapping": mapping,
    }


def config_from_args(args: argparse.Namespace) -> ModelHardNegativeConfig:
    return ModelHardNegativeConfig(
        top_k=args.top_k,
        min_rank=args.min_rank,
        max_jaccard=args.max_jaccard,
        max_overlap=args.max_overlap,
        min_token_len=args.min_token_len,
        keep_stopwords=args.keep_stopwords,
    )


def build_meta(args: argparse.Namespace) -> dict:
    return {
        "task": "msrvtt_model_mined_hard_negative",
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_csv": os.path.abspath(args.train_csv),
        "source_train_csv": os.path.abspath(args.source_train_csv),
        "test_csv": os.path.abspath(args.test_csv),
        "split_manifest": os.path.abspath(args.split_manifest),
        "val_csv": os.path.abspath(args.val_csv),
        "data_path": os.path.abspath(args.data_path),
        "features_path": os.path.abspath(args.features_path),
        "unfold_sentences": True,
        "miner_backend": "checkpoint_topk",
        "top_k": args.top_k,
        "min_rank": args.min_rank,
        "max_jaccard": args.max_jaccard,
        "max_overlap": args.max_overlap,
        "min_token_len": args.min_token_len,
        "keep_stopwords": args.keep_stopwords,
        "limit_queries": args.limit_queries,
        "query_start": args.query_start,
        "query_end": args.query_end,
        "seed": args.seed,
    }


def checkpoint_path_for(args: argparse.Namespace) -> str:
    return args.checkpoint_path or args.output + ".checkpoint.json"


def write_json(obj: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_resume_state(args: argparse.Namespace, expected_meta: dict) -> tuple[dict[str, dict], set[int]]:
    if not args.resume:
        return {}, set()
    for path in (checkpoint_path_for(args), args.output):
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        old_meta = obj.get("meta", {})
        mismatches = []
        for key, expected in expected_meta.items():
            if old_meta.get(key) != expected:
                mismatches.append((key, old_meta.get(key), expected))
        if mismatches:
            detail = "; ".join(f"{key}: old={old!r}, current={new!r}" for key, old, new in mismatches[:8])
            raise ValueError(f"Resume file is not compatible: {path}; {detail}")
        mapping = obj.get("mapping", {})
        if not isinstance(mapping, dict):
            raise ValueError(f"Resume file has no mapping object: {path}")
        processed = obj.get("processed_indices")
        if processed is None:
            processed_set = {int(v.get("anchor_index", key)) for key, v in mapping.items() if isinstance(v, dict)}
        else:
            processed_set = {int(idx) for idx in processed}
        print(f"[model-mined] resume loaded mapping={len(mapping)} processed={len(processed_set)} from {path}", flush=True)
        return mapping, processed_set
    return {}, set()


def write_checkpoint(
    args: argparse.Namespace,
    meta: dict,
    mapping: dict[str, dict],
    processed_indices: set[int],
    total_samples: int,
    elapsed_seconds: float,
) -> None:
    obj = {
        "meta": meta,
        "stats": summarize_mapping(mapping, total_samples, 0, elapsed_seconds),
        "processed_indices": sorted(processed_indices),
        "mapping": mapping,
    }
    write_json(obj, checkpoint_path_for(args))


def _compat_args_for_model(args: argparse.Namespace) -> argparse.Namespace:
    compat = argparse.Namespace(**vars(args))
    compat.baseline_checkpoint = args.checkpoint
    compat.output_dir = str(Path(args.output).parent / "_tmp_model_args")
    return compat


def build_task_args_for_checkpoint(args: argparse.Namespace):
    from scripts.diagnose_msrvtt_hard_negative_runtime import build_task_args

    task_args = build_task_args(_compat_args_for_model(args), args.checkpoint)
    task_args.num_thread_reader = args.num_workers
    return task_args


def build_train_dataset(args: argparse.Namespace):
    from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

    task_args = build_task_args_for_checkpoint(args)
    tokenizer = ClipTokenizer()
    return MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
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


def samples_from_dataset(dataset) -> list[CaptionSample]:
    samples = []
    for idx in range(len(dataset)):
        video_id, caption = dataset.sentences_dict[idx]
        samples.append(CaptionSample(idx, str(video_id), str(caption)))
    return samples


def select_query_samples(
    samples: Sequence[CaptionSample],
    query_start: int = 0,
    query_end: int = 0,
    limit_queries: int = 0,
) -> tuple[list[CaptionSample], int, int]:
    total = len(samples)
    start = int(query_start)
    end = total if int(query_end) == 0 else int(query_end)
    limit = int(limit_queries)
    if start < 0:
        raise ValueError(f"query_start must be >= 0, got {query_start}")
    if end < 0:
        raise ValueError(f"query_end must be >= 0, got {query_end}")
    if limit < 0:
        raise ValueError(f"limit_queries must be >= 0, got {limit_queries}")
    if start > total:
        raise ValueError(f"query_start={start} exceeds dataset length {total}")
    if end > total:
        raise ValueError(f"query_end={end} exceeds dataset length {total}")
    if end <= start:
        raise ValueError(f"query_end must be greater than query_start, got {start}:{end}")
    if limit > 0:
        end = min(end, start + limit)
    return list(samples[start:end]), start, end


def tensorize_text_batch(dataset, samples: Sequence[CaptionSample], start: int, end: int):
    import numpy as np
    import torch

    input_ids, input_mask, segment_ids = [], [], []
    for sample in samples[start:end]:
        ids, mask, seg, _ = dataset._get_text(sample.video_id, sample.caption, max_words=dataset.max_words)
        input_ids.append(ids)
        input_mask.append(mask)
        segment_ids.append(seg)
    return (
        torch.from_numpy(np.stack(input_ids, axis=0)).long(),
        torch.from_numpy(np.stack(input_mask, axis=0)).long(),
        torch.from_numpy(np.stack(segment_ids, axis=0)).long(),
    )


def load_video_batch(dataset, video_ids: Sequence[str], start: int, end: int):
    import torch

    video, video_mask = dataset._get_rawvideo(list(video_ids[start:end]))
    video = torch.from_numpy(video).unsqueeze(1).float()
    video_mask = torch.from_numpy(video_mask).unsqueeze(1).long()
    return video, video_mask


def load_model(args: argparse.Namespace, device):
    from main_task_retrieval import init_model

    task_args = build_task_args_for_checkpoint(args)
    task_args.init_model = args.checkpoint
    model = init_model(task_args, device, 1, 0)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model


def encode_all_videos(args: argparse.Namespace, model, dataset, video_ids: Sequence[str], device):
    import torch

    visual_outputs = []
    video_masks = []
    total = len(video_ids)
    with torch.no_grad():
        for start in range(0, total, args.video_batch_size):
            end = min(start + args.video_batch_size, total)
            video, video_mask = load_video_batch(dataset, video_ids, start, end)
            video = video.to(device)
            video_mask = video_mask.to(device)
            visual_output = model.get_visual_output(video, video_mask, shaped=False)
            visual_outputs.append(visual_output.detach().cpu())
            video_masks.append(video_mask.detach().cpu())
            print(f"[model-mined] encoded videos {end}/{total}", flush=True)
    return (
        torch.cat(visual_outputs, dim=0),
        torch.cat(video_masks, dim=0),
    )


def _update_topk(current_scores, current_indices, logits, video_offset: int, top_k: int):
    import torch

    k = min(top_k, logits.size(1))
    chunk_scores, chunk_indices = torch.topk(logits, k=k, dim=1)
    chunk_indices = chunk_indices + int(video_offset)
    if current_scores is None:
        return chunk_scores, chunk_indices

    merged_scores = torch.cat([current_scores, chunk_scores], dim=1)
    merged_indices = torch.cat([current_indices, chunk_indices], dim=1)
    keep = min(top_k, merged_scores.size(1))
    top_scores, order = torch.topk(merged_scores, k=keep, dim=1)
    top_indices = merged_indices.gather(1, order)
    return top_scores, top_indices


def mine_mapping_with_model(
    args: argparse.Namespace,
    model,
    dataset,
    query_samples: Sequence[CaptionSample],
    all_samples: Sequence[CaptionSample],
    video_ids: Sequence[str],
    device,
    initial_mapping: dict[str, dict] | None = None,
    processed_indices: set[int] | None = None,
) -> tuple[dict[str, dict], set[int]]:
    import torch

    config = config_from_args(args)
    mapping = dict(initial_mapping or {})
    processed = set(processed_indices or set())
    pending_samples = [sample for sample in query_samples if sample.sample_index not in processed]
    if not pending_samples:
        return mapping, processed

    video_to_sample_indices = build_video_to_sample_indices(all_samples)
    video_to_index = {video_id: idx for idx, video_id in enumerate(video_ids)}
    visual_output_all, video_mask_all = encode_all_videos(args, model, dataset, video_ids, device)
    total = len(pending_samples)
    start_time = time.time()
    last_checkpoint = 0

    with torch.no_grad():
        for start in range(0, total, args.text_batch_size):
            end = min(start + args.text_batch_size, total)
            batch_samples = pending_samples[start:end]
            ids, mask, seg = tensorize_text_batch(dataset, pending_samples, start, end)
            ids = ids.to(device)
            mask = mask.to(device)
            seg = seg.to(device)
            sequence_output, text_token = model.get_sequence_output(ids, seg, mask)
            top_scores = None
            top_indices = None
            positive_scores = [None] * len(batch_samples)

            for video_start in range(0, len(video_ids), args.video_chunk_size):
                video_end = min(video_start + args.video_chunk_size, len(video_ids))
                logits, _ = model.get_similarity_logits(
                    sequence_output,
                    text_token,
                    visual_output_all[video_start:video_end].to(device),
                    mask,
                    video_mask_all[video_start:video_end].to(device),
                    loose_type=model.loose_type,
                )
                top_scores, top_indices = _update_topk(top_scores, top_indices, logits, video_start, config.top_k)
                for row_idx, sample in enumerate(batch_samples):
                    pos_idx = video_to_index.get(sample.video_id)
                    if pos_idx is not None and video_start <= pos_idx < video_end:
                        positive_scores[row_idx] = float(logits[row_idx, pos_idx - video_start].detach().cpu().item())

            top_scores_cpu = top_scores.detach().cpu().tolist()
            top_indices_cpu = top_indices.detach().cpu().tolist()
            for row_idx, anchor in enumerate(batch_samples):
                ranked = [int(idx) for idx in top_indices_cpu[row_idx]]
                candidate = select_hard_negative_candidate(
                    anchor=anchor,
                    ranked_video_indices=ranked,
                    video_ids=video_ids,
                    samples=all_samples,
                    video_to_sample_indices=video_to_sample_indices,
                    config=config,
                )
                processed.add(anchor.sample_index)
                if candidate is None:
                    continue
                hard_video_pos = ranked.index(video_to_index[candidate.hard_video_id])
                hard_score = float(top_scores_cpu[row_idx][hard_video_pos])
                mapping[str(anchor.sample_index)] = _mapping_item(
                    anchor=anchor,
                    candidate=candidate,
                    positive_score=positive_scores[row_idx],
                    hard_score=hard_score,
                    include_captions=args.include_captions,
                )

            done = len(processed)
            if args.progress_interval > 0 and (done % args.progress_interval == 0 or end == total):
                print(
                    f"[model-mined] processed={done}/{len(query_samples)} mapping={len(mapping)} "
                    f"elapsed={(time.time() - start_time) / 60:.1f}m",
                    flush=True,
                )
            if args.checkpoint_interval > 0 and done - last_checkpoint >= args.checkpoint_interval:
                meta = build_meta(args)
                write_checkpoint(args, meta, mapping, processed, len(query_samples), time.time() - start_time)
                last_checkpoint = done

    return mapping, processed


def write_report(path: str | Path, args: argparse.Namespace, stats: dict, elapsed_seconds: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MSRVTT Model-Mined Hard Negative Map",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Output: `{args.output}`",
        f"- Queries: {stats['num_samples']}",
        f"- Query range: [{stats.get('query_start', 0)}, {stats.get('query_end', stats['num_samples'])})",
        f"- Mapping size: {stats['mapping_size']}",
        f"- Unmapped: {stats['unmapped_count']} ({stats['unmapped_rate']:.2%})",
        f"- top_k/min_rank: {args.top_k}/{args.min_rank}",
        f"- Text-risk filter: max_jaccard={args.max_jaccard}, max_overlap={args.max_overlap}",
        f"- Elapsed: {elapsed_seconds / 60:.1f} min",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in sorted(stats):
        lines.append(f"| {key} | {stats[key]} |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `model_rank` is the 1-based rank of the selected hard video inside the model top-k list.",
            "- Missing mappings are allowed; the dataloader marks them invalid and masks the explicit HN loss for those samples.",
            "- Run `scripts/audit_msrvtt_hard_negatives.py` on this output before training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import torch

    args = parse_args()
    validate_trusted_diagnostic_inputs(args)
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.min_rank < 1:
        raise ValueError("--min_rank must be >= 1")
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start_time = time.time()

    dataset = build_train_dataset(args)
    all_samples = samples_from_dataset(dataset)
    query_samples, query_start, query_end = select_query_samples(
        all_samples,
        query_start=args.query_start,
        query_end=args.query_end,
        limit_queries=args.limit_queries,
    )
    video_ids = list(dict.fromkeys(dataset.csv_video_ids))
    meta = build_meta(args)
    mapping, processed = load_resume_state(args, meta)

    print(
        f"[model-mined] query_range=[{query_start},{query_end}) queries={len(query_samples)} "
        f"videos={len(video_ids)} device={device} "
        f"top_k={args.top_k} min_rank={args.min_rank}",
        flush=True,
    )
    model = load_model(args, device)
    mapping, processed = mine_mapping_with_model(
        args=args,
        model=model,
        dataset=dataset,
        query_samples=query_samples,
        all_samples=all_samples,
        video_ids=video_ids,
        device=device,
        initial_mapping=mapping,
        processed_indices=processed,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    stats = summarize_mapping(mapping, len(query_samples), len(video_ids), elapsed)
    stats["query_start"] = query_start
    stats["query_end"] = query_end
    obj = {"meta": meta, "stats": stats, "mapping": mapping}
    write_json(obj, args.output)
    write_report(Path(args.output).with_suffix(".md"), args, stats, elapsed)
    if not args.keep_checkpoint:
        checkpoint_path = checkpoint_path_for(args)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    print(
        f"[model-mined] wrote {args.output} mapping={stats['mapping_size']} "
        f"unmapped={stats['unmapped_count']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
