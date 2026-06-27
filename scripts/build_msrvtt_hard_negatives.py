#!/usr/bin/env python3
"""Build query-based hard negative mappings for MSRVTT.

This script follows the two-stage idea from UACL hard negative sampling:

1. Retrieve query-neighbor candidates by TF-IDF cosine similarity.
2. Re-rank those candidates by BM25 and pick a fixed target rank.

The output maps each training caption sample to a hard-negative video.  It is
designed as an offline artifact for a later hard-negative sampler/loss.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


@dataclass(frozen=True)
class CaptionSample:
    sample_index: int
    video_id: str
    caption: str
    sen_id: int | None
    json_sentence_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MSRVTT query-based hard negative mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_csv",
        default="/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv",
        help="MSRVTT train csv. Usually contains only a video_id column.",
    )
    parser.add_argument(
        "--data_path",
        default="/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json",
        help="MSRVTT annotation json containing the sentences list.",
    )
    parser.add_argument(
        "--output",
        default="cache_dir/hard_negatives/msrvtt_train_hardneg.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--unfold_sentences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build mapping at caption-sample granularity, matching --expand_msrvtt_sentences.",
    )
    parser.add_argument(
        "--dense_top_k",
        type=int,
        default=500,
        help="Number of TF-IDF cosine candidates before BM25 re-ranking.",
    )
    parser.add_argument(
        "--target_rank",
        type=int,
        default=50,
        help="1-based BM25 rank to select from the dense candidates.",
    )
    parser.add_argument(
        "--max_query_terms",
        type=int,
        default=16,
        help="Use the highest-IDF query terms for candidate accumulation.",
    )
    parser.add_argument(
        "--max_posting_docs",
        type=int,
        default=50000,
        help="Skip a term during candidate retrieval if its posting list is larger than this.",
    )
    parser.add_argument(
        "--min_token_len",
        type=int,
        default=2,
        help="Minimum token length after regex tokenization.",
    )
    parser.add_argument(
        "--keep_stopwords",
        action="store_true",
        help="Keep common English stopwords instead of filtering them.",
    )
    parser.add_argument(
        "--exclude_same_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude captions from the same video_id to reduce false negatives.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="Debug only: keep the first N samples after filtering. 0 means all samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only for fallback hard negatives.",
    )
    parser.add_argument(
        "--include_captions",
        action="store_true",
        help="Include anchor/hard captions in output for inspection. Increases file size.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1000,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a compatible checkpoint if it exists.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint JSON path. Defaults to '<output>.checkpoint.json'.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="Write a resumable checkpoint every N newly processed samples. 0 disables checkpoints.",
    )
    parser.add_argument(
        "--keep_checkpoint",
        action="store_true",
        help="Keep the checkpoint after a successful final write.",
    )
    return parser.parse_args()


def read_train_video_ids(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train_csv not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a video_id column")
        video_ids = []
        for row in reader:
            vid = str(row.get("video_id", "")).strip()
            if vid:
                video_ids.append(vid)

    if not video_ids:
        raise ValueError(f"No video ids found in {csv_path}")
    return video_ids


def load_caption_samples(
    data_path: str,
    train_video_ids: Iterable[str],
    unfold_sentences: bool,
    limit_samples: int = 0,
) -> list[CaptionSample]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    train_set = set(train_video_ids)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "sentences" not in data:
        raise ValueError(f"Unexpected MSRVTT annotation schema: {data_path}")

    samples: list[CaptionSample] = []
    if unfold_sentences:
        for json_idx, item in enumerate(data["sentences"]):
            vid = item.get("video_id")
            cap = item.get("caption")
            if vid not in train_set or not isinstance(cap, str):
                continue
            samples.append(
                CaptionSample(
                    sample_index=len(samples),
                    video_id=vid,
                    caption=cap,
                    sen_id=item.get("sen_id"),
                    json_sentence_index=json_idx,
                )
            )
    else:
        by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for json_idx, item in enumerate(data["sentences"]):
            vid = item.get("video_id")
            if vid in train_set:
                by_video[vid].append((json_idx, item))
        for vid in train_video_ids:
            items = by_video.get(vid, [])
            if not items:
                continue
            json_idx, item = items[len(items) // 2]
            cap = item.get("caption", "")
            samples.append(
                CaptionSample(
                    sample_index=len(samples),
                    video_id=vid,
                    caption=str(cap),
                    sen_id=item.get("sen_id"),
                    json_sentence_index=json_idx,
                )
            )

    if limit_samples and limit_samples > 0:
        samples = samples[:limit_samples]
        samples = [
            CaptionSample(
                sample_index=i,
                video_id=s.video_id,
                caption=s.caption,
                sen_id=s.sen_id,
                json_sentence_index=s.json_sentence_index,
            )
            for i, s in enumerate(samples)
        ]

    if not samples:
        raise ValueError("No caption samples found")
    return samples


def tokenize(text: str, min_token_len: int, keep_stopwords: bool) -> list[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= min_token_len]
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def build_statistics(
    samples: list[CaptionSample],
    min_token_len: int,
    keep_stopwords: bool,
) -> tuple[list[Counter[str]], Counter[str], list[int], float]:
    doc_tfs: list[Counter[str]] = []
    df: Counter[str] = Counter()
    doc_lens: list[int] = []

    for sample in samples:
        tokens = tokenize(sample.caption, min_token_len, keep_stopwords)
        tf = Counter(tokens)
        doc_tfs.append(tf)
        df.update(tf.keys())
        doc_lens.append(sum(tf.values()))

    avgdl = sum(doc_lens) / max(1, len(doc_lens))
    return doc_tfs, df, doc_lens, avgdl


def compute_idf(df: Counter[str], num_docs: int) -> dict[str, float]:
    return {
        term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }


def tfidf_weight(tf: int, idf: float) -> float:
    if tf <= 0:
        return 0.0
    return (1.0 + math.log(float(tf))) * idf


def build_tfidf_postings(
    doc_tfs: list[Counter[str]],
    idf: dict[str, float],
) -> tuple[list[dict[str, float]], dict[str, list[tuple[int, float]]]]:
    doc_vecs: list[dict[str, float]] = []
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for doc_idx, tf in enumerate(doc_tfs):
        vec = {term: tfidf_weight(cnt, idf.get(term, 0.0)) for term, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {term: val / norm for term, val in vec.items() if val > 0}
        else:
            vec = {}
        doc_vecs.append(vec)
        for term, val in vec.items():
            postings[term].append((doc_idx, val))

    return doc_vecs, postings


def bm25_score(
    query_terms: Iterable[str],
    doc_tf: Counter[str],
    doc_len: int,
    avgdl: float,
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    denom_const = k1 * (1.0 - b + b * (doc_len / max(avgdl, 1e-12)))
    for term in query_terms:
        tf = doc_tf.get(term, 0)
        if tf <= 0:
            continue
        score += idf.get(term, 0.0) * (tf * (k1 + 1.0)) / (tf + denom_const)
    return score


def select_query_terms(
    doc_tf: Counter[str],
    idf: dict[str, float],
    max_query_terms: int,
) -> list[str]:
    terms = list(doc_tf.keys())
    terms.sort(key=lambda t: idf.get(t, 0.0), reverse=True)
    if max_query_terms > 0:
        terms = terms[:max_query_terms]
    return terms


def dense_candidates_for_query(
    query_idx: int,
    query_vec: dict[str, float],
    query_terms: list[str],
    postings: dict[str, list[tuple[int, float]]],
    samples: list[CaptionSample],
    dense_top_k: int,
    max_posting_docs: int,
    exclude_same_video: bool,
) -> tuple[list[tuple[int, float]], int, int]:
    scores: dict[int, float] = defaultdict(float)
    skipped_terms = 0
    raw_hits = 0
    anchor_video = samples[query_idx].video_id

    for term in query_terms:
        q_weight = query_vec.get(term, 0.0)
        if q_weight <= 0:
            continue
        posting = postings.get(term, [])
        if max_posting_docs > 0 and len(posting) > max_posting_docs:
            skipped_terms += 1
            continue
        raw_hits += len(posting)
        for doc_idx, doc_weight in posting:
            if doc_idx == query_idx:
                continue
            if exclude_same_video and samples[doc_idx].video_id == anchor_video:
                continue
            scores[doc_idx] += q_weight * doc_weight

    if not scores:
        return [], skipped_terms, raw_hits

    top_k = min(max(1, dense_top_k), len(scores))
    top = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
    return top, skipped_terms, raw_hits


def fallback_negative(
    query_idx: int,
    samples: list[CaptionSample],
    rng: random.Random,
    exclude_same_video: bool,
) -> int:
    anchor_video = samples[query_idx].video_id
    if len(samples) <= 1:
        raise ValueError("Need at least two samples for fallback negative selection")

    for _ in range(1000):
        idx = rng.randrange(len(samples))
        if idx == query_idx:
            continue
        if exclude_same_video and samples[idx].video_id == anchor_video:
            continue
        return idx

    for idx, sample in enumerate(samples):
        if idx == query_idx:
            continue
        if exclude_same_video and sample.video_id == anchor_video:
            continue
        return idx

    raise ValueError(f"No valid fallback negative for sample {query_idx}")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def checkpoint_path_for(args: argparse.Namespace) -> str:
    if args.checkpoint_path:
        return args.checkpoint_path
    return args.output + ".checkpoint.json"


def build_meta(args: argparse.Namespace) -> dict:
    return {
        "task": "msrvtt_query_hard_negative",
        "train_csv": os.path.abspath(args.train_csv),
        "data_path": os.path.abspath(args.data_path),
        "unfold_sentences": args.unfold_sentences,
        "dense_backend": "tfidf_cosine",
        "rerank_backend": "bm25",
        "dense_top_k": args.dense_top_k,
        "target_rank": args.target_rank,
        "max_query_terms": args.max_query_terms,
        "max_posting_docs": args.max_posting_docs,
        "exclude_same_video": args.exclude_same_video,
        "keep_stopwords": args.keep_stopwords,
        "min_token_len": args.min_token_len,
        "limit_samples": args.limit_samples,
        "seed": args.seed,
    }


def load_resume_mapping(args: argparse.Namespace, expected_meta: dict) -> tuple[dict[str, dict], str | None]:
    checkpoint_path = checkpoint_path_for(args)
    for path in [checkpoint_path, args.output]:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or not isinstance(obj.get("mapping"), dict):
            raise ValueError(f"Resume file has no mapping object: {path}")
        old_meta = obj.get("meta", {})
        mismatches = []
        for key, expected_val in expected_meta.items():
            if old_meta.get(key) != expected_val:
                mismatches.append((key, old_meta.get(key), expected_val))
        if mismatches:
            details = "; ".join(
                f"{key}: checkpoint={old!r}, current={new!r}"
                for key, old, new in mismatches[:8]
            )
            raise ValueError(f"Checkpoint is not compatible with current args: {path}; {details}")
        mapping = obj["mapping"]
        print(f"[hardneg] resume loaded {len(mapping)} entries from {path}", flush=True)
        return mapping, path
    return {}, None


def compute_final_stats(
    train_video_count: int,
    samples: list[CaptionSample],
    mapping: dict[str, dict],
    skipped_term_counts: list[int],
    elapsed_seconds: float,
) -> dict:
    values = list(mapping.values())
    dense_scores = [float(v.get("dense_score", 0.0) or 0.0) for v in values]
    bm25_scores = [float(v.get("bm25_score", 0.0) or 0.0) for v in values]
    candidate_counts = [int(v.get("candidate_count", 0) or 0) for v in values]
    fallback_count = sum(1 for v in values if v.get("dense_rank") is None or v.get("bm25_rank") is None)

    return {
        "num_train_videos": train_video_count,
        "num_samples": len(samples),
        "num_unique_videos_in_samples": len({s.video_id for s in samples}),
        "mapping_size": len(mapping),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / max(1, len(mapping)),
        "candidate_count_mean": statistics.fmean(candidate_counts) if candidate_counts else 0.0,
        "candidate_count_p50": percentile([float(v) for v in candidate_counts], 0.5),
        "candidate_count_p90": percentile([float(v) for v in candidate_counts], 0.9),
        "dense_score_mean": statistics.fmean(dense_scores) if dense_scores else 0.0,
        "dense_score_p50": percentile(dense_scores, 0.5),
        "bm25_score_mean": statistics.fmean(bm25_scores) if bm25_scores else 0.0,
        "bm25_score_p50": percentile(bm25_scores, 0.5),
        "skipped_query_terms_mean": statistics.fmean(skipped_term_counts) if skipped_term_counts else 0.0,
        "elapsed_seconds": elapsed_seconds,
    }


def write_checkpoint(
    args: argparse.Namespace,
    meta: dict,
    mapping: dict[str, dict],
    processed_until: int,
    elapsed_seconds: float,
) -> None:
    checkpoint_path = checkpoint_path_for(args)
    obj = {
        "meta": meta,
        "stats": {
            "partial": True,
            "processed_until": processed_until,
            "mapping_size": len(mapping),
            "elapsed_seconds": elapsed_seconds,
        },
        "mapping": mapping,
    }
    write_json(obj, checkpoint_path)
    print(
        f"[hardneg] checkpoint wrote {checkpoint_path} "
        f"processed_until={processed_until} mapping_size={len(mapping)}",
        flush=True,
    )


def build_mapping(args: argparse.Namespace) -> dict:
    start_time = time.time()
    meta = build_meta(args)
    train_video_ids = read_train_video_ids(args.train_csv)
    samples = load_caption_samples(
        args.data_path,
        train_video_ids,
        unfold_sentences=args.unfold_sentences,
        limit_samples=args.limit_samples,
    )
    print(
        f"[hardneg] loaded train_videos={len(train_video_ids)} samples={len(samples)} "
        f"unfold_sentences={args.unfold_sentences}",
        flush=True,
    )

    doc_tfs, df, doc_lens, avgdl = build_statistics(
        samples,
        min_token_len=args.min_token_len,
        keep_stopwords=args.keep_stopwords,
    )
    idf = compute_idf(df, len(samples))
    doc_vecs, postings = build_tfidf_postings(doc_tfs, idf)
    print(
        f"[hardneg] vocab={len(idf)} avgdl={avgdl:.2f} postings_terms={len(postings)}",
        flush=True,
    )

    rng = random.Random(args.seed)
    if args.resume:
        mapping, _resume_source = load_resume_mapping(args, meta)
    else:
        mapping = {}
    skipped_term_counts: list[int] = []
    fallback_count = 0
    new_since_checkpoint = 0

    for idx, sample in enumerate(samples):
        key = str(sample.sample_index)
        if key in mapping:
            continue

        query_terms = select_query_terms(doc_tfs[idx], idf, args.max_query_terms)
        dense_top, skipped_terms, raw_hits = dense_candidates_for_query(
            query_idx=idx,
            query_vec=doc_vecs[idx],
            query_terms=query_terms,
            postings=postings,
            samples=samples,
            dense_top_k=args.dense_top_k,
            max_posting_docs=args.max_posting_docs,
            exclude_same_video=args.exclude_same_video,
        )
        skipped_term_counts.append(skipped_terms)

        if dense_top:
            dense_rank_by_doc = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(dense_top)}
            dense_score_by_doc = dict(dense_top)
            bm25_ranked = []
            for doc_idx, dense_score in dense_top:
                score = bm25_score(
                    query_terms=query_terms,
                    doc_tf=doc_tfs[doc_idx],
                    doc_len=doc_lens[doc_idx],
                    avgdl=avgdl,
                    idf=idf,
                )
                bm25_ranked.append((doc_idx, score, dense_score))
            bm25_ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
            rank_pos = min(max(1, args.target_rank), len(bm25_ranked)) - 1
            hard_idx, hard_bm25, hard_dense = bm25_ranked[rank_pos]
            bm25_rank = rank_pos + 1
            dense_rank = dense_rank_by_doc.get(hard_idx)
            dense_score = dense_score_by_doc.get(hard_idx, hard_dense)
        else:
            hard_idx = fallback_negative(idx, samples, rng, args.exclude_same_video)
            hard_bm25 = 0.0
            dense_score = 0.0
            dense_rank = None
            bm25_rank = None
            raw_hits = 0
            fallback_count += 1

        hard = samples[hard_idx]
        item = {
            "anchor_index": sample.sample_index,
            "anchor_video_id": sample.video_id,
            "anchor_sen_id": sample.sen_id,
            "anchor_json_sentence_index": sample.json_sentence_index,
            "hard_index": hard.sample_index,
            "hard_video_id": hard.video_id,
            "hard_sen_id": hard.sen_id,
            "hard_json_sentence_index": hard.json_sentence_index,
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
            "dense_score": round(float(dense_score), 8),
            "bm25_score": round(float(hard_bm25), 8),
            "candidate_count": len(dense_top),
            "raw_posting_hits": raw_hits,
            "skipped_query_terms": skipped_terms,
        }
        if args.include_captions:
            item["anchor_caption"] = sample.caption
            item["hard_caption"] = hard.caption
        mapping[key] = item
        new_since_checkpoint += 1

        if args.progress_interval > 0 and (idx + 1) % args.progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"[hardneg] processed {idx + 1}/{len(samples)} "
                f"fallback={fallback_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if args.checkpoint_interval > 0 and new_since_checkpoint >= args.checkpoint_interval:
            write_checkpoint(
                args=args,
                meta=meta,
                mapping=mapping,
                processed_until=idx + 1,
                elapsed_seconds=time.time() - start_time,
            )
            new_since_checkpoint = 0

    stats = compute_final_stats(
        train_video_count=len(train_video_ids),
        samples=samples,
        mapping=mapping,
        skipped_term_counts=skipped_term_counts,
        elapsed_seconds=time.time() - start_time,
    )

    return {
        "meta": meta,
        "stats": stats,
        "mapping": mapping,
    }


def write_json(obj: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    if args.dense_top_k <= 0:
        raise ValueError("--dense_top_k must be > 0")
    if args.target_rank <= 0:
        raise ValueError("--target_rank must be > 0")

    result = build_mapping(args)
    write_json(result, args.output)
    checkpoint_path = checkpoint_path_for(args)
    if not args.keep_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    stats = result["stats"]
    print(f"[hardneg] wrote {args.output}", flush=True)
    print(
        "[hardneg] summary: "
        f"samples={stats['num_samples']} "
        f"fallback={stats['fallback_count']} ({stats['fallback_rate']:.4%}) "
        f"dense_mean={stats['dense_score_mean']:.4f} "
        f"bm25_mean={stats['bm25_score_mean']:.4f} "
        f"elapsed={stats['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[hardneg] interrupted", file=sys.stderr)
        raise
