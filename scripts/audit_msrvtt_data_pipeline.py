"""Audit the MSRVTT trusted-v1 data pipeline behind the RSPR experiments.

Answers three questions with measurements rather than inspection:
  1. Does the TQFS frame cache return the same tensor a fresh decode would?
  2. Are the trusted train/val/test splits disjoint, and do the caption
     expansions line up with the reported step count?
  3. How many same-video positives does a realistic training batch contain?
     (the one-to-many premise behind the PCME-style BCE)

Run from the repo root; needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.dataloader_msrvtt_retrieval import (  # noqa: E402
    MSRVTT_TrainDataLoader,
)
from dataloaders.msrvtt_protocol import load_trusted_manifest  # noqa: E402
from dataloaders.rawvideo_util import RawVideoExtractor  # noqa: E402
from dataloaders.tqfs_cache import TQFSFrameCache  # noqa: E402
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402


def _test_video_ids(test_csv):
    import csv

    with open(test_csv, newline="", encoding="utf-8") as handle:
        return [row["video_id"] for row in csv.DictReader(handle)]


def audit_cache(args) -> None:
    print("\n" + "=" * 70)
    print("1. TQFS cache integrity: cached tensor vs fresh decode")
    print("=" * 70)

    cache = TQFSFrameCache(
        args.tqfs_cache_dir,
        features_path=args.features_path,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        image_resolution=224,
    )
    extractor = RawVideoExtractor(framerate=args.feature_framerate, size=224)

    manifest = load_trusted_manifest(args.split_manifest)
    rng = np.random.default_rng(0)
    pools = {
        "train": manifest["train_video_ids"],
        "val": manifest["val_video_ids"],
        "test": _test_video_ids(args.test_csv),
    }
    sampled = []
    for split, ids in pools.items():
        picks = rng.choice(len(ids), size=min(args.cache_samples, len(ids)), replace=False)
        sampled.extend((split, ids[int(i)]) for i in picks)

    mismatches = 0
    for split, video_id in sampled:
        cached = cache.load(video_id)
        if cached is None:
            print(f"  [MISS] {split:5s} {video_id}: no cache entry")
            mismatches += 1
            continue
        video_path = os.path.join(args.features_path, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4", ".webm")
        fresh = extractor.get_tqfs_video_data(video_path, args.max_frames)["video"]
        fresh = torch.as_tensor(np.asarray(fresh, dtype=np.float32))
        if fresh.shape != cached.shape:
            print(f"  [SHAPE] {split:5s} {video_id}: cached={tuple(cached.shape)} fresh={tuple(fresh.shape)}")
            mismatches += 1
            continue
        delta = (fresh - cached).abs().max().item()
        status = "OK " if delta == 0.0 else "DIFF"
        if delta != 0.0:
            mismatches += 1
        print(f"  [{status}] {split:5s} {video_id:10s} shape={tuple(cached.shape)} max|fresh-cached|={delta:.3e}")

    print(f"\n  checked={len(sampled)}  mismatches={mismatches}")

    # A cache keyed only on (features_path, framerate, max_frames, resolution)
    # cannot distinguish slice_framepos / frame_order changes.
    config_path = os.path.join(args.tqfs_cache_dir, "cache_config.json")
    config = json.load(open(config_path, encoding="utf-8"))
    print(f"  cache key fields: {sorted(config)}")
    print("  NOTE: slice_framepos and frame_order are NOT part of the cache key")


def audit_splits(args) -> None:
    print("\n" + "=" * 70)
    print("2. Trusted split disjointness and caption expansion")
    print("=" * 70)

    manifest = load_trusted_manifest(args.split_manifest)
    train = set(manifest["train_video_ids"])
    val = set(manifest["val_video_ids"])
    test = set(_test_video_ids(args.test_csv))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = a & b
        flag = "LEAK" if overlap else "OK  "
        print(f"  [{flag}] {a_name} ∩ {b_name} = {len(overlap)}" + (f" e.g. {sorted(overlap)[:5]}" if overlap else ""))


def audit_batches(args) -> None:
    print("\n" + "=" * 70)
    print("3. Batch composition: same-video positives per training batch")
    print("=" * 70)

    tokenizer = ClipTokenizer()
    dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=32,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=True,
        frame_order=0,
        slice_framepos=3,
        strategy=1,
        split_manifest_path=args.split_manifest,
        tqfs_cache_dir=args.tqfs_cache_dir,
    )
    group_ids = np.array(
        [dataset.video_group_ids[video_id] for video_id, _caption in dataset.sentences_dict.values()],
        dtype=np.int64,
    )
    n_samples = group_ids.size
    n_videos = len(set(group_ids.tolist()))
    captions_per_video = Counter(Counter(group_ids.tolist()).values())
    print(f"  unfolded samples={n_samples}  distinct train videos={n_videos}")
    print(f"  captions-per-video histogram: {dict(sorted(captions_per_video.items()))}")
    print(f"  steps/epoch at effective batch {args.effective_batch}: "
          f"{n_samples / args.effective_batch:.1f}")

    # Replicate the sampler: DistributedSampler(shuffle=True) over 4 ranks,
    # then the per-rank slices are concatenated into the global batch.
    print(f"\n  simulating {args.batch_trials} global batches of {args.effective_batch}"
          f" (DistributedSampler-style shuffle, seed=epoch)")
    total_positive_rows = 0
    total_extra_positives = 0
    total_rows = 0
    max_group = 0
    for epoch in range(args.batch_trials):
        generator = torch.Generator()
        generator.manual_seed(epoch)
        permutation = torch.randperm(n_samples, generator=generator).numpy()
        world = args.world_size
        per_rank = args.effective_batch // world
        rank_stride = int(np.ceil(n_samples / world))
        rank_slices = [permutation[r * rank_stride:(r + 1) * rank_stride] for r in range(world)]
        n_batches = min(len(s) for s in rank_slices) // per_rank
        for b in range(n_batches):
            batch = np.concatenate([s[b * per_rank:(b + 1) * per_rank] for s in rank_slices])
            counts = Counter(group_ids[batch].tolist())
            sizes = np.array(list(counts.values()))
            total_rows += batch.size
            total_positive_rows += int(sizes[sizes > 1].sum())
            total_extra_positives += int((sizes[sizes > 1] - 1).sum())
            max_group = max(max_group, int(sizes.max()))

    share = total_positive_rows / total_rows
    print(f"  rows sharing their video with >=1 other row in the same batch: "
          f"{total_positive_rows}/{total_rows} = {share:.4%}")
    print(f"  mean extra positives per batch: {total_extra_positives / (total_rows / args.effective_batch):.2f}")
    print(f"  largest same-video group observed in one batch: {max_group}")
    print(f"  => a typical row has {1 + total_extra_positives / max(total_positive_rows, 1):.2f} "
          f"positives when it has any, but {1 - share:.2%} of rows are diagonal-only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps")
    parser.add_argument("--data-path", default="/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json")
    parser.add_argument("--train-csv", default="data/generated/msrvtt_trusted_v1/train.csv")
    parser.add_argument("--split-manifest", default="dataloaders/splits/msrvtt_trusted_v1_seed0.json")
    parser.add_argument("--test-csv", default="/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv")
    parser.add_argument("--tqfs-cache-dir", default="/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224")
    parser.add_argument("--feature-framerate", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--cache-samples", type=int, default=4)
    parser.add_argument("--batch-trials", type=int, default=1)
    parser.add_argument("--skip", nargs="*", default=[], choices=["cache", "splits", "batches"])
    args = parser.parse_args()

    if "cache" not in args.skip:
        audit_cache(args)
    if "splits" not in args.skip:
        audit_splits(args)
    if "batches" not in args.skip:
        audit_batches(args)


if __name__ == "__main__":
    main()
