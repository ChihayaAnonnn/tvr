#!/usr/bin/env python3
"""Precompute the shared trusted-v1 MSRVTT TQFS frame cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.rawvideo_util import RawVideoExtractor  # noqa: E402
from dataloaders.tqfs_cache import TQFSFrameCache  # noqa: E402

_WORKER_EXTRACTOR = None
_WORKER_CACHE = None
_WORKER_FEATURES_PATH = None
_WORKER_MAX_FRAMES = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        default=str(
            PROJECT_ROOT / "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
        ),
    )
    parser.add_argument(
        "--features-path",
        default="/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(
            PROJECT_ROOT / "cache_dir/tqfs/msrvtt_trusted_v1_f1_m8_r224"
        ),
    )
    parser.add_argument("--feature-framerate", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-resolution", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit to build the complete train+val cache.",
    )
    return parser.parse_args()


def _resolve_video_path(features_path, video_id):
    root = Path(features_path)
    mp4_path = root / f"{video_id}.mp4"
    if mp4_path.is_file():
        return mp4_path
    webm_path = root / f"{video_id}.webm"
    if webm_path.is_file():
        return webm_path
    raise FileNotFoundError(f"video file not found for video_id={video_id}")


def _init_worker(config):
    global _WORKER_CACHE, _WORKER_EXTRACTOR, _WORKER_FEATURES_PATH, _WORKER_MAX_FRAMES

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    _WORKER_FEATURES_PATH = config["features_path"]
    _WORKER_MAX_FRAMES = config["max_frames"]
    _WORKER_EXTRACTOR = RawVideoExtractor(
        framerate=config["feature_framerate"],
        size=config["image_resolution"],
    )
    _WORKER_CACHE = TQFSFrameCache(
        config["cache_dir"],
        features_path=config["features_path"],
        feature_framerate=config["feature_framerate"],
        max_frames=config["max_frames"],
        image_resolution=config["image_resolution"],
    )


def _build_one(video_id):
    cached = _WORKER_CACHE.load(video_id)
    if cached is not None:
        return "hit"
    video_path = _resolve_video_path(_WORKER_FEATURES_PATH, video_id)
    video = _WORKER_EXTRACTOR.get_tqfs_video_data(
        str(video_path), _WORKER_MAX_FRAMES
    )["video"]
    if getattr(video, "ndim", 0) != 4:
        raise RuntimeError(f"failed to decode video_id={video_id}: shape={video.shape}")
    _WORKER_CACHE.store(video_id, video)
    return "built"


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    video_ids = [
        *manifest["train_video_ids"],
        *manifest["val_video_ids"],
    ]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("trusted split contains duplicate train/val video IDs")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        video_ids = video_ids[: args.limit]

    config = {
        "features_path": str(Path(args.features_path).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "feature_framerate": args.feature_framerate,
        "max_frames": args.max_frames,
        "image_resolution": args.image_resolution,
    }
    TQFSFrameCache(
        config["cache_dir"],
        features_path=config["features_path"],
        feature_framerate=config["feature_framerate"],
        max_frames=config["max_frames"],
        image_resolution=config["image_resolution"],
    )

    counts = {"built": 0, "hit": 0}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:
        results = executor.map(_build_one, video_ids, chunksize=1)
        for status in tqdm(results, total=len(video_ids), desc="TQFS cache"):
            counts[status] += 1

    print(
        f"TQFS cache complete: total={len(video_ids)} "
        f"built={counts['built']} hit={counts['hit']} dir={config['cache_dir']}"
    )


if __name__ == "__main__":
    main()
