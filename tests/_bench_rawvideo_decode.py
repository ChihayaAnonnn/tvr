"""Standalone equivalence + benchmark for the video decoding pipeline.

Compares three implementations:
  - `old`   : the pre-fix per-frame-seek OpenCV code (verbatim snapshot below)
  - `cv2`   : the fixed sequential-decode OpenCV path (default backend)
  - `decord`: the decord-backed backend (opt-in via TVR_VIDEO_BACKEND=decord)

Reports:
  - `old` vs `cv2`   : must be bit-identical (max_abs_diff == 0)
  - `cv2` vs `decord`: expected close-but-not-identical (different codec
                       backends); we report max pixel drift for transparency
  - wall-clock speedups for each pair

Not a pytest so we can drive it against the tvr conda env directly.
"""

import glob
import os
import sys
import time

import cv2
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

sys.path.insert(0, "/data2/hxj/project/UATVR")

from dataloaders.rawvideo_util import (  # noqa: E402
    RawVideoExtractorCV2,
    RawVideoExtractorDecord,
)


def _old_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def old_video_to_tensor(video_file, preprocess, sample_fp=1,
                        start_time=None, end_time=None):
    """Verbatim copy of the pre-fix implementation."""
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration
    if start_time is not None:
        start_sec, end_sec = start_time, (end_time if end_time <= total_duration
                                          else total_duration)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    interval = 1
    if sample_fp > 0:
        interval = fps // sample_fp
    else:
        sample_fp = fps
    if interval == 0:
        interval = 1
    inds = [int(ind) for ind in range(0, fps, interval)]
    assert len(inds) >= sample_fp
    inds = inds[:sample_fp]
    ret = True
    images = []
    for sec in range(start_sec, end_sec + 1):
        if not ret:
            break
        sec_base = int(sec * fps)
        for ind in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(sec_base + int(ind)))
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
    cap.release()
    if len(images) > 0:
        video_data = th.tensor(np.stack(images))
    else:
        video_data = th.zeros(1)
    return {'video': video_data}


def compare(path, sample_fp):
    old_pre = _old_transform(224)
    cv2_ex = RawVideoExtractorCV2(size=224, framerate=sample_fp)
    de_ex = RawVideoExtractorDecord(size=224, framerate=sample_fp)

    old_out = old_video_to_tensor(path, old_pre, sample_fp=sample_fp)['video']
    cv2_out = cv2_ex.get_video_data(path)['video']
    de_out = de_ex.get_video_data(path)['video']

    if old_out.shape != cv2_out.shape or old_out.shape != de_out.shape:
        return (False, False, f"shape mismatch "
                              f"old={old_out.shape} cv2={cv2_out.shape} de={de_out.shape}")
    d_old_cv2 = (old_out.float() - cv2_out.float()).abs().max().item()
    d_cv2_de = (cv2_out.float() - de_out.float()).abs().max().item()
    return (d_old_cv2 < 1e-5,     # cv2 MUST be bit-identical to old
            d_cv2_de < 0.05,      # decord expected within ~0.05 of normalized value
            f"old↔cv2={d_old_cv2:.2e}  cv2↔decord={d_cv2_de:.2e}")


def bench(path, sample_fp, iters=5):
    old_pre = _old_transform(224)
    cv2_ex = RawVideoExtractorCV2(size=224, framerate=sample_fp)
    de_ex = RawVideoExtractorDecord(size=224, framerate=sample_fp)

    # Warm the OS page cache so we measure decoding, not disk IO.
    old_video_to_tensor(path, old_pre, sample_fp=sample_fp)
    cv2_ex.get_video_data(path)
    de_ex.get_video_data(path)

    t0 = time.perf_counter()
    for _ in range(iters):
        old_video_to_tensor(path, old_pre, sample_fp=sample_fp)
    old_dt = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        cv2_ex.get_video_data(path)
    cv2_dt = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        de_ex.get_video_data(path)
    de_dt = (time.perf_counter() - t0) / iters

    return old_dt, cv2_dt, de_dt


def main():
    root = "/data2/hxj/data/MSRVTT/videos/all"
    paths = sorted(glob.glob(os.path.join(root, "*.mp4")))[:8]
    assert paths, "no videos found"

    print("=== Equivalence check ===")
    ok_cv2_count = 0
    ok_de_count = 0
    for p in paths:
        for fp in (1, 4):
            ok_cv2, ok_de, msg = compare(p, fp)
            mark = f"[cv2={'OK' if ok_cv2 else 'FAIL'} decord={'OK' if ok_de else 'FAIL'}]"
            print(f"{mark} fp={fp} {os.path.basename(p)}: {msg}")
            ok_cv2_count += int(ok_cv2)
            ok_de_count += int(ok_de)

    print()
    print("=== Benchmark (mean ms per call, lower = better) ===")
    total_old, total_cv2, total_de = 0.0, 0.0, 0.0
    for p in paths:
        for fp in (1, 4):
            old_dt, cv2_dt, de_dt = bench(p, fp, iters=3)
            total_old += old_dt
            total_cv2 += cv2_dt
            total_de += de_dt
            print(f"fp={fp} {os.path.basename(p)}: "
                  f"old={old_dt*1000:7.2f}ms  "
                  f"cv2={cv2_dt*1000:7.2f}ms  "
                  f"decord={de_dt*1000:7.2f}ms  "
                  f"[cv2/old={old_dt/cv2_dt:5.2f}x  "
                  f"decord/old={old_dt/de_dt:5.2f}x  "
                  f"decord/cv2={cv2_dt/de_dt:5.2f}x]")
    print()
    n = len(paths) * 2
    print(f"TOTAL old={total_old*1000:.1f}ms  "
          f"cv2={total_cv2*1000:.1f}ms  decord={total_de*1000:.1f}ms")
    print(f"  cv2 speedup over old   : {total_old/total_cv2:.2f}x")
    print(f"  decord speedup over old: {total_old/total_de:.2f}x")
    print(f"  decord speedup over cv2: {total_cv2/total_de:.2f}x")
    print(f"Equivalence: cv2={ok_cv2_count}/{n}, decord={ok_de_count}/{n}")


if __name__ == "__main__":
    main()
