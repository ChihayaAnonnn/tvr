#!/usr/bin/env python
"""Cache N uniformly spaced frames for each video on the JSFUSION test grid.

The VLM judge (scripts/vlm_ambiguity_probe.py) has to look at the same 1000
videos thousands of times -- once per (caption, video) pair it is asked about.
Decoding on the fly would dominate the runtime, so decode once here and write
JPEGs. The grid is only 1000 videos, so the cache is small.

Frames are taken at uniform positions over the whole clip rather than from a
fixed window, matching what the dual encoder was trained on.

Usage:
    python scripts/extract_grid_frames.py \
        --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
        --video-dir /data2/hxj/data/MSRVTT/videos/all \
        --out .scratch/grid_frames --frames 8 --side 252
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2


def uniform_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # some containers do not report a count; fall back to a scan
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        idx = [round(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
        return [frames[i] for i in idx]
    want = {round(i * (total - 1) / max(n - 1, 1)) for i in range(n)}
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out.append(f)
        i += 1
    cap.release()
    return out


def resize_pad(img, side):
    """Letterbox to a square so the model never sees a stretched aspect ratio."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))))
    h, w = img.shape[:2]
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--side", type=int, default=252)
    a = ap.parse_args()

    vids = sorted({r["video_id"] for r in csv.DictReader(open(a.test_csv))})
    os.makedirs(a.out, exist_ok=True)
    missing, done = [], 0
    for k, vid in enumerate(vids):
        d = os.path.join(a.out, vid)
        if os.path.isdir(d) and len(os.listdir(d)) == a.frames:
            done += 1
            continue
        src = os.path.join(a.video_dir, vid + ".mp4")
        if not os.path.exists(src):
            missing.append(vid)
            continue
        fs = uniform_frames(src, a.frames)
        if len(fs) < a.frames:  # short clip: repeat the last frame
            fs = fs + [fs[-1]] * (a.frames - len(fs)) if fs else []
        if not fs:
            missing.append(vid)
            continue
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(fs[: a.frames]):
            cv2.imwrite(os.path.join(d, f"{i:02d}.jpg"), resize_pad(f, a.side),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        done += 1
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(vids)}", flush=True)
    print(f"cached {done}/{len(vids)} videos, missing {len(missing)}")
    if missing:
        print("missing:", missing[:20])


if __name__ == "__main__":
    main()
