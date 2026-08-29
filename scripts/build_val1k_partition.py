#!/usr/bin/env python
"""Build the msrvtt_trusted_v1_val1k partition for study oq1-kl-variance-hubness-v1.

Frozen protocol: experiments/oq1-kl-variance-hubness-v1/study.yaml (data_plane_splits).

Rule (deterministic, no free parameters):
  1. Take the 8500 trusted-v1 train video ids.
  2. Sort lexicographically, permute with numpy RandomState(seed=0), take the
     first 1000 as val1k videos; the remaining 7500 are train7500.
  3. For each val1k video, take the FIRST caption in MSRVTT_v2.json sentence
     order (lowest index / sen_id) -- "lowest_index_in_annotation_list".

Hard-fail assertions (study.yaml stage0):
  - val1k video ids disjoint from MSRVTT JSFUSION test video ids
  - val1k disjoint from train7500, and their union == train8500

Outputs (data/generated/msrvtt_trusted_v1_val1k/):
  train.csv     7500 rows, column: video_id       (same schema as trusted_v1 train.csv)
  val1k.csv     1000 rows, columns: video_id,sentence (same schema as trusted_v1 val.csv)
  manifest.json rule, seed, counts, sha256 of inputs/outputs, assertion results
"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC_TRAIN = REPO / "data/generated/msrvtt_trusted_v1/train.csv"
TEST_CSV = Path("/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv")
ANNOT = Path("/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json")
OUT_DIR = REPO / "data/generated/msrvtt_trusted_v1_val1k"

SEED = 0
N_VAL = 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with open(SRC_TRAIN, newline="") as f:
        rows = list(csv.DictReader(f))
    train8500 = [r["video_id"] for r in rows]
    assert len(train8500) == 8500, f"expected 8500 train videos, got {len(train8500)}"
    assert len(set(train8500)) == 8500, "duplicate video ids in trusted-v1 train.csv"

    with open(TEST_CSV, newline="") as f:
        test_ids = {r["video_id"] for r in csv.DictReader(f)}
    assert len(test_ids) == 1000, f"expected 1000 test videos, got {len(test_ids)}"

    ordered = sorted(train8500)
    perm = np.random.RandomState(SEED).permutation(len(ordered))
    val1k_ids = sorted(ordered[i] for i in perm[:N_VAL])
    train7500_ids = sorted(ordered[i] for i in perm[N_VAL:])

    # --- hard-fail assertions (study.yaml stage0_qualification) ---
    inter_test = set(val1k_ids) & test_ids
    assert not inter_test, f"val1k overlaps JSFUSION test: {sorted(inter_test)[:5]} ..."
    inter_train = set(val1k_ids) & set(train7500_ids)
    assert not inter_train, "val1k overlaps train7500"
    assert set(val1k_ids) | set(train7500_ids) == set(train8500), "partition is not exact"
    assert not (set(train7500_ids) & test_ids), "train7500 overlaps JSFUSION test"

    first_caption = {}
    with open(ANNOT) as f:
        sentences = json.load(f)["sentences"]
    for s in sentences:  # annotation order; keep only the first hit per video
        vid = s["video_id"]
        if vid not in first_caption:
            first_caption[vid] = s["caption"]
    missing = [v for v in val1k_ids if v not in first_caption]
    assert not missing, f"videos without captions: {missing[:5]}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = OUT_DIR / "train.csv"
    val_out = OUT_DIR / "val1k.csv"
    with open(train_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id"])
        for vid in train7500_ids:
            w.writerow([vid])
    with open(val_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "sentence"])
        for vid in val1k_ids:
            w.writerow([vid, first_caption[vid]])

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "study": "oq1-kl-variance-hubness-v1",
        "partition_ids": {
            "train": "msrvtt_trusted_v1_train7500",
            "val": "msrvtt_trusted_v1_val1k",
        },
        "rule": {
            "source": "data/generated/msrvtt_trusted_v1/train.csv",
            "video_selection": "sort ids lexicographically, numpy RandomState(0) "
                               "permutation, first 1000 -> val1k, rest -> train7500",
            "seed": SEED,
            "caption_selection": "lowest_index_in_annotation_list "
                                 "(first occurrence in MSRVTT_v2.json sentences)",
        },
        "counts": {"train7500": len(train7500_ids), "val1k": len(val1k_ids)},
        "assertions": {
            "val1k_disjoint_from_jsfusion_test": True,
            "val1k_disjoint_from_train7500": True,
            "union_equals_train8500": True,
            "train7500_disjoint_from_jsfusion_test": True,
        },
        "inputs_sha256": {
            str(SRC_TRAIN.relative_to(REPO)): sha256(SRC_TRAIN),
            str(TEST_CSV): sha256(TEST_CSV),
            str(ANNOT): sha256(ANNOT),
        },
        "outputs_sha256": {
            "train.csv": sha256(train_out),
            "val1k.csv": sha256(val_out),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head_at_build": git_sha,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"train7500 -> {train_out}  sha256 {manifest['outputs_sha256']['train.csv'][:16]}")
    print(f"val1k     -> {val_out}  sha256 {manifest['outputs_sha256']['val1k.csv'][:16]}")
    print("all assertions passed")


if __name__ == "__main__":
    sys.exit(main())
