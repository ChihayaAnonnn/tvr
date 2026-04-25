#!/usr/bin/env python3
"""
Compute per-token IDF weights from MSRVTT attribute corpus
and save as a PyTorch tensor (shape [vocab_size]).

Usage:
    python scripts/build_idf_weights.py \
        --attr_files deploy_qwen/attributes/msrvtt/final/msrvtt_train9k_attributes.json \
                     deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json \
        --output deploy_qwen/attributes/msrvtt/final/msrvtt_idf_weights.pt
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.tokenization_clip import SimpleTokenizer


VOCAB_SIZE = 49408


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attr_files", nargs="+", required=True,
        help="Attribute JSON files (key=video_id, value=text)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for .pt file",
    )
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()

    df = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    n_docs = 0

    for fpath in args.attr_files:
        print(f"Processing {fpath} ...")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for vid, text in data.items():
            token_ids = tokenizer.encode(text)
            unique_ids = set(token_ids)
            for tid in unique_ids:
                df[tid] += 1
            n_docs += 1

    print(f"Total documents: {n_docs}")
    print(f"Tokens with df>0: {(df > 0).sum().item()}")

    # IDF = log(N / (1 + df))
    idf = torch.log(torch.tensor(float(n_docs)) / (1.0 + df))

    # Tokens never seen get maximal IDF; cap at 0 (shouldn't go negative since df <= N)
    idf = idf.clamp(min=0.0)

    # Normalize to [0, 1] for stable training
    idf_max = idf.max()
    if idf_max > 0:
        idf = idf / idf_max

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(idf, args.output)
    print(f"Saved IDF weights to {args.output}  (shape={idf.shape})")

    # Print top-10 highest and lowest IDF tokens for sanity check
    nonzero_mask = df > 0
    idf_nonzero = idf[nonzero_mask]
    ids_nonzero = torch.arange(VOCAB_SIZE)[nonzero_mask]

    sorted_idx = idf_nonzero.argsort()
    print("\n--- Lowest IDF (most common) ---")
    for i in range(min(15, len(sorted_idx))):
        tid = ids_nonzero[sorted_idx[i]].item()
        print(f"  token_id={tid:5d}  idf={idf[tid]:.4f}  df={df[tid]:.0f}  word={tokenizer.decode([tid])!r}")

    print("\n--- Highest IDF (rarest) ---")
    for i in range(min(15, len(sorted_idx))):
        tid = ids_nonzero[sorted_idx[-(i + 1)]].item()
        print(f"  token_id={tid:5d}  idf={idf[tid]:.4f}  df={df[tid]:.0f}  word={tokenizer.decode([tid])!r}")


if __name__ == "__main__":
    main()
