"""Shared utilities for loading hard-negative mapping JSON files."""

from __future__ import annotations

import json
import os


def load_hard_negative_index(hard_negative_path: str, dataset_len: int) -> list[int]:
    """Load a mapping JSON as an anchor-index -> hard-index list.

    Invalid, missing, self-referential, and out-of-range links are represented
    by ``-1`` so callers can safely ignore them.
    """

    if dataset_len < 0:
        raise ValueError(f"dataset_len must be non-negative, got {dataset_len}")
    if not hard_negative_path:
        raise ValueError("hard_negative_path must be provided")
    if not os.path.exists(hard_negative_path):
        raise FileNotFoundError(f"hard negative mapping not found: {hard_negative_path}")

    with open(hard_negative_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping = obj.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"hard negative file has no mapping object: {hard_negative_path}")

    hard_index = [-1] * int(dataset_len)
    for key, item in mapping.items():
        if not isinstance(item, dict):
            continue
        try:
            anchor = int(item.get("anchor_index", key))
            hard = int(item["hard_index"])
        except (TypeError, ValueError, KeyError):
            continue
        if anchor < 0 or anchor >= dataset_len or hard < 0 or hard >= dataset_len:
            continue
        if hard == anchor:
            continue
        hard_index[anchor] = hard

    return hard_index
