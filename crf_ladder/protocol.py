"""Frozen CRF ladder protocol: levels, x264 recipe, split, and output path convention.

Referenced by every downstream stage as the single source of truth for the CRF
degradation contract. Nothing here performs I/O other than the pure functions
`split_crf_dev_audit` and `crf_output_path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TRAIN_CRFS: tuple[int, int, int] = (30, 38, 46)
EVAL_CRFS: tuple[int, int, int, int, int] = (34, 36, 42, 44, 50)

DEV_SIZE: int = 250
AUDIT_SIZE: int = 250


@dataclass(frozen=True)
class X264Recipe:
    preset: str = "veryfast"
    pixel_format: str = "yuv420p"
    color_range: str = "mpeg"
    color_primaries: str = "smpte170m"
    color_trc: str = "smpte170m"
    colorspace: str = "smpte170m"
    time_base_policy: str = "inverse_input_fps"
    keyint: int = 48
    min_keyint: int = 48
    scenecut: int = 0
    b_frames: int = 2


X264_RECIPE: X264Recipe = X264Recipe()


def split_crf_dev_audit(val_video_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Slice the trusted-v1 `val_video_ids` into CRF-dev (first 250) and CRF-audit (last 250).

    Order-preserving; does not shuffle. The trusted-v1 manifest schema is not
    mutated — this slicing is the only bridge between the trusted-v1 split and
    the CRF experiment split.
    """

    ids = list(val_video_ids)
    if len(ids) != DEV_SIZE + AUDIT_SIZE:
        raise ValueError(
            f"expected {DEV_SIZE + AUDIT_SIZE} val_video_ids, got {len(ids)}"
        )
    return ids[:DEV_SIZE], ids[DEV_SIZE:]


def crf_output_path(root: str | Path, crf: int, video_id: str) -> Path:
    """Return the canonical output path for a degraded video: ``<root>/crf<crf>/<video_id>.mp4``."""

    return Path(root) / f"crf{crf}" / f"{video_id}.mp4"
