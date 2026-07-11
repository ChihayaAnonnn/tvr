"""Deterministic, process-safe cache for preprocessed TQFS video frames."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

CACHE_VERSION = "msrvtt-tqfs-frames-v1"
ALGORITHM_VERSION = "tqfs-distinct-k-v2"


def _normalized_config(
    features_path,
    feature_framerate,
    max_frames,
    image_resolution,
):
    return {
        "cache_version": CACHE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "features_path": str(Path(features_path).resolve()),
        "feature_framerate": float(feature_framerate),
        "max_frames": int(max_frames),
        "image_resolution": int(image_resolution),
        "dtype": "float32",
        "layout": "frames,channels,height,width",
        "normalization": "openai-clip",
    }


class TQFSFrameCache:
    """Store one exact preprocessed frame tensor per video ID.

    Each entry is an atomic ``.npy`` file so all DDP ranks and DataLoader
    workers can safely populate the same cache concurrently.
    """

    def __init__(
        self,
        root,
        *,
        features_path,
        feature_framerate,
        max_frames,
        image_resolution,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_frames = int(max_frames)
        self.image_resolution = int(image_resolution)
        self.config = _normalized_config(
            features_path,
            feature_framerate,
            max_frames,
            image_resolution,
        )
        self._ensure_config()

    def _ensure_config(self):
        path = self.root / "cache_config.json"
        encoded = json.dumps(
            self.config, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != self.config:
                raise ValueError(
                    f"TQFS cache config mismatch at {path}: "
                    f"existing={existing} requested={self.config}"
                )
            return

        fd, temporary_name = tempfile.mkstemp(
            prefix=".cache_config.", suffix=".tmp", dir=self.root
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != self.config:
                    raise ValueError(
                        f"TQFS cache config mismatch at {path}: "
                        f"existing={existing} requested={self.config}"
                    )
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_video_id(video_id):
        video_id = str(video_id)
        if not video_id or Path(video_id).name != video_id:
            raise ValueError(f"invalid cache video_id={video_id!r}")
        return video_id

    def entry_path(self, video_id):
        return self.root / f"{self._validate_video_id(video_id)}.npy"

    def _validate_array(self, array, path):
        expected_tail = (3, self.image_resolution, self.image_resolution)
        if array.dtype != np.float32:
            raise ValueError(f"invalid TQFS cache dtype at {path}: {array.dtype}")
        if array.ndim != 4 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(f"invalid TQFS cache shape at {path}: {array.shape}")
        if not 1 <= int(array.shape[0]) <= self.max_frames:
            raise ValueError(
                f"invalid TQFS cache frame count at {path}: {array.shape[0]}"
            )

    def load(self, video_id):
        path = self.entry_path(video_id)
        if not path.is_file():
            return None
        array = np.load(path, allow_pickle=False)
        self._validate_array(array, path)
        return torch.from_numpy(array)

    def store(self, video_id, video):
        path = self.entry_path(video_id)
        if path.is_file():
            return self.load(video_id)

        if isinstance(video, torch.Tensor):
            array = video.detach().cpu().contiguous().numpy()
        else:
            array = np.asarray(video)
        if array.dtype != np.float32:
            raise ValueError(f"invalid TQFS cache dtype at {path}: {array.dtype}")
        array = np.ascontiguousarray(array)
        self._validate_array(array, path)

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=self.root
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return torch.from_numpy(array)
