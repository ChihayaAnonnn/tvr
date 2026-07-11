import json
import warnings

import numpy as np
import pytest
import torch

from dataloaders.dataloader_msrvtt_retrieval import _get_tqfs_video_data
from dataloaders.tqfs_cache import TQFSFrameCache
from dataloaders.tqfs_util import select_tqfs_indices


def _cache(tmp_path, **overrides):
    config = {
        "features_path": tmp_path / "videos",
        "feature_framerate": 1,
        "max_frames": 8,
        "image_resolution": 4,
    }
    config.update(overrides)
    return TQFSFrameCache(tmp_path / "cache", **config)


def test_tqfs_cache_roundtrip_and_config_manifest(tmp_path):
    cache = _cache(tmp_path)
    video = torch.randn(8, 3, 4, 4, dtype=torch.float32)

    stored = cache.store("video0", video)
    loaded = cache.load("video0")

    torch.testing.assert_close(stored, video)
    torch.testing.assert_close(loaded, video)
    config = json.loads(
        (tmp_path / "cache/cache_config.json").read_text(encoding="utf-8")
    )
    assert config["max_frames"] == 8
    assert config["algorithm_version"] == "tqfs-distinct-k-v2"


def test_tqfs_cache_rejects_incompatible_configuration(tmp_path):
    _cache(tmp_path)

    with pytest.raises(ValueError, match="config mismatch"):
        _cache(tmp_path, max_frames=4)


@pytest.mark.parametrize(
    "video",
    [
        torch.zeros(8, 3, 4, 4, dtype=torch.float16),
        torch.zeros(8, 4, 4, dtype=torch.float32),
        torch.zeros(9, 3, 4, 4, dtype=torch.float32),
    ],
)
def test_tqfs_cache_rejects_invalid_entries(tmp_path, video):
    cache = _cache(tmp_path)

    with pytest.raises(ValueError, match="invalid TQFS cache"):
        cache.store("video0", video)


def test_tqfs_cache_hit_skips_decode(tmp_path):
    cache = _cache(tmp_path)
    expected = torch.randn(8, 3, 4, 4)
    cache.store("video0", expected)

    class FailExtractor:
        def get_tqfs_video_data(self, *_args, **_kwargs):
            raise AssertionError("cache hit must skip decoding")

    result = _get_tqfs_video_data(
        FailExtractor(), cache, "video0", "video0.mp4", 8
    )

    torch.testing.assert_close(result["video"], expected)


def test_degenerate_tqfs_uses_distinct_cluster_count_without_warning():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(12)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = select_tqfs_indices(frames, 8)

    assert len(selected) == 8
    assert selected == sorted(set(selected))
    assert caught == []
