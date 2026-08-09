"""Task 4 — Distributed CRF batch sampler + synchronized paired decode tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

import numpy as np
import pytest
import torch

from dataloaders.crf_pairing import (
    RANK_TO_CRF,
    CRFBatchIndex,
    CRFDistributedBatchSampler,
    CRFTrainBatch,
    crf_train_collate,
)


def test_rank_to_crf_map_is_frozen():
    assert RANK_TO_CRF == {1: 30, 2: 38, 3: 46}


def test_every_local_batch_has_fixed_order_count():
    sampler = CRFDistributedBatchSampler(
        dataset_size=128,
        local_batch_size=16,
        rank=0,
        world_size=4,
        seed=7,
        order_fraction=0.25,
    )
    for batch in sampler:
        assert len(batch) == 16
        assert sum(item.secondary_rank is not None for item in batch) == max(1, round(16 * 0.25))
        assert all(item.primary_rank in (1, 2, 3) for item in batch)
        assert all(
            item.secondary_rank is None or item.secondary_rank != item.primary_rank
            for item in batch
        )


def test_sampler_is_epoch_deterministic():
    kw = dict(
        dataset_size=128,
        local_batch_size=16,
        rank=1,
        world_size=4,
        seed=7,
        order_fraction=0.20,
    )
    left = CRFDistributedBatchSampler(**kw)
    right = CRFDistributedBatchSampler(**kw)
    left.set_epoch(3)
    right.set_epoch(3)
    assert list(left) == list(right)


def test_sampler_drops_last_partial_global_batch():
    sampler = CRFDistributedBatchSampler(
        dataset_size=130,
        local_batch_size=16,
        rank=0,
        world_size=4,
        seed=0,
        order_fraction=0.20,
    )
    # G = 64; dataset_size=130 -> 2 complete global batches, len == 2.
    assert len(sampler) == 2


def test_primary_rank_counts_are_balanced_across_ranks_within_each_global_batch():
    kw = dict(
        dataset_size=192,
        local_batch_size=16,
        world_size=4,
        seed=11,
        order_fraction=0.25,
    )
    for epoch in range(3):
        per_rank = []
        for r in range(4):
            sampler = CRFDistributedBatchSampler(rank=r, **kw)
            sampler.set_epoch(epoch)
            per_rank.append(list(sampler))
        num_batches = len(per_rank[0])
        for b in range(num_batches):
            global_batch = [item for r in range(4) for item in per_rank[r][b]]
            counts = Counter(item.primary_rank for item in global_batch)
            assert set(counts) == {1, 2, 3}
            assert max(counts.values()) - min(counts.values()) <= 1


def test_primary_rank_marginal_is_near_uniform_across_epochs():
    kw = dict(
        dataset_size=1_536,
        local_batch_size=16,
        world_size=4,
        seed=13,
        order_fraction=0.20,
    )
    counts = Counter()
    for epoch in range(4):
        for r in range(4):
            sampler = CRFDistributedBatchSampler(rank=r, **kw)
            sampler.set_epoch(epoch)
            for batch in sampler:
                counts.update(item.primary_rank for item in batch)
    total = sum(counts.values())
    for rank_value in (1, 2, 3):
        assert abs(counts[rank_value] / total - 1 / 3) < 0.02


def test_order_fraction_one_makes_every_item_an_order_pair():
    sampler = CRFDistributedBatchSampler(
        dataset_size=64,
        local_batch_size=16,
        rank=0,
        world_size=1,
        seed=1,
        order_fraction=1.0,
    )
    for batch in sampler:
        assert all(item.secondary_rank is not None for item in batch)


def test_crf_batch_index_is_hashable_dataclass():
    idx = CRFBatchIndex(index=7, primary_rank=2, secondary_rank=3)
    # frozen -> hashable
    assert hash(idx) == hash(CRFBatchIndex(index=7, primary_rank=2, secondary_rank=3))
    assert idx.primary_rank == 2 and idx.secondary_rank == 3


def _fake_item(*, index: int, primary_rank: int, secondary_rank: int | None):
    """Simulate what MSRVTT_TrainDataLoader.__getitem__ returns in CRF mode."""

    T, C, H, W = 4, 3, 2, 2
    marker_clean = float(index)
    marker_primary = float(index) + 0.1 * primary_rank
    marker_secondary = (
        None if secondary_rank is None else float(index) + 0.1 * secondary_rank
    )
    return {
        "input_ids": np.full((30,), index, dtype=np.int64),
        "input_mask": np.ones((30,), dtype=np.int64),
        "segment_ids": np.zeros((30,), dtype=np.int64),
        "clean_video": np.full((T, 1, C, H, W), marker_clean, dtype=np.float32),
        "video_mask": np.ones((T,), dtype=np.int64),
        "group_ids": np.int64(index),
        "degraded_video": np.full((T, 1, C, H, W), marker_primary, dtype=np.float32),
        "primary_rank": np.int64(primary_rank),
        "secondary_video": (
            None
            if marker_secondary is None
            else np.full((T, 1, C, H, W), marker_secondary, dtype=np.float32)
        ),
        "secondary_rank": (0 if secondary_rank is None else int(secondary_rank)),
    }


def test_crf_collate_packs_order_slots_contiguously_with_traceable_indices():
    samples = [
        _fake_item(index=0, primary_rank=1, secondary_rank=None),
        _fake_item(index=1, primary_rank=2, secondary_rank=3),
        _fake_item(index=2, primary_rank=3, secondary_rank=None),
        _fake_item(index=3, primary_rank=1, secondary_rank=2),
    ]
    batch = crf_train_collate(samples)
    assert isinstance(batch, CRFTrainBatch)
    B = 4
    assert batch.clean_video.shape[0] == B
    assert batch.degraded_video.shape[0] == B
    assert batch.primary_rank.tolist() == [1, 2, 3, 1]
    # Order slots come from samples[1] and samples[3] in order.
    assert batch.order_batch_indices.tolist() == [1, 3]
    assert batch.order_rank.tolist() == [[2, 3], [1, 2]]
    # order_video markers must match secondary markers for samples[1]/[3].
    order_marker_1 = float(1) + 0.1 * 3
    order_marker_3 = float(3) + 0.1 * 2
    assert torch.allclose(batch.order_video[0], torch.full_like(batch.order_video[0], order_marker_1))
    assert torch.allclose(batch.order_video[1], torch.full_like(batch.order_video[1], order_marker_3))


def test_crf_collate_handles_no_order_slots():
    samples = [
        _fake_item(index=0, primary_rank=1, secondary_rank=None),
        _fake_item(index=1, primary_rank=2, secondary_rank=None),
    ]
    batch = crf_train_collate(samples)
    assert batch.order_video.shape[0] == 0
    assert batch.order_rank.shape[0] == 0
    assert batch.order_batch_indices.shape[0] == 0


def test_crf_train_batch_to_moves_tensor_fields_only():
    samples = [
        _fake_item(index=0, primary_rank=1, secondary_rank=2),
        _fake_item(index=1, primary_rank=3, secondary_rank=1),
    ]
    batch = crf_train_collate(samples)
    moved = batch.to("cpu")  # device swap that succeeds on CPU-only test hosts
    assert isinstance(moved, CRFTrainBatch)
    for f in fields(CRFTrainBatch):
        val = getattr(moved, f.name)
        if isinstance(val, torch.Tensor):
            assert val.device == torch.device("cpu")


def test_paired_frame_indices_are_shared(monkeypatch, tmp_path):
    """Fake decoder returns frame-numbered arrays; assert clean/primary/secondary
    select identical watermarks."""

    from dataloaders import crf_pairing

    watermark_calls = []

    def fake_decode(video_path, frame_indices, height, width):
        """Return a stack of frames whose pixel value equals the frame index."""
        marker = tuple(int(i) for i in frame_indices)
        watermark_calls.append((str(video_path), marker))
        arr = np.zeros((len(frame_indices), 1, 3, height, width), dtype=np.float32)
        for k, idx in enumerate(frame_indices):
            arr[k, 0].fill(float(idx))
        return arr, np.ones((len(frame_indices),), dtype=np.int64)

    monkeypatch.setattr(crf_pairing, "decode_video_at_indices", fake_decode)

    clean = tmp_path / "clean.mp4"
    primary = tmp_path / "primary.mp4"
    secondary = tmp_path / "secondary.mp4"
    for p in (clean, primary, secondary):
        p.write_bytes(b"stub")

    indices = np.array([2, 5, 9], dtype=np.int64)
    out = crf_pairing.decode_paired_video(
        clean_path=clean,
        primary_path=primary,
        secondary_path=secondary,
        frame_indices=indices,
        height=4,
        width=4,
    )
    # Same indices requested three times.
    requested = [call for call in watermark_calls if call[0] in (str(clean), str(primary), str(secondary))]
    assert {call[1] for call in requested} == {tuple(indices.tolist())}
    # Frames watermark to the requested indices in each path.
    for name in ("clean", "primary", "secondary"):
        frames = out[name]
        for k, idx in enumerate(indices):
            assert np.allclose(frames[k], float(idx))


def test_paired_decode_refuses_when_degraded_missing(monkeypatch, tmp_path):
    from dataloaders import crf_pairing

    def fake_decode(video_path, frame_indices, height, width):
        if "primary_missing" in str(video_path):
            raise FileNotFoundError(str(video_path))
        arr = np.zeros((len(frame_indices), 1, 3, height, width), dtype=np.float32)
        return arr, np.ones((len(frame_indices),), dtype=np.int64)

    monkeypatch.setattr(crf_pairing, "decode_video_at_indices", fake_decode)
    clean = tmp_path / "clean.mp4"
    primary = tmp_path / "primary_missing.mp4"
    clean.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="primary_missing"):
        crf_pairing.decode_paired_video(
            clean_path=clean,
            primary_path=primary,
            secondary_path=None,
            frame_indices=np.array([0, 1], dtype=np.int64),
            height=4,
            width=4,
        )


def test_paired_decode_reports_frame_count_mismatch(monkeypatch, tmp_path):
    from dataloaders import crf_pairing

    def fake_decode(video_path, frame_indices, height, width):
        # primary path returns fewer frames -> mismatch.
        if "primary" in str(video_path):
            arr = np.zeros((len(frame_indices) - 1, 1, 3, height, width), dtype=np.float32)
            return arr, np.ones((len(frame_indices) - 1,), dtype=np.int64)
        arr = np.zeros((len(frame_indices), 1, 3, height, width), dtype=np.float32)
        return arr, np.ones((len(frame_indices),), dtype=np.int64)

    monkeypatch.setattr(crf_pairing, "decode_video_at_indices", fake_decode)
    clean = tmp_path / "clean.mp4"
    primary = tmp_path / "primary.mp4"
    clean.write_bytes(b"x")
    primary.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="frame count mismatch"):
        crf_pairing.decode_paired_video(
            clean_path=clean,
            primary_path=primary,
            secondary_path=None,
            frame_indices=np.array([0, 1, 2], dtype=np.int64),
            height=4,
            width=4,
        )
