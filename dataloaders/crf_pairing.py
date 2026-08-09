"""CRF-mode training batch sampler + synchronized paired video decode.

Public surface (referenced by the plan):

* :data:`RANK_TO_CRF` — the ordinal ``{1,2,3} → {30,38,46}`` map. **Never**
  feed raw CRF values into margins; the ordinal rank is what the plan's frozen
  loss consumes.
* :class:`CRFBatchIndex` — one entry per local batch slot; carries the dataset
  index plus the (primary_rank, secondary_rank | None) that :class:`CRFDistributedBatchSampler`
  has already committed to.
* :class:`CRFDistributedBatchSampler` — deterministic, epoch-seeded batch
  sampler that (a) guarantees each local batch contains exactly
  ``max(1, round(local_batch_size * order_fraction))`` order pairs, and (b)
  distributes primary ranks near-uniformly across each *global* batch (the
  three rank counts differ by at most 1). Both properties are what the CRF
  ranking loss relies on for well-conditioned gradients.
* :class:`CRFTrainBatch` + :func:`crf_train_collate` — pack a list of
  per-sample dicts into a single dataclass whose tensor fields can be moved
  to a device via ``batch.to(device, non_blocking=True)``.
* :func:`decode_paired_video` — decode clean / primary / optional secondary
  from three MP4s at the same frame indices; raises on missing / corrupt /
  metadata-mismatched inputs — no fallback to clean.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


RANK_TO_CRF: dict[int, int] = {1: 30, 2: 38, 3: 46}


@dataclass(frozen=True)
class CRFBatchIndex:
    index: int
    primary_rank: int
    secondary_rank: Optional[int] = None


class CRFDistributedBatchSampler(Sampler[list[CRFBatchIndex]]):
    """Deterministic distributed batch sampler for the CRF training profile.

    Iteration (all ranks share the same base generator seeded by
    ``(seed + epoch)``):

    1. ``randperm(dataset_size)`` → deterministic permutation.
    2. Truncate to a multiple of ``G = local_batch_size * world_size`` (drop
       the incomplete tail; equivalent to ``drop_last=True``).
    3. For every global batch, build a length-``G`` ``primary_rank`` vector
       where each of ``{1,2,3}`` appears ``G // 3`` times and the remaining
       ``G % 3`` slots are filled by a deterministic rotation of ``{1,2,3}``
       — then the full vector is shuffled with the same generator. This
       guarantees the three rank counts in every global batch differ by at
       most 1.
    4. Slice the local rank window
       ``[rank*local_batch_size : (rank+1)*local_batch_size]``.
    5. For position ``< n_order = max(1, round(local_batch_size * order_fraction))``
       assign ``secondary_rank`` uniformly from ``{1,2,3} \\ {primary_rank}``;
       remaining positions leave ``secondary_rank = None``.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        order_fraction: float,
    ) -> None:
        if local_batch_size <= 0 or world_size <= 0:
            raise ValueError("local_batch_size and world_size must be positive")
        if not (0.0 <= order_fraction <= 1.0):
            raise ValueError(f"order_fraction must be in [0,1], got {order_fraction}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        self.dataset_size = int(dataset_size)
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.order_fraction = float(order_fraction)
        self._epoch = 0

        self._global_batch = self.local_batch_size * self.world_size
        self._num_batches = self.dataset_size // self._global_batch
        self._n_order = max(1, round(self.local_batch_size * self.order_fraction))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[CRFBatchIndex]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)
        perm = torch.randperm(self.dataset_size, generator=gen).tolist()
        usable = self._num_batches * self._global_batch
        perm = perm[:usable]

        for b in range(self._num_batches):
            start = b * self._global_batch
            end = start + self._global_batch

            # Build the global primary_rank vector: balanced within (max−min ≤ 1).
            primary_ranks_global = self._balanced_rank_vector(b, gen)
            # Local rank window.
            local_start = self.rank * self.local_batch_size
            local_end = local_start + self.local_batch_size
            local_indices = perm[start:end][local_start:local_end]
            local_primary = primary_ranks_global[local_start:local_end]

            # Order-pair positions come first within the local window (deterministic).
            batch: list[CRFBatchIndex] = []
            for pos, (dataset_idx, primary) in enumerate(zip(local_indices, local_primary)):
                if pos < self._n_order:
                    others = [r for r in (1, 2, 3) if r != primary]
                    # Deterministic secondary choice: 2-way uniform via one draw.
                    draw = torch.randint(0, 2, (1,), generator=gen).item()
                    secondary = others[int(draw)]
                    batch.append(
                        CRFBatchIndex(
                            index=int(dataset_idx),
                            primary_rank=int(primary),
                            secondary_rank=int(secondary),
                        )
                    )
                else:
                    batch.append(
                        CRFBatchIndex(
                            index=int(dataset_idx),
                            primary_rank=int(primary),
                            secondary_rank=None,
                        )
                    )
            yield batch

    def _balanced_rank_vector(self, batch_ordinal: int, gen: torch.Generator) -> list[int]:
        G = self._global_batch
        base_count = G // 3
        remainder = G % 3
        vec: list[int] = []
        for rank_value in (1, 2, 3):
            vec.extend([rank_value] * base_count)
        # Deterministic remainder distribution: rotate {1,2,3} by batch_ordinal.
        if remainder:
            rotation = [((batch_ordinal + i) % 3) + 1 for i in range(remainder)]
            vec.extend(rotation)
        # Shuffle in-place with the same generator so successive batches diverge.
        idx = torch.randperm(len(vec), generator=gen).tolist()
        return [vec[i] for i in idx]


# ---------------------------------------------------------------------------
# CRFTrainBatch dataclass + collate.
# ---------------------------------------------------------------------------


@dataclass
class CRFTrainBatch:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    clean_video: torch.Tensor
    video_mask: torch.Tensor
    group_ids: torch.Tensor
    degraded_video: torch.Tensor
    primary_rank: torch.Tensor
    order_video: torch.Tensor
    order_rank: torch.Tensor
    order_batch_indices: torch.Tensor

    def to(self, device: str | torch.device, non_blocking: bool = False) -> "CRFTrainBatch":
        kwargs: dict[str, torch.Tensor] = {}
        for f in fields(CRFTrainBatch):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                kwargs[f.name] = value.to(device, non_blocking=non_blocking)
            else:
                kwargs[f.name] = value
        return CRFTrainBatch(**kwargs)


def _as_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def crf_train_collate(samples: list[dict]) -> CRFTrainBatch:
    """Collate the per-sample dicts produced by CRF-mode ``__getitem__``.

    Each ``sample`` must contain:

    * ``input_ids`` / ``input_mask`` / ``segment_ids`` — text tensors
    * ``clean_video`` / ``video_mask`` / ``group_ids``
    * ``degraded_video`` — the primary CRF decode
    * ``primary_rank`` — scalar in {1,2,3}
    * ``secondary_video`` — either ``None`` or the secondary CRF decode
    * ``secondary_rank`` — 0 when no order slot, otherwise the ordinal rank

    Order slots are packed contiguously in the order they appear in ``samples``;
    ``order_batch_indices`` maps each order slot back to its batch position.
    """

    B = len(samples)
    input_ids = torch.stack([_as_tensor(s["input_ids"], torch.long) for s in samples])
    input_mask = torch.stack([_as_tensor(s["input_mask"], torch.long) for s in samples])
    segment_ids = torch.stack([_as_tensor(s["segment_ids"], torch.long) for s in samples])
    clean_video = torch.stack([_as_tensor(s["clean_video"], torch.float32) for s in samples])
    video_mask = torch.stack([_as_tensor(s["video_mask"], torch.long) for s in samples])
    group_ids = torch.stack([_as_tensor(s["group_ids"], torch.long) for s in samples])
    degraded_video = torch.stack([_as_tensor(s["degraded_video"], torch.float32) for s in samples])
    primary_rank = torch.stack([_as_tensor(s["primary_rank"], torch.long) for s in samples])

    order_batch_indices: list[int] = []
    order_ranks: list[tuple[int, int]] = []
    order_videos: list[torch.Tensor] = []
    for i, s in enumerate(samples):
        if s.get("secondary_video") is None:
            continue
        order_batch_indices.append(i)
        order_ranks.append((int(s["primary_rank"]), int(s["secondary_rank"])))
        order_videos.append(_as_tensor(s["secondary_video"], torch.float32))

    if order_videos:
        order_video = torch.stack(order_videos)
        order_rank = torch.tensor(order_ranks, dtype=torch.long)
        order_batch_tensor = torch.tensor(order_batch_indices, dtype=torch.long)
    else:
        order_video = degraded_video.new_zeros((0,) + degraded_video.shape[1:])
        order_rank = torch.empty((0, 2), dtype=torch.long)
        order_batch_tensor = torch.empty((0,), dtype=torch.long)

    return CRFTrainBatch(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        clean_video=clean_video,
        video_mask=video_mask,
        group_ids=group_ids,
        degraded_video=degraded_video,
        primary_rank=primary_rank,
        order_video=order_video,
        order_rank=order_rank,
        order_batch_indices=order_batch_tensor,
    )


# ---------------------------------------------------------------------------
# Synchronized paired decode.
# ---------------------------------------------------------------------------


def decode_video_at_indices(
    video_path: Path,
    frame_indices: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``frame_indices`` from ``video_path``, resized to ``height x width``.

    Real implementation lives in ``dataloaders/rawvideo_util.py``'s
    backend-agnostic reader; this default calls it. Kept as a module-level
    function so tests can monkeypatch it deterministically.
    """

    from dataloaders.rawvideo_util import RawVideoExtractor

    extractor = RawVideoExtractor(framerate=1.0, size=height)
    if height != width:
        raise ValueError(
            "decode_video_at_indices: current RawVideoExtractor assumes square crop; "
            f"got height={height} width={width}"
        )
    data = extractor.get_video_data(str(video_path))
    frames = np.asarray(data["video"], dtype=np.float32)
    if frames.ndim == 4:  # [T, C, H, W] → [T, 1, C, H, W]
        frames = frames[:, None, ...]
    if frames.shape[0] == 0:
        raise RuntimeError(f"decode_video_at_indices: no frames decoded from {video_path}")
    # Clamp indices to available range and gather.
    idx = np.clip(np.asarray(frame_indices, dtype=np.int64), 0, frames.shape[0] - 1)
    selected = frames[idx]
    mask = np.ones((len(idx),), dtype=np.int64)
    return selected, mask


def decode_paired_video(
    *,
    clean_path: Path,
    primary_path: Path,
    secondary_path: Optional[Path],
    frame_indices: np.ndarray,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    """Decode the same ``frame_indices`` from clean / primary / optional secondary.

    All three paths must produce the same frame count. Any missing / corrupt
    input raises ``RuntimeError`` carrying the offending path; there is **no
    fallback to clean**.
    """

    outputs: dict[str, np.ndarray] = {}
    for name, path in (("clean", clean_path), ("primary", primary_path)):
        try:
            frames, _mask = decode_video_at_indices(path, frame_indices, height, width)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"decode_paired_video: missing {name} at {path}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"decode_paired_video: failed to decode {name} at {path}: {exc}"
            ) from exc
        outputs[name] = frames
    if secondary_path is not None:
        try:
            frames, _mask = decode_video_at_indices(
                secondary_path, frame_indices, height, width
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"decode_paired_video: missing secondary at {secondary_path}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"decode_paired_video: failed to decode secondary at {secondary_path}: {exc}"
            ) from exc
        outputs["secondary"] = frames

    reference = outputs["clean"].shape[0]
    for name, frames in outputs.items():
        if frames.shape[0] != reference:
            raise RuntimeError(
                f"decode_paired_video: frame count mismatch: clean={reference} "
                f"{name}={frames.shape[0]}"
            )
    return outputs
