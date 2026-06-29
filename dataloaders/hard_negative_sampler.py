"""Hard-negative-aware batch sampler for distributed retrieval training."""

from __future__ import annotations

import math
import os
import random
from typing import Iterator

from dataloaders.hard_negative_mapping import load_hard_negative_index


class HardNegativeDistributedBatchSampler:
    """Pack anchor samples with their hard negatives in the same global batch.

    The sampler yields local micro-batches for the current DDP rank.  All ranks
    construct the same global batches deterministically and then take disjoint
    contiguous slices, so positives remain aligned while hard negatives enter
    the allgathered in-batch denominator.
    """

    def __init__(
        self,
        dataset,
        hard_negative_path: str,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        shuffle: bool = True,
        log_prefix: str = "[HardNegSampler]",
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if not hard_negative_path:
            raise ValueError("hard_negative_path must be provided")
        if not os.path.exists(hard_negative_path):
            raise FileNotFoundError(f"hard negative mapping not found: {hard_negative_path}")

        self.dataset = dataset
        self.hard_negative_path = hard_negative_path
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.log_prefix = log_prefix
        self.epoch = 0
        self._local_batches: list[list[int]] | None = None
        self._stats: dict[str, float | int] = {}

        self.hard_index = self._load_hard_index()
        valid = sum(1 for idx in self.hard_index if idx >= 0)
        if self.rank == 0:
            print(
                f"{self.log_prefix} loaded {valid}/{len(self.hard_index)} hard-negative links "
                f"from {hard_negative_path}",
                flush=True,
            )

    def _load_hard_index(self) -> list[int]:
        n = len(self.dataset)
        return load_hard_negative_index(self.hard_negative_path, n)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._local_batches = None

    def __iter__(self) -> Iterator[list[int]]:
        if self._local_batches is None:
            self._local_batches = self._build_local_batches()
        if self.rank == 0:
            print(
                f"{self.log_prefix} epoch={self.epoch} batches={len(self._local_batches)} "
                f"hit_rate={self._stats.get('hit_rate', 0.0):.4f} "
                f"paired_anchors={self._stats.get('paired_anchors', 0)} "
                f"dropped={self._stats.get('dropped_samples', 0)}",
                flush=True,
            )
        yield from self._local_batches

    def __len__(self) -> int:
        if self._local_batches is None:
            self._local_batches = self._build_local_batches()
        return len(self._local_batches)

    def _make_units(self, indices: list[int]) -> tuple[list[list[int]], int]:
        used: set[int] = set()
        units: list[list[int]] = []
        paired = 0

        for anchor in indices:
            if anchor in used:
                continue
            hard = self.hard_index[anchor] if 0 <= anchor < len(self.hard_index) else -1
            if hard >= 0 and hard not in used:
                units.append([anchor, hard])
                used.add(anchor)
                used.add(hard)
                paired += 1
            else:
                units.append([anchor])
                used.add(anchor)

        return units, paired

    def _pack_global_batches(self, units: list[list[int]], global_batch_size: int) -> list[list[int]]:
        pairs = [unit for unit in units if len(unit) == 2]
        singles = [unit[0] for unit in units if len(unit) == 1]
        single_pos = 0
        batches: list[list[int]] = []
        cur: list[int] = []

        def fill_with_singles() -> None:
            nonlocal single_pos, cur
            while len(cur) < global_batch_size and single_pos < len(singles):
                cur.append(singles[single_pos])
                single_pos += 1

        for pair in pairs:
            if len(cur) + 2 > global_batch_size:
                fill_with_singles()
                if len(cur) == global_batch_size:
                    batches.append(cur)
                elif not self.drop_last and cur:
                    batches.append(cur)
                cur = []
            cur.extend(pair)
            if len(cur) == global_batch_size:
                batches.append(cur)
                cur = []

        while single_pos < len(singles):
            fill_with_singles()
            if len(cur) == global_batch_size:
                batches.append(cur)
                cur = []
            elif not self.drop_last and cur:
                batches.append(cur)
                cur = []

        if cur and not self.drop_last:
            batches.append(cur)

        return batches

    def _build_local_batches(self) -> list[list[int]]:
        n = len(self.dataset)
        indices = list(range(n))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        global_batch_size = self.batch_size * self.num_replicas
        units, paired_anchors = self._make_units(indices)
        global_batches = self._pack_global_batches(units, global_batch_size)

        local_batches = []
        start = self.rank * self.batch_size
        end = start + self.batch_size
        for batch in global_batches:
            if len(batch) < global_batch_size:
                if self.drop_last:
                    continue
                pad_to = int(math.ceil(len(batch) / self.num_replicas) * self.num_replicas)
                if pad_to > len(batch):
                    batch = batch + batch[: pad_to - len(batch)]
            local = batch[start:end]
            if len(local) == self.batch_size or (local and not self.drop_last):
                local_batches.append(local)

        used_count = sum(len(batch) for batch in global_batches)
        dropped = max(0, n - min(n, used_count))
        hit_anchors = self._count_hit_anchors(global_batches)
        self._stats = {
            "paired_anchors": paired_anchors,
            "hit_anchors": hit_anchors,
            "hit_rate": hit_anchors / max(1, used_count),
            "global_batches": len(global_batches),
            "local_batches": len(local_batches),
            "dropped_samples": dropped,
        }
        return local_batches

    def _count_hit_anchors(self, global_batches: list[list[int]]) -> int:
        hits = 0
        for batch in global_batches:
            batch_set = set(batch)
            for anchor in batch:
                hard = self.hard_index[anchor] if 0 <= anchor < len(self.hard_index) else -1
                if hard in batch_set:
                    hits += 1
        return hits
