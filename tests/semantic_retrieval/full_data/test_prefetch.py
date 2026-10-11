"""Real video, ordering, RNG and worker lifecycle contracts for parallel supply."""
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from semantic_retrieval.data.video import decode_video
        video, mask = decode_video(self.paths[index], [0, 2], 16, decoder_threads=1)
        return video, mask, torch.tensor(index)


@pytest.fixture
def videos(tmp_path):
    paths = []
    for i in range(8):
        path = tmp_path/f'{i}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (16, 16))
        assert writer.isOpened()
        for frame in range(3):
            writer.write(np.full((16, 16, 3), 15*i+frame, np.uint8))
        writer.release(); paths.append(str(path))
    return VideoDataset(paths)


def collect(dataset, workers):
    from semantic_retrieval.full_data.loader import batch_loader
    batches = [[5, 1], [6, 0], [7, 3], [2, 4]]
    with batch_loader(dataset, batches, workers=workers, pin_memory=False) as iterator:
        return list(iterator)


def test_real_spawn_prefetch_preserves_batch_pixels_order_and_parent_rng(videos):
    torch.manual_seed(713); before = torch.get_rng_state().clone()
    expected = collect(videos, 0); after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual = collect(videos, 4)
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.cat([b[2] for b in actual]).tolist() == [5, 1, 6, 0, 7, 3, 2, 4]
    for left, right in zip(expected, actual):
        assert all(torch.equal(a, b) for a, b in zip(left, right))
    torch.set_rng_state(before)
    replay = collect(videos, 4)
    assert all(torch.equal(a, b) for left, right in zip(actual, replay) for a, b in zip(left, right))


def test_decode_failure_and_early_exit_close_owned_workers(videos):
    from semantic_retrieval.full_data.loader import batch_loader
    original = {p.pid for p in mp.active_children()}
    with batch_loader(videos, [[0], [1], [2], [3]], workers=4, pin_memory=False) as iterator:
        next(iterator)
    assert {p.pid for p in mp.active_children()} == original
    Path(videos.paths[0]).write_bytes(b'broken')
    with pytest.raises(ValueError, match='cannot open video'):
        collect(videos, 4)
    assert {p.pid for p in mp.active_children()} == original
