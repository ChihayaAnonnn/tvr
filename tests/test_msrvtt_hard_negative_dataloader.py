import numpy as np
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader


def test_get_hard_negative_video_returns_mapped_video_and_valid_flag():
    loader = MSRVTT_TrainDataLoader.__new__(MSRVTT_TrainDataLoader)
    loader.hard_index = [1, -1]
    loader.sentences_dict = {
        0: ("video0", "anchor caption"),
        1: ("video1", "hard caption"),
    }
    calls = []

    def fake_get_rawvideo(video_ids):
        calls.append(list(video_ids))
        return np.array([[[1.0]]], dtype=np.float32), np.array([[1]], dtype=np.int64)

    loader._get_rawvideo = fake_get_rawvideo

    video, mask, valid = loader._get_hard_negative_video("video0", 0)

    assert calls == [["video1"]]
    assert valid == np.int64(1)
    assert video.shape == (1, 1, 1)
    assert mask.shape == (1, 1)


def test_get_hard_negative_video_falls_back_to_anchor_when_mapping_missing():
    loader = MSRVTT_TrainDataLoader.__new__(MSRVTT_TrainDataLoader)
    loader.hard_index = [-1]
    loader.sentences_dict = {0: ("video0", "anchor caption")}
    calls = []

    def fake_get_rawvideo(video_ids):
        calls.append(list(video_ids))
        return np.array([[[2.0]]], dtype=np.float32), np.array([[1]], dtype=np.int64)

    loader._get_rawvideo = fake_get_rawvideo

    _video, _mask, valid = loader._get_hard_negative_video("video0", 0)

    assert calls == [["video0"]]
    assert valid == np.int64(0)
