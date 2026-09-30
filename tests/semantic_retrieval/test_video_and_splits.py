import json

import cv2
import numpy as np
import pytest
import torch


def test_sampling_is_uniform_and_repeatable():
    from semantic_retrieval.data.video import sample_frame_indices

    assert sample_frame_indices(10, 3) == [0, 4, 9]
    assert sample_frame_indices(2, 5) == [0, 1]
    with pytest.raises(ValueError):
        sample_frame_indices(0, 3)


def test_decode_short_video_and_missing_frames(tmp_path):
    from semantic_retrieval.data.video import decode_video

    path = tmp_path / "small.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
    assert writer.isOpened()
    for value in [20, 100, 200]:
        writer.write(np.full((16, 16, 3), value, np.uint8))
    writer.release()
    first, mask = decode_video(path, [0, 2, 99], 8)
    second, _ = decode_video(path, [0, 2, 99], 8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert mask.tolist() == [True, True, False]
    assert first[1].mean() > first[0].mean() and (first[2] == 0).all()
    with pytest.raises(FileNotFoundError):
        decode_video(tmp_path / "missing.mp4", [0], 8)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not video")
    with pytest.raises(ValueError):
        decode_video(bad, [0], 8)


def test_zero_fps_is_irrelevant_to_index_decode(tmp_path, monkeypatch):
    from semantic_retrieval.data.video import decode_video

    path = tmp_path / "video"
    path.touch()

    class Capture:
        def isOpened(self):
            return True

        def get(self, field):
            return 0

        def set(self, field, value):
            pass

        def read(self):
            return True, np.full((8, 8, 3), 127, np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    frames, mask = decode_video(path, [0, 1], 8)
    assert mask.all() and torch.isfinite(frames).all()


def test_split_source_hash_drift_is_rejected(tmp_path, monkeypatch):
    from semantic_retrieval.data import splits

    monkeypatch.setattr(splits, "SOURCE_TRAIN_VIDEO_COUNT", 2)
    monkeypatch.setattr(splits, "TEST_VIDEO_COUNT", 1)
    monkeypatch.setattr(splits, "DEFAULT_VAL_SIZE", 1)
    monkeypatch.setattr(splits, "DEFAULT_CAPTIONS_PER_VIDEO", 1)
    train = tmp_path / "train.csv"
    train.write_text("video_id\na\nb\n")
    test = tmp_path / "test.csv"
    test.write_text("video_id\nc\n")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "videos": [{"video_id": i} for i in "abc"],
                "sentences": [{"video_id": i, "caption": i, "sen_id": n} for n, i in enumerate("abc")],
            }
        )
    )
    manifest = splits.build_trusted_manifest(train, annotation, test, val_size=1, expected_captions=1)
    splits.validate_trusted_manifest(manifest, train, annotation, test)
    train.write_text("video_id\nb\na\n")
    with pytest.raises(ValueError, match="mismatch"):
        splits.validate_trusted_manifest(manifest, train, annotation, test)
