import hashlib
import cv2
import numpy as np
import pytest
from semantic_retrieval.laya_rerank.evidence import decode_evidence


def test_real_short_video(tmp_path):
    path = tmp_path/'short.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 2, (32, 24))
    assert writer.isOpened()
    for i in range(6): writer.write(np.full((24, 32, 3), i*30, dtype=np.uint8))
    writer.release()
    row = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), split='val', frame_count=6, fps=2.)
    images, indices = decode_evidence(row)
    assert indices == [0, 2, 4] and len(images) == 3 and images[0].size == (32, 24)
    row['split'] = 'test'
    with pytest.raises(ValueError): decode_evidence(row)
