"""Certified RGB evidence, before CLIP resize and normalization."""
import hashlib
from pathlib import Path
from .core import evidence_indices


def decode_evidence(record):
    import cv2
    from PIL import Image
    from semantic_retrieval.full_data.loader import official_frame_indices
    path = Path(record['path'])
    if record['split'] != 'val': raise ValueError('reranker accepts validation videos only')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    if h.hexdigest() != record['sha256']: raise ValueError('certified video changed')
    indices = evidence_indices(official_frame_indices(record['frame_count'], record['fps'], 12))
    capture = cv2.VideoCapture(str(path)); images = []
    try:
        if not capture.isOpened(): raise ValueError('cannot open certified video')
        for index in indices:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index): raise ValueError('cannot seek certified frame')
            ok, frame = capture.read()
            if not ok: raise ValueError('cannot decode certified frame')
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return images, indices
