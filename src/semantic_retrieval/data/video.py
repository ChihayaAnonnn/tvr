"""Index-based OpenCV decoding with the original CLIP resize/crop/normalization."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sample_frame_indices(frame_count, max_frames):
    if type(frame_count) is not int or type(max_frames) is not int or min(frame_count, max_frames) <= 0:
        raise ValueError("frame_count and max_frames must be positive integers")
    return np.linspace(0, frame_count - 1, min(frame_count, max_frames)).astype(int).tolist()


def decode_video(path, frame_indices, image_size, *, decoder_threads=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = list(frame_indices)
    if (
        not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or type(image_size) is not int
        or image_size <= 0
    ):
        raise ValueError("indices must be nonempty nonnegative integers; image_size must be positive")
    if decoder_threads is not None and (type(decoder_threads) is not int or not 1 <= decoder_threads <= 16):
        raise ValueError('decoder_threads must be an integer in [1,16]')
    capture = (cv2.VideoCapture(str(path)) if decoder_threads is None else
               cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads]))
    transform = Compose([Resize(image_size, interpolation=Image.BICUBIC), CenterCrop(image_size), ToTensor()])
    frames = torch.zeros((len(indices), 3, image_size, image_size))
    mask = torch.zeros(len(indices), dtype=torch.bool)
    mean = torch.tensor(CLIP_MEAN)[:, None, None]
    std = torch.tensor(CLIP_STD)[:, None, None]
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        for position, index in enumerate(indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                continue
            ok, frame = capture.read()
            if ok:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames[position] = (transform(image) - mean) / std
                mask[position] = True
        if not mask.any():
            raise ValueError(f"video has no decodable requested frames: {path}")
    finally:
        capture.release()
    return frames, mask
