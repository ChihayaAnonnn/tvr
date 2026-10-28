"""Local-only CLIP weight loading with integrity and exact compatibility checks."""

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import torch


def load_clip_state(path, expected_sha256, model):
    with open(path, "rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError("CLIP weight SHA-256 mismatch")
    scripted = False
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            scripted = any("/code/" in name for name in archive.namelist())
    if Path(path).suffix == ".npz":
        # Synthetic archived inputs keep numerical payloads in external NPZ.
        with np.load(path, allow_pickle=False) as archive:
            state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    else:
        state = (
            torch.jit.load(str(path), map_location="cpu").state_dict()
            if scripted
            else torch.load(path, map_location="cpu", weights_only=True)
        )
    if not isinstance(state, dict):
        raise ValueError("expected a CLIP state_dict")
    state = dict(state)
    metadata = dict(
        input_resolution=model.config.image_size,
        context_length=model.config.context_length,
        vocab_size=model.config.vocab_size,
    )
    for name, expected in metadata.items():
        if name in state:
            value = state.pop(name)
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or value.item() != expected:
                raise ValueError(f"invalid CLIP metadata: {name}")
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("CLIP state_dict key mismatch")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid CLIP parameter: {name}")
    model.load_state_dict(state, strict=True)
    model.float()
