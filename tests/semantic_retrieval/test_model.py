import hashlib
from dataclasses import replace

import pytest
import torch

from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.types import EncoderInputs


@pytest.fixture
def model_config(tmp_path):
    from semantic_retrieval.config import ModelConfig

    torch.manual_seed(7)
    clip = ClipConfig(8, 8, 64, 1, 4, 5, 20, 8, 1, 1, fixture=True)
    path = tmp_path / "clip.pt"
    torch.save(ClipBackbone(clip).state_dict(), path)
    return ModelConfig(clip, path, hashlib.sha256(path.read_bytes()).hexdigest(), 8, 1, 2, 4, 6)


@pytest.fixture
def model(model_config):
    from semantic_retrieval.models.model import SemanticRetrievalModel

    return SemanticRetrievalModel(model_config).eval()


@pytest.fixture
def inputs():
    return EncoderInputs(
        torch.tensor([[18, 2, 19, 0, 0], [18, 3, 4, 19, 0]]),
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool),
        torch.randn(3, 4, 3, 8, 8),
        torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
    )


def test_independent_counts_and_unit_coarse(model, inputs):
    f = model.encode(inputs)
    assert f.pooled_text.shape == (2, 8) and f.pooled_video.shape == (3, 8)
    q, v = model.coarse_embeddings(f)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(v.norm(dim=-1), torch.ones(3))
    assert type(model.train().encode(inputs)) is type(f)


def test_padding_invariance_and_empty_gallery(model, inputs):
    before = model.encode(inputs)
    changed = inputs.video.clone()
    changed[~inputs.frame_mask] = 100
    after = model.encode(replace(inputs, video=changed))
    torch.testing.assert_close(before.frame_tokens, after.frame_tokens)
    empty = model.encode(replace(inputs, video=inputs.video[:0], frame_mask=inputs.frame_mask[:0]))
    assert empty.pooled_video.shape == (0, 8)
    assert model.encode_query(inputs).pooled_text.shape == (2, 8)


def test_temporal_guards_and_disabled():
    from semantic_retrieval.models.temporal import TemporalEncoder

    t = TemporalEncoder(8, 0, 2, 4)
    f = torch.randn(2, 4, 8)
    m = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    out, p = t(f, m)
    torch.testing.assert_close(p, torch.stack([f[0, :2].mean(0), f[1, 0]]))
    assert not out[~m].any()
    for x, mask in [(f, torch.zeros_like(m)), (torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool))]:
        with pytest.raises(ValueError):
            t(x, mask)


def test_zero_norm_and_weight_binding(model_config):
    from semantic_retrieval.models.coarse import CoarseProjector
    from semantic_retrieval.models.model import SemanticRetrievalModel

    with pytest.raises(ValueError):
        CoarseProjector(8, 6).project_text(torch.zeros(1, 8))
    with pytest.raises(ValueError):
        SemanticRetrievalModel(replace(model_config, clip_sha256="0" * 64))
