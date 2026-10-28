import pytest
import torch
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig


@pytest.fixture
def tiny_clip():
    torch.manual_seed(713)
    return ClipBackbone(ClipConfig(64, 16, 64, 1, 4, 32, 128, 64, 1, 1, fixture=True))


@pytest.fixture
def paired_batch():
    ids = torch.tensor([[126, 4, 127, 0, 0, 0, 0, 0], [126, 5, 6, 127, 0, 0, 0, 0]])
    generator = torch.Generator().manual_seed(714)
    return (ids, ids != 0, torch.randn(2, 3, 3, 16, 16, generator=generator),
            torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool), torch.arange(2))
