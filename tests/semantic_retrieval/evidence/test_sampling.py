import torch
import pytest
from semantic_retrieval.evidence.sampling import SamplingState


def test_training_generator_restores_next_draw(config_factory):
    config = config_factory().sampling
    s = SamplingState(config, mode='train', model_sha256='a'*64)
    ids = (('q', 'a'), ('q', 'b'))
    first = s.noise(ids, device='cpu', dtype=torch.float32)
    saved = s.state_dict()
    expected = s.noise(ids, device='cpu', dtype=torch.float32)
    s.load_state_dict(saved)
    assert torch.equal(expected, s.noise(ids, device='cpu', dtype=torch.float32))
    assert not torch.equal(first, expected)
    bad = SamplingState(config, mode='train', model_sha256='b'*64)
    with pytest.raises(ValueError):
        bad.load_state_dict(saved)
