from dataclasses import replace
import pytest
import torch
from experiment_artifacts import ArtifactArrayStore
from semantic_retrieval.evidence.frame_cache import FrameFeatureCache, FrameCacheIdentity


def test_cache_rejects_wrong_identity_and_tamper(tmp_path):
    cache = FrameFeatureCache(ArtifactArrayStore(tmp_path/'arrays'))
    identity = FrameCacheIdentity(*['a'*64, 'b'*64, 'c'*64, 'd'*64])
    ref = cache.write('a', 0, torch.arange(8.), identity=identity)
    assert torch.equal(cache.read('a', 0, identity=identity), torch.arange(8.))
    with pytest.raises(ValueError):
        cache.read('a', 0, identity=replace(identity, clip_sha256='f'*64))
    with pytest.raises(ValueError):
        cache.write('a', 0, torch.ones(8), identity=identity)
    p = tmp_path/ref.path; p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        cache.read('a', 0, identity=identity)


def test_cache_rejects_gradients_and_symlinks(tmp_path):
    cache = FrameFeatureCache(ArtifactArrayStore(tmp_path/'arrays'))
    identity = FrameCacheIdentity(*(['a'*64]*4))
    with pytest.raises(ValueError):
        cache.write('a', 0, torch.ones(8, requires_grad=True), identity=identity)

def test_cache_symlink_index_rejected(tmp_path):
    cache = FrameFeatureCache(ArtifactArrayStore(tmp_path/'arrays'))
    identity = FrameCacheIdentity(*(['a'*64]*4))
    cache.write('a', 0, torch.ones(8), identity=identity)
    p = next(cache.root.glob('*.json'))
    original = p.read_bytes(); p.unlink()
    target = tmp_path/'outside.json'; target.write_bytes(original); p.symlink_to(target)
    with pytest.raises(ValueError):
        cache.read('a', 0, identity=identity)
