from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiment_artifacts import ArtifactArrayStore
from semantic_retrieval.types import VideoFeatures


def test_canonical_chunk_ties_and_empty():
    from semantic_retrieval.retrieval.index import CanonicalIndex, IndexIdentity

    identity = IndexIdentity("a" * 64, "g", "b" * 64)
    index = CanonicalIndex(["b", "a", "b", "c"], torch.tensor([[1.0, 0], [1.0, 0], [1.0, 0], [0.0, 1]]), identity)
    for chunk in (1, 2, 10):
        index.chunk_size = chunk
        assert index.search(torch.tensor([[1.0, 0]]), 20, expected_identity=identity) == [["a", "b", "c"]]
    assert index.search(torch.tensor([[1.0, 0]]), 2, expected_identity=identity) == [["a", "b"]]
    empty = CanonicalIndex([], torch.empty(0, 2), identity)
    assert empty.search(torch.ones(1, 2), 1, expected_identity=identity) == [[]]
    with pytest.raises(ValueError):
        index.search(torch.ones(1, 2), 0, expected_identity=identity)
    with pytest.raises(ValueError):
        CanonicalIndex(["a", "a"], torch.eye(2), identity)
    for key, val in [("model_sha256", "c" * 64), ("gallery_version", "other"), ("preprocessing_sha256", "c" * 64)]:
        with pytest.raises(ValueError):
            index.search(torch.ones(1, 2), 1, expected_identity=replace(identity, **{key: val}))


def test_cache_binding_and_corruption(tmp_path):
    from semantic_retrieval.retrieval.cache import FeatureCache
    from semantic_retrieval.retrieval.index import IndexIdentity

    identity = IndexIdentity("a" * 64, "g", "b" * 64)
    context = SimpleNamespace(array_store=ArtifactArrayStore(tmp_path / "arrays"))
    cache = FeatureCache(context)
    features = VideoFeatures(torch.ones(1, 2, 8), torch.ones(1, 8), torch.ones(1, 2, dtype=torch.bool))
    cache.write("a", features, identity=identity, content_sha256="c" * 64)
    loaded = cache.read("a", expected_identity=identity, content_sha256="c" * 64)
    torch.testing.assert_close(loaded.frame_tokens, features.frame_tokens)
    for ident, content in [(replace(identity, model_sha256="d" * 64), "c" * 64), (identity, "d" * 64)]:
        with pytest.raises(ValueError):
            cache.read("a", expected_identity=ident, content_sha256=content)
    array = next(tmp_path.rglob("*.npy"))
    array.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        cache.read("a", expected_identity=identity, content_sha256="c" * 64)
