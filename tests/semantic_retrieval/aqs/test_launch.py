import pytest


def test_rank_batches_preserve_complete_last_batch():
    from semantic_retrieval.aqs.launch import rank_batches
    global_batches = [tuple(range(64)), tuple(range(64, 88))]
    ranks = [rank_batches(global_batches, r, 2) for r in range(2)]
    assert [len(b) for b in ranks[0]] == [32, 12]
    for i, batch in enumerate(global_batches):
        assert ranks[0][i] + ranks[1][i] == list(batch)
    with pytest.raises(ValueError):
        rank_batches([(0, 1, 2)], 0, 2)


def test_launcher_rejects_test_and_wrong_topology():
    from semantic_retrieval.aqs.launch import validate_recipe
    recipe = dict(epochs=5, global_batch=64, learning_rate=5e-5, seed=713,
                  world_size=2, negatives=16, weight=.2, validation_only=True)
    validate_recipe(recipe)
    for key, value in [('world_size', 4), ('validation_only', False), ('global_batch', 63)]:
        with pytest.raises(ValueError):
            validate_recipe(dict(recipe, **{key: value}))


def test_relocated_index_requires_exact_original_bytes(tmp_path):
    from scripts.prepare_aqs_training import relocate_indices
    import hashlib
    path = tmp_path/'experiments/demo/inputs/raw.input-index.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original')
    sha = hashlib.sha256(b'original').hexdigest()
    old = dict(path='/removed/UATVR/experiments/demo/inputs/raw.input-index.json', sha256=sha)
    result, lineage = relocate_indices(tmp_path, [old])
    assert result == [dict(path=str(path), sha256=sha)]
    assert lineage[0]['original_path'] == old['path']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): relocate_indices(tmp_path, [old])
