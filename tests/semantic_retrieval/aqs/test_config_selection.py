import pytest
import torch


def test_config_roundtrip_and_no_scientific_defaults():
    from semantic_retrieval.aqs.config import AqsConfig
    config = AqsConfig(True, 2, .2)
    assert config.active
    assert AqsConfig.from_recipe(config.recipe()) == config
    assert not AqsConfig(True, 2, 0).active
    assert not AqsConfig(False, 2, .2).active
    with pytest.raises(ValueError):
        AqsConfig.from_recipe(dict(config.recipe(), unexpected=True))


@pytest.mark.parametrize('args', [(1, 2, .2), (True, True, .2), (True, 0, .2),
                                  (True, 2, True), (True, 2, -.1), (True, 2, float('nan')),
                                  (True, 2, float('inf'))])
def test_config_rejects_invalid(args):
    from semantic_retrieval.aqs.config import AqsConfig
    with pytest.raises(ValueError):
        AqsConfig(*args)


def test_pool_ignores_padding_and_aggregate_tokens_without_mutation():
    from semantic_retrieval.aqs.selection import pooled_frames
    x = torch.tensor([[[3., 0.], [float('nan'), 4.], [90., 90.]]], requires_grad=True)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    before, before_mask = x.detach().clone(), mask.clone()
    actual = pooled_frames(x, mask, frame_count=2)
    torch.testing.assert_close(actual, torch.tensor([[1., 0.]]), rtol=0, atol=0)
    assert not actual.requires_grad
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(mask, before_mask)


@pytest.mark.parametrize('defect', ['padding', 'zero', 'nan', 'mask', 'count'])
def test_pool_rejects_invalid(defect):
    from semantic_retrieval.aqs.selection import pooled_frames
    x, mask, count = torch.ones(1, 2, 3), torch.ones(1, 2), 2
    if defect == 'padding': mask.zero_()
    if defect == 'zero': x.zero_()
    if defect == 'nan': x[0, 0, 0] = float('nan')
    if defect == 'mask': mask[0, 0] = .5
    if defect == 'count': count = True
    with pytest.raises(ValueError):
        pooled_frames(x, mask, count)


@pytest.mark.parametrize('ids', [torch.tensor([1, 1]), torch.tensor([-1, 2]), torch.tensor([1., 2.])])
def test_ids_reject_duplicate_negative_or_wrong_dtype(ids):
    from semantic_retrieval.aqs.selection import validate_ids
    with pytest.raises(ValueError):
        validate_ids(ids, 2)


def test_hard_first_then_diverse_and_permutation():
    from semantic_retrieval.aqs.selection import select_negatives
    ids = torch.tensor([10, 20, 30, 40, 50])
    scores = torch.zeros(5, 5, requires_grad=True)
    with torch.no_grad(): scores[0] = torch.tensor([0., .9, .8, .7, .1])
    reps = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [-1., 0.], [0., -1.]], requires_grad=True)
    rng = torch.get_rng_state().clone()
    out = select_negatives(scores, reps, ids, 2)
    assert out.selected_ids[0].tolist() == [20, 40]
    assert not out.difficulty.requires_grad and not out.uniqueness.requires_grad
    assert torch.equal(torch.get_rng_state(), rng)
    perm = torch.tensor([4, 2, 0, 3, 1])
    other = select_negatives(scores[perm][:, perm], reps[perm], ids[perm], 2)
    assert torch.equal(other.selected_ids, out.selected_ids[perm])


@pytest.mark.parametrize('batch,maximum', [(1, 2), (3, 20), (4, 2)])
def test_ties_capacity_and_exclusion(batch, maximum):
    from semantic_retrieval.aqs.selection import select_negatives, summarize_selection
    ids = torch.arange(batch).flip(0)
    out = select_negatives(torch.zeros(batch, batch), torch.ones(batch, 3), ids, maximum)
    assert out.indices.shape == (batch, min(maximum, batch-1))
    for i in range(batch):
        expected = sorted(set(ids.tolist())-{int(ids[i])})[:maximum]
        assert out.selected_ids[i].tolist() == expected
    stats = summarize_selection(out)
    assert stats['mean_negatives'] == min(maximum, batch-1)
    assert stats['empty_rows'] == (1 if batch == 1 else 0)


@pytest.mark.parametrize('defect', ['nan_score', 'nan_repr', 'zero_repr', 'shape', 'ids', 'maximum'])
def test_selector_rejects_corrupt_input(defect):
    from semantic_retrieval.aqs.selection import select_negatives
    scores, reps, ids, maximum = torch.eye(3), torch.eye(3), torch.arange(3), 2
    if defect == 'nan_score': scores[0, 0] = float('nan')
    if defect == 'nan_repr': reps[0, 0] = float('nan')
    if defect == 'zero_repr': reps[0].zero_()
    if defect == 'shape': scores = scores[:, :2]
    if defect == 'ids': ids[0] = ids[1]
    if defect == 'maximum': maximum = True
    with pytest.raises(ValueError): select_negatives(scores, reps, ids, maximum)
