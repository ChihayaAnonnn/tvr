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
