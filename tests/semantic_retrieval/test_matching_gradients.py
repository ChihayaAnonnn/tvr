import pytest
import torch


@pytest.mark.parametrize("objective", ["pair", "presence"])
def test_shared_projection_updates_inference(model, inputs, objective):
    model.clip.requires_grad_(False)
    pairs = torch.tensor([[0, 0], [0, 1], [1, 2]])
    offsets = torch.tensor([0, 2, 3])
    features = model.encode(inputs)
    before = model.score_pairs(features, pairs).logits.detach().clone()
    p = model.score_pairs(features, pairs)
    loss = p.logits.square().sum() if objective == "pair" else model.score_presence(p, offsets).sum()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    loss.backward()
    assert model.matcher.text_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.clip.parameters())
    opt.step()
    after = model.score_pairs(model.encode(inputs), pairs).logits.detach()
    assert not torch.equal(before, after)


def test_permutation_empty_and_alignment(model, inputs):
    f = model.encode(inputs)
    pairs = torch.tensor([[0, 0], [0, 1]])
    p = model.score_pairs(f, pairs, return_alignment=True)
    assert p.attention.shape == (2, 5, 5)
    shuffled = model.score_pairs(f, pairs.flip(0))
    torch.testing.assert_close(p.logits, shuffled.logits.flip(0))
    torch.testing.assert_close(
        model.score_presence(p, torch.tensor([0, 2])), model.score_presence(shuffled, torch.tensor([0, 2]))
    )
    empty = model.score_pairs(f, torch.empty(0, 2, dtype=torch.long))
    assert model.score_presence(empty, torch.tensor([0, 0])).item() < -10
    for bad in [torch.tensor([[0, 3]]), torch.tensor([[-1, 0]])]:
        with pytest.raises(ValueError):
            model.score_pairs(f, bad)


def test_unfreezing_designated_clip_layer_has_gradient(model, inputs):
    model.clip.requires_grad_(False)
    model.clip.text_projection.requires_grad_(True)
    model.score_pairs(model.encode(inputs), torch.tensor([[0, 0]])).logits.sum().backward()
    assert model.clip.text_projection.grad.abs().sum() > 0
