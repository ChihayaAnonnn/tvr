import math
import pytest
import torch
from semantic_retrieval.evidence.matcher import mc_log_probabilities
from semantic_retrieval.evidence.objectives import gaussian_score_nll, probabilistic_contrastive


def test_small_scale_matches_point_softmax():
    loc = torch.tensor([2., -1., .5, 4.])
    got = mc_log_probabilities(loc, torch.full((4,), 1e-9), torch.tensor([0,3,4]), torch.randn(8,4), temperature=1.)
    torch.testing.assert_close(got[:3], loc[:3].log_softmax(0))
    assert got[-1].item() == 0.


def test_multiple_positives_extreme_logits_and_unknown_gradients():
    loc = torch.tensor([10000., -10000., 100000., 0.], requires_grad=True)
    scale = torch.ones(4, requires_grad=True)
    term = probabilistic_contrastive(loc, scale, torch.tensor([0,4]), torch.tensor([1,1,-1,0]), torch.zeros(2,4), temperature=1.)
    assert term.numerator.item() == 10000.
    term.numerator.backward()
    assert loc.grad[2] == scale.grad[2] == 0
    assert torch.isfinite(loc.grad).all()


def test_all_unknown_and_empty_have_connected_zero():
    for size in (0, 3):
        loc = torch.ones(size, requires_grad=True); scale = torch.ones(size, requires_grad=True)
        result = probabilistic_contrastive(loc, scale, torch.tensor([0,size]), torch.full((size,), -1), torch.zeros(2,size), temperature=1.)
        assert result.denominator == result.numerator == 0
        result.numerator.backward()
        assert loc.grad is not None and scale.grad is not None


def test_reference_target_detached_and_gaussian_nll():
    loc = torch.tensor([1., 9.], requires_grad=True); scale = torch.tensor([2., 1.], requires_grad=True)
    target = torch.tensor([3., 0.], requires_grad=True)
    result = gaussian_score_nll(loc, scale, target, torch.tensor([True,False]))
    assert result.numerator.item() == pytest.approx(math.log(2)+.5)
    result.numerator.backward()
    assert target.grad is None and loc.grad.tolist() == [-.5, 0.]
