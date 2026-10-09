import math
import torch
from semantic_retrieval.evidence.matcher import mc_log_probabilities
from semantic_retrieval.evidence.objectives import probabilistic_contrastive

def test_mc_averages_probabilities_before_log():
    loc = torch.zeros(3, requires_grad=True)
    scale = torch.ones(3, requires_grad=True)
    noise = torch.tensor([[0., 0., 0.], [3., -3., 0.]])
    offsets = torch.tensor([0, 3])
    actual = mc_log_probabilities(loc, scale, offsets, noise, temperature=1.)
    expected = torch.logsumexp(torch.log_softmax(loc[None] + scale[None] * noise, -1), 0) - math.log(2)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, torch.log_softmax(loc[None] + scale[None] * noise, -1).mean(0))
    (-actual[0]).backward()
    assert scale.grad.abs().sum() > 0 and torch.isfinite(scale.grad).all()

def test_unknown_never_enters_contrastive_denominator():
    loc = torch.tensor([1., 0., 1000.], requires_grad=True)
    result = probabilistic_contrastive(
        loc, torch.ones(3), torch.tensor([0, 3]),
        torch.tensor([1, 0, -1]), torch.zeros(2, 3), temperature=1.)
    expected = -torch.log_softmax(loc[:2], 0)[0]
    torch.testing.assert_close(result.numerator, expected)
    assert result.denominator.item() == 1

def test_eval_noise_is_candidate_permutation_invariant(config_factory):
    from semantic_retrieval.evidence.sampling import SamplingState
    s = SamplingState(config_factory().sampling, mode="eval", model_sha256="a"*64)
    ids = (("q0", "a"), ("q0", "b"))
    original = s.noise(ids, device=torch.device("cpu"), dtype=torch.float32)
    permuted = s.noise(ids[::-1], device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(original.flip(1), permuted, rtol=0, atol=0)

def test_scale_head_changes_inference_and_receives_gradient(model, inputs, config_factory):
    from semantic_retrieval.evidence.matcher import GaussianPairMatcher
    from semantic_retrieval.evidence.sampling import SamplingState
    torch.manual_seed(19)
    f = model.encode(inputs)
    matcher = GaussianPairMatcher(8, scale_floor=.01)
    sampling = SamplingState(config_factory().sampling, mode='eval', model_sha256='a'*64)
    pairs = torch.tensor([[0,0],[0,1],[0,2]])
    ids = (('q','a'),('q','b'),('q','c'))
    coverage = torch.zeros(3,4)
    before = matcher(f, pairs, torch.tensor([0,3]), pair_ids=ids, coverage=coverage, sampling=sampling)
    (-before.ranking_log_probability[0]).backward()
    assert sum(p.grad.abs().sum() for p in matcher.scale_head.parameters()) > 0
    with torch.no_grad():
        matcher.scale_head[-1].bias.add_(3.)
    after = matcher(f, pairs, torch.tensor([0,3]), pair_ids=ids, coverage=coverage, sampling=sampling)
    assert not torch.allclose(before.ranking_log_probability, after.ranking_log_probability)
    perm = torch.tensor([2,0,1])
    reordered = matcher(f, pairs[perm], torch.tensor([0,3]), pair_ids=tuple(ids[i] for i in perm), coverage=coverage[perm], sampling=sampling)
    torch.testing.assert_close(reordered.ranking_log_probability, after.ranking_log_probability[perm])


def test_empty_probability_segment_is_supported():
    got = mc_log_probabilities(torch.empty(0), torch.empty(0), torch.tensor([0,0]), torch.empty(4,0), temperature=1.)
    assert got.shape == (0,)
