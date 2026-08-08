import pytest
import torch

from uncertainty_modules.epistemic import EpistemicUncertaintyModule


def test_identical_ensemble_samples_have_zero_disagreement():
    base = torch.randn(2, 4)
    samples = base.unsqueeze(0).repeat(3, 1, 1)

    output = EpistemicUncertaintyModule().from_ensemble(samples)

    assert torch.equal(output.diagonal_variance, torch.zeros_like(base))
    assert torch.equal(output.uncertainty_score, torch.zeros(2))
    assert output.component_scores["ensemble"] is output.uncertainty_score


def test_mc_dropout_disagreement_increases_with_spread():
    module = EpistemicUncertaintyModule()
    base = torch.zeros(3, 2, 4)
    spread = base.clone()
    spread[0] = -1
    spread[2] = 1

    base_score = module.from_mc_dropout(base).uncertainty_score
    spread_score = module.from_mc_dropout(spread).uncertainty_score

    assert torch.all(spread_score > base_score)


@pytest.mark.parametrize("shape", [(1, 2, 4), (2, 4)])
def test_disagreement_rejects_invalid_samples(shape):
    with pytest.raises(ValueError, match="M >= 2"):
        EpistemicUncertaintyModule().from_ensemble(torch.randn(*shape))


@pytest.mark.parametrize(
    "samples",
    [
        torch.ones(2, 2, 4, dtype=torch.int64),
        torch.full((2, 2, 4), float("nan")),
    ],
)
def test_disagreement_rejects_nonfloating_or_nonfinite_samples(samples):
    with pytest.raises(ValueError, match="finite floating-point"):
        EpistemicUncertaintyModule().from_mc_dropout(samples)
