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


def test_cosine_prototype_distance_returns_nearest_index():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    output = EpistemicUncertaintyModule().from_prototypes(
        features,
        prototypes,
    )

    assert torch.allclose(
        output.uncertainty_score,
        torch.zeros(2),
        atol=1e-6,
    )
    assert output.component_scores["prototype"] is output.uncertainty_score
    assert output.nearest_prototype_distance is output.uncertainty_score
    assert torch.equal(output.nearest_prototype_index, torch.tensor([0, 2]))


def test_squared_euclidean_prototype_distance():
    features = torch.tensor([[2.0, 0.0]])
    prototypes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    output = EpistemicUncertaintyModule().from_prototypes(
        features,
        prototypes,
        metric="squared_euclidean",
    )

    assert output.nearest_prototype_distance.item() == pytest.approx(1.0)
    assert output.nearest_prototype_index.item() == 1


def test_prototype_distance_rejects_empty_or_mismatched_prototypes():
    module = EpistemicUncertaintyModule()
    with pytest.raises(ValueError, match="non-empty rank-2"):
        module.from_prototypes(torch.randn(2, 4), torch.empty(0, 4))
    with pytest.raises(ValueError, match="feature size"):
        module.from_prototypes(torch.randn(2, 4), torch.randn(3, 5))
    with pytest.raises(ValueError, match="dtype"):
        module.from_prototypes(
            torch.randn(2, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float64),
        )


def test_prototype_distance_rejects_nonfinite_values_and_unknown_metric():
    module = EpistemicUncertaintyModule()
    features = torch.randn(2, 4)
    prototypes = torch.randn(3, 4)
    prototypes[0, 0] = torch.inf

    with pytest.raises(ValueError, match="finite floating-point"):
        module.from_prototypes(features, prototypes)
    with pytest.raises(ValueError, match="metric"):
        module.from_prototypes(features, torch.randn(3, 4), metric="l1")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_prototype_distance_rejects_device_mismatch():
    with pytest.raises(ValueError, match="device"):
        EpistemicUncertaintyModule().from_prototypes(
            torch.randn(2, 4, device="cuda"),
            torch.randn(3, 4),
        )
