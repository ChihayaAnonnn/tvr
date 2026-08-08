import pytest
import torch

import uncertainty_modules
from uncertainty_modules.epistemic import EpistemicUncertaintyModule
from uncertainty_modules.types import EpistemicOutput


def test_package_exports_epistemic_public_api():
    assert (
        uncertainty_modules.EpistemicUncertaintyModule
        is EpistemicUncertaintyModule
    )
    assert uncertainty_modules.EpistemicOutput is EpistemicOutput


def test_identical_ensemble_samples_have_zero_disagreement():
    base = torch.randn(2, 4)
    samples = base.unsqueeze(0).repeat(3, 1, 1)

    output = EpistemicUncertaintyModule().from_ensemble(samples)

    assert torch.equal(output.diagonal_variance, torch.zeros_like(base))
    assert torch.equal(output.uncertainty_score, torch.zeros(2))
    assert output.component_scores["ensemble"] is output.uncertainty_score


def test_epistemic_component_scores_are_read_only():
    samples = torch.randn(3, 2, 4)
    output = EpistemicUncertaintyModule().from_ensemble(samples)

    with pytest.raises(TypeError):
        output.component_scores["ensemble"] = torch.zeros(2)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_squared_euclidean_supports_low_precision_cpu_and_autograd(dtype):
    features = torch.tensor(
        [[2.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    prototypes = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)

    output = EpistemicUncertaintyModule().from_prototypes(
        features,
        prototypes,
        metric="squared_euclidean",
    )
    output.uncertainty_score.sum().backward()

    assert output.uncertainty_score.dtype == dtype
    assert torch.isfinite(output.uncertainty_score).all()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_squared_euclidean_supports_cuda_float16_and_autograd():
    features = torch.tensor(
        [[2.0, 0.0]],
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    prototypes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0]],
        device="cuda",
        dtype=torch.float16,
    )

    output = EpistemicUncertaintyModule().from_prototypes(
        features,
        prototypes,
        metric="squared_euclidean",
    )
    output.uncertainty_score.sum().backward()

    assert output.uncertainty_score.device == features.device
    assert output.uncertainty_score.dtype == features.dtype
    assert torch.isfinite(output.uncertainty_score).all()
    assert torch.isfinite(features.grad).all()


def test_zero_vector_cosine_distance_is_finite():
    output = EpistemicUncertaintyModule().from_prototypes(
        torch.zeros(1, 4),
        torch.zeros(2, 4),
    )

    assert output.uncertainty_score.item() == pytest.approx(1.0)
    assert torch.isfinite(output.uncertainty_score).all()


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


def test_combine_calibrates_weights_and_applies_ood_threshold():
    scores = {
        "ensemble": torch.tensor([1.0, 3.0]),
        "prototype": torch.tensor([2.0, 6.0]),
    }
    calibration = {
        "ensemble": (1.0, 2.0),
        "prototype": (2.0, 4.0),
    }

    output = EpistemicUncertaintyModule().combine(
        scores,
        {"ensemble": 1.0, "prototype": 3.0},
        calibration,
        threshold=0.5,
    )

    assert torch.allclose(output.uncertainty_score, torch.tensor([0.0, 1.0]))
    assert output.component_scores == scores
    assert torch.equal(output.is_ood, torch.tensor([False, True]))


def test_combine_without_threshold_omits_ood_decision():
    output = EpistemicUncertaintyModule().combine(
        {"ensemble": torch.tensor([1.0, 2.0])},
        {"ensemble": 1.0},
        {"ensemble": (0.0, 1.0)},
    )

    assert output.is_ood is None


@pytest.mark.parametrize(
    ("weights", "calibration", "message"),
    [
        ({"ensemble": -1.0}, {"ensemble": (0.0, 1.0)}, "nonnegative"),
        ({"ensemble": float("nan")}, {"ensemble": (0.0, 1.0)}, "finite"),
        ({"ensemble": 0.0}, {"ensemble": (0.0, 1.0)}, "positive"),
        ({"ensemble": 1.0}, {}, "identical keys"),
        ({"ensemble": 1.0}, {"ensemble": (0.0, 0.0)}, "positive"),
        ({"ensemble": 1.0}, {"ensemble": (float("nan"), 1.0)}, "finite"),
    ],
)
def test_combine_rejects_invalid_configuration(
    weights,
    calibration,
    message,
):
    with pytest.raises(ValueError, match=message):
        EpistemicUncertaintyModule().combine(
            {"ensemble": torch.ones(2)},
            weights,
            calibration,
        )


def test_combine_rejects_empty_or_incompatible_scores():
    module = EpistemicUncertaintyModule()
    with pytest.raises(ValueError, match="must not be empty"):
        module.combine({}, {}, {})
    with pytest.raises(ValueError, match=r"floating-point \[B\]"):
        module.combine(
            {"ensemble": torch.ones(2, 1)},
            {"ensemble": 1.0},
            {"ensemble": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="share shape"):
        module.combine(
            {"ensemble": torch.ones(2), "prototype": torch.ones(3)},
            {"ensemble": 1.0, "prototype": 1.0},
            {"ensemble": (0.0, 1.0), "prototype": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="finite"):
        module.combine(
            {"ensemble": torch.tensor([1.0, float("inf")])},
            {"ensemble": 1.0},
            {"ensemble": (0.0, 1.0)},
        )


@pytest.mark.parametrize(
    "calibration_statistic",
    [torch.zeros(2, 1), torch.zeros(1, 2)],
)
def test_combine_rejects_calibration_that_changes_score_shape(
    calibration_statistic,
):
    with pytest.raises(ValueError, match="single value"):
        EpistemicUncertaintyModule().combine(
            {"ensemble": torch.ones(2)},
            {"ensemble": 1.0},
            {"ensemble": (calibration_statistic, 1.0)},
        )


def test_combine_scalarizes_single_value_calibration_tensors():
    output = EpistemicUncertaintyModule().combine(
        {"ensemble": torch.tensor([1.0, 2.0])},
        {"ensemble": 1.0},
        {"ensemble": (torch.zeros(1, 1), torch.ones(1, 1))},
    )

    assert output.uncertainty_score.shape == (2,)
    assert torch.equal(output.uncertainty_score, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_combine_rejects_nonfinite_ood_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be finite"):
        EpistemicUncertaintyModule().combine(
            {"ensemble": torch.ones(2)},
            {"ensemble": 1.0},
            {"ensemble": (0.0, 1.0)},
            threshold=threshold,
        )
