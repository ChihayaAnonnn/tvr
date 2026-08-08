import pytest
import torch

from uncertainty_modules.aleatoric import AleatoricUncertaintyModule
from uncertainty_modules.types import AleatoricOutput, ProbabilisticEmbedding


def test_forward_returns_documented_shapes():
    module = AleatoricUncertaintyModule(
        4,
        8,
        min_variance=1e-4,
        max_variance=2.0,
    )

    output = module(torch.randn(2, 3, 4))

    assert isinstance(output, AleatoricOutput)
    assert isinstance(output.embedding, ProbabilisticEmbedding)
    assert output.embedding.mean.shape == (2, 4)
    assert output.embedding.variance.shape == (2, 4)
    assert output.embedding.uncertainty_score.shape == (2,)
    assert output.local_uncertainty.shape == (2, 3)
    assert output.global_uncertainty.shape == (2,)
    assert output.relevance_weights.shape == (2, 3)
    assert output.aggregation_weights.shape == (2, 3)


@pytest.mark.parametrize(
    "args",
    [
        (0, 8, 1e-4, 2.0),
        (4, 0, 1e-4, 2.0),
        (4, 8, 0.0, 2.0),
        (4, 8, 2.0, 2.0),
    ],
)
def test_constructor_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        AleatoricUncertaintyModule(*args)


def test_masked_positions_are_zero_and_valid_weights_sum_to_one():
    module = AleatoricUncertaintyModule(4, 8)
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    output = module(torch.randn(2, 3, 4), mask)

    assert torch.equal(
        output.relevance_weights[~mask],
        torch.zeros(3),
    )
    assert torch.equal(
        output.aggregation_weights[~mask],
        torch.zeros(3),
    )
    assert torch.allclose(
        output.relevance_weights.sum(dim=1),
        torch.ones(2),
    )
    assert torch.allclose(
        output.aggregation_weights.sum(dim=1),
        torch.ones(2),
    )


def test_variance_is_finite_and_bounded():
    module = AleatoricUncertaintyModule(
        4,
        8,
        min_variance=0.1,
        max_variance=0.9,
    )

    output = module(torch.randn(2, 3, 4) * 1e4)

    assert torch.isfinite(output.embedding.variance).all()
    assert torch.all(output.embedding.variance >= 0.1)
    assert torch.all(output.embedding.variance <= 0.9)


def test_reliability_does_not_send_mean_loss_gradient_to_uncertainty_head():
    module = AleatoricUncertaintyModule(4, 8)

    module(torch.randn(2, 3, 4)).embedding.mean.sum().backward()

    assert any(
        parameter.grad is not None
        for parameter in module.relevance_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in module.uncertainty_head.parameters()
    )


def test_rejects_all_empty_mask():
    module = AleatoricUncertaintyModule(4, 8)
    mask = torch.tensor(
        [
            [True, False, False],
            [False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid"):
        module(torch.randn(2, 3, 4), mask)


def test_preserves_float64_dtype():
    module = AleatoricUncertaintyModule(4, 8).double()

    output = module(torch.randn(2, 1, 4, dtype=torch.float64))

    assert output.embedding.mean.dtype == torch.float64
    assert output.embedding.variance.dtype == torch.float64


def test_rejects_nonfinite_features():
    module = AleatoricUncertaintyModule(4, 8)
    features = torch.randn(2, 3, 4)
    features[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        module(features)


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4)])
def test_rejects_empty_batch_or_sequence(shape):
    module = AleatoricUncertaintyModule(4, 8)

    with pytest.raises(ValueError, match="non-empty"):
        module(torch.empty(shape))
