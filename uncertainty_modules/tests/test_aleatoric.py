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
