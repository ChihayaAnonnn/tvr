import pytest
import torch

from prob_models.retrieval_risk import (
    RetrievalRiskSidecar,
    score_distribution_features,
)


def test_features_are_candidate_permutation_invariant():
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.5, 0.4, 0.3]])

    actual = score_distribution_features(logits, "t2v")
    expected = score_distribution_features(logits[:, [2, 0, 1]], "t2v")

    torch.testing.assert_close(actual, expected)


def test_direction_selects_query_axis_and_adds_direction_code():
    logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 1.0]])

    t2v = score_distribution_features(logits, "t2v")
    v2t = score_distribution_features(logits, "v2t")

    assert t2v.shape == (2, 8)
    assert v2t.shape == (3, 8)
    torch.testing.assert_close(
        t2v[:, -2:], torch.tensor([[1.0, 0.0]]).expand(2, -1)
    )
    torch.testing.assert_close(
        v2t[:, -2:], torch.tensor([[0.0, 1.0]]).expand(3, -1)
    )


@pytest.mark.parametrize(
    ("logits", "direction", "temperature"),
    [
        (torch.ones(3), "t2v", 1.0),
        (torch.ones(2, 1), "t2v", 1.0),
        (torch.ones(2, 2), "bad", 1.0),
        (torch.ones(2, 2), "t2v", 0.0),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), "t2v", 1.0),
    ],
)
def test_features_reject_invalid_inputs(logits, direction, temperature):
    with pytest.raises(ValueError):
        score_distribution_features(logits, direction, temperature)


def test_sidecar_returns_binary_dirichlet_outputs():
    sidecar = RetrievalRiskSidecar(hidden_dim=16)

    output = sidecar(torch.randn(4, 5), "t2v")

    assert output["alpha"].shape == (4, 2)
    assert output["failure_risk"].shape == (4,)
    assert output["vacuity"].shape == (4,)
    assert output["features"].shape == (4, 8)
    assert torch.all(output["alpha"] > 1)
    torch.testing.assert_close(
        output["alpha"].sum(dim=1) * output["vacuity"],
        torch.full((4,), 2.0),
    )


def test_sidecar_detaches_score_features():
    logits = torch.randn(3, 3, requires_grad=True)
    sidecar = RetrievalRiskSidecar(hidden_dim=8)

    sidecar(logits, "t2v")["failure_risk"].sum().backward()

    assert logits.grad is None
