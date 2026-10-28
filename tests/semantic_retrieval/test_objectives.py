import math

import pytest
import torch

from semantic_retrieval.types import PairScores, Targets


def compute(labels, presence, logits=None):
    from semantic_retrieval.objectives.retrieval import compute_objective

    labels = torch.tensor([labels], dtype=torch.long)
    n = labels.shape[1]
    x = torch.zeros(n, requires_grad=True) if logits is None else logits
    p = PairScores(x, torch.zeros(n, 2), None)
    terms = compute_objective(
        x[None, :],
        p,
        torch.zeros(1, requires_grad=True),
        Targets(labels, torch.tensor([presence])),
        torch.tensor([[0, i] for i in range(n)], dtype=torch.long).reshape(n, 2),
        torch.tensor([0, n]),
        weights=dict(coarse=1, match=1, pair=1, exist=1, align=0, group=0),
    )
    return terms, x


def test_multi_positive_hand_calculation_and_unknown():
    terms, x = compute([1, 1, 0, -1], 1)
    assert terms["match"].item() == pytest.approx(math.log(3))
    assert terms["pair"].item() == pytest.approx(math.log(2))
    terms["total"].backward()
    assert x.grad[0] < 0 and x.grad[1] < 0 and x.grad[2] > 0 and x.grad[3] == 0
    changed, _ = compute([1, 1, 0, -1], 1, torch.tensor([0.0, 0.0, 0.0, 10000.0], requires_grad=True))
    torch.testing.assert_close(changed["total"], terms["total"])


@pytest.mark.parametrize("labels,presence", [([-1, -1], -1), ([], -1)])
def test_unknown_is_connected_zero(labels, presence):
    t, x = compute(labels, presence)
    assert t["total"].item() == 0 and all(t["valid_" + k + "_count"] == 0 for k in ["coarse", "match", "pair", "exist"])
    t["total"].backward()
    assert x.grad is not None


def test_all_negative_only_bce():
    t, _ = compute([0, 0], 0)
    assert t["coarse"] == t["match"] == 0 and t["valid_pair_count"] == 2 and t["valid_exist_count"] == 1


def test_unordered_packed_query_rejected():
    from semantic_retrieval.objectives.retrieval import compute_objective

    with pytest.raises(ValueError):
        compute_objective(
            torch.zeros(2, 1),
            PairScores(torch.zeros(2), torch.zeros(2, 1), None),
            torch.zeros(2),
            Targets(torch.ones(2, 1, dtype=torch.long), torch.ones(2, dtype=torch.long)),
            torch.tensor([[1, 0], [0, 0]]),
            torch.tensor([0, 1, 2]),
            weights={},
        )
