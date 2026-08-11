import pytest
import torch

from prob_models.retrieval_risk import (
    RetrievalRiskSidecar,
    positive_mask_from_group_ids,
    retrieval_hardness,
    retrieval_risk_objective,
)


def test_group_ids_create_multi_positive_mask():
    groups = torch.tensor([7, 7, 9])

    mask = positive_mask_from_group_ids(groups)

    assert mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_group_ids_support_rectangular_query_and_candidate_sets():
    query = torch.tensor([2, 5])
    candidate = torch.tensor([5, 2, 7])

    mask = positive_mask_from_group_ids(query, candidate)

    assert mask.tolist() == [[False, True, False], [True, False, False]]


def test_hardness_increases_when_negatives_overtake_all_positives():
    positive_mask = torch.tensor(
        [[True, True, False], [False, False, True]]
    )
    easy = torch.tensor([[5.0, 4.0, 0.0], [0.0, 1.0, 5.0]])
    hard = torch.tensor(
        [[1.0, 0.0, 2.0], [3.0, 2.0, 1.0]], requires_grad=True
    )

    easy_hardness = retrieval_hardness(easy, positive_mask)
    hard_hardness = retrieval_hardness(hard, positive_mask)

    assert torch.all(hard_hardness > easy_hardness)
    assert not hard_hardness.requires_grad


@pytest.mark.parametrize(
    ("logits", "positive_mask"),
    [
        (torch.ones(2, 2), torch.ones(2, 3, dtype=torch.bool)),
        (torch.ones(2, 2), torch.tensor([[False, False], [True, False]])),
        (torch.ones(2, 2), torch.tensor([[True, True], [False, True]])),
    ],
)
def test_hardness_rejects_invalid_positive_masks(logits, positive_mask):
    with pytest.raises(ValueError):
        retrieval_hardness(logits, positive_mask)


def test_risk_objective_updates_only_sidecar():
    logits = torch.randn(4, 4, requires_grad=True)
    sidecar = RetrievalRiskSidecar(hidden_dim=8)

    loss, output = retrieval_risk_objective(
        sidecar, logits, torch.eye(4, dtype=torch.bool)
    )
    loss.backward()

    assert logits.grad is None
    assert sum(
        parameter.grad.abs().sum() for parameter in sidecar.parameters()
    ) > 0
    assert {"edl", "fit", "rank", "t2v", "v2t"} <= output.keys()
    assert output["t2v"]["failure"].shape == (4,)
    assert output["v2t"]["failure"].shape == (4,)


def test_objective_respects_off_diagonal_multi_positive_matches():
    logits = torch.tensor(
        [[3.0, 4.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
    )
    mask = positive_mask_from_group_ids(torch.tensor([7, 7, 9]))

    _, output = retrieval_risk_objective(
        RetrievalRiskSidecar(hidden_dim=8), logits, mask
    )

    assert output["t2v"]["failure"].tolist() == [0.0, 0.0, 0.0]
    assert output["v2t"]["failure"].tolist() == [0.0, 0.0, 0.0]


def test_tiny_training_orders_ambiguous_queries_above_clear_queries():
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [8.0, -4.0, -4.0, -4.0],
            [-4.0, 8.0, -4.0, -4.0],
            [-4.0, -4.0, 1.0, 0.9],
            [-4.0, -4.0, 0.9, 1.0],
        ]
    )
    positive_mask = torch.eye(4, dtype=torch.bool)
    sidecar = RetrievalRiskSidecar(hidden_dim=8)
    optimizer = torch.optim.Adam(sidecar.parameters(), lr=0.03)

    for _ in range(120):
        loss, _ = retrieval_risk_objective(sidecar, logits, positive_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    risk = sidecar(logits, "t2v")["failure_risk"]
    assert risk[2:].mean() > risk[:2].mean()
