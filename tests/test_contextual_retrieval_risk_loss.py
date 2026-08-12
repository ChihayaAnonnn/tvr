import pytest
import torch

from prob_models.contextual_retrieval_risk import (
    ContextualRetrievalRiskSidecar,
    contextual_retrieval_risk_objective,
)


def _inputs():
    logits = torch.tensor(
        [[0.0, 4.0, 1.0], [3.0, 0.0, 5.0], [2.0, 1.0, 4.0]],
        requires_grad=True,
    )
    text = torch.randn(3, 6, requires_grad=True)
    video = torch.randn(3, 6, requires_grad=True)
    mask = torch.tensor(
        [[True, True, False], [True, False, False], [False, False, True]]
    )
    return logits, text, video, mask


def test_objective_uses_multi_positive_failures_and_bounded_weights():
    logits, text, video, mask = _inputs()
    sidecar = ContextualRetrievalRiskSidecar(embed_dim=6, topk=2, dropout=0.0)

    loss, output = contextual_retrieval_risk_objective(
        sidecar, logits, text, video, mask
    )

    assert loss.ndim == 0
    assert output["t2v"]["failure"].tolist() == [0.0, 1.0, 0.0]
    assert output["v2t"]["failure"].tolist() == [0.0, 0.0, 1.0]
    assert output["t2v"]["positive_weight"].item() == pytest.approx(2.0)
    assert output["v2t"]["positive_weight"].item() == pytest.approx(2.0)
    assert 0.25 <= output["t2v"]["positive_weight"].item() <= 4.0
    assert 0.25 <= output["v2t"]["positive_weight"].item() <= 4.0
    assert {"bce", "fit", "rank", "t2v", "v2t"} <= output.keys()
    assert {"failure_rate", "prediction_mean", "prediction_std", "target_mean", "target_std", "valid_ranking_fraction", "entropy", "saturation_fraction"} <= output["t2v"].keys()


def test_objective_ranking_tracks_harder_rows_and_updates_both_projections():
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [8.0, -4.0, -4.0, -4.0],
            [-4.0, 8.0, -4.0, -4.0],
            [-4.0, -4.0, 0.9, 1.0],
            [-4.0, -4.0, 1.0, 0.9],
        ],
        requires_grad=True,
    )
    text = torch.randn(4, 5, requires_grad=True)
    video = torch.randn(4, 5, requires_grad=True)
    sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=5, hidden_dim=8, topk=2, dropout=0.0
    )
    optimizer = torch.optim.Adam(sidecar.parameters(), lr=0.03)

    for _ in range(120):
        loss, _ = contextual_retrieval_risk_objective(
            sidecar, logits, text, video, torch.eye(4, dtype=torch.bool)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    risk = sidecar(logits, text, video, "t2v")["failure_risk"]
    assert risk[2:].mean() > risk[:2].mean()
    assert logits.grad is None and text.grad is None and video.grad is None
    assert sidecar.text_projection.weight.grad is not None
    assert sidecar.video_projection.weight.grad is not None


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True], [True, False]]),
        torch.tensor([[False, False], [True, False]]),
    ],
)
def test_objective_rejects_rows_without_positive_and_negative_candidates(mask):
    sidecar = ContextualRetrievalRiskSidecar(embed_dim=3, topk=2)
    with pytest.raises(ValueError):
        contextual_retrieval_risk_objective(
            sidecar,
            torch.ones(2, 2),
            torch.ones(2, 3),
            torch.ones(2, 3),
            mask,
        )
