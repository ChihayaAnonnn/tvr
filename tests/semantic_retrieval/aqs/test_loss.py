from dataclasses import replace
import pytest
import torch


def test_gradient_reaches_positive_and_selected_negative_only():
    from semantic_retrieval.aqs.selection import select_negatives
    from semantic_retrieval.aqs.loss import selected_t2v_loss
    raw = torch.tensor([[.2,.9,.4],[.8,.3,.6],[.5,.7,.1]], requires_grad=True)
    reps = torch.eye(3, requires_grad=True)
    selected = select_negatives(raw, reps, torch.arange(3), 1)
    logits = raw.clone().detach().requires_grad_()
    selected_t2v_loss(logits, selected).backward()
    rows = torch.arange(3)
    assert (logits.grad.diag() < 0).all()
    assert (logits.grad[rows, selected.indices[:, 0]] > 0).all()
    used = torch.eye(3, dtype=torch.bool)
    used[rows, selected.indices[:, 0]] = True
    assert (logits.grad[~used] == 0).all()
    assert raw.grad is None and reps.grad is None


@pytest.mark.parametrize('batch', [1, 3])
def test_empty_or_large_logits_stable(batch):
    from semantic_retrieval.aqs.selection import select_negatives
    from semantic_retrieval.aqs.loss import selected_t2v_loss
    logits = (torch.eye(batch)*1e6).requires_grad_()
    sel = select_negatives(logits, torch.eye(batch), torch.arange(batch), 2)
    loss = selected_t2v_loss(logits, sel)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    if batch == 1: assert loss.item() == 0


@pytest.mark.parametrize('indices', [torch.tensor([[0], [0], [0]]), torch.tensor([[-1], [0], [0]]),
                                     torch.tensor([[1, 1], [0, 0], [0, 0]])])
def test_forged_indices_rejected(indices):
    from semantic_retrieval.aqs.selection import select_negatives
    from semantic_retrieval.aqs.loss import selected_t2v_loss
    logits = torch.eye(3)
    sel = select_negatives(logits, torch.eye(3), torch.arange(3), 1)
    with pytest.raises(ValueError): selected_t2v_loss(logits, replace(sel, indices=indices))
