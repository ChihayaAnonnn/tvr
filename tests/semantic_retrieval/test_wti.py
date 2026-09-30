import pytest
import torch


def test_padding_does_not_beat_negative_similarity():
    from semantic_retrieval.models.wti import MaskedWTI

    score = MaskedWTI(2)(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[True]]),
        torch.tensor([[True, False]]),
    )
    assert score.item() == pytest.approx(-1.0)


def test_chunks_padding_and_backward():
    from semantic_retrieval.models.wti import MaskedWTI

    torch.manual_seed(5)
    model = MaskedWTI(4)
    text = torch.randn(2, 3, 4, requires_grad=True)
    frames = torch.randn(3, 2, 4, requires_grad=True)
    tm = torch.tensor([[True, True, False], [True, False, False]])
    fm = torch.tensor([[True, False], [True, True], [True, False]])
    full = model(text, frames, tm, fm)
    chunks = torch.cat([model(text, frames[i : i + 1], tm, fm[i : i + 1]) for i in range(3)], 1)
    torch.testing.assert_close(full, chunks)
    padded = model(
        torch.cat([text, torch.randn(2, 2, 4)], 1), frames, torch.cat([tm, torch.zeros(2, 2, dtype=torch.bool)], 1), fm
    )
    torch.testing.assert_close(full, padded)
    full.sum().backward()
    assert text.grad.isfinite().all() and frames.grad.isfinite().all()
    assert (text.grad[~tm] == 0).all() and (frames.grad[~fm] == 0).all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


@pytest.mark.parametrize("defect", ["empty", "nan"])
def test_invalid_matching_input(defect):
    from semantic_retrieval.models.wti import MaskedWTI

    text = torch.ones(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    if defect == "empty":
        mask[:] = False
    else:
        text[:] = float("nan")
    with pytest.raises(ValueError):
        MaskedWTI(2)(text, torch.ones(1, 1, 2), mask, torch.ones(1, 1, dtype=torch.bool))
