from dataclasses import replace

import pytest
import torch


@pytest.fixture
def alignment_config():
    from semantic_retrieval.config import AlignmentConfig

    return AlignmentConfig(0.1, 0.5, 500, 1e-5, 0)


def test_teacher_marginals_padding_detach_and_dustbin(alignment_config):
    from semantic_retrieval.objectives.alignment import _transport, dustbin_ot_target

    sim = torch.tensor([[[0.8, 0.2, 999], [0.1, 0.7, 999], [999, 999, 999]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    target = dustbin_ot_target(sim, mask, mask, config=alignment_config)
    assert not target.requires_grad and torch.isfinite(target).all()
    torch.testing.assert_close(target.sum(-1)[mask], torch.ones(2))
    assert not target[:, 2].any() and not target[:, :, 2].any()
    mass = _transport(sim[0, :2, :2].detach(), alignment_config)
    margins = torch.tensor([0.25, 0.25, 0.5], dtype=mass.dtype)
    torch.testing.assert_close(mass.sum(0), margins, atol=1e-5, rtol=0)
    torch.testing.assert_close(mass.sum(1), margins, atol=1e-5, rtol=0)
    one = torch.ones(1, 1, dtype=torch.bool)
    high = dustbin_ot_target(torch.ones(1, 1, 1), one, one, config=alignment_config)
    low = dustbin_ot_target(-torch.ones(1, 1, 1), one, one, config=alignment_config)
    assert low[0, 0, -1] > high[0, 0, -1]
    with pytest.raises(ValueError):
        dustbin_ot_target(sim, mask, mask, config=replace(alignment_config, max_iterations=1, tolerance=1e-14))
    with pytest.raises(ValueError):
        dustbin_ot_target(sim, torch.zeros_like(mask), mask, config=alignment_config)


def test_student_gradient_zero_targets_are_finite(alignment_config):
    from semantic_retrieval.objectives.alignment import alignment_loss

    logits = torch.randn(1, 2, 3, requires_grad=True)
    attention = logits.softmax(-1)
    target = torch.tensor([[[1.0, 0, 0], [0, 0, 0]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    loss = alignment_loss(attention, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0, 0].abs().sum() > 0 and not logits.grad[0, 1].any()


def test_positive_only_optional_alignment_updates_ranking(model_config, train_batch, tmp_path, alignment_config):
    from test_trainer import trainer, training_context

    from semantic_retrieval.models.model import SemanticRetrievalModel

    config = replace(model_config, alignment=alignment_config)
    sub = tmp_path / "align"
    sub.mkdir()
    context, cfg = training_context.__wrapped__(sub, config, only_align=True)
    m = SemanticRetrievalModel(config)
    t = trainer(m, context, cfg)
    before = m.score_pairs(m.encode(train_batch.inputs), train_batch.pair_indices).logits.detach().clone()
    metrics = t.train_step(train_batch)
    assert metrics["valid_align_count"] == 7
    after = m.score_pairs(m.encode(train_batch.inputs), train_batch.pair_indices).logits.detach()
    assert not torch.equal(before, after)
    negatives = replace(
        train_batch,
        targets=replace(
            train_batch.targets,
            relevance=torch.zeros(2, 3, dtype=torch.long),
            candidate_presence=torch.zeros(2, dtype=torch.long),
        ),
    )
    assert t.train_step(negatives)["stepped"] == 0
