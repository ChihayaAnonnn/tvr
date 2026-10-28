import hashlib
from dataclasses import replace

import pytest
import torch


def test_symmetric_known_only_loss_and_complete_link():
    from semantic_retrieval.models.grouping import EquivalencePredictor, conservative_groups, group_loss

    torch.manual_seed(2)
    predictor = EquivalencePredictor(6)
    a = torch.randn(3, 6, requires_grad=True)
    b = torch.randn(3, 6, requires_grad=True)
    logits = predictor(a, b)
    torch.testing.assert_close(logits, predictor(b, a))
    loss = group_loss(logits, torch.tensor([1, 0, -1]))
    loss.backward()
    assert not a.grad[2].any() and a.grad[:2].abs().sum() > 0
    p = torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, 0.9], [0.1, 0.9, 1.0]])
    assert conservative_groups(["a", "b", "c"], p, 0.8) == (("a", "b"), ("c",))
    order = torch.tensor([2, 0, 1])
    assert conservative_groups(["c", "a", "b"], p[order][:, order], 0.8) == (("a", "b"), ("c",))
    with pytest.raises(ValueError):
        conservative_groups(["a", "b", "c"], p + 0.5, 0.8)


def test_group_supervision_requires_audit_and_changes_shared_ranking(
    model_config, training_context, train_batch, tmp_path
):
    from test_trainer import trainer

    from semantic_retrieval.config import GroupingConfig
    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import assert_running

    context, cfg = training_context
    report = tmp_path / "review.md"
    report.write_text("synthetic validation fixture only")
    group = GroupingConfig(0.8, report, hashlib.sha256(report.read_bytes()).hexdigest(), ("a" * 64,))
    from test_trainer import training_context as context_fixture

    sub = tmp_path / "group"
    sub.mkdir()
    context, cfg = context_fixture.__wrapped__(sub, replace(model_config, grouping=group), only_group=True)
    model = SemanticRetrievalModel(cfg.model)
    t = trainer(model, context, cfg)
    batch = replace(
        train_batch,
        targets=replace(train_batch.targets, group_pairs=torch.tensor([[0, 1]]), group_labels=torch.tensor([0])),
    )
    with pytest.raises(ValueError, match="provenance"):
        t.train_step(batch)
    batch = replace(batch, group_provenance=("a" * 64,))
    before = model.score_pairs(model.encode(batch.inputs), batch.pair_indices).logits.detach().clone()
    metrics = t.train_step(batch)
    after = model.score_pairs(model.encode(batch.inputs), batch.pair_indices).logits.detach()
    assert not torch.equal(before, after)
    assert metrics["valid_group_count"] == 1 and model.matcher.text_projection.weight.grad.abs().sum() > 0
    assert_running(context)
