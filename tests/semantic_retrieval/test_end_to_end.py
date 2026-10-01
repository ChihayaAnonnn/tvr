from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiment_artifacts import ArtifactArrayStore
from semantic_retrieval.types import Decision, QueryInputs


def test_calibration_split_classes_and_unknown():
    from semantic_retrieval.evaluation.calibration import CalibrationIdentity, fit_calibration

    identity = CalibrationIdentity("a" * 64, "b" * 64, "g", "c" * 64, "d" * 64)
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0, 1000.0])
    y = torch.tensor([0, 0, 1, 1, -1])
    state = fit_calibration(x, y, x, y, split="val_calibration", identity=identity)
    assert state.pair_temperature > 0 and state.presence_temperature > 0
    changed = fit_calibration(x[:4], y[:4], x[:4], y[:4], split="val_calibration", identity=identity)
    assert state == changed
    for split in ("test", "train", "val_locked"):
        with pytest.raises(ValueError):
            fit_calibration(x, y, x, y, split=split, identity=identity)
    with pytest.raises(ValueError):
        fit_calibration(x, torch.ones(5, dtype=torch.long), x, y, split="val_calibration", identity=identity)


def make_service(model, inputs, tmp_path, policy=None):
    from semantic_retrieval.config import PolicyConfig
    from semantic_retrieval.evaluation.calibration import CalibrationIdentity, CalibrationState
    from semantic_retrieval.retrieval.cache import FeatureCache
    from semantic_retrieval.retrieval.index import CanonicalIndex, IndexIdentity
    from semantic_retrieval.runtime.checkpoint import model_sha256
    from semantic_retrieval.runtime.inference import RetrievalService, policy_sha256
    from semantic_retrieval.types import VideoFeatures

    model.eval().requires_grad_(False)
    identity = IndexIdentity(model_sha256(model), "g", "b" * 64)
    config = policy or PolicyConfig(1.0, 1.0, (1, 2, 3), 3)
    calibration = CalibrationState(
        CalibrationIdentity(identity.model_sha256, identity.preprocessing_sha256, "g", policy_sha256(config), "d" * 64),
        1.0,
        0.0,
        1.0,
        0.0,
    )
    cache = FeatureCache(SimpleNamespace(array_store=ArtifactArrayStore(tmp_path / "arrays")))
    with torch.no_grad():
        videos = model.encode_gallery(inputs.video, inputs.frame_mask)
        embeddings = model.project_gallery(videos)
    ids = ("a", "b", "c")
    hashes = {i: "e" * 64 for i in ids}
    for j, i in enumerate(ids):
        cache.write(
            i,
            VideoFeatures(videos.frame_tokens[j : j + 1], videos.pooled_video[j : j + 1], videos.frame_mask[j : j + 1]),
            identity=identity,
            content_sha256=hashes[i],
        )
    index = CanonicalIndex(ids, embeddings, identity)
    return RetrievalService(model, index, cache, calibration, config, content_sha256=hashes), calibration


def test_service_expansion_reuses_pairs_and_has_no_gt(model, inputs, tmp_path, monkeypatch):
    service, _ = make_service(model, inputs, tmp_path)
    counts = []
    original = model.score_pairs

    def observed(f, pairs, **kwargs):
        counts.append(len(pairs))
        return original(f, pairs, **kwargs)

    monkeypatch.setattr(model, "score_pairs", observed)
    q = QueryInputs(inputs.input_ids[:1], inputs.text_mask[:1])
    decision = service.retrieve(q, budget=3)
    assert decision.action == "abstain" and counts == [1, 1, 1]
    assert service.last_prediction.budget_used == 3
    assert service.retrieve(q, budget=3) == decision
    with pytest.raises(ValueError):
        service.retrieve(inputs, budget=3)
    model.matcher.b.data.add_(1)
    with pytest.raises(ValueError):
        service.retrieve(q, budget=3)


def test_return_set_and_metrics(model, inputs, tmp_path):
    from semantic_retrieval.config import PolicyConfig
    from semantic_retrieval.evaluation.episodes import EpisodePrediction, evaluate_episodes

    service, _ = make_service(model, inputs, tmp_path, PolicyConfig(0.0, 0.0, (3,), 3))
    decision = service.retrieve(QueryInputs(inputs.input_ids[:1], inputs.text_mask[:1]), budget=3)
    assert decision.action == "return" and decision.answer_ids == ("a", "b", "c")
    pred = EpisodePrediction("q", decision, ("a", "b", "c"), ("a",), (1, 3), 3)
    gt = SimpleNamespace(query_id="q", canonical_ids=("a", "b", "c"), relevance=(0, 1, 1), answer_in_gallery=1)
    metrics = evaluate_episodes([pred], [gt], split="val_locked")
    assert metrics["coverage"] == dict(numerator=1, denominator=1, value=1.0)
    assert metrics["set_recall"]["value"] == 1.0 and metrics["set_precision"]["value"] == pytest.approx(2 / 3)
    assert metrics["expansion_recovery"]["value"] == 1.0
    no = replace(pred, query_id="no")
    nogt = SimpleNamespace(query_id="no", canonical_ids=("a", "b", "c"), relevance=(0, 0, 0), answer_in_gallery=0)
    unknown = replace(pred, query_id="unknown", decision=Decision("abstain", (), "no support", None))
    unknown_gt = SimpleNamespace(
        query_id="unknown", canonical_ids=("a", "b", "c"), relevance=(-1, -1, -1), answer_in_gallery=-1
    )
    extra = evaluate_episodes([no, unknown], [nogt, unknown_gt], split="val_locked")
    assert extra["false_accept"]["value"] == 1 and extra["false_abstain"]["value"] is None
    with pytest.raises(ValueError):
        evaluate_episodes([pred], [gt], split="test")


def test_checkpoint_to_service(model, inputs, train_batch, training_context, tmp_path):
    from test_trainer import trainer

    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, save_checkpoint

    context, cfg = training_context
    t = trainer(model, context, cfg)
    t.train_step(train_batch)
    path = save_checkpoint(context, model, t.optimizer, t.scheduler, cfg, t.step)
    restored = SemanticRetrievalModel(cfg.model)
    load_checkpoint(path, restored, expected_config_sha256=config_sha256(cfg))
    service, _ = make_service(restored, inputs, tmp_path)
    assert service.retrieve(QueryInputs(inputs.input_ids[:1], inputs.text_mask[:1]), budget=3).action == "abstain"
