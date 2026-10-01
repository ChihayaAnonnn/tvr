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


def test_actual_train_calibrate_evaluate_entrypoints(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    import cv2
    import numpy as np
    from test_runtime_artifacts import commit_reservation, reservation

    from scripts import evaluate_semantic_retrieval as eval_cli
    from scripts import train_semantic_retrieval as train_cli
    from semantic_retrieval.config import (
        ClipConfig,
        DataConfig,
        DeploymentConfig,
        ModelConfig,
        PolicyConfig,
        SystemConfig,
        TrainConfig,
        config_payload,
        effective_manifest_config,
    )
    from semantic_retrieval.models.clip import ClipBackbone
    from semantic_retrieval.runtime.checkpoint import config_sha256

    source = tmp_path / "inputs"
    source.mkdir()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    clips = []
    for i in range(2):
        path = source / f"v{i}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (16, 16))
        writer.write(np.full((16, 16, 3), 40 + i * 80, dtype=np.uint8))
        writer.release()
        clips.append(path)
    pre = dict(max_text_length=5, max_frames=4, image_size=8)
    clip = ClipConfig(8, 8, 64, 1, 4, 5, 49408, 8, 1, 1, fixture=True)
    weights = source / "clip.pt"
    torch.save(ClipBackbone(clip).state_dict(), weights)
    mc = ModelConfig(clip, weights, digest(weights), 8, 0, 2, 4, 6)
    tc = TrainConfig(
        0.001,
        0.0,
        7,
        "constant",
        1,
        ("coarse.", "matcher.", "answerability."),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=1.0, align=0.0, group=0.0),
    )
    policy = PolicyConfig(0.5, 0.5, (2,), 2)

    def episode_file(split):
        rows = []
        for j in range(2):
            provenance = dict(source="synthetic audited fixture", reviewed=True, sha256="a" * 64)
            if j == 1:
                provenance["gallery_audit"] = dict(
                    gallery_version="fixture", scope="complete", reviewed=True, source="synthetic complete library"
                )
            rows.append(
                dict(
                    query_id=f"{split}-q{j}",
                    query_text="a video",
                    gallery_version="fixture",
                    candidate_ids=["a", "b"],
                    canonical_ids=["a", "b"],
                    relevance=["positive", "negative"] if j == 0 else ["negative", "negative"],
                    answer_in_candidates="yes" if j == 0 else "no",
                    answer_in_gallery="yes" if j == 0 else "no",
                    annotation_provenance=provenance,
                    split=split,
                    video_paths=[str(x) for x in clips],
                    content_sha256=[digest(x) for x in clips],
                    frame_counts=[1, 1],
                    source_ids=["s0", "s1"],
                    caption_group_id=f"{split}-caption{j}",
                )
            )
        path = source / f"{split}.json"
        path.write_text(json.dumps(dict(schema_version=1, gallery_version="fixture", preprocessing=pre, episodes=rows)))
        return path

    gallery = source / "gallery.json"
    gallery.write_text(
        json.dumps(
            dict(
                schema_version=1,
                gallery_version="fixture",
                preprocessing=pre,
                videos=[
                    dict(canonical_id=i, alias_id=i, video_path=str(path), content_sha256=digest(path), frame_count=1)
                    for i, path in zip(["a", "b"], clips)
                ],
            )
        )
    )

    def run(split, dep=None):
        path = episode_file(split)
        dc = DataConfig(path, digest(path), "fixture", split)
        cfg = SystemConfig(None, mc, tc, policy, dc, dep)
        inputs = [dict(path=str(weights), sha256=digest(weights))]
        refs = [dict(path=str(path), sha256=digest(path))]
        if dep:
            inputs.append(dict(path=str(dep.checkpoint_path), sha256=dep.checkpoint_sha256))
            refs.append(dict(path=str(gallery), sha256=digest(gallery)))
            if dep.calibration_path:
                refs.append(dict(path=str(dep.calibration_path), sha256=dep.calibration_sha256))
        phase = tmp_path / f"{split}-{dep.mode if dep else 'train'}"
        phase.mkdir()
        repo, rt = reservation.__wrapped__(
            phase,
            SimpleNamespace(
                param=dict(
                    config=effective_manifest_config(cfg, eval_split=split),
                    input_checkpoints=inputs,
                    dataset_splits=refs,
                )
            ),
        )
        rt = replace(rt, eval_split=split)
        cfg = replace(cfg, runtime=rt)
        commit_reservation(repo, rt)
        config = rt.run_dir / "provenance" / "system-config.json"
        config.write_text(json.dumps(config_payload(cfg)))
        module = train_cli if dep is None else eval_cli
        monkeypatch.setattr(module, "__file__", str(repo / "scripts" / "entry.py"))
        assert module.main(["--config", str(config)]) == 0
        status = json.loads((rt.run_dir / "status.json").read_text())
        assert status["state"] == "validating"
        return cfg, rt

    train_cfg, train_rt = run("train")
    checkpoint = next((train_rt.run_dir / "checkpoints").glob("*.pt"))
    cal_input = episode_file("val_calibration")
    dep = DeploymentConfig(
        checkpoint,
        digest(checkpoint),
        config_sha256(train_cfg),
        gallery,
        digest(gallery),
        None,
        None,
        digest(cal_input),
        "calibrate",
    )
    _, cal_rt = run("val_calibration", dep)
    calibration = cal_rt.run_dir / "provenance" / "calibration.json"
    dep = replace(dep, mode="evaluate", calibration_path=calibration, calibration_sha256=digest(calibration))
    _, eval_rt = run("val_locked", dep)
    assert len(list((eval_rt.run_dir / "metrics").glob("*.json"))) == 1


@pytest.mark.parametrize("defect", ["hash", "frame_count"])
def test_episode_and_gallery_canonical_content_must_agree(defect):
    from semantic_retrieval.data.gallery import audit_episode_gallery

    pre = SimpleNamespace(max_text_length=5, max_frames=4, image_size=8)
    gallery = SimpleNamespace(preprocessing=pre, content_sha256={"a": "a" * 64}, frame_counts={"a": 1})
    episode = SimpleNamespace(
        preprocessing=pre,
        canonical_ids=("a",),
        content_sha256=("b" * 64 if defect == "hash" else "a" * 64,),
        frame_counts=(2 if defect == "frame_count" else 1,),
    )
    with pytest.raises(ValueError, match="content|frame-count"):
        audit_episode_gallery([episode], gallery)


def test_preprocessing_changes_parameters_not_control_commit():
    from semantic_retrieval.data.episodes import EpisodePreprocessing
    from semantic_retrieval.data.gallery import preprocessing_sha256

    pre = EpisodePreprocessing(5, 4, 8)
    assert preprocessing_sha256(pre) == preprocessing_sha256(pre)
    assert preprocessing_sha256(pre) != preprocessing_sha256(replace(pre, max_frames=3))
