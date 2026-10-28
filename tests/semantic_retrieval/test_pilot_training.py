"""CPU evidence for the preregistered dynamic training protocol."""

import collections
import random
from dataclasses import replace

import pytest
import torch

from semantic_retrieval.models.answerability import AnswerabilityHead
from semantic_retrieval.models.pair_matcher import PairMatcher
from semantic_retrieval.types import EncodedFeatures, PairScores


def test_sampler_unique_global_queries_balanced_schedule_and_exact_restore():
    from semantic_retrieval.runtime.pilot import PilotSampler

    ids = tuple(f"q{i}" for i in range(37))
    before = random.getstate()
    sampler = PilotSampler(ids, seed=11)
    controls = collections.Counter()
    table = []
    for _ in range(120):
        rows, k, panel = sampler.next_batch()
        assert len(rows) == len(set(rows)) == 16
        controls[k, panel] += 1
        table.append(rows)
    assert controls == {(k, p): 15 for k in (4, 8, 16, 32) for p in ("present", "absent")}
    assert len(set(table)) > 100
    assert random.getstate() == before
    restored = PilotSampler(ids, seed=11)
    restored.load_state_dict(sampler.state_dict())
    assert restored.next_batch() == sampler.next_batch()
    with pytest.raises(ValueError):
        PilotSampler(ids[:-1], seed=11).load_state_dict(sampler.state_dict())
    assert PilotSampler(ids, seed=11).next_batch()[0] == table[0]


def test_sampler_shards_without_rank_overlap():
    from semantic_retrieval.runtime.pilot import PilotSampler

    samplers = [PilotSampler(tuple(map(str, range(17))), seed=3) for _ in range(4)]
    for _ in range(12):
        batches = [s.next_batch(rank=r, world_size=4)[0] for r, s in enumerate(samplers)]
        assert all(len(b) == 4 for b in batches)
        assert len(set(q for b in batches for q in b)) == 16


def test_score_only_features_count_and_detached_gradient():
    from semantic_retrieval.models.answerability import ScoreOnlyAnswerabilityHead, score_statistics

    scores = torch.tensor([3.0, 1.0, 0.0, -1.0], requires_grad=True)
    features = score_statistics(scores)
    probs = scores.softmax(0)
    expected = torch.stack(
        [
            scores[0],
            scores[1],
            scores[0] - scores[1],
            scores.mean(),
            scores.std(unbiased=False),
            -(probs * probs.log()).sum() / torch.tensor(4.0).log(),
            scores.logsumexp(0),
            torch.tensor(5.0).log(),
        ]
    )
    torch.testing.assert_close(features, expected)
    head = ScoreOnlyAnswerabilityHead()
    assert sum(p.numel() for p in head.parameters()) == 8326081
    assert sum(p.numel() for p in AnswerabilityHead(2050).parameters()) == 8413201
    out = head(PairScores(scores, torch.randn(4, 6), None), torch.tensor([0, 4]))
    out.sum().backward()
    assert scores.grad is None
    assert all(p.grad is not None for p in head.parameters())


def pair_inputs(dim=8):
    text, frames = torch.randn(2, 3, dim), torch.randn(3, 2, dim)
    return EncodedFeatures(
        text,
        text.mean(1),
        frames,
        frames.mean(1),
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(3, 2, dtype=torch.bool),
    )


def test_a0_disables_attention_and_residual_gradients():
    matcher = PairMatcher(8, wti_only=True)
    out = matcher(pair_inputs(), torch.tensor([[0, 0], [0, 1], [1, 2]]))
    out.logits.sum().backward()
    assert out.attention is None and out.dustbin_mass is None
    assert all(p.grad is None for p in matcher.mlp.parameters())
    assert matcher.dustbin.grad is None
    assert matcher.wti.text_weight.weight.grad is not None


@pytest.mark.parametrize("arm,expected", [("A2", False), ("A3", True), ("A4", False)])
def test_presence_gradient_isolation(model_config, inputs, arm, expected):
    from semantic_retrieval.models.model import SemanticRetrievalModel

    model = SemanticRetrievalModel(replace(model_config, pilot_arm=arm))
    pairs = model.score_pairs(model.encode(inputs), torch.tensor([[0, 0], [0, 1], [1, 2]]))
    model.score_presence(pairs, torch.tensor([0, 2, 3])).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.matcher.parameters()) == expected
    model.zero_grad(set_to_none=True)
    pairs = model.score_pairs(model.encode(inputs), torch.tensor([[0, 0], [0, 1], [1, 2]]))
    pairs.logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.matcher.parameters())


def test_controlled_panels_follow_ranking_and_preserve_annotations():
    from pathlib import Path

    from semantic_retrieval.data.episodes import Episode, EpisodePreprocessing
    from semantic_retrieval.runtime.pilot import controlled_panel

    ids = tuple(f"v{i:02d}" for i in range(34))
    e = Episode(
        "q",
        "text",
        "g",
        ids,
        ids,
        (1,) + (0,) * 33,
        1,
        1,
        {},
        "train",
        tuple(Path(i) for i in ids),
        ("a" * 64,) * 34,
        (1,) * 34,
        ("s",),
        "c",
        EpisodePreprocessing(32, 8, 224),
        {i: (i,) for i in ids},
    )
    order = list(reversed(ids))
    present = controlled_panel(e, order, 4, "present")
    absent = controlled_panel(e, order, 4, "absent")
    assert present.canonical_ids == ("v00", "v31", "v32", "v33")
    assert present.relevance == (1, 0, 0, 0) and present.answer_in_candidates == 1
    assert absent.canonical_ids == ("v30", "v31", "v32", "v33")
    assert absent.answer_in_candidates == 0 and absent.answer_in_gallery == 1
    with pytest.raises(ValueError):
        controlled_panel(replace(e, relevance=(1, 1) + (0,) * 32), order, 4, "present")


def test_global_benchmark_relations_include_gt_omitted_from_absent_panel(train_batch):
    from semantic_retrieval.runtime.trainer import global_coarse_relations

    batch = replace(
        train_batch, benchmark_positive_ids=((train_batch.canonical_ids[1],), (train_batch.canonical_ids[0],))
    )
    _, labels = global_coarse_relations(batch, torch.randn(len(batch.canonical_ids), 3))
    for q, positives in enumerate(batch.benchmark_positive_ids):
        assert labels[q].tolist() == [int(v in positives) for v in sorted(batch.canonical_ids)]


def test_pilot_configuration_binds_arm_prefixes_and_schedule(model_config, tmp_path):
    from types import SimpleNamespace

    from semantic_retrieval.config import PilotTrainConfig, TrainConfig
    from semantic_retrieval.runtime.pilot import pilot_trainable_prefixes, validate_pilot_config

    pilot = PilotTrainConfig("P1", 0, tmp_path / "orders.json", "a" * 64)
    train = TrainConfig(
        1e-4,
        0.01,
        0,
        "constant",
        120,
        pilot_trainable_prefixes("P1"),
        dict(coarse=1, match=1, pair=1, exist=0, align=0, group=0),
        pilot=pilot,
    )
    cfg = SimpleNamespace(
        model=replace(model_config, pilot_arm="P1", num_layers=0), train=train, data=SimpleNamespace(split="train")
    )
    validate_pilot_config(cfg, world_size=4)
    with pytest.raises(ValueError):
        validate_pilot_config(cfg, world_size=1)
    with pytest.raises(ValueError):
        validate_pilot_config(
            SimpleNamespace(model=cfg.model, train=replace(train, scheduler_steps=119), data=cfg.data), world_size=4
        )


def test_gradient_probe_preserves_rng_grads_parameters_and_optimizer_step(model_config, train_batch):
    from semantic_retrieval.config import TrainConfig
    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import model_sha256
    from semantic_retrieval.runtime.pilot import probe_matching_presence_gradients
    from semantic_retrieval.runtime.trainer import build_optimizer

    train = TrainConfig(
        1e-4,
        0.01,
        0,
        "constant",
        120,
        ("matcher.", "answerability."),
        dict(coarse=1, match=1, pair=1, exist=1, align=0, group=0),
    )
    model = SemanticRetrievalModel(replace(model_config, pilot_arm="A3"))
    opt, scheduler = build_optimizer(model, train)
    digest = model_sha256(model)
    rng = torch.get_rng_state().clone()
    values = probe_matching_presence_gradients(model, train_batch, train.loss_weights)
    assert values["matching_gradient_norm"] > 0 and values["presence_gradient_norm"] > 0
    assert -1 <= values["gradient_cosine"] <= 1
    assert model_sha256(model) == digest
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters()) and not opt.state and scheduler.last_epoch == 0


def parent_fixture(tmp_path, model_config):
    import json

    from semantic_retrieval.config import (
        DataConfig,
        PilotTrainConfig,
        PolicyConfig,
        RuntimeConfig,
        SystemConfig,
        TrainConfig,
        config_payload,
    )
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import config_sha256, model_sha256, rng_state
    from semantic_retrieval.runtime.pilot import PilotSampler, pilot_trainable_prefixes

    orders = tmp_path / "orders.json"
    orders.write_text("{}")
    data = tmp_path / "data.json"
    data.write_text("{}")
    runtime = RuntimeConfig(
        tmp_path / "study.yaml", tmp_path / "registry.jsonl", tmp_path / "p1", "p1", "a" * 64, "b" * 40, "train"
    )
    pilot = PilotTrainConfig("P1", 4, orders, file_sha256(orders))
    train = TrainConfig(
        1e-4,
        0.01,
        0,
        "constant",
        120,
        pilot_trainable_prefixes("P1"),
        dict(coarse=1, match=1, pair=1, exist=0, align=0, group=0),
        pilot=pilot,
    )
    cfg = SystemConfig(
        runtime,
        replace(model_config, pilot_arm="P1", num_layers=0),
        train,
        PolicyConfig(0.5, 0.5, (4, 8, 16, 32, 64, 128), 128),
        DataConfig(data, file_sha256(data), "g", "train"),
    )
    config_path = tmp_path / "parent-config.json"
    config_path.write_text(json.dumps(config_payload(cfg)))
    torch.manual_seed(0)
    model = SemanticRetrievalModel(cfg.model)
    with torch.no_grad():
        model.matcher.b.fill_(0.7)
        model.answerability.network[-1].bias.fill_(17.0)
    sampler = PilotSampler(tuple(str(i) for i in range(37)), seed=4)
    for _ in range(120):
        sampler.next_batch()
    metrics = dict(
        coarse=0.0,
        match=0.0,
        pair=0.0,
        exist=0.0,
        align=0.0,
        group=0.0,
        total=0.0,
        step=120,
        stepped=1,
        **{"valid_" + k + "_count": 0 for k in ("coarse", "match", "pair", "exist", "align", "group")},
    )
    checkpoint = tmp_path / "p1.pt"
    torch.save(
        dict(
            schema_version=2,
            config_sha256=config_sha256(cfg),
            step=120,
            model=model.state_dict(),
            optimizer={},
            scheduler={},
            rng=[rng_state()],
            progress=dict(next_iteration=120, metrics=metrics, sampler=sampler.state_dict()),
        ),
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                sha256=file_sha256(checkpoint),
                config_sha256=config_sha256(cfg),
                model_sha256=model_sha256(model),
            )
        )
    )
    fields = dict(
        parent_config_path=config_path,
        parent_config_sha256=file_sha256(config_path),
        parent_checkpoint_path=checkpoint,
        parent_checkpoint_sha256=file_sha256(checkpoint),
        parent_metadata_sha256=file_sha256(checkpoint.with_suffix(".json")),
        parent_run_id="p1",
    )
    return cfg, fields, model


def test_branch_parent_matching_exact_and_heads_fresh_identical(tmp_path, model_config):
    from semantic_retrieval.runtime.pilot import initialize_pilot_model, pilot_trainable_prefixes

    parent, fields, expected = parent_fixture(tmp_path, model_config)
    children = []
    for arm in ("A1", "A2", "A3", "A4"):
        cfg = replace(
            parent,
            model=replace(parent.model, pilot_arm=arm),
            train=replace(
                parent.train,
                pilot=replace(parent.train.pilot, arm=arm, **fields),
                trainable_prefixes=pilot_trainable_prefixes(arm),
                loss_weights={**parent.train.loss_weights, "exist": int(arm != "A1")},
            ),
        )
        model, state = initialize_pilot_model(cfg)
        assert state["iteration"] == 120
        for name, tensor in expected.state_dict().items():
            if not name.startswith("answerability."):
                torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
        children.append(model)
    for a, b in zip(children[1].answerability.parameters(), children[2].answerability.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert children[2].answerability.network[-1].bias != 17
    fields["parent_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        initialize_pilot_model(replace(cfg, train=replace(cfg.train, pilot=replace(cfg.train.pilot, **fields))))


def test_real_entry_batch_builder_consumes_dynamic_panels(tmp_path):
    import cv2
    import numpy as np

    from scripts.train_semantic_retrieval import next_pilot_batch
    from semantic_retrieval.data.episodes import Episode, EpisodePreprocessing, file_sha256
    from semantic_retrieval.runtime.pilot import PilotSampler

    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1, (16, 16))
    writer.write(np.full((16, 16, 3), 100, dtype=np.uint8))
    writer.release()
    ids = tuple(f"v{i:02d}" for i in range(33))
    base = Episode(
        "q0",
        "a video",
        "g",
        ids,
        ids,
        (1,) + (0,) * 32,
        1,
        1,
        {"reviewed": True, "source": "synthetic", "sha256": "a" * 64},
        "train",
        (video,) * 33,
        (file_sha256(video),) * 33,
        (1,) * 33,
        ("s",),
        "c",
        EpisodePreprocessing(32, 8, 8),
        {i: (i,) for i in ids},
    )
    inventory = {f"q{i}": replace(base, query_id=f"q{i}", caption_group_id=f"c{i}") for i in range(37)}
    orders = {q: list(reversed(ids)) for q in inventory}
    sampler = PilotSampler(tuple(inventory), seed=3)
    actual = []
    for _ in range(8):
        batch, k, panel = next_pilot_batch(inventory, orders, sampler, rank=0, world_size=4, device=torch.device("cpu"))
        assert len(batch.query_ids) == 4 and len(batch.pair_indices) == 4 * k
        assert batch.targets.candidate_presence.tolist() == [int(panel == "present")] * 4
        assert batch.inputs.input_ids.shape[0] == 4
        assert batch.inputs.video.shape[1:] == (8, 3, 8, 8)
        actual.append(batch.query_ids)
    assert len(set(actual)) == 8
    assert sampler.iteration == 8


def test_pilot_checkpoint_requires_phase_consistent_sampler_progress():
    from semantic_retrieval.runtime.checkpoint import validate_pilot_progress

    sampler = dict(iteration=120)
    validate_pilot_progress(dict(next_iteration=120, sampler=sampler), "P1", 120)
    validate_pilot_progress(dict(next_iteration=0, sampler=sampler), "A3", 0)
    for arm, step, progress in [
        ("P1", 0, None),
        ("P1", 120, dict(next_iteration=119, sampler=sampler)),
        ("A3", 1, dict(next_iteration=1, sampler=sampler)),
    ]:
        with pytest.raises(ValueError):
            validate_pilot_progress(progress, arm, step)
