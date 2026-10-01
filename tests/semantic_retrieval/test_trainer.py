import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_runtime_artifacts import commit_reservation, reservation

from semantic_retrieval.types import EpisodeBatch, Targets


@pytest.fixture
def train_batch(inputs):
    return EpisodeBatch(
        inputs,
        Targets(torch.tensor([[1, 0, -1], [-1, -1, 1]]), torch.ones(2, dtype=torch.long)),
        ("q0", "q1"),
        ("v0", "v1", "v2"),
        torch.tensor([[0, 0], [0, 1], [1, 2]]),
        torch.tensor([0, 2, 3]),
    )


@pytest.fixture
def training_context(tmp_path, model_config, *, only_group=False):
    from semantic_retrieval.config import PolicyConfig, SystemConfig, TrainConfig, effective_manifest_config
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run

    train = TrainConfig(
        0.001,
        0.0,
        4,
        "linear",
        4,
        ("temporal.", "coarse.", "matcher.", "answerability."),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=1.0, align=0.0, group=0.0),
    )
    if model_config.grouping is not None:
        train = replace(
            train,
            loss_weights={**train.loss_weights, "group": 1.0},
            trainable_prefixes=train.trainable_prefixes + ("grouping.",),
        )
    if only_group:
        train = replace(train, loss_weights={k: float(k == "group") for k in train.loss_weights})
    stub = SystemConfig(None, model_config, train, PolicyConfig(0.7, 0.6, (2, 4), 4))
    refs = [dict(path=str(model_config.clip_weights), sha256=model_config.clip_sha256)]
    repo, rt = reservation.__wrapped__(
        tmp_path,
        SimpleNamespace(
            param=dict(
                config=effective_manifest_config(stub, eval_split="val"),
                input_checkpoints=refs,
                dataset_splits=[
                    dict(
                        path=str(model_config.grouping.validation_report),
                        sha256=model_config.grouping.validation_report_sha256,
                    )
                ]
                if model_config.grouping is not None
                else [],
            )
        ),
    )
    cfg = replace(stub, runtime=rt)
    commit_reservation(repo, rt)
    context = prepare_run(rt, repo_root=repo, system_config=cfg)
    start_run(context)
    return context, cfg


def trainer(model, context, cfg):
    from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

    opt, sched = build_optimizer(model, cfg.train)
    return Trainer(model, opt, sched, context)


def test_steps_unknown_and_resume(model, training_context, train_batch, tmp_path):
    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, save_checkpoint

    context, cfg = training_context
    t = trainer(model, context, cfg)
    before = model.matcher.text_projection.weight.detach().clone()
    t.train_step(train_batch)
    assert not torch.equal(before, model.matcher.text_projection.weight)
    assert all(p.grad is None for p in model.clip.parameters())
    state = copy.deepcopy(model.state_dict())
    step = t.step
    unknown = replace(train_batch, targets=Targets(torch.full((2, 3), -1), torch.full((2,), -1)))
    assert t.train_step(unknown)["stepped"] == 0 and t.step == step
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    path = save_checkpoint(context, model, t.optimizer, t.scheduler, cfg, t.step)
    other = SemanticRetrievalModel(cfg.model)
    restored = trainer(other, context, cfg)
    restored.restore(load_checkpoint(path, other, expected_config_sha256=config_sha256(cfg)))
    t.train_step(train_batch)
    restored.train_step(train_batch)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, other.state_dict()[k], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_checkpoint(path, other, expected_config_sha256="0" * 64)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_checkpoint(path, other, expected_config_sha256=config_sha256(cfg))


def test_missing_optimizer_params_rejected(model, training_context):
    from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

    context, cfg = training_context
    opt, sched = build_optimizer(model, cfg.train)
    with pytest.raises(ValueError):
        Trainer(model, torch.optim.AdamW([model.matcher.b]), sched, context)


def test_effective_binding_rejects_mismatch(model, training_context):
    from semantic_retrieval.runtime.artifacts import prepare_run

    context, cfg = training_context
    # Running cannot be launched again, independently of config equality.
    with pytest.raises(ValueError):
        prepare_run(
            cfg.runtime,
            repo_root=context.repo_root,
            system_config=replace(cfg, train=replace(cfg.train, learning_rate=0.2)),
        )


def _rank_step(rank, init, context, cfg, batch, outdir):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    try:
        torch.manual_seed(101)
        m = SemanticRetrievalModel(cfg.model)
        opt, sched = build_optimizer(m, cfg.train)
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        ddp = DistributedDataParallel(m, find_unused_parameters=True)
        labels = batch.targets.relevance.clone()
        if rank == 1:
            labels[:] = -1
            labels[1, 2] = 1
        b = replace(
            batch,
            query_ids=tuple(f"r{rank}-{q}" for q in batch.query_ids),
            canonical_ids=tuple(f"r{rank}-{v}" for v in batch.canonical_ids),
            targets=Targets(labels, batch.targets.candidate_presence if rank == 0 else torch.full((2,), -1)),
        )
        t = Trainer(ddp, opt, sched, context)
        metrics = t.train_step(b)
        torch.save(dict(state=m.state_dict(), metrics=metrics), outdir / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_matches_global_known_only_objective(training_context, train_batch, tmp_path):
    import torch.multiprocessing as mp

    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.types import EncoderInputs

    context, cfg = training_context
    mp.spawn(_rank_step, args=(f"file://{tmp_path / 'gloo'}", context, cfg, train_batch, tmp_path), nprocs=2, join=True)
    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert left["metrics"]["valid_pair_count"] == 4
    for k, v in left["state"].items():
        torch.testing.assert_close(v, right["state"][k])
    labels = torch.full((4, 6), -1)
    labels[:2, :3] = train_batch.targets.relevance
    labels[3, 5] = 1
    inp = train_batch.inputs
    global_batch = EpisodeBatch(
        EncoderInputs(
            inp.input_ids.repeat(2, 1),
            inp.text_mask.repeat(2, 1),
            inp.video.repeat(2, 1, 1, 1, 1),
            inp.frame_mask.repeat(2, 1),
        ),
        Targets(labels, torch.tensor([1, 1, -1, -1])),
        ("r0-q0", "r0-q1", "r1-q0", "r1-q1"),
        tuple(f"r{r}-v{v}" for r in range(2) for v in range(3)),
        torch.cat([train_batch.pair_indices, train_batch.pair_indices + torch.tensor([2, 3])]),
        torch.tensor([0, 2, 3, 5, 6]),
    )
    torch.manual_seed(101)
    model = SemanticRetrievalModel(cfg.model)
    from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

    build_optimizer(model, cfg.train)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    t = Trainer(model, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0), context)
    t.train_step(global_batch)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, left["state"][k], rtol=1e-4, atol=1e-6)
