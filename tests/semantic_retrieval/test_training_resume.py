"""Exercise the real protected CLI, including controller-approved new attempts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from test_runtime_artifacts import commit_reservation, git, reservation

from scripts import manage_experiment_run as manager
from scripts import train_semantic_retrieval as train_cli
from semantic_retrieval.config import (
    DataConfig,
    PolicyConfig,
    SystemConfig,
    TrainConfig,
    config_payload,
    effective_manifest_config,
)
from semantic_retrieval.runtime.checkpoint import config_sha256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(tmp_path, model_config, *, unknown=False):
    from semantic_retrieval.models.clip import ClipBackbone

    clip = replace(model_config.clip, vocab_size=49408)
    weights = tmp_path / "cli-clip.pt"
    torch.save(ClipBackbone(clip).state_dict(), weights)
    model_config = replace(model_config, clip=clip, clip_weights=weights, clip_sha256=digest(weights))
    videos = []
    for i in range(2):
        path = tmp_path / f"video-{i}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (16, 16))
        writer.write(np.full((16, 16, 3), 40 + 80 * i, dtype=np.uint8))
        writer.release()
        videos.append(path)
    rows = []
    for i in range(2):
        rows.append(
            dict(
                query_id=f"q{i}",
                query_text="a video",
                gallery_version="fixture",
                candidate_ids=["a", "b"],
                canonical_ids=["a", "b"],
                relevance=["unknown", "unknown"] if unknown else ["positive", "negative"],
                answer_in_candidates="unknown" if unknown else "yes",
                answer_in_gallery="unknown" if unknown else "yes",
                annotation_provenance=dict(source="synthetic fixture", reviewed=True, sha256="a" * 64),
                split="train",
                video_paths=[str(p) for p in videos],
                content_sha256=[digest(p) for p in videos],
                frame_counts=[1, 1],
                source_ids=["s0", "s1"],
                caption_group_id=f"caption-{i}",
            )
        )
    episodes = tmp_path / "train.json"
    episodes.write_text(
        json.dumps(
            dict(
                schema_version=1,
                gallery_version="fixture",
                preprocessing=dict(max_text_length=5, max_frames=4, image_size=8),
                episodes=rows,
            )
        )
    )
    train = TrainConfig(
        0.001,
        0.0,
        7,
        "linear",
        4,
        ("coarse.", "matcher.", "answerability."),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=1.0, align=0.0, group=0.0),
        checkpoint_interval=2,
    )
    return SystemConfig(
        None,
        replace(model_config, num_layers=0),
        train,
        PolicyConfig(0.5, 0.5, (2,), 2),
        DataConfig(episodes, digest(episodes), "fixture", "train"),
    )


def initial_run(tmp_path, cfg):
    tmp_path.mkdir()
    repo, rt = reservation.__wrapped__(
        tmp_path,
        SimpleNamespace(
            param=dict(
                config=effective_manifest_config(cfg, eval_split="train"),
                input_checkpoints=[dict(path=str(cfg.model.clip_weights), sha256=cfg.model.clip_sha256)],
                dataset_splits=[dict(path=str(cfg.data.episode_manifest), sha256=cfg.data.episode_sha256)],
            )
        ),
    )
    rt = replace(rt, eval_split="train")
    cfg = replace(cfg, runtime=rt)
    commit_reservation(repo, rt)
    return repo, cfg


def invoke(repo, cfg, monkeypatch):
    path = cfg.runtime.run_dir / "provenance" / "system-config.json"
    path.write_text(json.dumps(config_payload(cfg)))
    monkeypatch.setattr(train_cli, "__file__", str(repo / "scripts" / "entry.py"))
    return train_cli.main(["--config", str(path)])


def new_attempt(repo, old, checkpoint, *, omit=None):
    from experiment_artifacts import transition_run_status
    from semantic_retrieval.config import ResumeConfig

    rt = old.runtime
    paths = manager.make_run_paths_from_dir(rt.run_dir)
    transition_run_status(paths, "running", "validating")
    transition_run_status(
        paths, "validating", "failed", exit_code=1, output_hashes=manager._all_output_hashes(rt.run_dir)
    )
    control = rt.study_path.parent
    summary = control / "failed-summary.json"
    manager.finalize_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=(control / "phase.json").relative_to(repo),
            registry=rt.registry_path.relative_to(repo),
            artifact_root=rt.run_dir.parents[3],
            summary=summary.relative_to(repo),
        )
    )
    report = repo / "resume-review.md"
    report.write_text("Synthetic test review: interruption after a committed checkpoint; authorize remaining work.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "terminal evidence and synthetic retry approval")
    resume = ResumeConfig(
        checkpoint, digest(checkpoint), digest(checkpoint.with_suffix(".json")), rt.run_id, report, digest(report)
    )
    cfg = replace(old, runtime=None, resume=resume)
    spec = json.loads((control / "phase.json").read_text())["runs"][0]
    spec.update(
        variant="resume",
        config=effective_manifest_config(cfg, eval_split="train"),
        parent_run_ids=[rt.run_id],
        selection_lineage=[
            dict(
                kind="training_resume",
                parent_run_id=rt.run_id,
                approved=True,
                report_path=report.relative_to(repo).as_posix(),
                report_sha256=digest(report),
                reason="Synthetic human-reviewed infrastructure interruption",
            )
        ],
    )
    spec["input_checkpoints"] += [
        dict(path=str(p), sha256=digest(p)) for p in (checkpoint, checkpoint.with_suffix(".json"))
    ]
    spec["dataset_splits"] += [dict(path=str(report), sha256=digest(report))]
    if omit == "checkpoint":
        spec["input_checkpoints"] = spec["input_checkpoints"][:1]
    if omit == "review":
        spec["selection_lineage"] = []
    if omit == "parent":
        spec["parent_run_ids"] = []
    phase = control / "resume-phase.json"
    phase.write_text(json.dumps(dict(phase="resume", study="fixture-study", runs=[spec])))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "new independently reserved resume phase")
    record = manager.reserve_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=phase.relative_to(repo),
            registry=rt.registry_path.relative_to(repo),
            artifact_root=rt.run_dir.parents[3],
        )
    )["runs"][0]
    manifest = json.loads((Path(record["artifact_path"]) / "manifest.json").read_text())
    fresh = replace(
        rt,
        run_id=record["run_id"],
        run_dir=Path(record["artifact_path"]),
        manifest_sha256=record["manifest_sha256"],
        source_sha=manifest["git_sha"],
    )
    commit_reservation(repo, fresh)
    return replace(cfg, runtime=fresh)


def interrupt_after_first_checkpoint(monkeypatch):
    save = train_cli.save_checkpoint
    saved = []

    def interrupted(*args, **kwargs):
        path = save(*args, **kwargs)
        saved.append(path)
        raise InterruptedError("synthetic interruption after checkpoint publication")

    monkeypatch.setattr(train_cli, "save_checkpoint", interrupted)
    return save, saved


@pytest.mark.parametrize("unknown", [False, True])
def test_periodic_resume_matches_uninterrupted_training(tmp_path, model_config, monkeypatch, unknown):
    cfg = configuration(tmp_path, model_config, unknown=unknown)
    repo, old = initial_run(tmp_path / "interrupted", cfg)
    save, saved = interrupt_after_first_checkpoint(monkeypatch)
    with pytest.raises(InterruptedError):
        invoke(repo, old, monkeypatch)
    checkpoint = saved[0]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["progress"]["next_iteration"] == 2
    assert payload["step"] == (0 if unknown else 2)
    fresh = new_attempt(repo, old, checkpoint)
    original = manager._all_output_hashes(old.runtime.run_dir)
    monkeypatch.setattr(train_cli, "save_checkpoint", save)
    assert invoke(repo, fresh, monkeypatch) == 0
    continuous_repo, continuous = initial_run(tmp_path / "continuous", cfg)
    assert invoke(continuous_repo, continuous, monkeypatch) == 0
    checkpoints = sorted(continuous.runtime.run_dir.joinpath("checkpoints").glob("*.pt"))
    assert len(checkpoints) == 2
    final = next(fresh.runtime.run_dir.joinpath("checkpoints").glob("*.pt"))
    actual = torch.load(final, map_location="cpu", weights_only=True)
    expected = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert actual["progress"]["next_iteration"] == 4
    assert actual["step"] == expected["step"] == (0 if unknown else 4)
    assert (
        json.loads(final.with_suffix(".json").read_text())["model_sha256"]
        == json.loads(checkpoints[-1].with_suffix(".json").read_text())["model_sha256"]
    )
    torch.testing.assert_close(actual["optimizer"], expected["optimizer"], rtol=0, atol=0)
    assert actual["scheduler"] == expected["scheduler"]
    assert config_sha256(old) == config_sha256(fresh)
    assert manager._all_output_hashes(old.runtime.run_dir) == original
    assert fresh.runtime.run_id != old.runtime.run_id


@pytest.mark.parametrize("defect", ["checkpoint", "review", "parent", "tamper"])
def test_resume_refuses_unreserved_or_unreviewed_inputs(tmp_path, model_config, monkeypatch, defect):
    repo, cfg = initial_run(tmp_path / "first", configuration(tmp_path, model_config))
    save, saved = interrupt_after_first_checkpoint(monkeypatch)
    with pytest.raises(InterruptedError):
        invoke(repo, cfg, monkeypatch)
    fresh = new_attempt(repo, cfg, saved[0], omit=defect)
    monkeypatch.setattr(train_cli, "save_checkpoint", save)
    if defect == "tamper":
        saved[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        invoke(repo, fresh, monkeypatch)
    assert json.loads((fresh.runtime.run_dir / "status.json").read_text())["state"] == "reserved"


def _rank_resume(rank, init, context, cfg, batch, outdir, collision):
    import random

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from semantic_retrieval.models.model import SemanticRetrievalModel
    from semantic_retrieval.runtime.checkpoint import load_checkpoint, save_checkpoint
    from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    try:
        torch.manual_seed(101)
        model = SemanticRetrievalModel(cfg.model)
        optimizer, scheduler = build_optimizer(model, cfg.train)
        worker = Trainer(DistributedDataParallel(model, find_unused_parameters=True), optimizer, scheduler, context)
        torch.rand(rank + 1)  # Distinct rank RNG states must survive the checkpoint.
        metrics = worker.train_step(batch)
        try:
            checkpoint = save_checkpoint(
                context, model, optimizer, scheduler, cfg, worker.step, progress=dict(next_iteration=1, metrics=metrics)
            )
        except ValueError as exc:
            if not collision:
                raise
            torch.save(dict(error=str(exc)), outdir / f"resume-rank-{rank}.pt")
            return
        assert not collision
        expected_random = (torch.rand(3), random.random(), np.random.rand(3))
        worker.train_step(batch)
        other = SemanticRetrievalModel(cfg.model)
        opt, sched = build_optimizer(other, cfg.train)
        restored = Trainer(DistributedDataParallel(other, find_unused_parameters=True), opt, sched, context)
        state = load_checkpoint(checkpoint, other, expected_config_sha256=config_sha256(cfg))
        restored.restore(state)
        torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
        assert random.random() == expected_random[1]
        np.testing.assert_array_equal(np.random.rand(3), expected_random[2])
        restored.train_step(batch)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, other.state_dict()[name], rtol=0, atol=0)
        torch.save(
            dict(step=restored.step, next_iteration=state.progress["next_iteration"], rng=state.rng_state["torch"]),
            outdir / f"resume-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("collision", [False, True])
def test_two_rank_restore_and_publication_failure(training_context, train_batch, tmp_path, collision):
    context, cfg = training_context
    checkpoint = context.paths.checkpoints / "iteration-00000001-step-00000001.pt"
    if collision:
        checkpoint.write_bytes(b"existing checkpoint must not be overwritten")
    torch.multiprocessing.spawn(
        _rank_resume,
        args=(f"file://{tmp_path / 'resume-gloo'}", context, cfg, train_batch, tmp_path, collision),
        nprocs=2,
        join=True,
    )
    results = [torch.load(tmp_path / f"resume-rank-{r}.pt", weights_only=True) for r in range(2)]
    if collision:
        assert all("FileExistsError" in result["error"] for result in results)
        assert checkpoint.read_bytes() == b"existing checkpoint must not be overwritten"
        assert not checkpoint.with_suffix(".json").exists()
    else:
        assert all(result["step"] == 2 and result["next_iteration"] == 1 for result in results)
        assert not torch.equal(results[0]["rng"], results[1]["rng"])
