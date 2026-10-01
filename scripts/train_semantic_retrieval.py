#!/usr/bin/env python3
"""Thin Scheme A entry point for the new retrieval system."""

import argparse
import os
import signal
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from semantic_retrieval.config import load_system_config
from semantic_retrieval.data.collate import collate_episodes
from semantic_retrieval.data.episodes import file_sha256, read_episodes
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, save_checkpoint
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.trainer import Trainer, build_optimizer


def main(argv=None):
    configure_fp32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--source-sha")
    args = parser.parse_args(argv)
    cfg = load_system_config(args.config)
    for name in ("run_id", "manifest_sha256", "source_sha"):
        if getattr(args, name) is not None and getattr(args, name) != getattr(cfg.runtime, name):
            raise ValueError("CLI identity differs from configuration")
    if cfg.data is None or cfg.data.split != "train":
        raise ValueError("training requires reserved train episodes")
    root = Path(__file__).resolve().parents[1]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    context = prepare_run(cfg.runtime, repo_root=root, system_config=cfg)

    def interrupted(signum, frame):
        raise InterruptedError(f"worker signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if world > 1:
        dist.barrier()
    if rank == 0:
        start_run(context)
    if world > 1:
        dist.barrier()
    try:
        if file_sha256(cfg.data.episode_manifest) != cfg.data.episode_sha256:
            raise ValueError("episode content changed")
        episodes = read_episodes(
            cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split=cfg.data.split
        )
        episodes = episodes[rank::world]
        if not episodes:
            raise ValueError("each rank requires at least one episode")
        batch = collate_episodes(episodes)
        inp = batch.inputs
        batch = replace(
            batch,
            inputs=replace(
                inp,
                input_ids=inp.input_ids.to(device),
                text_mask=inp.text_mask.to(device),
                video=inp.video.to(device),
                frame_mask=inp.frame_mask.to(device),
            ),
            targets=replace(
                batch.targets,
                relevance=batch.targets.relevance.to(device),
                candidate_presence=batch.targets.candidate_presence.to(device),
                group_pairs=batch.targets.group_pairs.to(device) if batch.targets.group_pairs is not None else None,
                group_labels=batch.targets.group_labels.to(device) if batch.targets.group_labels is not None else None,
            ),
            pair_indices=batch.pair_indices.to(device),
            query_offsets=batch.query_offsets.to(device),
        )
        torch.manual_seed(cfg.train.seed)
        model = SemanticRetrievalModel(cfg.model).to(device)
        opt, scheduler = build_optimizer(model, cfg.train)
        if world > 1:
            model = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True
            )
        worker = Trainer(model, opt, scheduler, context)
        next_iteration = 0
        if cfg.resume is not None:
            state = load_checkpoint(cfg.resume.checkpoint_path, worker.raw, expected_config_sha256=config_sha256(cfg))
            if state.progress is None or state.progress["next_iteration"] > cfg.train.scheduler_steps:
                raise ValueError("resume requires a checkpoint with compatible loop progress")
            worker.restore(state)
            next_iteration = state.progress["next_iteration"]
            metrics = state.progress["metrics"]
        last_checkpoint_iteration = None
        for iteration in range(next_iteration, cfg.train.scheduler_steps):
            metrics = worker.train_step(batch)
            next_iteration = iteration + 1
            if next_iteration % cfg.train.checkpoint_interval == 0:
                save_checkpoint(
                    context,
                    worker.raw,
                    opt,
                    scheduler,
                    cfg,
                    worker.step,
                    progress=dict(next_iteration=next_iteration, metrics=metrics),
                )
                last_checkpoint_iteration = next_iteration
        if last_checkpoint_iteration != next_iteration:
            save_checkpoint(
                context,
                worker.raw,
                opt,
                scheduler,
                cfg,
                worker.step,
                progress=dict(next_iteration=next_iteration, metrics=metrics),
            )
        if world > 1:
            dist.barrier()
        if rank == 0:
            publish_metrics(context, metrics)
            complete_run(context)
    except BaseException:
        if rank == 0:
            fail_run(context, exit_code=1)
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
