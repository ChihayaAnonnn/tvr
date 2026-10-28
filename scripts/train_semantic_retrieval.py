#!/usr/bin/env python3
"""Thin Scheme A entry point for the new retrieval system."""

import argparse
import json
import os
import signal
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from semantic_retrieval.config import load_system_config
from semantic_retrieval.data.collate import collate_episodes
from semantic_retrieval.data.episodes import file_sha256, read_bound_episodes
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, save_checkpoint
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.pilot import (
    PilotSampler,
    controlled_panel,
    initialize_pilot_model,
    probe_matching_presence_gradients,
    read_candidate_orders,
    validate_pilot_config,
)
from semantic_retrieval.runtime.trainer import Trainer, build_optimizer


def batch_to_device(batch, device):
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
    return batch


def training_progress(next_iteration, metrics, sampler):
    progress = dict(next_iteration=next_iteration, metrics=metrics)
    if sampler is not None:
        progress["sampler"] = sampler.state_dict()
    return progress


def next_pilot_batch(episode_lookup, orders, sampler, *, rank, world_size, device):
    query_ids, k, panel = sampler.next_batch(rank=rank, world_size=world_size)
    selected = [controlled_panel(episode_lookup[q], orders[q], k, panel) for q in query_ids]
    return batch_to_device(collate_episodes(selected), device), k, panel


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
    started = time.perf_counter()
    try:
        if file_sha256(cfg.data.episode_manifest) != cfg.data.episode_sha256:
            raise ValueError("episode content changed")
        episodes = read_bound_episodes(cfg.data)
        validate_pilot_config(cfg, world_size=world)
        sampler = None
        orders = None
        if cfg.train.pilot is not None:
            if any(getattr(e, "relevance_space", None) != "benchmark_annotation" for e in episodes):
                raise ValueError("pilot requires scoped benchmark annotation episodes")
            if getattr(cfg.data, "relevance_space", None) != "benchmark_annotation" or any(
                e.label_protocol != getattr(cfg.data, "label_protocol", None) for e in episodes
            ):
                raise ValueError("pilot episode label scope/protocol differs from reserved data config")
            if not cfg.model.clip.fixture and any(
                (e.preprocessing.max_text_length, e.preprocessing.max_frames, e.preprocessing.image_size)
                != (32, 8, 224)
                for e in episodes
            ):
                raise ValueError("pilot requires registered text/frame/image preprocessing")
            if any(len(getattr(e, "benchmark_positive_ids", ())) != 1 for e in episodes):
                raise ValueError("pilot requires one official canonical positive per query")
            orders = read_candidate_orders(cfg.train.pilot, episodes)
            sampler = PilotSampler(tuple(e.query_id for e in episodes), seed=cfg.train.pilot.sampler_seed)
            episode_lookup = {e.query_id: e for e in episodes}
            model, parent_sampler = initialize_pilot_model(cfg)
            if parent_sampler is not None:
                sampler.load_state_dict(parent_sampler)
        else:
            episodes = episodes[rank::world]
            if not episodes:
                raise ValueError("each rank requires at least one episode")
            batch = batch_to_device(collate_episodes(episodes), device)
            torch.manual_seed(cfg.train.seed)
            model = SemanticRetrievalModel(cfg.model)
        model = model.to(device)
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
            if sampler is not None:
                if "sampler" not in state.progress:
                    raise ValueError("pilot resume requires sampler checkpoint state")
                sampler.load_state_dict(state.progress["sampler"])
                expected_iteration = next_iteration + (120 if cfg.train.pilot.arm in {"A1", "A2", "A3", "A4"} else 0)
                if sampler.iteration != expected_iteration:
                    raise ValueError("pilot sampler and resumed training phase disagree")
        last_checkpoint_iteration = None
        for iteration in range(next_iteration, cfg.train.scheduler_steps):
            if sampler is not None:
                batch, k, panel = next_pilot_batch(
                    episode_lookup, orders, sampler, rank=rank, world_size=world, device=device
                )
                if (iteration + 1) % 20 == 0:
                    probes = probe_matching_presence_gradients(worker.raw, batch, cfg.train.loss_weights)
                    if rank == 0:
                        with (context.paths.logs / "pilot-gradients.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(iteration=iteration + 1, **probes)) + "\n")
            metrics = worker.train_step(batch)
            if sampler is not None and metrics["stepped"] != 1:
                raise ValueError("registered pilot update has no known-label optimizer step")
            next_iteration = iteration + 1
            if sampler is not None and rank == 0:
                with (context.paths.logs / "pilot-losses.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(dict(iteration=next_iteration, k=k, panel=panel, **metrics, **worker.last_timing))
                        + "\n"
                    )
            if next_iteration % cfg.train.checkpoint_interval == 0:
                save_checkpoint(
                    context,
                    worker.raw,
                    opt,
                    scheduler,
                    cfg,
                    worker.step,
                    progress=training_progress(next_iteration, metrics, sampler),
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
                progress=training_progress(next_iteration, metrics, sampler),
            )
        if world > 1:
            dist.barrier()
        if rank == 0:
            summary = dict(metrics)
            if sampler is not None:
                summary["pilot"] = dict(
                    arm=cfg.train.pilot.arm,
                    unique_queries_seen=len(sampler.seen),
                    query_inventory=len(sampler.query_ids),
                    global_batch=16,
                    rank_batch=4,
                    sampler_iteration=sampler.iteration,
                    parameter_count=sum(p.numel() for p in worker.raw.parameters()),
                    trainable_parameter_count=sum(p.numel() for p in worker.raw.parameters() if p.requires_grad),
                    presence_parameter_count=sum(p.numel() for p in worker.raw.answerability.parameters()),
                    elapsed_seconds=time.perf_counter() - started,
                    gpu_hours=(time.perf_counter() - started) * world / 3600 if device.type == "cuda" else 0.0,
                )
            publish_metrics(context, summary)
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
