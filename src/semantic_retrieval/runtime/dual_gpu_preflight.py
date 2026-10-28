"""Two-rank software probes; never a replacement for the four-rank science gate."""

import gc
import json
import math
import random
import time
from dataclasses import replace

import numpy as np
import torch
import torch.distributed as dist

from scripts.train_semantic_retrieval import batch_to_device, collate_episodes
from semantic_retrieval.data.episodes import read_episodes
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, save_checkpoint
from semantic_retrieval.runtime.pilot import PilotSampler, controlled_panel
from semantic_retrieval.runtime.pilot_candidates import _bound, write_json
from semantic_retrieval.runtime.pilot_qualification import _all, _gradient_probes, _same_ranks, _wrap
from semantic_retrieval.runtime.trainer import Trainer, build_optimizer

PROTOCOL = "dual-gpu-preflight-v1"


def validate_report(report, *, require_cuda):
    if (
        report.get("protocol") != PROTOCOL
        or report.get("software_only") is not True
        or report.get("world_size") != 2
        or report.get("optimizer") != "AdamW"
        or report.get("passed") is not True
        or sorted(r["rank"] for r in report.get("rank_records", [])) != [0, 1]
    ):
        raise ValueError("two-rank software report identity differs")
    if require_cuda and report.get("device") != "cuda":
        raise ValueError("CUDA evidence is required")
    tolerance = 1e-6 if report["device"] == "cuda" else 0.0
    resume = report["resume"]
    if any(
        not math.isfinite(resume[k]) or resume[k] > tolerance or resume[k] < 0
        for k in ("parameter_max_abs_delta", "optimizer_max_abs_delta")
    ) or any(
        resume.get(k) is not True
        for k in ("sampler_restored", "rng_restored", "scheduler_restored", "rank_consistency")
    ):
        raise ValueError("checkpoint restoration failed")
    for arm in ("A2", "A3", "A4"):
        probe = report["gradient_probes"][arm]
        norms = [probe["matching_gradient_norm"], probe["presence_gradient_norm"]]
        if (
            any(not math.isfinite(v) or v < 0 for v in norms)
            or norms[0] <= 0
            or (norms[1] > 0) != (arm == "A3")
            or probe.get("update_performed") is not True
        ):
            raise ValueError(f"{arm} gradient boundary failed")
    return report


def _draw(device):
    return (random.random(), float(np.random.random()), float(torch.rand(())), float(torch.rand((), device=device)))


def run_preflight(context, cfg, *, device):
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise ValueError("preflight requires exactly two distributed ranks")
    rank = dist.get_rank()
    cfg = replace(cfg, runtime=context.config)
    if cfg.train.pilot is not None or context.system_config != cfg:
        raise ValueError("preflight requires its exact bound standalone software configuration")
    _bound(context, dict(path=str(cfg.data.episode_manifest), sha256=cfg.data.episode_sha256))
    # Orders are explicit bound inputs from the producer, not regenerated from labels.
    order_path = cfg.data.episode_manifest.parent / "candidate-orders.json"
    from semantic_retrieval.data.episodes import file_sha256

    _bound(context, dict(path=str(order_path), sha256=file_sha256(order_path)))
    orders = json.loads(order_path.read_text())
    episodes = read_episodes(cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split="train")
    lookup = {e.query_id: e for e in episodes}
    sampler = PilotSampler(tuple(lookup), seed=0)

    def next_batch(s):
        global_ids, scheduled_k, scheduled_panel = s.next_batch(rank=None, world_size=4)
        ids = global_ids[rank * 8 : (rank + 1) * 8]
        if len(set(q for part in _all(ids) for q in part)) != 16:
            raise ValueError("two-rank global16/local8 queries are not unique")
        # Fixed K4/present minimizes resources; full K/panel qualification stays in Q0.
        selected = [controlled_panel(lookup[q], orders[q], 4, "present") for q in ids]
        return batch_to_device(collate_episodes(selected), device), (ids, scheduled_k, scheduled_panel)

    raw = SemanticRetrievalModel(cfg.model).to(device)
    optimizer, scheduler = build_optimizer(raw, cfg.train)
    wrapper = _wrap(raw, device)
    trainer = Trainer(wrapper, optimizer, scheduler, context)
    started = time.monotonic()
    batch, _ = next_batch(sampler)
    metrics = trainer.train_step(batch)
    _same_ranks(raw)
    random.seed(1000 + rank)
    np.random.seed(2000 + rank)
    torch.manual_seed(3000 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(4000 + rank)
    path = save_checkpoint(
        context,
        raw,
        optimizer,
        scheduler,
        cfg,
        trainer.step,
        progress=dict(next_iteration=1, metrics=metrics, sampler=sampler.state_dict()),
    )
    draw = _draw(device)
    batch, identity = next_batch(sampler)
    trainer.train_step(batch)
    continuous = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if p.requires_grad}
    moments = {
        i: {k: v.detach().cpu().clone() for k, v in row.items() if torch.is_tensor(v)}
        for i, row in optimizer.state_dict()["state"].items()
    }
    final_groups = optimizer.state_dict()["param_groups"]
    final_scheduler = scheduler.state_dict()
    del trainer, wrapper, optimizer, scheduler, raw
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw = SemanticRetrievalModel(cfg.model).to(device)
    optimizer, scheduler = build_optimizer(raw, cfg.train)
    wrapper = _wrap(raw, device)
    trainer = Trainer(wrapper, optimizer, scheduler, context)
    state = load_checkpoint(path, raw, expected_config_sha256=config_sha256(cfg))
    trainer.restore(state)
    rng_ok = _draw(device) == draw
    restored = PilotSampler(tuple(lookup), seed=0)
    restored.load_state_dict(state.progress["sampler"])
    del state
    # Loading a checkpoint is pure; it does not restore RNG until Trainer.restore.
    restored_batch, restored_identity = next_batch(restored)
    sampler_ok = restored_identity == identity and restored.state_dict() == sampler.state_dict()
    trainer.train_step(restored_batch)
    delta = max(
        float((continuous[n] - p.detach().cpu()).abs().max()) for n, p in raw.named_parameters() if p.requires_grad
    )
    current = optimizer.state_dict()
    moment_delta = max(
        float((v - current["state"][i][k].detach().cpu()).abs().max())
        for i, row in moments.items()
        for k, v in row.items()
    )
    scheduler_ok = scheduler.state_dict() == final_scheduler and current["param_groups"] == final_groups
    _same_ranks(raw)
    resume = dict(
        parameter_max_abs_delta=max(_all(delta)),
        optimizer_max_abs_delta=max(_all(moment_delta)),
        sampler_restored=all(_all(sampler_ok)),
        rng_restored=all(_all(rng_ok)),
        scheduler_restored=all(_all(scheduler_ok)),
        rank_consistency=True,
        continuous_steps=2,
        checkpoint_step=1,
        replayed_steps=1,
    )
    if rank == 0:
        write_json(context.paths.provenance / "dual-resume.json", resume)
    del trainer, wrapper, optimizer, scheduler, continuous, moments, current
    probes, a3 = _gradient_probes(context, cfg, raw, restored_batch, device)
    del a3, raw
    records = _all(
        dict(
            rank=rank,
            device=str(device),
            seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            gradient_probes=probes,
        )
    )
    report = dict(
        protocol=PROTOCOL,
        software_only=True,
        scientific_gate_pass=False,
        passed=True,
        optimizer="AdamW",
        world_size=2,
        device=device.type,
        precision="fp32",
        global_batch=16,
        local_batch=8,
        K=4,
        panel="present",
        source_sha=context.manifest["git_sha"],
        gradient_probes=probes,
        resume=resume,
        rank_records=records,
    )
    # Each rank validates its actual local gradient evidence, not only rank0's measurements.
    validate_report(report, require_cuda=device.type == "cuda")
    dist.barrier()
    if rank == 0:
        write_json(context.paths.provenance / "dual-gpu-preflight.json", report)
    return report


def validate_scope(manifest, study):
    if (
        manifest.get("study") != PROTOCOL
        or manifest.get("registry_path") != f"experiments/{PROTOCOL}/registry.jsonl"
        or manifest.get("dataset") != "synthetic"
        or manifest.get("stage") != "baseline"
        or manifest.get("variant") != "gradient-resume"
        or manifest.get("test_authorized") is not False
        or manifest["environment"].get("gpu_count") != 2
        or manifest["environment"].get("precision") != "fp32"
        or manifest["environment"].get("distributed_topology") != "torchrun-2"
        or study.get("software_only") is not True
        or study.get("scientific_authorized") is not False
        or study.get("test_authorized") is not False
        or study.get("physical_indices") != [3, 4]
    ):
        raise ValueError("unregistered two-GPU software scope")


def validate_manifest(repository, registry, manifest):
    import hashlib
    import shlex
    from pathlib import Path

    import yaml

    from semantic_retrieval.config import effective_manifest_config, load_system_config
    from semantic_retrieval.runtime.pilot_candidates import file_reference
    from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence

    control = Path(repository) / "experiments" / PROTOCOL
    study_path = control / "study.yaml"
    if not study_path.is_file():
        raise ValueError("dual preflight study is missing")
    study = yaml.safe_load(study_path.read_text())
    validate_scope(manifest, study)
    if (
        Path(registry) != control / "registry.jsonl"
        or Path(manifest["artifact_path"]).parent.parent.parent.name != PROTOCOL
        or manifest["seed"] != 0
    ):
        raise ValueError("preflight namespace/seed differs")
    ref = study["template"]
    if file_reference(ref["path"]) != ref:
        raise ValueError("preflight template hash differs")
    cfg = load_system_config(Path(ref["path"]))
    if manifest["config"] != effective_manifest_config(cfg, eval_split="train"):
        raise ValueError("preflight must reserve its exact software configuration")
    if (
        cfg.train.pilot is not None
        or cfg.model.pilot_arm != "P1"
        or cfg.model.clip.fixture
        or manifest["command_sha256"]
        != hashlib.sha256(
            (
                shlex.join([arg.replace("{artifact_run_dir}", manifest["artifact_path"]) for arg in study["command"]])
                + "\n"
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("preflight capacity/command differs")
    parent = verify_phase_evidence(repository, study["source_parent"], "Q0P")["runs"][0]
    inventory = json.loads((parent["paths"].provenance / "pilot-input-inventory.json").read_text())
    expected = {r["path"]: r["sha256"] for r in inventory["inputs"]}
    expected[ref["path"]] = ref["sha256"]
    expected[str(Path(repository) / study["source_parent"]["path"])] = study["source_parent"]["sha256"]
    actual = {r["path"]: r["sha256"] for f in ("input_checkpoints", "dataset_splits") for r in manifest[f]}
    if actual != expected or manifest["parent_run_ids"] != [parent["manifest"]["run_id"]]:
        raise ValueError("preflight producer/input identities differ")
