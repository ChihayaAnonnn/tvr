"""New checkpoint schema: hashed binary states, no embedded JSON tensors."""

import hashlib
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from experiment_artifacts import artifact_json_bytes, load_validated_run_lifecycle
from semantic_retrieval.config import config_payload


def config_sha256(config):
    payload = config_payload(config)
    payload.pop("runtime")
    return hashlib.sha256(artifact_json_bytes(payload)).hexdigest()


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        x = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(x.shape), str(x.dtype))).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    ns = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=(ns[0], torch.from_numpy(ns[1].astype(np.int64)), ns[2], ns[3], ns[4]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], ns[1].numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class TrainingState:
    step: int
    optimizer_state: dict
    scheduler_state: dict
    rng_state: dict


def assert_running(context):
    manifest, status, _ = load_validated_run_lifecycle(context.paths)
    if status["state"] != "running" or manifest != context.manifest:
        raise ValueError("worker must have exact running identity")


def save_checkpoint(context, model, optimizer, scheduler, config, step):
    assert_running(context)
    if context.system_config != config or type(step) is not int or step < 0:
        raise ValueError("checkpoint config/step mismatch")
    path = context.paths.checkpoints / f"step-{step:08d}.pt"
    rng = [rng_state()]
    rank = dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized():
        rng = [None] * dist.get_world_size()
        dist.all_gather_object(rng, rng_state())
    if rank != 0:
        return path
    payload = dict(
        schema_version=1,
        config_sha256=config_sha256(config),
        step=step,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        rng=rng,
    )
    with path.open("xb") as handle:
        torch.save(payload, handle)
    meta = dict(
        schema_version=1,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config_sha256=payload["config_sha256"],
        model_sha256=model_sha256(model),
    )
    with path.with_suffix(".json").open("xb") as handle:
        handle.write(artifact_json_bytes(meta))
    return path


def load_checkpoint(path, model, *, expected_config_sha256):
    meta = json.loads(path.with_suffix(".json").read_text())
    if (
        set(meta) != {"schema_version", "sha256", "config_sha256", "model_sha256"}
        or meta["schema_version"] != 1
        or meta["config_sha256"] != expected_config_sha256
        or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]
    ):
        raise ValueError("checkpoint hash/config mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(payload) != {"schema_version", "config_sha256", "step", "model", "optimizer", "scheduler", "rng"}
        or payload["schema_version"] != 1
        or payload["config_sha256"] != expected_config_sha256
        or type(payload["step"]) is not int
        or payload["step"] < 0
    ):
        raise ValueError("checkpoint state schema mismatch")
    current = model.state_dict()
    saved = payload["model"]
    if set(current) != set(saved) or any(
        not isinstance(saved[k], torch.Tensor)
        or current[k].shape != saved[k].shape
        or current[k].dtype != saved[k].dtype
        or not torch.isfinite(saved[k]).all()
        for k in current
    ):
        raise ValueError("checkpoint model state mismatch")
    if (
        not isinstance(payload["rng"], list)
        or len(payload["rng"]) != (dist.get_world_size() if dist.is_initialized() else 1)
        or any(not isinstance(rng, dict) or set(rng) != {"torch", "python", "numpy", "cuda"} for rng in payload["rng"])
        or not isinstance(payload["optimizer"], dict)
        or not isinstance(payload["scheduler"], dict)
    ):
        raise ValueError("checkpoint training state mismatch")
    # Validate the content identity before mutating the caller's model.
    digest = hashlib.sha256()
    for name, tensor in sorted(saved.items()):
        x = tensor.contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(x.shape), str(x.dtype))).encode())
        digest.update(x.numpy().tobytes())
    if digest.hexdigest() != meta["model_sha256"]:
        raise ValueError("checkpoint model identity mismatch")
    model.load_state_dict(saved, strict=True)
    return TrainingState(
        payload["step"],
        payload["optimizer"],
        payload["scheduler"],
        payload["rng"][dist.get_rank() if dist.is_initialized() else 0],
    )
