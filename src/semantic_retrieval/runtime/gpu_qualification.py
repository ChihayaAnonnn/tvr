"""Synthetic software qualification using the real training/deployment modules."""

import json
import random
import shutil
import time
from dataclasses import asdict, replace

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.config import DataConfig, ModelConfig, PolicyConfig, SystemConfig, TrainConfig, config_payload
from semantic_retrieval.data.collate import collate_episodes
from semantic_retrieval.data.episodes import file_sha256, read_episodes
from semantic_retrieval.evaluation.calibration import CalibrationIdentity, fit_calibration
from semantic_retrieval.evaluation.episodes import evaluate_episodes
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.retrieval.cache import FeatureCache
from semantic_retrieval.retrieval.index import CanonicalIndex, IndexIdentity
from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, model_sha256, save_checkpoint
from semantic_retrieval.runtime.inference import RetrievalService, policy_sha256
from semantic_retrieval.runtime.trainer import Trainer, build_optimizer
from semantic_retrieval.types import QueryInputs, Targets, VideoFeatures


GALLERY = "synthetic-code-qualification-v1"


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(artifact_json_bytes(value))


def require_four_gpu():
    if (
        not torch.cuda.is_available() or torch.cuda.device_count() != 4
        or not dist.is_initialized() or dist.get_world_size() != 4
        or dist.get_backend() != "nccl"
    ):
        raise ValueError("qualification requires four CUDA ranks with NCCL")


def build_fixture_inputs(context, *, tiny=False):
    """Called by an archived CPU prep worker; random weights have no scientific meaning."""
    source = context.paths.provenance / "synthetic-inputs"
    source.mkdir()
    torch.manual_seed(713)
    clip = (
        ClipConfig(8, 8, 64, 1, 4, 5, 49408, 8, 1, 1, fixture=True) if tiny
        else ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    )
    weights = context.paths.arrays / "random-clip.npz"
    backbone = ClipBackbone(clip)
    with weights.open("xb") as stream:
        np.savez(stream, **{name: tensor.numpy() for name, tensor in backbone.state_dict().items()})
    del backbone
    refs = [dict(path=str(weights), sha256=file_sha256(weights))]
    videos = []
    for i in range(3):
        path = source / f"video-{i}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (32, 32))
        if not writer.isOpened():
            raise ValueError("synthetic video writer failed")
        try:
            for j in range(2):
                frame = np.full((32, 32, 3), 30 + 60 * i + 15 * j, dtype=np.uint8)
                frame[:, i * 5 : i * 5 + 5, 0] = 220
                writer.write(frame)
        finally:
            writer.release()
        videos.append(path)
        refs.append(dict(path=str(path), sha256=file_sha256(path)))
    pre = dict(max_text_length=clip.context_length, max_frames=2, image_size=clip.image_size)
    for split in ("train", "val_calibration", "val_locked"):
        rows = []
        for i in range(8 if split == "train" else 2):
            positive = i < (4 if split == "train" else 1)
            unknown = split == "train" and i % 4 == 3
            provenance = dict(source="synthetic code fixture, not benchmark annotations", reviewed=True, sha256="a" * 64)
            if not positive and not unknown:
                provenance["gallery_audit"] = dict(
                    gallery_version=GALLERY, scope="complete", reviewed=True, source="synthetic complete fixture"
                )
            rows.append(dict(
                query_id=f"{split}-q{i}", query_text="a striped video" if positive else "a moving object",
                gallery_version=GALLERY, candidate_ids=["a", "b", "c"], canonical_ids=["a", "b", "c"],
                relevance=["unknown"] * 3 if unknown else (["positive", "negative", "negative"] if positive else ["negative"] * 3),
                answer_in_candidates="unknown" if unknown else ("yes" if positive else "no"),
                answer_in_gallery="unknown" if unknown else ("yes" if positive else "no"),
                annotation_provenance=provenance, split=split,
                video_paths=[str(p) for p in videos], content_sha256=[file_sha256(p) for p in videos],
                frame_counts=[2] * 3, source_ids=["synthetic-a", "synthetic-b", "synthetic-c"],
                caption_group_id=f"{split}-caption-{i}",
            ))
        path = source / f"{split}.json"
        write_json(path, dict(schema_version=1, gallery_version=GALLERY, preprocessing=pre, episodes=rows))
        refs.append(dict(path=str(path), sha256=file_sha256(path)))
    model = ModelConfig(clip, weights, refs[0]["sha256"], clip.embed_dim, 1, 2 if tiny else 8, 2, 6 if tiny else 256)
    train = TrainConfig(
        0.0001, 0.0, 713, "linear", 4,
        ("temporal.", "coarse.", "matcher.", "answerability."),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=1.0, align=0.0, group=0.0), checkpoint_interval=1,
    )
    data = DataConfig(source / "train.json", refs[-3]["sha256"], GALLERY, "train")
    cfg = SystemConfig(context.config, model, train, PolicyConfig(1.0, 1.0, (1, 2, 3), 3), data)
    write_json(context.paths.provenance / "input-inventory.json", dict(inputs=refs, software_only=True, random_weights=True))
    write_json(context.paths.provenance / "system-template.json", config_payload(cfg))
    return cfg, refs


def _move(batch, device):
    x, y = batch.inputs, batch.targets
    return replace(batch, inputs=replace(x, **{k: getattr(x, k).to(device) for k in ("input_ids", "text_mask", "video", "frame_mask")}),
                   targets=Targets(y.relevance.to(device), y.candidate_presence.to(device)),
                   pair_indices=batch.pair_indices.to(device), query_offsets=batch.query_offsets.to(device))


def _all(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _draw(device):
    return (random.random(), float(np.random.random()), float(torch.rand(())), float(torch.rand((), device=device)))


def qualification_episodes(context, cfg, split):
    if split not in {"train", "val_calibration", "val_locked"}:
        raise ValueError("qualification test remains sealed")
    path = cfg.data.episode_manifest if split == "train" else cfg.data.episode_manifest.parent / f"{split}.json"
    refs = {r["path"]: r["sha256"] for r in context.manifest["dataset_splits"]}
    if str(path) not in refs:
        raise ValueError("qualification split must be reserved")
    if file_sha256(path) != refs[str(path)]:
        raise ValueError("qualification split hash changed")
    return read_episodes(path, gallery_version=GALLERY, split=split)


def validate_qualification_template(cfg, path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ValueError("qualification template hash differs")
    expected = json.loads(path.read_text())
    actual = config_payload(cfg)
    expected.pop("runtime")
    actual.pop("runtime")
    if actual != expected:
        raise ValueError("qualification config differs from complete core-chain template")


def qualify_chain(context, cfg, *, device):
    """The CPU test calls this same chain with tiny capacity; only the CLI grants GPU PASS."""
    started = time.monotonic()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    episodes = qualification_episodes(context, cfg, "train")
    # Reject omitted or modified deployment inputs before doing any training.
    qualification_episodes(context, cfg, "val_calibration")
    qualification_episodes(context, cfg, "val_locked")
    batch = _move(collate_episodes(episodes[rank::world]), device)
    torch.manual_seed(cfg.train.seed)
    raw = SemanticRetrievalModel(cfg.model).to(device)
    opt, scheduler = build_optimizer(raw, cfg.train)
    model = DistributedDataParallel(raw, device_ids=[device.index] if device.type == "cuda" else None,
                                    find_unused_parameters=True) if world > 1 else raw
    worker = Trainer(model, opt, scheduler, context)
    before = {n: p.detach().clone() for n, p in raw.named_parameters() if p.requires_grad}
    metrics = worker.train_step(batch)
    local_counts = dict(
        coarse=int(((batch.targets.relevance == 1).any(1) & (batch.targets.relevance == 0).any(1)).sum()),
        match=int(((batch.targets.relevance == 1).any(1) & (batch.targets.relevance == 0).any(1)).sum()),
        pair=int((batch.targets.relevance[batch.pair_indices[:, 0], batch.pair_indices[:, 1]] >= 0).sum()),
        exist=int((batch.targets.candidate_presence >= 0).sum()),
    )
    counts = _all(local_counts)
    for key in local_counts:
        if metrics["valid_" + key + "_count"] != sum(c[key] for c in counts):
            raise ValueError("incorrect global known-label count")
    gradients = {}
    for prefix in cfg.train.trainable_prefixes:
        parameters = [p for n, p in raw.named_parameters() if n.startswith(prefix)]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("nonfinite gradient")
        norm = sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** 0.5
        if norm <= 0 or not any(not torch.equal(before[n], p) for n, p in raw.named_parameters() if n.startswith(prefix)):
            raise ValueError(f"missing gradient/update in {prefix}")
        gradients[prefix] = norm
    if any(p.grad is not None for p in raw.clip.parameters()):
        raise ValueError("unexpected update to frozen CLIP")
    if len(set(_all(model_sha256(raw)))) != 1 or any(g != gradients for g in _all(gradients)):
        raise ValueError("rank parameters or gradients differ")
    # Distinct RNG streams ensure this check is not accidentally reading rank zero's state.
    random.seed(1000 + rank)
    np.random.seed(2000 + rank)
    torch.manual_seed(3000 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(4000 + rank)
    path = save_checkpoint(context, raw, opt, scheduler, cfg, worker.step,
                           progress=dict(next_iteration=1, metrics=metrics))
    expected_draw = _draw(device)
    saved_hash = model_sha256(raw)
    saved_scheduler = scheduler.state_dict()
    unknown = replace(batch, targets=Targets(torch.full_like(batch.targets.relevance, -1),
                                            torch.full_like(batch.targets.candidate_presence, -1)))
    skipped = worker.train_step(unknown)
    if skipped["stepped"] or worker.step != 1 or saved_hash != model_sha256(raw) or scheduler.state_dict() != saved_scheduler:
        raise ValueError("unknown batch changed training state")
    worker.train_step(batch)
    continuous = {n: p.detach().clone() for n, p in raw.named_parameters() if p.requires_grad}
    moments = {p: {k: v.detach().clone() for k, v in values.items() if isinstance(v, torch.Tensor)}
               for p, values in opt.state_dict()["state"].items()}
    continuous_scheduler = scheduler.state_dict()
    del model, worker, opt, scheduler, raw, before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restored = SemanticRetrievalModel(cfg.model).to(device)
    opt, scheduler = build_optimizer(restored, cfg.train)
    wrapper = DistributedDataParallel(restored, device_ids=[device.index] if device.type == "cuda" else None,
                                      find_unused_parameters=True) if world > 1 else restored
    worker = Trainer(wrapper, opt, scheduler, context)
    state = load_checkpoint(path, restored, expected_config_sha256=config_sha256(cfg))
    worker.restore(state)
    if state.progress["next_iteration"] != 1 or worker.step != 1 or _draw(device) != expected_draw:
        raise ValueError("checkpoint cursor or rank RNG restore differs")
    worker.train_step(batch)
    delta = max(float((continuous[n] - p).abs().max()) for n, p in restored.named_parameters() if p.requires_grad)
    # CPU is bitwise; NCCL may choose different buckets after constructing a new reducer.
    tolerance = 1e-6 if device.type == "cuda" else 0.0
    if delta > tolerance or len(set(_all(model_sha256(restored)))) != 1:
        raise ValueError(f"resumed parameters differ: {delta}")
    current_moments = opt.state_dict()["state"]
    moment_delta = max(float((value - current_moments[p][k]).abs().max())
                       for p, values in moments.items() for k, value in values.items())
    if worker.step != 2 or scheduler.state_dict() != continuous_scheduler or moment_delta > tolerance:
        raise ValueError("resume lost optimizer moments/scheduler state")
    wrong_config = False
    try:
        load_checkpoint(path, restored, expected_config_sha256="0" * 64)
    except ValueError:
        wrong_config = True
    if not wrong_config:
        raise ValueError("incorrect checkpoint config accepted")
    report = dict(world_size=world, gradients=gradients, training_metrics=metrics, unknown_stepped=skipped["stepped"],
                  resume_parameter_max_abs_delta=max(_all(delta)), resume_parameter_tolerance=tolerance,
                  resume_optimizer_max_abs_delta=max(_all(moment_delta)),
                  rng_restored=1, rank_consistent=1, config_mismatch_rejected=1, checks_passed=10)
    if rank == 0:
        corrupt = context.paths.checkpoints / "planned-corrupt-copy.pt"
        shutil.copyfile(path, corrupt)
        shutil.copyfile(path.with_suffix(".json"), corrupt.with_suffix(".json"))
        with corrupt.open("ab") as stream:
            stream.write(b"planned corruption")
        try:
            load_checkpoint(corrupt, restored, expected_config_sha256=config_sha256(cfg))
        except ValueError:
            report["tamper_rejected"] = 1
        else:
            raise ValueError("corrupt checkpoint accepted")
        # Use a separate inference instance so deployment never modifies the DDP wrapper.
        del wrapper, worker, opt, scheduler
        restored.eval().requires_grad_(False)
        report.update(_deployment(context, cfg, restored, device))
        report["checks_passed"] += 5
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(context.paths.provenance / "qualification-report.json", report)
    if dist.is_initialized():
        dist.barrier()
    return report


def _deployment(context, cfg, model, device):
    source = cfg.data.episode_manifest.parent
    cal_episodes = qualification_episodes(context, cfg, "val_calibration")
    batch = _move(collate_episodes(cal_episodes), device)
    with torch.no_grad():
        f, pairs, presence, _, _, _ = model(batch.inputs, batch.pair_indices, batch.query_offsets)
        video = model.encode_gallery(batch.inputs.video, batch.inputs.frame_mask)
        embeddings = model.project_gallery(video)
    pre_hash = file_sha256(source / "val_calibration.json")
    ident = IndexIdentity(model_sha256(model), GALLERY, pre_hash)
    identity = CalibrationIdentity(ident.model_sha256, pre_hash, GALLERY, policy_sha256(cfg.policy), pre_hash)
    calibration = fit_calibration(pairs.logits, batch.targets.relevance[batch.pair_indices[:, 0], batch.pair_indices[:, 1]],
                                  presence, batch.targets.candidate_presence, split="val_calibration", identity=identity)
    write_json(context.paths.provenance / "calibration.json", asdict(calibration))
    for name, value in (("calibration_pair_logits", pairs.logits), ("calibration_presence_logits", presence)):
        context.array_store.write(name, value.cpu().numpy())
    cache = FeatureCache(context)
    hashes = dict(zip(cal_episodes[0].canonical_ids, cal_episodes[0].content_sha256))
    for j, canonical in enumerate(batch.canonical_ids):
        cache.write(canonical, VideoFeatures(video.frame_tokens[j:j+1], video.pooled_video[j:j+1], video.frame_mask[j:j+1]),
                    identity=ident, content_sha256=hashes[canonical])
    index = CanonicalIndex(batch.canonical_ids, embeddings, ident)
    service = RetrievalService(model, index, cache, calibration, cfg.policy, content_sha256=hashes)
    query = QueryInputs(batch.inputs.input_ids[:1], batch.inputs.text_mask[:1])
    decision = service.retrieve(query, budget=3)
    if decision.action != "abstain" or service.last_prediction.budget_used != 3 or service.last_prediction.attempted_k != (1, 2, 3):
        raise ValueError("bounded expansion failed")
    if service.retrieve(query, budget=0).action != "abstain":
        raise ValueError("zero budget did not terminate")
    # Fixed alternate policy exercises a successful return, not threshold tuning.
    return_policy = PolicyConfig(0.0, 0.0, (3,), 3)
    return_calibration = replace(calibration, identity=replace(calibration.identity, candidate_policy_sha256=policy_sha256(return_policy)))
    returning = RetrievalService(model, index, cache, return_calibration, return_policy, content_sha256=hashes)
    if set(returning.retrieve(query, budget=3).answer_ids) != set(batch.canonical_ids):
        raise ValueError("positive return path failed")
    locked = qualification_episodes(context, cfg, "val_locked")
    locked_batch = _move(collate_episodes(locked), device)
    predictions = []
    for i, episode in enumerate(locked):
        service.retrieve(QueryInputs(locked_batch.inputs.input_ids[i:i+1], locked_batch.inputs.text_mask[i:i+1]), budget=3)
        predictions.append(replace(service.last_prediction, query_id=episode.query_id))
    evaluated = evaluate_episodes(predictions, locked, split="val_locked")
    if evaluated["episode_count"] != 2 or evaluated["false_accept"]["denominator"] != 1:
        raise ValueError("locked evaluation denominator differs")
    write_json(context.paths.provenance / "locked-metrics.json", evaluated)
    return dict(expansion_budget_used=3, calibration_fitted=1, locked_evaluated=1, zero_budget_terminated=1, return_set_checked=1)
