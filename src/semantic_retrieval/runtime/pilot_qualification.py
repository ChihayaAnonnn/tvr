"""New synthetic v2 C1/C2/C3 qualification; CPU fixtures never grant GPU PASS."""

import gc
import hashlib
import os
import random
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from experiment_artifacts import artifact_json_bytes, strict_json_loads
from scripts.train_semantic_retrieval import next_pilot_batch
from semantic_retrieval.config import (
    DataConfig,
    ModelConfig,
    PilotTrainConfig,
    PolicyConfig,
    SystemConfig,
    TrainConfig,
    config_payload,
)
from semantic_retrieval.data.episodes import file_sha256, read_episodes
from semantic_retrieval.data.gallery import preprocessing_sha256
from semantic_retrieval.data.tokenizer import ClipTokenizer
from semantic_retrieval.evaluation.pilot_analysis import (
    calibrated_probabilities,
    export_model_scores,
    fit_pilot_calibration,
    stream_pilot_gallery,
)
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.objectives.retrieval import compute_objective
from semantic_retrieval.runtime.checkpoint import config_sha256, load_checkpoint, model_sha256, save_checkpoint
from semantic_retrieval.runtime.inference import policy_sha256
from semantic_retrieval.runtime.pilot import (
    PilotSampler,
    controlled_panel,
    pilot_trainable_prefixes,
    probe_matching_presence_gradients,
    read_candidate_orders,
    validate_pilot_config,
)
from semantic_retrieval.runtime.pilot_candidates import KS, _bound, file_reference, panel_groups, write_json
from semantic_retrieval.runtime.pilot_protocol import (
    PHYSICAL_INDICES,
    Q0_CHECKS,
    Q0_PROTOCOL,
)
from semantic_retrieval.runtime.trainer import Trainer, build_optimizer, global_coarse_relations
from semantic_retrieval.types import QueryInputs, Targets

GALLERY = "synthetic-pilot-v2-train"


def build_pilot_fixture_inputs(context, *, tiny=False):
    """Archived Q0P producer. Random weights/raw videos are software evidence only."""
    source = context.paths.provenance / "synthetic-pilot-v2"
    source.mkdir()
    torch.manual_seed(713)
    clip = (
        ClipConfig(8, 8, 64, 1, 4, 5, 49408, 8, 1, 1, fixture=True)
        if tiny
        else ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    )
    weights = context.paths.arrays / "random-pilot-clip.npz"
    raw_clip = ClipBackbone(clip)
    with weights.open("xb") as stream:
        np.savez(stream, **{k: v.numpy() for k, v in raw_clip.state_dict().items()})
    del raw_clip
    refs, videos = [file_reference(weights)], []
    ids = tuple(f"synthetic-c{i:03}" for i in range(129))
    for i, canonical in enumerate(ids):
        path = source / f"{canonical}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 32))
        if not writer.isOpened():
            raise ValueError("synthetic raw video writer failed")
        try:
            for frame in range(8):
                image = np.full((32, 32, 3), (i * 7 + frame * 3) % 240, dtype=np.uint8)
                image[:, (i % 24) : (i % 24) + 8, i % 3] = 255
                image[(i // 24) * 4 : (i // 24) * 4 + 4, :, (i + 1) % 3] = (i * 13 + frame) % 255
                writer.write(image)
        finally:
            writer.release()
        refs.append(file_reference(path))
        videos.append(
            dict(
                canonical_id=canonical,
                alias_id=f"synthetic-v{i:03}",
                video_path=str(path),
                content_sha256=refs[-1]["sha256"],
                frame_count=8,
            )
        )
    if len({v["content_sha256"] for v in videos}) != 129:
        raise ValueError("synthetic canonical videos must have unique bytes")
    relationships = {
        f"{split}-q{i:02}": [f"synthetic-v{i:03}"]
        for split, n in (("train", 16), ("val_calibration", 2), ("val_locked", 2))
        for i in range(n)
    }
    mappings = {v["alias_id"]: v["canonical_id"] for v in videos}
    files = {}
    for name, payload in dict(
        relations=relationships,
        id_map=mappings,
        annotation=dict(scope="synthetic_software_only", relationships=relationships),
        source_report=dict(scope="synthetic_software_only", real_benchmark_used=False),
        allowlist=dict(official_ids=sorted(mappings)),
        source_annotation=dict(scope="generated_synthetic"),
    ).items():
        path = source / f"{name}.json"
        write_json(path, payload)
        files[name] = file_reference(path)
        refs.append(files[name])
    protocol = dict(
        protocol_id=Q0_PROTOCOL,
        **{
            key: files[name]["sha256"]
            for key, name in (
                ("annotation_sha256", "annotation"),
                ("relation_sha256", "relations"),
                ("id_map_sha256", "id_map"),
                ("source_report_sha256", "source_report"),
                ("allowlist_sha256", "allowlist"),
                ("source_annotation_sha256", "source_annotation"),
            )
        },
    )
    pre = dict(max_text_length=min(32, clip.context_length), max_frames=8, image_size=clip.image_size)
    manifests = {}
    for split, n in (("train", 16), ("val_calibration", 2), ("val_locked", 2)):
        version = GALLERY if split == "train" else f"synthetic-pilot-v2-{split}"
        rows = []
        for i in range(n):
            rows.append(
                dict(
                    query_id=f"{split}-q{i:02}",
                    query_text=f"a moving striped object {i}",
                    gallery_version=version,
                    canonical_ids=list(ids),
                    candidate_ids=[v["alias_id"] for v in videos],
                    relevance=["positive" if j == i else "negative" for j in range(129)],
                    answer_in_candidates="yes",
                    answer_in_gallery="yes",
                    split=split,
                    annotation_provenance=dict(
                        source="benchmark_annotation", sha256=protocol["annotation_sha256"], protocol_id=Q0_PROTOCOL
                    ),
                    video_paths=[v["video_path"] for v in videos],
                    content_sha256=[v["content_sha256"] for v in videos],
                    frame_counts=[8] * 129,
                    source_ids=[v["alias_id"] for v in videos],
                    caption_group_id=f"synthetic-caption-{split}-{i}",
                    query_source_ids=[f"synthetic-v{i:03}"],
                    query_component_id=hashlib.sha256(f"synthetic-component-{i}".encode()).hexdigest(),
                )
            )
        episode_path, gallery_path = source / f"{split}.json", source / f"{split}-gallery.json"
        write_json(
            episode_path,
            dict(
                schema_version=2,
                gallery_version=version,
                preprocessing=pre,
                episodes=rows,
                relevance_space="benchmark_annotation",
                label_protocol=protocol,
                label_references={k: files[k] for k in ("relations", "id_map")},
            ),
        )
        write_json(gallery_path, dict(schema_version=1, gallery_version=version, preprocessing=pre, videos=videos))
        manifests[split] = dict(
            episodes=file_reference(episode_path), gallery=file_reference(gallery_path), gallery_version=version
        )
        refs.extend([manifests[split]["episodes"], manifests[split]["gallery"]])
    order_path = source / "candidate-orders.json"
    write_json(order_path, {q: list(ids) for q in relationships if q.startswith("train-")})
    refs.append(file_reference(order_path))
    model = ModelConfig(
        clip, weights, refs[0]["sha256"], clip.embed_dim, 0, 2 if tiny else 8, 8, 6 if tiny else 256, pilot_arm="P1"
    )
    train = TrainConfig(
        1e-4,
        0.01,
        0,
        "constant",
        120,
        pilot_trainable_prefixes("P1"),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=0.0, align=0.0, group=0.0),
        checkpoint_interval=20,
        pilot=PilotTrainConfig("P1", 0, order_path, file_sha256(order_path)),
    )
    data = DataConfig(
        Path(manifests["train"]["episodes"]["path"]),
        manifests["train"]["episodes"]["sha256"],
        GALLERY,
        "train",
        "benchmark_annotation",
        protocol,
    )
    cfg = SystemConfig(context.config, model, train, PolicyConfig(0.5, 0.5, KS, 128), data)
    template_path = context.paths.provenance / "pilot-system-template.json"
    write_json(template_path, config_payload(cfg))
    refs.append(file_reference(template_path))
    write_json(
        context.paths.provenance / "pilot-input-inventory.json",
        dict(
            schema_version=1,
            protocol=Q0_PROTOCOL,
            inputs=refs,
            manifests=manifests,
            software_only=True,
            random_weights=True,
            shared_synthetic_content_across_diagnostic_galleries=True,
        ),
    )
    return cfg, refs


def _all(value):
    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _wrap(raw, device):
    return (
        DistributedDataParallel(
            raw, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True
        )
        if dist.is_initialized()
        else raw
    )


def _draw(device):
    return (random.random(), float(np.random.random()), float(torch.rand((), device=device)))


def _same_ranks(model):
    if len(set(_all(model_sha256(model)))) != 1:
        raise ValueError("qualification model parameters differ across ranks")


def _planned_interruption():
    # This is an in-worker signal/restore probe. It never creates or reuses a cancelled run.
    old = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise InterruptedError("planned synthetic interruption")

    signal.signal(signal.SIGTERM, stop)
    try:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except InterruptedError:
            return True
        raise ValueError("planned worker interruption handler did not stop execution")
    finally:
        signal.signal(signal.SIGTERM, old)


def _training_probe(context, cfg, lookup, orders, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    sampler = PilotSampler(tuple(lookup), seed=cfg.train.pilot.sampler_seed)

    def next_batch(s):
        return next_pilot_batch(lookup, orders, s, rank=rank if world > 1 else None, world_size=4, device=device)

    raw = SemanticRetrievalModel(cfg.model).to(device)
    opt, scheduler = build_optimizer(raw, cfg.train)
    wrapper = _wrap(raw, device)
    trainer = Trainer(wrapper, opt, scheduler, context)
    start = time.monotonic()
    batch, k, panel = next_batch(sampler)
    decode_seconds = time.monotonic() - start
    if panel != "present":
        raise ValueError("registered qualification seed must start with a present panel")
    query_ids = [q for rows in _all(batch.query_ids) for q in rows]
    if len(query_ids) != 16 or len(set(query_ids)) != 16:
        raise ValueError("qualification requires unique global16/local4 pilot queries")
    start = time.monotonic()
    metrics = trainer.train_step(batch)
    update_seconds = time.monotonic() - start + decode_seconds
    _, coarse_labels = global_coarse_relations(batch, torch.zeros(len(batch.canonical_ids), 1, device=device))
    local_counts = dict(
        coarse=int(((coarse_labels == 1).any(1) & (coarse_labels == 0).any(1)).sum()),
        match=int(((batch.targets.relevance == 1).any(1) & (batch.targets.relevance == 0).any(1)).sum()),
        pair=len(batch.pair_indices),
        exist=len(batch.query_ids),
    )
    counts = _all(local_counts)
    if any(metrics["valid_" + key + "_count"] != sum(c[key] for c in counts) for key in local_counts):
        raise ValueError("pilot global known counts differ from bound official labels")
    _same_ranks(raw)
    random.seed(1000 + rank)
    np.random.seed(2000 + rank)
    torch.manual_seed(3000 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(4000 + rank)
    path = save_checkpoint(
        context,
        raw,
        opt,
        scheduler,
        cfg,
        trainer.step,
        progress=dict(next_iteration=1, metrics=metrics, sampler=sampler.state_dict()),
    )
    draw = _draw(device)
    before, saved_scheduler = model_sha256(raw), scheduler.state_dict()
    unknown = replace(
        batch,
        targets=Targets(
            torch.full_like(batch.targets.relevance, -1), torch.full_like(batch.targets.candidate_presence, -1)
        ),
        benchmark_positive_ids=None,
    )
    skipped = trainer.train_step(unknown)
    if (
        skipped["stepped"]
        or trainer.step != 1
        or model_sha256(raw) != before
        or scheduler.state_dict() != saved_scheduler
    ):
        raise ValueError("unknown pilot batch changed optimizer/model/scheduler")
    start = time.monotonic()
    next_actual, next_k, next_panel = next_batch(sampler)
    trainer.train_step(next_actual)
    second_seconds = time.monotonic() - start
    continuous = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if p.requires_grad}
    moments = {
        p: {key: value.detach().cpu().clone() for key, value in row.items() if torch.is_tensor(value)}
        for p, row in opt.state_dict()["state"].items()
    }
    final_scheduler = scheduler.state_dict()
    del trainer, wrapper, opt, scheduler, raw
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cancellation = _planned_interruption()
    raw = SemanticRetrievalModel(cfg.model).to(device)
    opt, scheduler = build_optimizer(raw, cfg.train)
    wrapper = _wrap(raw, device)
    trainer = Trainer(wrapper, opt, scheduler, context)
    state = load_checkpoint(path, raw, expected_config_sha256=config_sha256(cfg))
    trainer.restore(state)
    restored_sampler = PilotSampler(tuple(lookup), seed=cfg.train.pilot.sampler_seed)
    restored_sampler.load_state_dict(state.progress["sampler"])
    if _draw(device) != draw or trainer.step != 1:
        raise ValueError("rank RNG/update restore differs")
    restored_batch, restored_k, restored_panel = next_batch(restored_sampler)
    if (restored_batch.query_ids, restored_k, restored_panel) != (
        next_actual.query_ids,
        next_k,
        next_panel,
    ) or restored_sampler.state_dict() != sampler.state_dict():
        raise ValueError("sampler cursor/table/panels differ after checkpoint restoration")
    trainer.train_step(restored_batch)
    delta = max(
        float((continuous[n] - p.detach().cpu()).abs().max()) for n, p in raw.named_parameters() if p.requires_grad
    )
    current = opt.state_dict()["state"]
    moment_delta = max(
        float((value - current[p][key].detach().cpu()).abs().max())
        for p, row in moments.items()
        for key, value in row.items()
    )
    tolerance = 1e-6 if device.type == "cuda" else 0.0
    if delta > tolerance or moment_delta > tolerance or scheduler.state_dict() != final_scheduler:
        raise ValueError(f"pilot resumed parameters/moments differ: {delta}, {moment_delta}")
    _same_ranks(raw)
    tamper = None
    if rank == 0:
        corrupt = context.paths.checkpoints / "planned-corrupt-copy.pt"
        shutil.copyfile(path, corrupt)
        shutil.copyfile(path.with_suffix(".json"), corrupt.with_suffix(".json"))
        with corrupt.open("ab") as stream:
            stream.write(b"planned corruption")
        try:
            load_checkpoint(corrupt, raw, expected_config_sha256=config_sha256(cfg))
        except ValueError:
            tamper = True
        else:
            raise ValueError("tampered pilot checkpoint was accepted")
        try:
            load_checkpoint(path, raw, expected_config_sha256="0" * 64)
        except ValueError:
            pass
        else:
            raise ValueError("mismatched pilot configuration was accepted")
    if not any(_all(tamper)):
        raise ValueError("checkpoint hash tamper guard did not run")
    # A separate sampler proves dynamic K/present/absent panels and its balanced full120 table.
    table = PilotSampler(tuple(lookup), seed=0)
    schedule = {}
    for _ in range(120):
        qids, size, kind = table.next_batch()
        schedule[size, kind] = schedule.get((size, kind), 0) + 1
        selected = controlled_panel(lookup[qids[0]], orders[qids[0]], size, kind)
        if len(selected.canonical_ids) != size or selected.answer_in_candidates != int(kind == "present"):
            raise ValueError("dynamic pilot candidate panel or membership mismatch")
    if schedule != {(size, kind): 15 for size in (4, 8, 16, 32) for kind in ("present", "absent")}:
        raise ValueError("pilot dynamic panel table is not balanced")
    report = dict(
        training_metrics=metrics,
        resume_parameter_max_abs_delta=max(_all(delta)),
        resume_optimizer_max_abs_delta=max(_all(moment_delta)),
        unknown_stepped=skipped["stepped"],
        seconds_per_update=max(_all(max(update_seconds, second_seconds))),
        interruption_kind="planned in-worker SIGTERM; no terminal cancellation or retry",
        cancellation_guard=cancellation,
        sampled_panel_types=8,
        checkpoint=file_reference(path),
        checkpoint_metadata=file_reference(path.with_suffix(".json")),
    )
    if rank == 0:
        write_json(context.paths.provenance / "pilot-training-qualification.json", report)
    del trainer, wrapper, opt, scheduler
    return raw, batch, report


def _gradient_probes(context, cfg, p1, batch, device):
    probes, trained_a3 = {}, None
    state = {k: v.detach().cpu().clone() for k, v in p1.state_dict().items() if not k.startswith("answerability.")}
    for arm in ("A2", "A3", "A4"):
        torch.manual_seed(17)
        arm_cfg = replace(cfg.model, pilot_arm=arm)
        raw = SemanticRetrievalModel(arm_cfg).to(device)
        own = raw.state_dict()
        own.update(state)
        raw.load_state_dict(own)
        train = replace(
            cfg.train,
            trainable_prefixes=pilot_trainable_prefixes(arm),
            loss_weights={**cfg.train.loss_weights, "exist": 1.0},
        )
        opt, scheduler = build_optimizer(raw, train)
        wrapper = _wrap(raw, device)
        # Deliberately probe raw autograd after DDP construction and before real backward.
        probe = probe_matching_presence_gradients(raw, batch, train.loss_weights)
        if probe["matching_gradient_norm"] <= 0 or (probe["presence_gradient_norm"] > 0) != (arm == "A3"):
            raise ValueError(f"pilot {arm} matching/presence gradient path differs from protocol")
        before_matcher = raw.matcher.text_projection.weight.detach().clone()
        before_head = next(raw.answerability.parameters()).detach().clone()
        f, p, presence, q, v, _ = wrapper(batch.inputs, batch.pair_indices, batch.query_offsets)
        global_v, relations = global_coarse_relations(batch, v)
        terms = compute_objective(
            q @ global_v.T,
            p,
            presence,
            batch.targets,
            batch.pair_indices,
            batch.query_offsets,
            weights=train.loss_weights,
            coarse_relevance=relations,
        )
        opt.zero_grad(set_to_none=True)
        terms["total"].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in raw.parameters()):
            raise ValueError(f"pilot {arm} gradient is not finite")
        opt.step()
        scheduler.step()
        if torch.equal(before_matcher, raw.matcher.text_projection.weight) or torch.equal(
            before_head, next(raw.answerability.parameters())
        ):
            raise ValueError(f"pilot {arm} matching/presence parameters did not update")
        _same_ranks(raw)
        probes[arm] = dict(
            probe,
            head_parameter_count=sum(p.numel() for p in raw.answerability.parameters()),
            update_performed=True,
            head_kind="score_statistics_only" if arm == "A4" else "pair_features",
        )
        if arm == "A3":
            trained_a3 = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items() if not k.startswith("clip.")}
        del wrapper, opt, scheduler, raw, own, f, p, presence, q, v, terms, global_v
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    a3 = SemanticRetrievalModel(replace(cfg.model, pilot_arm="A3")).to(device)
    own = a3.state_dict()
    own.update(trained_a3)
    a3.load_state_dict(own)
    return probes, a3


def _calibration_probe(context, cfg, model, *, device):
    source = cfg.data.episode_manifest.parent
    cal_path = source / "val_calibration.json"
    cal_payload = strict_json_loads(cal_path.read_text())
    episodes = read_episodes(cal_path, gallery_version=cal_payload["gallery_version"], split="val_calibration")
    gallery_path, target_gallery = source / "val_calibration-gallery.json", source / "val_locked-gallery.json"
    gallery = stream_pilot_gallery(gallery_path, gallery_version=cal_payload["gallery_version"])
    ids = gallery.canonical_ids
    qids = tuple(e.query_id for e in episodes)
    positives = {e.query_id: e.benchmark_positive_ids for e in episodes}
    # Two separately recorded, controlled natural orders guarantee both event classes
    # while fitted probabilities remain the actual C3 fit (including saturation).
    order = np.stack(
        [
            np.arange(len(ids)),
            np.asarray(
                [j for j in range(len(ids)) if ids[j] != positives[qids[1]][0]] + [ids.index(positives[qids[1]][0])]
            ),
        ]
    )
    panels = [dict(track=g["track"], K=g["K"], **g["arrays"]) for g in panel_groups(qids, ids, order, positives)]
    canonical = np.asarray([[int(i in positives[q]) for i in ids] for q in qids])
    official = canonical.copy()
    tokenizer = ClipTokenizer()
    token_ids, mask = tokenizer.encode_batch(
        [e.query_text for e in episodes], episodes[0].preprocessing.max_text_length
    )
    descriptor = dict(
        version=cal_payload["gallery_version"],
        manifest_sha256=file_sha256(gallery_path),
        canonical_ids=list(ids),
        aliases={i: list(v) for i, v in gallery.aliases.items()},
    )
    fragment = export_model_scores(
        context,
        model,
        QueryInputs(token_ids.to(device), mask.to(device)),
        gallery.batches(),
        None,
        query_ids=qids,
        component_ids=[e.query_component_id for e in episodes],
        canonical_relevance=canonical,
        official_relevance=official,
        gallery=descriptor,
        panels=panels,
        arm="A3",
        seed=0,
        split="val_calibration",
        split_sha256=file_sha256(cal_path),
        label_protocol=cfg.data.label_protocol,
        preprocessing_sha256=preprocessing_sha256(episodes[0].preprocessing),
        policy_sha256=policy_sha256(cfg.policy),
        pair_batch_size=16,
    )
    states = {}
    for track in ("natural", "controlled"):
        rows = []
        for group in fragment["groups"]:
            if group["track"] != track:
                continue
            path = Path(group["arrays"]["path"])
            if not path.is_absolute():
                path = context.paths.root / path
            with np.load(path, allow_pickle=False) as arrays:
                row = dict(
                    K=group["K"],
                    query_ids=arrays["query_ids"],
                    canonical_ids=ids,
                    pair_logits=arrays["pair_logits"],
                    pair_labels=arrays["relevance"],
                    candidate_mask=arrays["candidate_mask"],
                    presence_logits=arrays["presence_logits"],
                    presence_labels=(arrays["relevance"] * arrays["candidate_mask"]).max(1),
                )
                rows.append(row)
        identity = dict(
            model_sha256=model_sha256(model),
            preprocessing_sha256=fragment["preprocessing_sha256"],
            policy_sha256=fragment["policy_sha256"],
            label_protocol=cfg.data.label_protocol,
            relevance_space="benchmark_annotation",
            fit_gallery_version=descriptor["version"],
            target_gallery_version=strict_json_loads(target_gallery.read_text())["gallery_version"],
            fit_gallery_manifest_sha256=file_sha256(gallery_path),
            target_gallery_manifest_sha256=file_sha256(target_gallery),
            split_sha256=file_sha256(cal_path),
            track=track,
            candidate_mechanism="synthetic-controlled-code-orders",
            k_weights={str(k): 1 / 6 for k in KS},
        )
        fitted = fit_pilot_calibration(rows, identity, split="val_calibration", has_presence=True)
        if any(fitted[head]["status"] != "available" for head in ("pair", "presence")):
            raise ValueError("synthetic known-class calibration unexpectedly unavailable")
        for head in ("pair", "presence"):
            probs = calibrated_probabilities(
                rows[0][head + "_logits"],
                fitted,
                head,
                track=track,
                target_gallery_manifest_sha256=identity["target_gallery_manifest_sha256"],
            )
            if not np.isfinite(probs).all():
                raise ValueError("nonfinite pilot fitted probabilities")
        for wrong_track, wrong_gallery in (
            ("controlled" if track == "natural" else "natural", identity["target_gallery_manifest_sha256"]),
            (track, identity["fit_gallery_manifest_sha256"]),
        ):
            try:
                calibrated_probabilities(
                    rows[0]["pair_logits"],
                    fitted,
                    "pair",
                    track=wrong_track,
                    target_gallery_manifest_sha256=wrong_gallery,
                )
            except ValueError:
                pass
            else:
                raise ValueError("cross-track or cross-gallery calibration was accepted")
        unavailable = fit_pilot_calibration(rows, identity, split="val_calibration", has_presence=False)
        if unavailable["presence"]["status"] != "unavailable":
            raise ValueError("untrained presence head received calibration")
        states[track] = fitted
    refs = [g["arrays"] for g in fragment["groups"]]
    write_json(
        context.paths.provenance / "pilot-calibration-qualification.json",
        dict(
            states=states,
            array_references=refs,
            scope="synthetic identity and fit plumbing only; shared raw content is not held-out scientific validation",
            controlled_order_fixture=True,
            fit_probabilities_unmodified=True,
        ),
    )
    return states, refs


def qualify_pilot_chain(context, cfg, *, device):
    started = time.monotonic()
    validate_pilot_config(cfg, world_size=4)
    if cfg.model.pilot_arm != "P1" or cfg.data.label_protocol["protocol_id"] != Q0_PROTOCOL:
        raise ValueError("new qualification requires its own P1 and synthetic-v2 protocol")
    # All new-chain assets, including diagnostic gallery/label inputs, must be reserved.
    for ref in context.manifest["input_checkpoints"] + context.manifest["dataset_splits"]:
        _bound(context, ref)
    for name in (
        "train.json",
        "val_calibration.json",
        "val_locked.json",
        "val_calibration-gallery.json",
        "val_locked-gallery.json",
        "relations.json",
        "id_map.json",
    ):
        path = cfg.data.episode_manifest.parent / name
        _bound(context, file_reference(path))
    episodes = read_episodes(cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split="train")
    if any(e.label_protocol != cfg.data.label_protocol for e in episodes):
        raise ValueError("qualification v2 label identity mismatch")
    lookup = {e.query_id: e for e in episodes}
    orders = read_candidate_orders(cfg.train.pilot, episodes)
    p1, batch, training = _training_probe(context, cfg, lookup, orders, device)
    probes, model = _gradient_probes(context, cfg, p1, batch, device)
    del p1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        write_json(
            context.paths.provenance / "pilot-gradient-qualification.json",
            dict(gradient_probes=probes, source_sha=context.manifest["git_sha"], synthetic_only=True),
        )
    rank_records = _all(
        dict(
            rank=rank,
            gpu_uuid=None
            if device.type != "cuda"
            else __import__("scripts.qualify_semantic_retrieval", fromlist=["local_device_uuid"]).local_device_uuid(
                rank
            ),
            query_count=len(batch.query_ids),
            query_identity_sha256=hashlib.sha256(artifact_json_bytes(batch.query_ids)).hexdigest(),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        )
    )
    report = dict(
        schema_version=1,
        protocol=Q0_PROTOCOL,
        source_sha=context.manifest["git_sha"],
        world_size=dist.get_world_size() if dist.is_initialized() else 1,
        device=device.type,
        precision="fp32",
        tf32=False,
        synthetic_only=True,
        physical_indices=list(PHYSICAL_INDICES),
        rank_records=rank_records,
        checks={key: True for key in Q0_CHECKS},
        gradient_probes=probes,
        **training,
    )
    # Keep partial evidence if a later C3 consumer fails, exactly as required for Q0.
    error = None
    if rank == 0:
        try:
            states, refs = _calibration_probe(context, cfg, model, device=device)
            report.update(calibration=states, array_references=refs, elapsed_seconds=time.monotonic() - started)
            write_json(context.paths.provenance / "pilot-qualification.json", report)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        result = [(report, error) if rank == 0 else None]
        dist.broadcast_object_list(result, src=0)
        report, error = result[0]
    if error is not None:
        raise ValueError("pilot C3 qualification failed after archived training evidence: " + error)
    return report
