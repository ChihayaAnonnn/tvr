"""Explicit FP32 training and known-relation distributed normalization."""

import math
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from semantic_retrieval.models.grouping import group_loss
from semantic_retrieval.models.pair_matcher import validate_offsets, validate_pairs
from semantic_retrieval.objectives.alignment import alignment_loss, dustbin_ot_target
from semantic_retrieval.objectives.retrieval import LOSS_NAMES, compute_objective
from semantic_retrieval.runtime.checkpoint import assert_running, restore_rng
from semantic_retrieval.runtime.distributed import gather_with_grad


def build_optimizer(model, config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    names = dict(model.named_parameters())
    if any(not any(n.startswith(prefix) for n in names) for prefix in config.trainable_prefixes):
        raise ValueError("unknown trainable prefix")
    for name, p in names.items():
        p.requires_grad_(name.startswith(config.trainable_prefixes))
    opt = torch.optim.AdamW(
        [p for p in names.values() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def factor(step):
        return max(0.0, 1 - step / config.scheduler_steps) if config.scheduler_name == "linear" else 1.0

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def global_coarse_relations(batch, video_embeddings):
    """Gather only reviewed relations; absence of a relation always remains unknown."""
    qids = batch.query_ids
    vids = batch.canonical_ids
    records = [
        (qids[q], vids[v], int(batch.targets.relevance[q, v]))
        for q in range(len(qids))
        for v in range(len(vids))
        if batch.targets.relevance[q, v] >= 0
    ]
    positives = getattr(batch, "benchmark_positive_ids", None)
    if positives is not None and (len(positives) != len(qids) or any(not row for row in positives)):
        raise ValueError("benchmark batches require positive IDs for every query")
    local = (vids, records, qids, positives)
    if dist.is_initialized():
        all_records = [None] * dist.get_world_size()
        dist.all_gather_object(all_records, local)
    else:
        all_records = [local]
    gallery = [i for ids, _, _, _ in all_records for i in ids]
    global_queries = [q for _, _, queries, _ in all_records for q in queries]
    if positives is not None and len(set(global_queries)) != len(global_queries):
        raise ValueError("pilot queries must be unique across ranks")
    known = {}
    for _, relations, queries, positives in all_records:
        if positives is not None:
            for query, relevant in zip(queries, positives):
                for video in set(gallery):
                    known[query, video] = int(video in relevant)
            continue
        for q, v, label in relations:
            if (q, v) in known and known[q, v] != label:
                raise ValueError("cross-rank annotation conflict")
            known[q, v] = label
    if dist.is_initialized():
        specs = [None] * dist.get_world_size()
        dist.all_gather_object(
            specs, (tuple(video_embeddings.shape), str(video_embeddings.dtype), video_embeddings.device.type)
        )
        if any(
            len(shape) != 2
            or shape[1] != video_embeddings.shape[1]
            or dtype != str(video_embeddings.dtype)
            or device != video_embeddings.device.type
            for shape, dtype, device in specs
        ):
            raise ValueError("gallery embedding dimensions/dtypes must agree across ranks")
        sizes = [shape[0] for shape, _, _ in specs]
        capacity = max(1, max(sizes))
        padded = torch.nn.functional.pad(video_embeddings, (0, 0, 0, capacity - len(video_embeddings)))
        gathered_padded = gather_with_grad(padded)
        gathered = torch.cat([gathered_padded[r * capacity : r * capacity + n] for r, n in enumerate(sizes)])
    else:
        gathered = video_embeddings
    # A canonical video appearing on more than one rank gets only one column.
    positions = {}
    for index, i in enumerate(gallery):
        positions.setdefault(i, index)
    ids = tuple(sorted(positions))
    indices = torch.tensor([positions[i] for i in ids], device=gathered.device, dtype=torch.long)
    values = gathered[indices]
    labels = torch.tensor(
        [[known.get((q, v), -1) for v in ids] for q in qids], dtype=torch.long, device=gathered.device
    ).reshape(len(qids), len(ids))
    return values, labels


class Trainer:
    def __init__(self, model, optimizer, scheduler, context):
        if context.system_config is None:
            raise ValueError("Trainer requires a bound SystemConfig")
        assert_running(context)
        self.model = model
        self.raw = model.module if isinstance(model, DistributedDataParallel) else model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.context = context
        self.step = 0
        verify_training_contract(self.raw, optimizer, scheduler, context, self.step)

    def restore(self, state):
        self.optimizer.load_state_dict(state.optimizer_state)
        self.scheduler.load_state_dict(state.scheduler_state)
        restore_rng(state.rng_state)
        self.step = state.step
        verify_training_contract(self.raw, self.optimizer, self.scheduler, self.context, self.step)

    def train_step(self, batch):
        assert_running(self.context)
        verify_training_contract(self.raw, self.optimizer, self.scheduler, self.context, self.step)
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise ValueError("training requires FP32")
        if len(set(batch.query_ids)) != len(batch.query_ids) or len(set(batch.canonical_ids)) != len(
            batch.canonical_ids
        ):
            raise ValueError("batch identities must be unique")
        if (
            len(batch.query_ids) != len(batch.inputs.input_ids)
            or len(batch.canonical_ids) != len(batch.inputs.video)
            or batch.targets.relevance.shape != (len(batch.query_ids), len(batch.canonical_ids))
        ):
            raise ValueError("batch identity shape mismatch")
        validate_pairs(batch.pair_indices, *batch.targets.relevance.shape)
        validate_offsets(batch.query_offsets, len(batch.pair_indices))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pilot = self.context.system_config.train.pilot is not None

        def timestamp():
            if pilot and batch.inputs.video.device.type == "cuda":
                torch.cuda.synchronize(batch.inputs.video.device)
            return time.perf_counter()

        started = timestamp()
        # forward is the same named matching pipeline, including when wrapped by DDP.
        gp = batch.targets.group_pairs
        weight = self.context.system_config.train.loss_weights["group"]
        config = self.raw.config.grouping
        if weight > 0:
            if config is None or gp is None or batch.group_provenance is None or len(batch.group_provenance) != len(gp):
                raise ValueError("group supervision requires configured annotation provenance")
            known = batch.targets.group_labels >= 0
            if any(
                d not in config.annotation_provenance_sha256
                for d, k in zip(batch.group_provenance, known.tolist())
                if k
            ):
                raise ValueError("group provenance is not authorized")
            if (
                (gp >= len(batch.pair_indices)).any()
                or (gp[:, 0] == gp[:, 1]).any()
                or not (batch.pair_indices[gp[:, 0], 0] == batch.pair_indices[gp[:, 1], 0]).all()
            ):
                raise ValueError("equivalence pairs must belong to the same query")
        align_config = self.raw.config.alignment
        align_weight = self.context.system_config.train.loss_weights["align"]
        if align_weight > 0 and align_config is None:
            raise ValueError("alignment weight requires explicit configuration")
        align_enabled = align_weight > 0 and self.step >= align_config.warmup_steps
        f, p, presence, q, v, group_logits = self.model(
            batch.inputs,
            batch.pair_indices,
            batch.query_offsets,
            group_pairs=gp if weight > 0 else None,
            return_alignment=align_enabled,
        )
        auxiliary = {"group": group_loss(group_logits, batch.targets.group_labels)} if group_logits is not None else {}
        align_count = 0
        if align_enabled:
            qi, vi = batch.pair_indices.unbind(1)
            positive = batch.targets.relevance[qi, vi] == 1
            if positive.any():
                tm = f.text_mask[qi[positive]]
                fm = f.frame_mask[vi[positive]]
                target = dustbin_ot_target(p.alignment_similarity[positive], tm, fm, config=align_config)
                auxiliary["align"] = alignment_loss(p.attention[positive], target, tm)
                align_count = int(tm.sum())
        global_v, relations = global_coarse_relations(batch, v)
        terms = compute_objective(
            q @ global_v.T,
            p,
            presence,
            batch.targets,
            batch.pair_indices,
            batch.query_offsets,
            weights=self.context.system_config.train.loss_weights,
            coarse_relevance=relations,
            auxiliary_losses=auxiliary,
        )
        terms["valid_align_count"] = torch.tensor(align_count, device=q.device)
        if group_logits is not None:
            terms["valid_group_count"] = (batch.targets.group_labels >= 0).sum()
        weights = self.context.system_config.train.loss_weights
        counts = torch.stack([terms["valid_" + k + "_count"] for k in LOSS_NAMES]).to(q.device)
        global_counts = counts.clone()
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world > 1:
            dist.all_reduce(global_counts)
        loss = terms["total"] * 0
        for i, k in enumerate(LOSS_NAMES):
            if global_counts[i] > 0:
                loss = loss + weights[k] * terms[k] * counts[i] * world / global_counts[i]
        active = any(weights[k] > 0 and global_counts[i] > 0 for i, k in enumerate(LOSS_NAMES))
        forward_done = timestamp()
        if active:
            loss.backward()
        backward_done = timestamp()
        if active:
            if pilot and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.raw.parameters()):
                raise ValueError("pilot has nonfinite gradients")
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
        optimizer_done = timestamp()
        self.last_timing = dict(
            forward_seconds=forward_done - started,
            backward_seconds=backward_done - forward_done,
            optimizer_seconds=optimizer_done - backward_done,
        )
        numerators = torch.stack([terms[k].detach() * counts[i] for i, k in enumerate(LOSS_NAMES)])
        if world > 1:
            dist.all_reduce(numerators)
        metrics = {
            k: float(numerators[i] / global_counts[i]) if global_counts[i] else 0.0 for i, k in enumerate(LOSS_NAMES)
        }
        metrics["total"] = sum(weights[k] * metrics[k] for k in LOSS_NAMES)
        metrics.update({"valid_" + k + "_count": int(global_counts[i]) for i, k in enumerate(LOSS_NAMES)})
        metrics.update(stepped=int(active), step=self.step)
        return metrics


def verify_training_contract(model, optimizer, scheduler, context, step):
    assert_running(context)
    config = context.system_config
    if config is None or model.config != config.model:
        raise ValueError("model config differs from bound reservation")
    cfg = config.train
    desired = {id(p) for n, p in model.named_parameters() if n.startswith(cfg.trainable_prefixes)}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if desired != trainable or set(actual) != desired or len(actual) != len(desired):
        raise ValueError("optimizer must cover exactly declared trainable parameters")
    if not isinstance(optimizer, torch.optim.AdamW) or any(
        g["weight_decay"] != cfg.weight_decay
        or g["betas"] != (0.9, 0.999)
        or g["eps"] != 1e-8
        or g["amsgrad"]
        or g["maximize"]
        for g in optimizer.param_groups
    ):
        raise ValueError("optimizer differs from registered AdamW settings")
    if (
        not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        or scheduler.optimizer is not optimizer
        or scheduler.last_epoch != step
        or any(lr != cfg.learning_rate for lr in scheduler.base_lrs)
    ):
        raise ValueError("scheduler differs from registered settings")

    def factor(s):
        return max(0.0, 1 - s / cfg.scheduler_steps) if cfg.scheduler_name == "linear" else 1.0

    if any(
        not math.isclose(fn(s), factor(s), rel_tol=1e-12, abs_tol=1e-12)
        for fn in scheduler.lr_lambdas
        for s in (0, step, step + 1, cfg.scheduler_steps)
    ):
        raise ValueError("scheduler strategy differs from registered configuration")
    if any(
        not math.isclose(g["lr"], cfg.learning_rate * factor(step), rel_tol=1e-12, abs_tol=1e-12)
        for g in optimizer.param_groups
    ):
        raise ValueError("optimizer learning rate differs from registered scheduler")
