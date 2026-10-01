"""Explicit FP32 training and known-relation distributed normalization."""

import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from semantic_retrieval.models.pair_matcher import validate_offsets, validate_pairs
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
    local = (vids, records)
    if dist.is_initialized():
        all_records = [None] * dist.get_world_size()
        dist.all_gather_object(all_records, local)
    else:
        all_records = [local]
    gallery = [i for ids, _ in all_records for i in ids]
    known = {}
    for _, relations in all_records:
        for q, v, label in relations:
            if (q, v) in known and known[q, v] != label:
                raise ValueError("cross-rank annotation conflict")
            known[q, v] = label
    gathered = gather_with_grad(video_embeddings)
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
        desired = {
            id(p)
            for n, p in self.raw.named_parameters()
            if n.startswith(context.system_config.train.trainable_prefixes)
        }
        trainable = {id(p) for p in self.raw.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if desired != trainable or set(actual) != desired or len(actual) != len(desired):
            raise ValueError("optimizer must cover exactly the declared trainable parameters")
        if scheduler.optimizer is not optimizer:
            raise ValueError("scheduler optimizer mismatch")

    def restore(self, state):
        self.optimizer.load_state_dict(state.optimizer_state)
        self.scheduler.load_state_dict(state.scheduler_state)
        restore_rng(state.rng_state)
        self.step = state.step

    def train_step(self, batch):
        assert_running(self.context)
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
        # forward is the same named matching pipeline, including when wrapped by DDP.
        f, p, presence, q, v = self.model(batch.inputs, batch.pair_indices, batch.query_offsets)
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
        )
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
        if active:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
        metrics = {k: float(value.detach()) for k, value in terms.items() if not k.startswith("valid_")}
        metrics.update({"valid_" + k + "_count": int(global_counts[i]) for i, k in enumerate(LOSS_NAMES)})
        metrics.update(stepped=int(active), step=self.step)
        return metrics
