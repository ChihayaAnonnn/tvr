"""Registered pilot arms, dynamic query batches and controlled candidate panels."""

import hashlib
import json
import random
from dataclasses import replace
from types import MappingProxyType

import torch

from semantic_retrieval.config import load_system_config
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.models.model import SemanticRetrievalModel
from semantic_retrieval.runtime.checkpoint import _validated_checkpoint, config_sha256

KS = (4, 8, 16, 32)
PANELS = ("present", "absent")


class PilotSampler:
    """All ranks reconstruct one global table using an isolated sampling RNG."""

    def __init__(self, query_ids, *, seed):
        self.query_ids = tuple(sorted(query_ids))
        if len(self.query_ids) < 16 or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("pilot requires at least 16 unique queries; padding is forbidden")
        self.seed = seed
        self.rng = random.Random(seed)
        self.order = list(self.query_ids)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.iteration = 0
        self.schedule = []
        self.seen = set()
        self.identity = hashlib.sha256(json.dumps([self.query_ids, seed]).encode()).hexdigest()

    def next_batch(self, *, rank=None, world_size=4):
        if world_size != 4 or rank is not None and rank not in range(4):
            raise ValueError("pilot requires four ranks")
        if self.iteration % 120 == 0:
            self.schedule = [(k, p) for k in KS for p in PANELS for _ in range(15)]
            self.rng.shuffle(self.schedule)
        chosen = []
        while len(chosen) < 16:
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            query = self.order[self.cursor]
            self.cursor += 1
            if query not in chosen:
                chosen.append(query)
        k, panel = self.schedule[self.iteration % 120]
        self.iteration += 1
        self.seen.update(chosen)
        rows = tuple(chosen if rank is None else chosen[rank * 4 : (rank + 1) * 4])
        return rows, k, panel

    def state_dict(self):
        return dict(
            schema_version=1,
            identity=self.identity,
            order=list(self.order),
            cursor=self.cursor,
            iteration=self.iteration,
            schedule=list(self.schedule),
            rng=self.rng.getstate(),
            seen=sorted(self.seen),
        )

    def load_state_dict(self, state):
        fields = {"schema_version", "identity", "order", "cursor", "iteration", "schedule", "rng", "seen"}
        if (
            not isinstance(state, dict)
            or set(state) != fields
            or state["schema_version"] != 1
            or state["identity"] != self.identity
            or sorted(state["order"]) != list(self.query_ids)
            or type(state["cursor"]) is not int
            or not 0 <= state["cursor"] <= len(self.order)
            or type(state["iteration"]) is not int
            or state["iteration"] < 0
            or not set(state["seen"]).issubset(self.query_ids)
        ):
            raise ValueError("sampler state identity/position mismatch")
        schedule = [tuple(s) for s in state["schedule"]]
        if state["iteration"] > 0 and sorted(schedule) != sorted(
            [(k, p) for k in KS for p in PANELS for _ in range(15)]
        ):
            raise ValueError("sampler schedule is not the registered balanced schedule")
        # Reconstruct from seed to authenticate the cursor, table, RNG and coverage together.
        expected = PilotSampler(self.query_ids, seed=self.seed)
        for _ in range(state["iteration"]):
            expected.next_batch()
        expected_state = expected.state_dict()
        normalized = dict(state, schedule=schedule)
        if normalized != expected_state:
            raise ValueError("sampler state differs from deterministic training history")
        self.order, self.cursor, self.iteration = expected.order, expected.cursor, expected.iteration
        self.schedule, self.rng, self.seen = expected.schedule, expected.rng, expected.seen


def read_candidate_orders(pilot, episodes):
    if file_sha256(pilot.candidate_order_path) != pilot.candidate_order_sha256:
        raise ValueError("candidate order SHA mismatch")
    orders = json.loads(pilot.candidate_order_path.read_text())
    if not isinstance(orders, dict) or set(orders) != {e.query_id for e in episodes}:
        raise ValueError("candidate order query identity mismatch")
    for e in episodes:
        row = orders[e.query_id]
        if not isinstance(row, list) or len(row) != len(set(row)) or any(not isinstance(x, str) for x in row):
            raise ValueError("candidate ordering requires unique canonical IDs")
        # Fail before training rather than only when the schedule eventually reaches K=32.
        controlled_panel(e, row, 32, "present")
        controlled_panel(e, row, 32, "absent")
    return orders


def controlled_panel(episode, candidate_order, k, panel):
    if k not in KS or panel not in PANELS:
        raise ValueError("unregistered training candidate panel")
    labels = dict(zip(episode.canonical_ids, episode.relevance))
    positives = [i for i, label in labels.items() if label == 1]
    if len(positives) != 1 or any(label == -1 for label in labels.values()):
        raise ValueError("pilot panels require exactly one known canonical positive and no unknown inventory labels")
    positive = positives[0]
    if positive not in candidate_order:
        raise ValueError("fixed candidate order does not contain the official positive")
    negatives = [i for i in candidate_order if i != positive]
    selected = negatives[: k - (panel == "present")] + ([positive] if panel == "present" else [])
    if len(selected) != k or not set(selected).issubset(labels):
        raise ValueError("episode inventory lacks required fixed-order panel candidates")
    selected = tuple(sorted(selected))
    positions = {i: p for p, i in enumerate(episode.canonical_ids)}
    idx = [positions[i] for i in selected]
    return replace(
        episode,
        candidate_ids=tuple(episode.candidate_ids[i] for i in idx),
        canonical_ids=selected,
        relevance=tuple(labels[i] for i in selected),
        answer_in_candidates=int(panel == "present"),
        video_paths=tuple(episode.video_paths[i] for i in idx),
        content_sha256=tuple(episode.content_sha256[i] for i in idx),
        frame_counts=tuple(episode.frame_counts[i] for i in idx),
        aliases=MappingProxyType({i: episode.aliases[i] for i in selected}),
        query_conditioned_group_labels=(),
    )


def pilot_trainable_prefixes(arm):
    matching = (
        "coarse.",
        "matcher.text_projection.",
        "matcher.frame_projection.",
        "matcher.wti.",
        "matcher.a_raw",
        "matcher.b",
    )
    if arm != "A0":
        matching += ("matcher.dustbin", "matcher.mlp.")
    return matching + (("answerability.",) if arm in {"A2", "A3", "A4"} else ())


def validate_pilot_config(config, *, world_size):
    p, t, m = config.train.pilot, config.train, config.model
    if p is None:
        if m.pilot_arm is not None:
            raise ValueError("pilot model requires explicit training protocol")
        return
    if (
        world_size != 4
        or m.pilot_arm != p.arm
        or t.seed not in (0, 1, 2)
        or t.scheduler_steps != (240 if p.arm == "A0" else 120)
        or t.learning_rate != 1e-4
        or t.weight_decay != 0.01
        or t.scheduler_name != "constant"
        or t.trainable_prefixes != pilot_trainable_prefixes(p.arm)
        or t.loss_weights != dict(coarse=1, match=1, pair=1, exist=int(p.arm in {"A2", "A3", "A4"}), align=0, group=0)
        or m.grouping is not None
        or m.alignment is not None
        or m.num_layers != 0
    ):
        raise ValueError("pilot training differs from registered arm/schedule/capacity/optimizer")
    if not m.clip.fixture and (
        m.dim != 512
        or m.coarse_dim != 256
        or m.num_heads != 8
        or m.max_frames != 8
        or m.clip.image_size != 224
        or m.clip.context_length != 77
        or m.clip.patch_size != 32
    ):
        raise ValueError("pilot requires registered frozen CLIP ViT-B/32 capacities")
    if config.data is None or config.data.split != "train":
        raise ValueError("pilot requires train-only data")


def initialize_pilot_model(config):
    """Load matching branches from a complete, independently hash-bound final P1."""
    torch.manual_seed(config.train.seed)
    model = SemanticRetrievalModel(config.model)
    p = config.train.pilot
    if p is None or p.arm in {"A0", "P1"}:
        return model, None
    for path, digest in (
        (p.parent_config_path, p.parent_config_sha256),
        (p.parent_checkpoint_path, p.parent_checkpoint_sha256),
        (p.parent_checkpoint_path.with_suffix(".json"), p.parent_metadata_sha256),
    ):
        if file_sha256(path) != digest:
            raise ValueError("P1 parent input hash mismatch")
    parent = load_system_config(p.parent_config_path)
    if (
        parent.model != replace(config.model, pilot_arm="P1")
        or parent.train.pilot is None
        or parent.train.pilot.arm != "P1"
        or parent.runtime.run_id != p.parent_run_id
        or parent.train.seed != config.train.seed
        or parent.data != config.data
        or parent.train.pilot.sampler_seed != p.sampler_seed
        or parent.train.pilot.candidate_order_sha256 != p.candidate_order_sha256
    ):
        raise ValueError("P1 parent model/data/label/sampler identity mismatch")
    validate_pilot_config(parent, world_size=4)
    # Parent has A3-shaped answerability head, so validate all tensors before importing matching state.
    with torch.random.fork_rng(devices=[]):
        parent_model = SemanticRetrievalModel(parent.model)
    payload = _validated_checkpoint(
        p.parent_checkpoint_path, parent_model, expected_config_sha256=config_sha256(parent)
    )
    progress = payload.get("progress")
    if payload["step"] != 120 or progress is None or progress["next_iteration"] != 120 or "sampler" not in progress:
        raise ValueError("branches require the final 120-update P1 checkpoint including sampler state")
    shared = {name: tensor for name, tensor in payload["model"].items() if not name.startswith("answerability.")}
    current = model.state_dict()
    if set(shared) != {n for n in current if not n.startswith("answerability.")}:
        raise ValueError("P1 matching tensor keys mismatch")
    current.update(shared)
    model.load_state_dict(current, strict=True)
    return model, progress["sampler"]


def probe_matching_presence_gradients(model, batch, weights):
    """Measure the two matcher gradient directions without accumulating .grad or stepping."""
    import torch.distributed as dist

    from semantic_retrieval.objectives.retrieval import compute_objective
    from semantic_retrieval.runtime.checkpoint import restore_rng, rng_state

    saved_rng = rng_state()
    try:
        _, pairs, presence, q, v, _ = model(batch.inputs, batch.pair_indices, batch.query_offsets)
        terms = compute_objective(
            q @ v.T, pairs, presence, batch.targets, batch.pair_indices, batch.query_offsets, weights=weights
        )
        parameters = [p for p in model.matcher.parameters() if p.requires_grad]
        matching = sum(weights[k] * terms[k] for k in ("match", "pair"))
        existence = weights["exist"] * terms["exist"]
        vectors = []
        for loss in (matching, existence):
            grads = (
                torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                if loss.requires_grad
                else [None] * len(parameters)
            )
            vector = torch.cat(
                [(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, grads)]
            ).detach()
            if dist.is_initialized():
                dist.all_reduce(vector)
                vector /= dist.get_world_size()
            vectors.append(vector)
        match, exist = vectors
        a, b = match.norm(), exist.norm()
        cosine = float((match @ exist / (a * b)).clamp(-1, 1)) if a > 0 and b > 0 else None
        return dict(matching_gradient_norm=float(a), presence_gradient_norm=float(b), gradient_cosine=cosine)
    finally:
        restore_rng(saved_rng)
