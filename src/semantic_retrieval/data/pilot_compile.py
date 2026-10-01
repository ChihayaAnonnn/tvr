"""Offline benchmark contracts and controlled panels, independent of model scores."""

from dataclasses import asdict, dataclass

from .benchmark_labels import candidate_presence
from .pilot_catalog import TARGETS, identity_digest


@dataclass(frozen=True)
class Panel:
    query_id: str
    candidate_ids: tuple[str, ...]
    answer_in_candidates: bool
    mode: str
    ranked_ids: tuple[str, ...]


def build_benchmark_panel(*, query_id, canonical_order, k, mode, labels, gallery_ids=None) -> Panel:
    order = tuple(canonical_order)
    gallery = set(labels.official_to_canonical.values()) if gallery_ids is None else set(gallery_ids)
    if (
        query_id not in labels.positives_by_query
        or len(set(order)) != len(order)
        or set(order) != gallery
        or type(k) is not int
        or k < 1
        or k > len(order)
        or mode not in {"natural", "present", "absent", "controlled_miss"}
    ):
        raise ValueError("invalid complete canonical order or panel capacity")
    positive = labels.positives_by_query[query_id]
    if len(positive) != 1 or not set(positive).issubset(gallery):
        raise ValueError("panel requires one gallery positive")
    negatives = tuple(i for i in order if i not in positive)
    ranked = order
    if mode == "present":
        candidates = negatives[: k - 1] + positive
    elif mode == "absent":
        candidates = negatives[:k]
    elif mode == "controlled_miss":
        if len(negatives) < 32:
            raise ValueError("controlled miss requires 32 negatives")
        ranked = negatives[:32] + positive + negatives[32:]
        candidates = ranked[:k]
    else:
        candidates = order[:k]
    if len(candidates) != k:
        raise ValueError("insufficient nonpositive candidates")
    return Panel(query_id, tuple(sorted(candidates)), candidate_presence(query_id, candidates, labels), mode, ranked)


def summarize_pilot_inputs(*, catalog, assignment, labels, targets=TARGETS) -> dict:
    videos = {e.official_video_id: e for e in catalog if e.kind == "video"}
    queries = {e.asset_id: e for e in catalog if e.kind == "query"}
    reasons = []
    splits = {}
    for split, ids in assignment.gallery_ids.items():
        ids = set(ids)
        canonical = {videos[i].canonical_id for i in ids if i in videos}
        selected = [e for e in queries.values() if e.official_video_id in ids]
        by_video = {}
        for q in selected:
            if q.asset_id not in labels.positives_by_query:
                reasons.append(f"{split}:missing_label")
                continue
            by_video.setdefault(q.official_video_id, []).append(q)
            if (
                assignment.asset_splits[q.asset_id] != split
                or assignment.asset_components[q.asset_id]
                != assignment.asset_components[videos[q.official_video_id].asset_id]
                or not set(labels.positives_by_query[q.asset_id]).issubset(canonical)
            ):
                reasons.append(f"{split}:source_or_gallery_truth_conflict")
        count = len(ids)
        components = {assignment.asset_components[videos[i].asset_id] for i in ids if i in videos}
        minimum = targets["train_min"] if split == "train" else targets[split]
        if count < minimum or len(canonical) < targets["canonical_min"] or set(by_video) != ids:
            reasons.append(f"{split}:quantity_or_query_gate")
        if split == "val_locked" and len(components) < targets["locked_components_min"]:
            reasons.append("val_locked:component_gate")
        splits[split] = dict(
            official_video_count=count,
            canonical_video_count=len(canonical),
            query_count=len(by_video),
            component_count=len(components),
            unused_official_count=len(assignment.unused_official_ids[split]),
            gallery_identity_sha256=identity_digest(sorted(ids)),
            positive_count_histogram={"1": len(by_video)},
        )
    return dict(
        schema_version=1,
        ready=not reasons,
        stop_reasons=sorted(set(reasons)),
        relevance_space="benchmark_annotation",
        semantic_review_performed=False,
        label_protocol=labels.protocol,
        splits=splits,
        assignment_sha256=identity_digest(asdict(assignment)),
    )


def compile_pilot_manifests(
    *, catalog, assignment, labels, input_summary, preprocessing, label_references, candidate_orders=None
) -> dict[str, dict]:
    if not input_summary["ready"] or input_summary["label_protocol"] != labels.protocol:
        raise ValueError("pilot input gates not ready")
    videos = {e.official_video_id: e for e in catalog if e.kind == "video"}
    queries = [e for e in catalog if e.kind == "query"]
    output = {}
    for split, official_ids in assignment.gallery_ids.items():
        canonical = {}
        for i in official_ids:
            e = videos[i]
            canonical.setdefault(e.canonical_id, e)
        gallery_version = f"{split}-{identity_digest(sorted(official_ids))}"
        rows = []
        for official in official_ids:
            eligible = [q for q in queries if q.official_video_id == official]
            if not eligible:
                raise ValueError("missing stable caption")

            # Numeric official caption IDs retain numeric ordering; fallback IDs
            # are immutable row IDs, never text/score-based selection.
            def caption_key(q):
                value = labels.caption_ids[q.asset_id]
                return (0, int(value)) if value.isdecimal() else (1, value)

            q = min(eligible, key=lambda q: (caption_key(q), q.asset_id))
            positive = labels.positives_by_query[q.asset_id]
            order = tuple(sorted(canonical)) if candidate_orders is None else tuple(candidate_orders[q.asset_id])
            if set(order) != set(canonical) or len(order) != len(canonical):
                raise ValueError("candidate order must be complete canonical gallery")
            # C1 sorted inventories are software contracts only. C2/E0 binds
            # actual CLIP orders. Keep top128 negatives + GT to bound JSON size.
            ids = tuple(sorted(tuple(i for i in order if i not in positive)[:128] + positive))
            members = [canonical[i] for i in ids]
            rows.append(
                dict(
                    query_id=q.asset_id,
                    query_text=labels.query_texts[q.asset_id],
                    gallery_version=gallery_version,
                    candidate_ids=[e.official_video_id for e in members],
                    canonical_ids=list(ids),
                    relevance=["positive" if i in positive else "negative" for i in ids],
                    answer_in_candidates="yes",
                    answer_in_gallery="yes",
                    annotation_provenance=dict(
                        source="benchmark_annotation", sha256=labels.annotation_sha256, protocol_id=labels.protocol_id
                    ),
                    split=split,
                    video_paths=[str(e.path) for e in members],
                    content_sha256=[e.sha256 for e in members],
                    frame_counts=[e.frame_count for e in members],
                    source_ids=[e.source_ids[0] for e in members],
                    caption_group_id=min(q.caption_group_ids),
                    query_source_ids=list(q.source_ids),
                    query_component_id=assignment.asset_components[q.asset_id],
                )
            )
        episodes = dict(
            schema_version=2,
            gallery_version=gallery_version,
            preprocessing=asdict(preprocessing),
            episodes=rows,
            relevance_space="benchmark_annotation",
            label_protocol=labels.protocol,
            label_references=label_references,
        )
        gallery = dict(
            schema_version=1,
            gallery_version=gallery_version,
            preprocessing=asdict(preprocessing),
            videos=[
                dict(
                    canonical_id=videos[i].canonical_id,
                    alias_id=i,
                    video_path=str(videos[i].path),
                    content_sha256=videos[i].sha256,
                    frame_count=videos[i].frame_count,
                )
                for i in official_ids
            ],
        )
        output[split] = dict(episodes=episodes, gallery=gallery)
    return output
