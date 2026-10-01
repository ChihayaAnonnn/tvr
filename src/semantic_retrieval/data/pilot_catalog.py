"""Bound training-source assets and component-wise deterministic pilot partitions."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .episodes import _digest, file_sha256

TARGETS = dict(
    val_locked=1000,
    val_calibration=500,
    val_development=500,
    train_min=6500,
    canonical_min=129,
    locked_components_min=800,
)
NAMESPACE = "semantic-answerability-pilot-v1:"


def identity_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CatalogEntry:
    asset_id: str
    kind: str
    path: Path
    sha256: str
    official_video_id: str
    canonical_id: str
    source_ids: tuple[str, ...]
    caption_group_ids: tuple[str, ...]
    frame_count: int | None
    origin_split: str
    source_report_sha256: str | None = None
    allowlist_sha256: str | None = None


@dataclass(frozen=True)
class SplitAssignment:
    asset_splits: dict
    asset_components: dict
    gallery_ids: dict
    component_ids: dict
    unused_official_ids: dict


def _safe_path(value, *, base, allowed_root):
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    root = Path(allowed_root).resolve()
    if not path.absolute().is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError("asset path escapes allowed root")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError("symlink asset/provenance is forbidden")
    return path.resolve()


def load_pilot_catalog(
    path: Path, *, expected_sha256: str, allowed_root: Path, source_confirmation: dict | None = None
) -> tuple[CatalogEntry, ...]:
    path = Path(path).absolute()
    path = _safe_path(path, base=Path.cwd(), allowed_root=path.parent)
    if file_sha256(path) != expected_sha256:
        raise ValueError("catalog SHA mismatch")
    payload = json.loads(path.read_text())
    if (
        set(payload) != {"schema_version", "origin_split", "protocol_id", "allowlist", "source_report", "entries"}
        or payload["schema_version"] != 1
        or payload["origin_split"] != "source_train"
    ):
        raise ValueError("catalog must have explicit train-only source provenance")
    bound = {}
    for name in ("allowlist", "source_report"):
        ref = payload[name]
        if set(ref) != {"path", "sha256"}:
            raise ValueError("source reference fields mismatch")
        source = _safe_path(ref["path"], base=path.parent, allowed_root=path.parent)
        if file_sha256(source) != ref["sha256"]:
            raise ValueError("source provenance SHA mismatch")
        bound[name] = json.loads(source.read_text())
    allow = bound["allowlist"]
    report = bound["source_report"]
    if (
        allow.get("origin_split") != "source_train"
        or report.get("origin_split") != "source_train"
        or report.get("source_verified") is not True
        or report.get("protocol_id") != payload["protocol_id"]
        or report.get("allowlist_sha256") != payload["allowlist"]["sha256"]
    ):
        raise ValueError("unverified training source identity")
    if (
        not isinstance(source_confirmation, dict)
        or set(source_confirmation) != {"path", "sha256"}
        or report.get("source_confirmation") != source_confirmation
    ):
        raise ValueError("independent source confirmation is required")
    confirmation_path = Path(source_confirmation["path"])
    if (
        not confirmation_path.is_absolute()
        or confirmation_path.resolve() != confirmation_path
        or confirmation_path.is_symlink()
        or file_sha256(confirmation_path) != source_confirmation["sha256"]
    ):
        raise ValueError("source confirmation SHA/path mismatch")
    confirmation = json.loads(confirmation_path.read_text())
    if (
        confirmation.get("status", {}).get("local_annotation_matches_published_captions")
        != "VERIFIED_EXACT_FOR_V2_TRAINING_ALLOWLIST_ONLY"
    ):
        raise ValueError("source confirmation lacks independently verified training captions")
    expected = confirmation.get("training_caption_comparison", {}).get("local_results", {}).get("MSRVTT_v2.json", {})
    approved = {(r["path"], r["sha256"]) for r in confirmation.get("local_assets", [])}
    for key in ("source_annotation", "allowlist_source"):
        ref = report.get(key)
        if not isinstance(ref, dict) or (ref.get("path"), ref.get("sha256")) not in approved:
            raise ValueError("source confirmation does not bind annotation/ID source")
    if report.get("caption_relation_sha256") != expected.get("relation_sha256"):
        raise ValueError("source confirmation training relation mismatch")
    allowed_ids = allow.get("official_video_ids")
    if (
        not isinstance(allowed_ids, list)
        or not allowed_ids
        or len(set(allowed_ids)) != len(allowed_ids)
        or not all(isinstance(x, str) and x for x in allowed_ids)
    ):
        raise ValueError("invalid source allowlist")
    if len(allowed_ids) != expected.get("training_video_count"):
        raise ValueError("source confirmation/allowlist quantity mismatch")
    entries = []
    video_ids = {}
    asset_ids = set()
    canonical = {}
    for raw in payload["entries"]:
        if set(raw) != set(CatalogEntry.__dataclass_fields__) - {"source_report_sha256", "allowlist_sha256"}:
            raise ValueError("catalog entry fields mismatch")
        if (
            raw["origin_split"] != "source_train"
            or raw["kind"] not in {"video", "query"}
            or raw["official_video_id"] not in allowed_ids
            or not _digest(raw["sha256"])
            or any(not isinstance(raw[k], str) or not raw[k] for k in ("asset_id", "canonical_id", "official_video_id"))
            or any(
                not isinstance(raw[k], list)
                or not raw[k]
                or not all(isinstance(x, str) and x for x in raw[k])
                or len(set(raw[k])) != len(raw[k])
                for k in ("source_ids", "caption_group_ids")
            )
        ):
            raise ValueError("catalog entry has unknown or inconsistent identity")
        if raw["asset_id"] in asset_ids:
            raise ValueError("duplicate asset identity")
        asset_ids.add(raw["asset_id"])
        raw = dict(raw)
        raw["path"] = _safe_path(
            raw["path"], base=path.parent, allowed_root=allowed_root if raw["kind"] == "video" else path.parent
        )
        raw["source_ids"] = tuple(sorted(raw["source_ids"]))
        raw["caption_group_ids"] = tuple(sorted(raw["caption_group_ids"]))
        entry = CatalogEntry(
            **raw,
            source_report_sha256=payload["source_report"]["sha256"],
            allowlist_sha256=payload["allowlist"]["sha256"],
        )
        if entry.kind == "video":
            if type(entry.frame_count) is not int or entry.frame_count <= 0 or entry.official_video_id in video_ids:
                raise ValueError("invalid frame count/duplicate official video identity")
            video_ids[entry.official_video_id] = entry
            signature = (entry.sha256, entry.frame_count)
            if entry.canonical_id in canonical and canonical[entry.canonical_id] != signature:
                raise ValueError("canonical content identity conflict")
            canonical[entry.canonical_id] = signature
        elif entry.frame_count is not None:
            raise ValueError("query does not carry video frame count")
        entries.append(entry)
    for entry in entries:
        if (
            entry.official_video_id not in video_ids
            or video_ids[entry.official_video_id].canonical_id != entry.canonical_id
            or not set(entry.source_ids) & set(video_ids[entry.official_video_id].source_ids)
        ):
            raise ValueError("query/video source identity conflict")
    # All metadata/path permission checks finish before any asset content open.
    verified = {}
    for entry in entries:
        if entry.path not in verified and entry.path.is_file():
            verified[entry.path] = file_sha256(entry.path)
        if verified.get(entry.path) != entry.sha256:
            raise ValueError("asset content SHA mismatch")
    return tuple(sorted(entries, key=lambda e: e.asset_id))


def assign_pilot_splits(entries, *, targets=TARGETS) -> SplitAssignment:
    if set(targets) != set(TARGETS) or any(type(v) is not int or v <= 0 for v in targets.values()):
        raise ValueError("invalid split targets")
    entries = tuple(sorted(entries, key=lambda e: e.asset_id))
    if len({e.asset_id for e in entries}) != len(entries):
        raise ValueError("duplicate asset identity")
    parent = {e.asset_id: e.asset_id for e in entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for e in entries:
        nodes = [("canonical", e.canonical_id), ("official", e.official_video_id)]
        if e.kind == "video":
            nodes.append(("content", e.sha256))
        nodes += [("source", x) for x in e.source_ids] + [("caption", x) for x in e.caption_group_ids]
        for node in nodes:
            if node in seen:
                parent[find(e.asset_id)] = find(seen[node])
            else:
                seen[node] = e.asset_id
    groups = {}
    for e in entries:
        groups.setdefault(find(e.asset_id), []).append(e)
    components = {identity_digest(sorted(e.asset_id for e in members)): members for members in groups.values()}
    order = sorted(components, key=lambda c: hashlib.sha256((NAMESPACE + c).encode()).hexdigest())
    splits = ("val_locked", "val_calibration", "val_development", "train")
    asset_splits = {}
    asset_components = {}
    official = {s: set() for s in splits}
    assigned = {s: [] for s in splits}
    index = 0
    for c in order:
        s = splits[index]
        for e in components[c]:
            asset_splits[e.asset_id] = s
            asset_components[e.asset_id] = c
            if e.kind == "video":
                official[s].add(e.official_video_id)
        assigned[s].append(c)
        if index < 3 and len(official[s]) >= targets[s]:
            index += 1
    videos = {e.official_video_id: e for e in entries if e.kind == "video"}
    gallery = {}
    unused = {}
    for s in splits:
        ids = sorted(official[s], key=lambda i: hashlib.sha256((NAMESPACE + i).encode()).hexdigest())
        keep = len(ids) if s == "train" else targets[s]
        gallery[s] = tuple(sorted(ids[:keep]))
        unused[s] = tuple(sorted(ids[keep:]))
        minimum = targets["train_min"] if s == "train" else targets[s]
        if len(gallery[s]) < minimum or len({videos[i].canonical_id for i in gallery[s]}) < targets["canonical_min"]:
            raise ValueError(f"quantity gate failed for {s}")
    locked_components = {asset_components[videos[i].asset_id] for i in gallery["val_locked"]}
    if len(locked_components) < targets["locked_components_min"]:
        raise ValueError("quantity gate failed for locked source components")
    return SplitAssignment(
        asset_splits, asset_components, gallery, {s: tuple(sorted(assigned[s])) for s in splits}, unused
    )
