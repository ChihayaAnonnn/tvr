"""Audited raw-content episodes; labels never enter inference tensor contracts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

LABELS = {"positive": 1, "negative": 0, "unknown": -1}
PRESENCE = {"yes": 1, "no": 0, "unknown": -1}
SPLITS = {"train", "val", "val_development", "val_calibration", "val_locked"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class EpisodePreprocessing:
    max_text_length: int
    max_frames: int
    image_size: int

    def __post_init__(self):
        if (
            any(type(x) is not int or x <= 0 for x in (self.max_text_length, self.max_frames, self.image_size))
            or self.max_text_length < 2
        ):
            raise ValueError("preprocessing capacities must be positive with text length >= 2")


@dataclass(frozen=True)
class Episode:
    query_id: str
    query_text: str
    gallery_version: str
    candidate_ids: tuple[str, ...]
    canonical_ids: tuple[str, ...]
    relevance: tuple[int, ...]
    answer_in_candidates: int
    answer_in_gallery: int
    annotation_provenance: Mapping
    split: str
    video_paths: tuple[Path, ...]
    content_sha256: tuple[str, ...]
    frame_counts: tuple[int, ...]
    source_ids: tuple[str, ...]
    caption_group_id: str
    preprocessing: EpisodePreprocessing
    aliases: Mapping[str, tuple[str, ...]]
    query_conditioned_group_labels: tuple[tuple[str, str, int], ...] = ()


def read_episodes(path: Path, *, gallery_version: str, split: str) -> list[Episode]:
    if split == "test":
        raise ValueError("test remains sealed")
    if split not in SPLITS:
        raise ValueError("unsupported episode split")
    path = Path(path)
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "gallery_version", "preprocessing", "episodes"}
        or payload["schema_version"] != 1
        or payload["gallery_version"] != gallery_version
    ):
        raise ValueError("episode manifest schema/gallery mismatch")
    if not isinstance(payload["episodes"], list) or not payload["episodes"]:
        raise ValueError("episodes must be a nonempty list")
    preprocessing = EpisodePreprocessing(**payload["preprocessing"])
    required = {
        "query_id",
        "query_text",
        "gallery_version",
        "candidate_ids",
        "canonical_ids",
        "relevance",
        "answer_in_candidates",
        "answer_in_gallery",
        "annotation_provenance",
        "split",
        "video_paths",
        "content_sha256",
        "frame_counts",
        "source_ids",
        "caption_group_id",
    }
    episodes = []
    query_ids = set()
    for row in payload["episodes"]:
        if (
            not isinstance(row, dict)
            or not required.issubset(row)
            or set(row) - required - {"query_conditioned_group_labels"}
        ):
            raise ValueError("episode row fields mismatch")
        if row["gallery_version"] != gallery_version or row["split"] != split:
            raise ValueError("episode split/gallery mismatch")
        for name in ("query_id", "query_text", "caption_group_id"):
            if not isinstance(row[name], str) or (name != "query_text" and not row[name]):
                raise ValueError(f"invalid {name}")
        if row["query_id"] in query_ids:
            raise ValueError("duplicate query identity")
        query_ids.add(row["query_id"])
        columns = (
            "candidate_ids",
            "canonical_ids",
            "relevance",
            "video_paths",
            "content_sha256",
            "frame_counts",
            "source_ids",
        )
        size = len(row["candidate_ids"])
        if any(not isinstance(row[name], list) or len(row[name]) != size for name in columns):
            raise ValueError("candidate field lengths mismatch")
        if len(set(row["candidate_ids"])) != size:
            raise ValueError("candidate aliases must be unique")
        provenance = row["annotation_provenance"]
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("source"), str)
            or not provenance["source"]
            or not _digest(provenance.get("sha256"))
            or type(provenance.get("reviewed")) is not bool
        ):
            raise ValueError("invalid annotation provenance")
        if (
            any(label not in LABELS for label in row["relevance"])
            or row["answer_in_candidates"] not in PRESENCE
            or row["answer_in_gallery"] not in PRESENCE
        ):
            raise ValueError("invalid three-state supervision")
        if not provenance["reviewed"] and (
            any(x != "unknown" for x in row["relevance"])
            or row["answer_in_candidates"] != "unknown"
            or row["answer_in_gallery"] != "unknown"
        ):
            raise ValueError("known labels require reviewed annotation provenance")
        merged = {}
        for alias, canonical, label, raw, digest, count, source in zip(*(row[name] for name in columns)):
            if (
                any(not isinstance(x, str) or not x for x in (alias, canonical, source))
                or not _digest(digest)
                or type(count) is not int
                or count <= 0
            ):
                raise ValueError("invalid candidate identity/content/frame count")
            video = (path.parent / Path(raw)).resolve()
            if not video.is_file() or file_sha256(video) != digest:
                raise ValueError("candidate content SHA-256 mismatch")
            label = LABELS[label]
            if canonical in merged:
                old = merged[canonical]
                if old["digest"] != digest or old["count"] != count:
                    raise ValueError("canonical aliases disagree on content")
                if old["label"] != -1 and label != -1 and old["label"] != label:
                    raise ValueError("canonical aliases have contradictory labels")
                if label != -1:
                    old["label"] = label
                old["aliases"].append(alias)
                old["sources"].add(source)
            else:
                merged[canonical] = dict(
                    path=video, digest=digest, count=count, label=label, aliases=[alias], sources={source}
                )
        ids = tuple(sorted(merged))
        labels = tuple(merged[i]["label"] for i in ids)
        candidate_presence = PRESENCE[row["answer_in_candidates"]]
        has_positive = 1 in labels
        complete = all(x != -1 for x in labels)
        if candidate_presence == 0 and has_positive or candidate_presence == 1 and complete and not has_positive:
            raise ValueError("candidate presence contradicts reliable pair labels")
        if has_positive:
            candidate_presence = 1
        elif complete:
            candidate_presence = 0
        elif candidate_presence == 0:
            raise ValueError("candidate absence cannot be proven with unknown pairs")
        gallery_presence = PRESENCE[row["answer_in_gallery"]]
        if gallery_presence == 0:
            audit = provenance.get("gallery_audit", {})
            if (
                not isinstance(audit, dict)
                or audit.get("gallery_version") != gallery_version
                or audit.get("scope") != "complete"
                or audit.get("reviewed") is not True
                or not audit.get("source")
            ):
                raise ValueError("gallery audit required for a no-answer declaration")
            if has_positive:
                raise ValueError("gallery absence contradicts positive candidates")
        elif has_positive:
            gallery_presence = 1
        groups = []
        for group in row.get("query_conditioned_group_labels", []):
            if (
                not provenance["reviewed"]
                or not provenance.get("equivalence_reviewed")
                or len(group) != 3
                or group[0] not in ids
                or group[1] not in ids
                or group[0] == group[1]
                or group[2] not in LABELS
            ):
                raise ValueError("group labels require reviewed query-conditioned equivalence provenance")
            groups.append((group[0], group[1], LABELS[group[2]]))
        # Retain all source lineage, including aliases, for leakage auditing.
        sources = tuple(sorted({source for item in merged.values() for source in item["sources"]}))
        episodes.append(
            Episode(
                row["query_id"],
                row["query_text"],
                gallery_version,
                tuple(min(merged[i]["aliases"]) for i in ids),
                ids,
                labels,
                candidate_presence,
                gallery_presence,
                MappingProxyType(provenance),
                split,
                tuple(merged[i]["path"] for i in ids),
                tuple(merged[i]["digest"] for i in ids),
                tuple(merged[i]["count"] for i in ids),
                sources,
                row["caption_group_id"],
                preprocessing,
                MappingProxyType({i: tuple(sorted(merged[i]["aliases"])) for i in ids}),
                tuple(groups),
            )
        )
    audit_episode_splits(episodes)
    return episodes


def audit_episode_splits(episodes):
    seen = {}
    contents = {}
    for episode in episodes:
        for canonical, digest in zip(episode.canonical_ids, episode.content_sha256):
            if canonical in contents and contents[canonical] != digest:
                raise ValueError("canonical content conflict across episodes")
            contents[canonical] = digest
        for kind, values in [
            ("canonical", episode.canonical_ids),
            ("source", episode.source_ids),
            ("caption", (episode.caption_group_id,)),
        ]:
            for value in values:
                key = (kind, value)
                if key in seen and seen[key] != episode.split:
                    raise ValueError(f"{kind} split leakage: {value}")
                seen[key] = episode.split
