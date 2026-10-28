"""Explicit raw gallery inventory; never infer full-library coverage from episode GT."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from semantic_retrieval.data.episodes import EpisodePreprocessing, file_sha256
from semantic_retrieval.data.video import decode_video, sample_frame_indices


@dataclass(frozen=True)
class Gallery:
    canonical_ids: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]]
    content_sha256: dict[str, str]
    frame_counts: dict[str, int]
    video: torch.Tensor
    frame_mask: torch.Tensor
    preprocessing: EpisodePreprocessing


def read_gallery(path, *, gallery_version):
    payload = json.loads(path.read_text())
    if (
        set(payload) != {"schema_version", "gallery_version", "preprocessing", "videos"}
        or payload["schema_version"] != 1
        or payload["gallery_version"] != gallery_version
    ):
        raise ValueError("gallery manifest identity/schema mismatch")
    pre = EpisodePreprocessing(**payload["preprocessing"])
    rows = {}
    aliases = {}
    alias_ids = set()
    for row in payload["videos"]:
        if (
            set(row) != {"canonical_id", "alias_id", "video_path", "content_sha256", "frame_count"}
            or not all(isinstance(row[k], str) and row[k] for k in ("canonical_id", "alias_id", "video_path"))
            or row["alias_id"] in alias_ids
            or type(row["frame_count"]) is not int
            or row["frame_count"] <= 0
        ):
            raise ValueError("invalid gallery row")
        raw = Path(row["video_path"])
        file = (raw if raw.is_absolute() else path.parent / raw).resolve()
        if file_sha256(file) != row["content_sha256"]:
            raise ValueError("gallery content hash mismatch")
        i = row["canonical_id"]
        previous = rows.get(i)
        if previous is not None and (previous[1], previous[2]) != (row["content_sha256"], row["frame_count"]):
            raise ValueError("canonical gallery content conflict")
        rows.setdefault(i, (file, row["content_sha256"], row["frame_count"]))
        aliases.setdefault(i, []).append(row["alias_id"])
        alias_ids.add(row["alias_id"])
    ids = tuple(sorted(rows))
    video = torch.zeros(len(ids), pre.max_frames, 3, pre.image_size, pre.image_size)
    mask = torch.zeros(len(ids), pre.max_frames, dtype=torch.bool)
    for j, i in enumerate(ids):
        file, _, count = rows[i]
        frames, valid = decode_video(
            file, frame_indices=sample_frame_indices(count, pre.max_frames), image_size=pre.image_size
        )
        video[j, : len(frames)] = frames
        mask[j, : len(frames)] = valid
    return Gallery(
        ids,
        {i: tuple(sorted(v)) for i, v in aliases.items()},
        {i: rows[i][1] for i in ids},
        {i: rows[i][2] for i in ids},
        video,
        mask,
        pre,
    )


def audit_episode_gallery(episodes, gallery):
    for e in episodes:
        if e.preprocessing != gallery.preprocessing:
            raise ValueError("query/gallery preprocessing mismatch")
        for i, digest, count in zip(e.canonical_ids, e.content_sha256, e.frame_counts):
            if i in gallery.content_sha256 and (
                digest != gallery.content_sha256[i] or count != gallery.frame_counts[i]
            ):
                raise ValueError("canonical episode/gallery content or frame-count conflict")


def preprocessing_sha256(preprocessing):
    """Consumer implementation and parameters, independent of control-plane commits."""
    import hashlib
    from dataclasses import asdict

    from experiment_artifacts import artifact_json_bytes

    root = Path(__file__).parent
    names = ("video.py", "tokenizer.py", "bpe_simple_vocab_16e6.txt.gz")
    payload = dict(
        schema_version=1,
        parameters=asdict(preprocessing),
        implementation={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    )
    return hashlib.sha256(artifact_json_bytes(payload)).hexdigest()
