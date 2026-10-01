import copy
import hashlib
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest


@pytest.fixture
def episode_manifest(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"v{i}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
        assert writer.isOpened()
        writer.write(np.full((16, 16, 3), 40 + i * 60, np.uint8))
        writer.release()
        paths.append(path)
    row = dict(
        query_id="q0",
        query_text="a video",
        gallery_version="g1",
        candidate_ids=["alias-b", "alias-a"],
        canonical_ids=["b", "a"],
        relevance=["positive", "negative"],
        answer_in_candidates="yes",
        answer_in_gallery="yes",
        annotation_provenance={"reviewed": True, "source": "synthetic-human-audit", "sha256": "a" * 64},
        split="train",
        video_paths=[str(paths[1]), str(paths[0])],
        content_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [paths[1], paths[0]]],
        frame_counts=[1, 1],
        source_ids=["s1", "s0"],
        caption_group_id="caption-0",
    )
    payload = {
        "schema_version": 1,
        "gallery_version": "g1",
        "preprocessing": {"max_text_length": 5, "max_frames": 4, "image_size": 8},
        "episodes": [row],
    }
    path = tmp_path / "episodes.json"

    def write(value=payload):
        path.write_text(json.dumps(value))
        return path

    return payload, write, paths


def test_single_multiple_miss_unknown_and_empty_candidates(episode_manifest):
    from semantic_retrieval.data.collate import collate_episodes
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    original = payload["episodes"][0]
    rows = []
    for i, (labels, presence, gallery) in enumerate(
        [
            (["positive", "negative"], "yes", "yes"),
            (["positive", "positive"], "yes", "yes"),
            (["negative", "negative"], "no", "yes"),
            (["unknown", "unknown"], "unknown", "unknown"),
            ([], "no", "unknown"),
        ]
    ):
        row = copy.deepcopy(original)
        row["query_id"] = f"q{i}"
        row["caption_group_id"] = f"caption-{i}"
        row.update(relevance=labels, answer_in_candidates=presence, answer_in_gallery=gallery)
        if not labels:
            for key in [
                "candidate_ids",
                "canonical_ids",
                "video_paths",
                "content_sha256",
                "frame_counts",
                "source_ids",
            ]:
                row[key] = []
        rows.append(row)
    payload["episodes"] = rows
    episodes = read_episodes(write(), gallery_version="g1", split="train")
    batch = collate_episodes(episodes)
    assert batch.canonical_ids == ("a", "b")
    assert batch.targets.candidate_presence.tolist() == [1, 1, 0, -1, 0]
    assert batch.query_offsets.tolist() == [0, 2, 4, 6, 8, 8]
    assert batch.inputs.video.shape == (2, 4, 3, 8, 8)
    assert batch.inputs.frame_mask.sum().item() == 2
    assert batch.targets.relevance[3].tolist() == [-1, -1]
    query = batch.inference_queries()
    assert not hasattr(query, "targets") and not hasattr(query, "relevance")
    empty = collate_episodes([episodes[-1]])
    assert empty.inputs.video.shape[0] == 0 and empty.pair_indices.shape == (0, 2)


@pytest.mark.parametrize("conflict", ["labels", "content", "presence"])
def test_alias_conflicts_are_rejected(episode_manifest, conflict):
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    row = payload["episodes"][0]
    if conflict == "presence":
        row["answer_in_candidates"] = "no"
    else:
        row["canonical_ids"] = ["b", "b"]
        if conflict == "labels":
            row["video_paths"][1] = row["video_paths"][0]
            row["content_sha256"][1] = row["content_sha256"][0]
        else:
            row["relevance"] = ["positive", "positive"]
    with pytest.raises(ValueError):
        read_episodes(write(), gallery_version="g1", split="train")


def test_reliable_label_wins_over_unknown_alias_without_duplication(episode_manifest):
    from semantic_retrieval.data.collate import collate_episodes
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    row = payload["episodes"][0]
    row.update(canonical_ids=["b", "b"], relevance=["unknown", "positive"])
    row["video_paths"][1] = row["video_paths"][0]
    row["content_sha256"][1] = row["content_sha256"][0]
    row["source_ids"][1] = row["source_ids"][0]
    batch = collate_episodes(read_episodes(write(), gallery_version="g1", split="train"))
    assert batch.canonical_ids == ("b",) and batch.targets.relevance.tolist() == [[1]]
    assert batch.query_offsets.tolist() == [0, 1]


def test_gallery_absence_requires_full_gallery_audit(episode_manifest):
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    row = payload["episodes"][0]
    row.update(relevance=["negative", "negative"], answer_in_candidates="no", answer_in_gallery="no")
    with pytest.raises(ValueError, match="gallery audit"):
        read_episodes(write(), gallery_version="g1", split="train")
    row["annotation_provenance"]["gallery_audit"] = {
        "gallery_version": "g1",
        "scope": "complete",
        "reviewed": True,
        "source": "synthetic-gallery-review",
    }
    assert read_episodes(write(), gallery_version="g1", split="train")[0].answer_in_gallery == 0
    row["answer_in_gallery"] = "unknown"
    assert read_episodes(write(), gallery_version="g1", split="train")[0].answer_in_gallery == -1


@pytest.mark.parametrize("identity", ["canonical", "source", "caption"])
def test_split_leakage_is_rejected(episode_manifest, identity):
    from semantic_retrieval.data.episodes import audit_episode_splits, read_episodes

    payload, write, _ = episode_manifest
    episode = read_episodes(write(), gallery_version="g1", split="train")[0]
    other = replace(
        episode,
        query_id="other",
        split="val_locked",
        canonical_ids=episode.canonical_ids if identity == "canonical" else ("other-a", "other-b"),
        source_ids=episode.source_ids if identity == "source" else ("other-s0", "other-s1"),
        caption_group_id=episode.caption_group_id if identity == "caption" else "other-caption",
    )
    with pytest.raises(ValueError, match="leak"):
        audit_episode_splits([episode, other])


@pytest.mark.parametrize("defect", ["hash", "gallery", "split", "provenance"])
def test_episode_input_binding_and_sealed_test(episode_manifest, defect):
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    row = payload["episodes"][0]
    if defect == "hash":
        row["content_sha256"][0] = "0" * 64
    elif defect == "gallery":
        row["gallery_version"] = "other"
    elif defect == "split":
        row["split"] = "test"
    else:
        row["annotation_provenance"]["reviewed"] = False
    with pytest.raises(ValueError):
        read_episodes(write(), gallery_version="g1", split="train")
    with pytest.raises(ValueError, match="sealed"):
        read_episodes(write(), gallery_version="g1", split="test")
