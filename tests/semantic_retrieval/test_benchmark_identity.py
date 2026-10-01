import copy
import hashlib
import json
from dataclasses import replace

import pytest
from test_episodes import episode_manifest as episode_manifest


def benchmark(payload, directory=None):
    payload = copy.deepcopy(payload)
    payload["schema_version"] = 2
    payload["relevance_space"] = "benchmark_annotation"
    payload["label_protocol"] = dict(
        protocol_id="fixture",
        annotation_sha256="a" * 64,
        relation_sha256="b" * 64,
        id_map_sha256="c" * 64,
        source_report_sha256="d" * 64,
        allowlist_sha256="e" * 64,
        source_annotation_sha256="f" * 64,
    )
    if directory is not None:
        refs = {}
        for name, value, key in [
            ("relations", {"q0": ["vb"]}, "relation_sha256"),
            ("id_map", {"va": "a", "vb": "b"}, "id_map_sha256"),
        ]:
            target = directory / f"{name}.json"
            target.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            refs[name] = {"path": str(target), "sha256": sha}
            payload["label_protocol"][key] = sha
        payload["label_references"] = refs
    row = payload["episodes"][0]
    row["query_source_ids"] = ["query-source"]
    row["query_component_id"] = "0" * 64
    row["annotation_provenance"] = {"source": "benchmark_annotation", "sha256": "a" * 64, "protocol_id": "fixture"}
    return payload


def test_benchmark_known_labels_do_not_claim_reviewed(episode_manifest):
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    episodes = read_episodes(write(benchmark(payload, write().parent)), gallery_version="g1", split="train")
    assert episodes[0].relevance_space == "benchmark_annotation"
    assert "reviewed" not in episodes[0].annotation_provenance
    assert episodes[0].query_source_ids == ("query-source",)
    assert episodes[0].label_protocol["protocol_id"] == "fixture"


@pytest.mark.parametrize("defect", ["scope", "protocol", "reviewed", "hash", "gallery_no", "false_label"])
def test_benchmark_scope_and_protocol_cannot_be_forged(episode_manifest, defect):
    from semantic_retrieval.data.episodes import read_episodes

    payload, write, _ = episode_manifest
    payload = benchmark(payload, write().parent)
    if defect == "scope":
        payload.pop("relevance_space")
    elif defect == "protocol":
        payload.pop("label_protocol")
    elif defect == "reviewed":
        payload["episodes"][0]["annotation_provenance"]["reviewed"] = True
    elif defect == "hash":
        payload["episodes"][0]["annotation_provenance"]["sha256"] = "9" * 64
    elif defect == "false_label":
        payload["episodes"][0]["relevance"] = ["negative", "negative"]
    elif defect == "gallery_no":
        payload["episodes"][0]["answer_in_gallery"] = "no"
    with pytest.raises(ValueError):
        read_episodes(write(payload), gallery_version="g1", split="train")


def test_query_lineage_leaks_even_when_query_source_video_not_a_candidate(episode_manifest):
    from semantic_retrieval.data.episodes import audit_episode_splits, read_episodes

    payload, write, _ = episode_manifest
    e = read_episodes(write(benchmark(payload, write().parent)), gallery_version="g1", split="train")[0]
    other = replace(e, split="val_locked", canonical_ids=(), content_sha256=(), source_ids=(), caption_group_id="other")
    with pytest.raises(ValueError, match="query"):
        audit_episode_splits([e, other])
