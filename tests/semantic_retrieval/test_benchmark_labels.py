"""Official relationships preserve one canonical positive without claiming semantic truth."""

import json
from dataclasses import replace

import pytest
from test_pilot_catalog import entries, write


def snapshot(tmp_path, rows=None):
    data = dict(
        schema_version=1,
        origin_split="source_train",
        protocol_id="fixture",
        source_annotation_sha256="a" * 64,
        source_report_sha256="b" * 64,
        allowlist_sha256="c" * 64,
        annotations=rows
        if rows is not None
        else [
            dict(query_id="q0", caption_id="0", official_video_id="0", query_text="hello"),
            dict(query_id="q1", caption_id="1", official_video_id="0", query_text="other"),
        ],
    )
    path = tmp_path / "snapshot.json"
    sha = write(path, data)
    return path, sha


def catalog(path=None, sha=None):
    items = entries(3)
    record_path = None
    record_sha = None
    if path is not None:
        record_path = path.parent / "training-records.json"
        rows = [dict(video_id="0", sen_id="0", caption="hello"), dict(video_id="0", sen_id="1", caption="other")]
        record_sha = write(record_path, dict(sentences=rows))
    return items + (
        replace(
            items[0],
            kind="query",
            asset_id="q0",
            frame_count=None,
            path=path,
            sha256=sha,
            source_report_sha256="b" * 64,
            allowlist_sha256="c" * 64,
            source_annotation_sha256="a" * 64,
            training_records_path=record_path,
            training_records_sha256=record_sha,
        ),
        replace(
            items[0],
            kind="query",
            asset_id="q1",
            frame_count=None,
            path=path,
            sha256=sha,
            source_report_sha256="b" * 64,
            allowlist_sha256="c" * 64,
            source_annotation_sha256="a" * 64,
            training_records_path=record_path,
            training_records_sha256=record_sha,
        ),
    )


def test_multiple_captions_are_queries_not_multiple_videos(tmp_path):
    from semantic_retrieval.data.benchmark_labels import candidate_presence, load_benchmark_labels

    path, sha = snapshot(tmp_path)
    labels = load_benchmark_labels(path, catalog=catalog(path, sha), expected_sha256=sha, protocol_id="fixture")
    assert labels.positives_by_query == {"q0": ("c0",), "q1": ("c0",)}
    assert labels.official_positives_by_query == {"q0": ("0",), "q1": ("0",)}
    assert candidate_presence("q0", ("c0", "c1"), labels) is True
    assert candidate_presence("q0", ("c1", "c2"), labels) is False
    with pytest.raises(ValueError):
        candidate_presence("missing", ("c0",), labels)
    with pytest.raises(ValueError):
        candidate_presence("q0", ("c0", "c0"), labels)


@pytest.mark.parametrize("defect", ["missing", "dangling", "contradiction", "scope", "hash"])
def test_bad_annotations_never_become_negatives(tmp_path, defect):
    from semantic_retrieval.data.benchmark_labels import load_benchmark_labels

    path, sha = snapshot(tmp_path)
    data = json.loads(path.read_text())
    if defect == "missing":
        data["annotations"] = data["annotations"][:1]
    elif defect == "dangling":
        data["annotations"][0]["official_video_id"] = "unlisted"
    elif defect == "contradiction":
        data["annotations"].append(dict(query_id="q0", caption_id="0", official_video_id="1", query_text="hello"))
    elif defect == "scope":
        data["origin_split"] = "mixed"
    newsha = write(path, data)
    with pytest.raises(ValueError):
        load_benchmark_labels(
            path,
            catalog=catalog(path, newsha),
            expected_sha256="0" * 64 if defect == "hash" else newsha,
            protocol_id="fixture",
        )


def test_aliases_merge_exact_content_not_semantic_neighbors(tmp_path):
    from semantic_retrieval.data.benchmark_labels import load_benchmark_labels

    items = list(catalog(tmp_path / "snapshot.json", None))
    items[1] = replace(items[1], canonical_id="c0", sha256=items[0].sha256)
    path, sha = snapshot(tmp_path)
    items = [replace(e, path=path, sha256=sha) if e.kind == "query" else e for e in items]
    labels = load_benchmark_labels(path, catalog=items, expected_sha256=sha, protocol_id="fixture")
    assert labels.official_to_canonical == {"0": "c0", "1": "c0", "2": "c2"}
    assert labels.positives_by_query["q0"] == ("c0",)


def test_snapshot_cannot_change_catalog_query_text_source(tmp_path):
    from semantic_retrieval.data.benchmark_labels import load_benchmark_labels

    path, sha = snapshot(tmp_path)
    items = list(catalog())
    for i, e in enumerate(items):
        if e.kind == "query":
            items[i] = replace(e, path=tmp_path / "different.json", sha256="9" * 64)
    with pytest.raises(ValueError, match="catalog"):
        load_benchmark_labels(path, catalog=items, expected_sha256=sha, protocol_id="fixture")


def test_source_annotation_identity_cannot_be_changed_in_snapshot(tmp_path):
    from semantic_retrieval.data.benchmark_labels import load_benchmark_labels

    path, sha = snapshot(tmp_path)
    data = json.loads(path.read_text())
    data["source_annotation_sha256"] = "9" * 64
    sha = write(path, data)
    with pytest.raises(ValueError, match="catalog"):
        load_benchmark_labels(path, catalog=catalog(path, sha), expected_sha256=sha, protocol_id="fixture")
