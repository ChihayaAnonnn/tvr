"""Panel truth and frozen manifests are generated offline without semantic labels."""

import pytest

from semantic_retrieval.data.benchmark_labels import BenchmarkLabels


def labels(size=129):
    ids = {f"v{i:03}": f"c{i:03}" for i in range(size)}
    return BenchmarkLabels(
        "fixture",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        {"q": ("c032",)},
        ids,
        {"q": ("v032",)},
        {"q": "hello"},
        {"q": "0"},
        "d" * 64,
        "e" * 64,
        "f" * 64,
    )


def test_controlled_panels_have_exact_truth_and_canonical_membership():
    from semantic_retrieval.data.pilot_compile import build_benchmark_panel

    label = labels()
    order = tuple(label.official_to_canonical.values())
    present = build_benchmark_panel(query_id="q", canonical_order=order, k=32, mode="present", labels=label)
    absent = build_benchmark_panel(query_id="q", canonical_order=order, k=32, mode="absent", labels=label)
    assert len(present.candidate_ids) == 32 and present.answer_in_candidates is True
    assert len(absent.candidate_ids) == 32 and absent.answer_in_candidates is False
    assert "c032" in present.candidate_ids and "c032" not in absent.candidate_ids
    for k, want in [(32, False), (64, True)]:
        miss = build_benchmark_panel(query_id="q", canonical_order=order, k=k, mode="controlled_miss", labels=label)
        assert miss.answer_in_candidates is want
        assert miss.ranked_ids[32:33] == ("c032",)
    assert (
        build_benchmark_panel(
            query_id="q", canonical_order=order, k=32, mode="natural", labels=label
        ).answer_in_candidates
        is False
    )


def test_absent_128_requires_129_distinct_candidates():
    from semantic_retrieval.data.pilot_compile import build_benchmark_panel

    for n in [128, 129]:
        label = labels(n)
        order = tuple(label.official_to_canonical.values())
        if n == 128:
            with pytest.raises(ValueError):
                build_benchmark_panel(query_id="q", canonical_order=order, k=128, mode="absent", labels=label)
        else:
            panel = build_benchmark_panel(query_id="q", canonical_order=order, k=128, mode="absent", labels=label)
            assert len(panel.candidate_ids) == 128 and not panel.answer_in_candidates


@pytest.mark.parametrize("defect", ["duplicate", "missing", "foreign", "zero"])
def test_invalid_order_or_capacity_rejected(defect):
    from semantic_retrieval.data.pilot_compile import build_benchmark_panel

    label = labels()
    order = list(label.official_to_canonical.values())
    k = 4
    if defect == "duplicate":
        order[-1] = order[0]
    elif defect == "missing":
        order.pop()
    elif defect == "foreign":
        order[-1] = "foreign"
    elif defect == "zero":
        k = 0
    with pytest.raises(ValueError):
        build_benchmark_panel(query_id="q", canonical_order=order, k=k, mode="present", labels=label)


def test_compile_read_and_joint_split_audit(tmp_path):
    import json
    from dataclasses import replace

    import cv2
    import numpy as np

    from semantic_retrieval.data.benchmark_labels import load_benchmark_labels
    from semantic_retrieval.data.episodes import EpisodePreprocessing, audit_episode_splits, file_sha256, read_episodes
    from semantic_retrieval.data.pilot_catalog import CatalogEntry, assign_pilot_splits
    from semantic_retrieval.data.pilot_compile import compile_pilot_manifests, summarize_pilot_inputs

    catalog = []
    rows = []
    for i in range(12):
        path = tmp_path / f"{i}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
        writer.write(np.full((16, 16, 3), i * 20, np.uint8))
        writer.release()
        video = CatalogEntry(
            f"v{i}", "video", path, file_sha256(path), f"v{i}", f"c{i}", (f"s{i}",), (f"g{i}",), 1, "source_train"
        )
        catalog += [video, replace(video, asset_id=f"q{i}", kind="query", frame_count=None)]
        rows.append(dict(query_id=f"q{i}", caption_id=str(i), official_video_id=f"v{i}", query_text=f"caption {i}"))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            dict(
                schema_version=1,
                origin_split="source_train",
                protocol_id="fixture",
                source_annotation_sha256="a" * 64,
                source_report_sha256="b" * 64,
                allowlist_sha256="c" * 64,
                annotations=rows,
            )
        )
    )
    projected = tmp_path / "training-records.json"
    projected.write_text(
        json.dumps(
            dict(
                sentences=[
                    dict(video_id=r["official_video_id"], sen_id=r["caption_id"], caption=r["query_text"]) for r in rows
                ]
            )
        )
    )
    catalog = [
        replace(
            e,
            path=snapshot,
            sha256=file_sha256(snapshot),
            source_report_sha256="b" * 64,
            allowlist_sha256="c" * 64,
            source_annotation_sha256="a" * 64,
            training_records_path=projected,
            training_records_sha256=file_sha256(projected),
        )
        if e.kind == "query"
        else e
        for e in catalog
    ]
    label = load_benchmark_labels(
        snapshot, catalog=catalog, expected_sha256=file_sha256(snapshot), protocol_id="fixture"
    )
    refs = {}
    for name, value in [("relations", label.official_positives_by_query), ("id_map", label.official_to_canonical)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        refs[name] = dict(path=str(path), sha256=file_sha256(path))
    targets = dict(
        val_locked=3, val_calibration=3, val_development=3, train_min=3, canonical_min=2, locked_components_min=2
    )
    assignment = assign_pilot_splits(catalog, targets=targets)
    summary = summarize_pilot_inputs(catalog=catalog, assignment=assignment, labels=label, targets=targets)
    assert summary["ready"]
    manifests = compile_pilot_manifests(
        catalog=catalog,
        assignment=assignment,
        labels=label,
        input_summary=summary,
        preprocessing=EpisodePreprocessing(32, 8, 224),
        label_references=refs,
    )
    all_episodes = []
    for split, payload in manifests.items():
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(payload["episodes"]))
        result = read_episodes(path, gallery_version=payload["episodes"]["gallery_version"], split=split)
        assert len(result) == 3 and all(e.answer_in_gallery == 1 and len(e.benchmark_positive_ids) == 1 for e in result)
        all_episodes.extend(result)
    audit_episode_splits(all_episodes)
    # Query source participates even when its GT was omitted in a controlled panel.
    bad = replace(all_episodes[0], query_source_ids=all_episodes[-1].query_source_ids)
    with pytest.raises(ValueError, match="source"):
        audit_episode_splits([bad, all_episodes[-1]])
    incomplete = replace(assignment, gallery_ids={**assignment.gallery_ids, "train": ()})
    report = summarize_pilot_inputs(catalog=catalog, assignment=incomplete, labels=label, targets=targets)
    assert not report["ready"] and report["stop_reasons"]
