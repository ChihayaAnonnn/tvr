"""Source checks must precede opening listed assets; components cannot cross splits."""

import hashlib
import json
from dataclasses import replace

import pytest


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fixture(tmp_path):
    video = tmp_path / "v.bin"
    video.write_bytes(b"video")
    caption = tmp_path / "q.json"
    caption.write_text('"caption"')
    allowlist = tmp_path / "allow.json"
    allowsha = write(allowlist, {"schema_version": 1, "origin_split": "source_train", "official_video_ids": ["v"]})
    trusted_ids = tmp_path / "source-ids.csv"
    trusted_ids.write_text("video_id\nv\n")
    trusted_ref = {"path": str(trusted_ids), "sha256": hashlib.sha256(trusted_ids.read_bytes()).hexdigest()}
    records = tmp_path / "training-records.json"
    records_sha = write(records, {"sentences": [{"video_id": "v", "caption": "caption", "sen_id": "0"}]})
    relation_sha = hashlib.sha256(b'["v","caption",1]\n').hexdigest()
    confirmation = tmp_path / "confirmation.json"
    source_ref = {"path": str(caption), "sha256": hashlib.sha256(caption.read_bytes()).hexdigest()}
    confirmation_sha = write(
        confirmation,
        {
            "status": {"local_annotation_matches_published_captions": "VERIFIED_EXACT_FOR_V2_TRAINING_ALLOWLIST_ONLY"},
            "local_assets": [source_ref, trusted_ref],
            "training_caption_comparison": {
                "local_results": {"MSRVTT_v2.json": {"training_video_count": 1, "relation_sha256": relation_sha}}
            },
        },
    )
    report = tmp_path / "source.json"
    reportsha = write(
        report,
        {
            "schema_version": 1,
            "origin_split": "source_train",
            "allowlist_sha256": allowsha,
            "protocol_id": "fixture",
            "source_verified": True,
            "source_confirmation": {"path": str(confirmation), "sha256": confirmation_sha},
            "source_annotation": source_ref,
            "allowlist_source": trusted_ref,
            "caption_relation_sha256": relation_sha,
            "training_records": {"path": str(records), "sha256": records_sha},
        },
    )
    rows = [
        dict(
            asset_id="v",
            kind="video",
            path=str(video),
            sha256=hashlib.sha256(b"video").hexdigest(),
            official_video_id="v",
            canonical_id="c",
            source_ids=["s"],
            caption_group_ids=["g"],
            frame_count=2,
            origin_split="source_train",
        ),
        dict(
            asset_id="q",
            kind="query",
            path=str(caption),
            sha256=hashlib.sha256(caption.read_bytes()).hexdigest(),
            official_video_id="v",
            canonical_id="c",
            source_ids=["s"],
            caption_group_ids=["g"],
            frame_count=None,
            origin_split="source_train",
        ),
    ]
    payload = dict(
        schema_version=1,
        origin_split="source_train",
        protocol_id="fixture",
        allowlist={"path": str(allowlist), "sha256": allowsha},
        source_report={"path": str(report), "sha256": reportsha},
        entries=rows,
    )
    path = tmp_path / "catalog.json"
    sha = write(path, payload)
    return path, sha, payload


def test_catalog_loads_only_bound_assets(tmp_path):
    from semantic_retrieval.data.pilot_catalog import load_pilot_catalog

    path, sha, _ = source_fixture(tmp_path)
    entries = load_pilot_catalog(
        path,
        expected_sha256=sha,
        allowed_root=tmp_path,
        source_confirmation={
            "path": str(tmp_path / "confirmation.json"),
            "sha256": hashlib.sha256((tmp_path / "confirmation.json").read_bytes()).hexdigest(),
        },
    )
    assert [e.asset_id for e in entries] == ["q", "v"]


@pytest.mark.parametrize("defect", ["test", "mixed", "unknown", "symlink", "escape", "hash", "allowlist", "identity"])
def test_invalid_identity_rejected_before_asset_read(tmp_path, defect):
    from semantic_retrieval.data.pilot_catalog import load_pilot_catalog

    path, sha, payload = source_fixture(tmp_path)
    if defect in {"test", "mixed", "unknown"}:
        payload["entries"][0]["origin_split"] = defect
    elif defect == "symlink":
        link = tmp_path / "link"
        link.symlink_to(tmp_path / "v.bin")
        payload["entries"][0]["path"] = str(link)
    elif defect == "escape":
        payload["entries"][0]["path"] = str(tmp_path.parent / "outside")
    elif defect == "hash":
        sha = "0" * 64
    elif defect == "allowlist":
        payload["entries"][0]["official_video_id"] = "not-authorized"
    elif defect == "identity":
        payload["entries"][1]["canonical_id"] = "different"
    # Metadata errors must win even when the first valid-looking asset is absent.
    (tmp_path / "v.bin").unlink()
    newsha = write(path, payload)
    with pytest.raises(ValueError):
        load_pilot_catalog(
            path,
            expected_sha256=sha if defect == "hash" else newsha,
            allowed_root=tmp_path,
            source_confirmation={
                "path": str(tmp_path / "confirmation.json"),
                "sha256": hashlib.sha256((tmp_path / "confirmation.json").read_bytes()).hexdigest(),
            },
        )


def entries(count=12):
    from semantic_retrieval.data.pilot_catalog import CatalogEntry

    return tuple(
        CatalogEntry(str(i), "video", None, f"{i:064x}", str(i), f"c{i}", (f"s{i}",), (f"g{i}",), 2, "source_train")
        for i in range(count)
    )


def test_component_partition_is_order_independent_and_query_sources_stay_together():
    from semantic_retrieval.data.pilot_catalog import assign_pilot_splits

    items = entries()
    query = replace(items[0], asset_id="q", kind="query", sha256="f" * 64, frame_count=None)
    targets = {
        "val_locked": 3,
        "val_calibration": 2,
        "val_development": 2,
        "train_min": 3,
        "canonical_min": 2,
        "locked_components_min": 2,
    }
    a = assign_pilot_splits(items + (query,), targets=targets)
    b = assign_pilot_splits(tuple(reversed(items + (query,))), targets=targets)
    assert a == b
    assert a.asset_splits["q"] == a.asset_splits["0"]
    assert [len(a.gallery_ids[s]) for s in ["val_locked", "val_calibration", "val_development", "train"]] == [
        3,
        2,
        2,
        5,
    ]
    assert a.asset_components["q"] == a.asset_components["0"]


def test_giant_component_does_not_get_split_to_meet_targets():
    from semantic_retrieval.data.pilot_catalog import assign_pilot_splits

    items = tuple(replace(e, source_ids=("shared",)) for e in entries())
    with pytest.raises(ValueError, match="quantity"):
        assign_pilot_splits(
            items,
            targets={
                "val_locked": 3,
                "val_calibration": 2,
                "val_development": 2,
                "train_min": 3,
                "canonical_min": 2,
                "locked_components_min": 2,
            },
        )


def test_self_signed_source_report_does_not_establish_training_identity(tmp_path):
    from semantic_retrieval.data.pilot_catalog import load_pilot_catalog

    path, sha, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="confirmation"):
        load_pilot_catalog(path, expected_sha256=sha, allowed_root=tmp_path)


def test_equal_sized_allowlist_cannot_replace_trusted_training_ids(tmp_path):
    from semantic_retrieval.data.pilot_catalog import load_pilot_catalog

    path, sha, payload = source_fixture(tmp_path)
    allow_path = tmp_path / "allow.json"
    new_allow = write(
        allow_path, dict(schema_version=1, origin_split="source_train", official_video_ids=["sealed-video"])
    )
    report_path = tmp_path / "source.json"
    report = json.loads(report_path.read_text())
    report["allowlist_sha256"] = new_allow
    payload["allowlist"]["sha256"] = new_allow
    payload["source_report"]["sha256"] = write(report_path, report)
    for row in payload["entries"]:
        row["official_video_id"] = "sealed-video"
    sha = write(path, payload)
    confirmation = tmp_path / "confirmation.json"
    with pytest.raises(ValueError, match="allowlist"):
        load_pilot_catalog(
            path,
            expected_sha256=sha,
            allowed_root=tmp_path,
            source_confirmation={
                "path": str(confirmation),
                "sha256": hashlib.sha256(confirmation.read_bytes()).hexdigest(),
            },
        )


def test_self_signed_caption_projection_must_match_independent_training_digest(tmp_path):
    from semantic_retrieval.data.pilot_catalog import load_pilot_catalog

    path, sha, payload = source_fixture(tmp_path)
    projected = tmp_path / "training-records.json"
    projected_sha = write(projected, {"sentences": [{"video_id": "v", "caption": "forged text", "sen_id": "0"}]})
    report_path = tmp_path / "source.json"
    report = json.loads(report_path.read_text())
    report["training_records"] = {"path": str(projected), "sha256": projected_sha}
    payload["source_report"]["sha256"] = write(report_path, report)
    sha = write(path, payload)
    confirmation = tmp_path / "confirmation.json"
    with pytest.raises(ValueError, match="caption"):
        load_pilot_catalog(
            path,
            expected_sha256=sha,
            allowed_root=tmp_path,
            source_confirmation={
                "path": str(confirmation),
                "sha256": hashlib.sha256(confirmation.read_bytes()).hexdigest(),
            },
        )
