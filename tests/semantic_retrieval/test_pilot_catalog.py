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
    confirmation = tmp_path / "confirmation.json"
    source_ref = {"path": str(caption), "sha256": hashlib.sha256(caption.read_bytes()).hexdigest()}
    confirmation_sha = write(
        confirmation,
        {
            "status": {"local_annotation_matches_published_captions": "VERIFIED_EXACT_FOR_V2_TRAINING_ALLOWLIST_ONLY"},
            "local_assets": [source_ref],
            "training_caption_comparison": {
                "local_results": {"MSRVTT_v2.json": {"training_video_count": 1, "relation_sha256": "a" * 64}}
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
            "allowlist_source": source_ref,
            "caption_relation_sha256": "a" * 64,
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
