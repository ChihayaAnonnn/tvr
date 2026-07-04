import pytest

from scripts.merge_msrvtt_model_mined_hard_negatives import merge_mapping_objects


def _part(anchor_start: int, count: int, num_samples: int):
    mapping = {}
    for offset in range(count):
        anchor = anchor_start + offset
        mapping[str(anchor)] = {
            "anchor_index": anchor,
            "anchor_video_id": f"video{anchor}",
            "hard_index": anchor + 100,
            "hard_video_id": f"video{anchor + 100}",
            "model_rank": 2,
            "positive_minus_hard": -0.5,
        }
    return {
        "meta": {"task": "msrvtt_model_mined_hard_negative", "query_start": anchor_start},
        "stats": {"num_samples": num_samples, "num_train_videos": 9000},
        "mapping": mapping,
    }


def test_merge_mapping_objects_combines_parts_and_recomputes_stats():
    merged = merge_mapping_objects([_part(0, 2, 3), _part(3, 2, 3)], source_paths=["p0.json", "p1.json"])

    assert list(merged["mapping"]) == ["0", "1", "3", "4"]
    assert merged["stats"]["num_samples"] == 6
    assert merged["stats"]["mapping_size"] == 4
    assert merged["stats"]["unmapped_count"] == 2
    assert merged["stats"]["num_train_videos"] == 9000
    assert merged["meta"]["task"] == "msrvtt_model_mined_hard_negative_merged"
    assert merged["meta"]["source_paths"] == ["p0.json", "p1.json"]


def test_merge_mapping_objects_rejects_duplicate_anchor_indices():
    with pytest.raises(ValueError, match="Duplicate anchor_index"):
        merge_mapping_objects([_part(0, 1, 1), _part(0, 1, 1)], source_paths=["p0.json", "p1.json"])
