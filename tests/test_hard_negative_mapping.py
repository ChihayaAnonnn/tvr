import json

import pytest

from dataloaders.hard_negative_mapping import load_hard_negative_index


def write_mapping(path, mapping):
    path.write_text(json.dumps({"mapping": mapping}), encoding="utf-8")


def test_load_hard_negative_index_skips_invalid_and_self_links(tmp_path):
    path = tmp_path / "hardneg.json"
    write_mapping(
        path,
        {
            "0": {"anchor_index": 0, "hard_index": 2},
            "1": {"anchor_index": 1, "hard_index": 1},
            "2": {"anchor_index": 2, "hard_index": 99},
            "4": {"anchor_index": 4, "hard_index": 3},
        },
    )

    hard_index = load_hard_negative_index(str(path), dataset_len=5)

    assert hard_index == [2, -1, -1, -1, 3]


def test_load_hard_negative_index_rejects_missing_mapping_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"items": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="no mapping object"):
        load_hard_negative_index(str(path), dataset_len=3)
