"""Filtering after a full JSON decode would expose sealed captions."""

import json

import pytest


def test_allowlist_gate_precedes_caption_decode_even_when_caption_field_comes_first():
    from semantic_retrieval.data.msrvtt_snapshot import training_annotations

    raw = b'{"meta":{"caption":"private"},"sentences":[{"caption":"private \\"{x}\\"","video_id":"test","sen_id":1},{"caption":"train \\u4e2d","video_id":"train","sen_id":2}]}'
    decoded = []

    def guarded(value):
        if b"private" in value:
            raise AssertionError("sealed caption decoded")
        decoded.append(value)
        return json.loads(value)

    rows, counts = training_annotations(raw, {"train"}, decode_caption=guarded)
    assert rows == [{"query_id": "train:2", "caption_id": "2", "official_video_id": "train", "query_text": "train 中"}]
    assert len(decoded) == 1 and counts["excluded_records"] == 1 and counts["decoded_excluded_caption_values"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sentences":[{"video_id":"train","caption":"one","sen_id":1},{"video_id":"train","caption":"two","sen_id":1}]}',
        b'{"sentences":[{"caption":"missing identity"}]}',
        b'{"sentences":[{"video_id":"train","caption":null}]}',
    ],
)
def test_missing_or_conflicting_caption_identity_is_not_silently_imported(raw):
    from semantic_retrieval.data.msrvtt_snapshot import training_annotations

    with pytest.raises(ValueError):
        training_annotations(raw, {"train"})
