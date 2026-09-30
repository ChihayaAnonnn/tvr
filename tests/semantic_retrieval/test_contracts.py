import json
from dataclasses import replace

import pytest
import torch


def inputs():
    from semantic_retrieval.types import EncoderInputs

    return EncoderInputs(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool),
                         torch.zeros(3, 2, 3, 4, 4), torch.ones(3, 2, dtype=torch.bool))


def test_independent_query_and_video_counts_and_empty_candidates():
    from semantic_retrieval.types import validate_inputs

    batch = inputs()
    validate_inputs(batch)
    validate_inputs(replace(batch, video=batch.video[:0], frame_mask=batch.frame_mask[:0]))


@pytest.mark.parametrize("field,value", [
    ("text_mask", torch.zeros(2, 3, dtype=torch.bool)),
    ("text_mask", torch.ones(2, 3)),
    ("frame_mask", torch.zeros(3, 2, dtype=torch.bool)),
    ("frame_mask", torch.ones(2, 2, dtype=torch.bool)),
    ("video", torch.zeros(3, 2, 4, 4)),
    ("input_ids", torch.ones(2, 3)),
])
def test_contracts_reject_invalid_shapes_masks_and_id_dtype(field, value):
    with pytest.raises(ValueError):
        replace(inputs(), **{field: value})


def test_unknown_targets_are_preserved_and_invalid_values_rejected():
    from semantic_retrieval.types import Targets, validate_targets

    target = Targets(torch.tensor([[1, 0, -1], [0, -1, 0]], dtype=torch.int8),
                     torch.tensor([1, -1], dtype=torch.int8))
    validate_targets(target)
    assert target.relevance.tolist() == [[1, 0, -1], [0, -1, 0]]
    with pytest.raises(ValueError):
        replace(target, relevance=torch.tensor([[2, 0, -1], [0, -1, 0]]))
    with pytest.raises(ValueError):
        replace(target, candidate_presence=torch.tensor([1]))
    with pytest.raises(ValueError):
        replace(target, group_pairs=torch.tensor([[0, 1]]))


def test_features_have_no_target_fields():
    from semantic_retrieval.types import EncodedFeatures, QueryInputs

    query = QueryInputs(torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool))
    features = EncodedFeatures(torch.zeros(2, 3, 8), torch.zeros(2, 8), torch.zeros(3, 2, 8),
                               torch.zeros(3, 8), query.text_mask, torch.ones(3, 2, dtype=torch.bool))
    assert not hasattr(features, "relevance")
    assert not hasattr(features, "candidate_presence")


def test_runtime_config_rejects_unknown_keys_and_malformed_identity(tmp_path):
    from semantic_retrieval.config import load_runtime_config

    payload = dict(study_path="study.yaml", registry_path="registry.jsonl", run_dir="run",
                   run_id="fixture", manifest_sha256="a" * 64, source_sha="b" * 40, eval_split="val")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = load_runtime_config(path)
    assert config.run_dir == tmp_path / "run"
    assert config.eval_split == "val"
    for extra in ({"use_uatvr_full": True}, {"manifest_sha256": "invalid"}, {"source_sha": "HEAD"}):
        path.write_text(json.dumps(payload | extra))
        with pytest.raises(ValueError):
            load_runtime_config(path)
