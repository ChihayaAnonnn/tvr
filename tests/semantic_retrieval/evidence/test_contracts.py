import pytest
import torch
from dataclasses import replace
from semantic_retrieval.evidence.types import ResourceCounts, TransitionTargets
from semantic_retrieval.evidence.config import load_evidence_config

def test_negative_gain_class_is_not_unknown():
    t = TransitionTargets(torch.tensor([0, 1, 2]),
                          torch.tensor([True, False, True]), "a" * 64)
    assert t.outcome_class[0].item() == 0 and t.valid_mask[0].item()
    assert not t.valid_mask[1].item()
    with pytest.raises(ValueError):
        ResourceCounts(-1, 0, 0, 0)

def test_config_rejects_unknown_fields_and_boolean_counts(config_factory, tmp_path):
    cfg = config_factory()
    payload = cfg.to_payload()
    payload["extra"] = 1
    p = tmp_path / "config.json"
    import json
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_evidence_config(p)
    with pytest.raises(ValueError):
        replace(cfg.sampling, train_samples=True)

def test_config_roundtrip_binds_all_effective_fields(config_factory, tmp_path):
    import json
    from semantic_retrieval.evidence.config import evidence_manifest_config
    cfg = config_factory()
    p = tmp_path/'c.json'; p.write_text(json.dumps(cfg.to_payload()))
    restored = load_evidence_config(p)
    assert restored.to_payload() == cfg.to_payload()
    bound = evidence_manifest_config(cfg, eval_split='val')
    assert bound['evaluation']['sampling']['eval_samples'] == 4
    assert bound['evaluation']['policy']['limits']['frame_requests'] == 32


def test_state_rejects_union_duplicates_and_overspend(state_factory):
    from semantic_retrieval.evidence.types import BudgetLedger, ZERO
    s = state_factory()
    assert s.state_sha256 == replace(s).state_sha256
    with pytest.raises(ValueError):
        replace(s, candidates=(s.candidates[0], s.candidates[0]))
    with pytest.raises(ValueError):
        replace(s.candidates[0], attempted_frame_ids=(0,))
    with pytest.raises(ValueError):
        BudgetLedger(ResourceCounts(1, 1, 1, 1), ResourceCounts(2, 0, 0, 0), ZERO)


@pytest.mark.parametrize('field,value', [('temperature', float('nan')), ('eval_samples', 0), ('seed', True)])
def test_sampling_rejects_invalid_numbers(config_factory, field, value):
    with pytest.raises(ValueError):
        replace(config_factory().sampling, **{field: value})


@pytest.mark.parametrize('stage,weights',[('reference',{'pc':0.,'score':1.,'value':0.}),('value',{'pc':1.,'score':0.,'value':0.}),('matcher',{'pc':0.,'score':0.,'value':1.})])
def test_training_stage_rejects_unused_only_loss(config_factory,stage,weights):
    cfg = config_factory()
    with pytest.raises(ValueError,match='stage.*loss'):
        replace(cfg,stage=stage,train=replace(cfg.train,loss_weights=weights))
