import pytest
import torch
from semantic_retrieval.evidence.calibration import fit_value_temperature
from semantic_retrieval.evidence.types import TransitionTargets


def test_calibration_only_held_out_role_and_exact_identity():
    logits = torch.tensor([[2.,0.,-2.], [-2.,0.,2.], [1.,0.,-1.]])
    targets = TransitionTargets(torch.tensor([0,2,2]), torch.ones(3,dtype=torch.bool), 'a'*64)
    identity = dict(producer_sha256='a'*64, model_sha256='b'*64, role='V_calibration',
                    relevance_space='reviewed_semantic', feature_schema_sha256='c'*64)
    for split in ('V_development', 'V_locked', 'test'):
        with pytest.raises(ValueError):
            fit_value_temperature(logits, targets, split=split, identity=identity)
    result = fit_value_temperature(logits, targets, split='V_calibration', identity=identity)
    out = result.apply(logits, expected_identity=identity)
    assert out.shape == (3,3) and result.temperature > 0
    before = torch.nn.functional.cross_entropy(logits, targets.outcome_class)
    assert torch.nn.functional.cross_entropy(out, targets.outcome_class) <= before+1e-6
    with pytest.raises(ValueError):
        result.apply(logits, expected_identity={**identity, 'model_sha256':'d'*64})
