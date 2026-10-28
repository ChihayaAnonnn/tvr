import torch
from semantic_retrieval.evidence.types import ActionValueDistribution, TransitionTargets
from semantic_retrieval.evidence.objectives import value_nll

def test_value_retains_harm_and_uses_weight_sum():
    probs = torch.tensor([[.6, .3, .1], [.1, .2, .7]])
    out = ActionValueDistribution(probs.log())
    torch.testing.assert_close(out.expected_gain, torch.tensor([-.5, .6]))
    target = TransitionTargets(torch.tensor([0, 2]),
                               torch.tensor([True, True]), "a" * 64)
    loss = value_nll(probs.log(), target, importance_weights=torch.tensor([1., 3.]))
    assert loss.denominator.item() == 4
    torch.testing.assert_close(loss.numerator, -probs[0,0].log() - 3*probs[1,2].log())

def test_value_step_cannot_update_matcher(config_factory, state_factory, observed_features):
    from semantic_retrieval.evidence.model import EvidenceMatcherModel
    from semantic_retrieval.evidence.value import EvidenceValueModel, ValueHead, build_value_features
    from semantic_retrieval.evidence.sampling import SamplingState
    from semantic_retrieval.evidence.views import propose_actions
    cfg = config_factory()
    matcher = EvidenceMatcherModel(cfg.model)
    model = EvidenceValueModel(matcher, ValueHead(matcher.matcher.feature_dim+19, 16))
    s = state_factory()
    from dataclasses import replace
    observed = replace(observed_features, features=replace(observed_features.features,
                       text_tokens=observed_features.features.text_tokens[:1], pooled_text=observed_features.features.pooled_text[:1],
                       text_mask=observed_features.features.text_mask[:1]),
                       pair_indices=observed_features.pair_indices[:2], query_offsets=torch.tensor([0,2]),
                       pair_ids=observed_features.pair_ids[:2], pair_coverage=observed_features.pair_coverage[:2])
    pred = matcher(observed, sampling=SamplingState(cfg.sampling,mode='eval',model_sha256='a'*64))
    f = build_value_features(s, propose_actions(s), pred)
    assert f.shape == (2, matcher.matcher.feature_dim+19) and not f.requires_grad
    out = model.value_head(f)
    out.logits.sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in matcher.parameters())
    assert all(p.grad is not None for p in model.value_head.parameters())

def test_invalid_importance_weights_rejected_and_unknown_ignored():
    import pytest
    from semantic_retrieval.evidence.objectives import mean_gain_mse
    targets = TransitionTargets(torch.tensor([0,2]), torch.tensor([True,False]), 'a'*64)
    logits = torch.zeros(2,3,requires_grad=True)
    term = value_nll(logits, targets)
    term.numerator.backward()
    assert not logits.grad[1].any()
    assert mean_gain_mse(torch.tensor([-1.,9.]), targets).numerator == 0
    for bad in (torch.tensor([-1.,1.]), torch.tensor([float('nan'),1.])):
        with pytest.raises(ValueError):
            value_nll(logits, targets, importance_weights=bad)
