import torch
from semantic_retrieval.evidence.evaluation import rank_candidates, score_outcomes
from semantic_retrieval.evidence.types import TransitionTargets

def test_ties_and_permutations_have_same_canonical_order():
    assert rank_candidates(torch.zeros(3), ("b", "c", "a")) == ("a", "b", "c")
    assert rank_candidates(torch.zeros(3), ("c", "a", "b")) == ("a", "b", "c")

def test_perfect_outcomes_have_zero_brier():
    logits = torch.tensor([[30., -30., -30.], [-30., -30., 30.]])
    targets = TransitionTargets(torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool), "a"*64)
    assert score_outcomes(logits, targets)["brier"] < 1e-12

def test_duplicate_captions_do_not_add_bootstrap_units():
    from semantic_retrieval.evidence.evaluation import paired_component_interval
    a = paired_component_interval([0.,0.,0.], [1.,1.,1.], ('x','x','x'),seed=0,samples=100,confidence_level=.95)
    assert a['component_count'] == 1 and a['defined'] is False
    b = paired_component_interval([0.,0.], [1.,-1.], ('x','y'),seed=0,samples=100,confidence_level=.95)
    c = paired_component_interval([0.,0.,0.,0.], [1.,1.,-1.,-1.], ('x','x','y','y'),seed=0,samples=100,confidence_level=.95)
    assert b == c
