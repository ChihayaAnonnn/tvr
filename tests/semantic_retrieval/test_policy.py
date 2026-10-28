from dataclasses import replace

import pytest
import torch

from semantic_retrieval.types import PairScores


@pytest.fixture
def policy():
    from semantic_retrieval.config import PolicyConfig
    from semantic_retrieval.retrieval.policy import RetrievalPolicy

    return RetrievalPolicy(PolicyConfig(0.7, 0.6, (2, 4, 8), 8))


def semantic(ids, probabilities):
    from semantic_retrieval.retrieval.semantic_set import SemanticSet

    p = torch.tensor(probabilities)
    return SemanticSet.build(ids, PairScores(torch.logit(p), torch.zeros(len(ids), 4), None))


def test_bounded_expansion_and_no_global_absence_claim(policy):
    s = semantic(["a"], [0.1])
    first = policy.decide(torch.tensor([0.1]), 0.1, s, attempted_k=[2], remaining_budget=6)
    assert (first.action, first.next_k) == ("expand", 4)
    end = policy.decide(torch.tensor([0.1]), 0.1, s, attempted_k=[2, 4, 8], remaining_budget=0)
    assert end.action == "abstain" and end.next_k is None and "gallery" not in end.reason
    assert policy.decide(torch.empty(0), 0.1, semantic([], []), attempted_k=[2], remaining_budget=0).action == "abstain"


def test_multiple_answers_and_conflict(policy):
    s = semantic(["b", "a"], [0.9, 0.8])
    d = policy.decide(torch.tensor([0.8, 0.9]), 0.9, s, attempted_k=[2], remaining_budget=6)
    assert d.action == "return" and d.answer_ids == ("a", "b")
    assert not s.conflict_signal
    assert (
        policy.decide(
            torch.tensor([0.8, 0.9]), 0.9, replace(s, conflict_signal=True), attempted_k=[2], remaining_budget=6
        ).action
        == "clarify"
    )


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_invalid_probabilities(policy, value):
    with pytest.raises(ValueError):
        policy.decide(torch.tensor([value]), 0.5, semantic(["a"], [0.5]), attempted_k=[2], remaining_budget=1)


def test_alias_and_permutation_stability():
    a = semantic(["b", "a", "b"], [0.9, 0.8, 0.9])
    b = semantic(["a", "b"], [0.8, 0.9])
    assert a == b
    with pytest.raises(ValueError):
        semantic(["a", "a"], [0.8, 0.9])
