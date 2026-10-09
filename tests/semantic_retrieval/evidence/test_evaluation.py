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


def test_report_externalizes_outcomes_and_scopes_calibration(tmp_path):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.evaluation import prediction_evidence
    targets = TransitionTargets(torch.tensor([0,2,1]),torch.tensor([True,True,False]),'a'*64)
    logits = torch.tensor([[10.,-10.,-10.],[-10.,-10.,10.],[0.,0.,0.]])
    report = prediction_evidence(store=ArtifactArrayStore(tmp_path/'arrays'),population=(logits,targets),selected=(logits,targets),
        rich=(torch.tensor([0.,0.]),torch.ones(2),torch.zeros(2),torch.ones(2,dtype=torch.bool)),
        before=[0.,1.],after=[1.,1.],components=('x','y'),seed=0)
    assert report.summary_metrics['population_valid_count'] == 2
    assert report.summary_metrics['selected_unknown_count'] == 1
    assert report.summary_metrics['selected_brier'] < 1e-12
    assert report.summary_metrics['rich_interval_coverage_95'] == 1
    assert report.summary_metrics['paired_component_count'] == 2
    assert len(report.array_references) >= 7
