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


def test_distribution_expected_gain_mse_matches_regression(tmp_path):
    import pytest
    from experiment_artifacts import ArtifactArrayStore,verify_array_reference
    from semantic_retrieval.evidence.evaluation import prediction_evidence
    targets=TransitionTargets(torch.tensor([2,0,1,0]),torch.tensor([True,False,True,True]),'a'*64)
    logits=torch.tensor([[0.,1.,2.],[3.,1.,0.],[1.,2.,0.],[1.,0.,2.]])
    gain=logits.double().softmax(-1)[:,2]-logits.double().softmax(-1)[:,0]
    expected=float(((gain[targets.valid_mask]-(targets.outcome_class[targets.valid_mask].double()-1))**2).mean())
    common=dict(selected=None,rich=None,before=[],after=[],components=(),seed=0)
    classification=prediction_evidence(store=ArtifactArrayStore(tmp_path/'class'/'arrays'),population=(logits,targets),**common)
    regression_store=ArtifactArrayStore(tmp_path/'mean'/'arrays')
    regression=prediction_evidence(store=regression_store,population=None,mean_population=(gain,targets),**common)
    assert classification.summary_metrics['population_gain_mse']==pytest.approx(expected)
    assert regression.summary_metrics['population_gain_mse']==pytest.approx(expected)
    assert regression.summary_metrics['population_valid_count']==3
    assert regression.summary_metrics['population_unknown_count']==1
    assert 'population_nll' not in regression.summary_metrics and 'population_brier' not in regression.summary_metrics
    assert regression.undefined_reasons['population_categorical_scores']=='mean regression does not provide categorical probabilities'
    arrays={a.key:verify_array_reference(regression_store.run_root,a.as_dict()) for a in regression.array_references}
    assert arrays['population_mean_gain'].shape==(4,)


def test_mean_evidence_without_valid_labels_is_undefined(tmp_path):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.evaluation import prediction_evidence
    target=TransitionTargets(torch.tensor([1]),torch.tensor([False]),'a'*64)
    report=prediction_evidence(store=ArtifactArrayStore(tmp_path/'arrays'),population=None,selected=None,rich=None,
        mean_selected=(torch.tensor([1.]),target),before=[],after=[],components=(),seed=0)
    assert report.summary_metrics['selected_valid_count']==0
    assert 'selected_gain_mse' not in report.summary_metrics
    assert report.undefined_reasons['selected_gain_mse']=='no known action outcomes'


def test_mean_evidence_rejects_nonfinite_and_ambiguous_scope(tmp_path):
    import pytest
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.evaluation import prediction_evidence
    target=TransitionTargets(torch.tensor([1]),torch.tensor([True]),'a'*64)
    common=dict(store=ArtifactArrayStore(tmp_path/'arrays'),selected=None,rich=None,before=[],after=[],components=(),seed=0)
    with pytest.raises(ValueError):prediction_evidence(population=None,mean_population=(torch.tensor([float('nan')]),target),**common)
    with pytest.raises(ValueError):prediction_evidence(population=(torch.zeros(1,3),target),mean_population=(torch.zeros(1),target),**common)
