from dataclasses import replace
import pytest
import torch
from semantic_retrieval.evidence.baselines import build_matcher_variant, build_initial_index, search_initial_candidates
from semantic_retrieval.retrieval.index import IndexIdentity
from semantic_retrieval.evidence.views import view_frame_ids
from semantic_retrieval.evidence.sampling import SamplingState


def test_initial_index_never_reads_extra_frames(state_factory, frame_reader, query_features):
    s = state_factory()
    s = replace(s, candidates=tuple(replace(c, observed_frame_ids=view_frame_ids(c.pool_frame_ids,4), attempted_frame_ids=view_frame_ids(c.pool_frame_ids,4)) for c in s.candidates))
    identity = IndexIdentity('a'*64, 'fixture-v1', 'b'*64)
    index = build_initial_index((s,), frame_reader, identity=identity)
    assert all(len(ids) == 4 for _,ids in frame_reader.requests)
    result = search_initial_candidates(query_features,index,2,expected_identity=identity)
    assert len(result) == 2 and all(set(row) == {'a','b'} for row in result)
    c = replace(s.candidates[0], observed_frame_ids=tuple(range(16)), attempted_frame_ids=tuple(range(16)))
    with pytest.raises(ValueError):
        build_initial_index((replace(s,candidates=(c,)),),frame_reader,identity=identity)


def test_variants_have_real_distinct_prediction_paths(config_factory, observed_features):
    cfg = config_factory()
    sampling = SamplingState(cfg.sampling, mode='eval', model_sha256='a'*64)
    gaussian = build_matcher_variant(cfg,variant='gaussian')
    assert gaussian.config.variant == 'gaussian'
    for variant in ('deterministic','fixed_scale','learned_temperature','standard_probabilistic'):
        model = build_matcher_variant(cfg,variant=variant)
        output = model(observed_features,sampling=sampling)
        assert torch.isfinite(output.ranking_log_probability).all()
        if variant in {'deterministic','learned_temperature','fixed_scale'}:
            assert all(not p.requires_grad for p in model.matcher.scale_head.parameters())
        if variant in {'deterministic','learned_temperature'}:
            assert torch.all(output.scale == 1)

def test_learned_temperature_receives_matching_gradient(config_factory, observed_features):
    from semantic_retrieval.evidence.training import matcher_loss
    from semantic_retrieval.evidence.types import EvidenceTargets
    cfg = config_factory()
    cfg = replace(cfg,model=replace(cfg.model,variant='learned_temperature'))
    model = build_matcher_variant(cfg,variant='learned_temperature')
    sampling = SamplingState(cfg.sampling,mode='eval',model_sha256='a'*64)
    output = model(observed_features,sampling=sampling)
    target = EvidenceTargets(torch.tensor([1,0,1,0]),torch.zeros(4),torch.ones(4,dtype=torch.bool),'a'*64)
    terms = matcher_loss(output,target,noise=torch.zeros(4,4),config=cfg)
    terms['pc'].numerator.backward()
    assert model.log_temperature.grad is not None and model.log_temperature.grad.abs() > 0

def test_candidate_selection_does_not_insert_annotated_positive():
    from semantic_retrieval.evidence.execution import bind_candidate_order
    from types import SimpleNamespace
    episode = SimpleNamespace(canonical_ids=('good','bad'),relevance=(1,0),relevance_space='reviewed_semantic',benchmark_positive_ids=())
    labels,presence = bind_candidate_order(episode,('bad',))
    assert labels == (0,) and presence == 0
    labels,presence = bind_candidate_order(episode,('unreviewed',))
    assert labels == (-1,) and presence == -1


def test_config_selects_real_value_head_and_rejects_mismatched_policy(config_factory):
    from semantic_retrieval.evidence.value import build_value_head,MeanRegressor,GeometryValue
    cfg = config_factory()
    mean = replace(cfg.value,kind='mean_regressor')
    assert isinstance(build_value_head(mean,23),MeanRegressor)
    assert isinstance(build_value_head(replace(cfg.value,kind='geometry'),23),GeometryValue)
    with pytest.raises(ValueError,match='policy/value'):
        replace(cfg,value=mean)


def test_random_baseline_is_replayable_and_candidate_order_independent(config_factory,state_factory):
    from semantic_retrieval.evidence.baselines import build_acquisition_baseline
    cfg = config_factory(); state = state_factory()
    policy = build_acquisition_baseline('random',seed=19,config=cfg)
    first = policy(state,None).action.candidate_id
    assert policy(state,None).action.candidate_id == first
    other = replace(state,candidates=tuple(reversed(state.candidates)))
    assert policy(other,None).action.candidate_id == first


def test_fixed_frames_advances_least_observed_candidate_first(config_factory,state_factory):
    from semantic_retrieval.evidence.baselines import build_acquisition_baseline
    state = state_factory()
    policy = build_acquisition_baseline('fixed_frames',seed=19,config=config_factory())
    first = policy(state,None).action
    candidate = next(c for c in state.candidates if c.canonical_id == first.candidate_id)
    updated = replace(candidate, observed_frame_ids=tuple(sorted(set(candidate.observed_frame_ids+first.frame_ids))),
                      attempted_frame_ids=tuple(sorted(set(candidate.attempted_frame_ids+first.frame_ids))))
    state = replace(state,candidates=tuple(updated if c.canonical_id == updated.canonical_id else c for c in state.candidates),version=state.version+1)
    assert policy(state,None).action.candidate_id != first.candidate_id


def test_initial_index_accepts_only_successful_initial_subset(state_factory,frame_reader):
    state = state_factory()
    initial = view_frame_ids(state.candidates[0].pool_frame_ids,4)
    candidates = tuple(replace(c,observed_frame_ids=initial[1:],attempted_frame_ids=initial) for c in state.candidates)
    state = replace(state,candidates=candidates)
    index = build_initial_index((state,),frame_reader,identity=IndexIdentity('a'*64,'fixture-v1','b'*64))
    assert len(index.ids) == 2
    assert all(ids == initial[1:] for _,ids in frame_reader.requests)
