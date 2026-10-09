import torch
from semantic_retrieval.evidence.model import EvidenceMatcherModel
from semantic_retrieval.evidence.training import freeze_reference, build_stage_optimizer

def test_stage_allowlist_and_reference_freeze(config_factory):
    cfg = config_factory()
    model = EvidenceMatcherModel(cfg.model)
    reference = freeze_reference(EvidenceMatcherModel(cfg.model))
    opt, _ = build_stage_optimizer(model, cfg.train)
    desired = {id(p) for n, p in model.named_parameters()
               if n.startswith(cfg.train.trainable_prefixes)}
    actual = {id(p) for g in opt.param_groups for p in g["params"]}
    assert actual == desired
    assert all(not p.requires_grad for p in model.clip.parameters())
    assert all(not p.requires_grad for p in reference.parameters())

def test_rich_teacher_frozen_raw_and_identical_view_invalid(config_factory, observed_features, rich_features):
    from semantic_retrieval.evidence.training import rich_score_targets
    reference = freeze_reference(EvidenceMatcherModel(config_factory().model))
    before = {k: v.clone() for k,v in reference.state_dict().items()}
    scores, valid = rich_score_targets(reference, rich_features, current_observed=observed_features)
    assert valid.tolist() == [True]*4 and not scores.requires_grad
    _, same = rich_score_targets(reference, observed_features, current_observed=observed_features)
    assert not same.any()
    assert all(torch.equal(v, before[k]) for k,v in reference.state_dict().items())
    assert all(p.grad is None for p in reference.parameters())


def test_matcher_losses_reach_loc_and_scale(config_factory, observed_features):
    from semantic_retrieval.evidence.training import matcher_loss
    from semantic_retrieval.evidence.sampling import SamplingState
    from semantic_retrieval.evidence.types import EvidenceTargets
    cfg = config_factory(); model = EvidenceMatcherModel(cfg.model)
    s = SamplingState(cfg.sampling, mode='eval', model_sha256='a'*64)
    pred = model(observed_features, sampling=s)
    targets = EvidenceTargets(torch.tensor([1,0,1,0]), torch.tensor([1.,2.,3.,4.]), torch.ones(4,dtype=torch.bool), 'b'*64)
    terms = matcher_loss(pred, targets, noise=s.noise(pred.pair_ids, device='cpu', dtype=torch.float32), config=cfg)
    assert terms['pc'].denominator == 2 and terms['score'].denominator == 4
    for term in terms.values():
        if term.denominator:
            grads = torch.autograd.grad(term.numerator, (pred.loc, pred.scale), retain_graph=True)
            assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_allowlist_rejects_clip_and_reference_scale(config_factory):
    from dataclasses import replace
    import pytest
    cfg = config_factory(); model = EvidenceMatcherModel(cfg.model)
    with pytest.raises(ValueError):
        build_stage_optimizer(model, replace(cfg.train, trainable_prefixes=('clip.',)))
    model.stage = 'reference'
    with pytest.raises(ValueError):
        build_stage_optimizer(model, replace(cfg.train, trainable_prefixes=('matcher.scale_head.',)))

def test_real_clip_frame_encoding_and_cache(config_factory, state_factory, inputs, tmp_path):
    from semantic_retrieval.evidence.frame_cache import FrameFeatureCache
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.types import QueryInputs
    model = EvidenceMatcherModel(config_factory().model)
    states = (state_factory(query_id='q0'), state_factory(query_id='q1'))
    def source(candidate, ids):
        return torch.stack([torch.full((3,8,8), i/16.) for i in ids])
    source.preprocessing_sha256 = 'b'*64
    out = model.encode_inputs(QueryInputs(inputs.input_ids, inputs.text_mask), states, source,
                              FrameFeatureCache(ArtifactArrayStore(tmp_path/'arrays')))
    assert out.features.frame_tokens.shape == (2,2,8)
    assert not out.features.frame_tokens.requires_grad
    assert out.pair_coverage.shape == (4,4)


def test_inference_timings_cover_temporal_and_mc_without_changing_scores(config_factory,observed_features):
    from semantic_retrieval.evidence.sampling import SamplingState
    model = EvidenceMatcherModel(config_factory().model).eval()
    sampling = SamplingState(config_factory().sampling,mode='eval',model_sha256='a'*64)
    expected = model(observed_features,sampling=sampling)
    timings = {}
    measured = model(observed_features,sampling=sampling,timings=timings)
    torch.testing.assert_close(expected.ranking_log_probability,measured.ranking_log_probability)
    assert timings['temporal_seconds'] > 0 and timings['mc_seconds'] > 0
