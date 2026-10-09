from semantic_retrieval.evidence.transitions import label_transition

def test_three_outcomes_and_unknown_are_distinct():
    truth = {"good": 1, "bad": 0, "unknown": -1}
    cases = [("good", "bad", 0), ("bad", "bad", 1), ("bad", "good", 2)]
    for before, after, expected in cases:
        t = label_transition(before, after, truth, producer_sha256="a" * 64)
        assert t.valid_mask.item() and t.outcome_class.item() == expected
    t = label_transition("unknown", "good", truth, producer_sha256="a" * 64)
    assert not t.valid_mask.item()

def test_all_negative_candidates_are_retained():
    t = label_transition('a', 'b', {'a':0,'b':0}, producer_sha256='a'*64)
    assert t.valid_mask.item() and t.outcome_class.item() == 1

def test_compile_real_frozen_matcher_and_verify_arrays(config_factory, state_factory, query_features, frame_reader, tmp_path):
    from dataclasses import replace
    import pytest
    import torch
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.encoding import pack_observations
    from semantic_retrieval.evidence.model import EvidenceMatcherModel
    from semantic_retrieval.evidence.sampling import SamplingState
    from semantic_retrieval.evidence.training import freeze_reference
    from semantic_retrieval.evidence.transitions import FrozenTransitionModel, TransitionProducerIdentity, compile_transitions, read_transitions
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.runtime.checkpoint import model_sha256
    cfg = config_factory(); model = freeze_reference(EvidenceMatcherModel(cfg.model))
    producer = TransitionProducerIdentity(model_sha256(model), *(['a'*64]*7))
    sampling = SamplingState(cfg.sampling, mode='eval', model_sha256=producer.matcher_sha256)
    query = replace(query_features, text_tokens=query_features.text_tokens[:1], pooled_text=query_features.pooled_text[:1], text_mask=query_features.text_mask[:1])
    def predict(state):
        return model(pack_observations((state,), query, frame_reader), sampling=sampling)
    def observe(state, action):
        rows = tuple(replace(c, observed_frame_ids=tuple(sorted(set(c.observed_frame_ids)|set(action.frame_ids))),
                             attempted_frame_ids=tuple(sorted(set(c.attempted_frame_ids)|set(action.frame_ids))))
                     if c.canonical_id == action.candidate_id else c for c in state.candidates)
        return replace(state, candidates=rows, version=state.version+1)
    adapter = FrozenTransitionModel(model, predict, observe, lambda state, actions, scores: scores.features[:len(actions)].detach())
    refs = compile_transitions((state_factory(),), propose_actions, adapter, {'q0':({'a':0,'b':0}, ('source0',))},
                               store=ArtifactArrayStore(tmp_path/'arrays'), producer_identity=producer)
    batch = read_transitions(refs, expected_producer=producer)
    assert len(batch.features) == 2
    assert batch.targets.outcome_class.tolist() == [1,1]
    assert batch.targets.valid_mask.all()
    with pytest.raises(ValueError):
        read_transitions(refs, expected_producer=replace(producer, config_sha256='b'*64))
    p = tmp_path/refs.arrays[0].path; p.write_bytes(b'damaged')
    with pytest.raises(ValueError):
        read_transitions(refs, expected_producer=producer)
