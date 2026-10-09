from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from semantic_retrieval.evidence.config import (
    EvidenceModelConfig, EvidencePolicyConfig, EvidenceSystemConfig, IndexConfig,
    InputReference, ParentConfig, SamplingConfig, StageTrainConfig, TransitionConfig,
    ValueConfig, ViewConfig,
)
from semantic_retrieval.evidence.types import (
    BudgetLedger, CandidateObservation, ExecutionProfile, ObservationState, ResourceCounts, ZERO,
)


@pytest.fixture
def config_factory(model_config):
    def make(stage='matcher', **overrides):
        ref = InputReference(model_config.clip_weights, model_config.clip_sha256)
        cfg = EvidenceSystemConfig(
            1, None, stage, True, EvidenceModelConfig(replace(model_config, max_frames=16), 'gaussian', .01),
            ViewConfig(16, (4, 8, 16), 'b'*64, 'opencv-rgb-v1'),
            SamplingConfig(4, 4, 1., 17, 'identity-normal-v1'),
            StageTrainConfig(.001, 0., 7, 'constant', 4, 1, ('temporal.', 'matcher.'), {'pc': 1., 'score': 1., 'value': 0.}),
            (), IndexConfig(None, 2, 'fixture-v1'), ParentConfig(ref, ref, ref, ref),
            TransitionConfig(ref, 'a'*64, 3, 2), ValueConfig(16, 'observable-action-v1', 1.),
            EvidencePolicyConfig(0., ResourceCounts(32, 32, 1024, 4096), (1., 1., 1., 1.)), None,
        )
        return replace(cfg, **overrides)
    return make


@pytest.fixture
def state_factory():
    def make(query_id='q0', canonical_ids=('a', 'b'), observed=(0, 15), frame_count=16):
        candidates = tuple(CandidateObservation(c, sha256(c.encode()).hexdigest(), frame_count,
                                               tuple(range(frame_count)), tuple(observed), tuple(observed)) for c in canonical_ids)
        return ObservationState(query_id, 'fixture-v1', candidates, (4, 8, 16),
                                ExecutionProfile(4, (1., 1., 1., 1.)),
                                BudgetLedger(ResourceCounts(32, 32, 1024, 4096), ZERO, ZERO), 0)
    return make


@pytest.fixture
def query_features(model, inputs):
    return model.encode_query(inputs)


@pytest.fixture
def frame_reader():
    def read(candidate, frame_ids):
        read.requests.append((candidate.canonical_id, frame_ids))
        return torch.tensor([[float(i), float(ord(candidate.canonical_id[0])), 1., 2., 3., 4., 5., 6.] for i in frame_ids])
    read.requests = []
    return read


@pytest.fixture
def observed_features(state_factory, query_features, frame_reader):
    from semantic_retrieval.evidence.encoding import pack_observations
    return pack_observations((state_factory(query_id='q0'), state_factory(query_id='q1')), query_features, frame_reader)


@pytest.fixture
def rich_features(state_factory, query_features, frame_reader):
    from semantic_retrieval.evidence.encoding import pack_observations
    return pack_observations((state_factory(query_id='q0', observed=tuple(range(16))),
                              state_factory(query_id='q1', observed=tuple(range(16)))), query_features, frame_reader)


@pytest.fixture
def journal_case(state_factory, tmp_path):
    from types import SimpleNamespace
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.policy import reserve_action, apply_outcome
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.types import ActionOutcome
    state = state_factory()
    action = propose_actions(state)[0]
    decision = reserve_action(state, action, predicted_counts=action.reservation)
    store = ArtifactArrayStore(tmp_path/'arrays')
    refs = tuple(store.write('frame_'+str(i), torch.arange(8.)+i) for i in action.frame_ids)
    outcome = ActionOutcome(action.action_id, action.frame_ids, (), action.reservation, .01, refs)
    return SimpleNamespace(state=state, decision=decision, outcome=outcome,
                           next_state=apply_outcome(state, action, outcome),
                           journal=ActionJournal(store.run_root/'provenance'/'evidence-actions', store))
