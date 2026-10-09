import pytest
from semantic_retrieval.evidence.runtime import prepare_evidence_run


def test_cli_runtime_cannot_be_omitted(config_factory,tmp_path):
    with pytest.raises(ValueError):
        prepare_evidence_run(config_factory(),repo_root=tmp_path)

def test_video_source_distinguishes_decode_failure_from_missing_input(tmp_path):
    import cv2
    import numpy as np
    import torch
    from semantic_retrieval.evidence.execution import VideoFrameSource, DecodeFailure
    from semantic_retrieval.evidence.types import CandidateObservation
    from hashlib import sha256
    path = tmp_path/'v.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(16,16))
    assert writer.isOpened(); writer.write(np.zeros((16,16,3),np.uint8)); writer.release()
    c = CandidateObservation('a',sha256(path.read_bytes()).hexdigest(),2,(0,1),(0,),(0,))
    source = VideoFrameSource({'a':(path,c.content_sha256)},8,'a'*64)
    assert source(c,(0,)).shape == (1,3,8,8)
    with pytest.raises(DecodeFailure):
        source(c,(1,))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        source(c,(0,))

def test_protected_evidence_context_rechecks_effective_config(config_factory,tmp_path):
    from dataclasses import replace
    from evidence.archive_fixture import reserve_evidence_fixture
    from semantic_retrieval.runtime.artifacts import start_run
    repo,cfg = reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    context = prepare_evidence_run(cfg,repo_root=repo)
    context.system_config = replace(cfg,sampling=replace(cfg.sampling,eval_samples=8))
    with pytest.raises(ValueError):
        start_run(context)
    context.system_config = cfg
    start_run(context)
    assert context.manifest['test_authorized'] is False

def test_prediction_budget_reserved_before_computation(state_factory,tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.runtime import run_policy_episode
    from semantic_retrieval.evidence.types import BudgetLedger,ResourceCounts,ZERO
    state = state_factory()
    state = replace(state,budget=BudgetLedger(ResourceCounts(0,0,0,0),ZERO,ZERO))
    calls = []
    def predict(s):
        calls.append(s)
        raise AssertionError('computation must not happen before budget reservation')
    models = SimpleNamespace(predict=predict,prediction_counts=lambda s:ResourceCounts(0,0,4,8),sampling_sha256='a'*64)
    store = ArtifactArrayStore(tmp_path/'arrays')
    journal = ActionJournal(store.run_root/'provenance'/'evidence-actions',store)
    with pytest.raises(ValueError):
        run_policy_episode(None,state,models,None,journal=journal)
    assert calls == []


def test_transition_input_rejects_different_frozen_matcher(config_factory,tmp_path):
    from dataclasses import replace,asdict
    from hashlib import sha256
    import json
    from semantic_retrieval.evidence.config import InputReference
    from semantic_retrieval.evidence.transitions import TransitionProducerIdentity
    from semantic_retrieval.evidence.execution import _read_transition_input
    producer = TransitionProducerIdentity(*(['a'*64]*8))
    path = tmp_path/'transitions.json'
    path.write_text(json.dumps(dict(producer=asdict(producer))))
    cfg = config_factory(stage='value')
    cfg = replace(cfg,transitions=replace(cfg.transitions,reference=InputReference(path,sha256(path.read_bytes()).hexdigest()),producer_sha256=producer.sha256))
    with pytest.raises(ValueError,match='frozen matcher'):
        _read_transition_input(cfg,matcher_sha256='b'*64)
