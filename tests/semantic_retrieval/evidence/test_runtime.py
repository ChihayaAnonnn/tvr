import pytest
from test_episodes import episode_manifest
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


def test_real_video_evaluation_partial_initial_decode_and_successor(config_factory,episode_manifest,tmp_path,monkeypatch):
    """Module integration on synthetic CPU inputs; this does not bypass a scientific CLI gate."""
    import json
    import torch
    from dataclasses import replace
    from types import SimpleNamespace
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.config import config_payload
    from semantic_retrieval.models.clip import ClipBackbone
    from semantic_retrieval.evidence.config import SplitBinding,InputReference,ParentConfig,EvidenceResumeConfig
    from semantic_retrieval.evidence.model import EvidenceMatcherModel
    from semantic_retrieval.evidence.execution import _evaluation
    from semantic_retrieval.evidence.types import digest
    from semantic_retrieval.data.episodes import file_sha256
    cfg = config_factory(stage='evaluation')
    clip = replace(cfg.model.base.clip,vocab_size=49408)
    weights = tmp_path/'bpe-clip.pt';torch.save(ClipBackbone(clip).state_dict(),weights)
    cfg = replace(cfg,model=replace(cfg.model,base=replace(cfg.model.base,clip=clip,clip_weights=weights,clip_sha256=file_sha256(weights))),
        policy=replace(cfg.policy,baseline='fixed_frames'),parents=ParentConfig(None,cfg.parents.matcher,None,None),
        transitions=replace(cfg.transitions,reference=None,producer_sha256=None))
    payload,write,_ = episode_manifest
    payload.update(schema_version=3,relevance_space='reviewed_semantic')
    row = payload['episodes'][0];row.update(split='val_development',query_source_ids=['query-source'],query_component_id='d'*64,frame_counts=[2,2])
    episodes = write()
    index = tmp_path/'index.json'
    index.write_text(json.dumps(dict(schema_version='evidence-initial-index-v1',clip_sha256=cfg.model.base.clip_sha256,
        preprocessing_sha256=cfg.views.preprocessing_sha256,view_sha256=digest(config_payload(cfg.views)),gallery_version='g1',roles={'V_development':['a','b']})))
    cfg = replace(cfg,splits=(SplitBinding('V_development',InputReference(episodes,file_sha256(episodes)),'reviewed_semantic'),),
        index=replace(cfg.index,reference=InputReference(index,file_sha256(index)),gallery_version='g1'))
    def context(name):
        store = ArtifactArrayStore(tmp_path/name/'arrays')
        return SimpleNamespace(array_store=store,paths=SimpleNamespace(root=store.run_root,provenance=store.run_root/'provenance'),
            config=SimpleNamespace(eval_split='val_development',source_sha='b'*40))
    from torch.multiprocessing import spawn
    training_cfg = replace(cfg,stage='reference',train=replace(cfg.train,loss_weights={'pc':1.,'score':0.,'value':0.}))
    training_path = tmp_path/'index-training-config.json';training_path.write_text(json.dumps(training_cfg.to_payload()))
    spawn(_real_index_worker,args=('file://'+str(tmp_path/'index-rendezvous'),str(training_path),str(tmp_path/'distributed-index')),nprocs=2,join=True)
    first = context('first'); model = EvidenceMatcherModel(cfg.model).eval()
    result = _evaluation(first,cfg,model,None)
    assert result['summary_metrics']['frame_requests'] == 4
    assert result['summary_metrics']['clip_frames'] == 2
    assert result['summary_metrics']['elapsed_seconds'] >= result['summary_metrics']['decode_seconds']
    report = json.loads((first.paths.provenance/'evidence-report.json').read_text())
    assert report['array_references'] and 'population_outcomes' in report['undefined_reasons']
    snapshots = list((first.paths.provenance/'evidence-recovery').glob('*.json'))
    snapshot = max(snapshots,key=lambda p:json.loads(p.read_text())['file_count'])
    resumed = replace(cfg,resume=EvidenceResumeConfig(InputReference(snapshot,file_sha256(snapshot)),0))
    second = context('successor')
    def repeated_decode(*args,**kwargs):
        raise AssertionError('successor repeated a restored frame decode')
    monkeypatch.setattr('semantic_retrieval.evidence.execution.decode_video',repeated_decode)
    actual = _evaluation(second,resumed,model,None)
    assert actual['summary_metrics']['frame_requests'] == 4
    assert actual['summary_metrics']['clip_frames'] == 2
    assert actual['summary_metrics']['decode_seconds'] == 0


def _real_index_worker(rank,rendezvous,config_path,run_root):
    import torch.distributed as dist
    from pathlib import Path
    from types import SimpleNamespace
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.config import load_evidence_config
    from semantic_retrieval.evidence.model import EvidenceMatcherModel
    from semantic_retrieval.evidence.execution import _episodes_and_sources
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        cfg = load_evidence_config(Path(config_path));model = EvidenceMatcherModel(cfg.model).eval()
        store = ArtifactArrayStore(Path(run_root)/'arrays')
        context = SimpleNamespace(array_store=store,paths=SimpleNamespace(provenance=store.run_root/'provenance'))
        episodes,_,_,cache = _episodes_and_sources(cfg,'V_development',model,context)
        assert len(episodes) == 1
        assert cache.store.run_root == store.run_root/'provenance'/f'index-rank-{rank}'
    finally:
        dist.destroy_process_group()
