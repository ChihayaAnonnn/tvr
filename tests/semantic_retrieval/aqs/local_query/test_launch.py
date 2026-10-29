import copy
import pytest
from dataclasses import asdict
from semantic_retrieval.aqs.local_launch import validate_launch, require_qualification
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.full_data.campaign import FORMAL


def fixture():
    return dict(study='uatvr-aqs-local-query-training-v1',dataset='synthetic',test_authorized=False,
        environment=dict(gpu_count=2,precision='fp32'),config=dict(
            model=dict(architecture='uatvr-aqs-local-query-v1',clip=asdict(FORMAL),local=LocalQueryConfig(True,.2,.1).recipe(),query_chunk=8,video_chunk=8),
            loss=dict(aqs=AqsConfig(True,16,.2).recipe()),optimizer=dict(kind='pinned_bert_adam'),scheduler=dict(kind='warmup_cosine'),
            evaluation=dict(mode='qualify',dsl=False,training=dict(epochs=5,global_batch=64,learning_rate=5e-5,seed=713,world_size=2,negatives=16,weight=.2,validation_only=True))))


def test_launch_mode_and_local_contract():
    m=fixture();assert validate_launch(m,'qualify')['recipe']['global_batch']==64
    for change in ['test','mode','local','clip','aqs']:
        wrong=copy.deepcopy(m)
        if change=='test':wrong['test_authorized']=True
        if change=='mode':wrong['config']['evaluation']['mode']='train'
        if change=='local':wrong['config']['model']['local']['enabled']=False
        if change=='clip':wrong['config']['model']['clip']['fixture']=True
        if change=='aqs':wrong['config']['loss']['aqs']['weight']=0.
        with pytest.raises(ValueError):validate_launch(wrong,'qualify')


def test_science_gate_requires_matching_terminal_qualification():
    m=fixture();report=dict(passed=True,source_files={'a':'b'},recipe=validate_launch(m,'qualify')['recipe'],local=m['config']['model']['local'],world_size=2,synthetic=True,test_accessed=False)
    require_qualification(report,report['source_files'],report['recipe'],report['local'],'completed')
    for key,value in [('passed',False),('world_size',4),('synthetic',False),('source_files',{}),('local',{})]:
        wrong=dict(report,**{key:value})
        with pytest.raises(ValueError):require_qualification(wrong,report['source_files'],report['recipe'],report['local'],'completed')
    with pytest.raises(ValueError):require_qualification(report,report['source_files'],report['recipe'],report['local'],'running')


def test_qualification_evidence_must_match_terminal_inventory():
    from semantic_retrieval.aqs.local_launch import qualification_evidence
    report=dict(manifest_sha256='c'*64)
    qualification_evidence(report,'a'*64,'a'*64,'c'*64)
    with pytest.raises(ValueError):qualification_evidence(report,'a'*64,'b'*64,'c'*64)
    with pytest.raises(ValueError):qualification_evidence(report,'a'*64,'a'*64,'d'*64)
