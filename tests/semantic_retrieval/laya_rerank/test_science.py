import numpy as np
import pytest
from semantic_retrieval.laya_rerank.science import summarize, merge_arrays, restore_blocks
from semantic_retrieval.laya_rerank.measurement import sample_queries, latency_summary
from semantic_retrieval.laya_rerank.store import ArrayCache


def test_global_metrics_use_all_queries_and_keep_r10():
    q=[{'video_id':str(i%2)} for i in range(4)]
    v=[{'video_id':str(i),'source_component':str(i)} for i in range(12)]
    a=np.tile(np.arange(12),(4,1));b=a.copy();b[1,:2]=[1,0]
    r=summarize(q,v,a,b)
    assert r['query_count']==4 and r['metrics']['baseline']['r1']==50
    assert r['metrics']['reranked']['r1']==75 and r['delta_r1_pp']==25
    assert r['metrics']['baseline']['r10']==r['metrics']['reranked']['r10']


def test_merge_rejects_overlap_missing_and_reorders():
    a=np.arange(12).reshape(4,3)
    np.testing.assert_array_equal(merge_arrays([(np.array([2,0]),a[[2,0]]),(np.array([3,1]),a[[3,1]])],4),a)
    with pytest.raises(ValueError): merge_arrays([(np.array([0,0]),a[:2])],4)
    with pytest.raises(ValueError): merge_arrays([(np.array([0]),a[:1])],4)


def test_restore_reuses_only_authenticated_blocks(tmp_path):
    src=ArrayCache(tmp_path/'old',{'model':'a'});src.write({'q':'first'},np.ones((10,3)))
    path,marker=src.paths({'q':'partial'});np.save(path,np.zeros((2,3)))
    assert restore_blocks(tmp_path/'old',tmp_path/'new',{'model':'a'})==1
    dst=ArrayCache(tmp_path/'new',{'model':'a'})
    assert dst.read({'q':'first'}).shape==(10,3) and dst.read({'q':'partial'}) is None
    with pytest.raises(ValueError):restore_blocks(tmp_path/'old',tmp_path/'wrong',{'model':'b'})


def test_sampling_never_uses_correctness_and_latency_units():
    q=[{'query_id':str(i),'query_text':'x'} for i in range(200)]
    first=sample_queries(q)
    assert len(first)==100 and len(set(first))==100
    assert [q[i]['query_id'] for i in first]==[list(reversed(q))[i]['query_id'] for i in sample_queries(list(reversed(q)))]
    r=latency_summary(np.array([.1,.2,.3]))
    assert r['median_ms']==200 and r['p95_ms']==pytest.approx(290)
    assert r['throughput_queries_s']==pytest.approx(5)


def test_single_query_preserves_archived_pooling_mask_semantics():
    import torch
    from types import SimpleNamespace
    from semantic_retrieval.full_data.upstream_uatvr.modeling import UATVR
    from semantic_retrieval.laya_rerank.measurement import preserve_archived_mask_semantics
    for count in (1,8,32):
        core=SimpleNamespace(_mean_pooling_for_similarity_sequence=lambda seq,mask:UATVR._mean_pooling_for_similarity_sequence(None,seq,mask))
        model=SimpleNamespace(core=core)
        preserve_archived_mask_semantics(model)
        original=torch.ones(count,34);before=original.clone()
        result=core._mean_pooling_for_similarity_sequence(torch.ones(count,32,4),original[:,:32].contiguous())
        assert torch.equal(original,before) and torch.equal(result,torch.ones(count,4))


def test_gpu_monitor_keeps_owned_exited_pid_but_rejects_pid_reuse(monkeypatch):
    from semantic_retrieval.laya_rerank import science_controller as c
    monkeypatch.setattr(c,'owned_processes',lambda nonce:[])
    monkeypatch.setattr(c,'ownership_start_ticks',lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
    assert c.measurement_owners(123,'nonce','ticks')=={123}
    monkeypatch.setattr(c,'ownership_start_ticks',lambda pid:'reused')
    assert c.measurement_owners(123,'nonce','ticks')==set()


def test_gpu_monitor_accepts_unreaped_owned_zombie():
    import os,subprocess,sys
    from semantic_retrieval.laya_rerank import science_controller as c
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
    try:
        ticks=c.ownership_start_ticks(child.pid)
        os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
        assert child.pid in c.measurement_owners(child.pid,'unused-nonce',ticks)
    finally:child.wait()
