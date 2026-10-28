import numpy as np
import pytest
from semantic_retrieval.laya_rerank.evaluator import evaluate, grouped_interval, validate_inputs


def fixture():
    videos = [dict(video_id=str(i), split='val', source_component=str(i)) for i in range(12)]
    queries = [dict(query_id='q'+str(i), query_text='description '+str(i), video_id=str(i), split='val') for i in range(4)]
    return queries, videos, np.tile(np.arange(12), (4, 1))


def test_resume_query_isolation_and_microbatch(tmp_path):
    q, v, order = fixture()
    calls = []
    def judge(texts, evidence):
        calls.extend(zip(texts, evidence))
        p = np.array([.9 if str(int(t.split()[-1])) == e else .1 for t,e in zip(texts,evidence)])
        return np.stack([p, 1-p], axis=1).astype('float32'), p
    def run(batch):
        return evaluate(q, v, order, judge, lambda video: video['video_id'], tmp_path, {'model':'sha'}, microbatch=batch)
    first = run(8)
    assert len(calls) == 40
    assert first['metrics']['reranked']['r1'] == 100
    second = run(1)
    assert len(calls) == 40 and first['metrics'] == second['metrics']
    changed = [dict(row) for row in q]; changed[0]['query_text'] = 'changed query'
    with pytest.raises(ValueError, match='identity'):
        evaluate(changed,v,order,judge,lambda x:x['video_id'],tmp_path,{'model':'sha'})


def test_incomplete_block_is_recomputed(tmp_path):
    q,v,a = fixture()
    count = [0]
    def interrupted(texts, frames):
        count[0] += 1
        if count[0] == 3: raise RuntimeError('interrupt')
        return np.zeros((len(texts),2),np.float32), np.full(len(texts),.5)
    with pytest.raises(RuntimeError): evaluate(q,v,a,interrupted,lambda x:x['video_id'],tmp_path,{'id':1})
    count[0] = 0
    def score(texts, frames):
        count[0] += len(texts)
        return np.zeros((len(texts),2),np.float32), np.full(len(texts),.5)
    result = evaluate(q,v,a,score,lambda x:x['video_id'],tmp_path,{'id':1})
    assert count[0] == 30
    assert result['metrics']['baseline'] == result['metrics']['reranked']


def test_reject_split_bad_output_and_coverage(tmp_path):
    q,v,a = fixture(); q[0]['split']='test'
    with pytest.raises(ValueError): validate_inputs(q,v,a)
    q[0]['split']='val'
    with pytest.raises(ValueError): validate_inputs(q,v,a,expected_queries=10000,expected_videos=500)
    with pytest.raises(ValueError):
        evaluate(q,v,a,lambda t,e:(np.zeros((len(t),2)), np.full(len(t),np.nan)),lambda x:x,tmp_path,{'id':1})


def test_grouped_bootstrap_uses_source_units():
    delta=np.array([1,1,-1,-1]); groups=['a','a','b','b']
    lo,hi=grouped_interval(delta,groups)
    assert lo == -100 and hi == 100
    assert grouped_interval(delta,groups) == (lo,hi)


def test_batch_and_independent_shards_match(tmp_path):
    from semantic_retrieval.laya_rerank.store import ArrayCache
    q,v,a=fixture()
    def score(texts,frames):
        p=np.array([.8 if str(int(t.split()[-1]))==e else .2 for t,e in zip(texts,frames)])
        return np.stack([p,1-p],axis=1).astype('float32'),p
    results=[]
    for name,batch in [('one',1),('eight',8)]:
        evaluate(q,v,a,score,lambda x:x['video_id'],tmp_path/name,{'id':1},microbatch=batch)
        # Read completion marker rather than assuming an unhashed output filename.
        import json
        marker=next(p for p in (tmp_path/name).glob('*.json') if p.name!='identity.json' and json.loads(p.read_text())['key']=={'kind':'full_rankings'})
        results.append(np.load(marker.with_suffix('.npy')))
    np.testing.assert_array_equal(*results)
    parts=[]
    for rank in range(2):
        indices=list(range(rank,len(q),2))
        path=tmp_path/str(rank)
        evaluate([q[i] for i in indices],v,a[indices],score,lambda x:x['video_id'],path,{'id':1})
        marker=next(p for p in path.glob('*.json') if p.name!='identity.json' and json.loads(p.read_text())['key']=={'kind':'full_rankings'})
        parts.append((indices,np.load(marker.with_suffix('.npy'))))
    merged=np.empty_like(results[0])
    for indices,ranks in parts:merged[indices]=ranks
    np.testing.assert_array_equal(merged,results[0])


def _process_shard(payload):
    import json
    q,v,a,path=payload
    def score(texts,evidence):
        p=np.array([.9 if str(int(t.split()[-1]))==e else .1 for t,e in zip(texts,evidence)])
        return np.column_stack((p,1-p)).astype('float32'),p
    evaluate(q,v,a,score,lambda row:row['video_id'],path,{'model':'synthetic'})
    marker=next(p for p in path.glob('*.json') if p.name!='identity.json' and json.loads(p.read_text())['key']=={'kind':'full_rankings'})
    return np.load(marker.with_suffix('.npy'))


def test_spawned_cpu_shards_merge(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    q,v,a=fixture()
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(_process_shard,([q[i] for i in range(rank,len(q),2)],v,a[rank::2],tmp_path/str(rank))) for rank in range(2)]
        parts=[job.result(timeout=40) for job in jobs]
    result=np.empty_like(a)
    result[::2],result[1::2]=parts
    expected=_process_shard((q,v,a,tmp_path/'all'))
    np.testing.assert_array_equal(result,expected)
