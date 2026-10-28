"""Authenticated validation-only data, full-coverage paired inference and publication."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import compare
from .evaluator import grouped_interval
from .store import ArrayCache, canonical, digest, atomic


def summarize(queries, videos, original, reranked):
    gallery={v['video_id']:i for i,v in enumerate(videos)}
    targets=np.array([gallery[q['video_id']] for q in queries],dtype=np.int64)
    metrics=compare(original,reranked,targets)
    if metrics['baseline']['r10']!=metrics['reranked']['r10']:raise ValueError('R@10 invariant violated')
    delta=(reranked[:,0]==targets).astype(int)-(original[:,0]==targets).astype(int)
    groups=[videos[t]['source_component'] for t in targets]
    interval=grouped_interval(delta,groups)
    improvement=float(100*delta.mean())
    verdict='positive_feasibility' if improvement>=.5 and interval[0]>0 else ('inconclusive' if improvement>0 else 'no_improvement')
    return dict(metrics=metrics,delta_r1_pp=improvement,ci95_pp=list(interval),verdict=verdict,
                query_count=len(queries),video_count=len(videos),source_group_count=len(set(groups)),
                bootstrap_seed=713,bootstrap_samples=10000,checkpoint_selected_on_validation=True,
                independent_test_conclusion=False,test_authorized=False,direction='t2v',dsl=False,precision='fp32')


def merge_arrays(parts,count):
    if not parts:raise ValueError('no shards')
    shape=parts[0][1].shape[1:];dtype=parts[0][1].dtype
    result=np.empty((count,*shape),dtype=dtype);seen=np.zeros(count,dtype=bool)
    for indices,values in parts:
        ids=np.asarray(indices)
        if (ids.dtype.kind not in 'iu' or ids.ndim!=1 or len(ids)!=len(values) or len(set(ids.tolist()))!=len(ids)
                or (ids<0).any() or (ids>=count).any() or seen[ids].any() or values.shape[1:]!=shape or values.dtype!=dtype):
            raise ValueError('overlapping or invalid shard coverage')
        seen[ids]=True;result[ids]=values
    if not seen.all():raise ValueError('incomplete shard coverage')
    return result


def restore_blocks(source,target,identity):
    """Only committed, hash-valid blocks cross into a new reserved run directory."""
    if not (Path(source)/'identity.json').is_file():raise ValueError('resume source has no authenticated identity')
    src=ArrayCache(source,identity);dst=ArrayCache(target,identity);count=0
    for marker in sorted(Path(source).glob('*.json')):
        if marker.name=='identity.json':continue
        key=json.loads(marker.read_text())['key'];array=src.read(key)
        if array is None:continue
        dst.write(key,array);count+=1
    return count


def put_array(root,key,value):
    from experiment_artifacts import ArtifactArrayStore
    return ArtifactArrayStore(root/'arrays').write(key,np.asarray(value)).as_dict()


def load_array(root,reference):
    from experiment_artifacts import verify_array_reference
    verify_array_reference(root,reference)
    return np.load(root/reference['path'],allow_pickle=False)


def fixed_config(repo):
    from .protocol import CONTROL,committed
    return json.loads(committed(repo,repo/CONTROL/'fixed.json'))


def cohort(repo,fixed):
    from experiment_artifacts import strict_json_load,verify_array_reference
    from semantic_retrieval.full_data.final import authenticated_parent
    from semantic_retrieval.full_data.runtime import settings_from_manifest
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    from semantic_retrieval.full_data.loader import read_complete
    from .evaluator import validate_inputs
    parent,selected=authenticated_parent(repo,{'selected':[fixed['parent']]},fixed['parent']['run_id'])
    settings=settings_from_manifest(parent)
    videos,captions=read_complete(settings.source_report,settings.source_report_sha256,manifest_input_references(repo,parent))
    gallery=sorted(i for i,v in videos.items() if v['split']=='val')
    rows=[videos[i] for i in gallery]
    queries=[dict(query_id=q['query_id'],query_text=q['query_text'],video_id=q['official_video_id'],split='val') for q in captions['val']]
    parent_root=Path(fixed['parent']['checkpoint']['path']).parents[1]
    with (parent_root/'metrics/selected_validation.json').open() as f:archived=strict_json_load(f)
    reference=next(r for r in archived['array_references'] if r['key']=='epoch_5_scores')
    verify_array_reference(parent_root,reference)
    scores=np.load(parent_root/reference['path'],allow_pickle=False)
    if scores.shape!=(10000,500) or scores.dtype!=np.float32 or not np.isfinite(scores).all():raise ValueError('archived baseline score shape/precision mismatch')
    order=np.argsort(-scores,axis=1,kind='stable')
    validate_inputs(queries,rows,order,expected_queries=10000,expected_videos=500)
    target=np.array([gallery.index(q['video_id']) for q in queries])
    metrics=compare(order,order,target)['baseline']
    for key in ('r1','r5','r10'):
        if abs(metrics[key]-archived['summary_metrics']['t2v_'+key])>1e-9:raise ValueError('archived baseline mismatch')
    return dict(parent=parent,selected=selected,settings=settings,rows=rows,queries=queries,order=order,scores=scores,
                baseline_metrics=metrics,baseline_score_reference=reference,baseline_root=str(parent_root))


def inference_identity(repo,fixed):
    from .protocol import CONTROL
    files=('judge.py','core.py','evidence.py','evaluator.py','visual_cache.py','science.py','measurement.py')
    return dict(inventory=digest(repo/CONTROL/'external.input-index.json'),precision='fp32',config=fixed,
                inference_sha256=hashlib.sha256(canonical({name:digest(Path(__file__).with_name(name)) for name in files})).hexdigest(),
                environment_lock_sha256=digest(repo/CONTROL/'environment.lock.txt'))


def visual_identity(repo):
    from .protocol import CONTROL
    return {'inventory':digest(repo/CONTROL/'external.input-index.json'),'precision':'fp32'}


def video_key(row):
    from .core import evidence_indices
    from semantic_retrieval.full_data.loader import official_frame_indices
    indices=evidence_indices(official_frame_indices(row['frame_count'],row['fps'],12))
    return {'video_sha256':row['sha256'],'indices':indices}


def build_visual(repo,root,rows,judge):
    import time,torch
    from .visual_cache import VisualCache
    from .evidence import decode_evidence
    from semantic_retrieval.full_data.loader import official_frame_indices
    cache=VisualCache(root/'arrays/visual-cache',visual_identity(repo))
    torch.cuda.synchronize();started=time.perf_counter();metadata=[]
    for i,row in enumerate(rows):
        key=video_key(row);cache.get(key,lambda row=row:decode_evidence(row)[0],judge)
        original=official_frame_indices(row['frame_count'],row['fps'],12)
        metadata.append([len(key['indices']),len(original)-len(set(original)),*key['indices'],*([-1]*(4-len(key['indices'])))])
        if (i+1)%100==0:print(f'visual_cache={i+1}/{len(rows)}',flush=True)
    torch.cuda.synchronize()
    result=dict(seconds=time.perf_counter()-started,disk_bytes=sum(p.stat().st_size for p in cache.store.root.iterdir() if p.is_file()),
                cpu_payload_bytes=sum(v['features'].numel()*v['features'].element_size()+8*len(v['ids']) for v in cache.memory.values()),
                video_count=len(rows),frame_metadata=put_array(root,'visual_frame_metadata',np.asarray(metadata,dtype=np.int64)))
    atomic(root/'provenance/visual-cache.json',canonical(result))
    return result


def read_visual(repo,root,rows):
    import torch
    if not (root/'arrays/visual-cache/identity.json').is_file():raise ValueError('visual cache is not complete')
    cache=ArrayCache(root/'arrays/visual-cache',visual_identity(repo));memory={}
    for row in rows:
        key=video_key(row)
        features=cache.read({'video':key,'kind':'features'});ids=cache.read({'video':key,'kind':'prefix_ids'})
        if features is None or ids is None or features.dtype!=np.float32 or ids.dtype!=np.int64:raise ValueError('incomplete visual cache')
        memory[row['video_id']]={'features':torch.from_numpy(features),'ids':ids.tolist()}
    return memory


def query_array(root,kind):
    identity=json.loads((root/'identity.json').read_text());result=ArrayCache(root,identity).read({'kind':kind})
    if result is None:raise ValueError('missing complete query output')
    return result


def accuracy_worker(repo,root,m,data,device):
    import torch,torch.distributed as dist
    from experiment_artifacts import manifest_sha256
    from .judge import LayaJudge
    from .evaluator import evaluate
    fixed=fixed_config(repo);rank=dist.get_rank();world=dist.get_world_size()
    judge=LayaJudge(fixed['weights'],device=str(device))
    if rank==0:build_visual(repo,root,data['rows'],judge)
    dist.barrier()
    memory=read_visual(repo,root,data['rows']);indices=np.arange(rank,len(data['queries']),world,dtype=np.int64)
    out=root/'arrays/query-blocks'/f'rank{rank}';identity=inference_identity(repo,fixed)
    resume=m['config']['evaluation'].get('resume_from')
    if resume:
        source=Path(resume)/'arrays/query-blocks'/f'rank{rank}'
        if source.exists():
            previous=json.loads((source/'identity.json').read_text())
            if previous['model']!=identity:raise ValueError('resume inference identity changed')
            restore_blocks(source,out,previous)
    result=evaluate([data['queries'][i] for i in indices],data['rows'],data['order'][indices],judge.score,
                    lambda row:memory[row['video_id']],out,identity,
                    progress=lambda done:print(f'rank={rank} queries={done}/{len(indices)}',flush=True))
    atomic(root/f'provenance/partial-rank{rank}.json',canonical({'query_count':len(indices),'partial':True}))
    dist.barrier()
    if rank==0:
        order_parts=[];numeric_parts=[]
        for r in range(world):
            ids=np.arange(r,len(data['queries']),world,dtype=np.int64);path=root/'arrays/query-blocks'/f'rank{r}'
            order_parts.append((ids,query_array(path,'full_rankings')))
            numeric_parts.append((ids,query_array(path,'query_logits_probability')))
        merged=merge_arrays(order_parts,10000);numeric=merge_arrays(numeric_parts,10000)
        report=summarize(data['queries'],data['rows'],data['order'],merged)
        gallery={v['video_id']:i for i,v in enumerate(data['rows'])}
        target=np.array([gallery[q['video_id']] for q in data['queries']],dtype=np.int64)
        groups=[data['rows'][t]['source_component'] for t in target]
        names,inverse=np.unique(groups,return_inverse=True)
        delta=(merged[:,0]==target).astype(int)-(data['order'][:,0]==target).astype(int)
        sums=np.bincount(inverse,weights=delta);counts=np.bincount(inverse);rng=np.random.default_rng(713)
        draws=np.empty(10000,dtype=np.float64)
        for b in range(10000):
            selection=rng.integers(len(names),size=len(names));draws[b]=100*sums[selection].sum()/counts[selection].sum()
        if not np.array_equal(np.quantile(draws,[.025,.975]),report['ci95_pp']):raise ValueError('bootstrap archive mismatch')
        report['arrays']={k:put_array(root,k,a) for k,a in dict(baseline_order=data['order'],reranked_order=merged,
                            logits_probability=numeric,targets=target,source_groups=inverse,bootstrap_delta_r1_pp=draws).items()}
        report.update(source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),baseline_score_reference=data['baseline_score_reference'],
                      baseline_source='authenticated archived original seed713 epoch05 full-validation FP32 scores',
                      visual_cache=json.loads((root/'provenance/visual-cache.json').read_text()),performance_comparison_completed=False)
        atomic(root/'provenance/laya-validation.json',canonical(report))
        print(canonical({k:report[k] for k in ('metrics','delta_r1_pp','ci95_pp','verdict')}).decode(),flush=True)
    dist.barrier()
