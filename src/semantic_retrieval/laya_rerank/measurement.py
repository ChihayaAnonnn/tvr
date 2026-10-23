"""Single-query online measurement; immutable hash selection and complete pipeline timing."""
import hashlib
import numpy as np


def sample_queries(queries,count=100):
    if len(queries)<count:raise ValueError('insufficient latency sample cohort')
    return sorted(range(len(queries)),key=lambda i:(hashlib.sha256(queries[i]['query_id'].encode()).digest(),queries[i]['query_id']))[:count]


def latency_summary(seconds):
    values=np.asarray(seconds,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<=0).any():raise ValueError('invalid latency samples')
    return dict(median_ms=float(np.median(values)*1000),p95_ms=float(np.quantile(values,.95)*1000),
                throughput_queries_s=float(len(values)/values.sum()),query_count=len(values))


def original_model(data,device):
    from types import SimpleNamespace
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.full_data.qualification import initial_clip,checkpoint_identity
    from semantic_retrieval.full_data.final import load_selected_model
    settings=data['settings'];parent=data['parent']
    model=BaselineModel('uatvr',settings.clip,initial_clip(settings),temporal_layers=4)
    identity=checkpoint_identity(SimpleNamespace(config=SimpleNamespace(source_sha=parent['git_sha'])),settings,'uatvr')
    identity['world_size']=parent['environment']['gpu_count']
    load_selected_model(data['selected']['selected_checkpoint']['path'],model,identity,expected_epoch=5)
    return model.to(device).eval().requires_grad_(False)


def prepare_original_cache(root,data,device):
    import time,torch
    from semantic_retrieval.full_data.loader import CaptionDataset
    from .science import put_array
    from .store import atomic,canonical
    model=original_model(data,device)
    q=data['queries'][0]
    captions=[dict(query_id=q['query_id'],query_text=q['query_text'],official_video_id=q['video_id'])]
    dataset=CaptionDataset({v['video_id']:v for v in data['rows']},captions)
    parts=[];torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,500,16):
            decoded=[dataset.decode(v['video_id']) for v in data['rows'][begin:begin+16]]
            values=model.encode_video(torch.stack([d[0] for d in decoded]).to(device),torch.stack([d[1] for d in decoded]).to(device))
            parts.append({k:v.cpu() for k,v in values.items()})
    torch.cuda.synchronize();compute_seconds=time.perf_counter()-started
    arrays={k:put_array(root,'original_'+k,torch.cat([p[k] for p in parts]).numpy()) for k in parts[0]}
    report=dict(seconds=time.perf_counter()-started,compute_seconds=compute_seconds,arrays=arrays,disk_bytes=sum((root/r['path']).stat().st_size for r in arrays.values()),
                cpu_payload_bytes=sum(r['byte_size'] for r in arrays.values()),video_count=500)
    atomic(root/'provenance/original-cache.json',canonical(report))
    return report


def process_rss_bytes():
    import os
    from pathlib import Path
    return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')


def measure_arm(repo,root,m,data,device,mode):
    import json,time,torch
    from pathlib import Path
    from semantic_retrieval.data.tokenizer import ClipTokenizer
    from .science import fixed_config,load_array,read_visual,put_array
    from .store import atomic,canonical
    from .core import rerank
    from .judge import LayaJudge
    arm,round_text=mode.split('-r');round_id=int(round_text)
    fixed=fixed_config(repo);accuracy=Path(m['config']['evaluation']['accuracy']['artifact_path'])
    torch.cuda.reset_peak_memory_stats()
    model=original_model(data,device)
    cache=json.loads((root/'provenance/original-cache.json').read_text())
    host={k:load_array(root,reference) for k,reference in cache['arrays'].items()}
    video={k:torch.from_numpy(v).to(device) for k,v in host.items()}
    judge=LayaJudge(fixed['weights'],device=str(device)) if arm=='enhanced' else None
    import gc
    gc.collect();before_rss=process_rss_bytes()
    visual=read_visual(repo,accuracy,data['rows']) if judge is not None else {}
    after_rss=process_rss_bytes()
    tokenizer=ClipTokenizer()
    torch.cuda.synchronize()
    resident=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    selected=np.asarray(sample_queries(data['queries']),dtype=np.int64)
    accuracy_report=json.loads((accuracy/'provenance/laya-validation.json').read_text())
    expected_enhanced=load_array(accuracy,accuracy_report['arrays']['reranked_order']) if judge else None
    archive_differences=[]
    @torch.inference_mode()
    def online(index):
        q=data['queries'][index]
        tokens,mask=tokenizer.encode_batch([q['query_text']],max_length=32)
        text=model.encode_text(tokens.to(device),mask.to(device))
        values=[]
        for start in range(0,500,64):
            values.append(model.score(text,{k:v[start:start+64] for k,v in video.items()}))
        scores=torch.cat(values,dim=1).float().cpu().numpy()[0]
        order=np.argsort(-scores,kind='stable')
        if judge is None:return order,scores,order
        probabilities=[]
        for start in (0,8):
            candidates=order[start:min(start+8,10)]
            _,p=judge.score([q['query_text']]*len(candidates),[visual[data['rows'][i]['video_id']] for i in candidates])
            probabilities.extend(p.tolist())
        result=rerank(order,np.asarray(probabilities))
        return result,scores,order
    def validate(index,result,base_order):
        if not np.array_equal(base_order[:10],data['order'][index,:10]):
            raise ValueError('single-query online baseline Top10 differs from archived baseline')
        if judge is not None and not np.array_equal(result[:10],expected_enhanced[index,:10]):
            raise ValueError('online enhanced Top10 differs from complete accuracy evaluation')
    assert_exclusive_gpu()
    for index in selected[:5]:
        result,_,base_order=online(int(index));validate(int(index),result,base_order)
    timings=[]
    for position,index in enumerate(selected):
        torch.cuda.synchronize();started=time.perf_counter()
        result,scores,base_order=online(int(index))
        torch.cuda.synchronize();timings.append(time.perf_counter()-started)
        validate(int(index),result,base_order)
        archive_differences.append(float(np.max(np.abs(scores-data['scores'][index]))))
        if (position+1)%25==0:
            assert_exclusive_gpu()
            print(f'measurement arm={arm} round={round_id} queries={position+1}/100',flush=True)
    report=dict(arm=arm,round=round_id,summary=latency_summary(np.asarray(timings)),query_batch=1,candidate_microbatch=8,
                resident_gpu=resident,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),cache_cpu_rss_delta_bytes=after_rss-before_rss,
                original_cache_cpu_payload_bytes=cache['cpu_payload_bytes'],
                laya_cache_cpu_payload_bytes=sum(v['features'].numel()*v['features'].element_size()+8*len(v['ids']) for v in visual.values()),
                max_single_query_vs_archived_score_abs_error=max(archive_differences),top10_matches_archived=True,
                timings=put_array(root,f'{arm}_round{round_id}_seconds',np.asarray(timings,dtype=np.float64)),
                selected_indices=put_array(root,f'{arm}_round{round_id}_indices',selected),test_authorized=False,
                timing_includes_tokenization_encoding_search_transfer_judge_sort=True,unrelated_gpu_processes_detected=False,
                physical_gpu_index=3)
    atomic(root/f'provenance/performance-{mode}.json',canonical(report))
    return report


def summarize_performance(root):
    import json
    from .science import load_array
    from .store import atomic,canonical
    reports={arm:[json.loads((root/f'provenance/performance-{arm}-r{r}.json').read_text()) for r in range(3)] for arm in ('baseline','enhanced')}
    for arm in reports:
        selected=[load_array(root,r['selected_indices']) for r in reports[arm]]
        if any(not np.array_equal(selected[0],s) for s in selected[1:]):raise ValueError('measurement cohort changed across rounds')
    if not np.array_equal(load_array(root,reports['baseline'][0]['selected_indices']),load_array(root,reports['enhanced'][0]['selected_indices'])):
        raise ValueError('measurement arms use different cohorts')
    result=dict(performance_comparison_completed=True,rounds=3,queries_per_round=100,warmup_queries_per_process=5,
                same_gpu=True,fresh_process_per_arm_round=True,alternating_arm_order=True,test_authorized=False)
    for arm,rows in reports.items():
        all_times=np.concatenate([load_array(root,r['timings']) for r in rows])
        result[arm]=dict(**latency_summary(all_times),
                        peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in rows),
                        peak_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in rows),
                        model_loaded_resident_gpu_allocated_bytes=max(r['resident_gpu']['allocated_bytes'] for r in rows),
                        model_loaded_resident_gpu_reserved_bytes=max(r['resident_gpu']['reserved_bytes'] for r in rows),
                        per_round=[r['summary'] for r in rows])
    result['relative_median_latency']=result['enhanced']['median_ms']/result['baseline']['median_ms']
    result['relative_p95_latency']=result['enhanced']['p95_ms']/result['baseline']['p95_ms']
    result['incremental_peak_gpu_allocated_bytes']=result['enhanced']['peak_gpu_allocated_bytes']-result['baseline']['peak_gpu_allocated_bytes']
    result['incremental_loaded_gpu_allocated_bytes']=result['enhanced']['model_loaded_resident_gpu_allocated_bytes']-result['baseline']['model_loaded_resident_gpu_allocated_bytes']
    result['original_cache']=json.loads((root/'provenance/original-cache.json').read_text())
    atomic(root/'provenance/laya-performance.json',canonical(result))
    return result


def assert_exclusive_gpu():
    import os,subprocess
    expected='GPU-8dab94cb-3e36-d6b4-61a8-a617b5dc7ede'
    text=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                        check=True,capture_output=True,text=True).stdout
    for line in text.splitlines():
        uuid,pid=(part.strip() for part in line.split(','))
        if uuid==expected and int(pid)!=os.getpid():raise ValueError('unrelated process on measurement GPU; performance comparison incomplete')
