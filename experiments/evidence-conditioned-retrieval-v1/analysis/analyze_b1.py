"""Control-plane aggregation of completed, hash-verified B1 outputs only."""
import json,hashlib
from pathlib import Path
import numpy as np
import sys,hashlib
REPO=Path(__file__).resolve().parents[3];C=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
records=[json.loads(line) for line in (C/'registry.jsonl').read_text().splitlines()]
def terminal(pattern):
 result=[r for r in records if r['event']=='completed' and pattern in r['run_id']]
 if len(result)!=1:raise ValueError('one completed producer required: '+pattern)
 return result[0]
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
H={'terminal':terminal}
from experiment_artifacts import verify_array_reference
from semantic_retrieval.evidence.data import read_split_binding
from semantic_retrieval.evidence.config import load_evidence_config
from semantic_retrieval.evidence.types import digest
from semantic_retrieval.evidence.evaluation import paired_component_interval
rows=[];labels={};components=None
for variant in ['deterministic','fixed-scale','learned-temperature','standard-probabilistic','gaussian']:
 r=H['terminal']('-evidence-b1-'+variant+'-');root=Path(r['artifact_path']);p=root/'provenance/evidence-report.json';report=json.loads(p.read_text());st=json.loads((root/'status.json').read_text());assert st['output_hashes'][p.relative_to(root).as_posix()]==ref(p)['sha256']
 cfg=load_evidence_config(C/'configs'/('B1-evidence-b1-'+variant+'.json'));ep=read_split_binding(next(s for s in cfg.splits if s.role=='V_development'));ids=[e.episode.query_id for e in ep];comps=tuple(e.episode.query_component_id for e in ep)
 assert report['identity']['episode_order_sha256']==digest(ids)
 if components is None:components=comps
 assert components==comps
 arrays={a['key']:verify_array_reference(root,a) for a in report['array_references']}
 lab=arrays['episode_labels'];assert lab.shape==(64,3) and np.isin(lab,[0,1]).all();labels[variant]=lab
 index=json.loads((root/'provenance/initial-index-V_development.json').read_text());assert index['schema_version']=='evidence-initial-index-report-v2'
 elapsed=json.loads((root/'provenance/controller-result.json').read_text())['elapsed_seconds']
 rows.append(dict(variant=variant,run_id=r['run_id'],manifest_sha256=r['manifest_sha256'],report=ref(p),summary_metrics=report['summary_metrics'],initial_correct=int(lab[:,0].sum()),final_correct=int(lab[:,1].sum()),candidate_present=int(lab[:,2].sum()),initial_to_final_interval=paired_component_interval(lab[:,0],lab[:,1],comps,seed=713,samples=2000,confidence_level=.95),index_cost=index,controller_seconds=elapsed,allocated_gpu_hours=elapsed/3600))
for row in rows:
 lab=labels[row['variant']];base=labels['deterministic'];assert np.array_equal(lab[:,2],base[:,2]);row['final_delta_vs_deterministic_interval']=paired_component_interval(base[:,1],lab[:,1],components,seed=713,samples=2000,confidence_level=.95)
result=dict(schema_version='evidence-stage-gate-v1',stage='development-matcher-comparison',sample_scope='64 unique source components, one query each, 64-video gallery and frozen top8 candidates; not official test benchmark',runs=rows,allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows),bootstrap=dict(unit='query source component',samples=2000,seed=713,confidence=.95),model_selection='none: preregistered Gaussian and deterministic transition producers retained regardless of B1 accuracy',latency_caveat='Five concurrent workers share host CPU and I/O; timings are actual concurrent-run observations, not isolated method latency comparisons.',test_authorized=False)
p=C/'reports/B1-development-comparison.json';p.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
lines=['# B1 development validation 对照','',result['sample_scope'],'','| matcher | 初始正确/64 | 最终正确/64 | corrected | worsened | 帧请求 | CLIP帧 | temporal | MC元素 | rich NLL |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 m=r['summary_metrics'];lines.append('| '+r['variant']+' | '+str(r['initial_correct'])+' | '+str(r['final_correct'])+' | '+str(m['corrections'])+' | '+str(m['worsenings'])+' | '+str(m['frame_requests'])+' | '+str(m['clip_frames'])+' | '+str(m['temporal_tokens'])+' | '+str(m['mc_elements'])+' | '+str(m.get('rich_gaussian_nll','undefined'))+' |')
lines+=['','完整 SHA-256 引用、组件配对区间、索引离线成本与实际运行时间见同名 JSON。','不能把相同资源上限说成实际成本相同。训练额外监督差异不支持单独的 uncertainty 因果归因。','五个并发 worker 共享 CPU/I/O，延迟不用于孤立方法速度比较。','本表不执行模型选择；按预注册仍使用 Gaussian 和 deterministic 编译转移。','test sealed。']
(C/'reports/B1-development-comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
