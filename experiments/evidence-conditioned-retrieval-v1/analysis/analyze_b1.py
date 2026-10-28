"""Control-plane aggregation of completed, hash-verified B1 outputs only."""
import json,hashlib
from pathlib import Path
import numpy as np
import sys
REPO=Path(__file__).resolve().parents[3];C=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from archive_reader import certify_phase,require
ARTIFACT=Path('/data2/hxj/project/UATVR/artifacts')
def phase(name):return certify_phase(REPO,C,(C/'phases'/(name+'.json')).relative_to(REPO),(C/'summaries'/(name+'.json')).relative_to(REPO),ARTIFACT)
# Authenticate every required arm and parent phase before reading model results.
completed=phase('B1-R2')
for name in ('D0','S1','S2'):phase(name)
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
from experiment_artifacts import verify_array_reference
from semantic_retrieval.evidence.data import read_split_binding
from semantic_retrieval.evidence.types import digest
from semantic_retrieval.evidence.evaluation import paired_component_interval
rows=[];labels={};components=None
for variant in ['deterministic','fixed-scale','learned-temperature','standard-probabilistic','gaussian']:
 run=next(r for r in completed if r.manifest['variant']=='evidence-b1-'+variant+'-r2');root=run.root;r=dict(run_id=run.manifest['run_id'],manifest_sha256=__import__('experiment_artifacts').manifest_sha256(run.manifest));p=root/'provenance/evidence-report.json';report=run.read_json('provenance/evidence-report.json')
 cfg=run.config();ep=read_split_binding(next(s for s in cfg.splits if s.role=='V_development'));ids=[e.episode.query_id for e in ep];comps=tuple(e.episode.query_component_id for e in ep)
 require(len(ids)==len(set(ids))==len(set(comps))==64)
 require(report['identity']['episode_order_sha256']==digest(ids))
 if components is None:components=comps
 require(components==comps)
 arrays={a['key']:verify_array_reference(root,a) for a in report['array_references']}
 lab=arrays['episode_labels'];require(lab.shape==(64,3) and np.isin(lab,[0,1]).all());labels[variant]=lab
 index=run.read_json('provenance/initial-index-V_development.json');require(index['schema_version']=='evidence-initial-index-report-v2')
 elapsed=run.read_json('provenance/controller-result.json')['elapsed_seconds']
 rows.append(dict(variant=variant,run_id=r['run_id'],manifest_sha256=r['manifest_sha256'],report=ref(p),summary_metrics=report['summary_metrics'],initial_correct=int(lab[:,0].sum()),final_correct=int(lab[:,1].sum()),candidate_present=int(lab[:,2].sum()),initial_to_final_interval=paired_component_interval(lab[:,0],lab[:,1],comps,seed=713,samples=2000,confidence_level=.95),index_cost=index,controller_seconds=elapsed,allocated_gpu_hours=elapsed/3600))
for row in rows:
 lab=labels[row['variant']];base=labels['deterministic'];require(np.array_equal(lab[:,2],base[:,2]));row['final_delta_vs_deterministic_interval']=paired_component_interval(base[:,1],lab[:,1],components,seed=713,samples=2000,confidence_level=.95)
result=dict(schema_version='evidence-stage-gate-v1',stage='development-matcher-comparison',sample_scope='64 unique source components, one query each, 64-video gallery and frozen top8 candidates; not official test benchmark',runs=rows,allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows),bootstrap=dict(unit='query source component',samples=2000,seed=713,confidence=.95),model_selection='none: preregistered Gaussian and deterministic transition producers retained regardless of B1 accuracy',latency_caveat='Five concurrent workers share host CPU and I/O; timings are actual concurrent-run observations, not isolated method latency comparisons.',test_authorized=False)
p=C/'reports/B1-development-comparison.json';p.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
lines=['# B1 development validation 对照','',result['sample_scope'],'','| matcher | 初始正确/64 | 最终正确/64 | corrected | worsened | 帧请求 | CLIP帧 | temporal | MC元素 | rich NLL |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 m=r['summary_metrics'];lines.append('| '+r['variant']+' | '+str(r['initial_correct'])+' | '+str(r['final_correct'])+' | '+str(m['corrections'])+' | '+str(m['worsenings'])+' | '+str(m['frame_requests'])+' | '+str(m['clip_frames'])+' | '+str(m['temporal_tokens'])+' | '+str(m['mc_elements'])+' | '+str(m.get('rich_gaussian_nll','undefined'))+' |')
lines+=['','完整 SHA-256 引用、组件配对区间、索引离线成本与实际运行时间见同名 JSON。','不能把相同资源上限说成实际成本相同。训练额外监督差异不支持单独的 uncertainty 因果归因。','五个并发 worker 共享 CPU/I/O，延迟不用于孤立方法速度比较。','本表不执行模型选择；按预注册仍使用 Gaussian 和 deterministic 编译转移。','test sealed。']
(C/'reports/B1-development-comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
