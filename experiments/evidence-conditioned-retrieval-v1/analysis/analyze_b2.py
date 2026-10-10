"""Hash-verified, full-cohort B2 gate; no model inference or outcome-driven selection."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import sys,hashlib
REPO=Path(__file__).resolve().parents[3];C=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from archive_reader import certify_phase,require
ARTIFACT=Path('/data2/hxj/project/UATVR/artifacts')
def phase(name):return certify_phase(REPO,C,(C/'phases'/(name+'.json')).relative_to(REPO),(C/'summaries'/(name+'.json')).relative_to(REPO),ARTIFACT)
# Freeze no outcomes into the gate until all six producers are authenticated.
completed=phase('S3-G')+phase('S3-D')
parents={name:phase(name) for name in ('D0','S1','S2')}
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
from semantic_retrieval.evidence.checkpoint import evidence_config_sha256
from semantic_retrieval.config import config_payload
from experiment_artifacts import verify_array_reference,ArrayReference
from semantic_retrieval.evidence.data import read_split_binding
from semantic_retrieval.evidence.execution import bind_candidate_order
from semantic_retrieval.evidence.transitions import TransitionProducerIdentity,TransitionReferences,read_transitions
from semantic_retrieval.evidence.types import digest
from semantic_retrieval.evidence.schemas import validate_evidence_document
rows=[];details=['# B2 全组件覆盖与转移明细','','每个角色全部64个独立来源组件；presence为-1未知、0漏召回、1存在。零动作组件仍列出。','','| matcher | role | source component | presence | 动作总数 | corrected | unchanged | worsened | unknown |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for phase_name,variant in [('S3-G','gaussian'),('S3-D','deterministic')]:
 for role in ['T_value','V_development','V_calibration']:
  suffix=role.replace('_','-').lower();run=next(r for r in completed if r.manifest['variant']=='evidence-transitions-'+variant+'-'+suffix);root=run.root;status=run.status;r=dict(run_id=run.manifest['run_id'],manifest_sha256=__import__('experiment_artifacts').manifest_sha256(run.manifest))
  cfg=run.config()
  binding=next(s for s in cfg.splits if s.role==role);episodes=read_split_binding(binding);queries=[e.episode.query_id for e in episodes];components=[e.episode.query_component_id for e in episodes]
  require(len(episodes)==64 and len(set(components))==64)
  p=root/'provenance/transitions.json';index=run.read_json('provenance/transitions.json')
  parent=next(r for r in parents['S2'] if cfg.parents.matcher.path.is_relative_to(r.root));meta=parent.read_json(cfg.parents.matcher.path.with_suffix('.json').relative_to(parent.root).as_posix())
  producer=TransitionProducerIdentity(meta['model_sha256']['matcher'],evidence_config_sha256(cfg),cfg.index.reference.sha256,binding.manifest.sha256,digest(config_payload(cfg.views)),digest(config_payload(cfg.sampling)),digest(dict(seed=cfg.transitions.state_seed,trajectories=cfg.transitions.trajectories,role=role)),digest('observable-action-v1'))
  if index['producer']!=asdict(producer):raise ValueError('transition producer differs from independently certified frozen matcher/config')
  arrays=tuple(ArrayReference(**dict(a,shape=tuple(a['shape']))) for a in index['arrays']);batch=read_transitions(TransitionReferences(p,ref(p)['sha256'],producer,arrays),expected_producer=producer)
  cover=root/'provenance/transition-coverage.json';payload=run.read_json('provenance/transition-coverage.json');validate_evidence_document('transition_coverage',payload)
  require(status['output_hashes'][p.relative_to(root).as_posix()]==ref(p)['sha256'] and status['output_hashes'][cover.relative_to(root).as_posix()]==ref(cover)['sha256'])
  require(payload['cohort_complete'] is True and payload['query_count']==64 and payload['producer_sha256']==producer.sha256 and payload['transition_index_sha256']==ref(p)['sha256'])
  require(payload['role_manifest_sha256']==binding.manifest.sha256 and producer.label_sha256==binding.manifest.sha256)
  require(payload['query_order_sha256']==digest(queries) and payload['component_order_sha256']==digest(components))
  require(payload['index_specification_sha256']==cfg.index.reference.sha256 and producer.index_sha256==cfg.index.reference.sha256 and ref(cfg.index.reference.path)['sha256']==cfg.index.reference.sha256)
  gallery=json.loads(cfg.index.reference.path.read_text())['roles'][role];require(payload['gallery_order_sha256']==digest(gallery))
  actual={a['key']:verify_array_reference(root,a) for a in payload['array_references']};presence=actual['candidate_presence'];selected=actual['selected_gallery_indices']
  require(presence.shape==(64,) and selected.shape==(64,8) and np.isin(presence,[-1,0,1]).all())
  for e,ids,seen in zip(episodes,selected,presence):
   require(np.unique(ids).size==8 and (ids>=0).all() and (ids<len(gallery)).all())
   _,truth=bind_candidate_order(e.episode,tuple(gallery[int(i)] for i in ids));require(truth==int(seen))
  allowed={digest(dict(sources=(component,))):component for component in components};require(set(batch.component_ids)<=set(allowed))
  classes=batch.targets.outcome_class.numpy();valid=batch.targets.valid_mask.numpy();require(np.isin(classes,[0,1,2]).all())
  corrected={c for c,v,k in zip(batch.component_ids,valid,classes) if v and k==2};worsened={c for c,v,k in zip(batch.component_ids,valid,classes) if v and k==0}
  for component,seen in zip(components,presence):
   mask=np.asarray(batch.component_ids)==digest(dict(sources=(component,)))
   counts=[int((mask & valid & (classes==k)).sum()) for k in (2,1,0)]
   details.append('| '+variant+' | '+role+' | '+component+' | '+str(int(seen))+' | '+str(int(mask.sum()))+' | '+' | '.join(map(str,counts))+' | '+str(int((mask & ~valid).sum()))+' |')
  elapsed=run.read_json('provenance/controller-result.json')['elapsed_seconds']
  rows.append(dict(matcher=variant,role=role,run_id=r['run_id'],manifest_sha256=r['manifest_sha256'],transitions=ref(p),coverage=ref(cover),full_cohort_components=64,components_with_any_legal_transition=len(set(batch.component_ids)),components_without_legal_transition=64-len(set(batch.component_ids)),candidate_present_components=int((presence==1).sum()),candidate_miss_components=int((presence==0).sum()),candidate_unknown_components=int((presence==-1).sum()),transitions_count=len(classes),valid_transitions=int(valid.sum()),unknown_transitions=int((~valid).sum()),corrected_transitions=int(((classes==2)&valid).sum()),unchanged_transitions=int(((classes==1)&valid).sum()),worsened_transitions=int(((classes==0)&valid).sum()),corrected_components=len(corrected),worsened_components=len(worsened),allocated_gpu_hours=elapsed/3600))
main=next(r for r in rows if r['matcher']=='gaussian' and r['role']=='V_development');passed=main['corrected_components']>=8 and main['worsened_components']>=4
detail_path=C/'reports/B2-component-coverage.md';detail_path.write_text('\n'.join(details)+'\n')
result=dict(component_details=ref(detail_path),schema_version='evidence-stage-gate-v1',stage='B2-transition-variation',passed=passed,thresholds=dict(corrected_components=8,worsened_components=4),runs=rows,allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows),scientific_interpretation='Sufficient preregistered variation to test value learning; no effectiveness claim.' if passed else 'Insufficient transition variation in this frozen pilot; stop B3/B4 and scaling. This is not a universal refutation of evidence-conditioned retrieval.',V_locked_model_evaluated=False,test_authorized=False)
(C/'reports/B2-transition-gate.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# B2 冻结转移可检验性 gate','',f'passed={passed}；Gaussian Vdev corrected组件={main["corrected_components"]}，worsened组件={main["worsened_components"]}；预注册门槛8/4。','','| matcher | role | 完整组件 | 候选漏召回 | 转移 | corrected组件 | worsened组件 |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['matcher']+' | '+r['role']+' | 64 | '+str(r['candidate_miss_components'])+' | '+str(r['transitions_count'])+' | '+str(r['corrected_components'])+' | '+str(r['worsened_components'])+' |')
lines+=['',result['scientific_interpretation'],'','全部六臂完成后判定；哈希验证完整 cohort、实际 top-K、unknown、零动作组件。V_locked 未做模型评价，test sealed。']
(C/'reports/B2-transition-gate.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
