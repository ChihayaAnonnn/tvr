"""Hash-verified, full-cohort B2 gate; no model inference or outcome-driven selection."""
import json
from pathlib import Path
from dataclasses import asdict
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
from experiment_artifacts import verify_array_reference,ArrayReference
from semantic_retrieval.evidence.config import load_evidence_config
from semantic_retrieval.evidence.data import read_split_binding
from semantic_retrieval.evidence.execution import bind_candidate_order
from semantic_retrieval.evidence.transitions import TransitionProducerIdentity,TransitionReferences,read_transitions
from semantic_retrieval.evidence.types import digest
from semantic_retrieval.evidence.schemas import validate_evidence_document
rows=[]
for phase,variant in [('S3-G','gaussian'),('S3-D','deterministic')]:
 for role in ['T_value','V_development','V_calibration']:
  suffix=role.replace('_','-').lower();r=H['terminal']('-evidence-transitions-'+variant+'-'+suffix+'-');root=Path(r['artifact_path']);status=json.loads((root/'status.json').read_text())
  cfg=load_evidence_config(C/'configs'/(phase+'-evidence-transitions-'+variant+'-'+suffix+'.json'))
  binding=next(s for s in cfg.splits if s.role==role);episodes=read_split_binding(binding);queries=[e.episode.query_id for e in episodes];components=[e.episode.query_component_id for e in episodes]
  assert len(episodes)==64 and len(set(components))==64
  p=root/'provenance/transitions.json';index=json.loads(p.read_text());producer=TransitionProducerIdentity(**index['producer'])
  arrays=tuple(ArrayReference(**dict(a,shape=tuple(a['shape']))) for a in index['arrays']);batch=read_transitions(TransitionReferences(p,ref(p)['sha256'],producer,arrays),expected_producer=producer)
  cover=root/'provenance/transition-coverage.json';payload=json.loads(cover.read_text());validate_evidence_document('transition_coverage',payload)
  assert status['output_hashes'][p.relative_to(root).as_posix()]==ref(p)['sha256'] and status['output_hashes'][cover.relative_to(root).as_posix()]==ref(cover)['sha256']
  assert payload['cohort_complete'] is True and payload['query_count']==64 and payload['producer_sha256']==producer.sha256 and payload['transition_index_sha256']==ref(p)['sha256']
  assert payload['role_manifest_sha256']==binding.manifest.sha256 and producer.label_sha256==binding.manifest.sha256
  assert payload['query_order_sha256']==digest(queries) and payload['component_order_sha256']==digest(components)
  assert payload['index_specification_sha256']==cfg.index.reference.sha256 and producer.index_sha256==cfg.index.reference.sha256 and ref(cfg.index.reference.path)['sha256']==cfg.index.reference.sha256
  gallery=json.loads(cfg.index.reference.path.read_text())['roles'][role];assert payload['gallery_order_sha256']==digest(gallery)
  actual={a['key']:verify_array_reference(root,a) for a in payload['array_references']};presence=actual['candidate_presence'];selected=actual['selected_gallery_indices']
  assert presence.shape==(64,) and selected.shape==(64,8) and np.isin(presence,[-1,0,1]).all()
  for e,ids,seen in zip(episodes,selected,presence):
   assert np.unique(ids).size==8 and (ids>=0).all() and (ids<len(gallery)).all()
   _,truth=bind_candidate_order(e.episode,tuple(gallery[int(i)] for i in ids));assert truth==int(seen)
  allowed={digest(dict(sources=(component,))):component for component in components};assert set(batch.component_ids)<=set(allowed)
  classes=batch.targets.outcome_class.numpy();valid=batch.targets.valid_mask.numpy();assert np.isin(classes,[0,1,2]).all()
  corrected={c for c,v,k in zip(batch.component_ids,valid,classes) if v and k==2};worsened={c for c,v,k in zip(batch.component_ids,valid,classes) if v and k==0}
  elapsed=json.loads((root/'provenance/controller-result.json').read_text())['elapsed_seconds']
  rows.append(dict(matcher=variant,role=role,run_id=r['run_id'],manifest_sha256=r['manifest_sha256'],transitions=ref(p),coverage=ref(cover),full_cohort_components=64,components_with_any_legal_transition=len(set(batch.component_ids)),components_without_legal_transition=64-len(set(batch.component_ids)),candidate_present_components=int((presence==1).sum()),candidate_miss_components=int((presence==0).sum()),candidate_unknown_components=int((presence==-1).sum()),transitions_count=len(classes),valid_transitions=int(valid.sum()),unknown_transitions=int((~valid).sum()),corrected_transitions=int(((classes==2)&valid).sum()),unchanged_transitions=int(((classes==1)&valid).sum()),worsened_transitions=int(((classes==0)&valid).sum()),corrected_components=len(corrected),worsened_components=len(worsened),allocated_gpu_hours=elapsed/3600))
main=next(r for r in rows if r['matcher']=='gaussian' and r['role']=='V_development');passed=main['corrected_components']>=8 and main['worsened_components']>=4
result=dict(schema_version='evidence-stage-gate-v1',stage='B2-transition-variation',passed=passed,thresholds=dict(corrected_components=8,worsened_components=4),runs=rows,allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in rows),scientific_interpretation='Sufficient preregistered variation to test value learning; no effectiveness claim.' if passed else 'Insufficient transition variation in this frozen pilot; stop B3/B4 and scaling. This is not a universal refutation of evidence-conditioned retrieval.',V_locked_model_evaluated=False,test_authorized=False)
(C/'reports/B2-transition-gate.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# B2 冻结转移可检验性 gate','',f'passed={passed}；Gaussian Vdev corrected组件={main["corrected_components"]}，worsened组件={main["worsened_components"]}；预注册门槛8/4。','','| matcher | role | 完整组件 | 候选漏召回 | 转移 | corrected组件 | worsened组件 |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['matcher']+' | '+r['role']+' | 64 | '+str(r['candidate_miss_components'])+' | '+str(r['transitions_count'])+' | '+str(r['corrected_components'])+' | '+str(r['worsened_components'])+' |')
lines+=['',result['scientific_interpretation'],'','全部六臂完成后判定；哈希验证完整 cohort、实际 top-K、unknown、零动作组件。V_locked 未做模型评价，test sealed。']
(C/'reports/B2-transition-gate.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
