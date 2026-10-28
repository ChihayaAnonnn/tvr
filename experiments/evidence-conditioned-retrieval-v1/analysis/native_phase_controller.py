"""Administrative client only: launch sealed workers and finalize Scheme A phases."""
import argparse,json,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
REPO=Path('/home/xujie/.codex/worktrees/ad3a/UATVR')
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts import manage_experiment_run as m
from scripts.run_dual_gpu_preflight import launch_once
from semantic_retrieval.evidence.config import load_evidence_config,evidence_manifest_config
from semantic_retrieval.evidence.frame_cache import publish_json
from experiment_artifacts import verify_array_reference
P=argparse.ArgumentParser();P.add_argument('operation',choices=['launch','finalize']);P.add_argument('phase');A=P.parse_args()
CONTROL=Path('experiments/evidence-conditioned-retrieval-v1');REGISTRY=CONTROL/'registry.jsonl';ARTIFACT=Path('/data2/hxj/project/UATVR/artifacts')
phase_path=CONTROL/'phases'/f'{A.phase}.json';phase=m._load_committed_phase(REPO,phase_path);records=m.load_registry(REPO/REGISTRY)
identities=[]
for spec in phase['runs']:
 r=m._reservation_for_spec(spec,records,ARTIFACT)
 identities.append((spec,SimpleNamespace(repo_root=REPO,registry=REGISTRY,run_dir=Path(r['artifact_path']),run_id=r['run_id'],manifest_sha256=r['manifest_sha256'])))
if A.operation=='launch':
 consumed=0.
 for r in records:
  if r['event'] not in {'completed','failed','cancelled'}:continue
  root=Path(r['artifact_path']);manifest,_=m._load_run(root)
  if manifest['dataset']=='synthetic':continue
  if manifest['config']['evaluation'].get('evidence_stage') not in {'reference','matcher','transitions','value','calibration','evaluation'}:continue
  result=root/'provenance/controller-result.json'
  if not result.exists():raise ValueError('cannot prove prior scientific GPU consumption')
  consumed+=json.loads(result.read_text())['elapsed_seconds']*manifest['environment']['gpu_count']/3600
 if consumed+len(identities)*2>32:raise ValueError('remaining campaign budget cannot cover sealed per-run limits')
 # Resolve all devices before spawn; reject overlap and any occupied card.
 devices=[]
 for spec,a in identities:
  setting=[s for s in spec['command'] if str(s).startswith('CUDA_VISIBLE_DEVICES=')]
  if len(setting)!=1:raise ValueError('device set must be sealed in the command')
  ids=setting[0].split('=',1)[1].split(',');devices.extend(ids)
 if len(set(devices))!=len(devices):raise ValueError('phase device allocations overlap')
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 usage={line.split(',')[0].strip():int(line.split(',')[2].strip()) for line in inventory.splitlines()}
 if any(usage[i]>100 for i in devices):raise ValueError('selected GPU is occupied; do not interfere')
 for spec,a in identities:
  manifest,status=m._load_run(a.run_dir)
  if status['state']!='reserved':raise ValueError('never relaunch a historical run')
  templates=[s for s in spec['selection_lineage'] if '/configs/' in s and s.endswith('.json')]
  if len(templates)!=1:raise ValueError('one committed configuration template required')
  path=REPO/templates[0];raw=subprocess.check_output(['git','show','HEAD:'+templates[0]],cwd=REPO)
  if raw!=path.read_bytes():raise ValueError('template differs from committed source')
  cfg=json.loads(raw);cfg['runtime']=dict(study_path=str(REPO/CONTROL/'study.yaml'),registry_path=str(REPO/REGISTRY),run_dir=str(a.run_dir),run_id=a.run_id,manifest_sha256=a.manifest_sha256,source_sha=manifest['git_sha'],eval_split=manifest['config']['evaluation']['eval_split'])
  target=a.run_dir/'provenance/worker-config.json'
  with target.open('x') as f:json.dump(cfg,f,indent=2)
  if evidence_manifest_config(load_evidence_config(target),eval_split=cfg['runtime']['eval_split'])!=manifest['config']:raise ValueError('effective config mismatch')
  with (a.run_dir/'provenance/device-schedule.txt').open('x') as f:f.write(inventory)
  publish_json(a.run_dir/'provenance/campaign-budget.json',dict(prior_gpu_hours=consumed,max_gpu_hours=32,max_run_wall_seconds=7200,phase_run_count=len(identities),max_concurrent_workers=len(identities),test_authorized=False))
  m.verify_run(a)
 def launch(pair):
  spec,a=pair;manifest,_=m._load_run(a.run_dir);v=m.verify_run(a)
  current=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  usage_now={line.split(',')[0].strip():int(line.split(',')[1]) for line in current.splitlines()}
  selected=next(s for s in spec['command'] if s.startswith('CUDA_VISIBLE_DEVICES=')).split('=',1)[1].split(',')
  if any(usage_now[i]>100 for i in selected):raise ValueError('selected GPU became occupied before queued worker; do not interfere')
  with (a.run_dir/'provenance/device-schedule-at-spawn.txt').open('x') as handle:handle.write(current)
  result=launch_once(v['command_argv'],cwd=REPO,run_dir=a.run_dir,command_sha256=manifest['command_sha256'],timeout=7200)
  print(json.dumps(dict(run_id=a.run_id,**result)),flush=True);return result
 with ThreadPoolExecutor(max_workers=len(identities)) as pool:results=list(pool.map(launch,identities))
 sys.exit(int(any(r['terminal_exit_code']!=0 for r in results)))
else:
 import torch
 keys={'path','key','shape','dtype','byte_size','format','sha256'}
 for spec,a in identities:
  m.verify_run(a);root=a.run_dir;manifest,status=m._load_run(root)
  result=json.loads((root/'provenance/controller-result.json').read_text());code=result['terminal_exit_code']
  if code==0 and status['state']!='validating':raise ValueError('successful worker missing validating state')
  count=[0]
  def walk(v,origin):
   if isinstance(v,dict):
    if set(v)==keys:
     bases=[p for p in (origin.parent,*origin.parents) if (p==root or p.is_relative_to(root)) and (p/v['path']).is_file()]
     if not bases:raise ValueError('array reference missing')
     verify_array_reference(bases[0],v);count[0]+=1
    else:
     for child in v.values():walk(child,origin)
   elif isinstance(v,(list,tuple)):
    for child in v:walk(child,origin)
   elif isinstance(v,torch.Tensor) and v.numel()*v.element_size()>1024**2:raise ValueError('unexternalized tensor')
  for path in root.rglob('*.json'):walk(json.loads(path.read_text()),path)
  for path in root.rglob('*.pt'):walk(torch.load(path,map_location='cpu',weights_only=False),path)
  publish_json(root/'provenance/controller-validation.json',dict(array_references_verified=count[0],raw_checkpoint_tensor_cap_bytes=1024**2,test_authorized=False))
  def transition(expected,next_state,exit_code=None):return m.transition_run(SimpleNamespace(**vars(a),expected=expected,next_state=next_state,exit_code=exit_code,output=[]))
  if status['state']=='running':transition('running','validating');status['state']='validating'
  transition(status['state'],'cancelled' if status['state']=='reserved' else 'completed' if code==0 else 'failed',code)
  print(json.dumps(dict(run_id=a.run_id,exit_code=code,array_references_verified=count[0])),flush=True)
 summary_path=CONTROL/'summaries'/f'{A.phase}.json'
 summary=m.finalize_phase(SimpleNamespace(repo_root=REPO,phase=phase_path,registry=REGISTRY,artifact_root=ARTIFACT,summary=summary_path))
 subprocess.run(['git','add',str(REGISTRY),str(summary_path)],cwd=REPO,check=True)
 subprocess.run(['git','commit','-m','experiment(evidence): archive '+phase['phase']],cwd=REPO,check=True)
 print(json.dumps(dict(phase=phase['phase'],terminal_counts=summary['terminal_counts'])))
