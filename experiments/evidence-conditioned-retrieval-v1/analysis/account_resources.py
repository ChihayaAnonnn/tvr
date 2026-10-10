"""Administrative final accounting; never launches or mutates a historical run."""
from pathlib import Path
import json,hashlib,sys,math
R=Path('/home/xujie/.codex/worktrees/ad3a/UATVR'); C=R/'experiments/evidence-conditioned-retrieval-v1'
sys.path[:0]=[str(R),str(R/'src')]
from scripts import manage_experiment_run as m
from experiment_artifacts import strict_json_loads
rows=m.load_registry(C/'registry.jsonl'); latest={r['run_id']:r for r in rows}
if any(r['state'] not in {'completed','failed','cancelled'} for r in latest.values()):raise ValueError('all reserved runs must be terminal before final accounting')
out=[]
for run_id,r in latest.items():
 root=Path(r['artifact_path']);manifest,status=m._load_run(root)
 if status['state']!=r['state'] or __import__('experiment_artifacts').manifest_sha256(manifest)!=r['manifest_sha256']:raise ValueError('terminal identity mismatch')
 if m._all_output_hashes(root)!=status['output_hashes']:raise ValueError('terminal output inventory mismatch: '+run_id)
 p=root/'provenance/controller-result.json'
 if not p.exists():raise ValueError('missing measured controller accounting: '+run_id)
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if status['output_hashes'].get('provenance/controller-result.json')!=h:raise ValueError('controller result not bound in terminal inventory')
 result=strict_json_loads(raw.decode());gpus=manifest['environment']['gpu_count']
 if set(result)=={'elapsed_seconds','observed_launcher_exit_code','reason','terminal_exit_code'}:
  seconds=result['elapsed_seconds'];receipt_schema='launch-once-v1'
 elif set(result)=={'elapsed_seconds','observed_launcher_exit_code','reason','terminal_exit_code','worker_spawned'}:
  if r['state']!='cancelled' or result['worker_spawned'] is not False or result['elapsed_seconds']!=0 or result['observed_launcher_exit_code'] is not None or result['terminal_exit_code']!=125:raise ValueError('unproven zero-work cancellation')
  seconds=result['elapsed_seconds'];receipt_schema='cancelled-before-spawn-v1'
 elif set(result)=={'allocated_gpu_hours','ended_at_utc','exit_code','timed_out','wall_seconds'}:
  seconds=result['wall_seconds'];receipt_schema='initial-Q0-wall-seconds-v1'
  if abs(result['allocated_gpu_hours']-seconds*gpus/3600)>1e-12:raise ValueError('legacy receipt GPU accounting mismatch')
 else:raise ValueError('unsupported controller accounting schema: '+run_id)
 if type(gpus) is not int or gpus<0 or type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('invalid resource accounting')
 kind='software' if manifest['dataset']=='synthetic' else 'cpu_preparation' if gpus==0 else 'scientific'
 out.append(dict(run_id=run_id,state=r['state'],kind=kind,source_sha=manifest['git_sha'],manifest_sha256=r['manifest_sha256'],controller_result=dict(path=str(p),sha256=h),controller_receipt_schema=receipt_schema,controller_seconds=seconds,gpu_count=gpus,allocated_gpu_hours=seconds*gpus/3600,output_inventory_files=len(status['output_hashes'])))
result=dict(schema_version='evidence-campaign-resource-accounting-v1',scope='All unique terminal registry runs, including software failures and never-spawned cancellations. Allocated GPU wall hours, not GPU-active energy or isolated method latency.',terminal_counts={s:sum(r['state']==s for r in out) for s in ('completed','failed','cancelled')},allocated_gpu_hours={k:sum(r['allocated_gpu_hours'] for r in out if r['kind']==k) for k in ('software','scientific','cpu_preparation')},runs=out,test_authorized=False)
result['compatibility_note']='Initial Q0 uses the hash-bound wall_seconds receipt, later clients use elapsed_seconds. Two final-accounting invocations rejected the legacy Q0 key and the explicit never-spawned cancellation schema before writing any report; exact-schema parsing, independent legacy GPU-hour equality and strict zero-work cancellation checks added. Historical files unchanged.'
result['allocated_gpu_hours']['total']=sum(r['allocated_gpu_hours'] for r in out)
if result['allocated_gpu_hours']['software']>8 or result['allocated_gpu_hours']['scientific']>32:raise ValueError('campaign budget exceeded')
(C/'reports/final-resource-accounting.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
