from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,subprocess,yaml
root=Path(__file__).resolve().parents[3]; study=root/'experiments/evidence-conditioned-retrieval-v1'
def read(name): return json.loads((study/name).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
checks={}
def check(name,condition):
    assert condition,name
    checks[name]=True
audit=read('reports/EXPERIMENT_AUDIT.json'); claim=read('reports/CLAIM_VERDICT.json'); pre=read('reports/evidence-precheck.json')
for name,h in audit['audited_input_hashes'].items(): check('audit_hash:'+name,digest(study/name)==h)
for name,h in claim['review_record']['input_hashes'].items(): check('claim_hash:'+name,digest(study/name)==h)
for name,h in pre['audited_input_hashes'].items(): check('precheck_hash:'+name,digest(root/name)==h)
for entry in pre['checks']:
    value=json.loads((root/entry['source']).read_text())
    for key in entry['json_pointer'].lstrip('/').split('/'): value=value[int(key)] if isinstance(value,list) else value[key]
    check('typed_value:'+entry['id'],type(value) is type(entry['value']) and value==entry['value'])
check('audit_response_hash',digest(study/'reports/EXPERIMENT_AUDIT.md')==audit['raw_response_sha256'])
check('claim_response_hash',digest(study/'reports/CLAIM_VERDICT.md')==claim['review_record']['raw_response_sha256'])
check('scoped_review_verdicts',audit['integrity_status']=='warn' and claim['claim_supported']=='no' and all(x['claim_supported']=='no' and not x['tested'] for x in claim['per_claim'].values()))
for review in (audit,claim): check('provisional:'+('audit' if review is audit else 'claim'),review['review_independence']=='same-family' and review['acceptance_status']=='provisional')
definition=yaml.safe_load((study/'study.yaml').read_text())
check('science_closed',definition['science']['authorized'] is False and definition['science']['authorized_stages']==[])
check('test_sealed',definition['test_authorized'] is False and claim['test_authorized'] is False)
check('B2_negative',definition['science']['gates']['value']['passed'] is False and read('reports/B2-transition-gate.json')['passed'] is False)
check('no_locked_model_selection',definition['science']['pilot_outcome']['V_locked_used_for_model_evaluation_or_selection'] is False)
byrun=defaultdict(list)
for line in (study/'registry.jsonl').read_text().splitlines():
    event=json.loads(line); byrun[event['run_id']].append(event)
check('35_runs',len(byrun)==35)
counts=Counter()
for identity,events in byrun.items():
    check('lifecycle:'+identity,len(events)==2 and events[0]['event']=='reserved' and events[1]['event'] in ('completed','failed','cancelled'))
    check('run_identity:'+identity,events[0]['manifest_sha256']==events[1]['manifest_sha256'] and events[0]['artifact_path']==events[1]['artifact_path'])
    check('run_no_test:'+identity,all(e['test_authorized'] is False for e in events))
    counts[events[1]['event']]+=1
resources=read('reports/final-resource-accounting.json')
check('terminal_counts',dict(counts)==resources['terminal_counts']=={'completed':25,'failed':6,'cancelled':4})
check('budget_totals',resources['allocated_gpu_hours']['scientific']<32 and resources['allocated_gpu_hours']['software']<8)
q0=read('reports/Q0-R5-final-source-qualification.json')
check('four_rank_qualification',q0['world_size']==4 and len(q0['checks'])==13 and all(q0['checks'].values()))
check('690_regression',read('reports/gain-mse-integration-review.json')['root_regression']['tests_passed']==690)
commands=[['git','diff','--exit-code','f32bc82','HEAD','--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py'],['git','diff','--exit-code','1d16e57','HEAD','--','tests'],['git','diff','--exit-code','--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py','tests'],['git','diff','--cached','--exit-code','--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py','tests'],['git','diff','--check']]
for i,argv in enumerate(commands):
    result=subprocess.run(argv,cwd=root,text=True,capture_output=True)
    check('git_check:'+str(i),result.returncode==0)
for review in (audit,claim):
    trace=root/(review['local_trace'] if review is audit else review['review_record']['local_trace'])
    check('trace_present:'+str(trace),trace.is_dir() and any(trace.glob('*.request.json')) and any(trace.glob('*.response.md')))
    result=subprocess.run(['git','check-ignore',str(trace.relative_to(root))],cwd=root,capture_output=True)
    check('trace_private:'+str(trace),result.returncode==0)
output={'schema_version':'evidence-final-review-verification-v1','checks_passed':len(checks),'checks':checks,'terminal_counts':dict(counts),'allocated_gpu_hours':resources['allocated_gpu_hours'],'git_commands':commands,'scope':'Final documentary hashes, typed aggregates, terminal metadata and executable/test identity. No scientific rerun or new numerical efficacy evidence. Final clean git status is checked after committing these review deliverables.'}
(study/'reports/final-review-verification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'terminal_counts':dict(counts),'science_closed':True,'test_sealed':True,'product_and_tests_unchanged':True}))
