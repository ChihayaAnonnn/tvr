"""Source-matched qualification, strict science scope, terminal inputs and budget gates."""
import json
from pathlib import Path
import subprocess
from experiment_artifacts import parse_registry_text,make_run_paths,load_validated_run_lifecycle,manifest_sha256
from .store import digest


def science_config(fixed,operation,accuracy=None,resume_from=None):
    if operation not in {'accuracy','performance'}:raise ValueError('unknown scientific operation')
    result={**fixed,'operation':operation,'max_seconds':28800 if operation=='accuracy' else 3600,'resume_from':resume_from}
    if operation=='performance':
        if not accuracy:raise ValueError('performance requires archived accuracy/cache inputs')
        result['accuracy']=accuracy
    return result


def authenticate_qualification(repo,study,manifest):
    from .protocol import CONTROL,STUDY,committed
    permission=study.get('gpu_qualification')
    if not permission:raise ValueError('new two-GPU inference/cache qualification has not passed and been archived')
    records=parse_registry_text(committed(repo,repo/CONTROL/'registry.jsonl').decode())
    rows=[r for r in records if r['event']=='completed' and r['run_id']==permission['run_id'] and r['manifest_sha256']==permission['manifest_sha256']]
    if len(rows)!=1:raise ValueError('two-GPU qualification requires committed completed registry evidence')
    root=Path(rows[0]['artifact_path']);paths=make_run_paths(root.parents[3],STUDY,permission['run_id'])
    qm,status,_=load_validated_run_lifecycle(paths)
    report=root/'provenance/laya-qualification.json'
    if (status['state']!='completed' or qm['study']!=STUDY or qm['environment']['gpu_count']!=2
            or manifest_sha256(qm)!=permission['manifest_sha256']
            or status['output_hashes'].get('provenance/laya-qualification.json')!=permission['report_sha256']
            or digest(report)!=permission['report_sha256']):raise ValueError('two-GPU qualification identity mismatch')
    for name,sha in status['output_hashes'].items():
        p=root/name
        if not p.resolve().is_relative_to(root) or digest(p)!=sha:raise ValueError('qualification output changed')
    r=json.loads(report.read_text())
    checks=('real_inference','visual_cache','shard_merge','batch_1_8','finite_fp32','exact_resume','cached_uncached_1e5','original_uatvr_inference')
    if (r.get('world_size')!=2 or r.get('device_type')!='cuda' or r.get('production_capacity') is not True
            or r.get('random_initialization') is not True or r.get('synthetic_only') is not True
            or r.get('source_sha')!=qm['git_sha'] or r.get('manifest_sha256')!=permission['manifest_sha256']
            or not all(r.get('checks',{}).get(k) is True for k in checks)):raise ValueError('incomplete two-GPU qualification')
    changed=subprocess.run(['git','diff','--name-only',qm['git_sha'],manifest['git_sha'],'--','src','scripts','experiment_artifacts.py'],
                           cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
    if changed:raise ValueError('executable source changed since qualification')
    return records


def authenticated_accuracy(repo,reference):
    from .protocol import CONTROL,STUDY,committed
    records=parse_registry_text(committed(repo,repo/CONTROL/'registry.jsonl').decode())
    matches=[r for r in records if r['run_id']==reference['run_id'] and r['event']=='completed' and r['manifest_sha256']==reference['manifest_sha256']]
    if len(matches)!=1 or matches[0]['artifact_path']!=reference['artifact_path']:raise ValueError('accuracy prerequisite not committed completed evidence')
    root=Path(reference['artifact_path']);paths=make_run_paths(root.parents[3],STUDY,reference['run_id'])
    manifest,status,_=load_validated_run_lifecycle(paths)
    path=root/'provenance/laya-validation.json'
    if (status['state']!='completed' or manifest_sha256(manifest)!=reference['manifest_sha256']
            or status['output_hashes'].get('provenance/laya-validation.json')!=reference['report_sha256']
            or digest(path)!=reference['report_sha256']):raise ValueError('accuracy report prerequisite identity mismatch')
    report=json.loads(path.read_text())
    if report['query_count']!=10000 or report['video_count']!=500 or report['test_authorized'] is not False:raise ValueError('accuracy prerequisite coverage invalid')
    for item in reference['input_indices']:
        committed(repo,Path(item['path']))
        if digest(item['path'])!=item['sha256']:raise ValueError('accuracy cache inventory changed')
    return manifest,status,report


def assert_science(repo,registry,manifest,operation,command,study):
    from .protocol import CONTROL,committed,validate_input_bindings,ENTRY
    from .campaign import budget_rows,projected_hours,validate_lineage
    if study.get('science_authorized') is not True:raise ValueError('scientific launch is not enabled in committed study')
    records=authenticate_qualification(repo,study,manifest)
    fixed=json.loads(committed(repo,repo/CONTROL/'fixed.json'))
    ev=manifest['config']['evaluation'];kind=ev.get('operation')
    accuracy=study.get('accuracy_result') if kind=='performance' else None
    expected=science_config(fixed,kind,accuracy,ev.get('resume_from'))
    cards=2 if kind=='accuracy' else 1
    devices='CUDA_VISIBLE_DEVICES=1,3' if cards==2 else 'CUDA_VISIBLE_DEVICES=3'
    if (ENTRY not in {Path(str(x)).name for x in command} or ev!=expected or manifest['dataset']!='msrvtt'
            or manifest['stage']!=('baseline' if kind=='accuracy' else 'efficiency')
            or manifest['test_authorized'] is not False or manifest['validation_authorized'] is not True
            or manifest['environment']['gpu_count']!=cards or manifest['environment']['precision']!='fp32'
            or manifest['environment']['distributed_topology']!=('torchrun-2' if cards==2 else 'single-process')
            or devices not in command):raise ValueError('science manifest differs from frozen scope/topology')
    if study['gpu_qualification']['run_id'] not in manifest['parent_run_ids']:raise ValueError('science must bind current qualification parent')
    from semantic_retrieval.full_data.final import authenticated_parent
    from semantic_retrieval.full_data.runtime import settings_from_manifest
    parent,_=authenticated_parent(repo,{'selected':[fixed['parent']]},fixed['parent']['run_id'])
    settings=settings_from_manifest(parent)
    validate_input_bindings(repo,manifest,fixed,{'clip_weights':str(settings.clip_weights),'clip_sha256':settings.clip_sha256})
    if kind=='performance':
        authenticated_accuracy(repo,accuracy)
        refs={r['path']:r['sha256'] for r in manifest['dataset_splits']}
        if any(refs.get(r['path'])!=r['sha256'] for r in accuracy['input_indices']):raise ValueError('performance caches are not reserved inputs')
    science_records=[]
    for record in records:
        if record['event']!='reserved':continue
        other=json.loads((Path(record['artifact_path'])/'manifest.json').read_text())
        if other['config']['evaluation'].get('operation')==kind:science_records.append((record,other))
    existing={r['run_id'] for r,_ in science_records};prospective=manifest['run_id'] not in existing
    completed={r['run_id'] for r in records if r['event']=='completed'}
    if prospective and existing&completed:raise ValueError('successful scientific operation cannot be repeated')
    if prospective and existing:validate_lineage(repo,manifest,records,allow_source_upgrade=False)
    if ev.get('resume_from'):
        previous=[r for r in records if r['artifact_path']==ev['resume_from'] and r['event'] in {'failed','cancelled'}]
        if len(previous)!=1 or previous[0]['run_id'] not in manifest['parent_run_ids']:raise ValueError('resume requires immutable failed/cancelled declared parent')
    total=projected_hours(budget_rows(repo,records,category='scientific'))
    if prospective:total+=cards*ev['max_seconds']/3600
    if total>24:raise ValueError('scientific cumulative GPU-hour budget exceeded')
