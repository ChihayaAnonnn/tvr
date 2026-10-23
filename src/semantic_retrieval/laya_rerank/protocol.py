"""Independent, fail-closed policy applied at reserve and worker preparation."""
from pathlib import Path
import json
import subprocess
from experiment_artifacts import parse_registry_text, make_run_paths, load_validated_run_lifecycle, manifest_sha256
from .store import digest

STUDY='laya-rerank-validation-v1'
CONTROL='experiments/'+STUDY
ENTRY='evaluate_laya_rerank.py'


def committed(repo,path):
    path=Path(path)
    raw=subprocess.run(['git','show','HEAD:'+path.relative_to(repo).as_posix()],cwd=repo,check=True,capture_output=True).stdout
    if path.read_bytes()!=raw: raise ValueError('Laya control file differs from committed bytes')
    return raw


def load_fixed(repo):
    return json.loads((Path(repo)/CONTROL/'fixed.json').read_text())


def assert_laya_authorized(*,repo_root,registry_path,manifest,operation,command):
    repo=Path(repo_root).resolve()
    entries={Path(str(x)).name for x in command}
    claimed=manifest.get('study')==STUDY or Path(registry_path).parent.name==STUDY
    if not claimed and ENTRY not in entries: return
    if manifest.get('study')!=STUDY or Path(registry_path).resolve()!=repo/CONTROL/'registry.jsonl':
        raise ValueError('Laya entry requires independent study and registry')
    # No qualification report is fabricated by CPU tests or inherited from old studies.
    study=json.loads(committed(repo,repo/CONTROL/'study.yaml'))
    permission=study.get('gpu_qualification')
    if not permission: raise ValueError('new four-GPU inference/cache qualification has not passed and been archived')
    records=parse_registry_text(committed(repo,repo/CONTROL/'registry.jsonl').decode())
    rows=[r for r in records if r['event']=='completed' and r['run_id']==permission['run_id']
          and r['manifest_sha256']==permission['manifest_sha256']]
    if len(rows)!=1: raise ValueError('four-GPU qualification requires committed completed registry evidence')
    root=Path(rows[0]['artifact_path']); paths=make_run_paths(root.parents[3],STUDY,permission['run_id'])
    qm,status,_=load_validated_run_lifecycle(paths)
    report=root/'provenance/laya-qualification.json'
    if (status['state']!='completed' or qm['study']!=STUDY or qm['environment']['gpu_count']!=4
            or manifest_sha256(qm)!=permission['manifest_sha256']
            or status['output_hashes'].get('provenance/laya-qualification.json')!=permission['report_sha256']
            or digest(report)!=permission['report_sha256']):
        raise ValueError('four-GPU qualification identity mismatch')
    for name,sha in status['output_hashes'].items():
        p=root/name
        if not p.resolve().is_relative_to(root) or digest(p)!=sha: raise ValueError('qualification output changed')
    r=json.loads(report.read_text())
    checks=('real_inference','visual_cache','shard_merge','batch_1_8','finite_fp32','exact_resume','cached_uncached_1e5')
    if (r.get('world_size')!=4 or r.get('device_type')!='cuda' or r.get('production_capacity') is not True
            or r.get('random_initialization') is not True or r.get('synthetic_only') is not True
            or r.get('source_sha')!=qm['git_sha'] or r.get('manifest_sha256')!=permission['manifest_sha256']
            or not all(r.get('checks',{}).get(k) is True for k in checks)):
        raise ValueError('four-GPU qualification missing required inference checks')
    changed=subprocess.run(['git','diff','--name-only',qm['git_sha'],manifest['git_sha'],'--','src','scripts',
                            'experiment_artifacts.py'],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
    if changed: raise ValueError('executable source changed since qualification')
    fixed=json.loads(committed(repo,repo/CONTROL/'fixed.json'))
    if (ENTRY not in entries or manifest['config']['evaluation']!=fixed or manifest['test_authorized'] is not False
            or manifest['validation_authorized'] is not True or manifest['dataset']!='msrvtt'
            or manifest['environment']['precision']!='fp32' or manifest['environment']['gpu_count']!=1):
        raise ValueError('Laya launch differs from fixed validation scope')
    from semantic_retrieval.full_data.final import authenticated_parent
    from semantic_retrieval.full_data.runtime import settings_from_manifest
    parent,_=authenticated_parent(repo,{'selected':[fixed['parent']]},fixed['parent']['run_id'])
    settings=settings_from_manifest(parent)
    validate_input_bindings(repo,manifest,fixed,{'clip_weights':str(settings.clip_weights),'clip_sha256':settings.clip_sha256})
    if study.get('science_authorized') is not True:
        raise ValueError('scientific launch is not enabled in committed study')


def validate_input_bindings(repo,manifest,fixed,parent_config):
    """Bind direct model inputs plus the authenticated parent provenance closure."""
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    refs=manifest_input_references(Path(repo),manifest)
    if not manifest['dataset_splits'] or not manifest['input_checkpoints']:
        raise ValueError('Laya consumed inputs must be bound in the reserved manifest')
    checkpoint=fixed['parent']['checkpoint']
    path=Path(checkpoint['path'])
    external=Path(repo)/CONTROL/'external.input-index.json'
    required={str(external):digest(external),checkpoint['path']:checkpoint['sha256'],
              str(path.with_suffix('.json')):checkpoint['sidecar_sha256'],
              parent_config['clip_weights']:parent_config['clip_sha256']}
    # The parent manifest closes over the source certificate and every input shard/hash.
    parent_manifest=path.parents[1]/'manifest.json'
    required[str(parent_manifest)]=digest(parent_manifest)
    if any(refs.get(p)!=sha for p,sha in required.items()):
        raise ValueError('Laya consumed inputs must be bound in the reserved manifest')
