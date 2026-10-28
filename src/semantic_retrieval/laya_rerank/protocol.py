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
    if not claimed and not {ENTRY,'qualify_laya_rerank.py'} & entries: return
    if manifest.get('study')!=STUDY or Path(registry_path).resolve()!=repo/CONTROL/'registry.jsonl':
        raise ValueError('Laya entry requires independent study and registry')
    # No qualification report is fabricated by CPU tests or inherited from old studies.
    study=json.loads(committed(repo,repo/CONTROL/'study.yaml'))
    if 'qualify_laya_rerank.py' in entries:
        from .campaign import qualification_config,projected_hours,budget_rows,validate_lineage,qualification_records
        fixed=json.loads(committed(repo,repo/CONTROL/'fixed.json'))
        records=parse_registry_text(committed(repo,repo/CONTROL/'registry.jsonl').decode())
        expected=qualification_config(fixed)
        inventory=repo/CONTROL/'qualification.input-index.json'
        if (study['software_qualification'].get('world_size')!=2
                or study['software_qualification'].get('authorization')!='2026-10-23_explicit_user_allow_two_gpu_laya_experiments'
                or manifest.get('config',{}).get('evaluation')!=expected
                or manifest['dataset']!='synthetic' or manifest['stage']!='baseline'
                or manifest['test_authorized'] is not False or manifest['validation_authorized'] is not True
                or manifest['environment']['gpu_count']!=2 or manifest['environment']['precision']!='fp32'
                or manifest['environment']['distributed_topology']!='torchrun-2'
                or manifest['input_checkpoints']!=[]
                or manifest['dataset_splits']!=[{'path':str(inventory),'sha256':digest(inventory)}]):
            raise ValueError('qualification differs from synthetic dual-rank scope')
        existing={r['run_id'] for r in qualification_records(records)}
        prospective=0 if manifest['run_id'] in existing else 1
        hours=projected_hours(budget_rows(repo,records))+prospective
        if hours>8:raise ValueError('software GPU budget exceeded')
        if 'CUDA_VISIBLE_DEVICES=1,3' not in command:raise ValueError('qualification devices differ from authorization')
        if prospective and existing:validate_lineage(repo,manifest,records,allow_source_upgrade=True)
        return
    from .science_policy import assert_science
    return assert_science(repo,Path(registry_path),manifest,operation,command,study)


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
