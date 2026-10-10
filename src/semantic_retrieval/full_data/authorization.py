"""Fail-closed independent study dispatch at reservation and worker startup."""
from pathlib import Path
import subprocess

import yaml

STUDY = 'full-data-retrieval-v1'
CONTROL = 'experiments/' + STUDY
ENTRIES = {'prepare_full_data_retrieval.py', 'train_full_data_retrieval.py', 'qualify_full_data_retrieval.py'}


def assert_full_data_authorized(*, repo_root, registry_path, manifest, operation, command=None):
    evaluation = manifest.get('config', {}).get('evaluation', {})
    argv = command if command is not None else manifest.get('command', ())
    entries = {Path(str(a)).name for a in argv} & ENTRIES
    scenario = evaluation.get('qualification_scenario', '')
    relevant = (manifest.get('study') == STUDY or STUDY in str(registry_path or '')
                or evaluation.get('config_kind') == STUDY or str(scenario).startswith('full_data_') or entries)
    if not relevant:
        return
    if (operation not in {'reserve', 'start'} or repo_root is None or registry_path is None
            or manifest.get('study') != STUDY):
        raise ValueError('full-data execution requires its independent study')
    repo = Path(repo_root).resolve()
    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = repo / registry
    if (registry != repo/CONTROL/'registry.jsonl'
            or manifest.get('registry_path') != CONTROL+'/registry.jsonl'
            or Path(manifest['artifact_path']).parents[2].name != STUDY):
        raise ValueError('full-data control/data namespace mismatch')
    study_path = registry.with_name('study.yaml')
    raw = subprocess.run(['git', 'show', 'HEAD:'+CONTROL+'/study.yaml'], cwd=repo,
                         check=True, capture_output=True).stdout
    if raw != study_path.read_bytes():
        raise ValueError('full-data protocol must match committed study')
    study = yaml.safe_load(raw)
    required = dict(study=STUDY, kind=STUDY, protocol_version=1, status='registered',
                    control_plane=CONTROL, data_plane='artifacts/'+STUDY, registry=CONTROL+'/registry.jsonl')
    if any(study.get(k) != v for k, v in required.items()):
        raise ValueError('full-data study identity/status mismatch')
    if (manifest.get('test_authorized') is not False or study.get('test_authorized') is not False
            or evaluation.get('eval_split') != 'val' or manifest.get('validation_authorized') is not True):
        raise ValueError('full-data test remains sealed; validation only')
    env = manifest['environment']
    if env['precision'] != 'fp32':
        raise ValueError('full-data execution requires FP32')
    if scenario == 'full_data_prepare:v1':
        if (entries != {'prepare_full_data_retrieval.py'}
                or study.get('cpu_preparation') != dict(authorized=True, scenario=scenario)
                or env['gpu_count'] != 0 or env['distributed_topology'] != 'single-process'
                or manifest['dataset'] != 'msrvtt' or manifest['stage'] != 'baseline'
                or manifest['variant'] != 'complete-source-input'
                or any(manifest['config'].get(k) for k in ('model', 'loss', 'optimizer', 'scheduler'))
                or evaluation != dict(eval_split='val', qualification_scenario=scenario)):
            raise ValueError('full-data preparation requires exact CPU-only permission')
        return
    raise ValueError('full-data execution lacks implemented qualified stage authorization')
