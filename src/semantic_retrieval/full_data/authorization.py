"""Fail-closed independent study dispatch at reservation and worker startup."""
from pathlib import Path
import subprocess
import hashlib

import yaml
from experiment_artifacts import (parse_registry_text, make_run_paths, load_validated_run_lifecycle,
                                  strict_json_load, manifest_sha256)
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import safe_path

STUDY = 'full-data-retrieval-v1'
CONTROL = 'experiments/' + STUDY
ENTRIES = {'prepare_full_data_retrieval.py', 'train_full_data_retrieval.py', 'qualify_full_data_retrieval.py', 'evaluate_full_data_test.py'}
PRETRAINED_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'


def _committed(repo, path):
    safe_path(path)
    raw = subprocess.run(['git', 'show', 'HEAD:'+path.relative_to(repo).as_posix()], cwd=repo,
                         check=True, capture_output=True).stdout
    if raw != path.read_bytes():
        raise ValueError('full-data protocol must match committed control file')
    return raw


def _terminal_report(repo, permission, filename):
    if not isinstance(permission, dict) or set(permission) != {'run_id', 'manifest_sha256', 'report_sha256'}:
        raise ValueError('missing full-data prerequisite evidence')
    records = parse_registry_text(_committed(repo, repo/CONTROL/'registry.jsonl').decode())
    matches = [r for r in records if r['event'] == 'completed' and r['run_id'] == permission['run_id']
               and r['manifest_sha256'] == permission['manifest_sha256']]
    if len(matches) != 1:
        raise ValueError('prerequisite requires one committed completed full-data run')
    root = Path(matches[0]['artifact_path'])
    paths = make_run_paths(root.parents[3], STUDY, permission['run_id'])
    manifest, status, _ = load_validated_run_lifecycle(paths)
    if (manifest['study'] != STUDY or status['state'] != 'completed'
            or manifest_sha256(manifest) != permission['manifest_sha256']):
        raise ValueError('prerequisite run/manifest identity mismatch')
    for name, sha in status['output_hashes'].items():
        path = paths.root/name; safe_path(path)
        if file_sha256(path) != sha:
            raise ValueError('prerequisite terminal output changed')
    path = paths.provenance/filename
    rel = path.relative_to(paths.root).as_posix()
    if (status['output_hashes'].get(rel) != permission['report_sha256']
            or file_sha256(path) != permission['report_sha256']):
        raise ValueError('prerequisite report is not authenticated terminal evidence')
    with path.open() as stream:
        report = strict_json_load(stream)
    if report.get('source_sha') != manifest['git_sha'] or report.get('manifest_sha256') != permission['manifest_sha256']:
        raise ValueError('prerequisite report source/manifest mismatch')
    return manifest, report, path


def validate_prerequisites(repo, study, manifest, settings):
    if not isinstance(study.get('gpu_qualification'), dict):
        raise ValueError('new full-data matching-topology GPU qualification has not passed')
    qm, report, _ = _terminal_report(repo, study['gpu_qualification'], 'full-data-qualification.json')
    checks = ('actual_forward_backward', 'intended_gradients', 'global_gradient_normalization', 'exact_resume',
              'rank_weights_equal', 'finite_fp32', 'metric_permutation', 'eval_rng_preserved', 'strict_checkpoint_rejection')
    if (report.get('schema_version') != 'full-data-qualification-v1' or report.get('world_size') != settings.world_size
            or settings.world_size != study['software_authorization']['world_size']
            or report.get('device_type') != 'cuda' or report.get('production_capacity') is not True
            or report.get('test_authorized') is not False or set(report.get('qualified_methods', ())) != {'uatvr', 'clip4clip'}
            or not all(report.get('checks', {}).get(k) is True for k in checks)):
        raise ValueError('incomplete production matching-CUDA-rank qualification')
    if settings.factor_arm is not None and report.get('factor_paths_qualified') is not True:
        raise ValueError('single-factor paths lack matching-source qualification')
    changed = subprocess.run(['git', 'diff', '--name-only', qm['git_sha'], manifest['git_sha'], '--',
                              'src/semantic_retrieval', 'scripts', 'experiment_artifacts.py', 'experiment_tracking.py'],
                             cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    if changed:
        raise ValueError('executable source changed since full-data qualification')
    pm, certificate, path = _terminal_report(repo, study.get('input_certification'), 'complete-preparation.json')
    if (certificate.get('schema_version') != 'full-data-preparation-v1'
            or certificate.get('train_video_count') != 8246 or certificate.get('validation_video_count') != 500
            or certificate.get('train_caption_count') != 164920 or certificate.get('validation_caption_count') != 10000
            or certificate.get('decoded_video_count') != 8746 or certificate.get('excluded_train_count') != 254
            or certificate.get('test_authorized') is not False or certificate.get('test_caption_values_decoded') != 0
            or settings.source_report != path or settings.source_report_sha256 != study['input_certification']['report_sha256']):
        raise ValueError('scientific worker does not bind complete approved source certificate')


def validate_resume_parent(repo, manifest, settings):
    ref = settings.resume
    if ref is None: return
    if ref.parent_run_id not in manifest['parent_run_ids'] or ref.parent_run_id == manifest['run_id']:
        raise ValueError('resume requires a distinct declared parent run')
    records = parse_registry_text(_committed(repo, repo/CONTROL/'registry.jsonl').decode())
    terminals = [r for r in records if r['run_id'] == ref.parent_run_id and r['event'] == 'failed']
    if len(terminals) != 1: raise ValueError('resume parent must have one committed failed terminal record')
    root = Path(terminals[0]['artifact_path'])
    paths = make_run_paths(root.parents[3], STUDY, ref.parent_run_id)
    parent, status, _ = load_validated_run_lifecycle(paths)
    if (status['state'] != 'failed' or manifest_sha256(parent) != terminals[0]['manifest_sha256']
            or parent['study'] != STUDY or ref.checkpoint.parent != paths.checkpoints):
        raise ValueError('resume parent checkpoint ownership/identity mismatch')
    for name, sha in status['output_hashes'].items():
        path = paths.root/name; safe_path(path)
        if file_sha256(path) != sha: raise ValueError('resume parent terminal output changed')
    for path, sha in ((ref.checkpoint, ref.checkpoint_sha256), (ref.checkpoint.with_suffix('.json'), ref.sidecar_sha256)):
        if status['output_hashes'].get(path.relative_to(root).as_posix()) != sha or file_sha256(path) != sha:
            raise ValueError('resume checkpoint is not authenticated parent output')
    def recipe(m):
        import copy
        cfg = copy.deepcopy(m['config']); cfg['evaluation'].pop('resume')
        return cfg
    if recipe(parent) != recipe(manifest): raise ValueError('resume scientific recipe/input identity changed')
    with ref.checkpoint.with_suffix('.json').open() as stream: meta = strict_json_load(stream)
    origin = meta['identity']['source_sha']
    if subprocess.run(['git', 'merge-base', '--is-ancestor', origin, parent['git_sha']], cwd=repo, capture_output=True).returncode:
        raise ValueError('resume origin is not the recorded source lineage')
    for source in (parent['git_sha'], origin):
        changed = subprocess.run(['git', 'diff', '--name-only', source, manifest['git_sha'], '--',
                                  'src/semantic_retrieval', 'scripts', 'experiment_artifacts.py', 'experiment_tracking.py'],
                                 cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        if changed: raise ValueError('resume executable compatibility cannot be proven; fresh run required')


def assert_full_data_authorized(*, repo_root, registry_path, manifest, operation, command=None):
    evaluation = manifest.get('config', {}).get('evaluation', {})
    argv = command if command is not None else manifest.get('command', ())
    entries = {Path(str(a)).name for a in argv} & ENTRIES
    scenario = evaluation.get('qualification_scenario', '')
    relevant = (manifest.get('study') == STUDY or STUDY in str(registry_path or '')
                or evaluation.get('config_kind') in {STUDY, 'full-data-test-v1'} or str(scenario).startswith('full_data_') or entries)
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
    raw = _committed(repo, study_path)
    study = yaml.safe_load(raw)
    required = dict(study=STUDY, kind=STUDY, protocol_version=1, status='registered',
                    control_plane=CONTROL, data_plane='artifacts/'+STUDY, registry=CONTROL+'/registry.jsonl')
    if any(study.get(k) != v for k, v in required.items()):
        raise ValueError('full-data study identity/status mismatch')
    if 'evaluate_full_data_test.py' in entries or evaluation.get('config_kind') == 'full-data-test-v1':
        from .final import assert_final_authorized
        assert_final_authorized(repo, study, manifest, entries)
        return
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
    from .runtime import settings_from_manifest, KIND
    if evaluation.get('config_kind') != KIND:
        raise ValueError('full-data execution requires explicit typed config marker')
    settings = settings_from_manifest(manifest)
    from .factors import validate_factor_permission
    validate_factor_permission(study, settings, manifest['variant'])
    if settings.factor_arm is not None and settings.stage == 'baseline' and operation == 'reserve':
        from .factors import validate_factor_slot
        keys=[]
        for record in parse_registry_text(_committed(repo,registry).decode()):
            if record['event'] != 'completed':continue
            with (Path(record['artifact_path'])/'manifest.json').open() as stream:
                previous=strict_json_load(stream)
            if manifest_sha256(previous)!=record['manifest_sha256']:
                raise ValueError('prior factor manifest identity changed')
            ev=previous['config']['evaluation']
            if ev.get('full_data_stage')=='baseline' and ev.get('factor_arm') is not None:
                keys.append((previous['config']['model']['method'],ev['factor_arm'],previous['seed']))
        validate_factor_slot(keys,(settings.method,settings.factor_arm,settings.seed),study['validation_factors']['max_scientific_runs'])
    if settings.stage != 'cpu_fixture':
        expected_loader = dict(workers_per_rank=settings.loader_workers, prefetch_batches_per_worker=2,
                               backend='spawn-v1', worker_timeout_seconds=90, ordered_batches=True)
        if study.get('loader_policy') != expected_loader:
            raise ValueError('loader differs from sealed study policy')
    if settings.stage != 'cpu_fixture' and (settings.world_size != study['software_authorization']['world_size']
            or settings.memory_limit_gib != study['resource_policy']['memory_limit_gib']):
        raise ValueError('GPU topology/memory settings differ from approved study')
    if manifest.get('seed') != settings.seed or manifest['stage'] != 'baseline':
        raise ValueError('full-data stage/seed mismatch')
    inputs = {r['path']: r['sha256'] for r in manifest['input_checkpoints']}
    if inputs.get(str(settings.clip_weights)) != settings.clip_sha256:
        raise ValueError('full-data CLIP identity is not reserved')
    if settings.stage in {'cpu_fixture', 'gpu_qualification'}:
        if (entries != {'qualify_full_data_retrieval.py'} or manifest['dataset'] != 'synthetic'
                or manifest['dataset_splits'] or len(manifest['input_checkpoints']) != 1):
            raise ValueError('qualification is synthetic only with exactly its CLIP input')
        permission = study['software_authorization']
        if settings.stage == 'cpu_fixture':
            if (permission.get('cpu_fixture') is not True or env['gpu_count'] != 0
                    or env['distributed_topology'] != 'single-process' or manifest['variant'] != 'full-data-cpu-fixture'):
                raise ValueError('CPU fixture requires authorized single-process synthetic execution')
        elif (permission.get('gpu') is not True or env['gpu_count'] != settings.world_size
              or env['distributed_topology'] != f'torchrun-{settings.world_size}' or manifest['variant'] != 'full-data-q0'
              or settings.clip_sha256 != permission.get('random_clip_sha256')):
            raise ValueError('GPU qualification requires approved CUDA ranks and registered random formal CLIP')
        return
    if (entries != {'train_full_data_retrieval.py'} or manifest['dataset'] != 'msrvtt'
            or env['gpu_count'] != settings.world_size or env['distributed_topology'] != f'torchrun-{settings.world_size}'
            or manifest['variant'] != (settings.method+'-'+settings.factor_arm if settings.factor_arm else settings.method) or settings.seed not in study['seeds']
            or settings.clip_sha256 != PRETRAINED_SHA):
        raise ValueError('scientific baseline requires frozen full-data approved-rank recipe')
    validate_prerequisites(repo, study, manifest, settings)
    science = study['science']
    if science.get('authorized') is not True or 'baseline' not in science.get('authorized_stages', ()):
        raise ValueError('full-data scientific baseline gate is closed')
    validate_resume_parent(repo, manifest, settings)
