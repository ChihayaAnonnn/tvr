"""Independent study permission and four-rank qualification gates, checked twice."""
from pathlib import Path
import hashlib
import subprocess

import yaml

from experiment_artifacts import load_validated_run_lifecycle, make_run_paths, parse_registry_text, strict_json_load
from semantic_retrieval.evidence.frame_cache import safe_path

KIND = 'evidence-retrieval-v1'
STUDY = 'evidence-conditioned-retrieval-v1'
CONTROL = 'experiments/'+STUDY
ENTRYPOINTS = {'train_evidence_retrieval.py','compile_evidence_transitions.py','evaluate_evidence_retrieval.py','qualify_evidence_retrieval.py'}


def _committed(repo, path):
    safe_path(path)
    rel = path.relative_to(repo).as_posix()
    raw = subprocess.run(['git','show','HEAD:'+rel],cwd=repo,check=True,capture_output=True).stdout
    if raw != path.read_bytes():
        raise ValueError('evidence protocol must match committed control file')
    return raw


def validate_value_qualification(qualified,requested,report):
    if any(qualified[k] != requested[k] for k in ('hidden_dim','feature_schema')) or requested['kind'] not in report.get('qualified_value_kinds',()):
        raise ValueError('science value capacity/schema/kind differs from qualified value path')


def validate_qualification(repo, study, manifest):
    q = study.get('four_gpu_qualification')
    if not isinstance(q,dict) or set(q) != {'run_id','manifest_sha256','report_sha256'}:
        raise ValueError('new evidence four-GPU qualification has not passed')
    registry = repo/CONTROL/'registry.jsonl'
    records = parse_registry_text(_committed(repo,registry).decode())
    completed = [r for r in records if r['run_id'] == q['run_id'] and r['event'] == 'completed' and r['manifest_sha256'] == q['manifest_sha256']]
    if len(completed) != 1:
        raise ValueError('qualification must have one committed completed terminal record')
    ref = completed[0]
    paths = make_run_paths(Path(ref['artifact_path']).parents[3],STUDY,q['run_id'])
    qm, status, _ = load_validated_run_lifecycle(paths)
    report_path = paths.provenance/'evidence-qualification.json'
    safe_path(report_path)
    relative = report_path.relative_to(paths.root).as_posix()
    if status['state'] != 'completed' or status['output_hashes'].get(relative) != q['report_sha256'] or hashlib.sha256(report_path.read_bytes()).hexdigest() != q['report_sha256']:
        raise ValueError('qualification report hash is not terminal evidence')
    with report_path.open() as h:
        report = strict_json_load(h)
    checks = ('real_clip_forward','loc_gradient','scale_gradient','value_gradient','unknown_rank','exact_resume','rank_weights_equal','action_recovery','finite_fp32')
    if report.get('schema_version') != 'evidence-qualification-v1' or report.get('world_size') != 4 or report.get('device_type') != 'cuda' or report.get('production_capacity') is not True or report.get('test_authorized') is not False or not all(report.get('checks',{}).get(k) is True for k in checks):
        raise ValueError('incomplete new-chain production four-CUDA-rank qualification')
    if report.get('source_sha') != qm['git_sha'] or report.get('manifest_sha256') != q['manifest_sha256']:
        raise ValueError('qualification source/manifest binding mismatch')
    # Control-only commits may follow qualification, but executable changes require requalification.
    changed = subprocess.run(['git','diff','--name-only',qm['git_sha'],manifest['git_sha'],'--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py'],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
    if changed:
        raise ValueError('executable source changed since four-rank qualification')
    def capacity(m):
        base = dict(m['base'])
        base.pop('clip_weights');base.pop('clip_sha256')
        return dict(base=base,scale_floor=m['scale_floor'])
    if capacity(qm['config']['model']) != capacity(manifest['config']['model']) or manifest['config']['model']['variant'] not in report.get('qualified_variants',()):
        raise ValueError('science model capacity/variant differs from qualified model')
    validate_value_qualification(qm['config']['evaluation']['value'],manifest['config']['evaluation']['value'],report)


def assert_evidence_authorized(*, repo_root, registry_path, manifest, operation, command=None):
    evaluation = manifest.get('config',{}).get('evaluation',{})
    marker = evaluation.get('config_kind') == KIND
    argv = command if command is not None else manifest.get('command',())
    entry = any(Path(str(a)).name in ENTRYPOINTS for a in argv)
    model = manifest.get('config',{}).get('model',{})
    structural = isinstance(model,dict) and {'base','variant','scale_floor'} <= set(model)
    named = manifest.get('study') == STUDY or STUDY in str(registry_path or '')
    if not (marker or entry or structural or named):
        return
    if not marker or manifest.get('study') != STUDY or operation not in {'reserve','start'} or repo_root is None or registry_path is None:
        raise ValueError('evidence entry/config requires its independent study and explicit marker')
    repo = Path(repo_root).resolve()
    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = repo/registry
    if registry != repo/CONTROL/'registry.jsonl' or manifest.get('registry_path') != CONTROL+'/registry.jsonl':
        raise ValueError('evidence registry namespace mismatch')
    if Path(manifest['artifact_path']).parents[2].name != STUDY:
        raise ValueError('evidence artifact namespace mismatch')
    study_path = registry.with_name('study.yaml')
    study = yaml.safe_load(_committed(repo,study_path))
    required = dict(study=STUDY,kind=KIND,protocol_version=1,status='registered',control_plane=CONTROL,
                    data_plane='artifacts/'+STUDY,registry=CONTROL+'/registry.jsonl',test_authorized=False)
    if not isinstance(study,dict) or any(study.get(k) != v for k,v in required.items()):
        raise ValueError('evidence study identity/status mismatch')
    if manifest.get('test_authorized') is not False or evaluation.get('eval_split') == 'test':
        raise ValueError('MSRVTT/MSVD test remain sealed')
    stage = evaluation.get('evidence_stage')
    env = manifest['environment']
    if env['precision'] != 'fp32':
        raise ValueError('evidence execution requires FP32')
    software = study.get('software_authorization',{})
    if stage == 'cpu_fixture':
        if evaluation.get('synthetic') is not True or not model['base']['clip']['fixture'] or env['gpu_count'] != 0 or software.get('cpu_fixture') is not True:
            raise ValueError('CPU qualification requires explicitly authorized tiny synthetic fixture')
        return
    if stage == 'gpu_qualification':
        if software.get('four_gpu') is not True or evaluation.get('synthetic') is not True or model['base']['clip']['fixture'] is not False or env['gpu_count'] != 4 or env['distributed_topology'] != 'torchrun-4':
            raise ValueError('GPU qualification requires authorized production capacity random weights, four CUDA ranks')
        if manifest['dataset'] != 'synthetic':
            raise ValueError('qualification cannot consume scientific data')
        if model['base']['clip_sha256'] != software.get('random_clip_sha256'):
            raise ValueError('qualification requires registered random CLIP weight bytes')
        from semantic_retrieval.models.clip import ClipConfig
        ClipConfig(**model['base']['clip'])
        return
    if stage not in {'reference','matcher','transitions','value','calibration','evaluation'}:
        raise ValueError('unregistered evidence stage')
    validate_qualification(repo,study,manifest)
    science = study.get('science',{})
    if science.get('authorized') is not True or stage not in science.get('authorized_stages',()):
        raise ValueError('scientific stage is not authorized by its study gate')
    gate = science.get('gates',{}).get(stage)
    if not isinstance(gate,dict) or gate.get('passed') is not True:
        raise ValueError('scientific predecessor gate has not passed')
    for report in gate.get('reports',()):
        path = repo/report['path']
        if hashlib.sha256(_committed(repo,path)).hexdigest() != report['sha256']:
            raise ValueError('scientific gate report identity mismatch')
    if not gate.get('reports'):
        raise ValueError('scientific gate requires committed auditable evidence')
