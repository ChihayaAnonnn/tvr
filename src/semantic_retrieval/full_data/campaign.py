"""Native campaign: authenticated Q0, then frozen complete-data baseline pairs."""
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from experiment_artifacts import load_registry
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.models.clip import ClipConfig
from .authorization import STUDY, CONTROL, _committed, _terminal_report, validate_prerequisites
from .controller import campaign_lock, idle_devices, run_phase, _budget_rows, budget_projection, _read
from .runtime import FullDataSettings, manifest_config

ARTIFACTS = Path('/data2/hxj/project/UATVR/artifacts')
PRETRAINED = ARTIFACTS/'evidence-conditioned-retrieval-v1/synthetic/baseline/synthetic-baseline-pretrained-clip-input-seed713-20261010T002713Z-619e2ff1c63b/arrays/ViT-B-32.pt'
RANDOM = ARTIFACTS/'semantic-answerability-pilot-v1/synthetic/baseline/synthetic-baseline-pilot-q0-prep-seed000-20261001T104445Z-4ff307be570d/arrays/random-pilot-clip.npz'
FORMAL = ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)


def scientific_inputs(repo):
    study = yaml.safe_load((repo/CONTROL/'study.yaml').read_bytes())
    _, certificate, path = _terminal_report(repo, study['input_certification'], 'complete-preparation.json')
    preparation = _read(repo/CONTROL/'phases/P0.json')
    raw_indices = [r for r in preparation['runs'][0]['dataset_splits'] if 'D0-9000-videos-' in r['path']]
    if len(raw_indices) != 10: raise ValueError('full raw-video inventory indices are incomplete')
    return [*raw_indices, dict(path=str(path), sha256=file_sha256(path)), *certificate['input_indices']], path


def _commit(repo, files, message):
    subprocess.run(['git', 'add', *map(str, files)], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-m', message], cwd=repo, check=True)


def _spec(settings, devices, inputs, parents):
    script = 'qualify_full_data_retrieval.py' if settings.stage == 'gpu_qualification' else 'train_full_data_retrieval.py'
    command = ['/usr/bin/env', 'CUDA_VISIBLE_DEVICES='+','.join(devices), 'OMP_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
               sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=4',
               'scripts/'+script, '--run-dir', '{artifact_run_dir}']
    return dict(dataset='synthetic' if settings.stage == 'gpu_qualification' else 'msrvtt', stage='baseline',
                variant='full-data-q0' if settings.stage == 'gpu_qualification' else settings.method, seed=settings.seed,
                scheduled_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                environment=dict(python=platform.python_version(), pytorch=str(torch.__version__), cuda=str(torch.version.cuda),
                                 gpu_model='NVIDIA A800 80GB PCIe', gpu_count=4, precision='fp32', distributed_topology='torchrun-4'),
                input_checkpoints=[dict(path=str(settings.clip_weights), sha256=settings.clip_sha256)], dataset_splits=inputs,
                config=manifest_config(settings), parent_run_ids=parents, selection_lineage=[], validation_authorized=True,
                test_authorized=False, expected_scalar_outputs=[], array_references=[], command=command)


def _prepare_phase(repo, name, specifications):
    from scripts import manage_experiment_run as manager
    registry = Path(CONTROL)/'registry.jsonl'
    with campaign_lock(repo/registry.with_name('registry.jsonl.campaign.lock')):
        manager._require_clean(repo)
        study = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
        records = load_registry(repo/registry)
        terminal = {r['run_id'] for r in records if r['event'] in {'completed', 'failed', 'cancelled'}}
        if any(r['event'] == 'reserved' and r['run_id'] not in terminal for r in records):
            raise ValueError('another reserved full-data phase requires exact recovery, refusing duplicates')
        rows = _budget_rows(repo, study, manager)
        for spec in specifications:
            category = 'software' if spec['config']['evaluation']['synthetic'] else 'scientific'
            rows.append(dict(category=category, gpu_count=4, state='reserved', elapsed_seconds=None,
                             max_wall_seconds=study['campaign'][category]['max_run_wall_seconds']))
        totals = budget_projection(rows)
        if any(totals[k] > study['campaign'][k]['max_cumulative_gpu_hours'] for k in totals):
            raise ValueError('new complete phase would exceed conservative campaign GPU budget')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        phase = Path(CONTROL)/'phases'/f'{name}-{stamp}.json'
        if (repo/phase).exists(): raise FileExistsError('phase identity already exists')
        publish_json(repo/phase, dict(phase=f'{name}-{stamp}', study=STUDY, runs=specifications))
        _commit(repo, [phase], f'research: define frozen full-data phase {name}')
        reservation = manager.reserve_phase(SimpleNamespace(repo_root=repo, phase=phase, registry=registry, artifact_root=ARTIFACTS))
        _commit(repo, [registry], f'archive: reserve full-data phase {name}')
    return phase, reservation['runs']


def _wait_four(repo, deadline, source):
    while True:
        # No tracked edits or duplicate watcher may be silently incorporated while waiting.
        actual = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        if actual != source: raise ValueError('campaign source changed while waiting for resources')
        devices = idle_devices()
        if len(devices) >= 4: return devices[:4]
        if time.monotonic() >= deadline: raise TimeoutError('four free GPUs unavailable within registered resource wait')
        print(f'WAITING: four idle GPUs required; available={devices}; no scientific run started', flush=True)
        time.sleep(min(30, max(0, deadline-time.monotonic())))


def _open_gate(repo, reservations):
    registry = repo/CONTROL/'registry.jsonl'
    with campaign_lock(registry.with_name('registry.jsonl.campaign.lock')):
        study = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
        q = reservations[0]; path = Path(q['artifact_path'])/'provenance/full-data-qualification.json'
        study['four_gpu_qualification'] = dict(run_id=q['run_id'], manifest_sha256=q['manifest_sha256'], report_sha256=file_sha256(path))
        inputs, certificate = scientific_inputs(repo)
        settings = FullDataSettings('baseline', 'uatvr', PRETRAINED, file_sha256(PRETRAINED), FORMAL, 713,
                                    certificate, file_sha256(certificate))
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        validate_prerequisites(repo, study, dict(git_sha=head), settings)
        study['science'] = dict(authorized=True, authorized_stages=['baseline'],
                               reason='predeclared_gate_satisfied_authenticated_complete_inputs_and_new_four_cuda_rank_Q0',
                               test='sealed_no_implicit_authorization')
        (registry.with_name('study.yaml')).write_text(yaml.safe_dump(study, sort_keys=False))
        _commit(repo, [Path(CONTROL)/'study.yaml'], 'research: record passing full-data Q0 and open predefined baseline gate')


def _report(repo):
    rows = []
    for record in load_registry(repo/CONTROL/'registry.jsonl'):
        if record['event'] != 'completed': continue
        root = Path(record['artifact_path']); manifest = _read(root/'manifest.json')
        if manifest['config']['evaluation'].get('full_data_stage') != 'baseline': continue
        status = _read(root/'status.json'); name = 'metrics/selected_validation.json'
        if status['output_hashes'].get(name) != file_sha256(root/name): raise ValueError('selected baseline metrics changed')
        result = _read(root/'provenance/baseline-result.json')
        rows.append(dict(run_id=record['run_id'], method=result['method'], seed=result['seed'],
                         selected_epoch=result['selected_epoch'], metrics=_read(root/name)['summary_metrics']))
    aggregates = {}
    for method in ('uatvr', 'clip4clip'):
        values = [r['metrics']['t2v_r1'] for r in rows if r['method'] == method]
        aggregates[method] = dict(seeds=len(values), t2v_r1_mean=float(np.mean(values)),
                                  t2v_r1_std=float(np.std(values, ddof=1)) if len(values) > 1 else None) if values else None
    path = Path(CONTROL)/'reports/full-baselines.json'
    if (repo/path).exists(): raise FileExistsError('full baseline report already published')
    publish_json(repo/path, dict(schema_version='full-data-baseline-comparison-v1', rows=rows, aggregates=aggregates,
                                training_videos=8246, validation_videos=500, validation_queries=10000,
                                test_authorized=False, paper_test_not_directly_comparable=True,
                                new_method_hypothesis_supported=False))
    _commit(repo, [path], 'research: summarize complete full-data baseline validation results')


def main(repo):
    from scripts import manage_experiment_run as manager
    repo = Path(repo).resolve(); manager._require_clean(repo)
    study = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
    load_registry(repo/CONTROL/'registry.jsonl')
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    deadline = time.monotonic()+study['campaign']['max_resource_wait_seconds']
    devices = _wait_four(repo, deadline, source)
    if study['four_gpu_qualification'] is None:
        settings = FullDataSettings('gpu_qualification', 'both', RANDOM, file_sha256(RANDOM), FORMAL, 713)
        phase, reservations = _prepare_phase(repo, 'Q0', [_spec(settings, devices, [], [])])
        if run_phase(repo, phase, ARTIFACTS): return 1
        _open_gate(repo, reservations)
    for seed in study['seeds']:
        current = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
        inputs, certificate = scientific_inputs(repo)
        existing = []
        for record in load_registry(repo/CONTROL/'registry.jsonl'):
            if record['event'] == 'completed':
                m = _read(Path(record['artifact_path'])/'manifest.json')
                if m['config']['evaluation'].get('full_data_stage') == 'baseline' and m['seed'] == seed:
                    existing.append(m['variant'])
        if set(existing) == {'uatvr', 'clip4clip'}: continue
        if existing: raise ValueError('partial scientific seed requires diagnosed exact recovery')
        specs = [_spec(FullDataSettings('baseline', method, PRETRAINED, file_sha256(PRETRAINED), FORMAL, seed,
                                       certificate, file_sha256(certificate)), devices, inputs,
                       [current['input_certification']['run_id'], current['four_gpu_qualification']['run_id']])
                 for method in ('uatvr', 'clip4clip')]
        phase, _ = _prepare_phase(repo, f'B-seed{seed}', specs)
        if run_phase(repo, phase, ARTIFACTS): return 1
    _report(repo)
    return 0
