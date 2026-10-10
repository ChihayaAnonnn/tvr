import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import manage_experiment_run as manager
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.full_data.runtime import FullDataSettings, manifest_config
from semantic_retrieval.models.clip import ClipBackbone
from full_data.test_runtime import fixture_clip
from test_runtime_artifacts import git


def reserve_cpu_worker(tmp_path):
    source = Path(__file__).resolve().parents[3]
    repo = tmp_path/'repo'; repo.mkdir()
    (repo/'.gitignore').write_text('/artifacts/\n')
    shutil.copytree(source/'src/semantic_retrieval', repo/'src/semantic_retrieval', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (repo/'scripts').mkdir()
    for name in ('experiment_artifacts.py', 'experiment_tracking.py'):
        shutil.copy2(source/name, repo/name)
    for name in ('manage_experiment_run.py', 'qualify_full_data_retrieval.py', 'run_full_data_phase.py'):
        shutil.copy2(source/'scripts'/name, repo/'scripts'/name)
    control = repo/'experiments/full-data-retrieval-v1'; control.mkdir(parents=True)
    shutil.copy2(source/'experiments/full-data-retrieval-v1/study.yaml', control/'study.yaml')
    registry = control/'registry.jsonl'; registry.touch()
    weights = repo/'tiny.npz'
    clip = ClipBackbone(fixture_clip())
    np.savez(weights, **{k: v.detach().numpy() for k, v in clip.state_dict().items()})
    cfg = FullDataSettings('cpu_fixture', 'both', weights, file_sha256(weights), clip.config, 713)
    command = ['/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
               'CUBLAS_WORKSPACE_CONFIG=:4096:8', sys.executable, 'scripts/qualify_full_data_retrieval.py',
               '--run-dir', '{artifact_run_dir}']
    run = dict(dataset='synthetic', stage='baseline', variant='full-data-cpu-fixture', seed=713,
               scheduled_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
               environment=dict(python=platform.python_version(), pytorch=str(torch.__version__), cuda=str(torch.version.cuda or 'none'),
                                gpu_model='none', gpu_count=0, precision='fp32', distributed_topology='single-process'),
               input_checkpoints=[dict(path=str(weights), sha256=file_sha256(weights))], dataset_splits=[],
               config=manifest_config(cfg), parent_run_ids=[], selection_lineage=[], validation_authorized=True,
               test_authorized=False, expected_scalar_outputs=[], array_references=[], command=command)
    phase = control/'phase.json'; phase.write_text(json.dumps(dict(phase='cpu-worker', study='full-data-retrieval-v1', runs=[run])))
    git(repo, 'init', '-q'); git(repo, 'config', 'user.email', 'test@example.com'); git(repo, 'config', 'user.name', 'Test')
    git(repo, 'add', '.'); git(repo, 'commit', '-qm', 'source and synthetic protocol')
    reservation = manager.reserve_phase(SimpleNamespace(repo_root=repo, phase=phase.relative_to(repo), registry=registry.relative_to(repo), artifact_root=repo/'artifacts'))['runs'][0]
    git(repo, 'add', str(registry)); git(repo, 'commit', '-qm', 'reserved')
    identity = SimpleNamespace(repo_root=repo, registry=registry.relative_to(repo), run_dir=reservation['artifact_path'],
                               run_id=reservation['run_id'], manifest_sha256=reservation['manifest_sha256'])
    return repo, identity, manager.verify_run(identity)['command_argv']


def test_protected_real_cpu_worker_checkpoints_metrics_and_idempotence(tmp_path):
    repo, identity, command = reserve_cpu_worker(tmp_path)
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    root = Path(identity.run_dir)
    manifest, status = manager._load_run(root)
    assert status['state'] == 'validating'
    report = json.loads((root/'provenance/full-data-qualification.json').read_bytes())
    assert report['production_capacity'] is False and report['world_size'] == 1 and report['device_type'] == 'cpu'
    assert report['qualified_methods'] == ['uatvr', 'clip4clip'] and all(report['checks'].values())
    assert len(list((root/'checkpoints').glob('*.pt'))) == 2
    inventory = manager._all_output_hashes(root)
    duplicate = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert duplicate.returncode != 0
    assert manager._all_output_hashes(root) == inventory
    manager.transition_run(SimpleNamespace(**vars(identity), expected='validating', next_state='completed', exit_code=0, output=[]))
    summary = manager.finalize_phase(SimpleNamespace(repo_root=repo, phase='experiments/full-data-retrieval-v1/phase.json',
        registry=identity.registry, artifact_root=repo/'artifacts', summary='experiments/full-data-retrieval-v1/summary.json'))
    assert summary['runs'][0]['state'] == 'completed'


def test_actual_cpu_controller_closes_log_writer_then_archives_batch(tmp_path):
    source = Path(__file__).resolve().parents[3]
    repo, identity, _ = reserve_cpu_worker(tmp_path)
    # Controller source must be present in the original committed fixture tree.
    assert (repo/'scripts/run_full_data_phase.py').exists()
    result = subprocess.run([sys.executable, 'scripts/run_full_data_phase.py', '--phase',
                             'experiments/full-data-retrieval-v1/phase.json'], cwd=repo,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    root = Path(identity.run_dir)
    _, status = manager._load_run(root)
    assert status['state'] == 'completed'
    assert 'logs/worker.log' in status['output_hashes']
    assert file_sha256(root/'logs/worker.log') == status['output_hashes']['logs/worker.log']
    assert json.loads((root/'provenance/controller-result.json').read_bytes())['spawned'] is True
    assert 'finalize' in git(repo, 'log', '-1', '--format=%s')
    head = git(repo, 'rev-parse', 'HEAD')
    repeated = subprocess.run([sys.executable, 'scripts/run_full_data_phase.py', '--phase',
                               'experiments/full-data-retrieval-v1/phase.json'], cwd=repo,
                              capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stdout+repeated.stderr
    assert git(repo, 'rev-parse', 'HEAD') == head


def test_launch_guard_authenticates_real_process_lock_and_sealed_plan(tmp_path):
    from semantic_retrieval.full_data import controller
    from semantic_retrieval.evidence.frame_cache import publish_json
    from experiment_artifacts import manifest_sha256
    repo, identity, _ = reserve_cpu_worker(tmp_path)
    root = Path(identity.run_dir); manifest, _ = manager._load_run(root)
    plan = dict(manifest_sha256=manifest_sha256(manifest), source_sha=manifest['git_sha'],
                command_sha256=manifest['command_sha256'], controller_sha256=file_sha256(Path(controller.__file__)),
                gpu_count=0, max_wall_seconds=1800)
    publish_json(root/'provenance/controller-plan.json', plan)
    publish_json(root/'provenance/controller-launch-intent.json', dict(**plan, controller_pid=os.getpid(),
                 controller_start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')', 1)[1].split()[19]))
    lock = repo/'experiments/full-data-retrieval-v1/registry.jsonl.campaign.lock'
    with controller.campaign_lock(lock): controller.validate_controller_launch(repo, root, manifest)
    import pytest
    with pytest.raises(ValueError, match='active locked'):
        controller.validate_controller_launch(repo, root, manifest)


def test_controller_sigterm_stops_owned_worker_before_terminal_inventory(tmp_path):
    from semantic_retrieval.full_data.controller import _group_alive
    repo, identity, _ = reserve_cpu_worker(tmp_path)
    root = Path(identity.run_dir)
    process = subprocess.Popen([sys.executable, 'scripts/run_full_data_phase.py', '--phase',
                               'experiments/full-data-retrieval-v1/phase.json'], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    launch = root/'provenance/controller-launch.json'; worker = None
    try:
        deadline = time.monotonic()+15
        while not launch.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert launch.exists()
        worker = json.loads(launch.read_bytes())['pid']
        process.send_signal(signal.SIGTERM)
        output = process.communicate(timeout=30)[0]
        assert not _group_alive(worker), output
        _, status = manager._load_run(root)
        assert status['state'] in {'failed', 'cancelled'}
        assert file_sha256(root/'logs/worker.log') == status['output_hashes']['logs/worker.log']
        receipt = json.loads((root/'provenance/controller-result.json').read_bytes())
        assert receipt['spawned'] is True and receipt['reason'] == 'controller_or_launch_failure'
    finally:
        if process.poll() is None: process.kill(); process.wait(timeout=5)
        if worker is not None and _group_alive(worker): os.killpg(worker, signal.SIGKILL)
