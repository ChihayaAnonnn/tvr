import json
import os
import subprocess
import shlex
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest

from evidence.archive_fixture import reserve_evidence_fixture


@pytest.fixture(scope='function')
def cpu_qualification_runs(tmp_path,config_factory):
    cfg = config_factory(stage='cpu_fixture')
    repo,cfg = reserve_evidence_fixture(tmp_path,cfg,copy_source=True)
    config = cfg.runtime.run_dir/'provenance/worker-config.json'
    config.parent.mkdir(exist_ok=True)
    config.write_text(json.dumps(cfg.to_payload()))
    with (cfg.runtime.run_dir/'manifest.json').open() as h:
        manifest = json.load(h)
    proc = subprocess.run(shlex.split((cfg.runtime.run_dir/'command.sh').read_text()),cwd=repo,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
    assert proc.returncode == 0,proc.stdout+proc.stderr
    from scripts import manage_experiment_run as manager
    from test_runtime_artifacts import git
    control = Path('experiments/evidence-conditioned-retrieval-v1')
    args = dict(repo_root=repo,registry=control/'registry.jsonl',run_dir=cfg.runtime.run_dir,run_id=cfg.runtime.run_id,manifest_sha256=cfg.runtime.manifest_sha256)
    manager.verify_run(SimpleNamespace(**args))
    manager.transition_run(SimpleNamespace(**args,expected='validating',next_state='completed',exit_code=0,output=[]))
    manager.finalize_phase(SimpleNamespace(repo_root=repo,registry=control/'registry.jsonl',phase=control/'phase.json',summary=control/'summary.json',artifact_root=repo/'artifacts'))
    git(repo,'add',str(control));git(repo,'commit','-qm','terminal batch')
    status = json.loads((cfg.runtime.run_dir/'status.json').read_text())
    report = json.loads((cfg.runtime.run_dir/'provenance/evidence-qualification.json').read_text())
    return {**report['continuous'],'status':status['state'],'checks':report['checks'],'scientific_authorized':report['scientific_authorized'],'test_authorized':report['test_authorized']},{**report['resumed'],'status':status['state']}


def test_protected_cpu_pipeline_and_resume(cpu_qualification_runs):
    continuous,resumed = cpu_qualification_runs
    assert continuous['status'] == resumed['status'] == 'completed'
    for k in ('model_sha256','action_ids','cost_counts','initial_ranking','ranking'):
        assert continuous[k] == resumed[k]
    for k in ('hidden_evidence_isolated','scale_affects_prediction','teacher_frozen','real_clip_forward','loc_gradient','scale_gradient','value_gradient','exact_resume','action_recovery','finite_fp32'):
        assert continuous['checks'][k] is True
    assert continuous['scientific_authorized'] is False
    assert continuous['test_authorized'] is False


def test_failed_worker_records_failure_without_terminal_overwrite(tmp_path,config_factory):
    cfg = config_factory(stage='cpu_fixture')
    # This budget cannot reserve the initial three-video view; failure precedes decoding.
    from semantic_retrieval.evidence.types import ResourceCounts
    cfg = replace(cfg,policy=replace(cfg.policy,limits=ResourceCounts(1,1,1,1)))
    repo,cfg = reserve_evidence_fixture(tmp_path,cfg,copy_source=True)
    path = cfg.runtime.run_dir/'provenance/worker-config.json';path.write_text(json.dumps(cfg.to_payload()))
    proc = subprocess.run(shlex.split((cfg.runtime.run_dir/'command.sh').read_text()),cwd=repo,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
    assert proc.returncode != 0
    failure = json.loads((cfg.runtime.run_dir/'provenance/worker_failure.json').read_text())
    assert failure['exit_code'] == 1
    from scripts import manage_experiment_run as manager
    control = Path('experiments/evidence-conditioned-retrieval-v1')
    args = dict(repo_root=repo,registry=control/'registry.jsonl',run_dir=cfg.runtime.run_dir,run_id=cfg.runtime.run_id,manifest_sha256=cfg.runtime.manifest_sha256)
    manager.verify_run(SimpleNamespace(**args))
    manager.transition_run(SimpleNamespace(**args,expected='running',next_state='validating',exit_code=None,output=[]))
    manager.transition_run(SimpleNamespace(**args,expected='validating',next_state='failed',exit_code=1,output=[]))
    assert json.loads((cfg.runtime.run_dir/'status.json').read_text())['state'] == 'failed'


def test_protected_four_gloo_ranks_train_and_restore(tmp_path,config_factory):
    cfg = config_factory(stage='cpu_fixture')
    repo,cfg = reserve_evidence_fixture(tmp_path,cfg,copy_source=True,world_size=4)
    path = cfg.runtime.run_dir/'provenance/worker-config.json';path.write_text(json.dumps(cfg.to_payload()))
    proc = subprocess.run(shlex.split((cfg.runtime.run_dir/'command.sh').read_text()),cwd=repo,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
    assert proc.returncode == 0,proc.stdout+proc.stderr
    report = json.loads((cfg.runtime.run_dir/'provenance/evidence-qualification.json').read_text())
    assert report['world_size'] == 4 and report['device_type'] == 'cpu'
    assert report['production_capacity'] is False
    assert all(report['checks'][k] for k in ('unknown_rank','rank_weights_equal','exact_resume','loc_gradient','scale_gradient','value_gradient','action_recovery'))
    from semantic_retrieval.evidence.checkpoint import load_evidence_model_weights,evidence_config_sha256,load_evidence_checkpoint
    from semantic_retrieval.evidence.model import EvidenceMatcherModel
    from semantic_retrieval.runtime.checkpoint import model_sha256
    import torch
    checkpoint = next((cfg.runtime.run_dir/'checkpoints').glob('*.pt'))
    model = EvidenceMatcherModel(cfg.model)
    rng = torch.get_rng_state().clone()
    load_evidence_model_weights(checkpoint,{'matcher':model},expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)
    sidecar = json.loads(checkpoint.with_suffix('.json').read_text())
    assert model_sha256(model) == sidecar['model_sha256']['matcher']
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError,match='world size'):
        load_evidence_checkpoint(checkpoint,{'matcher':model},expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)
