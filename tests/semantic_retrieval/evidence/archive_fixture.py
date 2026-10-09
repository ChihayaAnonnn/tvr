"""Temporary committed studies for real protected worker tests."""
import datetime
import json
import platform
import shutil
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts import manage_experiment_run as manager
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.evidence.config import evidence_manifest_config
from test_runtime_artifacts import git, commit_reservation


def reserve_evidence_fixture(tmp_path,cfg,*,copy_source=False,variant='cpu-fixture'):
    repo = tmp_path/'repo'; repo.mkdir()
    control = repo/'experiments/evidence-conditioned-retrieval-v1'; control.mkdir(parents=True)
    (repo/'.gitignore').write_text('/artifacts/\n')
    study = dict(study='evidence-conditioned-retrieval-v1',kind='evidence-retrieval-v1',protocol_version=1,status='registered',
                 control_plane='experiments/evidence-conditioned-retrieval-v1',data_plane='artifacts/evidence-conditioned-retrieval-v1',
                 registry='experiments/evidence-conditioned-retrieval-v1/registry.jsonl',test_authorized=False,
                 software_authorization=dict(cpu_fixture=True,four_gpu=False))
    (control/'study.yaml').write_text(yaml.safe_dump(study))
    registry = control/'registry.jsonl'; registry.touch()
    if copy_source:
        root = Path(__file__).resolve().parents[3]
        shutil.copytree(root/'src/semantic_retrieval',repo/'src/semantic_retrieval',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (repo/'scripts').mkdir()
        for file in ('experiment_artifacts.py','experiment_tracking.py'):
            shutil.copy2(root/file,repo/file)
        for file in ('qualify_evidence_retrieval.py','train_evidence_retrieval.py','compile_evidence_transitions.py','evaluate_evidence_retrieval.py','manage_experiment_run.py'):
            if (root/'scripts'/file).exists():
                shutil.copy2(root/'scripts'/file,repo/'scripts'/file)
    refs = {str(cfg.model.base.clip_weights):cfg.model.base.clip_sha256}
    refs.update({str(v.path):v.sha256 for f in fields(cfg.parents) if (v := getattr(cfg.parents,f.name)) is not None})
    if cfg.transitions.reference:
        refs[str(cfg.transitions.reference.path)] = cfg.transitions.reference.sha256
    checkpoints = [dict(path=k,sha256=v) for k,v in refs.items()]
    count = torch.cuda.device_count()
    command = [sys.executable,'scripts/qualify_evidence_retrieval.py','--config','{run_dir}/provenance/worker-config.json','--run-id','{run_id}']
    run = dict(dataset='synthetic',stage='baseline',variant=variant,seed=cfg.train.seed,
               scheduled_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
               environment=dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=str(torch.version.cuda or 'none'),
                                gpu_model=torch.cuda.get_device_name(0) if count else 'none',gpu_count=count,precision='fp32',distributed_topology='single-process'),
               input_checkpoints=checkpoints,dataset_splits=[],config=evidence_manifest_config(cfg,eval_split='val'),
               parent_run_ids=[],selection_lineage=[],validation_authorized=True,test_authorized=False,
               expected_scalar_outputs=[],array_references=[],command=command)
    phase = control/'phase.json'; phase.write_text(json.dumps(dict(phase='cpu-fixture',study=study['study'],runs=[run])))
    git(repo,'init','-q'); git(repo,'config','user.email','test@example.com'); git(repo,'config','user.name','Test')
    git(repo,'add','.'); git(repo,'commit','-qm','source and protocol')
    output = manager.reserve_phase(SimpleNamespace(repo_root=repo,phase=phase.relative_to(repo),registry=registry.relative_to(repo),artifact_root=repo/'artifacts'))
    identity = output['runs'][0]
    root = Path(identity['artifact_path'])
    manifest = json.loads((root/'manifest.json').read_text())
    rt = RuntimeConfig(control/'study.yaml',registry,root,identity['run_id'],identity['manifest_sha256'],manifest['git_sha'],'val')
    commit_reservation(repo,rt)
    return repo,replace(cfg,runtime=rt)
