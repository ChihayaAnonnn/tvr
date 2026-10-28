"""Real temporary Scheme A lifecycle tests for the read-only analysis verifier."""
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_manage_experiment_run import campaign_repo,_reserve,_run,REGISTRY
from scripts import manage_experiment_run as manager


def reader():
    path=Path(__file__).resolve().parents[1]/'experiments/evidence-conditioned-retrieval-v1/analysis/archive_reader.py'
    import sys
    spec=importlib.util.spec_from_file_location('evidence_archive_reader',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


@pytest.mark.parametrize('tamper',[None,'output','summary','phase','source'])
def test_real_archive_certificate_rejects_tampering(campaign_repo,tamper):
    repo,phase,artifact_root,payload=_reserve(campaign_repo);r=payload['runs'][0];root=Path(r['artifact_path'])
    def commit():
        subprocess.run(['git','add','-u'],cwd=repo,check=True)
        if (repo/REGISTRY.parent/'summaries/stage0-smoke.json').exists():subprocess.run(['git','add',str(REGISTRY.parent/'summaries/stage0-smoke.json')],cwd=repo,check=True)
        subprocess.run(['git','commit','-qm','archive fixture'],cwd=repo,check=True)
    commit()
    identity=SimpleNamespace(repo_root=repo,registry=REGISTRY,run_dir=root,run_id=r['run_id'],manifest_sha256=r['manifest_sha256'])
    manager.start_run(identity)
    (root/'provenance/value.json').write_text('{"value":1}')
    for before,after,code in [('running','validating',None),('validating','completed',0)]:
        manager.transition_run(SimpleNamespace(**vars(identity),expected=before,next_state=after,exit_code=code,output=[]))
    summary=REGISTRY.parent/'summaries/stage0-smoke.json'
    manager.finalize_phase(SimpleNamespace(repo_root=repo,phase=phase,registry=REGISTRY,artifact_root=artifact_root,summary=summary));commit()
    read=reader();args=(repo,repo/REGISTRY.parent,phase,summary,artifact_root)
    if tamper=='output':(root/'provenance/value.json').write_text('{"value":2}')
    elif tamper=='summary':
        p=repo/summary;data=json.loads(p.read_text());data['runs'][0]['output_hashes']['provenance/value.json']='a'*64;p.write_text(json.dumps(data));commit()
    elif tamper in {'phase','source'}:
        p=repo/phase;data=json.loads(p.read_text())
        if tamper=='phase':data['runs'][0]['config']['evaluation']['other']=True
        else:data['phase']='changed-phase'
        p.write_text(json.dumps(data));commit()
    if tamper is not None:
        with pytest.raises(ValueError):read.certify_phase(*args)
    else:
        run,=read.certify_phase(*args);assert run.read_json('provenance/value.json')=={'value':1}
