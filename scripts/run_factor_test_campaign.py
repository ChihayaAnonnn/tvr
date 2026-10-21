#!/usr/bin/env python3
"""One-shot authorized test of completed, validation-selected factor models."""
from pathlib import Path
import json
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiment_artifacts import load_registry
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.full_data.authorization import CONTROL,_committed,_terminal_report
from semantic_retrieval.full_data.campaign import _wait_devices,_prepare_phase,_spec,_open_gate,RANDOM,FORMAL,ARTIFACTS
from semantic_retrieval.full_data.controller import run_phase,campaign_lock
from semantic_retrieval.full_data.runtime import FullDataSettings,settings_from_manifest
from semantic_retrieval.full_data.final import authenticated_parent,final_config,ENTRY,test_variant


def main():
    from scripts import manage_experiment_run as manager
    with campaign_lock(ROOT/CONTROL/'factor-test.campaign.lock'):
        manager._require_clean(ROOT)
        study=yaml.safe_load(_committed(ROOT,ROOT/CONTROL/'study.yaml'))
        permission=study['final_test_factors']
        # Authenticate the complete frozen batch before any GPU reservation.
        parents=[authenticated_parent(ROOT,permission,f['run_id']) for f in permission['selected']]
        records=load_registry(ROOT/CONTROL/'registry.jsonl')
        selected_ids={p['run_id'] for p,s in parents}
        prior=[]
        for r in records:
            if r['event'] not in {'completed','failed','cancelled'}:continue
            m=json.loads((Path(r['artifact_path'])/'manifest.json').read_text())
            ev=m['config']['evaluation']
            if ev.get('config_kind')=='full-data-test-v1' and ev.get('parent_run_id') in selected_ids:prior.append(r)
        if prior:raise ValueError('factor tests already attempted; inspect archived outcomes, never blindly repeat')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        devices=_wait_devices(ROOT,time.monotonic()+study['campaign']['max_resource_wait_seconds'],head,study)
        qm,qr,_=_terminal_report(ROOT,study['gpu_qualification'],'full-data-qualification.json')
        changed=subprocess.check_output(['git','diff','--name-only',qm['git_sha'],head,'--','src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py'],cwd=ROOT,text=True).strip()
        if changed or qr.get('factor_paths_qualified') is not True:
            settings=FullDataSettings('gpu_qualification','both',RANDOM,file_sha256(RANDOM),FORMAL,713,world_size=2,factor_arm='caption_shuffle')
            phase,reservations=_prepare_phase(ROOT,'Q0-factor-test',[_spec(settings,devices,[],[])])
            if run_phase(ROOT,phase,ARTIFACTS):return 1
            _open_gate(ROOT,reservations)
        study=yaml.safe_load(_committed(ROOT,ROOT/CONTROL/'study.yaml'));specs=[]
        for parent,selected in parents:
            settings=settings_from_manifest(parent);spec=_spec(settings,devices,[],[]);ckpt=selected['selected_checkpoint']
            spec.update(stage='final',variant=test_variant(settings),config=final_config(parent,permission,selected),
                parent_run_ids=[parent['run_id'],study['gpu_qualification']['run_id']],
                dataset_splits=[permission['test_csv'],*permission['video_indices']],
                input_checkpoints=[dict(path=str(settings.clip_weights),sha256=settings.clip_sha256),dict(path=ckpt['path'],sha256=ckpt['sha256']),dict(path=str(Path(ckpt['path']).with_suffix('.json')),sha256=ckpt['sidecar_sha256'])],
                selection_lineage=[CONTROL+'/FACTOR_TEST_PROTOCOL.md'],test_authorized=True,validation_authorized=False)
            spec['command']=[a.replace('train_full_data_retrieval.py',ENTRY) for a in spec['command']];specs.append(spec)
        phase,_=_prepare_phase(ROOT,'T-jsfusion-completed-factors',specs)
        return run_phase(ROOT,phase,ARTIFACTS)


if __name__=='__main__':raise SystemExit(main())
