#!/usr/bin/env python3
"""Finite validation-only single-factor campaign using the existing guarded controller."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import yaml
from experiment_artifacts import load_registry,load_validated_run_lifecycle,make_run_paths,manifest_sha256
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.full_data.authorization import CONTROL,_terminal_report,_committed
from semantic_retrieval.full_data.campaign import (_spec,_prepare_phase,_open_gate,_wait_devices,_commit,
    scientific_inputs,PRETRAINED,RANDOM,FORMAL,ARTIFACTS)
from semantic_retrieval.full_data.controller import run_phase,_read,campaign_lock
from semantic_retrieval.full_data.runtime import FullDataSettings
from semantic_retrieval.full_data.factors import ARMS,validate_terminal_binding
from semantic_retrieval.evidence.frame_cache import publish_json


def completed():
    _committed(ROOT,ROOT/CONTROL/'registry.jsonl')
    found={};failures=[]
    for r in load_registry(ROOT/CONTROL/'registry.jsonl'):
        if r['event'] not in {'completed','failed','cancelled'}:continue
        root=Path(r['artifact_path'])
        m=_read(root/'manifest.json')
        if manifest_sha256(m)!=r['manifest_sha256']:raise ValueError('registry manifest changed')
        ev=m['config']['evaluation']
        if ev.get('full_data_stage')!='baseline' or ev.get('factor_arm') is None:continue
        m,status,_=load_validated_run_lifecycle(make_run_paths(ARTIFACTS,'full-data-retrieval-v1',r['run_id']))
        matches=[]
        for path in (ROOT/CONTROL/'summaries').glob('*.json'):
            document=json.loads(_committed(ROOT,path))
            matches.extend(x for x in document.get('runs',[]) if x['run_id']==r['run_id'])
        if len(matches)!=1:raise ValueError('factor terminal needs one committed phase summary')
        validate_terminal_binding(r,m,status,matches[0])
        key=(m['config']['model']['method'],ev['factor_arm'],m['seed'])
        if r['event']=='completed':
            if key in found:raise ValueError('duplicate successful factor run')
            found[key]=(r,m)
        else:failures.append(key)
    if any(k not in found for k in failures):raise ValueError('failed factor needs diagnosed new-identity recovery; no blind retry')
    return found


def report():
    rows=[]
    for (method,arm,seed),(record,manifest) in sorted(completed().items()):
        root=Path(record['artifact_path']);status=_read(root/'status.json')
        def bound(name):
            p=root/name
            if status['output_hashes'].get(name)!=file_sha256(p):raise ValueError('factor output changed')
            return _read(p)
        epochs=[bound(f'provenance/epoch-{i:02d}.json') for i in range(1,6)]
        selected=bound('metrics/selected_validation.json')
        trajectory=[e['metrics']['summary_metrics'] for e in epochs]
        rows.append(dict(method=method,arm=arm,seed=seed,run_id=record['run_id'],
            manifest_sha256=record['manifest_sha256'],epoch5=trajectory[-1],
            selected=selected['summary_metrics'],trajectory=trajectory))
    if len(rows)!=6:raise ValueError('six completed controls/treatments required')
    comparisons=[]
    for method in ('uatvr','clip4clip'):
        control=next(r for r in rows if r['method']==method and r['arm']=='control')
        for r in rows:
            if r['method']!=method or r['arm']=='control':continue
            comparisons.append(dict(method=method,arm=r['arm'],
                epoch5_t2v_r1_delta_pp=r['epoch5']['t2v_r1']-control['epoch5']['t2v_r1'],
                epoch5_v2t_r1_delta_pp=r['epoch5']['v2t_r1']-control['epoch5']['v2t_r1'],
                selected_t2v_r1_delta_pp=r['selected']['t2v_r1']-control['selected']['t2v_r1']))
    target=Path(CONTROL)/'reports/validation-factors-seed713.json'
    payload=dict(schema_version=1,test_access=False,split='validation',dsl=False,
        primary='fixed_epoch5_raw_t2v_r1',single_seed_exploratory=True,rows=rows,comparisons=comparisons)
    if (ROOT/target).exists():
        if _read(ROOT/target)!=payload:raise ValueError('existing factor report differs')
        return
    publish_json(ROOT/target,payload);_commit(ROOT,[target],'research: report all single-factor validation outcomes')


def _main():
    from scripts import manage_experiment_run as manager
    manager._require_clean(ROOT)
    existing=completed()
    if len(existing)>6:raise ValueError('factor run limit exceeded')
    if len(existing)==6:report();return 0
    study=yaml.safe_load((ROOT/CONTROL/'study.yaml').read_text())
    inputs,certificate=scientific_inputs(ROOT)
    def settings(stage,method,arm):
        synthetic=stage=='gpu_qualification';weights=RANDOM if synthetic else PRETRAINED
        return FullDataSettings(stage,method,weights,file_sha256(weights),FORMAL,713,
            None if synthetic else certificate,None if synthetic else file_sha256(certificate),
            world_size=2,factor_arm=arm)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    devices=_wait_devices(ROOT,time.monotonic()+study['campaign']['max_resource_wait_seconds'],head,study)
    # Reuse only a fully authenticated, matching-source qualification of new paths.
    qualified=False
    q=study.get('gpu_qualification')
    if q:
        qm,qr,_=_terminal_report(ROOT,q,'full-data-qualification.json')
        changed=subprocess.check_output(['git','diff','--name-only',qm['git_sha'],head,'--',
            'src/semantic_retrieval','scripts','experiment_artifacts.py','experiment_tracking.py'],cwd=ROOT,text=True).strip()
        qualified=not changed and qr.get('factor_paths_qualified') is True
    if not qualified:
        phase,runs=_prepare_phase(ROOT,'VF-Q0',[_spec(settings('gpu_qualification','both','caption_shuffle'),devices,[],[])])
        if run_phase(ROOT,phase,ARTIFACTS):return 1
        _open_gate(ROOT,runs)
    for method in ('uatvr','clip4clip'):
        for arm in ARMS:
            if (method,arm,713) in existing:continue
            study=yaml.safe_load((ROOT/CONTROL/'study.yaml').read_text())
            spec=_spec(settings('baseline',method,arm),devices,inputs,
                [study['input_certification']['run_id'],study['gpu_qualification']['run_id']])
            spec['variant']=method+'-'+arm
            spec['selection_lineage']=[CONTROL+'/VALIDATION_FACTORS.md']
            phase,_=_prepare_phase(ROOT,'VF-'+method+'-'+arm,[spec])
            if run_phase(ROOT,phase,ARTIFACTS):return 1
    report();return 0


def main():
    with campaign_lock(ROOT/CONTROL/'validation-factors.campaign.lock'):
        return _main()


if __name__=='__main__':raise SystemExit(main())
