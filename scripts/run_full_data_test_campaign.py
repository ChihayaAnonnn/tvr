#!/usr/bin/env python3
"""Native one-shot final tests after matching-source Q0, never retrain/select on test."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def main(repo):
    import json,time,subprocess,yaml
    from experiment_artifacts import load_registry
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.evidence.frame_cache import publish_json
    from semantic_retrieval.full_data.authorization import CONTROL,_committed
    from semantic_retrieval.full_data.campaign import _wait_devices,_prepare_phase,_spec,_open_gate,RANDOM,FORMAL,_commit,ARTIFACTS
    from semantic_retrieval.full_data.controller import run_phase
    from semantic_retrieval.full_data.runtime import FullDataSettings
    from semantic_retrieval.full_data.final import authenticated_parent,final_config,ENTRY
    from scripts import manage_experiment_run as manager
    manager._require_clean(repo);study=yaml.safe_load(_committed(repo,repo/CONTROL/'study.yaml'));load_registry(repo/CONTROL/'registry.jsonl')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    devices=_wait_devices(repo,time.monotonic()+study['campaign']['max_resource_wait_seconds'],source,study)
    if study['gpu_qualification'] is None:
        settings=FullDataSettings('gpu_qualification','both',RANDOM,file_sha256(RANDOM),FORMAL,713,world_size=2)
        phase,reservations=_prepare_phase(repo,'Q0-final-test',[_spec(settings,devices,[],[])])
        if run_phase(repo,phase,ARTIFACTS):return 1
        _open_gate(repo,reservations)
    study=yaml.safe_load(_committed(repo,repo/CONTROL/'study.yaml'));p=study['final_test'];specs=[]
    completed=[]
    for record in load_registry(repo/CONTROL/'registry.jsonl'):
        if record['event']=='completed':
            m=json.loads((Path(record['artifact_path'])/'manifest.json').read_text())
            if m['config']['evaluation'].get('config_kind')=='full-data-test-v1':completed.append(record)
    if completed:raise ValueError('final test already evaluated; refusing duplicate test exposure')
    for frozen in p['selected']:
        parent,selected=authenticated_parent(repo,p,frozen['run_id']);from semantic_retrieval.full_data.runtime import settings_from_manifest
        settings=settings_from_manifest(parent);spec=_spec(settings,devices,[],[])
        ckpt=selected['selected_checkpoint'];spec.update(stage='final',variant='test-'+settings.method,
            config=final_config(parent,p,selected),parent_run_ids=[parent['run_id'],study['gpu_qualification']['run_id']],
            dataset_splits=[p['test_csv'],*p['video_indices']],input_checkpoints=[dict(path=str(settings.clip_weights),sha256=settings.clip_sha256),dict(path=ckpt['path'],sha256=ckpt['sha256']),dict(path=str(Path(ckpt['path']).with_suffix('.json')),sha256=ckpt['sidecar_sha256'])],test_authorized=True,validation_authorized=False)
        spec['command']=[a.replace('train_full_data_retrieval.py',ENTRY) for a in spec['command']];specs.append(spec)
    phase,_=_prepare_phase(repo,'T-jsfusion-frozen-six',specs)
    if run_phase(repo,phase,ARTIFACTS):return 1
    rows=[]
    for record in load_registry(repo/CONTROL/'registry.jsonl'):
        if record['event']!='completed':continue
        root=Path(record['artifact_path']);m=json.loads((root/'manifest.json').read_text())
        if m['config']['evaluation'].get('config_kind')!='full-data-test-v1':continue
        s=json.loads((root/'status.json').read_text());name='metrics/final_test.json'
        if s['output_hashes'].get(name)!=file_sha256(root/name):raise ValueError('final test result changed')
        rows.append(dict(run_id=record['run_id'],method=m['config']['model']['method'],seed=m['seed'],metrics=json.loads((root/name).read_text())['summary_metrics']))
    import numpy as np
    aggregates={method:{metric:dict(mean=float(np.mean([r['metrics'][metric] for r in rows if r['method']==method])),seed_sd=float(np.std([r['metrics'][metric] for r in rows if r['method']==method],ddof=1))) for metric in ('t2v_r1','v2t_r1')} for method in ('uatvr','clip4clip')}
    path=Path(CONTROL)/'reports/jsfusion-final-test.json'
    if (repo/path).exists():raise FileExistsError('final report exists')
    publish_json(repo/path,dict(schema_version='full-data-jsfusion-comparison-v1',rows=rows,aggregates=aggregates,test_authorized=True,dsl=False,training_videos=8246,test_video_count=1000,test_query_count=1000,checkpoint_selection='validation_only',paper_training_protocol_differs=True,new_method_hypothesis_supported=False))
    _commit(repo,[path],'research: summarize frozen JSFUSION 1k final test results')
    return 0

if __name__=='__main__':raise SystemExit(main(ROOT))
