"""Reserve and launch the scoped two-card corrected HVP-Net reproduction."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from types import SimpleNamespace
import torch
import yaml
from experiment_artifacts import load_registry,manifest_sha256
from scripts import manage_experiment_run as manager
from scripts.local_query_campaign import commit,devices,read
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.campaign import RANDOM,PRETRAINED

REPO=Path(__file__).resolve().parents[1]
STUDY='hvp-net-clean-evaluation-v1';CONTROL=Path('experiments')/STUDY
ARTIFACTS=Path('/data2/hxj/project/UATVR/artifacts')
HVP=REPO/'research_refs/HVP-Net'


def shared_devices(study):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    used={x.split(',')[0].strip():int(x.split(',')[1]) for x in rows}
    policy=study['resource_policy'];chosen=policy['gpu_uuids']
    if any(used.get(x,999999)>policy['max_existing_memory_MiB'] for x in chosen):
        raise ValueError('insufficient reserved memory for read-only shared evaluation')
    return chosen


def source_files():
    names=subprocess.check_output(['git','ls-files'],cwd=HVP,text=True).splitlines()
    paths=[HVP/n for n in names]
    paths += [p.resolve() for p in (HVP/'.local-deps').rglob('*.py')]
    paths += [REPO/'scripts/hvp_clean_eval_worker.py',REPO/'scripts/hvp_clean_eval_campaign.py']
    return {str(p):file_sha256(p) for p in paths}


def prepare(mode):
    manager._require_clean(REPO);manager._require_clean(HVP)
    study=yaml.safe_load((REPO/CONTROL/'study.yaml').read_text());records=load_registry(REPO/CONTROL/'registry.jsonl')
    terminal={r['run_id'] for r in records if r['event'] in ('completed','failed','cancelled')}
    if any(r['event']=='reserved' and r['run_id'] not in terminal for r in records):raise ValueError('unresolved reservation')
    attempts=[r for r in records if r['event']=='reserved' and ('hvp-q0' if mode=='qualify' else 'hvp-clean-eval') in r['run_id']]
    if len(attempts)>=study['budget']['max_qualification_attempts' if mode=='qualify' else 'max_evaluation_attempts']:raise ValueError('attempt budget exhausted')
    chosen=shared_devices(study);files=source_files();parents=[]
    ev=dict(mode=mode,selected=read(REPO/CONTROL/'inputs/selected.json'),dependency_root=str((HVP/'.local-deps').resolve()),hvp_root=str(HVP),hvp_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HVP,text=True).strip(),source_files=files,qualification=None)
    if mode=='qualify' and attempts:parents=[attempts[-1]['run_id']]
    if mode=='evaluate':
        done=[r for r in records if r['event']=='completed' and 'hvp-q0' in r['run_id']
              and read(Path(r['artifact_path'])/'provenance/result.json').get('source_files')==files]
        if len(done)!=1:raise ValueError('one completed qualification required')
        root=Path(done[0]['artifact_path']);report=root/'provenance/result.json';q=read(report)
        if q.get('passed') is not True or q['source_files']!=files:raise ValueError('qualification source/config mismatch')
        ev['qualification']=dict(path=str(report),sha256=file_sha256(report),run_id=done[0]['run_id']);parents=[done[0]['run_id']]
    inputs=read(REPO/CONTROL/'inputs/test-references.json') if mode=='evaluate' else []
    weights=PRETRAINED if mode=='evaluate' else RANDOM
    selected=read(REPO/CONTROL/'inputs/selected.json')
    command=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+','.join(chosen),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src:.',sys.executable,
        '-m','torch.distributed.run','--standalone','--nproc_per_node=2','--log-dir','{artifact_run_dir}/logs/elastic','scripts/hvp_clean_eval_worker.py','--run-dir','{artifact_run_dir}']
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec=dict(dataset='msrvtt' if mode=='evaluate' else 'synthetic',stage='final' if mode=='evaluate' else 'baseline',variant='hvp-clean-eval-attempt'+str(len(attempts)+1) if mode=='evaluate' else 'hvp-q0-attempt'+str(len(attempts)+1),seed=42,scheduled_timestamp_utc=stamp,
        environment=dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=str(torch.version.cuda),gpu_model='NVIDIA A800 80GB PCIe',gpu_count=2,precision='mixed_fp16',distributed_topology='torchrun-2'),
        input_checkpoints=[dict(path=str(weights),sha256=file_sha256(weights)),dict(path=selected['path'],sha256=selected['sha256'])],dataset_splits=inputs,
        config=dict(model=study['recipe'],loss={},optimizer={},scheduler={},evaluation=ev),
        parent_run_ids=parents,selection_lineage=[dict(report_path=str(CONTROL/'PROTOCOL.md'),report_sha256=file_sha256(REPO/CONTROL/'PROTOCOL.md'),reason='User-authorized fixed epoch1 evaluation-only repair; no candidate caption input or checkpoint sweep.')],
        validation_authorized=False,test_authorized=mode=='evaluate',expected_scalar_outputs=[],array_references=[],command=command)
    phase=CONTROL/'phases'/f'{mode}-{stamp}.json';publish_json(REPO/phase,dict(phase=f'{mode}-{stamp}',study=STUDY,runs=[spec]))
    commit(REPO,[phase],f'research: freeze HVP {mode}')
    reservation=manager.reserve_phase(SimpleNamespace(repo_root=str(REPO),phase=str(phase),registry=str(CONTROL/'registry.jsonl'),artifact_root=str(ARTIFACTS)))
    commit(REPO,[CONTROL/'registry.jsonl'],f'archive: reserve HVP {mode}')
    root=Path(reservation['runs'][0]['artifact_path']);publish_json(root/'provenance/campaign.json',dict(phase=str(phase),mode=mode))
    print('RUN_DIR '+str(root),flush=True);return root


def execute(root):
    m=read(root/'manifest.json');mode=m['config']['evaluation']['mode']
    args=SimpleNamespace(repo_root=str(REPO),registry=m['registry_path'],run_dir=str(root),run_id=m['run_id'],manifest_sha256=manifest_sha256(m))
    command=manager.verify_run(args)['command_argv'];study=yaml.safe_load((REPO/CONTROL/'study.yaml').read_text());shared_devices(study)
    manager.start_run(args)
    logfile=root/'logs'/('qualification.log' if mode=='qualify' else 'evaluate.log')
    with logfile.open('x') as log:
        child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        publish_json(root/'provenance/process.json',dict(controller_pid=os.getpid(),torchrun_pid=child.pid,log_path=str(logfile)))
        print('LOG '+str(logfile),flush=True)
        try:code=child.wait(timeout=study['budget']['qualify_wall_seconds' if mode=='qualify' else 'evaluate_wall_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    manager.verify_run(args)
    if code==0 and read(root/'provenance/result.json').get('passed' if mode=='qualify' else 'completed') is not True:code=1
    if code==0:
        manager.transition_run(SimpleNamespace(**vars(args),expected='running',next_state='validating',exit_code=None,output=[]))
        manager.transition_run(SimpleNamespace(**vars(args),expected='validating',next_state='completed',exit_code=0,output=[]))
    else:
        manager.transition_run(SimpleNamespace(**vars(args),expected='running',next_state='validating',exit_code=None,output=[]))
        manager.transition_run(SimpleNamespace(**vars(args),expected='validating',next_state='failed',exit_code=code,output=[]))
    phase=read(root/'provenance/campaign.json')['phase'];summary=CONTROL/'summaries'/(Path(phase).stem+'.json')
    manager.finalize_phase(SimpleNamespace(repo_root=str(REPO),phase=phase,registry=m['registry_path'],artifact_root=str(ARTIFACTS),summary=str(summary)))
    commit(REPO,[CONTROL/'registry.jsonl',summary],f'archive: finalize HVP {mode}')
    return code

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','execute']);parser.add_argument('--mode',choices=['qualify','evaluate']);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.mode)
    else:raise SystemExit(execute(args.run_dir.resolve()))
