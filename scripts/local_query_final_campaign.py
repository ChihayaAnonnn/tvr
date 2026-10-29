"""Scoped one-checkpoint JSFUSION final test controller; no training."""
import argparse
import datetime
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
from scripts.local_query_campaign import CONTROL,ARTIFACTS,STUDY,read,commit,sources,devices
from semantic_retrieval.aqs.local_final import authenticate_selection
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.final import CSV
from semantic_retrieval.full_data.loader import read_complete
from semantic_retrieval.runtime.input_bindings import manifest_input_references


def freeze(repo):
    rows=[r for r in load_registry(repo/CONTROL/'registry.jsonl') if r['event']=='completed' and 'local-query-train' in r['run_id']]
    if len(rows)!=1:raise ValueError('exactly one completed validation-selected parent required')
    row=rows[0];root=Path(row['artifact_path']);parent=read(root/'manifest.json');result=read(root/'provenance/result.json')
    epoch=result['selected_epoch'];ckpt=root/f'checkpoints/epoch-{epoch:02d}.pt';meta=read(ckpt.with_suffix('.json'))
    frozen=dict(parent_run_id=row['run_id'],parent_manifest_sha256=row['manifest_sha256'],parent_result_sha256=file_sha256(root/'provenance/result.json'),selected_epoch=epoch,
        checkpoint=dict(path=str(ckpt),sha256=file_sha256(ckpt),metadata_sha256=file_sha256(ckpt.with_suffix('.json')),local_sha256=file_sha256(ckpt.with_suffix('.local.json'))),model_sha256=meta['model_sha256'])
    authenticate_selection(repo,frozen)
    refs=manifest_input_references(repo,parent);cert=parent['config']['evaluation']['source_report']
    videos,_=read_complete(Path(cert['path']),cert['sha256'],refs)
    forbidden=repo/CONTROL/'inputs/train-val-ids.json';forbidden.parent.mkdir(exist_ok=True)
    publish_json(forbidden,dict(parent_manifest_sha256=row['manifest_sha256'],ids=sorted(videos)))
    indices=[repo/'experiments/full-data-retrieval-v1/inputs'/f'jsfusion-1k-final-{i:03d}.input-index.json' for i in range(2)]
    permission=dict(authorized=True,authorization='2026-10-30_explicit_user_evaluate_this_ckpt_on_1k_test',frozen=frozen,
        test_csv=dict(path=str(CSV),sha256=file_sha256(CSV)),video_indices=[dict(path=str(p),sha256=file_sha256(p)) for p in indices],
        forbidden_ids=dict(path=str(forbidden),sha256=file_sha256(forbidden)),max_evaluations=1,dsl=False,beta=.2,tau=.1,
        inference_qualification='fresh_two_rank_production_random_then_frozen_checkpoint_no_test_data',max_gpu_hours=2)
    study=yaml.safe_load((repo/CONTROL/'study.yaml').read_text())
    if 'final_test' in study:raise ValueError('existing test permission cannot be overwritten')
    study['final_test']=permission
    (repo/CONTROL/'study.yaml').write_text(yaml.safe_dump(study,sort_keys=False))
    return permission


def prepare(repo,mode):
    manager._require_clean(repo);study=yaml.safe_load((repo/CONTROL/'study.yaml').read_text());permission=study['final_test']
    parent=authenticate_selection(repo,permission['frozen']);rows=load_registry(repo/CONTROL/'registry.jsonl')
    terminal={r['run_id'] for r in rows if r['event'] in ('completed','failed','cancelled')}
    if any(r['event']=='reserved' and r['run_id'] not in terminal for r in rows):raise ValueError('unresolved run needs verification')
    variant='local-query-final-q0' if mode=='qualify' else 'local-query-jsfusion1k'
    if any(r['event']=='reserved' and variant in r['run_id'] for r in rows):raise ValueError('one attempt only; successful or failed test cannot silently repeat')
    uuids=devices(study);parents=[permission['frozen']['parent_run_id']]
    ev=dict(mode=mode,dsl=False,frozen=permission['frozen'],source_files=sources(repo),forbidden_ids=permission['forbidden_ids'],qualification=None)
    inputs=[]
    if mode=='test':
        qualified=[r for r in rows if r['event']=='completed' and 'local-query-final-q0' in r['run_id']]
        if len(qualified)!=1:raise ValueError('fresh archived final-inference qualification required')
        q=qualified[0];qroot=Path(q['artifact_path']);report=read(qroot/'provenance/result.json');status=read(qroot/'status.json')
        sha=file_sha256(qroot/'provenance/result.json')
        if (status['state']!='completed' or status['output_hashes'].get('provenance/result.json')!=sha
                or report.get('passed') is not True or report['source_files']!=ev['source_files'] or report['frozen']!=ev['frozen']
                or report['manifest_sha256']!=q['manifest_sha256'] or report['test_accessed'] is not False):
            raise ValueError('qualification evidence/configuration mismatch')
        ev['qualification']=dict(path=str(qroot/'provenance/result.json'),sha256=sha,run_id=q['run_id']);parents.append(q['run_id'])
        inputs=[permission['test_csv'],*permission['video_indices'],permission['forbidden_ids']]
    ref=permission['frozen']['checkpoint']
    checkpoints=[parent['input_checkpoints'][0],dict(path=ref['path'],sha256=ref['sha256']),dict(path=str(Path(ref['path']).with_suffix('.json')),sha256=ref['metadata_sha256']),dict(path=str(Path(ref['path']).with_suffix('.local.json')),sha256=ref['local_sha256'])]
    command=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+','.join(uuids),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src:.',
        sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--log-dir','{artifact_run_dir}/logs/elastic','--module','semantic_retrieval.aqs.local_final','--run-dir','{artifact_run_dir}','--mode',mode]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec=dict(dataset='synthetic' if mode=='qualify' else 'msrvtt',stage='baseline' if mode=='qualify' else 'final',variant=variant,seed=713,scheduled_timestamp_utc=stamp,
        environment=dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=str(torch.version.cuda),gpu_model='NVIDIA A800 80GB PCIe',gpu_count=2,precision='fp32',distributed_topology='torchrun-2',gpu_uuids=uuids),
        config=dict(model=parent['config']['model'],loss={},optimizer={},scheduler={},evaluation=ev),input_checkpoints=checkpoints,dataset_splits=inputs,parent_run_ids=parents,
        selection_lineage=[dict(report_path=str(CONTROL/'FINAL_TEST_PROTOCOL.md'),report_sha256=file_sha256(repo/CONTROL/'FINAL_TEST_PROTOCOL.md'),reason='Explicit user scope: validation-selected epoch3 only; no test selection or DSL tuning.')],
        validation_authorized=mode=='qualify',test_authorized=mode=='test',expected_scalar_outputs=[],array_references=[],command=command)
    phase=CONTROL/'phases'/f'final-{mode}-{stamp}.json';publish_json(repo/phase,dict(study=STUDY,phase=f'final-{mode}-{stamp}',runs=[spec]))
    commit(repo,[phase],f'research: freeze local query final {mode}')
    reserved=manager.reserve_phase(SimpleNamespace(repo_root=str(repo),phase=str(phase),registry=str(CONTROL/'registry.jsonl'),artifact_root=str(ARTIFACTS)))
    commit(repo,[CONTROL/'registry.jsonl'],f'archive: reserve local query final {mode}')
    root=Path(reserved['runs'][0]['artifact_path']);publish_json(root/'provenance/campaign.json',dict(phase=str(phase),mode=mode))
    return root


def execute(repo,root):
    m=read(root/'manifest.json');mode=m['config']['evaluation']['mode']
    identity=dict(repo_root=str(repo),registry=m['registry_path'],run_dir=str(root),run_id=m['run_id'],manifest_sha256=manifest_sha256(m))
    args=SimpleNamespace(**identity);verified=manager.verify_run(args)
    devices(yaml.safe_load((repo/CONTROL/'study.yaml').read_text()));manager.start_run(args)
    code=1
    with (root/'logs'/('qualification.log' if mode=='qualify' else 'test.log')).open('x') as log:
        child=subprocess.Popen(verified['command_argv'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        publish_json(root/'provenance/process.json',dict(controller_pid=os.getpid(),torchrun_pid=child.pid))
        try:code=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    manager.verify_run(args)
    if code==0 and read(root/'provenance/result.json').get('passed' if mode=='qualify' else 'completed') is not True:code=1
    if code==0:
        manager.transition_run(SimpleNamespace(**identity,expected='running',next_state='validating',exit_code=None,output=[]))
        manager.transition_run(SimpleNamespace(**identity,expected='validating',next_state='completed',exit_code=0,output=[]))
    else:manager.transition_run(SimpleNamespace(**identity,expected='running',next_state='failed',exit_code=code,output=[]))
    phase=read(root/'provenance/campaign.json')['phase'];summary=CONTROL/'summaries'/(Path(phase).stem+'.json')
    manager.finalize_phase(SimpleNamespace(repo_root=str(repo),phase=phase,registry=m['registry_path'],artifact_root=str(ARTIFACTS),summary=str(summary)))
    commit(repo,[CONTROL/'registry.jsonl',summary],f'archive: finalize local query final {mode}')
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','prepare','execute']);parser.add_argument('--mode',choices=['qualify','test']);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if args.operation=='freeze':print(freeze(repo)['frozen']['checkpoint']['path'])
    elif args.operation=='prepare':print(prepare(repo,args.mode))
    else:raise SystemExit(execute(repo,args.run_dir.resolve()))
