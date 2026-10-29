"""Reserve, launch and archive one scoped local-query qualification/training run."""
import argparse
from dataclasses import asdict
import datetime
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
from types import SimpleNamespace

import torch
import yaml
from experiment_artifacts import load_registry, strict_json_load, manifest_sha256
from scripts import manage_experiment_run as manager
from scripts.prepare_aqs_training import relocate_indices
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.campaign import FORMAL, RANDOM, PRETRAINED, scientific_inputs

STUDY='uatvr-aqs-local-query-training-v1'
CONTROL=Path('experiments')/STUDY
ARTIFACTS=Path('/data2/hxj/project/UATVR/artifacts')


def read(path):
    with Path(path).open() as stream:return strict_json_load(stream)


def commit(repo,files,message):
    subprocess.run(['git','add',*map(str,files)],cwd=repo,check=True)
    subprocess.run(['git','commit','-m',message],cwd=repo,check=True)


def sources(repo):
    names=subprocess.check_output(['git','ls-files','src','scripts','experiment_artifacts.py'],cwd=repo,text=True).splitlines()
    return {name:file_sha256(repo/name) for name in names if name.endswith('.py')}


def devices(study):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    memory={r.split(',')[0].strip():int(r.split(',')[1]) for r in rows}
    chosen=study['resource_policy']['gpu_uuids']
    if len(chosen)!=2 or len(set(chosen))!=2 or any(memory.get(u,999999)>500 for u in chosen):
        raise ValueError('two registered GPUs must each have less than 500 MiB used')
    return chosen


def prepare(repo,mode):
    manager._require_clean(repo)
    study=yaml.safe_load((repo/CONTROL/'study.yaml').read_text());records=load_registry(repo/CONTROL/'registry.jsonl')
    terminals={r['run_id'] for r in records if r['event'] in {'completed','failed','cancelled'}}
    if any(r['event']=='reserved' and r['run_id'] not in terminals for r in records):raise ValueError('unresolved reservation; verify existing run first')
    if any((r['event']=='reserved' and ('local-query-q0' if mode=='qualify' else 'local-query-train') in r['run_id']) for r in records):
        raise ValueError('this study allows one attempt per stage; automatic successor not authorized')
    chosen=devices(study);recipe=study['recipe'];local=LocalQueryConfig(study['local']['enabled'],study['local']['beta'],study['local']['tau'])
    evaluation=dict(mode=mode,dsl=False,training=recipe,source_files=sources(repo),source_report=None,qualification=None)
    parents=[];inputs=[]
    env_spec=repo/'experiments/uatvr-aqs-training-v1/env-spec.json'
    evaluation['environment_spec']=dict(path=str(env_spec),sha256=file_sha256(env_spec))
    if mode=='train':
        rows=[r for r in records if r['event']=='completed' and 'local-query-q0' in r['run_id']]
        if len(rows)!=1:raise ValueError('exactly one completed qualification must be archived')
        row=rows[0];qroot=Path(row['artifact_path']);qreport=qroot/'provenance/result.json'
        evaluation['qualification']=dict(path=str(qreport),sha256=file_sha256(qreport),run_id=row['run_id'])
        from semantic_retrieval.aqs.local_launch import require_qualification,qualification_evidence
        qualification_evidence(read(qreport),file_sha256(qreport),read(qroot/'status.json')['output_hashes'].get('provenance/result.json'),manifest_sha256(read(qroot/'manifest.json')))
        if row['manifest_sha256']!=manifest_sha256(read(qroot/'manifest.json')):raise ValueError('qualification manifest differs from archived registry')
        require_qualification(read(qreport),evaluation['source_files'],recipe,local.recipe(),read(qroot/'status.json')['state'])
        parents=[row['run_id']]
        inputs,certificate=scientific_inputs(repo);inputs,_=relocate_indices(repo,inputs)
        evaluation['source_report']=dict(path=str(certificate),sha256=file_sha256(certificate))
        inputs.append(dict(path=str(qreport),sha256=file_sha256(qreport)))
    inputs.append(evaluation['environment_spec'])
    weights=RANDOM if mode=='qualify' else PRETRAINED
    weight_sha=file_sha256(weights)
    if weight_sha!=('4186a61a25330fbc2b567b948f55c8e5660484d393e70817fc3e95b7de72faa8' if mode=='qualify' else '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'):
        raise ValueError('unexpected CLIP input bytes')
    command=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+','.join(chosen),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src:.',
        sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '--log-dir','{artifact_run_dir}/logs/elastic','--module','semantic_retrieval.aqs.local_launch',
        '--run-dir','{artifact_run_dir}','--mode',mode]
    spec=dict(dataset='synthetic' if mode=='qualify' else 'msrvtt',stage='baseline',variant='local-query-q0' if mode=='qualify' else 'local-query-train',seed=recipe['seed'],
        scheduled_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        environment=dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=str(torch.version.cuda),gpu_model='NVIDIA A800 80GB PCIe',gpu_count=2,precision='fp32',distributed_topology='torchrun-2',gpu_uuids=chosen),
        input_checkpoints=[dict(path=str(weights),sha256=weight_sha)],dataset_splits=inputs,
        config=dict(model=dict(architecture='uatvr-aqs-local-query-v1',clip=asdict(FORMAL),local=local.recipe(),query_chunk=study['local']['query_chunk'],video_chunk=study['local']['video_chunk']),
            loss=dict(aqs=AqsConfig(True,recipe['negatives'],recipe['weight']).recipe()),optimizer=dict(kind='pinned_bert_adam'),scheduler=dict(kind='warmup_cosine'),evaluation=evaluation),
        parent_run_ids=parents,selection_lineage=[dict(report_path=str(CONTROL/'study.yaml'),report_sha256=file_sha256(repo/CONTROL/'study.yaml'),reason='User-authorized independent dual-GPU local-query run; qualification must precede scientific training.')],
        validation_authorized=True,test_authorized=False,expected_scalar_outputs=[],array_references=[],command=command)
    stamp=spec['scheduled_timestamp_utc'];phase=CONTROL/'phases'/f'{mode}-{stamp}.json'
    (repo/phase.parent).mkdir(exist_ok=True)
    publish_json(repo/phase,dict(phase=f'{mode}-{stamp}',study=STUDY,runs=[spec]))
    commit(repo,[phase],f'research: freeze local query {mode} phase')
    reservation=manager.reserve_phase(SimpleNamespace(repo_root=str(repo),phase=str(phase),registry=str(CONTROL/'registry.jsonl'),artifact_root=str(ARTIFACTS)))
    commit(repo,[CONTROL/'registry.jsonl'],f'archive: reserve local query {mode}')
    row=reservation['runs'][0]
    root=Path(row.get('artifact_path',row.get('run_dir','')))
    if not root.is_absolute():
        # Manager rows use run_dir; never infer a new or substitute run identity.
        raise ValueError('reservation did not return absolute run directory')
    publish_json(root/'provenance/campaign.json',dict(phase=str(phase),mode=mode))
    return root


def execute(repo,root):
    manifest=read(root/'manifest.json');mode=manifest['config']['evaluation']['mode']
    args=SimpleNamespace(repo_root=str(repo),registry=manifest['registry_path'],run_dir=str(root),run_id=manifest['run_id'],manifest_sha256=manifest_sha256(manifest))
    verified=manager.verify_run(args)
    study=yaml.safe_load((repo/CONTROL/'study.yaml').read_text());devices(study)
    manager.start_run(args)
    deadline=3600 if mode=='qualify' else 36*3600
    logfile=root/'logs'/('qualification.log' if mode=='qualify' else 'train.log')
    returncode=1
    with logfile.open('x') as log:
        log.write(f'LOCAL QUERY {mode} run_id={manifest["run_id"]} source={manifest["git_sha"]}\n');log.flush()
        child=subprocess.Popen(verified['command_argv'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        publish_json(root/'provenance/process.json',dict(controller_pid=os.getpid(),torchrun_pid=child.pid,log_path=str(logfile)))
        try:returncode=child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            log.write('Registered wall-time budget exhausted\n')
    # All worker writers are closed before terminal inventories and control commits.
    manager.verify_run(args)
    if returncode==0:
        result=read(root/'provenance/result.json')
        if (result.get('passed') is not True if mode=='qualify' else result.get('completed') is not True):
            returncode=1
    if returncode==0:
        manager.transition_run(SimpleNamespace(**vars(args),expected='running',next_state='validating',exit_code=None,output=[]))
        manager.transition_run(SimpleNamespace(**vars(args),expected='validating',next_state='completed',exit_code=0,output=[]))
    else:manager.transition_run(SimpleNamespace(**vars(args),expected='running',next_state='failed',exit_code=returncode,output=[]))
    campaign=read(root/'provenance/campaign.json');phase=campaign['phase']
    summary=CONTROL/'summaries'/(Path(phase).stem+'.json');(repo/summary.parent).mkdir(exist_ok=True)
    manager.finalize_phase(SimpleNamespace(repo_root=str(repo),phase=phase,registry=manifest['registry_path'],artifact_root=str(ARTIFACTS),summary=str(summary)))
    commit(repo,[CONTROL/'registry.jsonl',summary],f'archive: finalize local query {mode}')
    return returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','execute']);parser.add_argument('--mode',choices=['qualify','train']);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if args.operation=='prepare':print(prepare(repo,args.mode))
    else:raise SystemExit(execute(repo,args.run_dir.resolve()))


if __name__=='__main__':main()
