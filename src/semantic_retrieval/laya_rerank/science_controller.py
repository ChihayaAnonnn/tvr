"""Bounded scientific worker ownership and serial idle-GPU performance rounds."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from experiment_artifacts import manifest_sha256,parse_registry_text
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.full_data.controller import campaign_lock,idle_devices,_terminal,termination_signals
from semantic_retrieval.runtime.artifacts import publish_metrics
from .campaign import projected_hours,budget_rows,process_ticks,stop_owned,owned_processes
from .protocol import CONTROL,committed,assert_laya_authorized
from .store import atomic,canonical,digest


def wait_idle(devices,deadline):
    while not set(devices)<=set(idle_devices()):
        if time.monotonic()>=deadline:raise ValueError('stable idle GPU unavailable; performance comparison incomplete')
        print('waiting_for_idle_gpu='+','.join(devices),flush=True);time.sleep(15)


def ownership_start_ticks(pid):
    # Unlike live-worker validation, GPU cleanup monitoring accepts zombies.
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]


def measurement_owners(pid,nonce,start_ticks):
    """nvidia-smi can retain our just-exited process after /proc disappears."""
    owners={p for p,_ in owned_processes(nonce)}
    try:current=ownership_start_ticks(pid)
    except (FileNotFoundError,ProcessLookupError):current=None
    if current is None or current==start_ticks:owners.add(pid)
    return owners


def run_controller(repo,root):
    from .measurement import summarize_performance
    m=json.loads((root/'manifest.json').read_text());ev=m['config']['evaluation'];kind=ev['operation']
    registry=repo/CONTROL/'registry.jsonl'
    identity=SimpleNamespace(repo_root=repo,registry=registry.relative_to(repo),run_dir=root,run_id=m['run_id'],manifest_sha256=manifest_sha256(m))
    runtime=RuntimeConfig(registry.with_name('study.yaml'),registry,root,m['run_id'],manifest_sha256(m),m['git_sha'],'val')
    from scripts import manage_experiment_run as manager
    with campaign_lock(root.parents[3]/'laya-rerank-validation-v1/.controller.lock'),termination_signals():
        manager.verify_run(identity)
        assert_laya_authorized(repo_root=repo,registry_path=registry,manifest=m,operation='start',command=shlex.split((root/'command.sh').read_text()))
        records=parse_registry_text(committed(repo,registry).decode())
        if projected_hours(budget_rows(repo,records,category='scientific'))>24:raise ValueError('scientific GPU budget exceeded')
        devices=['1','3'] if kind=='accuracy' else ['3']
        wait_idle(devices,time.monotonic()+1800)
        manager.verify_run(identity);manager.start_run(identity)
        context=SimpleNamespace(paths=manager.make_run_paths_from_dir(root),config=runtime,sequence=0)
        started=time.monotonic();deadline=started+ev['max_seconds']-30;process=None;nonce=None;exit_code=1;reason='not_started'
        atomic(root/'provenance/controller-live.json',canonical(dict(pid=os.getpid(),start_ticks=process_ticks(os.getpid()),
            source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),command_sha256=m['command_sha256'],
            controller_file=str(Path(__file__).resolve()),controller_sha256=digest(Path(__file__)))))
        modes=['accuracy'] if kind=='accuracy' else ['prepare','baseline-r0','enhanced-r0','enhanced-r1','baseline-r1','baseline-r2','enhanced-r2']
        try:
            for mode in modes:
                if kind=='performance':wait_idle(devices,min(deadline,time.monotonic()+120))
                worker=[str(repo/'scripts/evaluate_laya_rerank.py'),'--run-dir',str(root),'--worker-mode',mode]
                command=[sys.executable,*(['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2'] if kind=='accuracy' else []),*worker]
                environment={**os.environ,'LAYA_LAUNCH_NONCE':os.urandom(32).hex()}
                if kind=='performance':environment.update(WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
                nonce=environment['LAYA_LAUNCH_NONCE']
                with (root/'logs'/f'{mode}.log').open('xb') as log:
                    process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
                    launch_ticks=process_ticks(process.pid)
                    atomic(root/f'provenance/launch-{mode}.json',canonical(dict(pid=process.pid,start_ticks=launch_ticks,mode=mode,nonce=nonce)))
                    try:
                        while process.poll() is None:
                            if time.monotonic()>=deadline:raise TimeoutError('scientific run deadline exceeded')
                            if kind=='performance':
                                owned=measurement_owners(process.pid,nonce,launch_ticks)
                                active=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
                                for line in active.splitlines():
                                    uuid,pid=[x.strip() for x in line.split(',')]
                                    if uuid=='GPU-8dab94cb-3e36-d6b4-61a8-a617b5dc7ede' and int(pid) not in owned:
                                        raise ValueError('unrelated GPU computation PID='+pid+' during performance round; comparison incomplete')
                            time.sleep(1)
                        observed=process.returncode
                    finally:stop_owned(process,nonce)
                if observed!=0:raise ValueError('scientific worker failed in '+mode+' exit='+str(observed))
            if kind=='accuracy':
                report=json.loads((root/'provenance/laya-validation.json').read_text())
                if report['query_count']!=10000 or report['video_count']!=500:raise ValueError('incomplete scientific coverage')
                summary={'delta_r1_pp':report['delta_r1_pp'],**{'baseline_'+k:v for k,v in report['metrics']['baseline'].items()},
                         **{'reranked_'+k:v for k,v in report['metrics']['reranked'].items()}}
                references=list(report['arrays'].values())
            else:
                report=summarize_performance(root)
                summary={'baseline_median_ms':report['baseline']['median_ms'],'enhanced_median_ms':report['enhanced']['median_ms'],
                         'relative_median_latency':report['relative_median_latency']}
                references=[]
            publish_metrics(context,dict(split='validation',summary_metrics=summary,array_references=references))
            manager.transition_run(SimpleNamespace(**vars(identity),expected='running',next_state='validating',exit_code=None,output=[]))
            exit_code=0;reason='completed'
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
            if process is not None:stop_owned(process,nonce)
        atomic(root/'provenance/controller-result.json',canonical(dict(elapsed_seconds=time.monotonic()-started,
               gpu_count=m['environment']['gpu_count'],source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),
               terminal_exit_code=exit_code,reason=reason,operation=kind)))
        _terminal(manager,identity,exit_code)
        return exit_code
