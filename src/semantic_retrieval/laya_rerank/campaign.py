"""Budgeted, owned-process qualification controller; scientific authorization stays separate."""
import math
from pathlib import Path


def qualification_config(fixed):
    return dict(config_kind='laya-qualification-v1',world_size=2,seed=713,synthetic=True,
                random_initialization=True,precision='fp32',max_seconds=1800,eval_split='val',
                weights_architecture=fixed['weights'],external_code=fixed['external_code'],
                frames=4,top_k=10,query_count=4,video_count=12,microbatch=8,uatvr_clip=fixed['qualification_uatvr_clip'])


def projected_hours(rows):
    total=0.
    for row in rows:
        terminal=row['state'] in {'completed','failed','cancelled'}
        value=row.get('elapsed_seconds' if terminal else 'max_seconds')
        cards=row.get('gpu_count')
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or type(cards) is not int or cards<1:
            raise ValueError('unproven GPU budget receipt')
        total+=cards*value/3600
    return total


def budget_rows(repo,records):
    import json
    from .store import digest
    rows=[]
    for run_id in sorted({r['run_id'] for r in records if r['event']=='reserved'}):
        ref=next(r for r in records if r['event']=='reserved' and r['run_id']==run_id)
        root=Path(ref['artifact_path']);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'status.json').read_text())
        if m['config']['evaluation']['config_kind']!='laya-qualification-v1':continue
        row=dict(state=s['state'],gpu_count=m['environment']['gpu_count'],max_seconds=1800)
        if s['state'] in {'completed','failed','cancelled'}:
            receipt=root/'provenance/controller-result.json'
            if s['output_hashes'].get('provenance/controller-result.json')!=digest(receipt):
                raise ValueError('budget receipt not terminal-bound')
            row['elapsed_seconds']=json.loads(receipt.read_text())['elapsed_seconds']
        rows.append(row)
    return rows


def run_controller(repo,root):
    import json,os,subprocess,sys,time
    from types import SimpleNamespace
    from scripts import manage_experiment_run as manager
    from experiment_artifacts import manifest_sha256,parse_registry_text
    from semantic_retrieval.config import RuntimeConfig
    from semantic_retrieval.runtime.artifacts import prepare_run,start_run,complete_run,publish_metrics
    from semantic_retrieval.runtime.distributed import configure_fp32
    from semantic_retrieval.full_data.controller import campaign_lock,idle_devices,_terminal,termination_signals
    from .protocol import CONTROL,committed
    from .store import atomic,canonical
    from .worker import LayaSystemConfig
    m=json.loads((root/'manifest.json').read_text());registry=repo/CONTROL/'registry.jsonl'
    identity=SimpleNamespace(repo_root=repo,registry=registry.relative_to(repo),run_dir=root,run_id=m['run_id'],manifest_sha256=manifest_sha256(m))
    runtime=RuntimeConfig(registry.with_name('study.yaml'),registry,root,m['run_id'],manifest_sha256(m),m['git_sha'],'val')
    configure_fp32()
    with campaign_lock(root.parents[3]/'laya-rerank-validation-v1/.controller.lock'),termination_signals():
        manager.verify_run(identity)
        records=parse_registry_text(committed(repo,registry).decode())
        if projected_hours(budget_rows(repo,records))>8: raise ValueError('software GPU-hour budget exceeded')
        # A global cap of three is stricter than two successors per failure class.
        if len({r['run_id'] for r in records if r['event']=='reserved'})>3: raise ValueError('qualification successor cap exceeded')
        if not {'1','3'}<=set(idle_devices()): raise ValueError('authorized GPU 1/3 are no longer idle')
        context=prepare_run(runtime,repo_root=repo,system_config=LayaSystemConfig(runtime,m['config']))
        start_run(context)
        started=time.monotonic();deadline=started+1770;exit_code=1;reason='not_started';process=None;nonce=None
        token=root/'provenance/controller-live.json'
        atomic(token,canonical({'pid':os.getpid(),'start_ticks':process_ticks(os.getpid()),'source_sha':m['git_sha'],'manifest_sha256':manifest_sha256(m),'command_sha256':m['command_sha256'],'controller_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}))
        try:
            for mode in ('interrupted','resumed','continuous'):
                command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                         str(repo/'scripts/qualify_laya_rerank.py'),'--run-dir',str(root),'--worker',mode]
                with (root/'logs'/f'{mode}.log').open('xb') as log:
                    nonce=os.urandom(32).hex()
                    process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'LAYA_LAUNCH_NONCE':nonce})
                    atomic(root/f'provenance/launch-{mode}.json',canonical({'pid':process.pid,'start_ticks':process_ticks(process.pid),'mode':mode,'nonce':nonce}))
                    try:observed=process.wait(timeout=max(1,deadline-time.monotonic()))
                    finally:stop_owned(process,nonce)
                if mode=='interrupted':
                    if observed==0 or any(not (root/f'provenance/interrupted-rank{r}.json').exists() for r in (0,1)):
                        raise ValueError('controlled interruption did not reach both committed first blocks')
                elif observed!=0:raise ValueError('qualification worker failed in '+mode)
            report=json.loads((root/'provenance/laya-qualification.json').read_text())
            if not all(report['checks'].values()):raise ValueError('qualification checks failed')
            publish_metrics(context,dict(split='validation',summary_metrics={'qualification_passed':1.0},array_references=[]))
            complete_run(context);exit_code=0;reason='passed'
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
            if process is not None:stop_owned(process,nonce)
        atomic(root/'provenance/controller-result.json',canonical(dict(elapsed_seconds=time.monotonic()-started,
               gpu_count=2,source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),terminal_exit_code=exit_code,reason=reason)))
        _terminal(manager,identity,exit_code)
        return exit_code


def process_ticks(pid):
    info=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    if info[0]=='Z':raise ValueError('controller process is a zombie')
    return info[19]


def validate_worker_launch(repo,root,m,mode):
    import fcntl,json,os
    from types import SimpleNamespace
    from scripts import manage_experiment_run as manager
    from experiment_artifacts import manifest_sha256
    from .protocol import CONTROL
    from .store import digest
    receipt=json.loads((root/'provenance/controller-live.json').read_text())
    expected={'source_sha':m['git_sha'],'manifest_sha256':manifest_sha256(m),
              'command_sha256':m['command_sha256'],'controller_sha256':digest(Path(__file__))}
    if ((root/'provenance/controller-result.json').exists() or any(receipt.get(k)!=v for k,v in expected.items())
            or process_ticks(receipt['pid'])!=receipt['start_ticks']):
        raise ValueError('controller launch identity changed')
    launch=json.loads((root/f'provenance/launch-{mode}.json').read_text())
    parent=int((Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[1])
    if (launch['mode']!=mode or parent!=launch['pid'] or os.environ.get('LAYA_LAUNCH_NONCE')!=launch['nonce']
            or process_ticks(launch['pid'])!=launch['start_ticks']):raise ValueError('worker is outside the owned launcher ancestry')
    fd=os.open(root.parents[3]/'laya-rerank-validation-v1/.controller.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise ValueError('qualification requires locked budget controller')
    finally:os.close(fd)
    manager.verify_run(SimpleNamespace(repo_root=repo,registry=Path(CONTROL)/'registry.jsonl',run_dir=root,
                                      run_id=m['run_id'],manifest_sha256=manifest_sha256(m)))


def owned_processes(nonce):
    """Find this launch's processes, including orphaned Elastic child sessions."""
    import os
    result=[];token=('LAYA_LAUNCH_NONCE='+nonce).encode()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid() or token not in (entry/'environ').read_bytes().split(b'\0'):continue
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z':result.append((int(entry.name),fields[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result


def stop_owned(process,nonce):
    """Signal only nonce-bound identities we created; prove all writers stopped."""
    import os,signal,time
    for sig in (signal.SIGTERM,signal.SIGKILL):
        until=time.monotonic()+10
        while True:
            live=owned_processes(nonce)
            if not live:break
            for pid,ticks in live:
                try:
                    if process_ticks(pid)==ticks:os.kill(pid,sig)
                except (FileNotFoundError,ProcessLookupError,ValueError):pass
            if time.monotonic()>=until:break
            process.poll();time.sleep(.1)
    process.wait(timeout=5)
    if owned_processes(nonce):raise RuntimeError('cannot prove all owned child sessions stopped; manual recovery required')
