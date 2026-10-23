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


def qualification_records(records):
    import json
    result=[]
    for r in records:
        if r['event']=='reserved':
            m=json.loads((Path(r['artifact_path'])/'manifest.json').read_text())
            if m['config']['evaluation']['config_kind']=='laya-qualification-v1':result.append(r)
    return result


def validate_lineage(repo,manifest,records,*,allow_source_upgrade):
    import json
    from .protocol import committed
    from .store import digest
    lineage=manifest.get('selection_lineage',[])
    if len(lineage)!=1 or not isinstance(lineage[0],dict):raise ValueError('successor requires one committed diagnosis/source-upgrade lineage')
    item=lineage[0];parent=item.get('parent_run_id');kind=item.get('kind')
    allowed={'diagnosed_software_successor'}|({'source_upgrade_qualification'} if allow_source_upgrade else set())
    if kind not in allowed or parent not in manifest['parent_run_ids'] or not item.get('reason'):
        raise ValueError('invalid successor/source-upgrade parent or rationale')
    terminal='completed' if kind=='source_upgrade_qualification' else ('failed','cancelled')
    if not any(r['run_id']==parent and (r['event']==terminal if isinstance(terminal,str) else r['event'] in terminal) for r in records):
        raise ValueError('lineage parent is not the required immutable terminal state')
    path=(repo/item['report_path']).resolve()
    if not path.is_relative_to(repo) or digest(path)!=item['report_sha256']:raise ValueError('lineage report hash mismatch')
    committed(repo,path)
    if kind=='diagnosed_software_successor':
        category=item.get('failure_class')
        if not isinstance(category,str) or not category:raise ValueError('retry requires diagnosed failure class')
        count=0
        for record in records:
            if record['event']!='reserved':continue
            previous=json.loads((Path(record['artifact_path'])/'manifest.json').read_text())
            count+=sum(r.get('kind')=='diagnosed_software_successor' and r.get('failure_class','controller_environment_topology')==category for r in previous['selection_lineage'] if isinstance(r,dict))
        if count>=2:raise ValueError('two diagnosed successors per failure class exhausted')


def budget_rows(repo,records,*,category='software'):
    import json
    from experiment_artifacts import make_run_paths,load_validated_run_lifecycle,manifest_sha256
    from .protocol import CONTROL,STUDY,committed
    from .store import digest
    if category not in {'software','scientific'}:raise ValueError('unknown budget category')
    anchors={}
    for path in sorted((repo/CONTROL/'summaries').glob('*.json')):
        summary=json.loads(committed(repo,path))
        for row in summary.get('runs',[]):
            anchors[row['run_id']]=row['output_hashes'].get('provenance/controller-result.json')
    rows=[]
    for ref in (r for r in records if r['event']=='reserved'):
        root=Path(ref['artifact_path']);paths=make_run_paths(root.parents[3],STUDY,ref['run_id'])
        m,s,_=load_validated_run_lifecycle(paths)
        if manifest_sha256(m)!=ref['manifest_sha256']:raise ValueError('budget run manifest identity changed')
        software=m['config']['evaluation']['config_kind']=='laya-qualification-v1'
        if software!=(category=='software'):continue
        row=dict(state=s['state'],gpu_count=m['environment']['gpu_count'],max_seconds=m['config']['evaluation']['max_seconds'])
        if s['state'] in {'completed','failed','cancelled'}:
            receipt=root/'provenance/controller-result.json';sha=digest(receipt)
            if s['output_hashes'].get('provenance/controller-result.json')!=sha or anchors.get(ref['run_id'])!=sha:
                raise ValueError('budget receipt not bound to committed terminal summary')
            payload=json.loads(receipt.read_text())
            if payload['manifest_sha256']!=ref['manifest_sha256'] or payload['gpu_count']!=row['gpu_count']:raise ValueError('budget receipt identity mismatch')
            row['elapsed_seconds']=payload['elapsed_seconds']
        rows.append(row)
    return rows


def run_controller(repo,root):
    import json,os,subprocess,sys,time
    from types import SimpleNamespace
    from scripts import manage_experiment_run as manager
    from experiment_artifacts import manifest_sha256,parse_registry_text
    from semantic_retrieval.config import RuntimeConfig
    from semantic_retrieval.runtime.artifacts import publish_metrics
    from semantic_retrieval.runtime.distributed import configure_fp32
    from semantic_retrieval.full_data.controller import campaign_lock,idle_devices,_terminal,termination_signals
    from .protocol import CONTROL,committed,assert_laya_authorized
    from .store import atomic,canonical
    m=json.loads((root/'manifest.json').read_text());registry=repo/CONTROL/'registry.jsonl'
    identity=SimpleNamespace(repo_root=repo,registry=registry.relative_to(repo),run_dir=root,run_id=m['run_id'],manifest_sha256=manifest_sha256(m))
    runtime=RuntimeConfig(registry.with_name('study.yaml'),registry,root,m['run_id'],manifest_sha256(m),m['git_sha'],'val')
    configure_fp32()
    with campaign_lock(root.parents[3]/'laya-rerank-validation-v1/.controller.lock'),termination_signals():
        manager.verify_run(identity)
        records=parse_registry_text(committed(repo,registry).decode())
        if projected_hours(budget_rows(repo,records))>8: raise ValueError('software GPU-hour budget exceeded')
        if not {'1','3'}<=set(idle_devices()): raise ValueError('authorized GPU 1/3 are no longer idle')
        import shlex
        assert_laya_authorized(repo_root=repo,registry_path=registry,manifest=m,operation='start',command=shlex.split((root/'command.sh').read_text()))
        manager.start_run(identity)
        context=SimpleNamespace(paths=manager.make_run_paths_from_dir(root),config=runtime,sequence=0)
        started=time.monotonic();deadline=started+1770;exit_code=1;reason='not_started';process=None;nonce=None
        token=root/'provenance/controller-live.json'
        atomic(token,canonical({'pid':os.getpid(),'start_ticks':process_ticks(os.getpid()),'source_sha':m['git_sha'],'manifest_sha256':manifest_sha256(m),'command_sha256':m['command_sha256'],'controller_file':str(Path(__file__).resolve()),'controller_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}))
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
            manager.transition_run(SimpleNamespace(**vars(identity),expected='running',next_state='validating',exit_code=None,output=[]))
            exit_code=0;reason='passed'
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
    import fcntl,json,os,time
    from types import SimpleNamespace
    from scripts import manage_experiment_run as manager
    from experiment_artifacts import manifest_sha256
    from .protocol import CONTROL
    from .store import digest
    receipt=json.loads((root/'provenance/controller-live.json').read_text())
    controller=Path(receipt.get('controller_file',__file__)).resolve()
    if controller not in {Path(__file__).resolve(),Path(__file__).with_name('science_controller.py').resolve()}:
        raise ValueError('unknown controller source')
    expected={'source_sha':m['git_sha'],'manifest_sha256':manifest_sha256(m),
              'command_sha256':m['command_sha256'],'controller_sha256':digest(controller)}
    if ((root/'provenance/controller-result.json').exists() or any(receipt.get(k)!=v for k,v in expected.items())
            or process_ticks(receipt['pid'])!=receipt['start_ticks']):
        raise ValueError('controller launch identity changed')
    launch_path=root/f'provenance/launch-{mode}.json'
    deadline=time.monotonic()+10
    while not launch_path.exists():
        if time.monotonic()>=deadline or process_ticks(receipt['pid'])!=receipt['start_ticks']:
            raise ValueError('controller did not publish worker launch receipt')
        time.sleep(.01)
    launch=json.loads(launch_path.read_text())
    parent=int((Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[1])
    belongs=(parent==launch['pid'] if int(os.environ.get('WORLD_SIZE','1'))>1 else os.getpid()==launch['pid'] and parent==receipt['pid'])
    if (launch['mode']!=mode or not belongs or os.environ.get('LAYA_LAUNCH_NONCE')!=launch['nonce']
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
