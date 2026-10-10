"""One locked, budgeted Scheme A phase; stop only process groups we create."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import yaml
from experiment_artifacts import load_registry, strict_json_load
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from .authorization import CONTROL, STUDY, _committed

TERMINAL = {'completed', 'failed', 'cancelled'}


def budget_projection(runs):
    totals = dict(software=0.0, scientific=0.0)
    for row in runs:
        category, cards = row['category'], row['gpu_count']
        if category not in totals or type(cards) is not int or not 0 <= cards <= 64:
            raise ValueError('invalid GPU budget category/count')
        state = row['state']
        if state not in TERMINAL | {'reserved', 'running', 'validating'}:
            raise ValueError('invalid GPU budget lifecycle state')
        seconds = row['elapsed_seconds'] if state in TERMINAL else row['max_wall_seconds']
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('GPU accounting requires proven finite elapsed/worst-case seconds')
        totals[category] += cards*seconds/3600
    return totals


def check_successors(failed_attempts, max_successors):
    if (type(failed_attempts) is not int or failed_attempts < 0
            or type(max_successors) is not int or max_successors < 0 or failed_attempts > max_successors):
        raise ValueError('diagnosed software successor cap exceeded or unproven')


@contextmanager
def termination_signals():
    """Normal scheduler/session termination must enter owned-worker cleanup."""
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f'controller received termination signal {signum}')
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for signum in previous: signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)


@contextmanager
def campaign_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('another campaign controller is active') from exc
        yield
    finally:
        os.close(fd)


def idle_devices():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    # Memory and utilization alone can briefly look idle between external batches.
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                          check=True, capture_output=True, text=True).stdout.splitlines()
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    identities = {r.split(',')[0].strip(): r.split(',')[1].strip() for r in uuids}
    free = []
    for line in result.stdout.splitlines():
        index, memory, utilization = (x.strip() for x in line.split(','))
        if int(memory) <= 256 and int(utilization) == 0 and identities[index] not in busy:
            free.append(index)
    return free


def select_shared_devices(rows, *, limit_gib, headroom_gib):
    if any(type(x) is not int or x < 1 for x in (limit_gib, headroom_gib)):
        raise ValueError('invalid shared-card memory policy')
    seen, eligible = set(), []
    for row in rows:
        index, total, used = row['index'], row['total_mib'], row['used_mib']
        if (not isinstance(index, str) or not index.isdigit() or index in seen
                or type(total) is not int or type(used) is not int or not 0 <= used <= total):
            raise ValueError('invalid GPU memory observation')
        seen.add(index)
        if total-used >= (limit_gib+headroom_gib)*1024: eligible.append((used, int(index), index))
    return [item[2] for item in sorted(eligible)]


def available_devices(study):
    policy = study.get('resource_policy', {'sharing_authorized': False})
    if policy.get('sharing_authorized') is not True: return idle_devices()
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, total, used = (s.strip() for s in line.split(','))
        rows.append(dict(index=index, total_mib=int(total), used_mib=int(used)))
    return select_shared_devices(rows, limit_gib=policy['memory_limit_gib'], headroom_gib=policy['headroom_gib'])


def _read(path):
    with Path(path).open() as stream: return strict_json_load(stream)


def validate_controller_launch(repo, root, manifest):
    """GPU workers cannot bypass active-phase budgeting by invoking their CLI directly."""
    from scripts import manage_experiment_run as manager
    plan = _read(root/'provenance/controller-plan.json')
    intent = _read(root/'provenance/controller-launch-intent.json')
    expected = dict(manifest_sha256=manager.manifest_sha256(manifest), source_sha=manifest['git_sha'],
                    command_sha256=manifest['command_sha256'], controller_sha256=file_sha256(Path(__file__)),
                    gpu_count=manifest['environment']['gpu_count'])
    if (any(plan.get(k) != v or intent.get(k) != v for k, v in expected.items())
            or set(intent) != set(plan) | {'controller_pid', 'controller_start_ticks'}
            or (root/'provenance/controller-result.json').exists()):
        raise ValueError('GPU controller launch identity mismatch')
    pid = intent['controller_pid']
    if type(pid) is not int or pid < 1:
        raise ValueError('invalid controller process identity')
    stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    if stat[19] != intent['controller_start_ticks'] or stat[0] == 'Z':
        raise ValueError('controller process no longer owns the launch')
    lock = repo/CONTROL/'registry.jsonl.campaign.lock'
    fd = os.open(lock, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise ValueError('GPU worker requires an active locked controller')
    finally: os.close(fd)
    study = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
    category = 'scientific' if manifest['config']['evaluation']['full_data_stage'] == 'baseline' else 'software'
    if plan['max_wall_seconds'] != study['campaign'][category]['max_run_wall_seconds']:
        raise ValueError('controller wall budget differs from frozen study')
    totals = budget_projection(_budget_rows(repo, study, manager))
    if any(totals[k] > study['campaign'][k]['max_cumulative_gpu_hours'] for k in totals):
        raise ValueError('GPU worker launch exceeds whole-phase budget')


def _budget_rows(repo, study, manager):
    rows = []
    for record in load_registry(repo/CONTROL/'registry.jsonl'):
        if record['event'] != 'reserved': continue
        root = Path(record['artifact_path']); manifest, status = manager._load_run(root)
        if manifest['study'] != STUDY or record['manifest_sha256'] != manager.manifest_sha256(manifest):
            raise ValueError('budget reservation identity mismatch')
        cards = manifest['environment']['gpu_count']
        if cards == 0: continue  # P0 is authenticated by its own terminal protocol, never GPU work.
        stage = manifest['config']['evaluation']['full_data_stage']
        category = 'scientific' if stage == 'baseline' else 'software'
        seconds = None
        if status['state'] in TERMINAL:
            name = 'provenance/controller-result.json'
            if status['output_hashes'].get(name) != file_sha256(root/name):
                raise ValueError('terminal GPU accounting receipt is not authenticated')
            result = _read(root/name)
            if (result['manifest_sha256'] != record['manifest_sha256'] or result['source_sha'] != manifest['git_sha']
                    or result['gpu_count'] != cards):
                raise ValueError('terminal GPU accounting identity mismatch')
            seconds = result['elapsed_seconds']
        rows.append(dict(category=category, gpu_count=cards, state=status['state'], elapsed_seconds=seconds,
                         max_wall_seconds=study['campaign'][category]['max_run_wall_seconds']))
    return rows


def _audit(root, event, **fields):
    path = root/'provenance/controller-audit.jsonl'
    previous = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    packet = dict(event=event, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), previous_sha256=previous, **fields)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as stream:
        stream.write(json.dumps(packet, sort_keys=True, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def _group_alive(pid):
    # Zombie processes cannot write; avoid confusing an unreaped child with an active writer.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            stat = (entry/'stat').read_text(); fields = stat[stat.rfind(')')+2:].split()
        except (FileNotFoundError, ProcessLookupError): continue
        if int(fields[2]) == pid and fields[0] != 'Z': return True
    return False


def _stop_owned_group(process):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if not _group_alive(process.pid): break
        os.killpg(process.pid, signum)
        deadline = time.monotonic()+10
        while _group_alive(process.pid) and time.monotonic() < deadline:
            process.poll(); time.sleep(.1)
    process.wait(timeout=5)
    if _group_alive(process.pid):
        raise RuntimeError('cannot prove owned worker writers stopped; manual recovery required')


def _identity(repo, registry, reservation):
    return SimpleNamespace(repo_root=repo, registry=registry.relative_to(repo), run_dir=reservation['artifact_path'],
                           run_id=reservation['run_id'], manifest_sha256=reservation['manifest_sha256'])


def _terminal(manager, identity, exit_code, cancelled=False):
    root = Path(identity.run_dir); _, status = manager._load_run(root)
    if status['state'] in TERMINAL: raise ValueError('controller refuses to rewrite terminal history')
    if status['state'] == 'running':
        manager.transition_run(SimpleNamespace(**vars(identity), expected='running', next_state='validating', exit_code=None, output=[]))
        status['state'] = 'validating'
    final = 'cancelled' if cancelled or status['state'] == 'reserved' else ('completed' if exit_code == 0 else 'failed')
    if final == 'completed' and status['state'] != 'validating': raise ValueError('worker did not complete validation')
    manager.transition_run(SimpleNamespace(**vars(identity), expected=status['state'], next_state=final,
                                            exit_code=exit_code, output=[]))
    return final


def _run_one(repo, identity, study, manager):
    verified = manager.verify_run(identity)
    if verified['state'] != 'reserved': raise ValueError('run is not a fresh verified reservation')
    root = Path(identity.run_dir); manifest, _ = manager._load_run(root)
    cards = manifest['environment']['gpu_count']
    stage = manifest['config']['evaluation']['full_data_stage']
    attempts = 0
    for record in load_registry(repo/CONTROL/'registry.jsonl'):
        if record['event'] != 'failed': continue
        prior, status = manager._load_run(Path(record['artifact_path']))
        if (prior['config']['evaluation'].get('full_data_stage') == stage and prior['seed'] == manifest['seed']
                and prior['config']['model'].get('method') == manifest['config']['model']['method']):
            receipt = Path(record['artifact_path'])/'provenance/controller-result.json'
            if status['output_hashes'].get('provenance/controller-result.json') != file_sha256(receipt):
                raise ValueError('failed-attempt accounting receipt changed')
            attempts += int(_read(receipt)['spawned'] is True)
    check_successors(attempts, study['campaign']['max_successors_per_failure'])
    category = 'scientific' if stage == 'baseline' else 'software'
    cap = study['campaign'][category]['max_run_wall_seconds']
    plan = dict(manifest_sha256=identity.manifest_sha256, source_sha=manifest['git_sha'],
                command_sha256=manifest['command_sha256'], controller_sha256=file_sha256(Path(__file__)),
                gpu_count=cards, max_wall_seconds=cap)
    plan_path = root/'provenance/controller-plan.json'
    if plan_path.exists():
        if _read(plan_path) != plan: raise ValueError('controller recovery plan changed')
    else: publish_json(plan_path, plan)
    intent = root/'provenance/controller-launch-intent.json'
    if intent.exists(): raise ValueError('prior launch intent requires manual integrity recovery')
    command = verified['command_argv']
    devices = next((s.split('=', 1)[1].split(',') for s in command if s.startswith('CUDA_VISIBLE_DEVICES=')), [])
    if cards and (len(devices) != cards or len(set(devices)) != cards): raise ValueError('sealed GPU devices differ from budget')
    deadline = time.monotonic()+study['campaign'].get('max_resource_wait_seconds', 21600)
    while cards and not set(devices).issubset(available_devices(study)):
        if time.monotonic() >= deadline:
            publish_json(root/'provenance/controller-result.json', dict(**plan, spawned=False, elapsed_seconds=0.0,
                         terminal_exit_code=75, reason='external_gpu_occupancy_timeout'))
            return _terminal(manager, identity, 75, cancelled=True)
        _audit(root, 'waiting_for_external_gpu_jobs', devices=devices)
        time.sleep(min(30, max(0, deadline-time.monotonic())))
    manager.verify_run(identity)  # Recheck committed source and every input after waiting.
    publish_json(intent, dict(**plan, controller_pid=os.getpid(),
                 controller_start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')', 1)[1].split()[19]))
    start = time.monotonic(); process = None; reason = 'worker_exit'; observed = None
    try:
        with (root/'logs/worker.log').open('xb') as output:
            process = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            publish_json(root/'provenance/controller-launch.json', dict(**plan, pid=process.pid,
                         process_start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]))
            try: observed = process.wait(timeout=max(1, cap-25))
            except subprocess.TimeoutExpired: reason = 'wall_timeout'
            finally: _stop_owned_group(process)
    except BaseException as exc:
        if process is not None: _stop_owned_group(process)
        # A launch/IO failure is diagnosed in its own terminal run, never retried here.
        reason = 'controller_or_launch_failure'
        _audit(root, reason, error=f'{type(exc).__name__}: {exc}')
    elapsed = time.monotonic()-start if process is not None else 0.0
    _, status = manager._load_run(root)
    exit_code = 0 if reason == 'worker_exit' and observed == 0 and status['state'] == 'validating' else 1
    publish_json(root/'provenance/controller-result.json', dict(**plan, spawned=process is not None,
                 elapsed_seconds=elapsed, terminal_exit_code=exit_code, observed_launcher_exit_code=observed, reason=reason))
    return _terminal(manager, identity, exit_code)


def run_phase(repo, phase_path, artifact_root):
    from scripts import manage_experiment_run as manager
    repo = Path(repo).resolve(); phase_path = Path(phase_path)
    registry = repo/CONTROL/'registry.jsonl'
    with termination_signals(), campaign_lock(registry.with_name('registry.jsonl.campaign.lock')):
        manager._require_clean(repo)
        study = yaml.safe_load(_committed(repo, repo/CONTROL/'study.yaml'))
        if study['study'] != STUDY: raise ValueError('wrong campaign study')
        phase = manager._load_committed_phase(repo, phase_path)
        if phase['study'] != STUDY: raise ValueError('wrong phase study')
        records = load_registry(registry)
        reservations = [manager._reservation_for_spec(s, records, Path(artifact_root).resolve()) for s in phase['runs']]
        totals = budget_projection(_budget_rows(repo, study, manager))
        if any(totals[k] > study['campaign'][k]['max_cumulative_gpu_hours'] for k in totals):
            raise ValueError('whole-phase worst-case GPU budget exceeded')
        states = []; failed = False
        for reservation in reservations:
            identity = _identity(repo, registry, reservation)
            _, existing = manager._load_run(Path(identity.run_dir))
            if existing['state'] in TERMINAL:
                # Exact output inventory is revalidated by finalization below.
                state = existing['state']
            elif failed:
                manager.verify_run(identity)
                root = Path(identity.run_dir); manifest, _ = manager._load_run(root)
                publish_json(root/'provenance/controller-result.json', dict(spawned=False, elapsed_seconds=0.0,
                             gpu_count=manifest['environment']['gpu_count'], manifest_sha256=identity.manifest_sha256,
                             source_sha=manifest['git_sha'], terminal_exit_code=1, reason='prior_phase_arm_failed'))
                state = _terminal(manager, identity, 1, cancelled=True)
            else: state = _run_one(repo, identity, study, manager)
            states.append(state); failed |= state != 'completed'
        summary = Path(CONTROL)/'summaries'/f'{phase_path.stem}.json'
        manager.finalize_phase(SimpleNamespace(repo_root=repo, phase=phase_path, registry=registry.relative_to(repo),
                                               artifact_root=Path(artifact_root), summary=summary))
        subprocess.run(['git', 'add', str(registry.relative_to(repo)), str(summary)], cwd=repo, check=True)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo).returncode:
            subprocess.run(['git', 'commit', '-m', f'archive: finalize full-data phase {phase["phase"]}'], cwd=repo, check=True)
        return 1 if failed else 0
