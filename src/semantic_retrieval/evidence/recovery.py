"""Immutable, hash-bound evaluation snapshots imported into unique successor runs."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import shutil
import stat

from experiment_artifacts import strict_json_load
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json,safe_path
from semantic_retrieval.evidence.types import digest,state_from_payload,require_digest
from semantic_retrieval.runtime.checkpoint import _publish_file


def _initial_path(store,query_id):
    return store.run_root/'provenance'/'evidence-initial'/(digest(query_id)+'.json')


def save_initial_state(store,state):
    path = _initial_path(store,state.query_id)
    payload = asdict(state)
    if path.exists():
        if read_initial_state(store,state.query_id) != state:
            raise ValueError('initial recovery state is immutable')
        return
    publish_json(path,payload)


def read_initial_state(store,query_id):
    path = _initial_path(store,query_id);safe_path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        state = state_from_payload(strict_json_load(handle))
    if state.query_id != query_id:
        raise ValueError('recovery query identity mismatch')
    return state


def _allowed(relative):
    return relative.parts[0] == 'arrays' or (relative.parts[:2] in (
        ('provenance','frame-cache'),('provenance','evidence-actions'),('provenance','evidence-initial'),('provenance','evidence-gallery')))


def _directory_inventory(directory,run_root):
    """Hash every leaf through a checked directory descriptor, without ancestor rescans."""
    safe_path(directory)
    if not directory.exists():
        return []
    files = []
    def identity(value):
        return value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_size,value.st_mtime_ns,value.st_ctime_ns
    def visit(parent,descriptor):
        before_directory = os.fstat(descriptor)
        for name in os.listdir(descriptor):
            path = Path(parent)/name
            expected = os.stat(name,dir_fd=descriptor,follow_symlinks=False)
            if stat.S_ISDIR(expected.st_mode):
                child = os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
                try:
                    if identity(expected) != identity(os.fstat(child)):
                        raise ValueError('unsafe recovery directory changed')
                    visit(path,child)
                finally:
                    os.close(child)
                if identity(expected) != identity(os.stat(name,dir_fd=descriptor,follow_symlinks=False)):
                    raise ValueError('unsafe recovery directory changed')
                continue
            if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
                raise ValueError('unsafe or nonprivate recovery artifact')
            if path.suffix not in {'.npy','.json'}:
                raise ValueError('partial recovery publication')
            try:
                fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=descriptor)
            except OSError as error:
                raise ValueError('unsafe recovery artifact') from error
            with os.fdopen(fd,'rb') as handle:
                before = os.fstat(handle.fileno())
                if identity(before) != identity(expected):
                    raise ValueError('unsafe recovery artifact changed')
                hasher = hashlib.sha256()
                for block in iter(lambda: handle.read(1024*1024),b''):
                    hasher.update(block)
                after = os.fstat(handle.fileno())
                if identity(before) != identity(after):
                    raise ValueError('recovery artifact changed during hashing')
            if identity(expected) != identity(os.stat(name,dir_fd=descriptor,follow_symlinks=False)):
                raise ValueError('unsafe recovery artifact changed')
            files.append(dict(path=path.relative_to(run_root).as_posix(),sha256=hasher.hexdigest()))
        if identity(before_directory) != identity(os.fstat(descriptor)):
            raise ValueError('unsafe recovery directory changed')
    expected_root = os.stat(directory,follow_symlinks=False)
    descriptor = os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        if identity(expected_root) != identity(os.fstat(descriptor)):
            raise ValueError('unsafe recovery directory changed')
        visit(directory,descriptor)
    finally:
        os.close(descriptor)
    safe_path(directory)
    if identity(expected_root) != identity(os.stat(directory,follow_symlinks=False)):
        raise ValueError('unsafe recovery directory changed')
    return sorted(files,key=lambda item: Path(item['path']))


def export_recovery(store,*,config_sha256,source_sha):
    require_digest(config_sha256)
    if len(source_sha) != 40 or any(c not in '0123456789abcdef' for c in source_sha):
        raise ValueError('recovery source identity malformed')
    files = []
    for directory in (store.root,*(store.run_root/'provenance'/name for name in ('frame-cache','evidence-actions','evidence-initial','evidence-gallery'))):
        files.extend(_directory_inventory(directory,store.run_root))
    if not files:
        raise ValueError('empty evaluation recovery snapshot')
    shards = []
    for start in range(0,len(files),128):
        part = dict(schema_version='evidence-recovery-files-v1',files=files[start:start+128])
        path = store.run_root/'provenance'/'evidence-recovery-files'/(digest(part)+'.json')
        if not path.exists():
            publish_json(path,part)
        shards.append(dict(path=path.relative_to(store.run_root).as_posix(),sha256=file_sha256(path)))
    payload = dict(schema_version='evidence-recovery-v1',config_sha256=config_sha256,source_sha=source_sha,file_count=len(files),shards=shards)
    path = store.run_root/'provenance'/'evidence-recovery'/(digest(payload)+'.json')
    if not path.exists():
        publish_json(path,payload)
    return path


def import_recovery(reference,store,*,config_sha256,source_sha):
    """Validate the entire snapshot before copying; never overwrite conflicting output."""
    safe_path(reference.path)
    if file_sha256(reference.path) != reference.sha256:
        raise ValueError('recovery root hash mismatch')
    with reference.path.open() as handle:
        payload = strict_json_load(handle)
    if set(payload) != {'schema_version','config_sha256','source_sha','file_count','shards'} or payload['schema_version'] != 'evidence-recovery-v1' or payload['config_sha256'] != config_sha256 or payload['source_sha'] != source_sha:
        raise ValueError('recovery configuration/source identity mismatch')
    root = reference.path.parents[2]
    files,seen = [],set()
    for ref in payload['shards']:
        if set(ref) != {'path','sha256'}:
            raise ValueError('invalid recovery shard reference')
        rel = Path(ref['path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[:2] != ('provenance','evidence-recovery-files'):
            raise ValueError('invalid recovery shard path')
        shard = root/rel;safe_path(shard)
        if file_sha256(shard) != ref['sha256']:
            raise ValueError('recovery shard hash mismatch')
        with shard.open() as handle:
            part = strict_json_load(handle)
        if set(part) != {'schema_version','files'} or part['schema_version'] != 'evidence-recovery-files-v1':
            raise ValueError('invalid recovery shard schema')
        for item in part['files']:
            if set(item) != {'path','sha256'}:
                raise ValueError('invalid recovery file reference')
            rel = Path(item['path']);require_digest(item['sha256'])
            if rel.is_absolute() or '..' in rel.parts or rel in seen or not _allowed(rel):
                raise ValueError('invalid recovery file path')
            seen.add(rel);path = root/rel;safe_path(path)
            if file_sha256(path) != item['sha256']:
                raise ValueError('recovery file hash mismatch')
            target = store.run_root/rel;safe_path(target)
            if target.exists() and file_sha256(target) != item['sha256']:
                raise ValueError('successor recovery output conflict')
            files.append((path,target,item['sha256']))
    if type(payload['file_count']) is not int or len(files) != payload['file_count']:
        raise ValueError('recovery file count mismatch')
    for source,target,sha in files:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            def copy(handle,source=source):
                with source.open('rb') as original:
                    shutil.copyfileobj(original,handle)
            _publish_file(target,copy)
        if file_sha256(target) != sha:
            raise ValueError('successor recovery copy hash mismatch')
