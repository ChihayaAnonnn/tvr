"""Immutable, hash-bound evaluation snapshots imported into unique successor runs."""
from dataclasses import asdict
from pathlib import Path
import shutil

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


def export_recovery(store,*,config_sha256,source_sha):
    require_digest(config_sha256)
    if len(source_sha) != 40 or any(c not in '0123456789abcdef' for c in source_sha):
        raise ValueError('recovery source identity malformed')
    files = []
    for directory in (store.root,*(store.run_root/'provenance'/name for name in ('frame-cache','evidence-actions','evidence-initial','evidence-gallery'))):
        safe_path(directory)
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                safe_path(path)
                if path.is_file():
                    if path.suffix not in {'.npy','.json'}:
                        raise ValueError('partial recovery publication')
                    files.append(dict(path=path.relative_to(store.run_root).as_posix(),sha256=file_sha256(path)))
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
