"""Authenticated complete-cohort CPU preparation and actual per-video decoding."""
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.data.msrvtt_snapshot import inspect_training_video
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.evidence.source_preparation import preparation_spec
from semantic_retrieval.full_data.data import compile_sources
from semantic_retrieval.runtime.input_bindings import context_input_references, write_input_indices


def _shards(directory, prefix, rows, references):
    for offset in range(0, len(rows), 1000):
        path = directory/f'{prefix}-{offset//1000:03d}.json'
        publish_json(path, dict(schema_version=1, rows=rows[offset:offset+1000]))
        references.append(dict(path=str(path), sha256=file_sha256(path)))


def export_complete(directory, compiled, video_directory, *, image_size=224):
    directory, video_directory = Path(directory), Path(video_directory)
    directory.mkdir()  # Never overwrite or silently resume partial exports.
    cv2.setNumThreads(1)
    membership = {i: 'train' for i in compiled['train_video_ids']}
    membership.update({i: 'val' for i in compiled['validation_video_ids']})
    def inspect(i):
        path = video_directory/(i+'.mp4')
        cap = cv2.VideoCapture(str(path))
        try:
            advertised, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if (not math.isfinite(advertised) or advertised < 1 or advertised != int(advertised)
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError('invalid video container frame count/fps')
        count = inspect_training_video(path, expected_sha256=compiled['content'][i],
                                       expected_frame_count=int(advertised), image_size=image_size, max_frames=12)
        return dict(video_id=i, split=membership[i], source_component=compiled['components'][i],
                    path=str(path), sha256=compiled['content'][i], frame_count=count, fps=fps)
    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = list(pool.map(inspect, sorted(membership)))
    references = []
    _shards(directory, 'videos', videos, references)
    _shards(directory, 'captions', compiled['captions'], references)
    assignments = [dict(video_id=i, source_component=c, split=membership.get(i, 'quarantined_train'))
                   for i, c in sorted(compiled['components'].items())]
    _shards(directory, 'sources', assignments, references)
    indices = write_input_indices(directory, 'complete-cohort', references)
    return dict(schema_version='full-data-preparation-v1', decoded_video_count=len(videos),
                train_video_count=len(compiled['train_video_ids']),
                validation_video_count=len(compiled['validation_video_ids']),
                excluded_train_count=len(compiled['excluded_train_ids']),
                caption_count=len(compiled['captions']),
                train_caption_count=sum(r['official_video_id'] in compiled['train_video_ids'] for r in compiled['captions']),
                validation_caption_count=sum(r['official_video_id'] in compiled['validation_video_ids'] for r in compiled['captions']),
                input_indices=indices, privacy=compiled['privacy'], metadata_privacy=compiled['metadata_privacy'],
                all_allowed_caption_relation_sha256=compiled['all_allowed_caption_relation_sha256'],
                test_authorized=False, test_caption_values_decoded=0,
                unknown_cross_url_visual_duplicates='not_excluded_explicit_limitation')


def prepare_complete(context):
    started = time.monotonic()
    # Reuse immutable raw-input identities, without the old study's pilot roles/permissions.
    spec = preparation_spec()
    references = context_input_references(context)
    paths = {}
    for name in ('annotation', 'allowlist', 'trusted_split'):
        path = Path(spec[name])
        if not path.is_absolute():
            path = context.repo_root/path
        if (references.get(str(path), references.get(spec[name])) != spec[name+'_sha256']
                or file_sha256(path) != spec[name+'_sha256']):
            raise ValueError('full-data input identity mismatch')
        paths[name] = path
    with paths['allowlist'].open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['video_id']:
            raise ValueError('invalid allowlist schema')
        ids = [r['video_id'] for r in reader]
    trusted = json.loads(paths['trusted_split'].read_bytes())
    if (len(ids) != 9000 or len(set(ids)) != 9000 or len(trusted['train_video_ids']) != 8500
            or len(trusted['val_video_ids']) != 500):
        raise ValueError('frozen full-data cohort cardinality changed')
    content = {}
    for i in sorted(ids):
        path = Path(spec['video_directory'])/(i+'.mp4')
        sha = references.get(str(path))
        if sha is None or file_sha256(path) != sha:
            raise ValueError('video missing from authenticated inventory or changed')
        content[i] = sha
    compiled = compile_sources(paths['annotation'].read_bytes(), set(ids), trusted, content)
    if (len(compiled['train_video_ids']) != 8246 or len(compiled['excluded_train_ids']) != 254
            or compiled['all_allowed_caption_relation_sha256'] != spec['caption_relation_sha256']):
        raise ValueError('approved source isolation/caption relation changed')
    result = export_complete(context.paths.provenance/'inputs', compiled, Path(spec['video_directory']))
    result.update(elapsed_seconds=time.monotonic()-started, source_sha=context.config.source_sha,
                  manifest_sha256=context.config.manifest_sha256)
    publish_json(context.paths.provenance/'complete-preparation.json', result)
    return result
