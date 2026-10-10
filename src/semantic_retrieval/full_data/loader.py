"""Authenticated full-caption datasets and pinned upstream 1 FPS frame policy."""
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiment_artifacts import strict_json_load
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.data.tokenizer import ClipTokenizer
from semantic_retrieval.data.video import decode_video
from semantic_retrieval.evidence.frame_cache import safe_path


def official_frame_indices(frame_count, fps, max_frames):
    if (type(frame_count) is not int or frame_count < 1 or type(max_frames) is not int or max_frames < 1
            or type(fps) not in (float, int) or not math.isfinite(fps) or fps < 1):
        raise ValueError('invalid 1 FPS frame sampling inputs')
    candidates = list(range(0, frame_count, int(fps)))
    if len(candidates) > max_frames:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, max_frames).astype(int)]
    return candidates


def read_complete(report_path, report_sha256, references, *, expected_train=8246, expected_val=500, captions_per_video=20):
    def read(path, sha):
        path = Path(path); safe_path(path)
        if references.get(str(path)) != sha or file_sha256(path) != sha:
            raise ValueError('complete cohort input hash/binding mismatch')
        if path.stat().st_size >= 1024**2:
            raise ValueError('complete cohort JSON must be sharded below 1 MiB')
        with path.open() as stream:
            return strict_json_load(stream)
    report_path = Path(report_path)
    report = read(report_path, report_sha256)
    if (report.get('schema_version') != 'full-data-preparation-v1' or report.get('test_authorized') is not False
            or report.get('test_caption_values_decoded') != 0 or report.get('train_video_count') != expected_train
            or report.get('validation_video_count') != expected_val
            or report.get('decoded_video_count') != expected_train+expected_val):
        raise ValueError('complete cohort certification scope mismatch')
    records = dict(videos=[], captions=[], sources=[])
    seen = set()
    for reference in report['input_indices']:
        index = read(reference['path'], reference['sha256'])
        if index.get('schema_version') != 1 or index.get('kind') != 'file_sha256_index':
            raise ValueError('invalid cohort input index')
        for item in index['files']:
            path = Path(item['path'])
            kind = path.name.split('-')[0]
            if path in seen or path.parent != report_path.parent/'inputs' or kind not in records:
                raise ValueError('invalid or duplicate cohort shard')
            seen.add(path)
            payload = read(path, item['sha256'])
            if set(payload) != {'schema_version', 'rows'} or payload['schema_version'] != 1 or not payload['rows']:
                raise ValueError('invalid cohort shard schema')
            records[kind].extend(payload['rows'])
    videos = {}
    for row in records['videos']:
        if (set(row) != {'video_id', 'split', 'source_component', 'path', 'sha256', 'frame_count', 'fps'}
                or row['video_id'] in videos or row['split'] not in {'train', 'val'}
                or not isinstance(row['source_component'], str) or re.fullmatch('[0-9a-f]{64}', row['source_component']) is None
                or references.get(row['path']) != row['sha256']):
            raise ValueError('video identity/split/content not bound')
        official_frame_indices(row['frame_count'], row['fps'], 12)
        videos[row['video_id']] = row
    if (sum(r['split'] == 'train' for r in videos.values()) != expected_train
            or sum(r['split'] == 'val' for r in videos.values()) != expected_val):
        raise ValueError('complete video coverage mismatch')
    ownership = {}
    for row in videos.values():
        component = row['source_component']
        if component in ownership and ownership[component] != row['split']:
            raise ValueError('source component crosses training and validation')
        ownership[component] = row['split']
    source_ids, excluded = set(), 0
    for row in records['sources']:
        if (set(row) != {'video_id', 'source_component', 'split'} or row['video_id'] in source_ids
                or row['split'] not in {'train', 'val', 'quarantined_train'}):
            raise ValueError('invalid source assignment schema/identity')
        source_ids.add(row['video_id'])
        if row['split'] == 'quarantined_train':
            excluded += 1
            if row['video_id'] in videos:
                raise ValueError('quarantined source entered complete cohort')
        elif (row['video_id'] not in videos or videos[row['video_id']]['split'] != row['split']
              or videos[row['video_id']]['source_component'] != row['source_component']):
            raise ValueError('source assignment disagrees with cohort video')
    if not set(videos) <= source_ids or excluded != report['excluded_train_count'] or len(source_ids) != len(videos)+excluded:
        raise ValueError('incomplete source assignment coverage')
    captions, query_ids, counts = {'train': [], 'val': []}, set(), {i: 0 for i in videos}
    for row in records['captions']:
        if (set(row) != {'query_id', 'caption_id', 'official_video_id', 'query_text'}
                or row['query_id'] in query_ids or row['official_video_id'] not in videos
                or not isinstance(row['caption_id'], str) or not row['caption_id']
                or row['query_id'] != row['official_video_id']+':'+row['caption_id']
                or not isinstance(row['query_text'], str) or not row['query_text'].strip()):
            raise ValueError('caption identity/content/relationship mismatch')
        query_ids.add(row['query_id']); counts[row['official_video_id']] += 1
        captions[videos[row['official_video_id']]['split']].append(row)
    if not counts or set(counts.values()) != {captions_per_video}:
        raise ValueError('complete caption coverage mismatch')
    if (len(captions['train']) != report['train_caption_count'] or len(captions['val']) != report['validation_caption_count']
            or sum(counts.values()) != report['caption_count']):
        raise ValueError('caption coverage disagrees with certification counts')
    for rows in captions.values():
        rows.sort(key=lambda r: r['query_id'])
    return videos, captions


class CaptionDataset(Dataset):
    def __init__(self, videos, captions, *, frames=12, image_size=224, text_length=32):
        self.videos, self.captions = videos, captions
        self.frames, self.image_size = frames, image_size
        self.video_ids = {i: n for n, i in enumerate(sorted(videos))}
        tokenizer = ClipTokenizer()
        self.tokens, self.text_masks = tokenizer.encode_batch([r['query_text'] for r in captions], max_length=text_length)

    def __len__(self):
        return len(self.captions)

    def text(self, index):
        return self.tokens[index], self.text_masks[index]

    def decode(self, video_id):
        row = self.videos[video_id]
        if file_sha256(row['path']) != row['sha256']:
            raise ValueError('video changed after cohort certification')
        indices = official_frame_indices(row['frame_count'], row['fps'], self.frames)
        frames, mask = decode_video(row['path'], indices, self.image_size, decoder_threads=1)
        if not mask.all():
            raise ValueError('certified frame decode failed; refusing partial training example')
        video = torch.zeros(self.frames, 3, self.image_size, self.image_size)
        valid = torch.zeros(self.frames, dtype=torch.bool)
        video[:len(frames)], valid[:len(frames)] = frames, True
        return video, valid

    def __getitem__(self, index):
        row = self.captions[index]
        video, frame_mask = self.decode(row['official_video_id'])
        return (*self.text(index), video, frame_mask, torch.tensor(self.video_ids[row['official_video_id']]))
