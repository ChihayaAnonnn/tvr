"""Read-only provenance audit; no captions, role assignment, or gate changes."""
import itertools
import json
import math
import re
from collections import defaultdict
from urllib.parse import parse_qs, urlparse

from semantic_retrieval.data.msrvtt_snapshot import Cursor, record_fields


def training_source_metadata(raw, allowed, *, decode_value=json.loads):
    if not allowed or any(not isinstance(i, str) or not i for i in allowed):
        raise ValueError('nonempty training allowlist required')
    cursor = Cursor(raw)
    cursor.expect(b'{')
    token = cursor.token()
    rows, seen, roots = [], set(), set()
    counts = dict(selected=0, excluded=0, excluded_values_decoded=0, captions_decoded=0)
    while token != b'}':
        key = json.loads(token)
        if key in roots:
            raise ValueError('duplicate metadata root')
        roots.add(key)
        cursor.expect(b':')
        if key != 'videos':
            cursor.value()  # Sentence values remain encoded bytes.
        else:
            cursor.expect(b'[')
            while True:
                start = cursor.pos
                token = cursor.token()
                if token == b']':
                    break
                cursor.pos = start
                fields = record_fields(cursor.value())
                video = json.loads(fields['video_id'])
                if not isinstance(video, str) or not video or video in seen:
                    raise ValueError('invalid or duplicate video identity')
                seen.add(video)
                if video in allowed:
                    names = ('url', 'start time', 'end time')
                    if not set(names) <= fields.keys():
                        raise ValueError('missing training source metadata')
                    rows.append(dict(video_id=video, **{k: decode_value(fields[k]) for k in names}))
                    counts['selected'] += 1
                else:
                    counts['excluded'] += 1
                token = cursor.token()
                if token == b']':
                    break
                if token != b',':
                    raise ValueError('invalid metadata delimiter')
        token = cursor.token()
        if token == b'}':
            break
        if token != b',':
            raise ValueError('invalid root delimiter')
        token = cursor.token()
    if 'videos' not in roots or raw[cursor.pos:].strip() or {r['video_id'] for r in rows} != allowed:
        raise ValueError('incomplete metadata or trailing bytes')
    return rows, counts


def summarize_sources(rows, train, validation, content):
    allowed = train | validation
    ids = [r['video_id'] for r in rows]
    if (not train or not validation or train & validation or len(ids) != len(set(ids))
            or set(ids) != allowed or set(content) != allowed
            or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in content.values())):
        raise ValueError('source inventory/membership mismatch')
    groups, hashes = defaultdict(list), defaultdict(list)
    for r in rows:
        parsed = urlparse(r['url'])
        source = parse_qs(parsed.query).get('v', [])
        times = (r['start time'], r['end time'])
        if (parsed.scheme != 'https' or parsed.hostname not in {'youtube.com','www.youtube.com'}
                or len(source) != 1 or re.fullmatch('[A-Za-z0-9_-]{11}',source[0]) is None
                or any(type(t) not in {int,float} or not math.isfinite(t) for t in times)
                or not 0 <= times[0] < times[1]):
            raise ValueError('invalid original source or interval')
        groups[source[0]].append(r)
        hashes[content[r['video_id']]].append(r['video_id'])
    cross = [g for g in groups.values() if any(r['video_id'] in train for r in g)
             and any(r['video_id'] in validation for r in g)]
    pairs = []
    for group in cross:
        for a,b in itertools.combinations(group,2):
            if (a['video_id'] in train) == (b['video_id'] in train):
                continue
            overlap = min(a['end time'],b['end time']) - max(a['start time'],b['start time'])
            if overlap > 0:
                a,b = (a,b) if a['video_id'] in train else (b,a)
                pairs.append(dict(train_video_id=a['video_id'],val_video_id=b['video_id'],overlap_seconds=round(overlap,6)))
    return dict(source_groups=len(groups),multi_clip_source_groups=sum(len(g)>1 for g in groups.values()),
                cross_train_validation_source_groups=len(cross),
                train_videos_in_cross_source_groups=sum(r['video_id'] in train for g in cross for r in g),
                validation_videos_in_cross_source_groups=sum(r['video_id'] in validation for g in cross for r in g),
                cross_partition_temporally_overlapping_pairs=len(pairs),
                overlapping_pair_examples=sorted(pairs,key=lambda p:(-p['overlap_seconds'],p['train_video_id'],p['val_video_id']))[:10],
                exact_duplicate_content_groups=sum(len(g)>1 for g in hashes.values()),
                cross_partition_exact_duplicate_groups=sum(any(i in train for i in g) and any(i in validation for i in g) for g in hashes.values()))
