"""Allowlist-first full-cohort compilation; no pilot or caption subsampling."""
import json

from semantic_retrieval.data.msrvtt_snapshot import training_annotations, caption_relation_digest
from semantic_retrieval.evidence.source_audit import training_source_metadata
from semantic_retrieval.evidence.source_preparation import partition_sources


def compile_sources(raw, allowlist, trusted, content, *, decode_caption=json.loads):
    train_ids, val_ids = trusted['train_video_ids'], trusted['val_video_ids']
    train, val = set(train_ids), set(val_ids)
    if (not train or not val or len(train) != len(train_ids) or len(val) != len(val_ids)
            or train & val or train | val != set(allowlist) or set(content) != set(allowlist)):
        raise ValueError('full-data membership mismatch')
    metadata, metadata_privacy = training_source_metadata(raw, set(allowlist))
    captions, privacy = training_annotations(raw, set(allowlist), decode_caption=decode_caption)
    if {r['official_video_id'] for r in captions} != set(allowlist):
        raise ValueError('incomplete caption coverage')
    _, components, excluded = partition_sources(metadata, train, val, content, seed=0)
    eligible = train - excluded
    if not eligible:
        raise ValueError('no source-isolated training videos')
    return dict(schema_version='full-data-sources-v1', train_video_ids=sorted(eligible),
                validation_video_ids=sorted(val), excluded_train_ids=sorted(excluded),
                components=components, content=content,
                captions=sorted((r for r in captions if r['official_video_id'] not in excluded),
                                key=lambda r: r['query_id']),
                all_allowed_caption_relation_sha256=caption_relation_digest(captions),
                privacy=privacy, metadata_privacy=metadata_privacy)
