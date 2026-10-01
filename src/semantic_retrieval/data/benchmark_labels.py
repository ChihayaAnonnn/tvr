"""Official benchmark membership, explicitly distinct from semantic relevance."""
import json
from dataclasses import dataclass
from pathlib import Path
from .episodes import file_sha256, _digest
from .pilot_catalog import identity_digest


@dataclass(frozen=True)
class BenchmarkLabels:
    protocol_id: str
    annotation_sha256: str
    relation_sha256: str
    id_map_sha256: str
    positives_by_query: dict
    official_to_canonical: dict
    official_positives_by_query: dict
    query_texts: dict
    caption_ids: dict
    source_report_sha256: str
    allowlist_sha256: str
    source_annotation_sha256: str
    duplicate_rows: int = 0

    @property
    def protocol(self):
        return dict(protocol_id=self.protocol_id,annotation_sha256=self.annotation_sha256,
                    relation_sha256=self.relation_sha256,id_map_sha256=self.id_map_sha256,
                    source_report_sha256=self.source_report_sha256,allowlist_sha256=self.allowlist_sha256,
                    source_annotation_sha256=self.source_annotation_sha256)


def load_benchmark_labels(snapshot, *, catalog, expected_sha256, protocol_id) -> BenchmarkLabels:
    path=Path(snapshot)
    if file_sha256(path)!=expected_sha256:
        raise ValueError('benchmark snapshot SHA mismatch')
    payload=json.loads(path.read_text())
    if (set(payload)!={'schema_version','origin_split','protocol_id','source_annotation_sha256',
                       'source_report_sha256','allowlist_sha256','annotations'}
            or payload['schema_version']!=1 or payload['origin_split']!='source_train'
            or payload['protocol_id']!=protocol_id
            or any(not _digest(payload[k]) for k in ('source_annotation_sha256','source_report_sha256','allowlist_sha256'))):
        raise ValueError('benchmark snapshot provenance/scope mismatch')
    videos={e.official_video_id:e.canonical_id for e in catalog if e.kind=='video'}
    queries={e.asset_id:e for e in catalog if e.kind=='query'}
    official={};texts={};captions={};duplicates=0
    for row in payload['annotations']:
        if (set(row)!={'query_id','caption_id','official_video_id','query_text'}
                or any(not isinstance(row[k],str) or not row[k] for k in row)
                or row['query_id'] not in queries or row['official_video_id'] not in videos
                or queries[row['query_id']].official_video_id != row['official_video_id']):
            raise ValueError('missing/invalid official relationship')
        q=row['query_id']; value=(row['official_video_id'],row['query_text'],row['caption_id'])
        if q in official:
            if (official[q][0],texts[q],captions[q])!=value:
                raise ValueError('contradictory duplicate annotation')
            duplicates+=1;continue
        official[q]=(row['official_video_id'],);texts[q]=row['query_text'];captions[q]=row['caption_id']
    if set(queries)!=set(official):
        raise ValueError('missing query annotations cannot become negative')
    positives={q:tuple(sorted({videos[v] for v in ids})) for q,ids in official.items()}
    if any(len(ids)!=1 for ids in positives.values()):
        raise ValueError('pilot requires exactly one canonical positive')
    return BenchmarkLabels(protocol_id,expected_sha256,identity_digest(official),identity_digest(videos),
                           positives,videos,official,texts,captions,payload['source_report_sha256'],
                           payload['allowlist_sha256'],payload['source_annotation_sha256'],duplicates)


def candidate_presence(query_id, candidate_canonical_ids, labels) -> bool:
    ids=tuple(candidate_canonical_ids)
    if (query_id not in labels.positives_by_query or len(set(ids))!=len(ids)
            or not set(ids).issubset(set(labels.official_to_canonical.values()))):
        raise ValueError('unknown query/candidate or duplicate canonical candidate')
    return bool(set(ids)&set(labels.positives_by_query[query_id]))
