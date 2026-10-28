"""Source-disjoint role bindings reuse the audited episode reader."""
from dataclasses import dataclass

from experiment_artifacts import strict_json_load
from semantic_retrieval.data.episodes import Episode, read_episodes, file_sha256
from semantic_retrieval.evidence.config import ROLES
from semantic_retrieval.evidence.frame_cache import safe_path


@dataclass(frozen=True)
class EvidenceEpisode:
    episode: Episode
    role: str
    source_component_ids: tuple[str, ...]

    def __post_init__(self):
        if self.role not in ROLES or not self.source_component_ids:
            raise ValueError('explicit source component identities required')


def read_split_binding(binding):
    p = binding.manifest.path
    safe_path(p)
    if file_sha256(p) != binding.manifest.sha256:
        raise ValueError('split manifest hash mismatch')
    with p.open() as h:
        payload = strict_json_load(h)
    split = {'T_match':'train', 'T_value':'train', 'V_development':'val_development',
             'V_calibration':'val_calibration', 'V_locked':'val_locked'}[binding.role]
    episodes = read_episodes(p, gallery_version=payload['gallery_version'], split=split)
    result = []
    for e in episodes:
        if e.relevance_space != binding.relevance_space:
            raise ValueError('split relevance scope mismatch')
        components = set(e.source_ids) | set(e.query_source_ids)
        if e.query_component_id:
            components.add(e.query_component_id)
        if not components or not e.caption_group_id:
            raise ValueError('query/candidate source metadata required')
        result.append(EvidenceEpisode(e, binding.role, tuple(sorted(components))))
    audit_component_roles(result)
    return tuple(result)


def audit_component_roles(bindings):
    items = []
    for item in bindings:
        if isinstance(item, EvidenceEpisode):
            items.append(item)
        else:
            items.extend(read_split_binding(item))
    seen = {}
    for item in items:
        e = item.episode
        identities = [('source', x) for x in item.source_component_ids]
        identities += [('content', x) for x in e.content_sha256]
        identities += [('canonical', x) for x in e.canonical_ids]
        identities += [('query', e.query_id), ('caption', e.caption_group_id)]
        for key in identities:
            if key in seen and seen[key] != item.role:
                raise ValueError(f'source role leakage: {key}')
            seen[key] = item.role
