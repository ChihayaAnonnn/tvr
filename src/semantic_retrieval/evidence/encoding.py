"""Pack observations without merging evidence across queries."""
import torch
from semantic_retrieval.types import EncodedFeatures
from semantic_retrieval.evidence.types import ObservedFeatures
from semantic_retrieval.evidence.views import coverage_vector


def pack_observations(states, queries, frame_reader):
    if len(states) != len(queries.text_tokens):
        raise ValueError('one query feature row required per observation state')
    keys, rows, positions, pairs, ids, offsets = [], [], [], [], [], [0]
    coverage = []
    d = queries.pooled_text.shape[-1]
    for q, state in enumerate(states):
        for c in state.candidates:
            # Reader identity binds preprocessing and encoder; content prevents stale canonical reuse.
            key = (c.canonical_id, c.content_sha256, c.observed_frame_ids,
                   getattr(frame_reader, 'preprocessing_sha256', None), getattr(frame_reader, 'clip_sha256', None))
            if key not in keys:
                row = frame_reader(c, c.observed_frame_ids)
                if row.shape != (len(c.observed_frame_ids), d) or row.requires_grad or not torch.isfinite(row).all():
                    raise ValueError('reader must return detached finite observed CLIP vectors')
                keys.append(key); rows.append(row.to(queries.pooled_text.device))
                positions.append(tuple(c.pool_frame_ids.index(i) for i in c.observed_frame_ids))
            coverage.append(coverage_vector(c))
            pairs.append((q, keys.index(key))); ids.append((state.query_id, c.canonical_id))
        offsets.append(len(pairs))
    device = queries.pooled_text.device
    n = max(len(r) for r in rows)
    frames = queries.pooled_text.new_zeros((len(rows), n, d))
    mask = torch.zeros((len(rows), n), dtype=torch.bool, device=device)
    pos = torch.zeros((len(rows), n), dtype=torch.long, device=device)
    for i, (r, p) in enumerate(zip(rows, positions)):
        frames[i, :len(r)] = r; mask[i, :len(r)] = True
        pos[i, :len(r)] = torch.tensor(p, device=device)
    pooled = frames.sum(1)/mask.sum(1, keepdim=True)
    f = EncodedFeatures(queries.text_tokens, queries.pooled_text, frames, pooled, queries.text_mask, mask)
    return ObservedFeatures(f, pos, torch.tensor(pairs, dtype=torch.long, device=device),
                            torch.tensor(offsets, dtype=torch.long, device=device), tuple(ids), tuple(keys), torch.stack(coverage).to(device))


def encode_observations(state, frame_cache, *, query_features):
    return pack_observations((state,), query_features, frame_cache)
