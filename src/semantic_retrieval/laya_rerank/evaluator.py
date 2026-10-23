"""Complete permutations and immutable per-query blocks; labels stay outside the judge."""
import hashlib
import numpy as np
from .core import compare, rerank, validate_ranking
from .store import ArrayCache, canonical


def validate_inputs(queries, videos, rankings, *, expected_queries=None, expected_videos=None):
    a = np.asarray(rankings)
    if (a.shape != (len(queries), len(videos)) or not queries or len(videos) < 10
            or (expected_queries is not None and len(queries) != expected_queries)
            or (expected_videos is not None and len(videos) != expected_videos)):
        raise ValueError('incomplete validation coverage')
    ids = [v['video_id'] for v in videos]
    if len(set(ids)) != len(ids) or any(v['split'] != 'val' or not v['source_component'] for v in videos):
        raise ValueError('gallery identity/split invalid')
    if (len({q['query_id'] for q in queries}) != len(queries)
            or any(q['split'] != 'val' or q['video_id'] not in ids
                   or not isinstance(q['query_text'], str) or not q['query_text'].strip() for q in queries)):
        raise ValueError('query identity/split invalid')
    for row in a: validate_ranking(row)
    return a


def grouped_interval(delta, groups, *, seed=713, samples=10000):
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != (len(groups),) or not len(d) or not np.isfinite(d).all():
        raise ValueError('invalid bootstrap inputs')
    names, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=d)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(len(names), size=len(names))
        means[i] = 100*sums[selected].sum()/counts[selected].sum()
    return tuple(float(x) for x in np.quantile(means, [.025,.975]))


def evaluate(queries, videos, rankings, score, evidence, output, identity, *, microbatch=8, progress=None):
    """CPU orchestration; callers must authenticate inputs and enforce launch gates first.

    score receives only query strings and visual evidence, never evaluation labels.
    Each committed block has exactly ten (two logits, probability) rows.
    """
    if microbatch not in (1,8): raise ValueError('fixed microbatch must be 1 or 8')
    a = validate_inputs(queries, videos, rankings)
    cohort = hashlib.sha256(canonical({'queries': queries, 'videos': videos,
                                      'rankings_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                                      'rankings_dtype':str(a.dtype)})).hexdigest()
    store = ArrayCache(output, {'model':identity, 'cohort':cohort})
    full = []; all_blocks = []
    for q, row in zip(queries, a):
        key = {'query_id':q['query_id'], 'query_sha256':hashlib.sha256(q['query_text'].encode()).hexdigest(),
               'candidates':[videos[i]['video_id'] for i in row[:10]]}
        block = store.read(key)
        if block is None:
            parts=[]
            for start in range(0,10,microbatch):
                batch=row[start:min(start+microbatch,10)]
                z,p=score([q['query_text']]*len(batch),[evidence(videos[i]) for i in batch])
                z,p=np.asarray(z),np.asarray(p)
                if z.shape != (len(batch),2) or p.shape != (len(batch),) or not np.isfinite(z).all():
                    raise ValueError('invalid paired judge outputs')
                parts.append(np.column_stack([z,p]))
            block=np.concatenate(parts)
            rerank(row,block[:,2])  # Validate before publishing any completion marker.
            store.write(key,block)
        if block.shape != (10,3): raise ValueError('invalid completed query block')
        full.append(rerank(row,block[:,2])); all_blocks.append(block)
        if progress is not None and (len(full)%250==0 or len(full)==len(queries)):progress(len(full))
    b=np.stack(full)
    gallery={v['video_id']:i for i,v in enumerate(videos)}
    targets=np.array([gallery[q['video_id']] for q in queries])
    metrics=compare(a,b,targets)
    delta=(b[:,0] == targets).astype(int)-(a[:,0] == targets).astype(int)
    groups=[videos[gallery[q['video_id']]]['source_component'] for q in queries]
    interval=grouped_interval(delta,groups)
    improvement=float(100*delta.mean())
    verdict='positive_feasibility' if improvement >= .5 and interval[0] > 0 else ('inconclusive' if improvement > 0 else 'no_improvement')
    store.write({'kind':'full_rankings'},b)
    store.write({'kind':'query_logits_probability'},np.stack(all_blocks))
    store.write({'kind':'targets'},targets)
    return {'metrics':metrics,'delta_r1_pp':improvement,'ci95_pp':list(interval),'verdict':verdict,
            'query_count':len(queries),'video_count':len(videos),'bootstrap_seed':713,'bootstrap_samples':10000,
            'checkpoint_selected_on_validation':True,'independent_test_conclusion':False}
