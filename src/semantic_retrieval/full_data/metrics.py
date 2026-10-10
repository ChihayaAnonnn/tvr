"""Complete matrix metrics; one-based ranks, canonical identity tie breaking."""
import numpy as np


def _identities(values, *, unique):
    values = list(values)
    if (not values or any(not isinstance(i, str) or not i for i in values)
            or unique and len(values) != len(set(values))):
        raise ValueError('invalid or duplicate retrieval identities')
    return values


def _summary(ranks):
    return {**{f'R{k}': float(np.mean(ranks <= k) * 100) for k in (1, 5, 10)},
            'MdR': float(np.median(ranks)), 'MnR': float(np.mean(ranks))}


def retrieval_metrics(scores, query_video_ids, gallery_ids, *, query_ids):
    positives = _identities(query_video_ids, unique=False)
    gallery = _identities(gallery_ids, unique=True)
    queries = _identities(query_ids, unique=True)
    scores = np.asarray(scores)
    if (scores.shape != (len(queries), len(gallery)) or len(positives) != len(queries)
            or not np.issubdtype(scores.dtype, np.floating) or not np.isfinite(scores).all()
            or set(positives) != set(gallery)):
        raise ValueError('invalid, nonfinite or incomplete retrieval matrix')
    # Sort identities first, then stable score order: independent of input permutations.
    vg = np.array(sorted(range(len(gallery)), key=lambda i: gallery[i]))
    vorder = vg[np.argsort(-scores[:, vg], axis=1, kind='stable')]
    target = np.array([gallery.index(i) for i in positives])
    t2v = np.argmax(vorder == target[:, None], axis=1).astype(np.int64) + 1
    qg = np.array(sorted(range(len(queries)), key=lambda i: queries[i]))
    qorder = qg[np.argsort(-scores[qg, :].T, axis=1, kind='stable')]
    relevant = np.asarray(positives)[qorder] == np.asarray(gallery)[:, None]
    v2t = np.argmax(relevant, axis=1).astype(np.int64) + 1
    return dict(t2v=_summary(t2v), v2t=_summary(v2t), t2v_ranks=t2v, v2t_ranks=v2t)
