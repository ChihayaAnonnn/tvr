"""Pure ranking operations: never discard candidates or silently repair invalid input."""
import numpy as np


def evidence_indices(indices):
    if not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('nonempty nonnegative integer frame indices required')
    unique = sorted(set(indices))
    if list(indices) != sorted(indices):
        raise ValueError('frames must be chronological')
    return [unique[i] for i in np.linspace(0, len(unique)-1, min(4, len(unique))).astype(int)]


def validate_ranking(order):
    a = np.asarray(order)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a) or not np.array_equal(np.sort(a), np.arange(len(a))):
        raise ValueError('ranking must be a complete gallery permutation')
    return a


def rerank(order, probabilities):
    a = validate_ranking(order)
    p = np.asarray(probabilities)
    if len(a) < 10 or p.shape != (10,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('exactly ten finite probabilities required')
    return np.concatenate((a[:10][np.argsort(-p, kind='stable')], a[10:]))


def compare(original, reranked, targets):
    a, b, t = np.asarray(original), np.asarray(reranked), np.asarray(targets)
    if a.ndim != 2 or a.shape != b.shape or t.shape != (len(a),) or t.dtype.kind not in 'iu' or not len(a):
        raise ValueError('invalid paired evaluation shapes')
    for x, y in zip(a, b):
        validate_ranking(x); validate_ranking(y)
        if not np.array_equal(x[10:], y[10:]) or set(x[:10]) != set(y[:10]):
            raise ValueError('reranking changed candidate membership or tail')
    if ((t < 0) | (t >= a.shape[1])).any():
        raise ValueError('target outside gallery')
    ra = np.argmax(a == t[:, None], axis=1)+1
    rb = np.argmax(b == t[:, None], axis=1)+1
    def metrics(r):
        return dict(r1=float(100*np.mean(r <= 1)), r5=float(100*np.mean(r <= 5)),
                    r10=float(100*np.mean(r <= 10)), mrr=float(np.mean(1/r)))
    return dict(baseline=metrics(ra), reranked=metrics(rb),
                corrected=int(np.sum((ra > 1) & (rb == 1))), worsened=int(np.sum((ra == 1) & (rb > 1))))
