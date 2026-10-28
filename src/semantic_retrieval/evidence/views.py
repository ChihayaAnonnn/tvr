"""Deterministic nested temporal evidence; no pixel content informs actions."""
import torch

from semantic_retrieval.evidence.types import EvidenceAction, ResourceCounts, digest, positive


def build_frame_pool(frame_count: int, pool_size: int):
    positive(frame_count, integer=True); positive(pool_size, integer=True)
    n = min(frame_count, pool_size)
    if n == 1:
        return (0,)
    return tuple(i*(frame_count-1)//(n-1) for i in range(n))


def priority_order(pool):
    if not pool or tuple(sorted(set(pool))) != pool:
        raise ValueError('nonempty sorted unique frame pool required')
    selected = [pool[0]]
    if len(pool) > 1:
        selected.append(pool[-1])
    while len(selected) < len(pool):
        selected.append(min((i for i in pool if i not in selected), key=lambda i: (-min(abs(i-j) for j in selected), i)))
    return tuple(selected)


def view_frame_ids(pool, count):
    positive(count, integer=True)
    return tuple(sorted(priority_order(pool)[:count]))


def coverage_vector(candidate):
    observed = candidate.observed_frame_ids
    f = candidate.frame_count
    if f == 1:
        return torch.tensor([1., 0., 0., 0.])
    boundaries = (0, *observed, f-1)
    gap = max(b-a for a, b in zip(boundaries[:-1], boundaries[1:]))
    return torch.tensor([len(observed)/len(candidate.pool_frame_ids), observed[0]/(f-1), observed[-1]/(f-1), gap/(f-1)])


def propose_actions(state):
    actions = []
    for c in state.candidates:
        for level in state.view_levels:
            new = tuple(i for i in view_frame_ids(c.pool_frame_ids, level) if i not in c.attempted_frame_ids)
            if new:
                break
        else:
            continue
        upper = ResourceCounts(len(new), len(new), sum(len(v.observed_frame_ids) for v in state.candidates)+len(new), state.profile.mc_samples*len(state.candidates))
        cost = sum(a*b for a, b in zip(upper.values(), state.profile.cost_weights))
        identity = dict(state=state.state_sha256, candidate=c.canonical_id, frames=new)
        actions.append(EvidenceAction(digest(identity), state.state_sha256, c.canonical_id, new,
                                      tuple(c.pool_frame_ids.index(i) for i in new), upper, cost))
    return tuple(actions)
