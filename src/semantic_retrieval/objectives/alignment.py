"""Detached, masked balanced dustbin transport in the log domain."""

import torch


def _transport(similarity, config):
    n, m = similarity.shape
    if not n or not m or not torch.isfinite(similarity).all() or (similarity.abs() > 1 + 1e-6).any():
        raise ValueError("valid cosine similarities required")
    s = similarity.detach().double().clamp(-1, 1)
    cost = s.new_full((n + 1, m + 1), config.dustbin_cost)
    cost[:n, :m] = 1 - s
    cost[-1, -1] = 0
    a = s.new_ones(n + 1)
    a[-1] = m
    a /= n + m
    b = s.new_ones(m + 1)
    b[-1] = n
    b /= n + m
    kernel = -cost / config.epsilon
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    for _ in range(config.max_iterations):
        u = a.log() - torch.logsumexp(kernel + v[None, :], dim=1)
        v = b.log() - torch.logsumexp(kernel + u[:, None], dim=0)
        plan = torch.exp(kernel + u[:, None] + v[None, :])
        error = torch.maximum((plan.sum(1) - a).abs().max(), (plan.sum(0) - b).abs().max())
        if error <= config.tolerance:
            return plan
    raise ValueError("dustbin transport did not converge to required marginals")


def dustbin_ot_target(similarity, text_mask, frame_mask, *, config):
    if (
        similarity.ndim != 3
        or text_mask.dtype != torch.bool
        or frame_mask.dtype != torch.bool
        or text_mask.shape != similarity.shape[:2]
        or frame_mask.shape != (similarity.shape[0], similarity.shape[2])
        or not text_mask.any(-1).all()
        or not frame_mask.any(-1).all()
    ):
        raise ValueError("invalid alignment masks/shapes")
    out = torch.zeros(
        similarity.shape[0], similarity.shape[1], similarity.shape[2] + 1, dtype=torch.float32, device=similarity.device
    )
    with torch.no_grad(), torch.autocast(device_type=similarity.device.type, enabled=False):
        for row in range(len(similarity)):
            ti = text_mask[row].nonzero().flatten()
            fi = frame_mask[row].nonzero().flatten()
            plan = _transport(similarity[row][ti][:, fi], config)
            distribution = plan[:-1] / plan[:-1].sum(-1, keepdim=True)
            out[row, ti[:, None], fi[None, :]] = distribution[:, :-1].float()
            out[row, ti, -1] = distribution[:, -1].float()
    return out.detach()


def alignment_loss(attention, target, text_mask):
    if (
        attention.shape != target.shape
        or attention.ndim != 3
        or text_mask.shape != attention.shape[:2]
        or text_mask.dtype != torch.bool
        or not torch.isfinite(attention).all()
        or not torch.isfinite(target).all()
        or (attention < 0).any()
        or (target < 0).any()
        or target.requires_grad
    ):
        raise ValueError("invalid student/teacher distributions")
    if not text_mask.any():
        return attention.sum() * 0
    if not torch.allclose(
        target.sum(-1)[text_mask], torch.ones_like(target.sum(-1)[text_mask]), atol=1e-5
    ) or not torch.allclose(attention.sum(-1)[text_mask], torch.ones_like(attention.sum(-1)[text_mask]), atol=1e-5):
        raise ValueError("valid token distributions must sum to one")
    valid_target = target[text_mask]
    student = attention[text_mask].clamp_min(1e-12)
    return (torch.xlogy(valid_target, valid_target) - valid_target * student.log()).sum(-1).mean()
