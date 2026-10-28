"""Detached, deterministic video selection and side-effect-free representations."""
import torch


def validate_ids(video_ids, batch):
    if (not isinstance(video_ids, torch.Tensor) or video_ids.dtype != torch.long
            or type(batch) is not int or batch < 1 or video_ids.shape != (batch,)
            or (video_ids < 0).any()):
        raise ValueError('invalid canonical video identities')
    if len(video_ids.unique()) != batch:
        raise ValueError('global batch requires unique canonical video identities')


@torch.no_grad()
def pooled_frames(tokens, mask, frame_count):
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or not tokens.is_floating_point() or min(tokens.shape) < 1
            or not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape[:2]
            or mask.device != tokens.device or not ((mask == 0) | (mask == 1)).all()
            or type(frame_count) is not int or not 1 <= frame_count <= tokens.shape[1]):
        raise ValueError('invalid frame pooling input')
    frames = tokens.detach()[:, :frame_count].float()
    valid = mask[:, :frame_count].bool()
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError('all-padding video')
    if not torch.isfinite(frames[valid]).all():
        raise ValueError('nonfinite valid frame')
    selected = torch.where(valid[..., None], frames, torch.zeros_like(frames))
    mean = selected.sum(1) / counts[:, None]
    norms = mean.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-8).any():
        raise ValueError('invalid pooled video representation')
    return mean / norms


from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    indices: torch.Tensor
    selected_ids: torch.Tensor
    difficulty: torch.Tensor
    uniqueness: torch.Tensor


@torch.no_grad()
def select_negatives(scores, video_repr, video_ids, negatives):
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 2
            or scores.shape[0] != scores.shape[1] or not scores.is_floating_point()
            or not torch.isfinite(scores).all() or type(negatives) is not int or negatives < 1):
        raise ValueError('invalid AQS score matrix or negative count')
    batch = len(scores)
    validate_ids(video_ids, batch)
    if (not isinstance(video_repr, torch.Tensor) or video_repr.ndim != 2
            or video_repr.shape[0] != batch or video_repr.shape[1] < 1
            or not video_repr.is_floating_point() or video_repr.device != scores.device
            or video_ids.device != scores.device or not torch.isfinite(video_repr).all()):
        raise ValueError('invalid AQS video representation')
    reps = video_repr.detach().float()
    norms = reps.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-8).any():
        raise ValueError('invalid AQS representation norm')
    if batch == 1:
        index = torch.empty((1, 0), device=scores.device, dtype=torch.long)
        value = torch.empty((1, 0), device=scores.device, dtype=torch.float32)
        return Selection(index, index.clone(), value, value.clone())
    order = torch.argsort(video_ids, stable=True)
    reps = (reps / norms)[order]
    available = video_ids[:, None] != video_ids[order][None, :]
    values = scores.detach().float()[:, order]
    if not torch.isfinite(values).all():
        raise ValueError('AQS scores overflow FP32')
    low = values.masked_fill(~available, torch.inf).min(dim=1, keepdim=True).values
    high = values.masked_fill(~available, -torch.inf).max(dim=1, keepdim=True).values
    span = high-low
    if not torch.isfinite(span).all():
        raise ValueError('AQS score range overflow')
    h = torch.where(span <= 1e-8, torch.ones_like(values), (values-low)/span.clamp_min(1e-8))
    distance = (1-(reps @ reps.T).clamp(-1, 1))/2
    uniqueness = torch.ones_like(h)
    indices, difficulties, diversities = [], [], []
    rows = torch.arange(batch, device=scores.device)
    for _ in range(min(negatives, batch-1)):
        quality = (h*uniqueness).masked_fill(~available, -torch.inf)
        j = quality.argmax(dim=1)
        indices.append(order[j])
        difficulties.append(h[rows, j])
        diversities.append(uniqueness[rows, j])
        available[rows, j] = False
        uniqueness = torch.minimum(uniqueness, distance[:, j].T)
    index = torch.stack(indices, dim=1)
    return Selection(index, video_ids[index], torch.stack(difficulties, dim=1),
                     torch.stack(diversities, dim=1))


def summarize_selection(selection):
    batch, count = selection.indices.shape
    return dict(mean_negatives=float(count),
                mean_difficulty=float(selection.difficulty.mean()) if count else 0.0,
                mean_uniqueness=float(selection.uniqueness.mean()) if count else 0.0,
                empty_rows=float(batch if count == 0 else 0))
