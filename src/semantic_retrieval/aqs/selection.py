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
