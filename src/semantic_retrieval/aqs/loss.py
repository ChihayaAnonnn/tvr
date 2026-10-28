"""T2V auxiliary loss on original temperature-scaled, differentiable logits."""
import torch
import torch.nn.functional as F
from .selection import Selection


def selected_t2v_loss(logits, selection):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or len(logits) < 1
            or logits.shape[0] != logits.shape[1] or not logits.is_floating_point()
            or not torch.isfinite(logits).all() or not isinstance(selection, Selection)):
        raise ValueError('invalid selected-loss logits or selection')
    indices = selection.indices
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
            or indices.ndim != 2 or indices.shape[0] != len(logits) or indices.device != logits.device
            or indices.shape[1] > len(logits)-1 or (indices < 0).any() or (indices >= len(logits)).any()):
        raise ValueError('invalid selected negative indices')
    rows = torch.arange(len(logits), device=logits.device)
    if (indices == rows[:, None]).any():
        raise ValueError('positive cannot be a selected negative')
    if indices.shape[1] > 1 and (indices.sort(dim=1).values.diff(dim=1) == 0).any():
        raise ValueError('repeated selected negative')
    if indices.shape[1] == 0:
        return logits.diagonal().mean()*0.0
    columns = torch.cat((logits.diagonal()[:, None], logits.gather(1, indices)), dim=1)
    return F.cross_entropy(columns, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
