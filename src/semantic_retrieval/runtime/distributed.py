"""Equal-shape rank-order gather with standard autograd collective semantics."""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def configure_fp32():
    """Establish the precision required by the protected runtime before binding."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _distributed(values):
    if not dist.is_available() or not dist.is_initialized():
        return False
    specifications = [None] * dist.get_world_size()
    dist.all_gather_object(specifications, (tuple(values.shape), str(values.dtype), values.device.type))
    if values.ndim == 0 or any(item != specifications[0] for item in specifications):
        raise ValueError("gather requires equal non-scalar shape, dtype and device type on every rank")
    return True


def gather_with_grad(values):
    if not _distributed(values):
        return values
    return torch.cat(all_gather(values), dim=0)


def gather_metadata(values):
    detached = values.detach()
    if not _distributed(detached):
        return detached
    # NCCL requires contiguous send and receive buffers; encoder outputs may
    # be transposed views. Preserve logical values without mutating the caller.
    detached = detached.contiguous()
    gathered = [torch.empty_like(detached) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, detached)
    return torch.cat(gathered, dim=0)
