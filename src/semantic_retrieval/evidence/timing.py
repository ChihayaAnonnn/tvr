"""Synchronized wall-clock measurements for inference operation accounting."""
from contextlib import contextmanager
import time
import torch


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def measure(totals,name,device):
    if totals is None:
        yield
        return
    synchronize(device);started = time.monotonic()
    try:
        yield
    finally:
        synchronize(device)
        totals[name] = totals.get(name,0.)+time.monotonic()-started
