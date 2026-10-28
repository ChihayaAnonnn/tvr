"""Read-only, seeded two-device environment witness; no training or downloads."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch

spec = json.loads((Path(__file__).resolve().parents[1]/'experiments/uatvr-aqs-training-v1/env-spec.json').read_text())
for package, version in spec['packages'].items():
    assert importlib.metadata.version(package) == version, package
assert torch.cuda.device_count() == 2
for device in range(2):
    generator = torch.Generator().manual_seed(713)
    x = torch.randn(64, 64, generator=generator)
    actual = (x.to(device) @ x.to(device)).cpu()
    assert torch.allclose(actual, x @ x, atol=1e-4, rtol=1e-4)
    print('WITNESS', device, torch.cuda.get_device_name(device), hashlib.sha256(actual.numpy().tobytes()).hexdigest())
