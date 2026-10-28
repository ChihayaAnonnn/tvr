"""Recoverable training RNG and identity-addressed evaluation integration."""
import hashlib
import math
from dataclasses import asdict

import torch

from semantic_retrieval.evidence.types import digest, require_digest


class SamplingState:
    def __init__(self, config, *, mode, model_sha256):
        if mode not in {'train', 'eval'}:
            raise ValueError('sampling mode must be train/eval')
        require_digest(model_sha256)
        self.config = config
        self.mode = mode
        self.model_sha256 = model_sha256
        self.generator = torch.Generator(device='cpu').manual_seed(config.seed)
        self.identity = digest(dict(config=asdict(config), mode=mode, model=model_sha256))

    def noise(self, pair_ids, *, device, dtype):
        if len(set(pair_ids)) != len(pair_ids) or any(len(p) != 2 or not all(isinstance(x, str) and x for x in p) for p in pair_ids):
            raise ValueError('unique query/canonical pair identities required')
        m = self.config.train_samples if self.mode == 'train' else self.config.eval_samples
        if self.mode == 'train':
            return torch.randn(m, len(pair_ids), generator=self.generator, dtype=torch.float32).to(device=device, dtype=dtype)
        rows = []
        for sample in range(m):
            row = []
            for pair in pair_ids:
                h = hashlib.sha256((self.identity+'\0'+digest(dict(pair=pair, sample=sample))).encode()).digest()
                # Open-interval uniforms avoid log(0), portable across device and candidate order.
                u = (int.from_bytes(h[:8], 'big')+.5)/2**64
                v = (int.from_bytes(h[8:16], 'big')+.5)/2**64
                u = min(max(u, 2**-53), 1-2**-53)
                row.append(math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v))
            rows.append(row)
        return torch.tensor(rows, device=device, dtype=dtype).reshape(m, len(pair_ids))

    def state_dict(self):
        return dict(identity=self.identity, generator=self.generator.get_state().clone())

    def load_state_dict(self, value):
        if set(value) != {'identity', 'generator'} or value['identity'] != self.identity:
            raise ValueError('MC sampling identity mismatch')
        self.generator.set_state(value['generator'].cpu())
