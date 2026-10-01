"""Independent normalized coarse projections."""

import torch
from torch import nn


class CoarseProjector(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.text = nn.Linear(dim, output_dim, bias=False)
        self.video = nn.Linear(dim, output_dim, bias=False)

    def _project(self, x, layer):
        if x.ndim != 2 or x.shape[-1] != layer.in_features or not torch.isfinite(x).all():
            raise ValueError("invalid pooled features")
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = layer(x.float())
            norm = y.norm(dim=-1, keepdim=True)
            if (norm <= 1e-12).any():
                raise ValueError("zero norm coarse representation")
            return y / norm

    def project_text(self, x):
        return self._project(x, self.text)

    def project_video(self, x):
        return self._project(x, self.video)

    def forward(self, text_pool, video_pool):
        return self.project_text(text_pool), self.project_video(video_pool)
