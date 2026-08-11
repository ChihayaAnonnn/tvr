from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _validate_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 2-D tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def closed_form_sampled_distance(text_mu, video_mu, text_logvar, video_logvar):
    text_mu = _validate_matrix("text_mu", text_mu)
    video_mu = _validate_matrix("video_mu", video_mu)
    text_logvar = _validate_matrix("text_logvar", text_logvar)
    video_logvar = _validate_matrix("video_logvar", video_logvar)
    if text_mu.shape != text_logvar.shape or video_mu.shape != video_logvar.shape:
        raise ValueError("each mean and logvar pair must have equal shapes")
    if text_mu.size(1) != video_mu.size(1):
        raise ValueError("text and video embedding dimensions must match")
    mean_distance = torch.cdist(text_mu.float(), video_mu.float()).square()
    variance_distance = (
        text_logvar.float().exp().sum(-1, keepdim=True)
        + video_logvar.float().exp().sum(-1).unsqueeze(0)
    )
    return mean_distance + variance_distance


class CSDLogVarSidecar(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=128, logvar_min=-12.0,
                 logvar_max=4.0, init_logvar=-8.0, initial_scale=5.0,
                 initial_shift=5.0):
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        if not logvar_min < logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        if not logvar_min <= init_logvar <= logvar_max:
            raise ValueError("init_logvar must lie inside the clamp bounds")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        self.logvar_min, self.logvar_max = float(logvar_min), float(logvar_max)
        self.init_logvar = float(init_logvar)
        self.initial_scale, self.initial_shift = float(initial_scale), float(initial_shift)
        self.text_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim)
        )
        self.video_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim)
        )
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale))))
        self.shift = nn.Parameter(torch.tensor(initial_shift))
        self.reset_output_parameters()

    def reset_output_parameters(self):
        for head in (self.text_head, self.video_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, self.init_logvar)
        with torch.no_grad():
            self.raw_scale.fill_(math.log(math.expm1(self.initial_scale)))
            self.shift.fill_(self.initial_shift)

    def forward(self, text_mu, video_mu):
        text_mu = _validate_matrix("text_mu", text_mu)
        video_mu = _validate_matrix("video_mu", video_mu)
        text_logvar = self.text_head(text_mu.detach().float()).clamp(
            self.logvar_min, self.logvar_max
        )
        video_logvar = self.video_head(video_mu.detach().float()).clamp(
            self.logvar_min, self.logvar_max
        )
        mean_distance = torch.cdist(text_mu.detach().float(), video_mu.detach().float()).square()
        variance_distance = (
            text_logvar.exp().sum(-1, keepdim=True)
            + video_logvar.exp().sum(-1).unsqueeze(0)
        )
        distance = mean_distance + variance_distance
        scale = F.softplus(self.raw_scale)
        return {
            "text_logvar": text_logvar, "video_logvar": video_logvar,
            "u_text": text_logvar.exp().mean(-1), "u_video": video_logvar.exp().mean(-1),
            "distance": distance, "mean_distance": mean_distance,
            "variance_distance": variance_distance,
            "match_logits": -scale * distance + self.shift,
            "scale": scale, "shift": self.shift,
        }
