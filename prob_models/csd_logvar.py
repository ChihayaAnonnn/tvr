from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from prob_models.retrieval_risk import (
    _pairwise_uncertainty_ranking_loss,
    _robust_rank_normalize,
    retrieval_hardness,
)


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


def gaussian_vib_loss(mean, logvar):
    mean = _validate_matrix("mean", mean).detach().float()
    logvar = _validate_matrix("logvar", logvar).float()
    if mean.shape != logvar.shape:
        raise ValueError("mean and logvar must have equal shapes")
    return (-0.5 * (1.0 + logvar - mean.square() - logvar.exp())).mean()


def _grounding_terms(uncertainty, scores, positive_mask, temperature):
    hardness = retrieval_hardness(scores.detach(), positive_mask, temperature)
    fit = F.smooth_l1_loss(uncertainty, _robust_rank_normalize(hardness))
    rank = _pairwise_uncertainty_ranking_loss(uncertainty, hardness)
    return fit + rank, fit, rank, hardness


def csd_logvar_objective(sidecar, text_mu, video_mu, wti_logits, positive_mask,
                         w_csd=1.0, w_grounding=0.05, w_vib=1e-4,
                         hardness_temperature=1.0):
    if min(w_csd, w_grounding, w_vib) < 0:
        raise ValueError("loss weights must be non-negative")
    if hardness_temperature <= 0:
        raise ValueError("hardness_temperature must be positive")
    logits = _validate_matrix("wti_logits", wti_logits).detach().float()
    mask = torch.as_tensor(positive_mask, device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape or not mask.any(1).all() or not mask.any(0).all():
        raise ValueError("every text and video requires at least one positive")
    if mask.all(1).any() or mask.all(0).any():
        raise ValueError("every text and video requires at least one negative")
    output = sidecar(text_mu.detach(), video_mu.detach())
    target = mask.to(output["match_logits"].dtype)
    elements = F.binary_cross_entropy_with_logits(output["match_logits"], target, reduction="none")
    csd, csd_positive, csd_negative = elements.mean(), elements[mask].mean(), elements[~mask].mean()
    tg, tf, tr, text_hardness = _grounding_terms(output["u_text"], logits, mask, hardness_temperature)
    vg, vf, vr, video_hardness = _grounding_terms(output["u_video"], logits.T, mask.T, hardness_temperature)
    grounding, grounding_fit, grounding_rank = 0.5 * (tg + vg), 0.5 * (tf + vf), 0.5 * (tr + vr)
    vib = 0.5 * (gaussian_vib_loss(text_mu, output["text_logvar"]) + gaussian_vib_loss(video_mu, output["video_logvar"]))
    total = w_csd * csd + w_grounding * grounding + w_vib * vib
    diagnostics = {
        **{name: value.detach() for name, value in output.items()},
        "csd": csd.detach(), "csd_positive": csd_positive.detach(),
        "csd_negative": csd_negative.detach(), "grounding": grounding.detach(),
        "grounding_fit": grounding_fit.detach(), "grounding_rank": grounding_rank.detach(),
        "vib": vib.detach(), "text_hardness": text_hardness, "video_hardness": video_hardness,
    }
    for modality in ("text", "video"):
        values = output[f"{modality}_logvar"]
        diagnostics[f"{modality}_clamp_low"] = values.eq(sidecar.logvar_min).float().mean().detach()
        diagnostics[f"{modality}_clamp_high"] = values.eq(sidecar.logvar_max).float().mean().detach()
    return total, diagnostics
