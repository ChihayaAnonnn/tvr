"""Mask-aware diagonal-Gaussian embeddings for RSPR.

The module keeps every distribution parameter in log-variance semantics:
``standard_deviation = exp(0.5 * logvar)``.  Probability calculations are
performed in FP32 so they remain stable when upstream CLIP/DSA features use
mixed precision.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DistributionOutput:
    """Named output of :class:`ReparameterizedDistributionHead`."""

    center: torch.Tensor
    dispersion: torch.Tensor
    attention_entropy: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    samples: torch.Tensor
    anchor_kl: torch.Tensor


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _fp32_context(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _require_fp32_parameters(module: nn.Module) -> None:
    non_fp32 = [name for name, parameter in module.named_parameters() if parameter.dtype != torch.float32]
    if non_fp32:
        raise RuntimeError(
            f"{module.__class__.__name__} parameters must remain in FP32; non-FP32 parameters={non_fp32}"
        )


class MaskedStatPool(nn.Module):
    """Learn a masked center, per-dimension dispersion, and attention entropy."""

    def __init__(self, dim: int, hidden_dim: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.dim = _positive_integer(dim, "dim")
        self.hidden_dim = _positive_integer(self.dim if hidden_dim is None else hidden_dim, "hidden_dim")
        self.eps = _positive_finite(eps, "eps")

        self.features = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.score = nn.Linear(self.hidden_dim, 1, bias=False)

    def _validate_inputs(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, sequence, dim], " f"got shape={tuple(tokens.shape)}")
        if tokens.size(-1) != self.dim:
            raise ValueError(f"token dimension must be {self.dim}, got {tokens.size(-1)}")
        if mask.ndim != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(
                "mask must have shape [batch, sequence] matching tokens, "
                f"got mask={tuple(mask.shape)} tokens={tuple(tokens.shape)}"
            )

        valid = mask.to(device=tokens.device, dtype=torch.bool)
        invalid_rows = (~valid.any(dim=1)).nonzero(as_tuple=False).reshape(-1)
        if invalid_rows.numel():
            raise ValueError(
                "every sample must contain at least one valid position; "
                f"invalid batch indices={invalid_rows.tolist()}"
            )
        return valid

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_fp32_parameters(self)
        valid = self._validate_inputs(tokens, mask)

        with _fp32_context(tokens):
            work = tokens.float().masked_fill(~valid.unsqueeze(-1), 0.0)
            logits = self.score(self.features(work)).squeeze(-1)
            logits = logits.masked_fill(~valid, float("-inf"))
            attention = torch.softmax(logits, dim=-1)

            center = torch.einsum("bn,bnd->bd", attention, work)
            dispersion = torch.einsum(
                "bn,bnd->bd",
                attention,
                (work - center.unsqueeze(1)).square(),
            )

            valid_count = valid.sum(dim=1, keepdim=True)
            raw_entropy = -(attention * attention.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
            entropy = torch.where(
                valid_count > 1,
                raw_entropy / valid_count.to(dtype=torch.float32).log().clamp_min(self.eps),
                torch.zeros_like(raw_entropy),
            )

        return center, dispersion, entropy


def antithetic_standard_normal(
    batch_size: int,
    sample_count: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw interleaved ``[epsilon, -epsilon]`` standard-normal pairs."""

    batch_size = _positive_integer(batch_size, "batch_size")
    dim = _positive_integer(dim, "dim")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or sample_count % 2 != 0
    ):
        raise ValueError("sample_count must be a positive even integer")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("dtype must be a floating-point torch dtype")

    base = torch.randn(
        batch_size,
        sample_count // 2,
        dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return torch.stack((base, -base), dim=2).reshape(batch_size, sample_count, dim)


class ReparameterizedDistributionHead(nn.Module):
    """Parameterize and sample a mask-aware diagonal Gaussian distribution."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        logvar_min: float = -8.0,
        logvar_max: float = 2.0,
        prior_std: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = _positive_integer(dim, "dim")
        self.hidden_dim = _positive_integer(self.dim if hidden_dim is None else hidden_dim, "hidden_dim")
        self.eps = _positive_finite(eps, "eps")
        self.prior_std = _positive_finite(prior_std, "prior_std")
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        if (
            not math.isfinite(self.logvar_min)
            or not math.isfinite(self.logvar_max)
            or self.logvar_min >= self.logvar_max
        ):
            raise ValueError("logvar_min must be finite and smaller than logvar_max")

        prior_logvar = math.log(self.prior_std**2)
        if not self.logvar_min <= prior_logvar <= self.logvar_max:
            raise ValueError(
                "prior_std implies a log-variance outside the configured bounds: "
                f"logvar={prior_logvar:.6g} bounds=[{self.logvar_min}, {self.logvar_max}]"
            )

        self.pool = MaskedStatPool(dim=self.dim, hidden_dim=self.hidden_dim, eps=self.eps)
        self.mean_norm = nn.LayerNorm(2 * self.dim)
        self.mean_head = nn.Sequential(
            nn.Linear(2 * self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )
        self.logvar_norm = nn.LayerNorm(2 * self.dim + 1)
        self.logvar_head = nn.Sequential(
            nn.Linear(2 * self.dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        nn.init.zeros_(self.logvar_head[-1].weight)
        nn.init.constant_(self.logvar_head[-1].bias, prior_logvar)

    @staticmethod
    def _validate_sample_count(sample_count: int, mean_only: bool) -> None:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise ValueError("sample_count must be an integer")
        if mean_only:
            if sample_count != 1:
                raise ValueError("mean-only sampling requires sample_count=1")
            return
        if sample_count <= 0 or sample_count % 2 != 0:
            raise ValueError("sample_count must be a positive even integer")

    def _prepare_noise(
        self,
        noise: torch.Tensor | None,
        *,
        batch_size: int,
        sample_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected_shape = (batch_size, sample_count, self.dim)
        if noise is None:
            return antithetic_standard_normal(
                batch_size,
                sample_count,
                self.dim,
                device=device,
                dtype=torch.float32,
            )
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
        if not noise.is_floating_point():
            raise ValueError("noise must use a floating-point dtype")
        return noise.to(device=device, dtype=torch.float32)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        sample_count: int,
        noise: torch.Tensor | None = None,
        mean_only: bool = False,
        detach_samples: bool = False,
    ) -> DistributionOutput:
        _require_fp32_parameters(self)
        self._validate_sample_count(sample_count, mean_only)
        center, dispersion, entropy = self.pool(tokens, mask)

        with _fp32_context(tokens):
            center = center.float()
            dispersion = dispersion.float()
            entropy = entropy.float()

            mean_features = self.mean_norm(torch.cat((center, dispersion), dim=-1))
            mean = center + self.mean_head(mean_features)

            logvar_features = self.logvar_norm(torch.cat((center, dispersion, entropy), dim=-1))
            logvar = self.logvar_head(logvar_features).float()
            logvar = logvar.clamp(self.logvar_min, self.logvar_max)

            variance = logvar.exp()
            prior_variance = self.prior_std**2
            anchor_kl = (
                0.5
                * (
                    (variance + (mean.float() - center.detach()).square()) / prior_variance
                    - 1.0
                    + math.log(prior_variance)
                    - logvar
                ).mean()
            )

            if mean_only:
                if noise is not None:
                    expected_shape = (tokens.size(0), 1, self.dim)
                    if tuple(noise.shape) != expected_shape:
                        raise ValueError("noise must have shape " f"{expected_shape}, got {tuple(noise.shape)}")
                samples = F.normalize(mean.float(), dim=-1, eps=self.eps).unsqueeze(1)
            else:
                prepared_noise = self._prepare_noise(
                    noise,
                    batch_size=tokens.size(0),
                    sample_count=sample_count,
                    device=mean.device,
                )
                standard_deviation = torch.exp(0.5 * logvar)
                raw_samples = mean.float().unsqueeze(1) + (standard_deviation.unsqueeze(1) * prepared_noise)
                samples = F.normalize(raw_samples, dim=-1, eps=self.eps)

            if detach_samples:
                samples = samples.detach()

        return DistributionOutput(
            center=center,
            dispersion=dispersion,
            attention_entropy=entropy,
            mean=mean,
            logvar=logvar,
            samples=samples,
            anchor_kl=anchor_kl,
        )
