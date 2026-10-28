"""Validation-only probability calibration with explicit deployment identity."""

import math
import re
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationIdentity:
    model_sha256: str
    preprocessing_sha256: str
    gallery_version: str
    candidate_policy_sha256: str
    calibration_split_sha256: str

    def __post_init__(self):
        if not self.gallery_version or not all(
            isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x)
            for x in (
                self.model_sha256,
                self.preprocessing_sha256,
                self.candidate_policy_sha256,
                self.calibration_split_sha256,
            )
        ):
            raise ValueError("invalid calibration identity")


@dataclass(frozen=True)
class CalibrationState:
    identity: CalibrationIdentity
    pair_temperature: float
    pair_bias: float
    presence_temperature: float
    presence_bias: float

    def __post_init__(self):
        if (
            not isinstance(self.identity, CalibrationIdentity)
            or not all(
                math.isfinite(x)
                for x in (self.pair_temperature, self.pair_bias, self.presence_temperature, self.presence_bias)
            )
            or self.pair_temperature <= 0
            or self.presence_temperature <= 0
        ):
            raise ValueError("invalid calibration parameters")

    def pair_probabilities(self, logits):
        return torch.sigmoid(logits / self.pair_temperature + self.pair_bias)

    def presence_probabilities(self, logits):
        return torch.sigmoid(logits / self.presence_temperature + self.presence_bias)


def _fit(logits, labels):
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or not torch.isfinite(logits).all()
        or labels.dtype not in (torch.int32, torch.int64)
        or not ((labels == -1) | (labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("invalid calibration arrays")
    known = labels >= 0
    x = logits[known].detach().cpu().double()
    y = labels[known].detach().cpu().double()
    if not (y == 0).any() or not (y == 1).any():
        raise ValueError("calibration requires both known classes")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_temperature, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / torch.exp(log_temperature) + bias, y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_temperature.exp().detach())
    b = float(bias.detach())
    if not math.isfinite(t) or t <= 0 or not math.isfinite(b):
        raise ValueError("calibration solver failed")
    return t, b


def fit_calibration(pair_logits, pair_labels, presence_logits, presence_labels, *, split, identity):
    if split != "val_calibration":
        raise ValueError("calibration only accepts val_calibration; test is sealed")
    pt, pb = _fit(pair_logits, pair_labels)
    et, eb = _fit(presence_logits, presence_labels)
    return CalibrationState(identity, pt, pb, et, eb)
