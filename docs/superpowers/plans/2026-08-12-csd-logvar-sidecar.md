# Independent CSD Log-Variance Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately reported text/video Gaussian log-variance signal, trained by PCME++ closed-form distance and detached bidirectional retrieval hardness, without changing WTI retrieval or the frozen A1 risk sidecar.

**Architecture:** A symmetric `CSDLogVarSidecar` consumes detached normalized June text/video means and emits explicit diagonal `log(sigma^2)`. Its loss combines multi-positive CSD matching BCE, directionally mapped hardness grounding, and VIB. A2-only training freezes every existing tensor; paired evaluation collects one uncertainty vector per text and unique video, audits invariance, and selects one bidirectional checkpoint.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy, pytest, SHA-256, existing UATVR/MSRVTT DDP and trusted-v1 pipeline.

## Global Constraints

- Initialize every A2 run from `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`.
- Preserve raw WTI matrices, T2V/V2T retrieval metrics, A1 failure-risk arrays, and A1 vacuity arrays exactly in paired FP32 evaluation.
- Train only `csd_logvar_sidecar.text_head.*`, `csd_logvar_sidecar.video_head.*`, `csd_logvar_sidecar.raw_scale`, and `csd_logvar_sidecar.shift`.
- Keep CLIP, WTI, SAP, legacy probabilistic modules, and `retrieval_risk_sidecar.*` frozen and in evaluation mode.
- Use explicit `log(sigma^2)` semantics; never pass A2 tensors to legacy `sample_gaussian_tensors`.
- Use `Linear(512, 128) -> SiLU -> Linear(128, 512)`, final weight `0`, final bias `-8`, and clamp bounds `[-12, 4]`.
- Use `d_ij = ||mu_text_i - mu_video_j||^2 + sum(exp(text_logvar_i)) + sum(exp(video_logvar_j))` and `match_logit_ij = -softplus(raw_scale) * d_ij + shift`.
- Initialize the effective scale and shift to `5.0`; initialize `raw_scale` with inverse softplus.
- Construct positives from DDP-gathered `group_ids`; an identity target is forbidden for training and multi-caption evaluation.
- Use weights `w_csd=1.0`, `w_csd_grounding=0.05`, and `w_csd_vib=1e-4` for A2-CG.
- Map `u_text` to T2V hardness and `u_video` to V2T hardness.
- Do not rerank with CSD and do not feed A2 uncertainty into the A1 sidecar.
- Keep all new behavior opt-in so commands without `--use_csd_logvar_sidecar` are unchanged.
- Reject a checkpoint on non-finite values, any paired-invariance failure, or a clamp rate greater than or equal to `0.05`.
- Select by maximum `min(T2V rho, V2T rho)`, then mean AUROC, then the earlier epoch.
- Do not commit logs, checkpoints, generated data, or experiment clones.

## File Map

- `prob_models/csd_logvar.py`: explicit log-variance heads, CSD distance, VIB, and combined objective.
- `prob_models/csd_logvar_protocol.py`: immutable-array hashing, state audit, acceptance, and selection.
- `prob_models/retrieval_risk_metrics.py`: rank/failure metrics plus A2 distribution and chunked CSD summaries.
- `modules/modeling_mulit.py`: frozen June means and the A2-only early-return path.
- `main_task_retrieval.py`: CLI, freeze/mode control, evaluation collection, paired registration, and final test.
- `experiment_tracking.py`: opt-in A2 manifest contract.
- `run_train_msrvtt_csd_logvar_bg.sh`: registered CG/C/G launcher.
- `tests/test_csd_logvar*.py`: mathematical, integration, protocol, and launcher verification.

---

### Task 1: Explicit log-variance heads and closed-form distance

**Files:**
- Create: `prob_models/csd_logvar.py`
- Create: `tests/test_csd_logvar.py`

**Interfaces:**
- Produces `CSDLogVarSidecar(embed_dim=512, hidden_dim=128, logvar_min=-12.0, logvar_max=4.0, init_logvar=-8.0, initial_scale=5.0, initial_shift=5.0)`.
- Produces `closed_form_sampled_distance(text_mu, video_mu, text_logvar, video_logvar) -> Tensor[num_text, num_video]`.
- `forward(text_mu, video_mu)` returns `text_logvar`, `video_logvar`, `u_text`, `u_video`, `distance`, `mean_distance`, `variance_distance`, `match_logits`, `scale`, and `shift`.

- [ ] **Step 1: Write failing construction and semantics tests**

```python
import math

import pytest
import torch

from prob_models.csd_logvar import CSDLogVarSidecar, closed_form_sampled_distance


def test_sidecar_has_registered_shape_and_initialization():
    sidecar = CSDLogVarSidecar()
    assert sum(parameter.numel() for parameter in sidecar.parameters()) == 263_426
    assert torch.count_nonzero(sidecar.text_head[-1].weight) == 0
    assert torch.count_nonzero(sidecar.video_head[-1].weight) == 0
    torch.testing.assert_close(sidecar.text_head[-1].bias, torch.full((512,), -8.0))
    torch.testing.assert_close(torch.nn.functional.softplus(sidecar.raw_scale), torch.tensor(5.0))
    assert float(sidecar.shift) == 5.0


def test_closed_form_distance_uses_log_variance_not_log_standard_deviation():
    text_mu = torch.tensor([[0.0, 1.0]])
    video_mu = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    text_logvar = torch.log(torch.tensor([[0.25, 1.0]]))
    video_logvar = torch.log(torch.tensor([[4.0, 0.25], [1.0, 1.0]]))
    actual = closed_form_sampled_distance(text_mu, video_mu, text_logvar, video_logvar)
    torch.testing.assert_close(actual, torch.tensor([[9.5, 7.25]]))


def test_forward_clamps_logvar_and_reports_components():
    sidecar = CSDLogVarSidecar(embed_dim=2, hidden_dim=3)
    text_mu = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
    video_mu = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
    output = sidecar(text_mu, video_mu)
    assert output["text_logvar"].shape == (2, 2)
    assert output["video_logvar"].shape == (3, 2)
    torch.testing.assert_close(output["distance"], output["mean_distance"] + output["variance_distance"])
    torch.testing.assert_close(output["u_text"], output["text_logvar"].exp().mean(dim=-1))


@pytest.mark.parametrize("kwargs", [
    {"embed_dim": 0}, {"hidden_dim": 0},
    {"logvar_min": 4.0, "logvar_max": -12.0},
    {"init_logvar": -13.0}, {"initial_scale": 0.0},
])
def test_sidecar_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CSDLogVarSidecar(**kwargs)
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar.py
```

Expected: collection fails because `prob_models.csd_logvar` does not exist.

- [ ] **Step 3: Implement validation, exact distance, and the sidecar**

```python
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
    variance_distance = text_logvar.float().exp().sum(-1, keepdim=True) + video_logvar.float().exp().sum(-1).unsqueeze(0)
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
        self.text_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim))
        self.video_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embed_dim))
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
        text_logvar = self.text_head(text_mu.detach().float()).clamp(self.logvar_min, self.logvar_max)
        video_logvar = self.video_head(video_mu.detach().float()).clamp(self.logvar_min, self.logvar_max)
        mean_distance = torch.cdist(text_mu.detach().float(), video_mu.detach().float()).square()
        variance_distance = text_logvar.exp().sum(-1, keepdim=True) + video_logvar.exp().sum(-1).unsqueeze(0)
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
```

- [ ] **Step 4: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar.py
git add prob_models/csd_logvar.py tests/test_csd_logvar.py
git commit -m "feat: add independent CSD logvar heads"
```

---

### Task 2: Multi-positive CSD, grounding, and VIB objective

**Files:**
- Modify: `prob_models/csd_logvar.py`
- Create: `tests/test_csd_logvar_loss.py`

**Interfaces:**
- Produces `gaussian_vib_loss(mean, logvar) -> scalar Tensor`.
- Produces `csd_logvar_objective(sidecar, text_mu, video_mu, wti_logits, positive_mask, w_csd=1.0, w_grounding=0.05, w_vib=1e-4, hardness_temperature=1.0)`.
- Returns a differentiable total and detached loss/distance/clamp diagnostics.

- [ ] **Step 1: Write failing objective and detach tests**

```python
import pytest
import torch

from prob_models.csd_logvar import CSDLogVarSidecar, csd_logvar_objective
from prob_models.retrieval_risk import positive_mask_from_group_ids


def _fixture():
    torch.manual_seed(3)
    text_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    video_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    logits = torch.tensor([[4., 3., 0.], [3., 4., 0.], [0., 0., 5.]], requires_grad=True)
    mask = positive_mask_from_group_ids(torch.tensor([7, 7, 9]))
    return text_mu, video_mu, logits, mask


def test_objective_accepts_multi_positives_and_detaches_inputs():
    text_mu, video_mu, logits, mask = _fixture()
    sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    loss, output = csd_logvar_objective(sidecar, text_mu, video_mu, logits, mask)
    loss.backward()
    assert text_mu.grad is None and video_mu.grad is None and logits.grad is None
    assert sum(parameter.grad.abs().sum() for parameter in sidecar.parameters()) > 0
    assert output["text_hardness"].shape == (3,)
    assert output["video_hardness"].shape == (3,)


def test_g_only_removes_scale_and_shift_gradients():
    text_mu, video_mu, logits, mask = _fixture()
    sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    loss, _ = csd_logvar_objective(sidecar, text_mu, video_mu, logits, mask,
                                   w_csd=0.0, w_grounding=0.05, w_vib=1e-4)
    loss.backward()
    assert sidecar.raw_scale.grad is None or torch.count_nonzero(sidecar.raw_scale.grad) == 0
    assert sidecar.shift.grad is None or torch.count_nonzero(sidecar.shift.grad) == 0


def test_objective_rejects_missing_positives_and_negative_weights():
    means = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
    sidecar = CSDLogVarSidecar(embed_dim=2, hidden_dim=2)
    with pytest.raises(ValueError, match="positive"):
        csd_logvar_objective(sidecar, means, means, torch.eye(3), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="weights"):
        csd_logvar_objective(sidecar, means, means, torch.eye(3), torch.eye(3, dtype=torch.bool), w_csd=-1)
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_loss.py
```

Expected: import fails for `csd_logvar_objective`.

- [ ] **Step 3: Implement VIB and directional grounding using the existing hardness**

```python
from prob_models.retrieval_risk import (
    _pairwise_uncertainty_ranking_loss,
    _robust_rank_normalize,
    retrieval_hardness,
)


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
```

- [ ] **Step 4: Implement the weighted objective**

```python
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
        **output, "csd": csd.detach(), "csd_positive": csd_positive.detach(),
        "csd_negative": csd_negative.detach(), "grounding": grounding.detach(),
        "grounding_fit": grounding_fit.detach(), "grounding_rank": grounding_rank.detach(),
        "vib": vib.detach(), "text_hardness": text_hardness, "video_hardness": video_hardness,
    }
    for modality in ("text", "video"):
        values = output[f"{modality}_logvar"]
        diagnostics[f"{modality}_clamp_low"] = values.eq(sidecar.logvar_min).float().mean().detach()
        diagnostics[f"{modality}_clamp_high"] = values.eq(sidecar.logvar_max).float().mean().detach()
    return total, diagnostics
```

- [ ] **Step 5: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar.py tests/test_csd_logvar_loss.py
git add prob_models/csd_logvar.py tests/test_csd_logvar_loss.py
git commit -m "feat: add CSD and hardness logvar objective"
```

---

### Task 3: Opt-in CLI and experiment manifest contract

**Files:**
- Modify: `main_task_retrieval.py` in `get_args()`
- Modify: `experiment_tracking.py` in `build_experiment_manifest()`
- Modify: `tests/test_experiment_tracking.py`
- Create: `tests/test_csd_logvar_config.py`

**Interfaces:**
- Adds `--use_csd_logvar_sidecar`, `--csd_logvar_sidecar_only`, `--w_csd`, `--w_csd_grounding`, `--w_csd_vib`, `--csd_logvar_hidden_dim`, `--csd_logvar_min`, `--csd_logvar_max`, `--csd_logvar_init`, `--csd_initial_scale`, `--csd_initial_shift`, `--csd_hardness_temperature`, and `--csd_bootstrap_samples`.
- Adds manifest section `csd_logvar` only when the feature is enabled.

- [ ] **Step 1: Write failing CLI defaults and validation tests**

```python
import sys

import pytest

from main_task_retrieval import get_args


def _parse(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", [
        "main_task_retrieval.py", "--do_train", "--expand_msrvtt_sentences",
        "--output_dir", "/tmp/csd-logvar-config-test", *extra,
    ])
    return get_args()


def test_old_cli_keeps_csd_logvar_disabled(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.use_csd_logvar_sidecar is False
    assert args.csd_logvar_sidecar_only is False
    assert (args.w_csd, args.w_csd_grounding, args.w_csd_vib) == (1.0, 0.05, 1e-4)
    assert (args.csd_logvar_min, args.csd_logvar_max, args.csd_logvar_init) == (-12.0, 4.0, -8.0)


@pytest.mark.parametrize("extra", [
    ["--csd_logvar_sidecar_only"],
    ["--use_csd_logvar_sidecar", "--csd_logvar_sidecar_only", "--use_uatvr_full"],
    ["--use_csd_logvar_sidecar", "--use_uatvr_full", "--w_csd", "-1"],
    ["--use_csd_logvar_sidecar", "--use_uatvr_full", "--csd_logvar_min", "4", "--csd_logvar_max", "-12"],
])
def test_invalid_csd_cli_contracts_fail(monkeypatch, extra):
    with pytest.raises(ValueError):
        _parse(monkeypatch, extra)
```

The second case lacks `--use_retrieval_risk_sidecar`; A2-only mode requires A1 to be instantiated for paired-output checks.

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_config.py
```

Expected: argparse rejects the new flags.

- [ ] **Step 3: Add exact arguments and validation**

```python
parser.add_argument("--use_csd_logvar_sidecar", action="store_true")
parser.add_argument("--csd_logvar_sidecar_only", action="store_true")
parser.add_argument("--w_csd", type=float, default=1.0)
parser.add_argument("--w_csd_grounding", type=float, default=0.05)
parser.add_argument("--w_csd_vib", type=float, default=1e-4)
parser.add_argument("--csd_logvar_hidden_dim", type=int, default=128)
parser.add_argument("--csd_logvar_min", type=float, default=-12.0)
parser.add_argument("--csd_logvar_max", type=float, default=4.0)
parser.add_argument("--csd_logvar_init", type=float, default=-8.0)
parser.add_argument("--csd_initial_scale", type=float, default=5.0)
parser.add_argument("--csd_initial_shift", type=float, default=5.0)
parser.add_argument("--csd_hardness_temperature", type=float, default=1.0)
parser.add_argument("--csd_bootstrap_samples", type=int, default=1000)

if args.use_csd_logvar_sidecar and not args.use_uatvr_full:
    raise ValueError("use_csd_logvar_sidecar requires --use_uatvr_full")
if args.csd_logvar_sidecar_only and not args.use_csd_logvar_sidecar:
    raise ValueError("csd_logvar_sidecar_only requires --use_csd_logvar_sidecar")
if args.csd_logvar_sidecar_only and not args.use_retrieval_risk_sidecar:
    raise ValueError("csd_logvar_sidecar_only requires --use_retrieval_risk_sidecar")
if min(args.w_csd, args.w_csd_grounding, args.w_csd_vib) < 0:
    raise ValueError("CSD loss weights must be non-negative")
if args.use_csd_logvar_sidecar and args.w_csd == 0 and args.w_csd_grounding == 0:
    raise ValueError("CSD or grounding weight must be positive")
if args.csd_logvar_hidden_dim <= 0:
    raise ValueError("csd_logvar_hidden_dim must be positive")
if not args.csd_logvar_min < args.csd_logvar_max:
    raise ValueError("csd_logvar_min must be smaller than csd_logvar_max")
if not args.csd_logvar_min <= args.csd_logvar_init <= args.csd_logvar_max:
    raise ValueError("csd_logvar_init must lie inside the clamp bounds")
if args.csd_initial_scale <= 0 or args.csd_hardness_temperature <= 0:
    raise ValueError("CSD scales and temperatures must be positive")
if args.csd_bootstrap_samples < 0:
    raise ValueError("csd_bootstrap_samples must be non-negative")
```

- [ ] **Step 4: Write the failing manifest test**

```python
def test_experiment_manifest_records_csd_logvar_contract():
    args = _crf_args_namespace(
        aleatoric_crf_training=False, use_retrieval_risk_sidecar=True,
        use_csd_logvar_sidecar=True, csd_logvar_sidecar_only=True,
        w_csd=1.0, w_csd_grounding=0.05, w_csd_vib=1e-4,
        csd_logvar_hidden_dim=128, csd_logvar_min=-12.0,
        csd_logvar_max=4.0, csd_logvar_init=-8.0,
        csd_initial_scale=5.0, csd_initial_shift=5.0,
        csd_hardness_temperature=1.0, csd_bootstrap_samples=1000,
        init_model="/tmp/a1-bin.4",
    )
    payload = build_experiment_manifest(
        args, split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 4}, git_state={"commit": "abc"},
    )
    assert payload["csd_logvar"] == {
        "enabled": True, "sidecar_only": True,
        "weights": {"csd": 1.0, "grounding": 0.05, "vib": 1e-4},
        "hidden_dim": 128, "bounds": [-12.0, 4.0], "init_logvar": -8.0,
        "initial_scale": 5.0, "initial_shift": 5.0,
        "hardness_temperature": 1.0, "bootstrap_samples": 1000,
        "init_model": "/tmp/a1-bin.4",
        "score_path": "auxiliary_only_raw_wti_unchanged",
    }
```

- [ ] **Step 5: Implement the conditional manifest section**

```python
if bool(getattr(args, "use_csd_logvar_sidecar", False)):
    payload["csd_logvar"] = {
        "enabled": True,
        "sidecar_only": bool(getattr(args, "csd_logvar_sidecar_only", False)),
        "weights": {
            "csd": float(getattr(args, "w_csd", 1.0)),
            "grounding": float(getattr(args, "w_csd_grounding", 0.05)),
            "vib": float(getattr(args, "w_csd_vib", 1e-4)),
        },
        "hidden_dim": int(getattr(args, "csd_logvar_hidden_dim", 128)),
        "bounds": [float(getattr(args, "csd_logvar_min", -12.0)),
                   float(getattr(args, "csd_logvar_max", 4.0))],
        "init_logvar": float(getattr(args, "csd_logvar_init", -8.0)),
        "initial_scale": float(getattr(args, "csd_initial_scale", 5.0)),
        "initial_shift": float(getattr(args, "csd_initial_shift", 5.0)),
        "hardness_temperature": float(getattr(args, "csd_hardness_temperature", 1.0)),
        "bootstrap_samples": int(getattr(args, "csd_bootstrap_samples", 1000)),
        "init_model": getattr(args, "init_model", ""),
        "score_path": "auxiliary_only_raw_wti_unchanged",
    }
```

- [ ] **Step 6: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_config.py tests/test_experiment_tracking.py
git add main_task_retrieval.py experiment_tracking.py tests/test_csd_logvar_config.py tests/test_experiment_tracking.py
git commit -m "feat: register CSD logvar experiment contract"
```

---

### Task 4: June-model A2-only integration and exact freeze boundary

**Files:**
- Modify: `modules/modeling_mulit.py` in `UATVR.__init__`, `UATVR.forward`, and `UATVR._loose_similarity`
- Modify: `main_task_retrieval.py` near the A1 freeze helpers and `train_epoch()`
- Create: `tests/test_csd_logvar_integration.py`

**Interfaces:**
- Adds `UATVR._csd_logvar_terms(wti_logits, text_mu, video_mu, group_ids)`.
- Adds `configure_csd_logvar_sidecar_training(model, args)` and `set_csd_logvar_training_mode(model)`.
- Adds training result keys prefixed by `csd_logvar_` while preserving the A1 early-return behavior.

- [ ] **Step 1: Write failing freeze-allowlist and mode tests**

```python
from types import SimpleNamespace

import torch
from torch import nn

from main_task_retrieval import configure_csd_logvar_sidecar_training, set_csd_logvar_training_mode
from prob_models.csd_logvar import CSDLogVarSidecar


class _TinyA2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.retrieval_risk_sidecar = nn.Linear(4, 2)
        self.csd_logvar_sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
        self.csd_logvar_sidecar_only = True


def test_a2_only_freezes_every_parameter_outside_exact_prefix():
    model = _TinyA2Model()
    configure_csd_logvar_sidecar_training(model, SimpleNamespace(
        use_csd_logvar_sidecar=True, csd_logvar_sidecar_only=True))
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith("csd_logvar_sidecar.") for name in trainable)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 64


def test_a2_mode_keeps_only_new_sidecar_in_train_mode():
    model = _TinyA2Model().train()
    set_csd_logvar_training_mode(model)
    assert model.training is True
    assert model.backbone.training is False
    assert model.retrieval_risk_sidecar.training is False
    assert model.csd_logvar_sidecar.training is True
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_integration.py
```

Expected: import fails for the new configuration helpers.

- [ ] **Step 3: Instantiate the sidecar and restore its registered output initialization after `self.apply()`**

```python
self.use_csd_logvar_sidecar = bool(getattr(self.task_config, "use_csd_logvar_sidecar", False))
self.csd_logvar_sidecar_only = bool(getattr(self.task_config, "csd_logvar_sidecar_only", False))
self.w_csd = float(getattr(self.task_config, "w_csd", 1.0))
self.w_csd_grounding = float(getattr(self.task_config, "w_csd_grounding", 0.05))
self.w_csd_vib = float(getattr(self.task_config, "w_csd_vib", 1e-4))
self.csd_hardness_temperature = float(getattr(self.task_config, "csd_hardness_temperature", 1.0))
if self.use_csd_logvar_sidecar:
    self.csd_logvar_sidecar = CSDLogVarSidecar(
        embed_dim=embed_dim,
        hidden_dim=int(getattr(self.task_config, "csd_logvar_hidden_dim", 128)),
        logvar_min=float(getattr(self.task_config, "csd_logvar_min", -12.0)),
        logvar_max=float(getattr(self.task_config, "csd_logvar_max", 4.0)),
        init_logvar=float(getattr(self.task_config, "csd_logvar_init", -8.0)),
        initial_scale=float(getattr(self.task_config, "csd_initial_scale", 5.0)),
        initial_shift=float(getattr(self.task_config, "csd_initial_shift", 5.0)),
    )

self.apply(self.init_weights)
if self.use_csd_logvar_sidecar:
    self.csd_logvar_sidecar.reset_output_parameters()
```

Keep one parent `self.apply(self.init_weights)` call; the reset must immediately follow it so the parent initializer cannot overwrite the registered final bias.

- [ ] **Step 4: Implement exact freeze and train/eval-mode helpers**

```python
def configure_csd_logvar_sidecar_training(model, args):
    enabled = bool(getattr(args, "use_csd_logvar_sidecar", False))
    sidecar_only = bool(getattr(args, "csd_logvar_sidecar_only", False))
    if sidecar_only and not enabled:
        raise ValueError("csd_logvar_sidecar_only requires use_csd_logvar_sidecar")
    if not sidecar_only:
        return
    if not hasattr(model, "csd_logvar_sidecar"):
        raise ValueError("model does not expose csd_logvar_sidecar")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.csd_logvar_sidecar.parameters():
        parameter.requires_grad = True
    unexpected = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and not name.startswith("csd_logvar_sidecar.")]
    if unexpected:
        raise RuntimeError(f"unexpected A2 trainable parameters: {unexpected}")


def set_csd_logvar_training_mode(model):
    inner = model.module if hasattr(model, "module") else model
    if not getattr(inner, "csd_logvar_sidecar_only", False):
        return
    model.eval()
    model.training = True
    inner.training = True
    inner.csd_logvar_sidecar.train()
```

Call the configure helper after A1 configuration and the mode helper after `model.train()` in each epoch.

- [ ] **Step 5: Write a failing UATVR loss-isolation test**

```python
from modules.modeling_mulit import UATVR


def _tiny_uatvr_for_a2():
    model = object.__new__(UATVR)
    nn.Module.__init__(model)
    model.use_csd_logvar_sidecar = True
    model.csd_logvar_sidecar_only = True
    model.w_csd, model.w_csd_grounding, model.w_csd_vib = 1.0, 0.05, 1e-4
    model.csd_hardness_temperature = 1.0
    model.csd_logvar_sidecar = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    return model


def test_uatvr_a2_terms_update_only_new_sidecar():
    model = _tiny_uatvr_for_a2()
    wti = torch.randn(3, 3, requires_grad=True)
    text_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    video_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    terms = model._csd_logvar_terms(wti, text_mu, video_mu, torch.tensor([7, 7, 9]))
    terms["csd_logvar_loss"].backward()
    assert wti.grad is None and text_mu.grad is None and video_mu.grad is None


def test_sidecar_state_dict_round_trip_is_exact(tmp_path):
    torch.manual_seed(19)
    source = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    text_mu = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    video_mu = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    expected = source(text_mu, video_mu)
    checkpoint = tmp_path / "csd-logvar.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = CSDLogVarSidecar(embed_dim=4, hidden_dim=3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(text_mu, video_mu)
    for name in ("text_logvar", "video_logvar", "match_logits"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
```

- [ ] **Step 6: Implement the multi-positive A2 terms and early return**

```python
def _csd_logvar_terms(self, wti_logits, text_mu, video_mu, group_ids):
    if group_ids is None:
        raise ValueError("group_ids are required when CSD logvar sidecar is enabled")
    group_ids = torch.as_tensor(group_ids, device=wti_logits.device).reshape(-1)
    if tuple(wti_logits.shape) != (group_ids.numel(), group_ids.numel()):
        raise ValueError("group_ids must align with square training logits")
    total, output = csd_logvar_objective(
        self.csd_logvar_sidecar, text_mu, video_mu, wti_logits,
        positive_mask_from_group_ids(group_ids), self.w_csd,
        self.w_csd_grounding, self.w_csd_vib, self.csd_hardness_temperature,
    )
    names = ("csd", "csd_positive", "csd_negative", "grounding", "grounding_fit",
             "grounding_rank", "vib", "mean_distance", "variance_distance", "scale",
             "shift", "text_clamp_low", "text_clamp_high", "video_clamp_low", "video_clamp_high")
    terms = {"csd_logvar_loss": total}
    terms.update({f"csd_logvar_{name}": output[name].detach().mean() for name in names})
    return terms
```

Immediately after WTI logits are computed, add the A2-only early return using normalized `text_pooled`, normalized `mu_video`, and gathered `global_group_ids`. Return scalar zeros for every legacy loss. In `forward()`, zero `sim_loss` when either sidecar-only mode is active, add `csd_logvar_loss`, and expose all A2 diagnostics.

```python
if self.training and self.csd_logvar_sidecar_only:
    csd_terms = self._csd_logvar_terms(
        wti_logits, text_pooled, mu_video, global_group_ids
    )
    zero = wti_logits.new_zeros(())
    return {
        "retrieve_logits": wti_logits,
        "MIL_loss": zero, "evidential_loss": zero,
        "neg_reg_loss": zero, "orth_loss": zero,
        "retrieval_risk_loss": zero, "retrieval_risk_edl": zero,
        "retrieval_risk_fit": zero, "retrieval_risk_rank": zero,
        **csd_terms,
    }
```

- [ ] **Step 7: Pass group IDs for either sidecar and retain A2 diagnostics for logging**

```python
sidecar_kwargs = {}
if getattr(args, "use_retrieval_risk_sidecar", False) or getattr(args, "use_csd_logvar_sidecar", False):
    sidecar_kwargs["group_ids"] = group_ids
loss_output = model(input_ids, segment_ids, input_mask, video, video_mask, **sidecar_kwargs)
if isinstance(loss_output, dict):
    if getattr(args, "use_csd_logvar_sidecar", False):
        loss_terms = {key: value.detach() for key, value in loss_output.items()
                      if key.startswith("csd_logvar_") and key != "csd_logvar_loss"}
    loss = loss_output["total"]
else:
    loss = loss_output
```

Accumulate scalar window means and log CSD positive/negative BCE, grounding fit/rank, VIB, mean/variance distance, scale/shift, and all clamp fractions.

- [ ] **Step 8: Run integration and A1 regression tests; commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_csd_logvar.py tests/test_csd_logvar_loss.py \
  tests/test_csd_logvar_integration.py tests/test_retrieval_risk_features.py \
  tests/test_retrieval_risk_loss.py tests/test_retrieval_risk_integration.py
git add modules/modeling_mulit.py main_task_retrieval.py tests/test_csd_logvar_integration.py
git commit -m "feat: integrate frozen CSD logvar training"
```

---

### Task 5: Intrinsic-uncertainty collection and directional metrics

**Files:**
- Modify: `modules/modeling_mulit.py` in `_loose_similarity()` and `get_similarity_logits()`
- Modify: `main_task_retrieval.py` in `_run_on_single_gpu()` and `eval_epoch()`
- Modify: `prob_models/retrieval_risk_metrics.py`
- Create: `tests/test_csd_logvar_metrics.py`
- Modify: `tests/test_directional_eval.py`

**Interfaces:**
- Adds `get_similarity_logits(..., return_csd_aux=False)` without changing the false case.
- Adds `eval_epoch(..., report_csd_logvar=None)`; `None` follows the CLI flag and `False` supports paired preflight.
- Produces `evaluate_intrinsic_uncertainty()`, `summarize_logvar()`, and chunked `summarize_csd_components()`.
- Collects one text record per caption and one video record per unique video.

- [ ] **Step 1: Write failing metric and multi-caption tests**

```python
import numpy as np

from prob_models.retrieval_risk_metrics import (
    build_eval_positive_mask, evaluate_intrinsic_uncertainty,
    summarize_csd_components, summarize_logvar,
)


def test_intrinsic_metrics_use_one_value_per_directional_query():
    scores = np.array([[4., 0.], [3., 0.], [0., 4.], [5., 3.]], dtype=np.float32)
    mask = build_eval_positive_mask(4, 2, cut_off_points=[2, 4])
    t2v = evaluate_intrinsic_uncertainty(scores, mask, np.array([.1, .2, .3, .9]), 20, 0)
    v2t = evaluate_intrinsic_uncertainty(scores.T, mask.T, np.array([.1, .9]), 20, 1)
    assert t2v["num_queries"] == 4 and v2t["num_queries"] == 2


def test_logvar_summary_reports_clamp_rates():
    summary = summarize_logvar(np.array([[-12., -8.], [4., -7.]]), -12., 4.)
    assert summary["clamp_low_fraction"] == 0.25
    assert summary["clamp_high_fraction"] == 0.25
    assert summary["variance"]["mean"] > 0


def test_chunked_csd_summary_matches_direct_components():
    text_mu = np.array([[1., 0.], [0., 1.]])
    video_mu = np.array([[1., 0.], [-1., 0.]])
    text_logvar = np.log(np.full((2, 2), .25))
    video_logvar = np.log(np.full((2, 2), .5))
    summary = summarize_csd_components(text_mu, video_mu, text_logvar, video_logvar, chunk_size=1)
    assert summary["mean_distance"]["mean"] == 2.0
    assert summary["variance_distance"]["mean"] == 1.5
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_metrics.py
```

Expected: imports fail for the new metric functions.

- [ ] **Step 3: Implement intrinsic and distribution summaries**

```python
def evaluate_intrinsic_uncertainty(scores, positive_mask, uncertainty,
                                   bootstrap_samples=1000, seed=0):
    ranks = positive_ranks(scores, positive_mask)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if uncertainty.shape != ranks.shape or not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty must be finite with one value per query")
    failure = ranks > 0
    value = _binary_auroc(failure, uncertainty)
    ci_low, ci_high = _bootstrap_auroc(failure, uncertainty, int(bootstrap_samples), int(seed))
    rho = spearmanr(uncertainty, ranks).statistic
    if not np.isfinite(rho):
        raise ValueError("Spearman correlation requires non-constant inputs")
    return {
        "num_queries": int(ranks.size), "num_failures": int(failure.sum()),
        "spearman_uncertainty_vs_rank": float(rho),
        "failure_auroc": {"value": float(value), "ci_low": ci_low, "ci_high": ci_high},
        "risk_coverage": _risk_coverage(ranks, uncertainty),
        "uncertainty_stats": _summary(uncertainty),
    }


def summarize_logvar(logvar, lower, upper):
    values = np.asarray(logvar, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("logvar must be a finite 2-D array")
    return {
        "logvar": _summary(values.reshape(-1)),
        "variance": _summary(np.exp(values).reshape(-1)),
        "clamp_low_fraction": float(np.mean(values == float(lower))),
        "clamp_high_fraction": float(np.mean(values == float(upper))),
    }
```

Implement `summarize_csd_components()` without materializing the full text-by-video matrix:

```python
def _stream_summary(chunks):
    count, total, square_total = 0, 0.0, 0.0
    minimum, maximum = np.inf, -np.inf
    for chunk in chunks:
        values = np.asarray(chunk, dtype=np.float64)
        count += values.size
        total += float(values.sum())
        square_total += float(np.square(values).sum())
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    mean = total / count
    variance = max(square_total / count - mean * mean, 0.0)
    return {"min": minimum, "max": maximum, "mean": mean, "std": float(np.sqrt(variance))}


def summarize_csd_components(text_mu, video_mu, text_logvar, video_logvar, chunk_size=256):
    text_mu = np.asarray(text_mu, dtype=np.float64)
    video_mu = np.asarray(video_mu, dtype=np.float64)
    text_variance = np.exp(np.asarray(text_logvar, dtype=np.float64)).sum(1)
    video_variance = np.exp(np.asarray(video_logvar, dtype=np.float64)).sum(1)
    if text_mu.shape != np.asarray(text_logvar).shape:
        raise ValueError("text mean/logvar shapes must match")
    if video_mu.shape != np.asarray(video_logvar).shape:
        raise ValueError("video mean/logvar shapes must match")
    if chunk_size <= 0 or text_mu.shape[1] != video_mu.shape[1]:
        raise ValueError("chunk_size and embedding dimensions must be valid")

    def mean_chunks():
        video_norm = np.square(video_mu).sum(1)[None, :]
        for start in range(0, len(text_mu), chunk_size):
            text = text_mu[start:start + chunk_size]
            yield np.maximum(
                np.square(text).sum(1)[:, None] + video_norm - 2.0 * text @ video_mu.T,
                0.0,
            )

    def variance_chunks():
        for start in range(0, len(text_variance), chunk_size):
            yield text_variance[start:start + chunk_size, None] + video_variance[None, :]

    return {
        "mean_distance": _stream_summary(mean_chunks()),
        "variance_distance": _stream_summary(variance_chunks()),
    }
```

- [ ] **Step 4: Expose intrinsic tensors without changing default retrieval output**

```python
if not self.training and return_csd_aux and self.use_csd_logvar_sidecar:
    csd_output = self.csd_logvar_sidecar(text_pooled.detach(), mu_video.detach())
    return wti_logits, {
        "text_mu": text_pooled.detach(), "video_mu": mu_video.detach(),
        "text_logvar": csd_output["text_logvar"].detach(),
        "video_logvar": csd_output["video_logvar"].detach(),
    }
```

Thread `return_csd_aux=False` through `get_similarity_logits()`. The false path must retain the exact existing `(retrieve_logits, {})` interface.

- [ ] **Step 5: Collect each modality once and merge worker outputs**

```python
collect_csd = bool(getattr(args, "use_csd_logvar_sidecar", False))
text_payload, video_payload = [], []
for text_index, b1 in enumerate(batch_list_t):
    for chunk_index, chunk in enumerate(video_chunks):
        logits, aux = model.get_similarity_logits(
            sequence_output, text_token, visual_output, visual_hidden,
            input_mask, video_mask, loose_type=model.loose_type,
            return_csd_aux=collect_csd,
        )
        if collect_csd and chunk_index == 0:
            text_payload.append({name: aux[name].float().cpu()
                                 for name in ("text_mu", "text_logvar")})
        if collect_csd and text_index == 0:
            video_payload.append({name: aux[name].float().cpu()
                                  for name in ("video_mu", "video_logvar")})
```

Return structured similarity and intrinsic payload only in A2 mode. In `parallel_apply`, concatenate text payloads in worker order, use video payload from worker zero, and assert counts match the raw matrix axes.

```python
def _merge_csd_eval_outputs(outputs):
    similarity_rows = [row for output in outputs for row in output["similarity_rows"]]
    text_mu = torch.cat([output["text_mu"] for output in outputs], dim=0)
    text_logvar = torch.cat([output["text_logvar"] for output in outputs], dim=0)
    video_mu = outputs[0]["video_mu"]
    video_logvar = outputs[0]["video_logvar"]
    matrix = np.concatenate(tuple(similarity_rows), axis=0)
    if text_mu.size(0) != matrix.shape[0] or video_mu.size(0) != matrix.shape[1]:
        raise ValueError("intrinsic uncertainty axes do not align with retrieval matrix")
    return matrix, {
        "text_mu": text_mu, "text_logvar": text_logvar,
        "video_mu": video_mu, "video_logvar": video_logvar,
    }
```

- [ ] **Step 6: Attach separate T2V-text and V2T-video payloads**

```python
if report_csd_logvar is None:
    report_csd_logvar = bool(getattr(args, "use_csd_logvar_sidecar", False))
if report_csd_logvar:
    assert intrinsic is not None
    result["csd_logvar"] = {
        "t2v": evaluate_intrinsic_uncertainty(
            metric_t2v_matrix, positive_mask,
            intrinsic["text_logvar"].exp().mean(-1).numpy(), args.csd_bootstrap_samples, 0),
        "v2t": evaluate_intrinsic_uncertainty(
            metric_v2t_matrix.T, positive_mask.T,
            intrinsic["video_logvar"].exp().mean(-1).numpy(), args.csd_bootstrap_samples, 1),
        "text_distribution": summarize_logvar(
            intrinsic["text_logvar"].numpy(), args.csd_logvar_min, args.csd_logvar_max),
        "video_distribution": summarize_logvar(
            intrinsic["video_logvar"].numpy(), args.csd_logvar_min, args.csd_logvar_max),
        "distance_components": summarize_csd_components(
            intrinsic["text_mu"].numpy(), intrinsic["video_mu"].numpy(),
            intrinsic["text_logvar"].numpy(), intrinsic["video_logvar"].numpy()),
    }
```

Build `positive_mask` once when either A1 or A2 reporting is enabled. Keep A1 under `result["retrieval_risk"]`.

- [ ] **Step 7: Run metrics and directionality regressions; commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_csd_logvar_metrics.py tests/test_directional_eval.py \
  tests/test_retrieval_risk_metrics.py
git add modules/modeling_mulit.py main_task_retrieval.py prob_models/retrieval_risk_metrics.py \
  tests/test_csd_logvar_metrics.py tests/test_directional_eval.py
git commit -m "feat: evaluate intrinsic CSD uncertainty"
```

---

### Task 6: Paired invariance, state audit, and bidirectional selection

**Files:**
- Create: `prob_models/csd_logvar_protocol.py`
- Create: `tests/test_csd_logvar_protocol.py`
- Modify: `main_task_retrieval.py` in `_compute_retrieval_risk_metrics()`, `eval_epoch()`, and the training selection loop

**Interfaces:**
- Produces `array_sha256(array) -> str` and `non_csd_state_dict_mismatches(base_checkpoint, candidate_checkpoint) -> list[str]`.
- Produces `compare_paired_invariance(baseline, candidate)`, `choose_csd_logvar_checkpoint(candidates)`, and `csd_logvar_acceptance(metrics)`.
- Writes `a1_paired_baseline.json` before epoch 1 and `best_csd_logvar_checkpoint.json` after every validation epoch.

- [ ] **Step 1: Write failing digest, audit, and selector tests**

```python
import numpy as np
import pytest
import torch

from prob_models.csd_logvar_protocol import (
    array_sha256, choose_csd_logvar_checkpoint,
    non_csd_state_dict_mismatches,
)


def test_array_digest_changes_on_one_value():
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first.copy()
    second[1, 2] += 1
    assert array_sha256(first) != array_sha256(second)


def test_state_audit_ignores_only_a2_keys(tmp_path):
    base = {"backbone": torch.tensor([1.]), "retrieval_risk_sidecar.w": torch.tensor([2.])}
    candidate = {**base, "csd_logvar_sidecar.raw_scale": torch.tensor(3.)}
    base_path, candidate_path = tmp_path / "base.pt", tmp_path / "candidate.pt"
    torch.save(base, base_path)
    torch.save(candidate, candidate_path)
    assert non_csd_state_dict_mismatches(base_path, candidate_path) == []
    candidate["backbone"] = torch.tensor([1.5])
    torch.save(candidate, candidate_path)
    assert non_csd_state_dict_mismatches(base_path, candidate_path) == ["backbone"]


def test_selector_uses_min_rho_then_mean_auroc_then_earlier_epoch():
    candidates = [
        {"epoch": 0, "path": "a", "valid": True, "rho": [.30, .21], "auroc": [.70, .70]},
        {"epoch": 1, "path": "b", "valid": True, "rho": [.22, .22], "auroc": [.65, .65]},
        {"epoch": 2, "path": "c", "valid": True, "rho": [.22, .22], "auroc": [.80, .80]},
        {"epoch": 3, "path": "d", "valid": True, "rho": [.22, .22], "auroc": [.80, .80]},
    ]
    assert choose_csd_logvar_checkpoint(candidates)["path"] == "c"


def test_selector_rejects_invalid_candidates():
    with pytest.raises(ValueError, match="valid"):
        choose_csd_logvar_checkpoint([
            {"epoch": 0, "path": "a", "valid": False, "rho": [.9, .9], "auroc": [.9, .9]}
        ])
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_protocol.py
```

Expected: collection fails because `prob_models.csd_logvar_protocol` does not exist.

- [ ] **Step 3: Implement deterministic hashes and numerical state comparison**

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


A2_PREFIX = "csd_logvar_sidecar."


def array_sha256(array):
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def non_csd_state_dict_mismatches(base_checkpoint, candidate_checkpoint):
    base = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=True)
    candidate = torch.load(Path(candidate_checkpoint), map_location="cpu", weights_only=True)
    expected = {name for name in base if not name.startswith(A2_PREFIX)}
    actual = {name for name in candidate if not name.startswith(A2_PREFIX)}
    mismatches = sorted(expected.symmetric_difference(actual))
    for name in sorted(expected.intersection(actual)):
        left, right = base[name], candidate[name]
        if left.shape != right.shape or not torch.equal(left.to(right.dtype), right):
            mismatches.append(name)
    return sorted(set(mismatches))
```

The dtype cast reproduces the existing FP16-checkpoint-to-FP32-model conversion and then requires exact equality in the candidate checkpoint's dtype.

- [ ] **Step 4: Add WTI and exact A1-output hashes to evaluation**

Retain CPU `failure_risk` and `vacuity` arrays long enough to hash them. Copy and hash raw WTI before DSL mutation; after directional retrieval and A1 risk results have been constructed, attach:

```python
result["paired_digest"] = {
    "raw_wti_sha256": array_sha256(risk_similarity_matrix),
    "a1_failure_risk_sha256": {
        direction: risk_array_digests[direction]["failure_risk"] for direction in directions
    },
    "a1_vacuity_sha256": {
        direction: risk_array_digests[direction]["vacuity"] for direction in directions
    },
    "retrieval": {direction: result[direction] for direction in directions},
}
```

`compare_paired_invariance()` must compare each field exactly and return individual booleans plus `passed`.

```python
def compare_paired_invariance(baseline, candidate):
    checks = {
        "raw_wti": baseline["raw_wti_sha256"] == candidate["raw_wti_sha256"],
        "a1_failure_risk": baseline["a1_failure_risk_sha256"] == candidate["a1_failure_risk_sha256"],
        "a1_vacuity": baseline["a1_vacuity_sha256"] == candidate["a1_vacuity_sha256"],
        "retrieval": baseline["retrieval"] == candidate["retrieval"],
    }
    return {"passed": all(checks.values()), "checks": checks}
```

- [ ] **Step 5: Implement rejection, selection, and acceptance**

```python
def choose_csd_logvar_checkpoint(candidates):
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError("no valid CSD logvar checkpoint")
    return max(valid, key=lambda item: (
        min(item["rho"]), sum(item["auroc"]) / 2.0, -int(item["epoch"])))


def _monotone_selective_accuracy(direction_metrics):
    curve = direction_metrics["risk_coverage"]
    values = [curve[f"accuracy_at_{point:.2f}"] for point in (.25, .50, .75, 1.00)]
    return all(left >= right for left, right in zip(values, values[1:]))


def csd_logvar_acceptance(metrics):
    t2v, v2t = metrics["t2v"], metrics["v2t"]
    distributions = (metrics["text_distribution"], metrics["video_distribution"])
    checks = {
        "rho": min(t2v["spearman_uncertainty_vs_rank"],
                   v2t["spearman_uncertainty_vs_rank"]) >= .20,
        "auroc": min(t2v["failure_auroc"]["value"],
                     v2t["failure_auroc"]["value"]) > .5,
        "selective_accuracy": _monotone_selective_accuracy(t2v) and _monotone_selective_accuracy(v2t),
        "nonzero_variance": all(item["variance"]["std"] > 0 for item in distributions),
        "clamp_rate": all(item["clamp_low_fraction"] + item["clamp_high_fraction"] < .05
                          for item in distributions),
    }
    return {"passed": all(checks.values()), "checks": checks}
```

Before selection, recursively verify numeric metrics are finite. A candidate is valid only when finiteness, paired invariance, an empty non-A2 mismatch list, and each modality's low-plus-high clamp fraction below `0.05` hold.

- [ ] **Step 6: Register baseline and replace R@1 selection only in A2-only mode**

Before epoch 1 on rank zero, disable intrinsic metrics because the registered initialization is constant. Reuse an existing registration during a 3-to-5 epoch continuation so the audit reference remains the original A1 bin.4 and prior candidates remain eligible:

```python
if getattr(args, "csd_logvar_sidecar_only", False):
    baseline_path = Path(args.output_dir) / "a1_paired_baseline.json"
    selection_path = Path(args.output_dir) / "best_csd_logvar_checkpoint.json"
    if baseline_path.exists():
        paired_registration = json.loads(baseline_path.read_text())
        paired_baseline = paired_registration["paired_digest"]
        a1_reference_checkpoint = paired_registration["reference_checkpoint"]
    else:
        baseline_eval = eval_epoch(
            args, model, eval_dataloader, device, n_gpu, report_csd_logvar=False
        )
        paired_baseline = baseline_eval["paired_digest"]
        a1_reference_checkpoint = args.init_model
        atomic_write_json(baseline_path, {
            "reference_checkpoint": a1_reference_checkpoint,
            "paired_digest": paired_baseline,
        })
    csd_candidates = []
    if args.resume_model and selection_path.exists():
        csd_candidates = json.loads(selection_path.read_text())["candidates"]
```

After each checkpoint validation, build one bidirectional candidate, compare it to `paired_baseline`, call `non_csd_state_dict_mismatches(a1_reference_checkpoint, output_model_file)`, and write both `candidates` and `winner` to `best_csd_logvar_checkpoint.json`. Keep existing R@1 selectors unchanged outside A2-only mode. Final test must load the one selected A2 checkpoint once and evaluate both directions together.

- [ ] **Step 7: Run protocol and selection regressions; commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_csd_logvar_protocol.py tests/test_directional_eval.py \
  tests/test_retrieval_risk_metrics.py tests/test_run_train_msrvtt_bg.py
git add prob_models/csd_logvar_protocol.py main_task_retrieval.py tests/test_csd_logvar_protocol.py
git commit -m "feat: select invariant bidirectional logvar checkpoints"
```

---

### Task 7: Reproducible A2-CG/C/G launcher

**Files:**
- Create: `run_train_msrvtt_csd_logvar_bg.sh`
- Create: `tests/test_csd_logvar_launcher.py`

**Interfaces:**
- Default `CSD_VARIANT=CG` launches three epochs.
- `CSD_VARIANT=C` disables grounding; `CSD_VARIANT=G` disables CSD.
- `BASE_CKPT`, `CSD_EPOCHS`, `RUN_TAG`, and trailing arguments remain explicit overrides.

- [ ] **Step 1: Write failing static launcher tests**

```python
import os
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_csd_logvar_bg.sh"


def test_launcher_registers_a1_bin4_and_primary_contract():
    text = SCRIPT.read_text()
    assert "ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4" in text
    for flag in ("--use_uatvr_full", "--use_retrieval_risk_sidecar",
                 "--use_csd_logvar_sidecar", "--csd_logvar_sidecar_only",
                 "--w_csd", "--w_csd_grounding", "--w_csd_vib", "--run_final_test"):
        assert flag in text
    assert 'CSD_EPOCHS="${CSD_EPOCHS:-3}"' in text


def test_launcher_exposes_all_variants():
    text = SCRIPT.read_text()
    assert all(pattern in text for pattern in ('"CG")', '"C")', '"G")'))


def test_launcher_is_executable_and_parses():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_launcher.py
```

Expected: the launcher does not exist.

- [ ] **Step 3: Implement the launcher and variant gate**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CKPT="${BASE_CKPT:-${ROOT_DIR}/ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4}"
CSD_VARIANT="${CSD_VARIANT:-CG}"
CSD_EPOCHS="${CSD_EPOCHS:-3}"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing BASE_CKPT=${BASE_CKPT}" >&2
    exit 2
fi
case "${CSD_VARIANT}" in
    "CG") CSD_WEIGHT=1.0; GROUNDING_WEIGHT=0.05 ;;
    "C") CSD_WEIGHT=1.0; GROUNDING_WEIGHT=0.0 ;;
    "G") CSD_WEIGHT=0.0; GROUNDING_WEIGHT=0.05 ;;
    *) echo "Unsupported CSD_VARIANT=${CSD_VARIANT}; expected CG, C, or G" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-csd_logvar_a2_${CSD_VARIANT,,}}" \
EXPERIMENT_DESC="${EXPERIMENT_DESC:-A2-${CSD_VARIANT} independent CSD logvar sidecar}" \
bash "${ROOT_DIR}/run_train_msrvtt_bg.sh" \
    --use_uatvr_full --use_ada_norm --init_model "${BASE_CKPT}" \
    --use_retrieval_risk_sidecar --w_retrieval_risk 0.0 \
    --retrieval_risk_temperature 1.0 --retrieval_risk_hidden_dim 32 \
    --retrieval_risk_bootstrap_samples 1000 \
    --use_csd_logvar_sidecar --csd_logvar_sidecar_only \
    --w_csd "${CSD_WEIGHT}" --w_csd_grounding "${GROUNDING_WEIGHT}" --w_csd_vib 0.0001 \
    --csd_logvar_hidden_dim 128 --csd_logvar_min -12 --csd_logvar_max 4 \
    --csd_logvar_init -8 --csd_initial_scale 5 --csd_initial_shift 5 \
    --csd_hardness_temperature 1 --csd_bootstrap_samples 1000 \
    --epochs "${CSD_EPOCHS}" --DSL True --dsl_temperature 2.0 --run_final_test "$@"
```

- [ ] **Step 4: Make executable, run GREEN, and commit**

```bash
chmod +x run_train_msrvtt_csd_logvar_bg.sh
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_csd_logvar_launcher.py
git add run_train_msrvtt_csd_logvar_bg.sh tests/test_csd_logvar_launcher.py
git commit -m "feat: add reproducible CSD logvar experiments"
```

---

### Task 8: Full regression and one-optimizer-step verification

**Files:**
- Modify only a previously listed source or test if verification exposes a defect.

**Interfaces:**
- Produces evidence that legacy behavior is unchanged, exactly 263,426 A2 scalars train, checkpoints round-trip, and one real optimizer step is finite.

- [ ] **Step 1: Run the focused suite**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_csd_logvar.py tests/test_csd_logvar_loss.py \
  tests/test_csd_logvar_config.py tests/test_csd_logvar_integration.py \
  tests/test_csd_logvar_metrics.py tests/test_csd_logvar_protocol.py \
  tests/test_csd_logvar_launcher.py tests/test_retrieval_risk_features.py \
  tests/test_retrieval_risk_loss.py tests/test_retrieval_risk_integration.py \
  tests/test_retrieval_risk_metrics.py tests/test_directional_eval.py \
  tests/test_experiment_tracking.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and diff hygiene checks**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m compileall -q \
  prob_models/csd_logvar.py prob_models/csd_logvar_protocol.py \
  modules/modeling_mulit.py main_task_retrieval.py experiment_tracking.py
bash -n run_train_msrvtt_csd_logvar_bg.sh
git diff --check
```

Expected: all commands exit zero and `git diff --check` prints nothing.

- [ ] **Step 3: Run one real optimizer-step smoke test**

```bash
SMOKE_STEPS=1 CSD_VARIANT=CG CSD_EPOCHS=1 \
RUN_TAG=csd_logvar_a2_cg_smoke bash run_train_msrvtt_csd_logvar_bg.sh
```

Expected log evidence includes `SMOKE_STEPS cap=1`, trainable count `263426`, finite CSD/grounding/VIB diagnostics, and the smoke-validation skip. Audit the saved checkpoint and confirm every changed key starts with `csd_logvar_sidecar.`.

- [ ] **Step 4: Commit a correction only if smoke verification required one**

```bash
git add prob_models modules main_task_retrieval.py experiment_tracking.py tests run_train_msrvtt_csd_logvar_bg.sh
git commit -m "fix: satisfy CSD logvar smoke contract"
```

If no correction was needed, skip this command. Never add the smoke log or checkpoint.

---

### Task 9: Registered A2-CG run and conditional extension

**Files:**
- Generated but not committed: the A2-CG log and checkpoint directory.
- Create after completion: `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md`

**Interfaces:**
- Produces the paired A1 baseline, three validation checkpoints, one selected checkpoint, and one JSFUSION test evaluation.

- [ ] **Step 1: Start the registered three-epoch run**

```bash
CSD_VARIANT=CG CSD_EPOCHS=3 RUN_TAG=csd_logvar_a2_cg \
bash run_train_msrvtt_csd_logvar_bg.sh
```

- [ ] **Step 2: Resolve the run directory and validate artifacts**

```bash
A2_OUTPUT_DIR="$(find ckpts -maxdepth 1 -type d -name 'ckpt_msrvtt_*csd_logvar_a2_cg*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "${A2_OUTPUT_DIR}"
test -f "${A2_OUTPUT_DIR}/a1_paired_baseline.json"
test -f "${A2_OUTPUT_DIR}/best_csd_logvar_checkpoint.json"
test -f "${A2_OUTPUT_DIR}/final_test.json"
```

- [ ] **Step 3: Apply the registered continuation rule**

Continue only when epoch index 2 has a strictly larger selection tuple than epoch index 1. Resume model and optimizer into the same output directory:

```bash
BASE_CKPT="${A2_OUTPUT_DIR}/pytorch_model.bin.2" \
OUTPUT_DIR="${A2_OUTPUT_DIR}" CSD_VARIANT=CG CSD_EPOCHS=5 \
RUN_TAG=csd_logvar_a2_cg_resume bash run_train_msrvtt_csd_logvar_bg.sh \
  --resume_model "${A2_OUTPUT_DIR}/pytorch_opt.bin.2"
```

If epoch index 2 is not strictly better, stop at three epochs.

- [ ] **Step 4: Write the primary result section**

Record base checkpoint, code commit, manifest, paired digests, non-A2 audit, per-epoch loss diagnostics, selected epoch, both directions' rho/AUROC/AURC/selective accuracy, distribution/clamp statistics, CSD components, and acceptance checks. State separately whether A2-CG passed and whether A1 remained unchanged.

---

### Task 10: Conditional ablations and final interpretation

**Files:**
- Modify: `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md`
- Generated but not committed: A2-C and A2-G logs/checkpoints.

**Interfaces:**
- Runs A2-C and A2-G only after A2-CG passes every acceptance check.
- Produces an A1/A2-CG/A2-C/A2-G comparison and evidence-bounded conclusion.

- [ ] **Step 1: Stop if the combined experiment fails**

Read the selected checkpoint's validation `acceptance` in `best_csd_logvar_checkpoint.json`. If it is false, do not launch ablations; finish the report with failed checks and retain A1. Report JSFUSION metrics separately without using test data to decide whether ablations run.

- [ ] **Step 2: Launch independent ablations after a pass**

```bash
CSD_VARIANT=C CSD_EPOCHS=3 RUN_TAG=csd_logvar_a2_c bash run_train_msrvtt_csd_logvar_bg.sh
CSD_VARIANT=G CSD_EPOCHS=3 RUN_TAG=csd_logvar_a2_g bash run_train_msrvtt_csd_logvar_bg.sh
```

Run sequentially and apply the same continuation rule independently.

- [ ] **Step 3: Complete the comparison and registered interpretation**

For each executed variant, report selected epoch, paired invariance, T2V/V2T rho, AUROC with confidence intervals, AURC, 25/50/75/100% selective accuracy, variance standard deviation, clamp fractions, and pass/fail.

- A2-CG and A2-G pass while A2-C fails: retain hardness-grounded logvar and make no CSD-contribution claim.
- A2-C passes while A2-G fails: describe logvar as probabilistic matching uncertainty, not retrieval-hardness uncertainty.
- Only A2-CG passes: claim complementarity and retain both losses.
- No A2 variant passes the bidirectional gate: retain A1 and report intrinsic Gaussian uncertainty as unsupported.

- [ ] **Step 4: Validate and commit only the report**

```bash
rg -n "[T]BD|[T]ODO|[P]LACEHOLDER|\?\?\?" docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md && exit 1 || true
git diff --check
git add docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md
git commit -m "docs: report independent CSD logvar experiments"
```
