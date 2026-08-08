# Epistemic Uncertainty Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已完成的独立包中实现 Ensemble、MC Dropout、Prototype 三种 Epistemic 估计，以及显式校准的组合分数和 OOD 判定。

**Architecture:** Epistemic 模块不调用 encoder，只消费 `[M,B,D]` 多次表示或 `[B,D]` 特征与 `[K,D]` prototypes。所有估计结果使用统一不可变输出；组合逻辑只使用调用方提供的权重、校准统计和阈值。

**Tech Stack:** Python 3、PyTorch、pytest

## Global Constraints

- 依赖已通过测试的 Aleatoric 计划产物，但不修改项目现有检索代码。
- 支持 Ensemble disagreement、MC Dropout disagreement 和 Prototype distance 三种来源。
- 多次采样必须满足 `M >= 2`；prototype 必须满足 `K >= 1`。
- 默认 prototype 指标为 cosine distance，同时支持 squared Euclidean distance。
- 不硬编码校准统计、组合权重或 OOD 阈值。
- 先测试失败，再写最小实现。

---

## 文件职责

- `uncertainty_modules/types.py`：新增统一的 `EpistemicOutput`。
- `uncertainty_modules/epistemic.py`：三种估计、分量组合和 OOD 判定。
- `uncertainty_modules/tests/test_epistemic.py`：数值、错误处理和组合测试。
- `uncertainty_modules/__init__.py`：导出 Epistemic 公共 API。
- `uncertainty_modules/README.md`：增加三种估计的独立使用说明。

### Task 1: Ensemble 与 MC Dropout disagreement

**Files:**
- Modify: `uncertainty_modules/types.py`
- Create: `uncertainty_modules/epistemic.py`
- Create: `uncertainty_modules/tests/test_epistemic.py`

**Interfaces:**
- Consumes: `samples: torch.Tensor[M,B,D]`。
- Produces: `EpistemicOutput`、`from_ensemble`、`from_mc_dropout`。

- [ ] **Step 1: 写 disagreement 失败测试**

```python
import pytest
import torch

from uncertainty_modules.epistemic import EpistemicUncertaintyModule


def test_identical_ensemble_samples_have_zero_disagreement():
    base = torch.randn(2, 4)
    output = EpistemicUncertaintyModule().from_ensemble(base.unsqueeze(0).repeat(3, 1, 1))
    assert torch.equal(output.diagonal_variance, torch.zeros_like(base))
    assert torch.equal(output.uncertainty_score, torch.zeros(2))
    assert "ensemble" in output.component_scores


def test_mc_dropout_disagreement_increases_with_spread():
    module = EpistemicUncertaintyModule()
    base = torch.zeros(3, 2, 4)
    spread = base.clone()
    spread[0] = -1
    spread[2] = 1
    assert torch.all(module.from_mc_dropout(spread).uncertainty_score > module.from_mc_dropout(base).uncertainty_score)


@pytest.mark.parametrize("shape", [(1, 2, 4), (2, 4)])
def test_disagreement_rejects_invalid_samples(shape):
    with pytest.raises(ValueError):
        EpistemicUncertaintyModule().from_ensemble(torch.randn(*shape))
```

- [ ] **Step 2: 运行测试并确认因 Epistemic 模块不存在而失败**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: FAIL，错误包含缺失模块或类名。

- [ ] **Step 3: 新增结果类型和 disagreement 实现**

```python
# 追加到 uncertainty_modules/types.py
from dataclasses import field
from typing import Mapping


@dataclass(frozen=True)
class EpistemicOutput:
    uncertainty_score: torch.Tensor
    component_scores: Mapping[str, torch.Tensor] = field(default_factory=dict)
    diagonal_variance: torch.Tensor | None = None
    nearest_prototype_distance: torch.Tensor | None = None
    nearest_prototype_index: torch.Tensor | None = None
    is_ood: torch.Tensor | None = None
```

```python
# uncertainty_modules/epistemic.py
import torch
from torch import nn

from .types import EpistemicOutput


class EpistemicUncertaintyModule(nn.Module):
    @staticmethod
    def _validate_samples(samples: torch.Tensor) -> None:
        if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] == 0 or samples.shape[2] == 0:
            raise ValueError("samples must have shape [M, B, D] with M >= 2")
        if not samples.is_floating_point() or not torch.isfinite(samples).all():
            raise ValueError("samples must be finite floating-point values")

    def _disagreement(self, samples: torch.Tensor, name: str) -> EpistemicOutput:
        self._validate_samples(samples)
        diagonal = torch.var(samples, dim=0, unbiased=False)
        score = diagonal.mean(dim=-1)
        return EpistemicOutput(score, {name: score}, diagonal_variance=diagonal)

    def from_ensemble(self, samples: torch.Tensor) -> EpistemicOutput:
        return self._disagreement(samples, "ensemble")

    def from_mc_dropout(self, samples: torch.Tensor) -> EpistemicOutput:
        return self._disagreement(samples, "mc_dropout")
```

- [ ] **Step 4: 运行 Task 1 测试**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: `4 passed`。

- [ ] **Step 5: 提交 Task 1**

```bash
git add uncertainty_modules/types.py uncertainty_modules/epistemic.py uncertainty_modules/tests/test_epistemic.py
git commit -m "feat: add epistemic disagreement estimators"
```

### Task 2: Prototype distance

**Files:**
- Modify: `uncertainty_modules/epistemic.py`
- Modify: `uncertainty_modules/tests/test_epistemic.py`

**Interfaces:**
- Consumes: `features: [B,D]`、`prototypes: [K,D]`、`metric: Literal["cosine", "squared_euclidean"]`。
- Produces: 最近距离、索引和名为 `prototype` 的分量分数。

- [ ] **Step 1: 添加两种距离和非法输入的失败测试**

```python
def test_cosine_prototype_distance_returns_nearest_index():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    output = EpistemicUncertaintyModule().from_prototypes(features, prototypes)
    assert torch.allclose(output.uncertainty_score, torch.zeros(2), atol=1e-6)
    assert torch.equal(output.nearest_prototype_index, torch.tensor([0, 2]))


def test_squared_euclidean_prototype_distance():
    features = torch.tensor([[2.0, 0.0]])
    prototypes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    output = EpistemicUncertaintyModule().from_prototypes(features, prototypes, metric="squared_euclidean")
    assert output.nearest_prototype_distance.item() == pytest.approx(1.0)
    assert output.nearest_prototype_index.item() == 1


def test_prototype_distance_rejects_empty_or_mismatched_prototypes():
    module = EpistemicUncertaintyModule()
    with pytest.raises(ValueError):
        module.from_prototypes(torch.randn(2, 4), torch.empty(0, 4))
    with pytest.raises(ValueError):
        module.from_prototypes(torch.randn(2, 4), torch.randn(3, 5))
```

- [ ] **Step 2: 运行 prototype 测试并确认因方法不存在而失败**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: FAIL，错误包含 `has no attribute 'from_prototypes'`。

- [ ] **Step 3: 实现 prototype distance**

```python
from torch.nn import functional as F


def from_prototypes(self, features: torch.Tensor, prototypes: torch.Tensor, metric: str = "cosine") -> EpistemicOutput:
    if features.ndim != 2 or prototypes.ndim != 2 or features.shape[0] == 0 or prototypes.shape[0] == 0:
        raise ValueError("features and prototypes must be non-empty rank-2 tensors")
    if features.shape[1] != prototypes.shape[1] or features.device != prototypes.device or features.dtype != prototypes.dtype:
        raise ValueError("features and prototypes must share feature size, device, and dtype")
    if not features.is_floating_point() or not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
        raise ValueError("features and prototypes must be finite floating-point values")
    if metric == "cosine":
        distances = 1.0 - F.normalize(features, dim=-1, eps=1e-12) @ F.normalize(prototypes, dim=-1, eps=1e-12).transpose(0, 1)
    elif metric == "squared_euclidean":
        distances = torch.cdist(features, prototypes).square()
    else:
        raise ValueError("metric must be 'cosine' or 'squared_euclidean'")
    nearest_distance, nearest_index = distances.min(dim=1)
    return EpistemicOutput(nearest_distance, {"prototype": nearest_distance}, nearest_prototype_distance=nearest_distance, nearest_prototype_index=nearest_index)
```

将该方法放入 `EpistemicUncertaintyModule` 类中。

- [ ] **Step 4: 运行全部 Epistemic 测试**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: 全部 PASS。

- [ ] **Step 5: 提交 Task 2**

```bash
git add uncertainty_modules/epistemic.py uncertainty_modules/tests/test_epistemic.py
git commit -m "feat: add prototype epistemic uncertainty"
```

### Task 3: 校准组合与 OOD 判定

**Files:**
- Modify: `uncertainty_modules/epistemic.py`
- Modify: `uncertainty_modules/tests/test_epistemic.py`

**Interfaces:**
- Consumes: `scores: Mapping[str, Tensor[B]]`、`weights: Mapping[str,float]`、`calibration: Mapping[str,tuple[mean,std]]`、可选 `threshold`。
- Produces: z-score 后的加权均值、原始分量和可选 `is_ood`。

- [ ] **Step 1: 写组合和输入校验失败测试**

```python
def test_combine_calibrates_weights_and_applies_ood_threshold():
    scores = {"ensemble": torch.tensor([1.0, 3.0]), "prototype": torch.tensor([2.0, 6.0])}
    calibration = {"ensemble": (1.0, 2.0), "prototype": (2.0, 4.0)}
    output = EpistemicUncertaintyModule().combine(scores, {"ensemble": 1.0, "prototype": 3.0}, calibration, threshold=0.5)
    assert torch.allclose(output.uncertainty_score, torch.tensor([0.0, 1.0]))
    assert torch.equal(output.is_ood, torch.tensor([False, True]))


@pytest.mark.parametrize(
    "weights,calibration",
    [({"ensemble": -1.0}, {"ensemble": (0.0, 1.0)}), ({"ensemble": 1.0}, {}), ({"ensemble": 1.0}, {"ensemble": (0.0, 0.0)})],
)
def test_combine_rejects_invalid_configuration(weights, calibration):
    with pytest.raises(ValueError):
        EpistemicUncertaintyModule().combine({"ensemble": torch.ones(2)}, weights, calibration)
```

- [ ] **Step 2: 运行测试并确认因 `combine` 不存在而失败**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: FAIL，错误包含 `has no attribute 'combine'`。

- [ ] **Step 3: 实现显式校准组合**

```python
from collections.abc import Mapping


def combine(self, scores: Mapping[str, torch.Tensor], weights: Mapping[str, float], calibration: Mapping[str, tuple[float | torch.Tensor, float | torch.Tensor]], threshold: float | None = None) -> EpistemicOutput:
    if not scores:
        raise ValueError("scores must not be empty")
    if set(scores) != set(weights) or set(scores) != set(calibration):
        raise ValueError("scores, weights, and calibration must have identical keys")
    first = next(iter(scores.values()))
    if first.ndim != 1 or first.shape[0] == 0 or not first.is_floating_point():
        raise ValueError("every score must be a non-empty floating-point [B] tensor")
    weighted = torch.zeros_like(first)
    total_weight = 0.0
    for name, score in scores.items():
        if score.shape != first.shape or score.device != first.device or score.dtype != first.dtype or not torch.isfinite(score).all():
            raise ValueError("all scores must share shape, device, dtype, and be finite")
        weight = float(weights[name])
        if weight < 0:
            raise ValueError("weights must be nonnegative")
        mean, std = calibration[name]
        mean_tensor = torch.as_tensor(mean, dtype=score.dtype, device=score.device)
        std_tensor = torch.as_tensor(std, dtype=score.dtype, device=score.device)
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all() or torch.any(std_tensor <= 0):
            raise ValueError("calibration standard deviations must be finite and positive")
        weighted = weighted + weight * ((score - mean_tensor) / std_tensor)
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")
    combined = weighted / total_weight
    is_ood = None if threshold is None else combined > float(threshold)
    return EpistemicOutput(combined, dict(scores), is_ood=is_ood)
```

将该方法放入 `EpistemicUncertaintyModule` 类中。

- [ ] **Step 4: 运行 Epistemic 全量测试**

Run: `python -m pytest uncertainty_modules/tests/test_epistemic.py -q`

Expected: 全部 PASS，无 warning。

- [ ] **Step 5: 提交 Task 3**

```bash
git add uncertainty_modules/epistemic.py uncertainty_modules/tests/test_epistemic.py
git commit -m "feat: combine calibrated epistemic scores"
```

### Task 4: 公开接口、中文文档与独立包终验

**Files:**
- Modify: `uncertainty_modules/__init__.py`
- Modify: `uncertainty_modules/README.md`

**Interfaces:**
- Produces: `from uncertainty_modules import EpistemicOutput, EpistemicUncertaintyModule`。

- [ ] **Step 1: 先验证公开符号尚未导出**

Run: `python -c "from uncertainty_modules import EpistemicOutput, EpistemicUncertaintyModule"`

Expected: FAIL，缺失至少一个公开符号。

- [ ] **Step 2: 添加公开导出**

```python
from .epistemic import EpistemicUncertaintyModule
from .types import EpistemicOutput

# 将下列名称加入现有 __all__：
"EpistemicOutput",
"EpistemicUncertaintyModule",
```

- [ ] **Step 3: 扩展 README**

README 增加以下可运行形式的示例，并明确多次前向和 prototypes 由调用方产生：

```python
epistemic = EpistemicUncertaintyModule()
ensemble_output = epistemic.from_ensemble(ensemble_embeddings)       # [M,B,D]
dropout_output = epistemic.from_mc_dropout(mc_dropout_embeddings)    # [M,B,D]
prototype_output = epistemic.from_prototypes(features, prototypes)   # [B,D], [K,D]
combined = epistemic.combine(
    {"ensemble": ensemble_output.uncertainty_score, "prototype": prototype_output.uncertainty_score},
    {"ensemble": 1.0, "prototype": 1.0},
    {"ensemble": (ensemble_mean, ensemble_std), "prototype": (prototype_mean, prototype_std)},
    threshold=ood_threshold,
)
```

- [ ] **Step 4: 运行整个独立包的终验**

Run: `python -m pytest uncertainty_modules/tests -q`

Expected: 全部 PASS，无 error 或 warning。

Run: `python -c "import torch; from uncertainty_modules import AleatoricUncertaintyModule, EpistemicUncertaintyModule; a=AleatoricUncertaintyModule(4,8); ao=a(torch.randn(2,3,4)); e=EpistemicUncertaintyModule(); eo=e.from_ensemble(torch.randn(3,2,4)); assert ao.embedding.variance.shape==(2,4) and eo.uncertainty_score.shape==(2,)"`

Expected: 退出码 0。

Run: `git diff --check -- uncertainty_modules`

Expected: 无输出，退出码 0。

- [ ] **Step 5: 提交 Task 4**

```bash
git add uncertainty_modules/__init__.py uncertainty_modules/README.md
git commit -m "docs: document epistemic uncertainty module"
```
