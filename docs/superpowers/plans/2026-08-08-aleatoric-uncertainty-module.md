# Aleatoric Uncertainty Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立 PyTorch 包中实现双分支 Aleatoric 不确定性估计、概率聚合和配套训练损失。

**Architecture:** 模态无关模块接收 `[B,N,D]` 特征，由 relevance 与 uncertainty 两套独立 MLP 分别预测语义权重和 token 对角方差。可靠性在进入 pooling 前 detach；输出包含有界对角方差的概率嵌入和可解释的局部、全局及聚合权重。

**Tech Stack:** Python 3、PyTorch、pytest

## Global Constraints

- 只新增顶层 `uncertainty_modules` 包，不读取、导入或修改项目现有检索代码。
- 只依赖 PyTorch 与 Python 标准库。
- `mask=True` 表示有效 token；任一样本全空时抛出 `ValueError`。
- 视频和文本通过同一类的不同实例使用，参数不共享。
- 先测试失败，再写最小实现；每个任务结束时运行该任务的精确测试。

---

## 文件职责

- `uncertainty_modules/types.py`：不可变输出数据类。
- `uncertainty_modules/aleatoric.py`：双分支预测、mask、可靠性聚合和对角方差传播。
- `uncertainty_modules/losses.py`：ranking、semantic consistency 和 variance prior 损失。
- `uncertainty_modules/__init__.py`：稳定的公开导入接口。
- `uncertainty_modules/tests/test_aleatoric.py`：Aleatoric 行为、校验和梯度隔离测试。
- `uncertainty_modules/tests/test_losses.py`：三个独立损失的行为测试。
- `uncertainty_modules/README.md`：独立使用示例和张量约定。

### Task 1: 结果类型、基本前向和输入校验

**Files:**
- Create: `uncertainty_modules/types.py`
- Create: `uncertainty_modules/aleatoric.py`
- Create: `uncertainty_modules/tests/test_aleatoric.py`

**Interfaces:**
- Consumes: `features: torch.Tensor[B,N,D]`、可选 `mask: torch.BoolTensor[B,N]`。
- Produces: `ProbabilisticEmbedding`、`AleatoricOutput`、`AleatoricUncertaintyModule.forward(...) -> AleatoricOutput`。

- [ ] **Step 1: 写结果形状和构造参数的失败测试**

```python
import pytest
import torch

from uncertainty_modules.aleatoric import AleatoricUncertaintyModule
from uncertainty_modules.types import AleatoricOutput, ProbabilisticEmbedding


def test_forward_returns_documented_shapes():
    module = AleatoricUncertaintyModule(4, 8, min_variance=1e-4, max_variance=2.0)
    output = module(torch.randn(2, 3, 4))
    assert isinstance(output, AleatoricOutput)
    assert isinstance(output.embedding, ProbabilisticEmbedding)
    assert output.embedding.mean.shape == (2, 4)
    assert output.embedding.variance.shape == (2, 4)
    assert output.embedding.uncertainty_score.shape == (2,)
    assert output.local_uncertainty.shape == (2, 3)
    assert output.global_uncertainty.shape == (2,)
    assert output.relevance_weights.shape == (2, 3)
    assert output.aggregation_weights.shape == (2, 3)


@pytest.mark.parametrize("args", [(0, 8, 1e-4, 2.0), (4, 0, 1e-4, 2.0), (4, 8, 0.0, 2.0), (4, 8, 2.0, 2.0)])
def test_constructor_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        AleatoricUncertaintyModule(*args)
```

- [ ] **Step 2: 运行测试并确认因模块尚不存在而失败**

Run: `python -m pytest uncertainty_modules/tests/test_aleatoric.py -q`

Expected: FAIL，错误包含 `ModuleNotFoundError` 或缺失类名。

- [ ] **Step 3: 实现不可变结果类型与最小前向**

```python
# uncertainty_modules/types.py
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ProbabilisticEmbedding:
    mean: torch.Tensor
    variance: torch.Tensor
    uncertainty_score: torch.Tensor


@dataclass(frozen=True)
class AleatoricOutput:
    embedding: ProbabilisticEmbedding
    local_uncertainty: torch.Tensor
    global_uncertainty: torch.Tensor
    relevance_weights: torch.Tensor
    aggregation_weights: torch.Tensor
```

```python
# uncertainty_modules/aleatoric.py
import torch
from torch import nn
from torch.nn import functional as F

from .types import AleatoricOutput, ProbabilisticEmbedding


class AleatoricUncertaintyModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, min_variance: float = 1e-6, max_variance: float = 10.0):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if min_variance <= 0 or max_variance <= min_variance:
            raise ValueError("variance bounds must satisfy 0 < min_variance < max_variance")
        self.input_dim = input_dim
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.relevance_head = self._head(input_dim, hidden_dim, 1)
        self.uncertainty_head = self._head(input_dim, hidden_dim, input_dim)
        self.global_uncertainty_head = self._head(input_dim, hidden_dim, input_dim)

    @staticmethod
    def _head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def _bounded_variance(self, raw: torch.Tensor) -> torch.Tensor:
        positive = F.softplus(raw)
        unit = positive / (1.0 + positive)
        return self.min_variance + (self.max_variance - self.min_variance) * unit

    def _validate(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim or not features.is_floating_point():
            raise ValueError("features must be floating point with shape [B, N, input_dim]")
        if mask is None:
            return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.dtype is not torch.bool or mask.shape != features.shape[:2] or mask.device != features.device:
            raise ValueError("mask must be boolean [B, N] on the features device")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every sample must contain at least one valid element")
        return mask

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> AleatoricOutput:
        mask = self._validate(features, mask)
        relevance_logits = self.relevance_head(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        relevance = torch.softmax(relevance_logits, dim=1)
        token_variance = self._bounded_variance(self.uncertainty_head(features))
        local = token_variance.mean(dim=-1)
        reliability = torch.exp(-local.detach()) * mask.to(features.dtype)
        aggregation = relevance * reliability
        aggregation = aggregation / aggregation.sum(dim=1, keepdim=True)
        mean = torch.sum(aggregation.unsqueeze(-1) * features, dim=1)
        global_variance = self._bounded_variance(self.global_uncertainty_head(mean))
        propagated = torch.sum(aggregation.unsqueeze(-1).square() * token_variance, dim=1)
        total_positive = propagated + global_variance
        total_variance = self.min_variance + (self.max_variance - self.min_variance) * total_positive / (1.0 + total_positive)
        embedding = ProbabilisticEmbedding(mean, total_variance, total_variance.mean(dim=-1))
        return AleatoricOutput(embedding, local, global_variance.mean(dim=-1), relevance, aggregation)
```

- [ ] **Step 4: 运行 Task 1 测试并确认通过**

Run: `python -m pytest uncertainty_modules/tests/test_aleatoric.py -q`

Expected: `5 passed`。

- [ ] **Step 5: 提交 Task 1**

```bash
git add uncertainty_modules/types.py uncertainty_modules/aleatoric.py uncertainty_modules/tests/test_aleatoric.py
git commit -m "feat: add aleatoric uncertainty forward pass"
```

### Task 2: Mask、数值边界和梯度隔离

**Files:**
- Modify: `uncertainty_modules/tests/test_aleatoric.py`
- Modify: `uncertainty_modules/aleatoric.py`

**Interfaces:**
- Consumes: Task 1 的 `AleatoricUncertaintyModule`。
- Produces: 严格 mask 行为、有限有界方差、device/dtype 保持和分支隔离保证。

- [ ] **Step 1: 添加失败测试**

```python
def test_masked_positions_are_zero_and_valid_weights_sum_to_one():
    module = AleatoricUncertaintyModule(4, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    output = module(torch.randn(2, 3, 4), mask)
    assert torch.equal(output.relevance_weights[~mask], torch.zeros(3))
    assert torch.equal(output.aggregation_weights[~mask], torch.zeros(3))
    assert torch.allclose(output.relevance_weights.sum(1), torch.ones(2))
    assert torch.allclose(output.aggregation_weights.sum(1), torch.ones(2))


def test_variance_is_finite_and_bounded():
    module = AleatoricUncertaintyModule(4, 8, min_variance=0.1, max_variance=0.9)
    output = module(torch.randn(2, 3, 4) * 1e4)
    assert torch.isfinite(output.embedding.variance).all()
    assert torch.all(output.embedding.variance >= 0.1)
    assert torch.all(output.embedding.variance <= 0.9)


def test_reliability_does_not_send_mean_loss_gradient_to_uncertainty_head():
    module = AleatoricUncertaintyModule(4, 8)
    module(torch.randn(2, 3, 4)).embedding.mean.sum().backward()
    assert any(parameter.grad is not None for parameter in module.relevance_head.parameters())
    assert all(parameter.grad is None for parameter in module.uncertainty_head.parameters())


def test_rejects_all_empty_mask():
    module = AleatoricUncertaintyModule(4, 8)
    with pytest.raises(ValueError, match="at least one valid"):
        module(torch.randn(2, 3, 4), torch.tensor([[True, False, False], [False, False, False]]))


def test_preserves_float64_dtype():
    module = AleatoricUncertaintyModule(4, 8).double()
    output = module(torch.randn(2, 1, 4, dtype=torch.float64))
    assert output.embedding.mean.dtype == torch.float64
    assert output.embedding.variance.dtype == torch.float64


def test_rejects_nonfinite_features():
    module = AleatoricUncertaintyModule(4, 8)
    features = torch.randn(2, 3, 4)
    features[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        module(features)
```

- [ ] **Step 2: 运行新增测试并确认至少一个测试因边界或异常消息不符而失败**

Run: `python -m pytest uncertainty_modules/tests/test_aleatoric.py -q`

Expected: `test_rejects_nonfinite_features` FAIL，因为 Task 1 尚未校验有限值；确认其他失败（如有）来自约定行为而不是测试语法。

- [ ] **Step 3: 按测试修正实现**

```python
# 在 _validate 开头补充空 batch/序列校验，并在 forward 中保证无效位置显式归零：
if features.shape[0] == 0 or features.shape[1] == 0:
    raise ValueError("features must contain a non-empty batch and sequence")

relevance = torch.softmax(relevance_logits, dim=1)
relevance = relevance.masked_fill(~mask, 0.0)

if not torch.isfinite(features).all():
    raise ValueError("features must contain only finite values")
```

- [ ] **Step 4: 运行全部 Aleatoric 测试**

Run: `python -m pytest uncertainty_modules/tests/test_aleatoric.py -q`

Expected: 全部 PASS，且无 warning。

- [ ] **Step 5: 提交 Task 2**

```bash
git add uncertainty_modules/aleatoric.py uncertainty_modules/tests/test_aleatoric.py
git commit -m "test: cover aleatoric stability and gradient isolation"
```

### Task 3: Aleatoric 训练损失

**Files:**
- Create: `uncertainty_modules/losses.py`
- Create: `uncertainty_modules/tests/test_losses.py`

**Interfaces:**
- Produces: `uncertainty_ranking_loss(clean, degraded, margin=0.1)`、`semantic_consistency_loss(clean_mean, degraded_mean)`、`variance_prior_loss(variance, target_log_variance)`。

- [ ] **Step 1: 写三个损失的失败测试**

```python
import pytest
import torch

from uncertainty_modules.losses import semantic_consistency_loss, uncertainty_ranking_loss, variance_prior_loss


def test_ranking_loss_rewards_higher_degraded_uncertainty():
    clean = torch.tensor([0.2, 0.4])
    assert uncertainty_ranking_loss(clean, torch.tensor([0.5, 0.7]), margin=0.1) < uncertainty_ranking_loss(clean, torch.tensor([0.1, 0.2]), margin=0.1)


def test_semantic_consistency_is_zero_for_identical_embeddings():
    embedding = torch.randn(3, 4)
    assert semantic_consistency_loss(embedding, embedding).item() == pytest.approx(0.0)


def test_variance_prior_is_zero_at_target():
    variance = torch.full((2, 4), 0.5)
    target = torch.log(torch.tensor(0.5))
    assert variance_prior_loss(variance, target).item() == pytest.approx(0.0)


def test_variance_prior_rejects_nonpositive_variance():
    with pytest.raises(ValueError, match="positive"):
        variance_prior_loss(torch.tensor([0.0]), 0.0)
```

- [ ] **Step 2: 运行测试并确认因 `losses` 模块不存在而失败**

Run: `python -m pytest uncertainty_modules/tests/test_losses.py -q`

Expected: FAIL，错误包含 `ModuleNotFoundError`。

- [ ] **Step 3: 实现最小损失函数**

```python
import torch
from torch.nn import functional as F


def _same_shape(left: torch.Tensor, right: torch.Tensor, names: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{names} must have the same shape")


def uncertainty_ranking_loss(clean: torch.Tensor, degraded: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    _same_shape(clean, degraded, "clean and degraded")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    return F.relu(float(margin) + clean - degraded).mean()


def semantic_consistency_loss(clean_mean: torch.Tensor, degraded_mean: torch.Tensor) -> torch.Tensor:
    _same_shape(clean_mean, degraded_mean, "clean_mean and degraded_mean")
    return F.mse_loss(clean_mean, degraded_mean)


def variance_prior_loss(variance: torch.Tensor, target_log_variance: float | torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError("variance must be finite and positive")
    target = torch.as_tensor(target_log_variance, dtype=variance.dtype, device=variance.device)
    return torch.mean((torch.log(variance) - target).square())
```

- [ ] **Step 4: 运行损失测试并确认通过**

Run: `python -m pytest uncertainty_modules/tests/test_losses.py -q`

Expected: `4 passed`。

- [ ] **Step 5: 提交 Task 3**

```bash
git add uncertainty_modules/losses.py uncertainty_modules/tests/test_losses.py
git commit -m "feat: add aleatoric uncertainty losses"
```

### Task 4: 公开接口、中文文档与 Aleatoric 验证

**Files:**
- Create: `uncertainty_modules/__init__.py`
- Create: `uncertainty_modules/README.md`
- Modify: `uncertainty_modules/tests/test_aleatoric.py`

**Interfaces:**
- Produces: 从 `uncertainty_modules` 直接导入全部 Aleatoric 公共 API。

- [ ] **Step 1: 添加公开导入和双模态独立实例 smoke test**

```python
def test_video_and_text_instances_are_independent():
    video_module = AleatoricUncertaintyModule(4, 8)
    text_module = AleatoricUncertaintyModule(4, 8)
    assert next(video_module.parameters()) is not next(text_module.parameters())
    assert video_module(torch.randn(2, 5, 4)).embedding.mean.shape == (2, 4)
    assert text_module(torch.randn(2, 7, 4)).embedding.mean.shape == (2, 4)
```

- [ ] **Step 2: 运行测试并确认新增公开导入尚未满足**

Run: `python -c "from uncertainty_modules import AleatoricUncertaintyModule, ProbabilisticEmbedding, uncertainty_ranking_loss"`

Expected: FAIL，缺失至少一个公开符号。

- [ ] **Step 3: 新增公开导出与 README 示例**

```python
# uncertainty_modules/__init__.py
from .aleatoric import AleatoricUncertaintyModule
from .losses import semantic_consistency_loss, uncertainty_ranking_loss, variance_prior_loss
from .types import AleatoricOutput, ProbabilisticEmbedding

__all__ = [
    "AleatoricOutput",
    "AleatoricUncertaintyModule",
    "ProbabilisticEmbedding",
    "semantic_consistency_loss",
    "uncertainty_ranking_loss",
    "variance_prior_loss",
]
```

README 必须记录 `[B,N,D]`、`mask=True`、video/text 分开实例化、三个损失的用途，以及“尚未接入主流程”的边界，并包含下列可运行示例：

```python
video_uncertainty = AleatoricUncertaintyModule(input_dim=512, hidden_dim=256)
text_uncertainty = AleatoricUncertaintyModule(input_dim=512, hidden_dim=256)
video_output = video_uncertainty(video_tokens, video_mask)
text_output = text_uncertainty(text_tokens, text_mask)
```

- [ ] **Step 4: 运行 Aleatoric 全量验证**

Run: `python -m pytest uncertainty_modules/tests/test_aleatoric.py uncertainty_modules/tests/test_losses.py -q`

Expected: 全部 PASS，无 error 或 warning。

Run: `python -c "import torch; from uncertainty_modules import AleatoricUncertaintyModule; m=AleatoricUncertaintyModule(4,8); o=m(torch.randn(2,3,4)); (o.embedding.mean.sum()+o.embedding.variance.sum()).backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)"`

Expected: 退出码 0。

- [ ] **Step 5: 提交 Task 4**

```bash
git add uncertainty_modules/__init__.py uncertainty_modules/README.md uncertainty_modules/tests/test_aleatoric.py
git commit -m "docs: document standalone aleatoric module"
```
