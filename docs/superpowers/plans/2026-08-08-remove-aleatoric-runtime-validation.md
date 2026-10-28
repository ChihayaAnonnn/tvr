# Remove Aleatoric Runtime Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 `AleatoricUncertaintyModule._validate`，让 Aleatoric 前向过程信任固定训练链路的输入契约。

**Architecture:** `forward` 不再校验 feature 或显式 mask，只在 `mask=None` 时创建 `[B,N]` 全有效布尔 mask。构造参数校验、方差数值稳定性和全部正常输入行为保持不变。

**Tech Stack:** Python 3、PyTorch、pytest

## Global Constraints

- 只修改独立的 `uncertainty_modules` 包及其测试和文档，不接入项目主流程。
- 删除 `_validate` 方法，而不是保留空壳或改名后的等价校验。
- 调用方负责保证 `features: [B,N,D]` 和显式 `mask: [B,N]` 正确。
- `mask=None` 时仍自动创建与 features 同 device 的全有效布尔 mask。
- 保留 constructor 对维度、方差上下界和有限性的校验。
- 保留 log-space reliability 聚合与所有内部数值稳定性保护。

---

## 文件职责

- `uncertainty_modules/aleatoric.py`：删除运行时输入校验，并在 `forward` 内创建默认 mask。
- `uncertainty_modules/tests/test_aleatoric.py`：验证 `_validate` 已删除、默认 mask 和正常显式 mask 行为不回归；删除与新契约冲突的异常输入测试。
- `uncertainty_modules/README.md`：明确 Aleatoric `forward` 信任调用方输入。

### Task 1: 删除 Aleatoric 运行时输入校验

**Files:**
- Modify: `uncertainty_modules/aleatoric.py`
- Modify: `uncertainty_modules/tests/test_aleatoric.py`
- Modify: `uncertainty_modules/README.md`

**Interfaces:**
- Consumes: `AleatoricUncertaintyModule.forward(features: Tensor[B,N,D], mask: Tensor[B,N] | None = None)`。
- Produces: 相同的 `AleatoricOutput`；类中不再存在 `_validate`；`mask=None` 时仍使用全有效 mask。

- [ ] **Step 1: 写 `_validate` 必须不存在的失败测试**

在 `uncertainty_modules/tests/test_aleatoric.py` 中加入：

```python
def test_module_has_no_runtime_input_validator():
    module = AleatoricUncertaintyModule(4, 8)

    assert not hasattr(module, "_validate")
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest uncertainty_modules/tests/test_aleatoric.py::test_module_has_no_runtime_input_validator -q`

Expected: FAIL，断言显示当前实例仍具有 `_validate` 属性。

- [ ] **Step 3: 删除方法、内联默认 mask，并同步测试与文档**

从 `uncertainty_modules/aleatoric.py` 完整删除 `_validate` 方法，并将 `forward` 开头：

```python
mask = self._validate(features, mask)
```

替换为：

```python
if mask is None:
    mask = torch.ones(
        features.shape[:2],
        dtype=torch.bool,
        device=features.device,
    )
```

从 `uncertainty_modules/tests/test_aleatoric.py` 删除以下与新契约冲突的测试，保留正常显式 mask、默认 mask、dtype/device 和数值稳定性测试：

```python
def test_rejects_all_empty_mask(): ...
def test_rejects_nonfinite_features(): ...
def test_rejects_empty_batch_or_sequence(shape): ...
```

在 `uncertainty_modules/README.md` 的输入说明后加入：

```markdown
`forward` 信任固定训练链路提供的输入，不执行 shape、有限值或显式 mask 校验；调用方需要保证输入满足上述约定。只有在 `mask=None` 时，模块才会创建全有效 mask。
```

- [ ] **Step 4: 运行完整 Aleatoric 回归与 smoke test**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest uncertainty_modules/tests/test_aleatoric.py uncertainty_modules/tests/test_losses.py -q`

Expected: `25 passed`，无 error 或 warning。

Run: `/home/xujie/.conda/envs/tvr/bin/python -c "import torch; from uncertainty_modules import AleatoricUncertaintyModule; module=AleatoricUncertaintyModule(4,8); assert not hasattr(module,'_validate'); output=module(torch.randn(2,3,4)); (output.embedding.mean.sum()+output.embedding.variance.sum()).backward(); assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters() if parameter.grad is not None)"`

Expected: 退出码 0。

Run: `git diff --check -- uncertainty_modules`

Expected: 无输出，退出码 0。

- [ ] **Step 5: 提交变更**

```bash
git add uncertainty_modules/aleatoric.py uncertainty_modules/tests/test_aleatoric.py uncertainty_modules/README.md
git commit -m "refactor: trust aleatoric input contract"
```
