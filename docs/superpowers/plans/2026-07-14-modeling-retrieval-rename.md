# UATVR Retrieval Model Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将活动检索模型从历史误拼写路径 `modules.modeling_mulit` 彻底迁移到 `modules.modeling_retrieval`，且不改变模型行为或 checkpoint 参数键。

**Architecture:** 采用无兼容层的直接模块重命名。测试先声明新模块存在、旧模块不存在，再移动活动实现并更新唯一生产导入；随后同步测试文件名和项目文档，最后用引用扫描、完整测试与静态检查证明迁移闭合。

**Tech Stack:** Python 3、PyTorch、pytest、Ruff、Bash、Git

## Global Constraints

- 不保留 `modules/modeling_mulit.py` 兼容转发文件。
- 新活动路径固定为 `modules/modeling_retrieval.py`。
- 不修改 `UATVR` 类名、公开方法签名、模型结构、forward、WTI、loss、数据协议、训练参数或 state dict 参数名。
- 不删除或重构 `modules/modeling.py` 与 `prob_models/`。
- 保留工作树中已有的无关改动，尤其是 `modules/backbone_adapter.py`、`tests/test_backbone_adapter.py` 以及用户删除的历史文档。
- 不启动训练进程。

---

### Task 1: 以测试驱动完成模块路径迁移

**Files:**
- Rename: `modules/modeling_mulit.py` to `modules/modeling_retrieval.py`
- Rename: `tests/test_modeling_mulit_losses.py` to `tests/test_modeling_retrieval.py`
- Modify: `main_task_retrieval.py:28`
- Test: `tests/test_modeling_retrieval.py`

**Interfaces:**
- Consumes: 现有 `modules.modeling_mulit.UATVR` 类及其全部方法和 state dict 参数名。
- Produces: `modules.modeling_retrieval.UATVR`，类定义与行为保持不变；`modules.modeling_mulit` 不再存在。

- [ ] **Step 1: 在现有测试文件中加入模块命名契约测试**

在 `tests/test_modeling_mulit_losses.py` 的导入区保留现有 `import importlib`，加入：

```python
def test_active_model_module_uses_retrieval_name():
    assert importlib.util.find_spec("modules.modeling_retrieval") is not None
    assert importlib.util.find_spec("modules.modeling_mulit") is None
```

- [ ] **Step 2: 运行契约测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_modeling_mulit_losses.py::test_active_model_module_uses_retrieval_name
```

Expected: FAIL；`modules.modeling_retrieval` 尚不存在，且旧模块仍可发现。

- [ ] **Step 3: 移动活动模型和测试文件**

使用补丁完成两个纯移动：

```text
modules/modeling_mulit.py -> modules/modeling_retrieval.py
tests/test_modeling_mulit_losses.py -> tests/test_modeling_retrieval.py
```

移动时不修改 `UATVR` 实现内容。

- [ ] **Step 4: 更新生产和测试导入**

将 `main_task_retrieval.py` 改为：

```python
from modules.modeling_retrieval import UATVR
```

将 `tests/test_modeling_retrieval.py` 的模型加载改为：

```python
UATVR = importlib.import_module("modules.modeling_retrieval").UATVR
```

- [ ] **Step 5: 运行模块迁移定向测试并确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_modeling_retrieval.py::test_active_model_module_uses_retrieval_name
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_modeling_retrieval.py
```

Expected: 契约测试 PASS，随后整个模型测试文件 PASS。

- [ ] **Step 6: 检查本任务差异只包含路径迁移和导入更新**

Run:

```bash
git diff --stat -- modules/modeling_mulit.py modules/modeling_retrieval.py tests/test_modeling_mulit_losses.py tests/test_modeling_retrieval.py main_task_retrieval.py
git diff --find-renames --summary
```

Expected: Git 将两个大文件识别为高相似度 rename；`main_task_retrieval.py` 只有一行导入路径变化，测试除契约测试和导入路径外无行为变化。

- [ ] **Step 7: 提交模块迁移**

```bash
git add main_task_retrieval.py modules/modeling_mulit.py modules/modeling_retrieval.py tests/test_modeling_mulit_losses.py tests/test_modeling_retrieval.py
git commit -m "refactor: rename active retrieval model module"
```

### Task 2: 同步项目事实源和代理入口文档

**Files:**
- Modify: `AGENTS.md:25,39,54`
- Modify: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md:38`
- Modify: `docs/superpowers/specs/2026-07-14-modeling-retrieval-rename-design.md`

**Interfaces:**
- Consumes: Task 1 产生的活动路径 `modules/modeling_retrieval.py`。
- Produces: 所有当前项目文档只将 `modules/modeling_retrieval.py` 标记为主模型；设计文档保留旧名称作为迁移历史说明。

- [ ] **Step 1: 更新活动链路和主模型路径**

将 `AGENTS.md` 中的活动入口和表格改为：

```text
run_train_msrvtt_bg.sh -> train_msrvtt.sh -> main_task_retrieval.py -> modules/modeling_retrieval.py
```

并将稳定事实改为：

```text
当前主模型文件为 modeling_retrieval.py；历史拼写 modeling_mulit.py 已彻底移除，不保留兼容入口。
```

将 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 的 P0 主入口路径同步为 `modules/modeling_retrieval.py`。

- [ ] **Step 2: 将已批准设计文档更新为完成态路径说明**

在设计文档的背景中保留旧路径作为历史事实，并增加：

```text
活动主模型现位于 modules/modeling_retrieval.py；modules/modeling_mulit.py 仅作为迁移前历史名称出现。
```

- [ ] **Step 3: 扫描旧名称引用**

Run:

```bash
rg -n "modeling_mulit" AGENTS.md main_task_retrieval.py modules tests docs/project docs/superpowers/specs/2026-07-14-modeling-retrieval-rename-design.md
```

Expected: 只有已批准设计文档中的迁移历史和测试中的“旧路径不可导入”负向契约允许命中；活动源码、`AGENTS.md` 与项目 SSOT 不得命中。

- [ ] **Step 4: 提交文档同步**

```bash
git add AGENTS.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md docs/superpowers/specs/2026-07-14-modeling-retrieval-rename-design.md
git commit -m "docs: update active retrieval model path"
```

### Task 3: 完整验证迁移闭合

**Files:**
- Verify: `main_task_retrieval.py`
- Verify: `modules/modeling_retrieval.py`
- Verify: `tests/test_modeling_retrieval.py`
- Verify: `AGENTS.md`
- Verify: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`

**Interfaces:**
- Consumes: Task 1–2 的代码与文档迁移结果。
- Produces: 新模块路径可用、旧路径不存在、项目测试和静态检查通过的验证证据。

- [ ] **Step 1: 验证 Python 模块发现语义**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/python - <<'PY'
import importlib.util

assert importlib.util.find_spec("modules.modeling_retrieval") is not None
assert importlib.util.find_spec("modules.modeling_mulit") is None
print("module rename contract: ok")
PY
```

Expected: 输出 `module rename contract: ok`。

- [ ] **Step 2: 运行完整项目测试**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests
```

Expected: 所有 `tests/` 测试通过；不得运行根目录无范围的 `pytest -q`。

- [ ] **Step 3: 运行静态检查**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/ruff check main_task_retrieval.py modules/modeling_retrieval.py tests/test_modeling_retrieval.py
```

Expected: Ruff 退出码 0。

- [ ] **Step 4: 检查补丁完整性和用户改动隔离**

Run:

```bash
git diff --check HEAD~2..HEAD
git status --short
git log -3 --oneline
```

Expected: 无 whitespace error；用户原有 `modules/backbone_adapter.py`、`tests/test_backbone_adapter.py` 和历史文档删除仍保持原状态，没有被迁移提交吸收。
