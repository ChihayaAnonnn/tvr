# UATVR 主检索模型文件重命名设计

## 背景

当前活动主模型位于 `modules/modeling_mulit.py`。文件名中的 `mulit` 是历史拼写错误，无法准确表达该模块承担的文本—视频检索模型职责，也容易与未被主入口使用的历史文件 `modules/modeling.py` 混淆。

主训练与评测入口 `main_task_retrieval.py` 只使用 `modules.modeling_mulit.UATVR`。后续 P0/P2/P3 工作会继续修改这一活动模型，因此应在继续开发前完成一次纯命名迁移。

## 决策

将活动模型文件彻底重命名为：

```text
modules/modeling_retrieval.py
```

不保留 `modules/modeling_mulit.py` 兼容转发文件。迁移完成后，旧 Python 模块路径 `modules.modeling_mulit` 必须不可导入。

选择 `modeling_retrieval.py` 而不是 `modeling_wti.py`，因为文件职责是承载项目的活动检索模型；当前虽然是 WTI-only P0，后续仍可能在独立规格下加入 uncertainty-aware matching 或 candidate-conditioned alignment。选择它而不是 `modeling_uatvr.py`，是因为职责命名比项目名重复更清晰。

## 迁移范围

本次只进行结构性重命名：

1. 将 `modules/modeling_mulit.py` 移动为 `modules/modeling_retrieval.py`。
2. 将 `main_task_retrieval.py` 的模型导入改为 `modules.modeling_retrieval`。
3. 将 `tests/test_modeling_mulit_losses.py` 重命名为 `tests/test_modeling_retrieval.py`，并更新其导入路径。
4. 更新 `AGENTS.md` 与 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 中的活动模型路径和历史拼写说明。
5. 检查受版本控制文件中不再残留 `modeling_mulit` 引用。

以下内容不在本次范围内：

- 不修改 `UATVR` 类名或公开方法签名。
- 不修改模型结构、forward、WTI、loss、数据协议或训练参数。
- 不修改 state dict 参数名，因此当前活动模型 checkpoint key 不因文件重命名而变化。
- 不删除或重构历史文件 `modules/modeling.py` 与 `prob_models/`。
- 不触碰工作树中已有的无关 backbone 和文档改动。

## 兼容与失败语义

本次采用显式破坏旧导入路径的迁移策略：

- 新代码必须使用 `from modules.modeling_retrieval import UATVR`。
- 旧代码若继续导入 `modules.modeling_mulit`，应立即得到 `ModuleNotFoundError`，从而暴露遗漏引用。
- 不提供静默别名、弃用警告期或双入口，避免后续修改落入错误文件。
- checkpoint 序列化保存的是参数键而不是 Python 源文件路径，因此纯文件重命名不迁移参数键；现有 checkpoint 仍受当前 backbone 与退役参数校验约束。

## 测试设计

实施遵循以下验证顺序：

1. RED：先将测试期望改为导入 `modules.modeling_retrieval`，确认在源文件尚未移动时因模块不存在而失败。
2. GREEN：移动活动模型文件并更新生产导入，使定向导入测试通过。
3. 验证 `importlib.util.find_spec("modules.modeling_mulit")` 返回 `None`，证明没有兼容残留。
4. 使用 `rg` 检查受版本控制的源码、脚本、测试和项目文档中不存在 `modeling_mulit`。
5. 运行：

   ```bash
   /home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_modeling_retrieval.py
   /home/xujie/miniconda3/envs/ret/bin/pytest -q tests
   /home/xujie/miniconda3/envs/ret/bin/ruff check main_task_retrieval.py modules/modeling_retrieval.py tests/test_modeling_retrieval.py
   ```

## 完成条件

- 活动模型只存在于 `modules/modeling_retrieval.py`。
- 主训练/评测入口和测试只导入新路径。
- `AGENTS.md` 与科研路线图记录新路径。
- 旧模块路径无法导入且无受版本控制引用残留。
- 定向测试、完整项目测试与静态检查通过。
- 除上述迁移文件外，用户已有工作树改动保持不变。
