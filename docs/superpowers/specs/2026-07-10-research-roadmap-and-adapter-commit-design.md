# 科研路线统一与 EVA Adapter 提交设计

## 目标

在不夸大现有实验结论的前提下，统一 UATVR 的项目状态、实验归档、后续路线与论文表述；同时把当前未提交的 EVA02-CLIP-B/16 adapter 工作树整理为可验证、可回滚的独立提交。

本轮文档归档与 adapter 实现保持为两个语义独立的本地提交：

1. `docs: align research roadmap and current status`（Task 1 文档口径提交）
2. `feat: add EVA-CLIP backbone adapter`

不推送远端，不启动长期训练。

## 文档信息架构

### 1. 主事实源：`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`

该文档成为项目唯一持续维护的科研主文档，负责回答以下问题：

- 当前最可靠的基线和历史最高结果分别是什么；
- 哪些实验结论已经得到日志支持，哪些仍只是弱证据；
- 当前 SAP/不确定性路线的核心矛盾是什么；
- Hard Negative、UACL、global SAP score 与 QC-SAP 为什么终止或冻结；
- EVA backbone 当前完成到什么程度；
- 下一轮实验如何保持公平归因，以及达到什么条件才继续投入；
- 论文目前可以主张什么、不能主张什么。

关键事实采用“已证实 / 弱证据 / 未验证”三档标记，避免把单次高点、未来设想或工程接入写成稳定科研结论。

### 2. 快照入口：`docs/project/STATUS.md`

`STATUS.md` 压缩为一页式快照，只保留：

- 更新时间与当前阶段；
- 关键指标表；
- 已终止、已冻结、进行中三类状态；
- 当前代码与工作树状态；
- 唯一下一步 P0；
- 指向 Roadmap、UACL/HN 归档和日志报告的链接。

该文档不再重复架构长文、历史实验全过程和完整路线论证。

### 3. 兼容入口：`docs/project/plan.md`

保留文件以避免旧链接失效，但删除已经过期的待验证方向列表。文件只说明历史计划已经合并到 Roadmap，并提供当前主线、归档和论文大纲的链接。

### 4. 实验归档：`docs/project/UACL_HARD_NEG_PLAN.md`

该文档从“实施计划”转为“实验归档与边界结论”，保留必要的实现追溯信息，并补齐 UACL 第 4 epoch 最终结果：

| 实验 | Epoch-4 T2V R@1 | Epoch-4 V2T R@1 |
|---|---:|---:|
| random, `w=0.005, kl=0` | 49.3 | 47.3 |
| closest, `w=0.01, kl=1e-4`, seed 42 | 49.2 | 47.6 |
| closest, `w=0.01, kl=1e-4`, seed 43 | 49.4 | 47.2 |
| closest, `w=0.005, kl=0` | 49.0 | 47.0 |

归档结论统一为：UACL 达到过 49.3 门槛，单个 seed 达到 49.4，但没有形成跨配置、跨种子的稳定增益，因此冻结路线；不得再表述为“四组均未达到 49.3”。

## 科研结论口径

### 稳定事实

- legacy B1-only v2 repeat1 的 T2V R@1 为 49.3，仅作历史主线参照；历史 50.0 未稳定复现。
- legacy loss-zero hygiene 的 T2V R@1 为 48.2，仅作历史归因参考，不是真正的 trusted-v1 WTI-only/backbone-only 基线。
- Hard Negative 多条实现路线均未形成稳定 Top-1 收益，主线终止。
- global `prob_mu`、AnchorWTI 与 QC-SAP 的正负分数 gap 接近零，不继续扩展简单 score 融合或 gate sweep。

### 需要限定的事实

- UACL 第 4 epoch 四组结果为 49.3、49.2、49.4、49.0；它达到过门槛，但优势处于噪声级且不稳定，因此冻结而不是按“未达门槛”终止。
- hygiene 48.2 与 B1-only 49.3 的差异不能在非完全匹配配置下全部归因于某一个辅助 loss。
- ret_gap 增长不能作为验证 R@1 提升的替代指标。

### 尚未验证

- EVA02 是否提高检索指标；
- 当前不确定性是否能区分正确与错误 Top-1；
- 逐锚点可学习方差、pair-level uncertainty 与属性双视图是否有效；
- MSVD 上的泛化收益。

## EVA Adapter 提交边界

adapter/code 提交包含：

- `modules/backbone_adapter.py`
- `modules/modeling_mulit.py`
- `main_task_retrieval.py`
- `train_msrvtt.sh`
- `eval.sh`
- `tests/test_backbone_adapter.py`
- `tests/test_main_task_hard_negative_args.py`
- `tests/test_modeling_mulit_losses.py`

该提交同时保留当前工作树中的 QC-SAP 矩形评估诊断修复及相应测试，因为它属于同一批未提交的模型/评估稳定化改动。不得纳入 checkpoint、日志、模型权重或 `ref/` 第三方代码。

Task 1 文档口径提交的范围为：

- `.gitignore`
- `AGENTS.md`
- `docs/README.md`
- `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`
- `docs/project/STATUS.md`
- `docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md`

`docs/superpowers/plans/2026-07-10-trusted-experiment-foundation.md` 是此前独立加入的实施计划，不属于 Task 1。`docs/project/plan.md` 与 `docs/project/UACL_HARD_NEG_PLAN.md` 也不在 Task 1 范围内，不得将它们表述为本次已提交或已更新的归档文件。

## EVA 公平实验设计

EVA 首轮重跑不能只与 4GPU hygiene=48.2 直接比较。梯度累积不会合并每次 forward 的 in-batch 对比矩阵，降低 micro-batch 会改变负样本数量。因此，在 trusted-v1 完成代码实施后，必须使用相同 seed、GPU 数、`batch_size`、`gradient_accumulation_steps` 和每次 forward 全局 batch，分别运行 EVA02 与 OpenAI-CLIP hygiene WTI-only 对照。

首个 seed 只有在相对匹配 OpenAI CLIP 对照出现明确提升时才补第二 seed；EVA 相对匹配对照的变化是首要判断依据，legacy 48.2/49.3 只作次要历史参考。

## 验证与提交门槛

文档验证：

- 在 Task 1 实际提交范围内搜索并清除所有“UACL 第 4 epoch 尚未完成/未达到 49.3”的旧表述；
- 检查 `AGENTS.md`、Roadmap、STATUS 与本设计说明之间不存在相互矛盾的完成度描述；`plan.md` 与 UACL/HN 归档不属于 Task 1，不能记入本次验证覆盖；
- `git diff --check` 无空白错误；
- Markdown 中不出现未解释的占位内容。

代码验证：

- `/home/xujie/miniconda3/envs/ret/bin/pytest -q tests`
- 对本次触及的 Python 文件运行 Ruff，并修复本次提交范围内的错误；
- `bash -n train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh`
- `main_task_retrieval.py --help` 能列出 backbone 和 score 参数；
- 使用本地 EVA02 权重完成单样本文本、图像前向烟测，输出形状正确且数值有限。

提交完成后，工作树应无本任务遗留修改；不执行 push、merge 或长期训练。
