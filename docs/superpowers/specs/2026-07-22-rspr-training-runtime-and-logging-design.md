# RSPR 端到端训练与科研最小工程修订设计

**日期：** 2026-07-22  
**状态：** 激进精简方案已确认，待书面复核

## 1. 目标

本修订统一 RSPR 核心版的端到端训练语义，固定后台训练使用的 Python
环境，并以真实实验能否跑通为主要验收依据，避免为科研代码增加生产级校验、
测试和诊断框架。

修订只覆盖以下范围：

1. 从 OpenAI CLIP 预训练权重开始，在一次作业内端到端训练 CLIP 可训练层、
   DSA、WTI 和 RSPR，不依赖已训练的 UATVR 全模型 checkpoint；
2. 训练和评估入口显式使用 `tvr` 环境中的 Python 与 `torchrun`；
3. 精简启动日志和 step 日志，只保留判断训练是否正常所需的 RSPR 数值；
4. 将“科研实验优先、最小工程校验”写入 `AGENT.md`。

不修改 RSPR 分布、匹配、排序、损失权重、Top-R 算法、数据协议或 A0–A8
参数映射。

## 2. 初始化与端到端训练语义

### 2.1 初始化起点

训练使用项目配置的 OpenAI CLIP 预训练权重初始化 CLIP。DSA、WTI 权重头和
RSPR 概率头按当前模型初始化规则创建，随后在同一个训练作业中联合优化。

规范的首轮训练不传入 `--init_model`。该参数不表示 RSPR 的固定前置阶段，只保留
给用户显式指定模型权重，包括外部初始化或中断续训；对应 optimizer 状态仍由
现有 `--resume_model` 参数加载。

### 2.2 单次端到端训练

RSPR 核心配置的固定约束为：

- `rspr_mode=stochastic`；
- `RSPR_FREEZE_CLIP=0`、`RSPR_FREEZE_DSA=0`；
- `FREEZE_LAYER_NUM=8`，冻结 CLIP 前 8 个 Transformer block，最后 4 个 block
  及既有允许训练的归一化、投影参数使用 `coef_lr` 控制的低学习率；
- DSA、WTI 和 RSPR 从第一个 optimizer step 起可训练；
- DSA loss 与 probability loss 从第一个 step 起使用完整目标权重；
- stochastic rank loss 与 anchor KL 在第一个 epoch 内从 0 线性增长到目标权重；
- 默认在同一个 optimizer、同一个学习率计划中连续训练 5 个 epoch；
- 不生成或加载用于阶段衔接的中间 checkpoint。

`rspr_warmup_epochs=1` 只表示单次训练内部的损失权重 warm-up，不表示模型冻结
状态切换，也不重建 optimizer。若后续 GPU smoke 发现第一个 epoch 数值不稳定，
应先依据日志调整损失权重或 warm-up 长度，不默认恢复两次独立训练。

### 2.3 采用单次训练的依据

RSPR 概率均值头以确定性聚合中心为残差起点，最后一层零初始化；`logvar` 从
`prior_std` 对应的先验值开始并受固定上下界约束。因此概率旁路不是任意尺度的
随机输出。与此同时，CLIP 只开放后部 block 且使用较低学习率，排序与锚定损失
已有训练内 warm-up。上述保护已经覆盖原 Stage A 的主要稳定作用，无需用两次
作业和一次 optimizer 重启实现同一目的。

单次训练还保证所有消融共享连续的优化器状态、学习率计划和总训练步数，避免
Stage A checkpoint 选择及阶段切换成为额外实验变量。

### 2.4 A0–A8 与 A4 legacy

A0–A8 使用相同 OpenAI CLIP 权重起点、相同数据划分、总 epoch 数和优化日程，
只改变消融矩阵定义的组件或权重。不得为某一消融额外执行概率头预热或加载阶段
checkpoint。

`rspr_mode=legacy` 继续表示旧 UATVR 概率分支、MIL 和 KL 损失结构的消融。A4 与
其他消融一样从相同 CLIP 权重起点训练，不要求历史 UATVR 全模型 checkpoint
兼容性。新 RSPR 损失不得与 legacy MIL/KL 混合。

## 3. `tvr` 运行环境

### 3.1 解释器选择

公共入口默认使用：

```text
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun
```

调用者可以用同名环境变量覆盖路径。脚本直接执行这两个绝对路径，不依赖
非交互后台 worker 中的 `conda activate` 或调用者的 `PATH`。

训练入口和评估入口共享这一约定，避免训练成功但独立评估落入另一个 Python
环境。

### 3.2 最小运行时接入

训练和评估脚本直接定义上述两个变量，并分别用它们替换当前的 `python3` 和
`torchrun` 命令。不新增公共 runtime helper、环境目录一致性检查、依赖 import
preflight 或 controller/worker 运行时传递框架。

若路径、PyTorch 或 OpenCV 不可用，命令按 Bash `set -e` 的现有行为直接失败并
在训练日志中保留原始错误。科研环境固定且由当前用户控制，不为其他部署环境
增加兼容层。

worker 使用 `TVR_PYTHON` 构建 trusted split，并使用 `TVR_TORCHRUN` 启动训练；
`eval.sh` 使用相同变量。

## 4. 日志设计

### 4.1 启动日志

保留现有四个分组：

- `Training`；
- `Model`；
- `RSPR`；
- `Protocol`。

删除未筛选的 `[Other]` 全参数倾倒。输出仍保留实验 manifest，完整可复现参数
以 `experiment_manifest.json` 为事实源。

训练入口额外打印一行运行时摘要：

```text
[Runtime] python=<path> torchrun=<path>
```

### 4.2 step 日志

每 `n_display` 个 optimizer step 输出一行，不增加额外日志行。RSPR
`mean/stochastic` 模式的格式包含：

```text
loss=<total> [dsa=<v> prob=<v> rank=<v> anchor=<v>]
u_pair=<mean>
variance_t=<mean> variance_v=<mean>
lr=<clip>/<other>
time=<seconds> eta=<duration>
```

`off/legacy` 模式省略整个 RSPR 诊断片段，不输出零值或占位符。

这些字段全部复用 `UATVR._assemble_training_loss` 已经生成的
`last_loss_diagnostics`。不得为日志新增分位数计算、额外 forward、随机采样或
全矩阵概率匹配。

### 4.3 epoch 日志

保留现有 epoch 平均 loss 与耗时行，不新增 RSPR epoch 摘要、分位数、跨 batch
聚合器或独立诊断文件。完整的不确定性分析在正式实验结果阶段按需要离线计算，
不进入训练热路径。

### 4.4 计算位置

`UATVR._assemble_training_loss` 继续生成现有无梯度诊断：

- `dsa`、`prob`、`rank`、`anchor`；
- `pair_uncertainty_mean`；
- `text_variance_mean`、`video_variance_mean`。

`train_epoch` 只负责格式化当前 step。日志代码只读取 detached 值，不参与总损失
或反向传播。

## 5. 科研最小工程原则

`AGENT.md` 增加“科研项目工作原则”，内容固定为：

- 本仓库是科研实验项目，首要目标是让假设能够快速、可复现地进入真实训练与
  评估；不按生产服务标准扩展工程设施；
- 对 shell 参数、日志格式、文档和局部配置等低风险改动，默认不新增测试矩阵、
  抽象层、兼容层或重复 preflight；
- 对核心数学公式、梯度路径、数据划分、正负样本语义和 checkpoint 兼容性仍做
  必要验证，因为这些错误会直接使实验结论失效；
- 验收优先使用最小真实训练/评估 smoke；只有出现可复现故障，或用户明确要求，
  才补充针对性测试和诊断；
- 日志只保留判断 loss、学习率、速度和关键研究变量是否正常所需的信息，不为
  可能的未来分析预先增加统计。

这项原则约束后续 Agent 的默认行为，但不覆盖用户在具体任务中明确要求的测试、
审计或生产级可靠性工作。

## 6. 文档同步

以下事实源必须同步：

- RSPR 设计文档的端到端初始化与训练流程；
- RSPR 实施计划中的 global constraints、训练命令、A4 和 final checklist；
- Stage 1 实验协议中的单次训练命令及运行环境前置条件。

文档不得继续把 Stage A/B 作为固定训练协议，也不得要求任一消融加载历史 UATVR
全模型 checkpoint。`--init_model` 不得用于隐式阶段衔接。

## 7. 验收方式

本修订不新增自动化测试、文档契约测试、runtime 失败组合测试或诊断聚合测试。
若已有测试仅因规范默认值或日志文本发生预期变化，可最小更新其期望值，不扩大
覆盖范围。

工程修改完成后直接启动规范 A3 训练。观察到第一个 `n_display=20` optimizer
step 日志后确认：

1. 进程在 `tvr` 环境成功启动并完成 split 构建、模型初始化、forward、backward
   和 optimizer step；
2. total、dsa、prob、rank、anchor 与 `U_pair` 均为有限值；
3. 日志显示 CLIP/other 两组学习率、step 时间和 ETA；
4. 未出现 CUDA OOM、NaN/Inf 或 traceback。

smoke 使用独立 `RUN_ID`、`OUTPUT_DIR` 和 `TRAIN_PID_FILE`，不得覆盖正式实验。
达到首个 20-step 日志后，向 PID 文件记录的 worker 进程组发送 `TERM` 并确认其
退出，不继续完整 5-epoch 训练；正式实验由用户另行启动。若 smoke 失败，针对
实际错误做最小诊断，不预先建立通用测试框架。
