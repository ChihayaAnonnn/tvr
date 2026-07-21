# RSPR 训练初始化、运行环境与精简日志修订设计

**日期：** 2026-07-22  
**状态：** 待用户确认

## 1. 目标

本修订统一 RSPR 第一阶段的训练初始化语义，固定后台训练使用的 Python
环境，并让日志既能支持数值验收，又不被无关参数和逐步冗余统计淹没。

修订只覆盖以下范围：

1. Stage A 从 OpenAI CLIP 预训练权重初始化，而不是依赖已训练的 UATVR
   全模型 checkpoint；
2. 训练和评估入口显式使用 `tvr` 环境中的 Python 与 `torchrun`；
3. 精简启动日志和 step 日志，补齐 RSPR 不确定性及 `logvar` 诊断。

不修改 RSPR 分布、匹配、排序、损失权重、Top-R 算法、数据协议或 A0–A8
参数映射。

## 2. 初始化语义

### 2.1 Stage A

Stage A 使用项目配置的 OpenAI CLIP 预训练权重初始化 CLIP。DSA、WTI 权重头和
RSPR 概率头按当前模型初始化规则创建，随后共同进入分阶段训练。

Stage A 的固定约束为：

- `rspr_mode=stochastic`；
- CLIP 完全冻结；
- DSA 保持可训练；
- probability loss 与 anchor KL 生效；
- stochastic rank loss 权重为 0；
- 不要求传入 `--init_model`。

### 2.2 Stage B

Stage B 从 Stage A 输出的 checkpoint 初始化，解冻配置允许的 CLIP 层，并启用
stochastic rank loss。`--init_model` 在 Stage B 表示阶段衔接，而不是加载历史
UATVR 基线。

### 2.3 A4 legacy

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

### 3.2 启动前检查

在创建后台 worker 前完成快速检查：

- 两个路径存在且可执行；
- `TVR_PYTHON` 能导入 `torch` 和 `cv2`；
- `TVR_TORCHRUN` 属于同一环境目录，除非调用者显式同时覆盖两个路径。

检查失败时退出码为 2，错误信息包含失败路径和覆盖变量名称。检查不得启动
split 构建、分布式进程或训练。

controller 将解析后的运行时路径显式传给 detached worker；worker 使用
`TVR_PYTHON` 构建 trusted split，并使用 `TVR_TORCHRUN` 启动训练。

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
logvar_t=<mean>/<p95>
logvar_v=<mean>/<p95>
lr=<clip>/<other>
time=<seconds> eta=<duration>
```

`off/legacy` 模式省略整个 RSPR 诊断片段，不输出零值或占位符。

step 统计均来自当前 forward 已生成的 detached tensor；不得触发额外模型
forward、随机采样或全矩阵概率匹配。

### 4.3 epoch 日志

在 epoch 结束时只为 `mean/stochastic` 输出一行完整分布摘要：

```text
[RSPR epoch] logvar_t min/mean/p50/p95/max=...
             logvar_v min/mean/p50/p95/max=...
             u_pair_mean=...
```

epoch 摘要按本 epoch 各 forward 的样本数加权聚合。为控制内存，不保存完整
`logvar` 张量；每个 forward 只向 CPU 累积 detached 标量统计。min/max 取全
epoch 极值，mean、p50 和 p95 为按 batch 样本数加权的诊断估计。字段名称明确
其为训练诊断，不将其表述为校准置信度。

### 4.4 计算位置

`UATVR._assemble_training_loss` 负责生成无梯度诊断：

- `dsa`、`prob`、`rank`、`anchor`；
- `pair_uncertainty_mean`；
- text/video `logvar_min/mean/p50/p95/max`；
- text/video batch size。

`train_epoch` 负责格式化 step 日志和聚合 epoch 摘要。日志代码只读取 detached
值，不参与总损失或反向传播。

## 5. 文档同步

以下事实源必须同步：

- RSPR 设计文档的 Stage A 初始化描述；
- RSPR 实施计划中的 global constraints、Stage A/A4 和 final checklist；
- Stage 1 实验协议中的阶段说明及运行环境前置条件。

文档不得继续要求 Stage A 或 A4 加载历史 UATVR 全模型 checkpoint。

## 6. 测试与验收

实现按测试先行完成，至少覆盖：

1. 默认 runtime 路径指向 `tvr` 环境；
2. `TVR_PYTHON`、`TVR_TORCHRUN` 可成对覆盖；
3. 路径不可执行或依赖导入失败时，在启动 worker/torchrun 前退出 2；
4. controller 将 runtime 选择传给 detached worker；
5. worker 的 split builder 和训练分别使用指定 Python/torchrun；
6. 启动日志不再包含 `[Other]`；
7. RSPR step 日志包含四项未加权 loss、`U_pair` 和两模态 logvar mean/p95；
8. off/legacy 不输出 RSPR 占位字段；
9. epoch 摘要包含两模态 min/mean/p50/p95/max；
10. 日志统计全部 detached，反向梯度和总损失公式保持不变；
11. 文档扫描确认 Stage A 从 CLIP 初始化，且不存在历史 UATVR checkpoint
    前置要求。

最终运行：RSPR 定向测试、训练入口测试、Ruff、Bash 语法检查和完整 CPU 测试。
本修订不启动真实训练；20-step GPU smoke 仍作为代码修改后的下一阶段验收。
