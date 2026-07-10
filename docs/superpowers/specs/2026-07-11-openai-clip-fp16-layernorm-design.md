# OpenAI CLIP FP16 LayerNorm 设计规格

**日期**：2026-07-11

**状态**：已确认，待实施

**适用主线**：`train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_mulit.py` → `modules/module_clip.py`

## 1. 背景与结论

单卡 A800 80GB 运行 trusted-v1 OpenAI CLIP hygiene、`batch=256`、`max_frames=8` 时，首个视觉前向发生 OOM。该批次会同时编码 2048 张图像；ViT-B/16 的视觉 token 张量为 `[197, 2048, 768]`。当前自定义 `LayerNorm` 在每次调用中将完整输入显式复制为 FP32，单份 FP32 张量约 1.154 GiB。

本机只读基准表明，native FP16 LayerNorm 可消除这类整块 FP32 副本，并且在代表性输入上的输出与梯度均与旧实现高度接近。但这不能直接证明最终检索指标完全不变，也不能保证仅靠 LayerNorm 优化即可让 batch 256 完整训练通过。

因此采用以下决策：

- OpenAI CLIP 自定义 LayerNorm 默认使用 native FP16 执行；
- 保留显式 FP32 回退开关，支持严格 A/B；
- LayerNorm 的 master `weight`/`bias` 继续保留 FP32；
- 本次不加入 activation checkpointing，不启用全模型 AMP；
- 精度选择必须进入日志与实验 manifest。

## 2. 目标与非目标

### 2.1 目标

- 消除 CUDA FP16 输入在 LayerNorm 中生成整块 FP32 激活副本的显存开销。
- 默认启用 FP16 LayerNorm，同时提供可追溯的 FP32 回退模式。
- 保持 LayerNorm 参数为 FP32 master，由现有优化器继续更新。
- 对输出、梯度、极值稳定性、配置传播和实验 provenance 建立自动回归门禁。

### 2.2 非目标

- 不把整个 UATVR 强制转换为 FP16。
- 不接入 `torch.autocast`、`GradScaler` 或 Apex AMP。
- 不修改 WTI、多正例 InfoNCE、SAP、不确定性模块或 dataloader 数值精度。
- 不在同一提交加入 activation checkpointing、GradCache 或 backbone 冻结。
- 不承诺 FP16 LayerNorm 单独使 batch 256 一定可训练；容量必须由后续实测确认。

## 3. 方案比较

评估三种方案：

1. **直接永久删除 FP32 路径**：改动最小，但无法进行回退和可信 A/B，拒绝采用。
2. **默认 FP16、保留 FP32 开关**：显存收益明确，实验可追溯，出现数值问题时可立即回退。采用该方案。
3. **全模型 AMP**：覆盖范围大，当前 `--fp16` 尚未接入真实 AMP 数据流，回归面远超本问题，暂不采用。

## 4. 精度接口与执行语义

新增 CLI：

```text
--clip_layer_norm_precision {fp16,fp32}
```

默认值为 `fp16`。训练与评估脚本显式传递环境变量：

```text
CLIP_LAYER_NORM_PRECISION=fp16|fp32
```

### 4.1 FP16 模式

仅当输入同时满足以下条件时走 native FP16：

- `x.is_cuda`；
- `x.dtype == torch.float16`；
- 配置为 `fp16`。

执行逻辑等价于：

```python
F.layer_norm(
    x,
    normalized_shape,
    weight.to(dtype=x.dtype),
    bias.to(dtype=x.dtype),
    eps,
)
```

其中仅 768 维左右的 affine 向量发生临时转换。模型中保存和优化的 `weight`/`bias` 仍为 FP32，梯度通过 dtype cast 返回 FP32 master 参数。

### 4.2 FP32 回退模式

`fp32` 模式完整保留原始 OpenAI CLIP 行为：输入临时转换到 FP32，执行 LayerNorm 后再转换回原 dtype。

以下情况即使默认配置为 `fp16`，也安全使用 FP32：

- 输入本身为 FP32；
- CPU 上的 FP16 输入；
- 非 FP16 dtype（包括当前未纳入本设计的 BF16）。

### 4.3 影响边界

精度设置应用于项目复用的 `modules.module_clip.LayerNorm`。实际显存收益主要来自 OpenAI CLIP 视觉 Transformer；文本 Transformer token 数较少，收益次要。`seqTransf` 当前接收 FP32 输入，因此不会被强制降为 FP16。

EVA backbone 使用自身 LayerNorm 实现，本开关不改变 EVA 数值路径。若 `backbone_type=eva_clip`，配置仍写入 provenance，但日志必须明确其对 EVA 不生效，避免误读。

## 5. 配置传播与可追溯性

- `main_task_retrieval.py` 解析并校验精度选项，默认 `fp16`。
- `UATVR` 初始化时将选项传播到所有项目自定义 CLIP LayerNorm 实例。
- 有效参数日志的 Model 分组记录 `clip_layer_norm_precision`。
- `experiment_manifest.json` 的 backbone 字段记录该值。
- `train_msrvtt.sh` 和 `eval.sh` 默认显式设置并传入 `fp16`，允许环境变量覆盖为 `fp32`。
- checkpoint 参数结构不变；旧 checkpoint 可在两种模式下加载，不新增权重键。

## 6. 错误处理

- CLI 出现非 `fp16|fp32` 值时在模型构建前失败。
- LayerNorm 内部出现未知 precision 时抛出明确 `ValueError`，不得静默回退。
- FP16 路径若输出或梯度出现非有限值，由单元测试和短 smoke 阶段阻断，不通过自动切换掩盖问题。

## 7. 测试与验收

实施严格遵循 TDD，先新增失败测试，再修改生产代码。

### 7.1 CPU 单元测试

- CLI 默认值为 `fp16`，显式 `fp32` 可解析，非法值被拒绝。
- FP32 输入在两种配置下均保持 FP32 数值路径。
- CPU FP16 输入安全回退到旧 FP32 计算，不依赖 CPU half LayerNorm kernel。
- FP32 模式与变更前公式严格一致。
- 模型初始化会把配置传播到所有项目自定义 CLIP LayerNorm。
- experiment manifest 记录精度；train/eval 脚本传参和 shell 语法正确。

### 7.2 CUDA 条件测试

- FP16 模式输出 dtype 为 FP16，forward/backward 均为 finite。
- 与 FP32 回退的输出、输入梯度 cosine 均不低于 `0.9999`，最大绝对误差不高于 `1e-2`。
- 覆盖普通随机输入、低方差、大偏置和接近 FP16 范围上限的输入。
- 使用 saved-tensor 或峰值显存门禁，确认 FP16 模式不保存与完整输入同大小的 FP32 LayerNorm 副本。

### 7.3 项目回归

- 运行 LayerNorm/CLIP/modeling/tracking/CLI 聚焦测试。
- 运行项目 `tests/` 全量测试、Ruff、`py_compile`、shell syntax 和 `git diff --check`。
- 不启动长期训练；交付单步显存 smoke 命令，由用户决定是否运行完整实验。

## 8. 科研解释边界

FP16 LayerNorm 是运行时与显存优化，不是模型创新。任何新基线必须记录精度模式；FP16 与 FP32 结果不得在未说明配置的情况下混合汇总。

首次正式采用前，至少完成：

1. 同 checkpoint 的输出与梯度回归；
2. 单步 forward/backward 显存验证；
3. 同 seed 的短程 loss/val 对照。

若 FP16 出现非有限梯度或明确的 val 退化，应使用 `CLIP_LAYER_NORM_PRECISION=fp32` 回退，再单独评估 activation checkpointing。
