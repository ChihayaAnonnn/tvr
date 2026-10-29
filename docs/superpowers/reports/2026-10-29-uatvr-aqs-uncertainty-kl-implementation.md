# UATVR+AQS 不确定性 KL 融合实施交付

日期：2026-10-29（北京时间）。分支：`codex/uatvr-aqs-training-v1`。
验证代码提交：`aa96c45e1909f0b6d025d418a93ed26d13c34b30`。本报告及实施清单是后续文档提交，不改变已验证的 Python 源码。

## 结果

设计与实施计划的 Task1–7 软件范围已完成。新评分同时进入完整双向 CE、AQS 困难度和辅助 CE、矩形推理。旧 AQS/vendor/启动器源文件保持不变，融合关闭或 eta0 的退化行为经精确回归核对。

交付六个独立模块：`fusion_config.py`、`fusion_math.py`、`fusion_core.py`、`fusion_model.py`、`fusion_training.py`、`fusion_runtime.py`，均位于 `src/semantic_retrieval/aqs/`。使用示例与边界见 [软件接口说明](../../agent/uatvr-aqs-uncertainty-kl.md)。

## 关键合同

- 复用原概率头，评分明确采用 `logvar=2*clip(logsigma,-4,2)`。原采样/MIL/legacy VIB 语义和权重保持，未暗中修复旧约定差异。
- 对称 Gaussian KL 按维归一化，融合为 `WTI-eta*D`，之后沿用原 logit scale。eta 显式必填，示例0.1只用于软件测试。
- 选择使用 detached 融合分数，损失使用有梯度 logits；隔离 MIL/VIB 后确认均值/方差头与 encoder 仍收到检索梯度。
- 推理解析评分，不进行 Gaussian 采样；矩形、singleton、分块、排列、重复调用、RNG 与输入不变性经核对。
- 新 `.fusion.json` 绑定评分身份、实际源码字节、完整 recipe、ID、输入与 world size；恢复分为只读载荷验证和状态应用，集体验证完成前不修改模型/optimizer/global RNG。
- 离线 recipe 解析器不启动 CUDA、不读数据、不创建实验或授予研究资格；旧 manifest、缺失或未知字段、DSL/test 请求拒绝。
- 真实独立进程在屏蔽外部 reranker 导入/路径、限制写入 pytest 临时目录时完成新模式 forward、eval、step 与 save/restore。

## 验证证据

最终命令（在仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest tests/semantic_retrieval/aqs tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py -q --tb=short
```

实际结果：**147 passed, 1 warning in 88.06s**。warning 是原 PyTorch TypedStorage 弃用提示，无测试失败。测试先经历了预期的功能缺失/错误行为失败，再实施并通过；独立审查发现的两个问题也有修复前失败与修复后通过的回归。

核对内容包括：

- `torch.distributions.kl_divergence` FP64 oracle 与真实一阶梯度，FP32 autocast 隔离；近零、固定最小方差、维度2/3/4/8/64/512，保存张量无完整三维 pair 特征。
- 四种融合/AQS 开关的训练、旧路径精确等价、新模型/新优化器下一步精确恢复。
- CPU 双 rank Gloo 对比串行全局目标/梯度，配置/ID/输入/概率/optimizer 错误同步拒绝。
- 单 rank checkpoint 读取失败时，所有 rank 拒绝且模型、优化器、RNG fingerprints 不变；legacy 采样溢出同步拒绝。
- 新路径依赖与写入隔离、离线配置严格解析、旧 AQS 与 baseline 回归。

`git diff --check` 无错误；以下范围对设计提交 `7223d26` 的差异为空：

```text
src/semantic_retrieval/full_data/upstream_uatvr/
src/semantic_retrieval/aqs/{core,model,config,training}.py
scripts/prepare_aqs_training.py
src/semantic_retrieval/aqs/launch.py
```

## 审查修订

独立只读代码审查及复核的最终结论为无未解决的重要问题。已落实的主要修订：

1. 新恢复实现将完整模型/optimizer/数组/RNG 校验置于 collective staging 中，排除“一个 rank 已恢复，另一个失败”的状态分叉。
2. KL 快速矩阵计算出现近零消减时，采用依据点积规模与 FP32 精度的误差界，进入稳定直接计算。使用有界二维 scratch 与解析 backward，原 `1e-5/2e-4` 数值验收容差保持。
3. 有限概率参数仍可使 legacy Gaussian 采样溢出，因此在内部 collective 检查采样和 MIL/VIB，并同步最终 loss 有限性。

## 限制与未执行范围

本次是 CPU 软件验证，完整 CLIP ViT-B/32 的 CUDA 数值、显存、吞吐和资格尚未验证。稳定近零回退仅支持一阶导数，适用于当前训练，不提供高阶微分合同。

本版未新增跨评分块缓存、自动 warm-start、生产 GPU launcher 或新研究控制面。没有预留/启动科学实验，没有使用 test，也没有修改历史运行或 Laya study。未产生新模型 Recall、校准或不确定性有效性结论；后续科学执行必须按当时适用的归档与 gate 建立独立方案。
