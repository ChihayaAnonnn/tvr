# 重建系统四卡复验：PASS

2026-10-01，物理 **0、2、3、4** 四张 A800 80GB，FP32、NCCL、四个 CUDA rank。
正式 ViT-B/32 容量，随机初始化 CLIP，合成原始视频与文本，默认关闭语义分组/OT。
worker 退出码 **0**；**15 项**链路检查通过；链路检查耗时 **40.69 秒**。
PyTorch allocated 峰值每 rank 约 **1.414 GiB**；CUDA context 等使进程总显存约 2.2 GiB。
0、2 卡同时有其他任务，本次耗时不用于性能比较。

run：`synthetic-baseline-rebuilt-prod-e2e-retry1-seed713-20261001T081514Z-df6ef932049f`。
source：`df6ef932049f3e16daf43ce53ef04d589c403997`。
manifest SHA-256：`7cd498f33bd94dc25ecb65663a6b39a7b3ed2894933e6f6a61586817cd754a5f`。
数据面：`artifacts/contextual-risk-v2/synthetic/baseline/` 下该 run 的唯一目录。
终态及完整输出哈希见 [机器摘要](rebuilt-code-q0-four-gpu-retest-20261001.json)。

## 检查结果

- 原始视频解码、tokenizer、正式 CLIP 容量、时序编码、coarse 检索、共享 matcher、候选存在头接通。
- `temporal/coarse/matcher/answerability` 四组梯度有限且非零；四 rank 梯度/更新后参数一致。
  known 计数为 coarse=3、match=3、pair=18、exist=6；全 unknown 步不更新优化器或 scheduler。
- checkpoint 恢复模型、优化器、scheduler、各 rank 的 Python/NumPy/CPU/CUDA 随机状态及循环位置。
  两步连续训练与断点续训的参数最大绝对差 **0.0**，优化器状态最大绝对差 **0.0**，均低于容差 `1e-6`。
  错误配置和计划生成的损坏 checkpoint 副本均被拒绝。
- 完成实际 validation calibration、canonical 索引/缓存、有限扩检、零预算终止及返回集合检查。
  拟合校准的 pair/presence 各仍有一个 FP32 概率恰为 1；饱和本身不是错误。
  扩检分支使用单独保存的控制 calibration，预算 3、K 为 1→2→3；locked 评估使用实际拟合 calibration。
- 两个合成 locked episode 的指标与分母正确保存。该样本仅用于链路验收，不报告科学效果。

证据文件位于 run-local `provenance/training-qualification.json`、`qualification-report.json`、
`calibration.json`、`expansion-control-calibration.json`、`locked-metrics.json`、`hardware.json`。
数值数组通过 run-local NPY 引用外置，已验哈希；指标与校准 logits 的引用留在 JSON。

## 归档与适用范围

从干净源码、完整已提交 reserved 批次和 sealed command 启动；禁止 torchrun 自动重试。
`verify-run` 在终态提交前、源码仍精确匹配时通过，状态为 completed。
之后正常 `finalize-phase` 生成单 run completed 批次，与摘要一起提交。
后续文档更新不改变该次运行的 source SHA；归档复核使用历史 manifest、audit 和输出哈希。

本次是首次失败后的新唯一运行：第一个 parent 为 completed synthetic prep；第二个 parent 为首次失败。
首次 127 个内容文件、status/audit 全部保持原字节，控制面通过人工授权的 same-run failed correction 封存，
详见 [人工封存报告](../incidents/2026-10-01-rebuilt-q0-manual-seal.json)。原无效 runtime status 未自动修复。
CPU 全回归 **333 passed**，独立 guard 审阅通过，见 [前置证据](../incidents/2026-10-01-rebuilt-q0-retest-prerequisites.md)。

重建系统的合成四卡软件资格已通过。正式官方标签适配器、动态 minibatch 科学实验及可选分组/OT
效果尚未验证；既有失败研究不恢复授权，MSRVTT/MSVD test 仍 sealed。
