# 实验分析：B1only_v2

- **日志**：`000227_train_msrvtt.log`
- **日期**：2026-06-17 00:02 → 2026-06-17 10:50
- **配置**：batch=64 × 4GPU × 1accum = 256 eff | w_evid=0 | w_neg=0 | 无置信度加权

## 逐 Epoch 指标

| Epoch | T2V R@1 | V2T R@1 | ret_gap | ret_pos | ret_neg | epistemic_v | logsig_v |
|-------|---------|---------|---------|---------|---------|-------------|----------|
| 0 | 47.3 | 46.8 | — | — | — | — | — |
| 1 | 47.5 | 47.9 | 10.4 | 39.1 | 28.7 | 0.66 | −0.78 |
| 2 | 47.9 | 47.8 | 12.8 | 51.9 | 39.0 | 0.90 | −1.41 |
| 3 | **48.5** | 47.0 | 14.4 | 50.5 | 36.1 | 0.99 | −1.50 |
| 4 | 47.9 | 48.4 | 16.4 | 46.8 | 30.4 | 1.00 | −1.50 |

## 最佳结果

- **Best T2V R@1**：48.5 @ Epoch 3
- **Best V2T R@1**：48.4 @ Epoch 4
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260617_000227/pytorch_model.bin.3`

## 与历史对比

| 实验 | T2V R@1 | ret_gap | 置信度模式 | 架构 |
|------|:---:|:---:|------|------|
| Exp 1 | 50.0 | ~18 | per-video unsq(1) | NIG gamma + anchors.detach() |
| Dir1_colwise | 49.3 | 27.5 | per-video unsq(0) | anchor_proj + proj.detach() |
| PairConfMLP | 48.6 | 26.3 | per-pair MLP | anchor_proj + proj.detach() |
| **B1only_v2** | **48.5** | **16.4** | **无** | **anchor_proj + proj.detach()** |
| baseline | 49.4 | — | 无 | NIG gamma（无 detach） |

## 关键发现

1. **B1-only 未复现 50.0**：T2V 48.5，低于 Exp 1（50.0）和 baseline（49.4）
2. **ret_gap 显著低于有置信度加权的版本**：16.4 vs 27.5（Dir1_colwise），差 11 点。表征学习从 Epoch 1 就落后
3. **当前架构的 B1 detach 实际效果是 −0.9 vs baseline**（48.5 vs 49.4），不是 Exp 1 的 +0.6
4. **根因**：当前 SAP 架构（anchor_proj + Dirichlet-only head + 非学习不确定性）与 Exp 1（NIG gamma + 双层 head）不同。B1 detach 在这两套架构上的效果不直接等价
5. **logsigma_v 仍撞 −1.5**：Epoch 3 开始饱和

## 结论

B1 detach 的收益依赖于具体的 SAP 表征路径。Exp 1 中 NIG gamma 与 anchors.detach() 的组合提供了 +0.6 增益，但当前 anchor_proj + projected.detach() 组合反而低于 baseline。下一步应回到 Exp 1 commit (`21348d1`) 做减法，或在当前架构上重新验证是否为随机波动（同配置再跑一轮）。
