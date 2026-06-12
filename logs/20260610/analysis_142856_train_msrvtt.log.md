# 实验分析：PlanC_scalar_warmup500_wEvid0

- **日志**：`142856_train_msrvtt.log`
- **日期**：2026-06-10 ~ 2026-06-11
- **配置**：batch=256 × 4GPU × 1accum = 1024 eff | warmup=500 | w_evid=0 | w_neg=0

## 逐 Epoch 指标

| Epoch | T2V R@1 | V2T R@1 | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|---------|---------|---------|-------------|--------|------------|
| 0 | 46.0 | 46.6 | — | — | — | — |
| 1 | 47.3 | 47.0 | 11.8 | 0.43 | 0.50 | -1.02 |
| 2 | **47.8** | 47.2 | 15.5 | 0.22 | 0.47 | -1.50 |
| 3 | 46.7 | 45.9 | 17.4 | 0.22 | 0.49 | -1.50 |

## 最佳结果

- **Best T2V R@1**：47.8 @ Epoch 2
- **Best V2T R@1**：47.2 @ Epoch 2
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260610_142856/pytorch_model.bin.2`

## 与历史对比

| 实验 | T2V R@1 | 关键差异 |
|------|---------|---------|
| Exp 1 (NIG) | 50.0 | NIG + gamma投影 + eff_batch=256 |
| **本次** | **47.8** | scalar + 无投影 + eff_batch=1024 |

## 关键发现

1. 下降幅度大（-2.2），两个独立因素叠加：无投影层 + 有效 batch=1024 (steps/epoch 仅 176)
2. u_mode=0.50（非常均匀），Dirichlet 未学会区分锚点
3. epistemic 坍缩到 0.22（常数），与 NIG 问题一致

## 结论

有效 batch 1024 → 优化步数仅 Exp 1 的 12%，模型未充分收敛。需加回投影层并减小有效 batch。
