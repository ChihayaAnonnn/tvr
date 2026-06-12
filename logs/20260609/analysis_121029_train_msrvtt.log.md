# 实验分析：PlanA+B1_symmetric_warmup500_wEvid0

- **日志**：`121029_train_msrvtt.log`
- **日期**：2026-06-09 ~ 2026-06-10
- **配置**：batch=64 × 4GPU × 2accum = 512 eff | warmup=500 | w_evid=0 | w_neg=0

## 逐 Epoch 指标

| Epoch | T2V R@1 | V2T R@1 | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|---------|---------|---------|-------------|--------|------------|
| 0 | 46.1 | 45.5 | — | — | — | — |
| 1 | 46.4 | 46.2 | — | — | — | — |
| 2 | 46.7 | 46.3 | — | — | — | — |
| 3 | 47.7 | 46.6 | — | — | — | — |
| 4 | **48.1** | 46.6 | 21.7 | 0.12 | 0.35 | -1.50 |

## 最佳结果

- **Best T2V R@1**：48.1 @ Epoch 4
- **Best V2T R@1**：46.6 @ Epoch 3-4
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260609_121029/pytorch_model.bin.4`

## 与历史对比

| 实验 | T2V R@1 | 关键差异 |
|------|---------|---------|
| Exp 1 | 50.0 | 仅视频置信度 (unsqueeze(1)) |
| **本次** | **48.1** | 视频+文本置信度 (unsqueeze(0) × unsqueeze(1)) |

## 关键发现

1. -1.9 vs Exp 1 — 文本侧 Gaussian log-variance 与视频侧 NIG epistemic 是不兼容的不确定性类型
2. 混合两种不同类型的不确定性引入噪声
3. 需要只使用视频侧的 epistemic，不混合文本 Gaussian variance

## 结论

失败的修正尝试。视频 NIG 和文本 Gaussian 是不兼容的不确定性来源，不应混合。
