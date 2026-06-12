# 实验分析：PlanA+B1_colwise_warmup500_wEvid0

- **日志**：`013304_train_msrvtt.log`
- **日期**：2026-06-10
- **配置**：batch=64 × 4GPU × 2accum = 512 eff | warmup=500 | w_evid=0 | w_neg=0

## 逐 Epoch 指标

| Epoch | T2V R@1 | V2T R@1 | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|---------|---------|---------|-------------|--------|------------|
| 0 | 45.2 | 46.7 | — | — | — | — |
| 1 | 47.3 | 48.0 | 13.5 | 1.92 | 0.47 | -0.24 |
| 2 | **47.7** | 47.8 | 15.6 | 0.22 | 0.35 | -1.50 |
| 3 | 46.2 | 45.8 | 17.2 | 0.19 | 0.34 | -1.50 |

## 最佳结果

- **Best T2V R@1**：47.7 @ Epoch 2
- **Best V2T R@1**：48.0 @ Epoch 1
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260610_013304/pytorch_model.bin.2`

## 与历史对比

| 实验 | T2V R@1 | 置信度方向 |
|------|---------|-----------|
| Exp 1 | 50.0 | unsqueeze(1) — 行方向（T2V 不受扰） |
| symmetric | 48.1 | 列×行（视频+文本混合） |
| **本次** | **47.7** | unsqueeze(0) — 列方向（T2V 受扰） |

## 关键发现

1. unsqueeze(0) 列方向折扣将不确定性直接作用于 T2V 主 loss
2. 训练初期 epistemic≈6.6, confidence≈0.13 且各样本均匀 → 等同随机噪声
3. T2V 被无意义的噪声干扰，收敛质量下降
4. Exp 1 的 unsqueeze(1) bug 恰好避开了这个问题

## 结论

未校准的不确定性用于 T2V 主 loss 是毒药。Exp 1 的 +0.6 来自 B1，非来自置信度加权。
