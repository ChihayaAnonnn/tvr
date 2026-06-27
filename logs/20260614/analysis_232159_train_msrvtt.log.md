# 实验分析：PairConf_warmup500

- **日志**：`232159_train_msrvtt.log`
- **日期**：2026-06-14 23:22 → 2026-06-15 10:21
- **配置**：batch=64 × 4GPU × 1accum = 256 eff | warmup=500 | w_evid=0 | w_neg=0

## 逐 Epoch 指标

| Epoch | T2V R@1 | V2T R@1 | ret_gap | ret_pos | ret_neg | epistemic_v | logsigma_v |
|-------|---------|---------|---------|---------|---------|-------------|------------|
| 0 | 46.9 | 46.0 | — | — | — | — | — |
| 1 | 47.7 | 47.6 | 10.36 | 38.7 | 28.3 | 0.33 | −1.40 |
| 2 | **48.6** | 46.6 | 12.89 | 51.4 | 38.5 | 0.71 | −1.50 |
| 3 | 48.6 | 46.9 | 14.60 | 50.0 | 35.4 | 0.99 | −1.50 |
| 4 | 48.4 | 47.5 | 16.66 | 46.4 | 29.7 | 1.00 | −1.50 |

## 最佳结果

- **Best T2V R@1**：48.6 @ Epoch 2/3
- **Best V2T R@1**：47.6 @ Epoch 1
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260614_232159/pytorch_model.bin.2`

## 与历史对比

| 实验 | T2V R@1 | ret_gap | 置信度模式 |
|------|---------|---------|-----------|
| baseline | 49.4 | — | 无加权 |
| Exp 1 (A+B1) | 50.0 | 18.0 | per-video, unsqueeze(1)（实际无效） |
| Dir1_colwise | 49.3 | 27.5 | per-video, unsqueeze(0)（有效但无贡献） |
| **本次** | **48.6** | **16.7** | **per-pair [B,B]** |

## 关键发现

1. **T2V 下降 0.7 vs baseline**：48.6 < 49.4，per-pair 置信度未产生正向收益
2. **ret_gap 崩塌**：16.7 vs 27.5（Dir1_colwise），降幅 39%。负样本分数被整体抬高（ret_neg=30 vs 7），表征判别力显著退化
3. **ret_pos 也异常升高**：46 vs 35（Dir1_colwise）。整个相似度矩阵水位上升，但正负分离度反而下降
4. **根因：梯度冲突**。confidence 梯度路径（text_pooled → text_token → text encoder）与 WTI 梯度路径共享 text encoder 骨干。confidence 要求 text 表征同时做到：(a) 集中在 matching video 的特定 anchor（提高对角 confidence），(b) 分散在 non-matching video 的所有 anchor（压低非对角 confidence）。这与 WTI 的"推高匹配对、压低不匹配对"目标产生干扰
5. **logsigma_v 仍撞 −1.5 下界**：Epoch 2 即饱和，MIL 方差惩罚无改善

## 结论

per-pair 置信度引入了 text encoder 的额外梯度路径，导致表征质量下降（ret_gap −10.8）。问题不在于"正负抵消"，而在于两条梯度流共享 text encoder 产生冲突。下一步可以：(a) detach text_pooled 切断 confidence 到 text encoder 的梯度，或 (b) 回退到 B1-only（已验证 50.0）并探索非加权方向。
