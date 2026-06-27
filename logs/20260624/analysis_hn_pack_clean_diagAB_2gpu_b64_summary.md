# Clean-HN 诊断实验 A/B 总结

- **日志 A**：`logs/20260624/hn_pack_clean_diagA_seed42_pack43_2gpu_b64_train_msrvtt.log`
- **日志 B**：`logs/20260624/hn_pack_clean_diagB_seed43_pack42_2gpu_b64_train_msrvtt.log`
- **分析时间**：2026-06-26 09:36
- **目的**：拆分 clean-HN 结果波动来源，观察模型 seed 与 hard-negative packing seed 哪个更可疑。
- **结论**：两个 2GPU 诊断实验都没有超过 B1-only v2；Diag B 明显优于 Diag A，说明 pack_seed=42 比 pack_seed=43 更有利，但 clean-HN 的 49.7 高点仍未被稳定解释或复现。

## 对照结果

| 实验 | GPU | seed | hard_negative_pack_seed | Best T2V R@1 | 对应 V2T R@1 | Best epoch / ckpt | 结论 |
|------|----:|-----:|------------------------:|-------------:|-------------:|-------------------|------|
| clean-HN repeat1 | 4 | 42 | 42 | **49.7** | 46.6 | E5 / `pytorch_model.bin.4` | 当前 clean-HN 高点 |
| clean-HN repeat2 | 4 | 43 | 43 | **48.1** | 46.2 | E2 / `pytorch_model.bin.1` | 未复现 49.7 |
| Diag A | 2 | 42 | 43 | **47.8** | 47.6 | E3 / `pytorch_model.bin.2` | 固定 seed=42 不足以复现高点 |
| Diag B | 2 | 43 | 42 | **48.6** | 46.8 / 49.0 | E3 log-best；E2 更均衡 | pack_seed=42 更好，但仍低于 B1 |
| B1-only v2 repeat1 | 2/默认配置 | 42 | 无 | **49.3** | 47.6 | E5 / `pytorch_model.bin.4` | 当前强基线 |

说明：Diag B 的 T2V R@1 在 epoch2 与 epoch3 同为 48.6；日志最终 best checkpoint 指向 epoch3，但 epoch2 的 V2T R@1=49.0，更适合作为双向平衡观察点。

## 诊断判断

原始假设：

| 可能情况 | 预期 | 实际 |
|----------|------|------|
| 模型 seed 是主因 | Diag A 高，Diag B 低 | 不成立：Diag A=47.8，Diag B=48.6 |
| packing seed 是主因 | Diag A 低，Diag B 高 | 部分成立：pack_seed=42 的 Diag B 明显更好 |
| repeat1 是偶然高点 | A/B 都无法接近 49.7 | 当前更接近该判断 |

需要注意：A/B 诊断使用 2GPU 并行，per-GPU micro-batch=32；repeat1/repeat2 使用 4GPU，per-GPU micro-batch=16。因此 A/B 能说明 2GPU 条件下 pack_seed=42 更好，但不能完全排除 world size / per-GPU micro-batch 对结果的影响。

## 关键发现

- `seed=42` 本身不是充分条件：Diag A 使用 seed=42，但 pack_seed=43 时只有 47.8。
- `pack_seed=42` 是正向线索：Diag B 使用 pack_seed=42，在 2GPU 下达到 48.6，是两组诊断中较好的一组。
- clean-HN 的高点 49.7 仍未稳定复现；目前所有后续复验最高只有 48.6，低于 B1-only v2 的 49.3。
- 两个诊断实验均显示 ret_gap 后期继续增长，但 T2V R@1 在中期达峰后下降，说明当前 hard-negative packing 可能会增强分数间隔，却不稳定改善 top-1 排序。

## 结论与下一步

clean-HN batch packing 目前不适合作为主线稳定贡献。它的收益对 seed、packing 顺序以及 GPU/world-size 配置敏感，且 A/B 诊断后仍未超过 B1-only v2。

建议路线：

1. **暂停继续扩展 clean-HN batch packing 的训练组合**：暂时不要加 `w_mil=0.01`，也不要直接提高 batch 追高点。
2. **若必须补一个严格控制实验**：优先跑 2GPU 的 `seed=42, pack_seed=42`，判断 2GPU/world-size 是否会压低原 repeat1 高点；但这属于诊断补充，不是主线推进。
3. **更值得推进的是数据验证流程**：从 train 里切内部 val，基于内部验证集评估 hard-negative map 质量、clean 阈值和假负例比例，再决定是否重新设计 hard-negative 数据。
4. **论文表述上应保守**：可以写 raw map 质量差、clean map 能产生高点，但 batch packing 收益未稳定复现；不能写成稳定提升方法。
