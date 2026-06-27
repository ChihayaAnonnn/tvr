# 实验分析：Clean HN diag B, seed=43, pack_seed=42, 2GPU global batch=64, w_mil=0

- **日志**：`logs/20260624/hn_pack_clean_diagB_seed43_pack42_2gpu_b64_train_msrvtt.log`
- **诊断 TSV**：`logs/causal_chain/20260625/ckpt_msrvtt_20260624_hn_pack_clean_diagB_seed43_pack42_2gpu_b64.tsv`；`logs/causal_chain/20260626/ckpt_msrvtt_20260624_hn_pack_clean_diagB_seed43_pack42_2gpu_b64.tsv`
- **分析时间**：2026-06-26 09:36
- **配置**：batch=64 | GPU=2 | accum=1 | effective=64 | per-GPU micro=32 | warmup=500
- **结论**：Diag B 最佳 T2V R@1 为 48.6，高于 Diag A 的 47.8 和 4GPU clean-HN repeat2 的 48.1，但仍低于 B1-only v2 的 49.3；pack_seed=42 比 pack_seed=43 更有利，但不足以恢复 repeat1 的 49.7。

## 实验配置

| 项目 | 值 |
|------|----|
| experiment_desc | Clean HN diag B, seed=43, pack_seed=42, 2GPU global batch=64, w_mil=0 |
| seed | 43 |
| hard_negative_pack_seed | 42 |
| hard_negative_path | `cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json` |
| loaded hard-negative links | 163592 / 180000 |
| use_hard_negative_packing | True |
| hard_negative_rate | 0.5 |
| batch_size | 64 |
| gradient_accumulation_steps | 1 |
| world_size | 2 |
| w_mil / w_evidential / w_neg_reg | 0.0 / 0.0 / 0.0 |
| w_uncertainty_reg / w_orth | 0.001 / 0.1 |
| peak GPU memory | 24.93 GB |

说明：该实验跨日期运行，因此诊断 TSV 被拆到 20260625 和 20260626 两个目录；报告表格已按 epoch 合并。该实验与 4GPU clean-HN repeat 的全局 batch 相同，但 per-GPU micro-batch 和 world size 不同。

## Hard-Negative Sampler

| Sampler Epoch | hit_rate | paired_anchors | dropped |
|---------------|---------:|---------------:|--------:|
| 0 | 0.2551 | 45628 | 32 |
| 1 | 0.2552 | 45633 | 32 |
| 2 | 0.2548 | 45535 | 32 |
| 3 | 0.2563 | 45827 | 32 |
| 4 | 0.2554 | 45666 | 32 |

Sampler 正常启动，hit rate 稳定在约 25.5%，与其它 clean-HN 实验一致。

## 逐 Epoch 指标

| Epoch | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | V2T R@5 | V2T R@10 | Best so far | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|--------:|--------:|---------:|--------:|--------:|---------:|------------:|--------:|------------:|-------:|-----------:|
| 1 | 47.6 | 73.7 | 82.5 | 45.3 | 74.4 | 83.5 | 47.6 | 10.162 | 0.9958 | 0.5941 | -0.0327 |
| 2 | **48.6** | **76.7** | **85.2** | **49.0** | **75.5** | **84.8** | **48.6** | 12.811 | 0.9998 | 0.5981 | -0.1277 |
| 3 | **48.6** | 75.0 | 84.5 | 46.8 | 74.7 | 83.8 | **48.6** | 14.688 | 0.9998 | 0.5972 | -0.2008 |
| 4 | 47.8 | 75.4 | 84.4 | 47.0 | 74.7 | 84.3 | 48.6 | 16.353 | 0.9998 | 0.5968 | -0.2403 |
| 5 | 47.1 | 75.1 | 84.0 | 46.2 | 74.3 | 84.2 | 48.6 | **17.246** | 0.9998 | 0.5967 | -0.2525 |

## 最佳结果

- **Best T2V R@1**：48.6 @ Epoch 2 / Epoch 3
- **对应 V2T R@1**：49.0 @ Epoch 2；46.8 @ Epoch 3
- **Best V2T R@1**：49.0 @ Epoch 2
- **日志最终 Best checkpoint**：`ckpts/ckpt_msrvtt_20260624_hn_pack_clean_diagB_seed43_pack42_2gpu_b64/pytorch_model.bin.2`
- **更均衡 checkpoint**：`ckpts/ckpt_msrvtt_20260624_hn_pack_clean_diagB_seed43_pack42_2gpu_b64/pytorch_model.bin.1`

说明：T2V R@1 在 Epoch 2 与 Epoch 3 同为 48.6，日志最终 `Best so far` 指向 epoch3 的 `pytorch_model.bin.2`；但 epoch2 同时具备更高的 V2T R@1、T2V R@5 和 T2V R@10，因此若考虑双向平衡，epoch2 更值得保留。

## 与历史对比

| 实验 | T2V R@1 | V2T R@1 | 关键差异 |
|------|--------:|--------:|----------|
| Diag B（当前） | **48.6** | 46.8 / 49.0 | seed=43，pack_seed=42，2GPU；T2V tied at epoch2/3 |
| Diag A | **47.8** | 47.6 | seed=42，pack_seed=43，2GPU |
| clean-HN repeat2 | **48.1** | 46.2 | seed=43，pack_seed=43，4GPU |
| clean-HN repeat1 | **49.7** | 46.6 | seed=42，pack_seed=42，4GPU |
| B1-only v2 repeat1 | **49.3** | 47.6 | 无 hard-negative packing |

## 关键发现

- Diag B 比 Diag A 高 0.8 T2V R@1，说明在 2GPU 诊断设置下，pack_seed=42 明显优于 pack_seed=43。
- Diag B 比 4GPU clean-HN repeat2 高 0.5，但仍比 B1-only v2 低 0.7，比 clean-HN repeat1 低 1.1。
- Epoch2 是该实验最均衡的 checkpoint：T2V R@1=48.6，V2T R@1=49.0，T2V R@10=85.2。
- ret_gap 持续上升到 17.246，但 T2V R@1 在 epoch2/3 达峰后下降，说明训练后期并未带来更好 top-1 排序。

## 结论

Diag B 支持“hard-negative packing 顺序会影响结果”的判断：pack_seed=42 比 pack_seed=43 更好。但它仍未超过 B1-only v2，也未接近 clean-HN repeat1 的 49.7，因此 pack_seed=42 只是必要线索，不是充分条件。clean-HN batch packing 目前仍应归类为不稳定方向。
