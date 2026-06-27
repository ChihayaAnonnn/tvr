# 实验分析：Clean HN diag A, seed=42, pack_seed=43, 2GPU global batch=64, w_mil=0

- **日志**：`logs/20260624/hn_pack_clean_diagA_seed42_pack43_2gpu_b64_train_msrvtt.log`
- **诊断 TSV**：`logs/causal_chain/20260625/ckpt_msrvtt_20260624_hn_pack_clean_diagA_seed42_pack43_2gpu_b64.tsv`
- **分析时间**：2026-06-26 09:36
- **配置**：batch=64 | GPU=2 | accum=1 | effective=64 | per-GPU micro=32 | warmup=500
- **结论**：Diag A 最佳 T2V R@1 为 47.8，低于 B1-only v2 的 49.3，也低于 clean-HN repeat2 的 48.1；固定模型 seed=42 但改用 pack_seed=43 并不能复现 repeat1 的 49.7。

## 实验配置

| 项目 | 值 |
|------|----|
| experiment_desc | Clean HN diag A, seed=42, pack_seed=43, 2GPU global batch=64, w_mil=0 |
| seed | 42 |
| hard_negative_pack_seed | 43 |
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

说明：该实验使用 2GPU 并行，shell `--batch_size=64` 表示全局 batch；每卡 micro-batch 为 32。它与此前 4GPU clean-HN repeat 的全局 batch 相同，但 per-GPU micro-batch 和 world size 不同，因此不能视为完全同配置复现。

## Hard-Negative Sampler

| Sampler Epoch | hit_rate | paired_anchors | dropped |
|---------------|---------:|---------------:|--------:|
| 0 | 0.2552 | 45633 | 32 |
| 1 | 0.2548 | 45535 | 32 |
| 2 | 0.2563 | 45827 | 32 |
| 3 | 0.2554 | 45666 | 32 |
| 4 | 0.2556 | 45700 | 32 |

Sampler 正常启动，clean map 正常加载，hit rate 稳定在约 25.5%。

## 逐 Epoch 指标

| Epoch | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | V2T R@5 | V2T R@10 | Best so far | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|--------:|--------:|---------:|--------:|--------:|---------:|------------:|--------:|------------:|-------:|-----------:|
| 1 | 46.0 | 75.1 | 83.2 | 46.5 | 74.6 | 84.1 | 46.0 | 10.206 | 0.9959 | 0.5831 | -0.0326 |
| 2 | 47.5 | 75.2 | 84.8 | 46.2 | 74.4 | 84.4 | 47.5 | 12.591 | 0.9997 | 0.5881 | -0.1277 |
| 3 | **47.8** | **76.1** | 84.2 | **47.6** | **76.0** | 82.9 | **47.8** | 14.350 | 0.9998 | 0.5881 | -0.2008 |
| 4 | 47.6 | **76.1** | **84.9** | 46.5 | 75.4 | 82.9 | 47.8 | 15.848 | 0.9998 | 0.5882 | -0.2403 |
| 5 | 46.7 | 75.7 | **84.9** | 46.3 | 75.4 | 83.1 | 47.8 | **16.639** | 0.9998 | 0.5882 | -0.2525 |

## 最佳结果

- **Best T2V R@1**：47.8 @ Epoch 3
- **对应 V2T R@1**：47.6
- **Best V2T R@1**：47.6 @ Epoch 3
- **Checkpoint**：`ckpts/ckpt_msrvtt_20260624_hn_pack_clean_diagA_seed42_pack43_2gpu_b64/pytorch_model.bin.2`

## 与历史对比

| 实验 | T2V R@1 | V2T R@1 | 关键差异 |
|------|--------:|--------:|----------|
| Diag A（当前） | **47.8** | 47.6 | seed=42，pack_seed=43，2GPU |
| Diag B | **48.6** | 46.8 | seed=43，pack_seed=42，2GPU；日志最终 best checkpoint 为 epoch3 |
| clean-HN repeat2 | **48.1** | 46.2 | seed=43，pack_seed=43，4GPU |
| clean-HN repeat1 | **49.7** | 46.6 | seed=42，pack_seed=42，4GPU |
| B1-only v2 repeat1 | **49.3** | 47.6 | 无 hard-negative packing |

## 关键发现

- Diag A 使用 repeat1 的模型 seed=42，但将 hard-negative packing seed 改为 43 后，T2V 最佳只有 47.8。
- 该结果低于 clean-HN repeat2 的 48.1，更低于 B1-only v2 的 49.3，说明模型 seed=42 本身不足以解释 repeat1 的 49.7。
- ret_gap 从 10.206 增长到 16.639，但 T2V R@1 在 epoch3 达峰后下降，继续说明 ret_gap 增大不等价于 top-1 检索改善。
- 与 Diag B 相比，pack_seed=43 的 2GPU 诊断明显更差，hard-negative packing 顺序可能是重要不稳定因素之一。

## 结论

Diag A 不支持“只要使用 repeat1 的模型 seed=42 就能复现 clean-HN 高点”的假设。当前证据更倾向于：repeat1 的 49.7 依赖 seed、hard-negative packing 顺序、以及 4GPU/world-size 配置的组合；其中 pack_seed=43 在 2GPU 诊断中表现较差。
