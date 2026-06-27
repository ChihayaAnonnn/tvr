# 实验分析：Clean HN packing repeat2, 4GPU global batch=64, w_mil=0

- **日志**：`logs/20260624/hn_pack_clean_wmil0_repeat2_4gpu_b64_train_msrvtt.log`
- **诊断 TSV**：`logs/causal_chain/20260624/ckpt_msrvtt_20260624_hn_pack_clean_wmil0_repeat2_4gpu_b64.tsv`
- **分析时间**：2026-06-24 22:18
- **配置**：batch=64 | GPU=4 | accum=1 | effective=64 | per-GPU micro=16 | warmup=500
- **结论**：本次 clean-HN repeat2 最佳 T2V R@1 仅为 48.1，未复现 repeat1 的 49.7，也低于 B1-only v2 的 49.3；clean-HN 方向显示出高点潜力，但当前稳定性不足。

## 实验配置

| 项目 | 值 |
|------|----|
| experiment_desc | Clean HN packing repeat2, 4GPU global batch=64, w_mil=0 |
| seed | 43 |
| hard_negative_pack_seed | 43 |
| hard_negative_path | `cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json` |
| loaded hard-negative links | 163592 / 180000 |
| use_hard_negative_packing | True |
| hard_negative_rate | 0.5 |
| batch_size | 64 |
| gradient_accumulation_steps | 1 |
| world_size | 4 |
| w_mil / w_evidential / w_neg_reg | 0.0 / 0.0 / 0.0 |
| w_uncertainty_reg / w_orth | 0.001 / 0.1 |
| CLIP lr / new module lr | 1e-7 / 1e-4 |
| peak GPU memory | 13.23 GB |

说明：本项目的 shell `--batch_size` 表示目标全局 batch。该实验 `batch_size=64`、`accum=1`、`world_size=4`，因此每卡 micro-batch 为 16；每次 forward 后 all-gather 的全局对比 batch 为 64。

## Hard-Negative Sampler

| Sampler Epoch | hit_rate | paired_anchors | dropped |
|---------------|---------:|---------------:|--------:|
| 0 | 0.2552 | 45633 | 32 |
| 1 | 0.2548 | 45535 | 32 |
| 2 | 0.2563 | 45827 | 32 |
| 3 | 0.2554 | 45666 | 32 |
| 4 | 0.2556 | 45700 | 32 |

Sampler 行为稳定，hit rate 与 clean-HN repeat1 基本一致，因此本次低结果不像是 hard-negative 映射路径错误或 sampler 未生效导致。

## 逐 Epoch 指标

| Epoch | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | V2T R@5 | V2T R@10 | Best so far | ret_gap | epistemic_v | u_mode | logsigma_v |
|-------|--------:|--------:|---------:|--------:|--------:|---------:|------------:|--------:|------------:|-------:|-----------:|
| 1 | 47.0 | 75.3 | 83.2 | 46.0 | 74.8 | 84.2 | 47.0 | 10.236 | 0.9959 | 0.5950 | -0.0326 |
| 2 | **48.1** | **76.1** | **84.7** | 46.2 | 74.5 | 84.1 | **48.1** | 12.762 | 0.9998 | 0.5981 | -0.1277 |
| 3 | 47.3 | 75.7 | 83.9 | **47.7** | 74.2 | 83.6 | 48.1 | 14.588 | 0.9998 | 0.5970 | -0.2007 |
| 4 | 47.4 | 75.4 | 84.4 | 46.1 | 74.2 | 83.0 | 48.1 | 16.103 | 0.9998 | 0.5967 | -0.2402 |
| 5 | 47.6 | 74.8 | 83.6 | 45.8 | 74.0 | 83.3 | 48.1 | **16.850** | 0.9998 | 0.5966 | -0.2524 |

## 最佳结果

- **Best T2V R@1**：48.1 @ Epoch 2
- **对应 V2T R@1**：46.2
- **Best V2T R@1**：47.7 @ Epoch 3
- **T2V best checkpoint**：`ckpts/ckpt_msrvtt_20260624_hn_pack_clean_wmil0_repeat2_4gpu_b64/pytorch_model.bin.1`
- **V2T best checkpoint**：`ckpts/ckpt_msrvtt_20260624_hn_pack_clean_wmil0_repeat2_4gpu_b64/pytorch_model.bin.2`

## 与历史对比

| 实验 | T2V R@1 | V2T R@1 | 关键差异 |
|------|--------:|--------:|----------|
| clean-HN repeat2（当前） | **48.1** | 46.2 | clean map，seed=43，hard_negative_pack_seed=43，w_mil=0 |
| clean-HN repeat1 | **49.7** | 46.6 | 同 clean map，同 batch 配置，但使用 repeat1 seed/packing 顺序 |
| B1-only v2 repeat1 | **49.3** | 47.6 | 无 hard-negative packing，当前强基线 |
| raw HN packing repeat1 | **48.1** | 48.7 | raw map 未清洗，假负例和 caption 复用严重 |
| Exp1 历史最高 | **50.0** | 未在本日志对比中重新确认 | 历史最高未稳定复现，不能作为当前 clean-HN 的直接对照 |

## 关键发现

- 本次 clean-HN repeat2 未复现 repeat1 的 49.7，最佳 T2V R@1 只有 48.1，低于 B1-only v2 repeat1 的 49.3。
- sampler 启动与 clean map 加载均正常：`163592/180000` hard-negative links 被加载，5 个 epoch 的 hit rate 稳定在约 25.5%。
- T2V 最佳出现在 Epoch 2，之后 ret_gap 持续上升，但 T2V R@1 没有继续上升，说明训练诊断里的分数间隔增大并没有转化为检索 top-1 改善。
- V2T 最佳出现在 Epoch 3，与 T2V 最佳 epoch 不一致；该实验没有形成双向一致收益。
- repeat1 的 49.7 仍然是 clean-HN 的正向高点，但 repeat2 表明该收益对随机种子或 hard-negative packing 顺序敏感。

## 结论

clean-HN batch packing 目前不能直接作为稳定主线结论。更严谨的下一步不是立刻加 `w_mil` 或提高 batch，而是拆分不稳定来源：分别固定模型 seed 与 hard-negative packing seed，判断波动主要来自模型初始化/训练随机性，还是来自 hard-negative batch 组装顺序。

建议后续优先跑两组诊断实验：

| 实验 | seed | hard_negative_pack_seed | 目的 |
|------|-----:|------------------------:|------|
| A | 42 | 43 | 固定 repeat1 的模型 seed，只改变 hard-negative packing 顺序 |
| B | 43 | 42 | 固定 repeat1 的 packing seed，只改变模型 seed |

若 A 高、B 低，则模型 seed 更敏感；若 A 低、B 高，则 batch packing 顺序更敏感；若二者都低，则 clean-HN repeat1 的 49.7 更可能是偶然高点，应暂停扩大该路线。
