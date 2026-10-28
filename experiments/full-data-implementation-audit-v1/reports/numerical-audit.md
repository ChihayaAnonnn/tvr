# 官方数值一致性审计（2026-10-13）

成功运行：[msrvtt-baseline-cpu-audit-seed713-20261013T095250Z-908a67297b16](/data2/hxj/project/UATVR/artifacts/full-data-implementation-audit-v1/msrvtt/baseline/msrvtt-baseline-cpu-audit-seed713-20261013T095250Z-908a67297b16/provenance/audit-result.json)。source `908a67297b16`，预留提交 `8dadb32`，归档提交 `0f1507b`；manifest SHA-256 `b99e93550f59e6d02c5ad5724f2bc94da87674b9b1f6b4c1a98ccd2680009d1c`。

生产容量 CLIP ViT-B/32、CPU FP32、两个真实 train 视频（video0/video1000）、12 帧、共享初值和 RNG、一个真实 Gloo rank。执行未改动的固定官方模型和 BertAdam；两次优化器 step，其中第一次 warmup LR 为 0，第二次验证实际更新。预先固定 atol=2e-6、rtol=2e-4；输入要求完全相同。

| 方法 | 项目 | 最大绝对误差 | 通过 |
|---|---|---:|---|
| uatvr | eval_logits | 0 | True |
| uatvr | input_0_frame_mask | 0 | True |
| uatvr | input_0_text_mask | 0 | True |
| uatvr | input_0_tokens | 0 | True |
| uatvr | input_0_video | 0 | True |
| uatvr | input_1_frame_mask | 0 | True |
| uatvr | input_1_text_mask | 0 | True |
| uatvr | input_1_tokens | 0 | True |
| uatvr | input_1_video | 0 | True |
| uatvr | step0_contrastive | 0 | True |
| uatvr | step0_gradients | 0 | True |
| uatvr | step0_kl | 0 | True |
| uatvr | step0_logits | 0 | True |
| uatvr | step0_loss | 0 | True |
| uatvr | step0_mil | 0 | True |
| uatvr | step0_parameters | 0 | True |
| uatvr | step1_contrastive | 0 | True |
| uatvr | step1_gradients | 0 | True |
| uatvr | step1_kl | 0 | True |
| uatvr | step1_logits | 0 | True |
| uatvr | step1_loss | 0 | True |
| uatvr | step1_mil | 0 | True |
| uatvr | step1_parameters | 0 | True |
| uatvr | text_pooled | 0 | True |
| uatvr | valid_video_tokens | 0 | True |
| clip4clip | eval_logits | 5.7220459e-06 | True |
| clip4clip | input_0_frame_mask | 0 | True |
| clip4clip | input_0_text_mask | 0 | True |
| clip4clip | input_0_tokens | 0 | True |
| clip4clip | input_0_video | 0 | True |
| clip4clip | input_1_frame_mask | 0 | True |
| clip4clip | input_1_text_mask | 0 | True |
| clip4clip | input_1_tokens | 0 | True |
| clip4clip | input_1_video | 0 | True |
| clip4clip | step0_contrastive | 2.60770321e-07 | True |
| clip4clip | step0_gradients | 4.17232513e-06 | True |
| clip4clip | step0_kl | 0 | True |
| clip4clip | step0_logits | 5.7220459e-06 | True |
| clip4clip | step0_loss | 2.60770321e-07 | True |
| clip4clip | step0_mil | 0 | True |
| clip4clip | step0_parameters | 0 | True |
| clip4clip | step1_contrastive | 2.60770321e-07 | True |
| clip4clip | step1_gradients | 4.17232513e-06 | True |
| clip4clip | step1_kl | 0 | True |
| clip4clip | step1_logits | 5.7220459e-06 | True |
| clip4clip | step1_loss | 2.60770321e-07 | True |
| clip4clip | step1_mil | 0 | True |
| clip4clip | step1_parameters | 3.7252903e-09 | True |
| clip4clip | text_pooled | 0 | True |
| clip4clip | valid_video_tokens | 0 | True |

## 结论与范围

两套方法全部 25 项检查通过。UATVR 本批次全部误差为 0；CLIP4Clip 的 logits/梯度有 FP32 小差异，均满足逐元素绝对＋相对容差，参数更新最大误差 3.73e-9。此结果支持所测输入、共享权重下的计算一致性，不能证明整个训练配方、初始化过程或论文指标复现。

UATVR 的输入解码比较使用官方 **非活动 rawvideo 路径**，未验证其活动的 JPEG/随机偏移路径一致。官方初始化与当前适配器共享初值，因此未独立证明随机初始化过程相同。冻结策略独立按固定官方 trainer 规则执行并校验。官方多 rank 的 slice-only backward 与当前修正后的全局梯度明确不同；单 rank 数值通过不覆盖该差异。

实际回归命令：
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data/test_audit.py tests/semantic_retrieval/full_data/test_models_training.py
```
结果：17 passed，9.09 秒，1 个上游 TypedStorage 弃用 warning；包含真实双 Gloo rank 对 serial 全局目标的梯度回归。此次没有新 GPU 验证，也没有新科学训练。

成功 A2 的 worker 内部墙钟计时 173.94 秒，失败 A1 为 5.33 秒，合计 179.27 秒（不含测试、预留、哈希前置验证及分析开销）；GPU 0 小时。A1 因审计输入索引混淆 NPY 内容哈希和文件字节哈希而失败，已完整保留并归档；修复后新增回归测试，以新身份执行 A2。历史实验产物经核验未损坏、未修改。A1 的非法失败迁移已先 verify-run 后按合法状态链恢复。A2 在最终归档前 verify-run 通过。报告阶段再次逐文件验证 A1 的 46 项与 A2 的 51 项终态输出哈希，全部匹配。

独立审查完成：总限时、子进程清理、独立冻结策略、输入哈希与失败迁移修复无阻断项；独立核验 A2 全部 51 个输出 SHA、50 项数值检查、30 行轨迹及参数变化/温度/吞吐，均匹配。无重要过度因果归因。唯一 Minor 文案已修正为“未修改历史模型/checkpoint”，以区别本次临时模型真实 optimizer 更新。
