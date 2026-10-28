# oq1-kl-variance-hubness-v1 · 控制面

> **当前结论：HALT / inconclusive（2026-08-30）。** Stage 1 regime gate 未通过，
> Stage 2–6 不再执行，test 保持 sealed。见 [最终判定](summaries/stage1-arm0-baseline.md)。
> 冻结的 `study.yaml` 保留注册时状态，不是继续执行许可。

预注册协议见 `PREREGISTRATION.md`（散文体，签署于 2026-08-28）与 `study.yaml`
（冻结，commit 20a4a1d 起）。归档机制沿用 `docs/agent/experiment-artifacts.md` 的
Scheme A 生命周期（reserve-phase → verify-run → worker → transition-run →
finalize-phase），数据面在 `artifacts/oq1-kl-variance-hubness-v1/`。

与 contextual-risk-v2 的差异：
- 无 legacy metrics contract——所有 metrics 一律走 `schemas/metrics.schema.json`。
- 设备不硬编码：phase JSON 里携带 reserve 期 GPU 探测结果（manifest 绑定），
  `scripts/oq1_stage0_launch.sh` 在 launch 期 fail-closed 复核（<70000 MiB 空闲即拒启）。
- 数据分区：`data/generated/msrvtt_trusted_v1_val1k/`（构建脚本
  `scripts/build_val1k_partition.py`，manifest 含全部输入/输出 sha256）。

## 归档层 stage 槽位映射（2026-08-29 记录）

归档契约（`experiment_artifacts.py` ALLOWED_STAGES）的 stage 词表是跨 study 固定的
（`tests/test_experiment_artifacts.py:374` 断言其与旧 study schema 枚举一致），故本
study 的协议 stage 名映射到归档槽位；study.yaml 的 stages/stop_gates 名称不变，
仅 run-id 与 artifact 路径使用右列槽位：

| study.yaml stage | 归档槽位 |
|---|---|
| stage0_qualification | baseline（variant qual_*） |
| stage1_baseline_regime | baseline（variant arm0） |
| stage2_correction_envelope | calibration |
| stage3_arm_ablation | loss_ablation |
| stage4_multiseed | multiseed |
| stage5_mechanism | robustness |
| stage6_test_confirmation | final |
| stage7_external_replication | （v1 不预算；届时另定） |

## Arm 实现的冻结常数与协议解释（2026-08-29 记录，commit 1c0cc4e）

study.yaml 只冻结"各臂共享同一训练配置、打分为按项加权的高斯 KL cost"；下列实现
自由度在此一次性定死，之后与 study.yaml 同等对待（改动 = 新 study）：

1. **μ/σ 头**：复用仓内 PIENet + UncertaintyModuleImage（PCME 系）。μ 经 l2 归一化；
   头输出 `logsigma = log σ`（`prob_models/tensor_utils.py` 采样约定），KL 里的
   log σ² = 2·logsigma。**log σ 统一 clamp 到 [−3, 3]**（两模态、全部臂相同）。
2. **打分**：`modules/oq1_probabilistic_scoring.py`，cost 逐维分解后**对维度取均值**
   （等价于统一 1/512 温度缩放，保持排序不变），logits = `logit_scale · (−cost)`。
   已对暴力展开逐元素验证（<1e−12），arm1 与闭式 KL(v‖t) 恰差被弃掉的常数 −D/2。
3. **损失**：所有臂（含 arm0）只用双向 CrossEn（t2v + v2t 各半）。原 June 路径的
   MIL-NCE 与 VIB 辅助损失**全部关闭**——arm0 无概率头无从计算，且 VIB 会把 σ 拉向 1，
   直接干扰被测机制。lam 权重为 {0,1} 固定常数；0 权重项仍乘进计算图
   （零梯度参与），保证 DDP find_unused_parameters=False 与臂间参数量不变式成立。
4. **arm0**：不构造 pie/uncertain 模块（"无概率头"按字面执行）；概率臂不构造 WTI
   的 text/video_weight_fc。arm0 与概率臂参数量不同是注册设计的一部分
   （invariant 只约束 arm1/2/4/5/6 相互一致）。
5. **precision: fp32 的解释**：沿用本 study 归档运行时的既有绑定
   `environment.precision = clip_layer_norm_precision`，即 **LayerNorm fp32 +
   master 权重 fp32**；bf16 autocast 与 TF32 与生产 June 基线一致保持开启。
   （contextual-risk-v2 的"exact fp32"只约束其 A3 sidecar，不迁移到本 study。）
6. **确定性模式**（stage0 契约与后续所有臂训练）：`--strict_determinism` →
   cudnn.benchmark=False、cudnn.deterministic=True、
   `torch.use_deterministic_algorithms(True)`、要求 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
7. **launcher 链**：phase JSON command → `scripts/oq1_arm_launch.sh`（launch 期
   GPU 复核 + verify-run）→ `run_train_msrvtt_bg.sh` 的 `OQ1_ARM` 分支（固定命令行；
   启动前重建 trusted split 并按已提交 manifest 校验 val1k 分区字节一致）。
   训练集 train7500 经 dataloader 的 subset 校验（必须 ⊆ trusted train8500，仅
   oq1 臂激活时放宽相等为子集）。
8. **确定性探针**：`--determinism_probe_steps N` 截断在 N 步并落盘
   `provenance/determinism-probe-step-losses-rank0.npy` + 参数/buffer sha256；
   两次同 seed 运行逐字节一致即过门。
9. **评测精度 = 精确 fp32（2026-08-30 实现审查追加，commit 912a452）**：bf16
   autocast 评测把相似度矩阵量化到每行 ~198 个唯一分数（1000 候选），大规模精确
   并列使 rank/hubness 指标失真（E-89b/E-90，顺带解开 OQ-5 的 1012 分母）。oq1
   runs 的评测一律走 exact-fp32 路径；训练数值模式不变（bf16 autocast + TF32）。
   受影响的 attempt-1 phase（seed000 completed 但 eval 矩阵为 bf16）已 finalize
   存档，Stage 1 分析只使用 r2 phase 的数据。
10. **流程教训**：源码提交前必须先清账所有活跃预留（归档契约禁止 reserved 状态
    跨源码漂移做任何转换，含取消）；控制器 stdout 不可接管道（head 之类会经
    SIGPIPE 杀死看护进程，worker 虽存活但失去排水保障）。
