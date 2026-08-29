# oq1-kl-variance-hubness-v1 · 控制面

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
