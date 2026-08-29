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
