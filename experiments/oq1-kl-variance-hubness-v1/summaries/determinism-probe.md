# Stage 0 · 确定性契约 —— PASS（2026-08-30）

`determinism_first_200_steps_bitwise_equal_same_seed`（study.yaml stage0_qualification
最后一门）已通过：同 seed、同配置、同四卡的两次独立 arm1 训练，前 200 个优化步
**逐位一致**。

## 两次运行

| run | Run ID | 终态 | 用时 |
|---|---|---|---|
| a | `msrvtt-baseline-det_probe_arm1_a-seed000-20260829T164500Z-0761a744dc85` | `completed`, exit 0 | 6m46s |
| b | `msrvtt-baseline-det_probe_arm1_b-seed000-20260829T170000Z-0761a744dc85` | `completed`, exit 0 | 8m23s |

配置：arm1_full_akl，seed 0，train7500（trusted-v1 val1k 划分），batch 256（4×64，
AllGather 后全局负例池 256），fp32 LayerNorm + bf16 autocast + TF32，
`--strict_determinism`（cudnn.benchmark=False / deterministic=True /
`use_deterministic_algorithms(True)` / `CUBLAS_WORKSPACE_CONFIG=:4096:8`），
`--determinism_probe_steps 200`。GPU {0,1,3,4}（launch 期复核 ≥70000 MiB）。
实现 commit `1c0cc4e`，run-id 内嵌 reserve 期 HEAD `0761a744dc85`。

## 逐位一致的证据（a 与 b 完全相同的字节）

- `provenance/determinism-probe-step-losses-rank0.npy`（200 × float64）：
  两边 sha256 均为 `7039f69f722dbfa6ced52cb4594803c072b453af1a8aaa19c3795734b4035ab7`，
  `cmp` 逐字节一致。
- `provenance/determinism-probe-rank0.json`：参数摘要（全部 named_parameters 按名
  排序串接的 sha256）两边均为 `6c1f9b10323c1bee…`；文件本身 diff 为空。
- `metrics/selected_validation.json` 两边哈希一致（`197cc03a…`）。
- 数值走向（train7500, DSL n/a, t2v+v2t 训练损失）：step1 5.5449（= ln 256，
  确认四卡 AllGather 负例池正确）→ step200 4.4448，单调下降为主。

## 途中一次拒启（证据保留）

run b 首次 launch（20260829T165028Z）时研究者本人的 UGR sweep 占据 GPU0/1
（各余 58.7 GiB < 70000 MiB 下限），`scripts/oq1_arm_launch.sh` fail-closed 拒启，
探测证据保留于 `provenance/gpu-probe-launch-refused-20260829T165028Z.json`。
预留未消耗（状态仍 reserved），GPU 释放后以同一 reservation 重试成功——这正是
study.yaml `on_insufficient_gpus: refuse_launch_and_archive_probe_as_evidence`
的设计行为，顺带完成了该路径的实弹验证。

## 含义

- **Stage 0 四门全部关闭**（对账见 `archive-qualification.md`）：seed 是训练的
  全部随机性来源这一点已被验证为事实而非假设。Stage 4 三 seed 配对检验的
  方差可全部归因于 seed 与 arm，无实现噪声分量。
- 后续所有臂训练沿用完全相同的启动链
  （phase JSON → `oq1_arm_launch.sh` → `run_train_msrvtt_bg.sh` OQ1_ARM 分支）
  与 `--strict_determinism`；探针与正式训练唯一差别是 `--determinism_probe_steps`。

**结论：Stage 0 完成 4/4，Stage 1（arm0 基线 + P0 regime 门）解锁。**
