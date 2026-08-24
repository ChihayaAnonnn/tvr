# 06 · Assets —— 权威路径清单

> 所有路径相对仓库根 `/data2/hxj/project/UATVR`，除非另行标注绝对路径。
> 🔴 **`WT/` 已不存在（2026-08-24 删除）。** 原 `.worktrees/reliable-retrieval-experiment-campaign/`，
> 其 37 个 commit 已 fast-forward 进 master（ADR-10），worktree 与 ckpt 已删除回收 37 G。
> 下文任何 `WT/…` 路径都**只作历史记录读，不要去访问**。代码/配置去 master 找；
> 那批 ckpt 的超参与日志见 `artifacts/dead-lines/sap-fix1-campaign-provenance/`。
> **本文件的 SHA-256 全部为本 session `sha256sum` 实测值**，与 `PAPER.md` 记载冲突处见 ledger D-02。
>
> ✅ **原 `/tmp` 易失产物已于 2026-08-24 全部转存进仓库**（见下文第 4 节），`/tmp` 不再是任何证据的唯一载体。

---

## 1. Checkpoints

| 角色 | 路径 | winner 文件 | SHA-256（实测） |
|---|---|---|---|
| Base retriever（June arch） | `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/` | `pytorch_model.bin.4` | `ee79f8a2e7be460f63cfb16404bc4b24dc1f622021f4d7d7f9a24dc96d030964` |
| **A1 risk sidecar**（主结果） | `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/` | `pytorch_model.bin.4` | `a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607` |
| A2-CG（负结果，保留） | `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/` | `pytorch_model.bin.2` | `e539108aa8296eaf1e3c71d968117b5f153cd8d7c444282c8b39e9998c8635be` |
| A3 contextual risk v1 | `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/` | `pytorch_model.bin.3` | `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c` |
| CRF donor（crf_head 供体） | `ckpts/ckpt_msrvtt_crf_20260810_025752_crf_ladder_full/` | `pytorch_model.bin.2` | `306b15eb843d037cb1cda54668d41a018602efda2142394c2f0009f9bf1e04bd` |
| **Hybrid（CRF + 检索双达标）** | `ckpts/hybrid_june_crf_transplant/`（2026-08-24 从易失的 `/tmp/hybrid_ckpt/` 转存，SHA 已校验一致） | `pytorch_model.bin.hybrid`（+ `training_args.json`） | `3fd971f6266e9e015ac85a7fc64f403b894a44216e1c808ae0ab5587ec22ba2a` |
| Stage 1 A3 winner（FAIL 的那个） | `artifacts/contextual-risk-v2/msrvtt/baseline/msrvtt-baseline-validation_baselines-seed000-20260818T040000Z-1fdd78f384d1/checkpoints/`（**注册 run，未受 worktree 删除影响**；同目录另有 `metrics/`、`logs/train.log`、`arrays/sha256/`） | `pytorch_model.bin.2` | `e503946f0c7d861a2ea6e3e49eecb30e8f51ecb67a83e994fd1d8a26938d565f` |
| A3 DSL-T2 post-hoc eval | `ckpts/ckpt_msrvtt_20260813_214730_a3_dsl_t2_eval/` | — | 未测 |
| MSVD June baseline | `ckpts/ckpt_msvd_20260813_034603_msvd_june_baseline/` | `pytorch_model.bin.0`–`.4` | 未测 |
| CLIP backbone asset | `ViT-B/16`（`pretrained_clip_name`） | — | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

**⚠️ `PAPER.md:51` 写的是 ViT-B/32，是错的**（D-03）。两份 `training_args.json` 均为 `ViT-B/16`。

### Fix 1 系列 ckpt —— 🔴 **已于 2026-08-24 删除，权重不可恢复**

> 原在 `WT/ckpts/`。worktree 删除时权重与大 JSON 转储一并丢弃（37 G）。
> **这批 run 不可重新评测**，只保留了每个 run 的 `runtime.log` / `training_manifest.json` /
> `training_args.json` / `checkpoint_epoch.*.published.json`，位于
> **`artifacts/dead-lines/sap-fix1-campaign-provenance/<run_name>/`**（35 files, 572 K）。
> 删除依据见 ADR-10 收尾段：8 个 run 均不在 `critical-runs.json` 与 `inventory.jsonl`。
> 下面的清单保留为**命名与 winner-shard 索引**，不再是可用路径。

原 `WT/ckpts/` 下：
`ckpt_msrvtt_20260818_215201_fix1_smoke` · `…_20260818_215400_fix1_smoke` ·
`…_20260818_220203_fix1_poc_w0p1`（v1 PoC winner `bin.2`）·
`…_20260819_212850_fix1_v2_s0`（winner `bin.2`）·
`…_20260820_150233_fix1_v2_s1`（winner `bin.0`）·
`…_20260820_174558_fix1_v2_s2`（winner `bin.2`）·
`…_20260821_164109_sapfeat_full_sap_s0`（SAP 证伪 study 唯一跑起来的臂，winner `bin.1`）。

> 分支上的 37 个 commit 现已在 master（HEAD `d3a3cd9`），另有 bundle
> `artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle`。但 **master 本身仍未推送任何 remote**（见 ADR-10）。

---

## 2. 数据与 split

| 名称 | 路径 | 说明 |
|---|---|---|
| MSRVTT 标注 | `/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json` | `training_args.json:data_path` |
| MSRVTT 帧源 | `/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/` | `features_path`（另有 `msrvtt_224_3fps`） |
| MSRVTT train csv | `/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv` | `run_train_msrvtt_crf_bg.sh:77` |
| JSFUSION test csv | `/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv` | 1000 视频 × 1 caption；**V2T 侧有 1012 分母异常**（D-01/OQ-5） |
| **trusted-v1 val split** | `dataloaders/splits/msrvtt_trusted_v1_seed0.json` | SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`；500 视频 × 20 caption = 10000 pair |
| CRF dev audit split | `dataloaders/splits/crf_dev_audit_seed0.json` | preflight 校验用 |
| **MSRVTT CRF ladder** | `data/generated/msrvtt_crf/{crf30,34,36,38,42,44,46,50}` | 2.3 GB（本地生成，非 git 追踪） |
| MSVD CRF ladder | `data/generated/msvd_crf/` | 同上 |
| trusted-v1 物化目录 | `data/generated/msrvtt_trusted_v1/` | |
| MSVD 数据 | `/data2/hxj/data/MSVD/{YouTubeClips,desc_files}` | ⚠️ `.avi` decoder 有 `fps=0` bug（OQ-4） |

> 论文/study 里报告的 CRF ladder 是 6 级 `{clean, 34, 36, 42, 44, 50}`；
> 磁盘上多出 `crf30/38/46` 三级，**未进入任何 gate**。

---

## 3. Registered artifacts（SDD Scheme A 数据平面）

根：`artifacts/contextual-risk-v2/msrvtt/feature_ablation/`

| run | 目录名 |
|---|---|
| A1（主结果来源） | `msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/` |
| A3 v1 | `msrvtt-feature_ablation-a3_selected_legacy-seed000-20260813T031640Z-3bd3df8befb3/` |

每个 run 下 `checkpoints/{training_args.json, final_test.json}`。
**`final_test.json` 是 PAPER Table 1/Table 3 全部数字的唯一真源** —— 引用数字时优先读它，不要抄 PAPER。

> ⚠️ Fix 1 v1/v2 与 SAP study **从未生成数据平面 artifact**（原 `WT/artifacts/` 就不存在，
> 也无 `registry.jsonl`）。正因为没注册，2026-08-24 删 worktree 时它们的权重无法保留；
> 现在只剩 `artifacts/dead-lines/sap-fix1-campaign-provenance/` 里的日志与超参。
> **教训：没走 `manage_experiment_run.py` 注册的 run，在清理时就是可丢弃的。**

---

## 4. ✅ 原 /tmp 易失产物 —— 已全部转存进仓库（2026-08-24）

> 转存时 `sha256sum -c` 全部通过。`/tmp` 下的原件可能已随重启消失，**一律引用下表的仓库内路径**。

| 现路径（权威） | 原 /tmp 路径 | 内容 | 为什么重要 |
|---|---|---|---|
| `ckpts/hybrid_june_crf_transplant/pytorch_model.bin.hybrid` | `/tmp/hybrid_ckpt/` | CRF+检索 hybrid 权重 | **PAPER Table 4 的唯一载体** |
| `ckpts/hybrid_june_crf_transplant/training_args.json` | 同上 | 对应配置 | |
| `artifacts/rescued_from_tmp/hybrid_audit/crf_audit_report.json` | `/tmp/hybrid_audit/` | CRF ladder audit（15/15 pairwise） | E-60/E-61 的证据 |
| `artifacts/rescued_from_tmp/hybrid_audit/crf_audit_per_video.tsv` | 同上 | per-video Δu | E-62 的证据 |
| `artifacts/rescued_from_tmp/frozen_equiv/frozen_retrieval_equivalence.json` | `/tmp/frozen_equiv/` | 逐位等价性证明 | **PAPER §3.1 核心 claim（E-01…E-06）的唯一证据** |
| `artifacts/rescued_from_tmp/frozen_equiv/run.log` | 同上 | 运行日志 | |
| `artifacts/rescued_from_tmp/frozen_equiv/sim_{june,a1,a3,june__repeat}_{fp32,bf16}.npy` | 同上 | 8 个 10000×500 相似度矩阵 | **OQ-1 hubness 测量直接复用这 8 个矩阵即可，不必重跑 forward** |

> `research/legacy/PAPER.md` 的 §3.1 与 Appendix A 已同步改成上表的仓库内路径（2026-08-24）。

---

## 5. 脚本

| 用途 | 路径 | 状态 |
|---|---|---|
| 冻结检索器逐位等价性验证 | `scripts/verify_frozen_retrieval_equivalence.py` | **untracked** |
| CRF ladder 评测/审计 | `scripts/evaluate_crf_ladder.py` | tracked |
| uncertainty↔retrieval 相关性诊断 | `scripts/diagnose_uncertainty_retrieval_correlation.py` | tracked |
| ~~SAP 不确定性评测~~ | 🔴 **已删除**（`scripts/evaluate_sap_uncertainty.py`，2026-08-24） | 它 import 的 `query_models.module_sap` 在 master 已不存在，无法运行且无恢复价值。SAP 线见 `02-dead-ends.md` DE-3/DE-4 |
| 构建 MSRVTT CRF ladder | `scripts/build_msrvtt_crf_ladder.py` | tracked |
| 构建 MSVD CRF ladder | `scripts/build_msvd_crf_ladder.py` | tracked |
| 构建 trusted-v1 split | `scripts/build_msrvtt_trusted_split.py` | tracked |
| SDD run 生命周期管理 | `scripts/manage_experiment_run.py` | tracked（reserve/finalize，见 AGENTS.md） |
| 出图 | `paper/figs/make_figures.py` | tracked |

---

## 6. Launcher（仓库根）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `run_train_msrvtt_bg.sh` | MSRVTT June baseline（**MEMORY.md 记的标准训练模板**：`EXPERIMENT_DESC=… CUDA_VISIBLE_DEVICES=… bash run_train_msrvtt_bg.sh`） | tracked |
| `run_train_msrvtt_risk_sidecar_bg.sh` | A1 sidecar | **untracked** |
| `run_train_msrvtt_contextual_risk_bg.sh` | A3 contextual risk | tracked |
| `run_train_msrvtt_csd_logvar_bg.sh` | A2-CG | tracked |
| `run_train_msrvtt_crf_bg.sh` | CRF ladder 训练（含 preflight 校验 split 与 ladder 完整性） | tracked |
| `run_train_msvd_june_bg.sh` | MSVD baseline（单卡，`DATA_PATH=/data2/hxj/data/MSVD`，`CUDA_ID=1`，`MASTER_PORT=29561`） | **untracked**；受 PC-3 fail-closed 约束 |
| `run_train_msvd_risk_sidecar_bg.sh` | MSVD sidecar | **untracked**；受 PC-3 约束 |
| `run_train_msrvtt_fix1_4gpu_bg.sh` | Fix 1 v2 三 seed | 已随 ADR-10 合并进 master，tracked |

---

## 7. 图

`paper/figs/`：`fig1_architecture` · `fig2_a1_vs_a2cg` · `fig3_crf_ladder` ·
`fig4_risk_coverage` · `fig5_dsl_sweep`，每个都有 `.pdf` + `.png`；生成脚本 `make_figures.py`。
fig4 用的是 `final_test.json` 里的真实 per-coverage 点，不是拟合曲线。

---

## 8. 关键源码位点（改动前先看）

| 位点 | 作用 |
|---|---|
| `main_task_retrieval.py:4065-4088` `_eval_autocast_context` | 按 sidecar flag 选 bf16/fp32 —— **0.3–0.5 点 R@1 散布的根因** |
| `main_task_retrieval.py:4637` | A3 `failure_risk = sigmoid(failure_logit)` 计算点 |
| `main_task_retrieval.py:6326` / `:6357` | **严格** `load_state_dict` —— 见 PC-6，不得放宽 |
| `modules/until_module.py:100` `init_preweight` | 宽容加载路径（`--init_model` 走这里，可吃下 47 个 `sap.*` 键） |
| `prob_models/contextual_retrieval_risk.py:259` | A3 sidecar 头 |
| `prob_models/uncertainty_protocol.py` `validate_selected_checkpoint` | 结构性封死了旧 ckpt 走严格路径 |
| `modules/modeling_mulit.py:21` | master 上仍 `import SemanticAnchorProbing`（`:361` `confidence_mlp`，`:1020` `sap_out`） |
| `query_models/module_sap.py` | master 上**存在**（7798 字节）；只在分支上被删 |

---

## 9. 文档索引

| 文件 | 内容 | 可信度 |
|---|---|---|
| `PAPER.md` | 论文 v0.1（2026-08-13） | ⚠️ 已知 3 处缺陷 D-01/D-02/D-03，且 §7 Future Work 已过时 |
| `findings.md` | 逐条发现流水 | ⚠️ `:112-122` 有作废/更正框；`:209` 有缺陷 D-06 |
| `todo.md` | 2026-08-18 生成，§2 是协议约束原文 | 可信 |
| `progress.md` | 只覆盖 2026-08-13 | 已过时，用 `05-session-log.md` |
| `task_plan.md` | 2026-08-13 的计划 | 已过时 |
| `AGENTS.md` | 实验归档强制规则 1–8 | **强制，仍生效** |
| `related_work/dab_eccv2026/NOTES.md` | DAB 批判性阅读 | 可信（外部数字均标了出处） |
| `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md` | Stage 1 HALT 决定 | 权威 |
| `WT/experiments/sap-contextual-uncertainty-v1/study.yaml` | SAP 证伪 study（`do_not_resume: true`） | 权威 |
| `WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md` | 47 个 `sap.*` 键分析 | 权威 |
| `docs/superpowers/tables/2026-08-11_r1_and_rho.md` | 2026-08-11 逐小时流水 | 权威 |
