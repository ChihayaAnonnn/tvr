# 03 · Decision Records（ADR 风格）

> 格式：**日期 / 决策 / 背景 / 后果**。倒序无所谓，这里按时间正序排，便于理解演化。
> `WT/` = `.worktrees/reliable-retrieval-experiment-campaign/`。

---

## ADR-01 · 2026-08-11 —— 采用 frozen-retriever + sidecar 契约（放弃改打分路径）

**决策。** 方案 A 第一阶段固定为**旁路（sidecar）训练**，不再延续 "route 1+3" 那种
修改检索打分的做法。契约固定为：

```
Init checkpoint:      ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4
Architecture:         --use_uatvr_full --use_ada_norm
Trainable parameters: retrieval_risk_sidecar.* only
Retrieval score path: raw WTI; no confidence/risk/vacuity multiplication
w_retrieval_risk:     0.05
retrieval_risk_temperature: 1.0
Final retrieval eval: DSL T=2.0
Acceptance:           retrieval unchanged; rho >= 0.20; AUROC > 0.5; selective accuracy improves
```

**背景。** 此前所有"把 uncertainty 乘进 sim_matrix"的尝试都同时污染了检索指标和不确定性指标，
无法分开判定。而 June arch + DSL T=2 已经把 R@1 打到发表门槛（52.6/52.5），
不应该再冒险动它。

**后果。**
- 所有后续 sidecar（A1 / A2-CG / A3 / CSD logvar）都继承这个契约，检索指标可比。
- 2026-08-17 用 `scripts/verify_frozen_retrieval_equivalence.py` 把它升级为**可证明的**契约：
  同一精度下三份 ckpt 的 pre-DSL WTI 矩阵逐位相同（ledger E-01）。
- 副作用：sidecar 无法改变 ranking，因此**任何"A3 提升了 R@1"的说法都是错的**
  （`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md:53`）。

**来源。** `docs/superpowers/tables/2026-08-11_r1_and_rho.md:253-268`。

---

## ADR-02 · 2026-08-11 —— CRF probe 走"权重移植"而非 fine-tune

**决策。** 从 audit-passing 的 CRF donor ckpt（HEAD arch, R@1=14）提取 `crf_head` +
`video_aleatoric` 权重，直接覆盖到 June ckpt（R@1=52.6）上，得到 hybrid ckpt。

**背景。** 三个候选方案都要付出 R@1 风险或训练时间；实测 5 epoch 的 CRF 训练会把
HEAD arch 的 T2V R@1 从 21.1 拉到 12.9。而 `crf_head` 的输入是 `[B,T,D]` frame features，
June arch 与 HEAD arch 的 `visual_cls` **是同一个 CLIP 输出张量**，所以权重可以无缝复用。

**后果。** 0 训练成本，R@1 保持 52.6/52.5，CRF audit `three_pair_accuracy = 1.0000`。
同时这个移植本身变成了论文的一个 claim（"probe 是模块化、backbone-agnostic 的"）。
代价：hybrid ckpt 落在 `/tmp/hybrid_ckpt/`，**易失**（见 `06-assets.md`）。

**来源。** `docs/superpowers/tables/2026-08-11_r1_and_rho.md:148-179,225-251`。

---

## ADR-03 · 2026-08-12 —— 选定 trusted-v1 split，JSFUSION test 封存

**决策。** 所有 selection / 消融 / 阈值 / K / 校准只在 `dataloaders/splits/msrvtt_trusted_v1_seed0.json`
（SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`，本 session 实测一致）
定义的 validation split 上做；MSRVTT JSFUSION test 与 MSVD test 保持 sealed，
每个被接受的 ckpt 只允许评一次。

**背景。** 早期结果在 JSFUSION 上反复评测、反复挑 checkpoint，已经无法区分"真信号"与"过拟合到 test"。

**后果。**
- 正面：A2-CG 的失败是**无歧义**的（val 与 test 都失败）；A1 的 val T2V ρ (0.514) **低于** test (0.576)，
  这本身是"未过拟合 val"的证据（`PAPER.md:117`）。
- 负面：trusted-v1 val 的 **V2T 只有 500 query / 35 failure**，AUROC CI 半宽约 ±0.09，
  这个低功效 regime 直接导致了 A3 在 val 上 FAIL 而在 test 上 PASS 的矛盾（ledger E-30 vs E-31）。
  **多加 seed 不会缩小这个 CI**（同一批 500 query）。
- 该约束被写进 `AGENTS.md:11`（规则 6）。

**来源。** `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:10`；`AGENTS.md:11`；
`WT/experiments/sap-contextual-uncertainty-v1/study.yaml`（power note）。

---

## ADR-04 · 2026-08-12 —— 把 `paired_invariance` 从硬 gate 降级为 audit-only 字段

**决策。** commit `3bd3df8`：`paired_invariance` 保留为**审计字段**，
frozen-boundary 的**唯一 load-bearing 检查**改为权重层的 `non_a3_mismatches`（SHA-256 逐张量比对）。

**背景。** Lineage 5 上 6 个 forward digest 中有 5 个出现 bit 级漂移（raw WTI、text/video means、
A1 failure risk、A1 vacuity），但 `non_a3_mismatches` 为空、检索 rank 与 R@K 逐位相同。
根因是 DataLoader 顺序 + 4-rank DDP 在 ULP 级的非确定性，不是 sidecar 污染。
若继续把 bit-exact 当硬边界，**这条 pipeline 下的每一个 candidate 都会被拦掉**。

**后果。** A3 v1 得以完成并注册。代价是"逐位可复现"这条更强的保证被放弃；
后来 2026-08-17 的 `verify_frozen_retrieval_equivalence.py` 在**受控单进程**下重新拿回了它（ledger E-01）。

**来源。** `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:33,125`。

---

## ADR-05 · 2026-08-18 —— Stage 1 gate FAIL，按 SDD 协议在 Stage 1 停止 campaign

**决策。** `evaluate_stage1_gate` 返回 `next_authorized_stage = null`，
contextual-risk-v2 campaign **就地终止**并 finalize（commit `eb2ee0b`，archive summary + registry 一致）。
不准备 Stage 2，不用不同超参 silently re-run Stage 1。

**背景。** A3 winner（ep 2）未能击败零参数 baseline：mean AUROC Δ = −0.0287，
mean AURC Δ = +0.0019；V2T ρ = 0.19735 < 0.20（差 0.003）。

**后果。**
- Controller 的 `assert_phase_authorized` 会拒绝任何 Stage 2 预留（predecessor gate 未过），
  所以这不是"自愿约束"而是**代码强制**。
- 任何 A3/SAP 重设计必须开**新的 pre-registered study**（`todo.md:45`）。
- 产出一个可写进论文的 negative result（见 `02-dead-ends.md` DE-2）。
- 未被采纳的诱惑：把 V2T ρ 阈值从 0.20 降到 0.19，或者换个 seed 重跑 —— **两者都是 post-hoc selection。**

**来源。** `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:11-30,80-96`；`todo.md:13-17,26`。

---

## ADR-06 · 2026-08-18 —— 恢复冻结的 `{0,2,3,4}` GPU 拓扑

**决策。** commit `c448240`，5 个文件 14 处 `0,1,2,3` → `0,2,3,4`
（launcher 1 处、`main_task_retrieval.py` 3 处、tests 10 处）。
采用**直接编辑**而非 revert 三个原始 commit（`05f57fa` / `62069e0` / `3fe60cf`），
结果等价但单 commit 更清晰。测试验证 145/145 通过。

**背景。** Stage 1 期间因外部服务占用 GPU 4 而临时切到 `{0,1,2,3}`；Stage 1 已 finalize，
"no work pending"，应回到 qualified 状态。

**后果。** 冻结拓扑重新是 `{0,2,3,4}`。
⚠️ **但注意冲突：** 后续的 sap-failure-supervision-v2 与 sap-contextual-uncertainty-v1
都声明 `topology: [0, 1, 2, 3]`，并注明 "GPU 4 remains occupied by an external VLLM service"
（`WT/experiments/sap-contextual-uncertainty-v1/study.yaml`，`hardware_policy`）。
**启动任何实验前必须先确认当前实际可用拓扑，不要照抄文档。**

**来源。** `todo.md:27-30`；`WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:93-96`。

---

## ADR-07 · 2026-08-21 —— 关闭 SAP 线（study `sap-contextual-uncertainty-v1` 标记 falsified）

**决策。** `status: falsified`, `completeness: partial`, `do_not_resume: true`。
6 个臂只跑了 1 个（`full_sap` seed 0，2/3 epoch，随后 rc=137 SIGKILL），
但 primary gate（raw SAP 描述子 direct AUROC ≥ 0.55）**独立地、无歧义地失败**，
故不再申请续跑授权。

**背景。** 一次 code trace 发现：A3 sidecar 从来收不到任何 SAP uncertainty 输出，
所以 sap-failure-supervision v1/v2 验证的是 score geometry 而不是 SAP 设计。
本 study 就是补上那根线、然后测它。

**后果。**
- 结论：*"Encoder-internal evidential uncertainty (K=16 anchors, Dirichlet evidence, mixture logsigma)
  is not a usable failure signal for this retrieval model, and the A3 sidecar's demonstrated skill
  comes from top-k score geometry."*
- **明确的 scope limit：** 这只证伪了 **PRETRAINED（冻结）** 的 SAP uncertainty。
  联合训练的 SAP 未被测试，而且是**有意不测**的 —— 因为 F3 表明架构根本没给 σ 一条通往
  ranking function 的路，"那条路，而不是 uncertainty head，才是值得改的东西"。
- 这个决策直接生成了 `00-north-star.md` 的 thesis 前半段。

**来源。** `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:282-455`。

---

## ADR-08 · 2026-08-22 —— 在分支上**删除** SAP 代码（master 尚未合并）

**决策。** 在分支 `feature/reliable-retrieval-experiment-campaign` 上分 6 个阶段删除 SAP：
`3477da8`(phase 0 安全网) → `3ca9240`(phase 1 退役 no-op 的 `w_sap_failure_bce`) →
`b9801a9`(phase 2 从 A3 sidecar 解线) → `4020fc1`(phase 4 partial，删三个死 launcher) →
`266a2b6`(phase 3 删模块与辅助损失) → `d3a3cd9`(phase 3 step 5，CLI flag 清理)。

删除范围：`query_models/module_sap.py` 整文件；`UATVR.sap` 与 `sap_out` 分支；
`confidence_mlp` / `anchor_attn` / `anchor_ent` / `weighted_logits`；
MIL-NCE、evidential NLL、negative-evidence、anchor-orthogonality 四个辅助损失；
CLI 开关 `--w_mil --w_evidential --w_neg_reg --w_orth`（`findings.md:249-252` 另记 `--num_queries`）。
**未动：** 文本侧概率路径（`pie_net_text` / `prob_text` / text `logsigma`）、`modules/modeling.py`。

**背景。** ADR-07 关闭了 SAP 线；同时确认 SAP **从来不在检索打分路径上**
（排序完全由 `weighted_token_wise_intersection` 产生，打分函数里不存在任何池化后的视频向量），
因此删除是 metric-neutral。

**后果。**
- 实测验证 metric-neutral：删除后 `d3a3cd9` 与对照 `4020fc1` 在 trusted-v1 val 上
  T2V R@1 76.5 / V2T R@1 93.0 / T2V risk AUROC 0.8285 / V2T 0.7253 **逐位相同**。
- 替代实现 `UATVR._pooled_video_embedding()`（帧 token masked mean + L2 归一化），
  只喂 A3 sidecar / CSD log-var sidecar / aux `video_mu` 诊断，**不是排序组件**。
- 行为变化（必须声明）：全量训练（不开 sidecar）时训练目标从 `wti_logits * confidence`
  变成裸 `wti_logits`。历史 run 全部开着 sidecar，故不受影响；但**未来的全量重训不等价于 A1 那个配方**。
- 遗留风险：旧 ckpt 仍含 47 个 `sap.*` key（8,943,361 参数），走宽容的
  `init_preweight`（`--init_model`）没问题。见 `07-protocol-constraints.md` PC-6。
- ⚠️ **删除只在分支上，master 未删且仍 import**（本 session 实测，详见
  `01-evidence-ledger.md` 的 "E-46 仓库现状冲突" 表与缺陷 D-06）。
  **这个决策目前只完成了一半 —— 见 ADR-10。**

**来源。** `findings.md:206-267`；`WT` 的 `git log --oneline`；
`WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`。

---

## ADR-10 · 2026-08-24 已决 —— 分支 `feature/reliable-retrieval-experiment-campaign` 的 37 个 commit 全部合并进 master

**状态：RESOLVED（2026-08-24，研究者选择方案 (a)）。**

**决策与执行。** 研究者明确选择"(a) 整个合并进 master"。执行方式为
`git merge --ff-only feature/reliable-retrieval-experiment-campaign`：
因为 `git rev-list --count feature/...--master` = **0**（master 完全包含在分支中），
这是一次纯 **fast-forward**，master 由 `2185fb2` 推进到 `d3a3cd9`，
不产生 merge commit，37 个 commit 以线性历史完整保留。
61 files changed, 13901 insertions(+), 575 deletions(-)。

**后果。**
- `query_models/module_sap.py` 已删除，SAP 在主线彻底消失（ADR-08 至此才真正落地）。
- 三个 study 的控制面（`sap-contextual-uncertainty-v1` / `sap-failure-supervision-v1` / `v2`）、
  `contextual-risk-v2` 的 7 个 phase JSON 与全部 stage1 summaries、
  `run_reliability_campaign.py` / `reliability_campaign.py` / `retrieval_risk_baselines.py`、
  以及 2026-08-18→08-21 的训练日志，全部进入 master。
- 孤儿脚本 `scripts/evaluate_sap_uncertainty.py`（未跟踪，import 已删除的模块）
  移至 `research/legacy/dead-code/`。
- **合并后测试：747 passed / 26 skipped**（`/home/xujie/.conda/envs/tvr/bin/python -m pytest tests/ -q`）。
  这同时为 `01-evidence-ledger.md` 记录的"测试基线冲突"给出了当前主线的权威数字。
- 缺陷 **D-06** 自动消解：`findings.md:209` 关于 SAP 已删除的表述现在与 master 相符。

**收尾（2026-08-24，研究者"全删吧"）。** 合并完成后分支与其 worktree 已冗余，全部清除：
- `git worktree remove --force .worktrees/reliable-retrieval-experiment-campaign`（回收 **37 G**）；
- `git branch -d feature/reliable-retrieval-experiment-campaign`（`-d` 而非 `-D`，
  git 确认 fully merged 后才删，ref 曾指向 `d3a3cd9`）。

删除前的两项核查与抢救：
1. worktree 里 8 个 ckpt **全部不在 `experiments/contextual-risk-v2/legacy/critical-runs.json`**，
   也**不在 `artifacts/contextual-risk-v2/legacy/inventory.jsonl`**（0 行命中）——
   即从未被登记为 critical 资产。文档侧的引用（`06-assets.md` / `01-evidence-ledger.md` /
   两个 `study.yaml` / `2026-08-21-sap-removal.md`）都是 SAP 死线的 provenance 叙述，非活依赖。
2. 每个 ckpt 目录的小文件已抢救至
   **`artifacts/dead-lines/sap-fix1-campaign-provenance/`**（35 files, 572 K）：
   `runtime.log` / `training_manifest.json` / `training_args.json` / `checkpoint_epoch.*.published.json`。
   丢弃的是 `pytorch_model.bin.*`、`pytorch_opt.bin.*`，以及 3 GB 量级的
   `best_contextual_risk_checkpoint.json` / `contextual_risk_candidates.json` / `a3_paired_baseline.json`。
   **代价：SAP/Fix-1 这批 run 不再能重新评测，只能查超参与日志。** 这是可接受的，
   因为 `02-dead-ends.md` DE-3/DE-4 已明确禁止重试该方向。

分支 ref 虽已消失，**37 个 commit 仍完整存在于 master**（`d3a3cd9` 就是当前 HEAD），
另有 `artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle` 可独立恢复该 ref。
删除后 `git fsck --connectivity-only` 干净，`git worktree list` 只剩主工作区。

**残留风险。** master 仍比 `origin/main` 领先约 177 个 commit 且**没有 upstream、未推送**。
合并前已打 bundle：`artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle`（4.5 MB，
`git bundle verify` 报 complete history），但那只覆盖到 `d3a3cd9`，
**不覆盖此后 master 上的任何新提交**。推一个 remote 仍是未决事项。

**下面是决策前的原始记录，保留备查。**

---

**（原始待决记录）状态：OPEN（尚未决定）。**

**事实（本 session 实测）。**
- `git rev-list --count master..feature/reliable-retrieval-experiment-campaign` = **37**。
- `git branch -a --contains d3a3cd9` 只返回该本地分支 → **不在任何 remote 上**。
- `git for-each-ref refs/heads/` 显示两个本地分支**都没有 upstream**；
  master 本身也比 `origin/main` 领先 140 个 commit 且未推送。
- 也就是说：**这 37 个 commit 没有任何远端备份，全部只存在于这台机器的磁盘上。**

**这 37 个 commit 里有什么（已知的）。**
- SAP 删除的 6 个 commit（`3477da8`→…→`d3a3cd9`，见 ADR-08）。
- Fix 1 的实现 `16d4f96`（`_sap_failure_bce_loss`）与 pre-removal baseline `67fba49`。
- 三个 study 的控制面：`experiments/sap-failure-supervision-v1/`、
  `experiments/sap-failure-supervision-v2/`、`experiments/sap-contextual-uncertainty-v1/`
  （后者的 `study.yaml` 是 SAP 证伪的**唯一权威记录**，`02-dead-ends.md` DE-3/DE-4 全部依赖它）。
- launcher `run_train_msrvtt_fix1_4gpu_bg.sh`。
- 报告 `docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`（47 个 `sap.*` key 的清单）。
- 关联的非 git 资产（在该分支 worktree 目录下，本身不受 git 跟踪）：
  `WT/ckpts/ckpt_msrvtt_2026081{8,9,20}_*fix1*`（6 个 ckpt 目录）与 `WT/logs/2026081{8,9}`、`WT/logs/202608{20,21}`。

**待决的问题。** (a) 合并进 master？(b) 只挑 `experiments/` 与 `docs/` 的科研记录 cherry-pick，
把 SAP 删除留在分支上？(c) 至少先推一个 remote 备份？

**建议（未执行，需研究者决策）：** 无论最终怎么处置，**先做备份**。
证伪记录（`sap-contextual-uncertainty-v1/study.yaml`）与 checkpoint key 清单是不可重建的科研证据。

**来源。** 本 session 的 `git rev-list` / `git branch -a --contains` / `git for-each-ref` 实测；
`WT/` 的 `git log --oneline -8`。

---

## ADR-09 · 2026-08-22 —— 放弃 SAP 这条线，但继续做视频文本 / 多模态检索

**决策。** UATVR **未归档**。UATVR 与 UGR 是两个并行独立项目。UATVR 继续做
video-text / 多模态检索方向，**仅放弃 SAP 这一条技术路线**。

**背景。** ADR-07/08 之后需要明确"是不是整个项目停掉"。答案是否。

**后果。** 新的研究命题在同日的 brainstorm 中形成，见 `00-north-star.md`
（variance vs geometry 分离 + conformal risk-controlled selective retrieval，
以 CRF probe 作为 shift 的分层变量）。

**来源。** 跨 session 记忆 `MEMORY.md → project_direction_shift.md`（仓库内无对应文档，`[未溯源]`）。
