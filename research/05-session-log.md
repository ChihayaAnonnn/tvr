# 05 · Session Log（倒序，append-only）

> 规则：**只在最上方追加新条目，不修改历史条目。** 需要更正历史结论时，
> 写一条新条目说明"撤回/更正了什么"，并同步更新 `01-evidence-ledger.md` 的状态标记。
> 早期条目由 `progress.md` / `todo.md` / study.yaml 的日期反向重建，标注为 *(重建)*。

---

## 2026-08-24 (4) —— `research/` 纳入版本管理；孤儿脚本删除

**`scripts/evaluate_sap_uncertainty.py` 已真删**（研究者授权）。它 import 的
`query_models.module_sap` 在 master 已不存在，无法运行；`research/legacy/dead-code/`
随之清空移除。`06-assets.md` 对应行改为"已删除"。

**`research/` + `related_work/` 提交进 master。** 此前 `research/`（本 anti-drift 体系本体）
一直是 untracked —— 一次误删就全没了。同时提交：
- `AGENTS.md` 的强制前置阅读指针（与 `research/README.md` 配套，必须同一次提交）；
- `related_work/dab_eccv2026/{NOTES.md,dab.txt}` —— `00-north-star.md` 与 E-70…E-72
  按**行号**引用 NOTES.md，不提交就是悬空引用。
- `related_work/dab_eccv2026/dab.pdf`（2.8 MB）**故意不提交**，第三方论文原文保持本地。

**仍未决：** master 领先 `origin/main` ~177 commit，**无 upstream、从未推送**。
现在 `research/` 至少在 git 里了，但仍只存在于这一台机器。

---

## 2026-08-24 (3) —— 合并后清理：worktree + 分支删除，回收 37 G

**执行。** 研究者授权"全删吧"。
`git worktree remove --force .worktrees/reliable-retrieval-experiment-campaign` +
`git branch -d feature/reliable-retrieval-experiment-campaign`（fully-merged 校验通过）。

**删除前核查。** 8 个 ckpt 均**不在** `critical-runs.json`，也**不在**
`artifacts/contextual-risk-v2/legacy/inventory.jsonl`（0 行命中）——从未登记为 critical。
文档里的引用（`06-assets.md`、`01-evidence-ledger.md`、两个 `study.yaml`、
`docs/superpowers/plans/2026-08-21-sap-removal.md`）都是死线 provenance 叙述，不是活依赖。
> 这一步是上一轮"审计漏查 inventory.jsonl"的教训：**删 ckpt 前必须同时查
> `critical-runs.json` 和 `inventory.jsonl` 两处。**

**抢救。** 35 个小文件 → `artifacts/dead-lines/sap-fix1-campaign-provenance/`（572 K）：
每个 run 的 `runtime.log` / `training_manifest.json` / `training_args.json` /
`checkpoint_epoch.*.published.json`。丢弃权重与 3 GB 量级的 JSON 转储
（`best_contextual_risk_checkpoint.json` / `contextual_risk_candidates.json` /
`a3_paired_baseline.json`）——**这批 run 从此不可重新评测，只可查超参与日志。**

**验证。** `git worktree list` 只剩主工作区；`git branch` 只剩 master @ `d3a3cd9`；
`git fsck --connectivity-only` 无输出；`.git` = 7.4 M；`/data2` 70%（2.2 T avail）；
bundle `reliable-retrieval-campaign_d3a3cd9.bundle` 完好（verify: complete history）。

**仍未决。**（a）master 领先 `origin/main` ~177 commit、无 upstream、未推送，
bundle 只覆盖到 `d3a3cd9`，**不含 `research/`**；（b）`research/legacy/dead-code/
evaluate_sap_uncertainty.py` 是否直接删掉；（c）`06-assets.md` 里的 `/tmp` 易失路径
与 `PAPER.md` Appendix A 的 `/tmp/hybrid_ckpt` 指向仍需改为 `ckpts/hybrid_june_crf_transplant/`。

---

## 2026-08-24 (2) —— ADR-10 结案：37 个 commit 合并进 master，SAP 主线删除落地

**决策。** 研究者选择方案 (a)：整个分支合并进 master。

**执行。** `git rev-list --count feature/...--master` = 0，即 master 完全包含在分支中，
所以是纯 **fast-forward**（`git merge --ff-only`），master `2185fb2` → `d3a3cd9`，
无 merge commit，37 个 commit 线性保留。61 files changed, +13901 / −575。
合并前确认分支不触碰工作区里未提交的 `.gitignore` / `AGENTS.md`，无冲突。

**结果。**
- `query_models/module_sap.py` 删除；全仓库（除 `research/` 的历史叙述外）无 `module_sap` /
  `SemanticAnchorProbing` 残留引用。
- 三个 SAP study 控制面、`contextual-risk-v2` 的 7 个 phase JSON 与全部 stage1 summaries、
  `reliability_campaign.py` / `retrieval_risk_baselines.py` / `run_reliability_campaign.py`、
  2026-08-18→08-21 训练日志，全部进入 master。
- 孤儿脚本 `scripts/evaluate_sap_uncertainty.py`（未跟踪、import 已删模块）
  移入 `research/legacy/dead-code/`。
- **测试：747 passed / 26 skipped**，132.88s，`/home/xujie/.conda/envs/tvr/bin/python`。
  注意默认 `python3` 缺 `einops`，跑测试必须用 `tvr` 环境。

**文档同步。** ADR-10 → RESOLVED（保留决策前原始记录备查）；DE-3 的"仅在分支删除"警告改写为
"master 已删除"；E-46 与缺陷 D-06 状态更新，D-06 标为 RESOLVED。

**残留。** master 比 `origin/main` 领先约 177 个 commit，**无 upstream、未推送**。
bundle 只覆盖到 `d3a3cd9`，不含此后的提交。推 remote 仍未做。

---

## 2026-08-24 (1) —— 工作区清理：`.git` 156G → 7.9M，回收约 210 GB

**发生了什么。** 按 `CLEANUP-PROPOSAL.md` 执行清理。审计**推翻了原始假设**：无需任何 git 历史改写——
整个可达仓库（所有 ref + reflog + worktree index）只有 40.0 MB / 2,897 objects，
`.git` 里其余全是不可达垃圾；另发现一个**此前无人知晓的 76.5 GB 孤立 Git-LFS 缓存**
（`.git/lfs/objects`，20 个对象，最大 9.3 GB），系某次尝试用 LFS 往 GitHub 推 checkpoint 的残留。
删除前三重独立验证：`git lfs ls-files --all` 为空、任何 ref 中都不存在 `.gitattributes`、
无任何 LFS 指针文件。

**先做的保护性操作（清理之前）。**
- `/tmp/hybrid_ckpt/` → `ckpts/hybrid_june_crf_transplant/`，SHA-256 校验一致。
  这是 PAPER Table 4 移植结果的唯一 artifact，原先躺在 98% 满的 `/` 上。
- `/tmp/frozen_equiv/`（306 MB，含 §3.1 逐位等价性的唯一证据 + **8 个可直接用于 OQ-1 hubness 测量的
  相似度矩阵**）与 `/tmp/hybrid_audit/` → `artifacts/rescued_from_tmp/`，9 个文件全部 sha256 校验通过。
- ADR-10 的无备份分支 → `artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle`（4.5 MB），
  `git bundle verify` 报告 "records a complete history"。**ADR-10 的处置决策仍未做，只是不再有丢失风险。**
- 两个 detached worktree 的独有内容（各 13 行 `train_msrvtt.sh` diff + 2.5M/1.9M 独有日志）
  → `artifacts/rescued_worktrees/`；其 5.5 GB ckpt 各自移入 `ckpts/` 后才注销 worktree。

**执行内容。** Tier 0（垃圾 pack + 各类 cache 1.86 GB）、Tier 1（`git gc --prune=now` 回收 76.93 GB、
删除孤立 LFS 缓存 76.5 GB、注销 3 个已完全合并的 worktree）、Tier 2 中经批准的两项
（11 个无引用 ckpt 39.28 GB、`research_refs/` 13.42 GB）。
**未执行**：`deploy_qwen/`（60 GB）、`cache_dir/`（17.9 GB）、`artifacts/` 三项保留。

**过程中纠正的一处审计疏漏。** 审计称 11 个候选 ckpt "无任何引用"，实测每个都有 2 处命中：
一处是 `CLEANUP-PROPOSAL.md` 自身（自引用），另一处是审计**漏查的**
`artifacts/contextual-risk-v2/legacy/inventory.jsonl`。核查后确认全部 11 个在该 inventory 中
classification 均为 `historical`（非 `critical`，全库共 76 条 critical），且均不在
`experiments/contextual-risk-v2/legacy/critical-runs.json` 中，其 SHA-256 已在 inventory 留档，
删除方才放行。

**验证结果。** `.git` 156G → **7.9M**；`git fsck --connectivity-only` 无输出（干净）；
`master`(2185fb2) 与 `feature/reliable-retrieval-experiment-campaign`(d3a3cd9) 两个 ref 完好；
必留的 `.worktrees/reliable-retrieval-experiment-campaign` 未受影响；`/data2` 由 73% 降至 70%。

**文档整理。** 根目录五份历史文档移入 `research/legacy/` 并附 `README.md` 说明各自的已知缺陷；
`research/README.md` 的"相关但不在本目录"段同步更新。

**下一步。** 与 2026-08-23 条目相同：OQ-1 / OQ-5 的先后仍待研究者决定；ADR-10 的处置仍待决策。
另新增两项待办：（a）`06-assets.md` 中 `/tmp` 易失清单需按本次转存结果更新路径；
（b）`PAPER.md` Appendix A 引用的 `/tmp/hybrid_ckpt/` 路径应改指 `ckpts/hybrid_june_crf_transplant/`。

---

## 2026-08-23 —— 建立 `research/` 持久记录目录

**发生了什么。** 通读 `PAPER.md` / `findings.md` / `todo.md` / `progress.md` / `task_plan.md` /
`related_work/dab_eccv2026/NOTES.md` / `AGENT(S).md` / `docs/superpowers/{reports,tables,specs}` /
`WT/experiments/*/study.yaml` 与 `WT/experiments/contextual-risk-v2/summaries/`，
蒸馏成 8 份 anti-drift 文档。过程中对多个数字与路径做了**直接核验**（`sha256sum`、
解析 `final_test.json` / `frozen_retrieval_equivalence.json`、读 `training_args.json`、`git ls-tree`）。

**新发现的缺陷（原文档里没有）。**
- D-02：`PAPER.md:95` 把 A1 sidecar 的 SHA-256 当成了 base ckpt 的 SHA。
- D-03：`PAPER.md:51` 写 CLIP-ViT-B/32，实际是 **ViT-B/16**。
- D-06：`findings.md:209` 声称 SAP 已删除，但 master 上仍在且仍被 import。
- ADR-10：`feature/reliable-retrieval-experiment-campaign` 的 37 个 commit 无任何远端备份。

**产物。** `research/{README,00-north-star,01-evidence-ledger,02-dead-ends,03-decisions,
04-open-questions,05-session-log,06-assets,07-protocol-constraints}.md`；
`AGENTS.md` 末尾追加指针块。

**下一步。** 研究者决定 OQ-1（hubness 测量）与 OQ-5（分母对账）的先后；
同时对 ADR-10 做出处置决策（建议先备份）。

---

## 2026-08-22 —— uncertainty × multimodal 领域调研 + 新 thesis 提出；SAP 代码删除

**发生了什么。**
1. 做了一轮 uncertainty × multimodal 的 landscape survey，形成新的研究命题：
   *representation variance 与 predictive uncertainty 是两套独立机制；概率 embedding 的
   accuracy 增益主要是隐式 hubness 校正；可用的 uncertainty 信号完全来自 candidate-set score geometry*，
   建设性一半是 conformal risk-controlled selective retrieval + CRF 分层的 conditional coverage。
   **该 thesis 是 PROPOSED，未验证。**
2. 在分支 `feature/reliable-retrieval-experiment-campaign` 上完成 SAP 删除（6 个 commit）。
3. 确认项目方向：UATVR 未归档，继续做 video-text / 多模态检索，**仅放弃 SAP 这条线**。
4. 更新 `findings.md`：新增 §2.7（SAP 删除 + "mu_video 从未进入检索打分"的 dataflow 更正），
   并把 C4 的描述从 "SAP + SpatialEnhancer + AdaNorm + DSL" 改为 "SpatialEnhancer + AdaNorm + WTI + DSL"。

**产物。** commits `3477da8` / `3ca9240` / `b9801a9` / `4020fc1` / `266a2b6` / `d3a3cd9`；
`findings.md §2.7`（`findings.md:206-267`）；
thesis 本身**只存在于跨 session 记忆，仓库内无文档**（`[未溯源]`）。

**下一步（当时）。** 把 thesis 变成可执行的 pilot。→ 已写入 `00-north-star.md`。

*来源：`findings.md:17,206-267`；`WT` 的 `git log`；`MEMORY.md → project_direction_shift.md`。*

---

## 2026-08-21 —— SAP 证伪定案 + DAB 关联工作笔记

**发生了什么。**
1. 跑 study `sap-contextual-uncertainty-v1`：16:37 第一次启动 29 秒内 fail-closed
   （`experiment_tracking._CONTEXTUAL_FEATURE_BLOCKS` 不认识 `full_sap`/`sap_only`，无 GPU 时间浪费）；
   16:41 第二次启动 run `20260821_164109_sapfeat_full_sap_s0`；17:38 被 SIGKILL(rc=137)，
   原因未定（dmesg 不可读，主机内存 20/1007 GB，GPU 峰值 5.64 GB，**不像资源限制，最像外部干预**）。
   driver 遵守 fail-closed，放弃剩余 5 个臂。
2. 尽管只跑了 1/6 个臂，primary gate 独立失败：SAP 5 个描述子 V2T direct AUROC
   0.4561–0.5441（阈值 0.55）。study 标记 `status: falsified`, `do_not_resume: true`，findings F1–F5。
3. 写 `related_work/dab_eccv2026/NOTES.md`：DAB（arXiv 2607.20984, ECCV 2026）的批判性阅读，
   提出 hubness 假设，并指出其 §4.4 从相反立场佐证了我们的负结果。
4. 执行 control arm equivalence check：三个 v2 winner 的 eval-only replay **逐位复现**
   0.7230 / 0.7233 / 0.7254 → 反证 SAP 从未被训练过。

**产物。** `WT/experiments/sap-contextual-uncertainty-v1/study.yaml`（含 outcome 块）；
`WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`；
`related_work/dab_eccv2026/{NOTES.md,dab.pdf,dab.txt}`；
ckpt `WT/ckpts/…20260821_164109_sapfeat_full_sap_s0/pytorch_model.bin.1`。

**下一步（当时）。** 删除 SAP 代码。→ 2026-08-22 完成。

---

## 2026-08-20 —— Fix 1 v2 三 seed 结果 + 指标名更正 + 撤回"符号翻转"

**发生了什么。**
1. 完成 sap-failure-supervision-v2 的 seed 1 / seed 2（4-GPU {0,1,2,3}）。
   三 seed V2T A3 AUROC 0.7230 / 0.7233 / 0.7254，mean **0.7239**，sd **0.0013**，
   最差 CI95 lower 0.6300，R@1 V2T 全部 93.0。**全部 pre-registered gate PASS。**
2. **指标名更正：** 原写 "SAP u_mode AUROC" 是错的；被测量的是 **A3 contextual sidecar 的
   `failure_risk = sigmoid(failure_logit)` 的 AUROC**（`main_task_retrieval.py:4637` /
   `prob_models/contextual_retrieval_risk.py:259`）。全仓库当时没有任何地方对 `u_mode` 直接算过 AUROC。
   数字未变，只改名与归因。
3. **撤回"符号翻转"证据：** ep0 那一行是随机初始化的 contextual sidecar
   （三 seed 的 init 分别 0.2296 / 0.6414 / 0.3461），不是"未加 Fix 1 的基线"。

**产物。** ckpts `WT/ckpts/ckpt_msrvtt_20260820_150233_fix1_v2_s1/pytorch_model.bin.0`、
`WT/ckpts/ckpt_msrvtt_20260820_174558_fix1_v2_s2/pytorch_model.bin.2`；
`WT/logs/20260820/{150233,174558}_fix1_v2_s*_…`；`findings.md §2.6` 的两个更正框。
**数据平面未生成**（`registry.jsonl` / `artifacts/` 至今不存在）。

**下一步（当时）。** 补 w=0 对照臂。→ 实际走向：2026-08-21 发现整条路径是 NO-OP，对照臂已无意义。

---

## 2026-08-19 —— Fix 1 v2 启动（4-GPU 三 seed 串行），seed 0 完成

**发生了什么。** Study `sap-failure-supervision-v2`，topology {0,1,2,3}，
launcher `run_train_msrvtt_fix1_4gpu_bg.sh`，其余配置与 v1 相同，3 seeds 串行。
seed 0 winner = ep3，V2T A3 AUROC 0.7230，CI95 [0.6300, 0.8097]。

**产物。** `WT/ckpts/ckpt_msrvtt_20260819_212850_fix1_v2_s0/pytorch_model.bin.2`；
`WT/logs/20260819/212850_fix1_v2_s0_…`。

*来源（重建）：`findings.md:154-165`。*

---

## 2026-08-18 —— Stage 1 finalize (FAIL) + 拓扑恢复 + Fix 1 v1 PoC

**发生了什么。**
1. Stage 1 A3（attempt 5）训练完成，3 epoch，winner = `pytorch_model.bin.2`。
   **gate FAIL**：`a3_improves_strongest = false`（mean AUROC Δ −0.0287，mean AURC Δ +0.0019）；
   `directional_rho_gte_0_20 = false`（V2T ρ 0.19735）。`next_authorized_stage = null`，
   campaign 按 SDD 协议在 Stage 1 停止。核心发现：**零参数 entropy scorer（mean AUROC 0.8300）
   击败 128-dim A3 MLP（0.8013）**。
2. 诊断 SAP + uncertainty 设计，定位 **5 个断链**（见 `02-dead-ends.md` DE-3）。
3. 确认 Stage 1 训练时 `contextual_risk_sidecar_only=True`，SAP 前向被跑但**未接收任何 loss**。
4. 协议清理：finalize Stage 1（commit `eb2ee0b`）；恢复冻结的 `{0,2,3,4}` GPU 拓扑
   （commit `c448240`，5 文件 14 处，测试 145/145 通过）。
5. 晚间跑 Fix 1 v1 PoC（2-GPU {2,3}，w=0.1，3 epoch）：V2T A3 AUROC 0.5661 → 0.7175(winner ep2) / 0.7244(ep3)，
   R@1 V2T 稳定 93.0。**为跑通删除/no-op 了 4 处 A3 protocol guards**（见 `07-protocol-constraints.md` PC-7）。

**产物。** `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md` + `stage1-gate.json`；
`todo.md`（本文件生成于当日）；
`WT/ckpts/ckpt_msrvtt_20260818_220203_fix1_poc_w0p1/pytorch_model.bin.2`；`WT/logs/20260818/`。

**下一步（当时）。** 4-GPU 多 seed 复现 Fix 1。→ 2026-08-19/20 完成。

---

## 2026-08-17 —— frozen-retriever 等价性证明

**发生了什么。** 写 `scripts/verify_frozen_retrieval_equivalence.py`，在同一 canonical 架构、
同一 pinned 精度下比较 June / A1 / A3 三份 ckpt 的 pre-DSL WTI 相似度矩阵（10000×500）：
**逐位相同**（fp32 SHA `8ebf27ae…`，bf16 SHA `25ac139d…`，含 repeat-run determinism control）。
定位 0.3–0.5 点 R@1 散布的根因为 `main_task_retrieval.py:4065-4088` 的 `_eval_autocast_context`
按 sidecar flag 选择 bf16 / fp32。

**产物。** `scripts/verify_frozen_retrieval_equivalence.py`（untracked）；
`/tmp/frozen_equiv/`（⚠️ 易失：JSON + 8 个 `sim_*.npy`）；`PAPER.md:53`；`findings.md:33-38`。

*来源（重建）：`PAPER.md:53`；`findings.md:33-38`。*

---

## 2026-08-13 —— 论文初稿 PAPER.md v0.1 + 5 张图

**发生了什么。**
1. 盘点 project docs / plans / reports；定位 A1 sidecar 数值证据；写 `findings.md`、`task_plan.md`、`progress.md`。
2. 派研究 agent 做 2024–2026 竞品扫描（结果进 PAPER §2 与 Appendix C）。
3. 出 5 张图（fig1 architecture / fig2 A1 vs A2-CG / fig3 CRF ladder /
   fig4 risk-coverage（用 `final_test.json` 的真实 per-coverage 点）/ fig5 DSL sweep）。
4. 写完 `PAPER.md` 全部章节。
5. 同日另跑 A3 winner 的固定 DSL T=2 post-hoc 评测：T2V R@1 **52.9**（vs 历史 52.6，+0.3 ≈ 3 个 query，
   报告自己注明"不应过度解读"，且明确 **A3 不能改变 ranking**）。

**产物。** `PAPER.md`（251 行）；`findings.md`；`task_plan.md`；`progress.md`；
`paper/figs/fig{1..5}_*.{pdf,png}` + `paper/figs/make_figures.py`；
`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md`；
`ckpts/ckpt_msrvtt_20260813_214730_a3_dsl_t2_eval/`。

*来源：`progress.md:3-26`；`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md`。*

---

## 2026-08-12 —— A2-CG 负结果注册 + A3 v1 训练与 JSFUSION one-shot

**发生了什么。**
1. A2-CG（CSD 512-D log-variance）训练并注册为**失败**：val 上每个 gate 都 fail
   （T2V ρ −0.1472，AUROC 0.4064，selective accuracy 非单调，方差塌成常数）。
   选中 ckpt 完整保留 A1（frozen boundary 审计通过）。
   协议偏离：JSFUSION 被评了两次（continuation 前后），仅描述性。
2. A3 contextual risk v1 训练（3 epoch + continuation 到 5），val 上 epoch idx 3 **PASS**
   五项检查（V2T ρ 0.200715，靠 0.000069 的差距选中）；JSFUSION one-shot（**DSL=false**）：
   A3 在 AUROC/AURC/coverage 上全面优于 A1。
3. 协议变更 ADR-04：`paired_invariance` 降为 audit-only（commit `3bd3df8`）。

**产物。** `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/`（winner `bin.2`，SHA `e539108a…`）；
`ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/`（winner `bin.3`，SHA `2e6f662b…`）；
`docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md`；
`docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md`。

*来源：两份 report 文件本身。*

---

## 2026-08-11 —— R@1 目标达成（June + DSL T=2）+ CRF hybrid + A1 契约确立

**发生了什么。**（这一天信息量最大，`docs/superpowers/tables/2026-08-11_r1_and_rho.md` 是逐小时流水账）
1. 诊断 SAP 不确定性坍缩，写 `2026-08-11_sap_collapse_investigation_and_fix.md`（路 1+3 修复）。
2. 发现所有 ckpt 的 ρ ≪ 0.20 → 触发决策规则 `drop_uncertainty_claim`，卖点转向 compression-robust probing。
3. Arch restoration（恢复 June 的 SAP + SpatialEnhancer + AdaNorm）。
4. **DSL 温度扫描：T=2.0 给出 T2V 52.6 / V2T 52.5，R@1 硬指标达成。**
5. **CRF head 权重移植方案（ADR-02）**：hybrid ckpt 同时满足 R@1 52.6/52.5 与 CRF audit 1.0000。
6. 23:04 确立 **A1 retrieval-risk sidecar 契约（ADR-01）**，当晚 23:19:59 起训。

**产物。** `docs/superpowers/tables/2026-08-11_r1_and_rho.md`；
`docs/superpowers/reports/2026-08-11_sap_collapse_investigation_and_fix.md`；
`/tmp/hybrid_ckpt/pytorch_model.bin.hybrid` + `/tmp/hybrid_audit/crf_audit_report.json`（⚠️ 均在 /tmp）；
`ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/`；`logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log`。

---

## 条目模板（复制到最上方使用）

```markdown
## YYYY-MM-DD —— <一句话标题>

**发生了什么。** <做了什么、观察到什么。数字必须带 split/protocol。>

**产物。** <ckpt / log / report / commit / figure 的具体路径。/tmp 下的要标 ⚠️ 易失。>

**证据登记。** <是否需要在 01-evidence-ledger.md 新增/改状态的行？写明 ID。>

**撤回/更正。** <本次是否推翻了此前的结论？写明推翻了哪条，并同步改 ledger 状态为 RETRACTED/CAVEATED。>

**下一步。** <一句话。若是新实验，指出它对应 04-open-questions.md 的哪个 OQ。>
```
