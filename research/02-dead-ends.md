# 02 · Dead Ends —— 已证伪的方向，不要重试

> 每条给出：**做了什么 / 为什么死 / 证据指针 / 什么条件下才允许重开**。
> 未满足"重开条件"就再提这些方向 = 浪费 GPU 时间，且违反 `07-protocol-constraints.md`。
> `WT/` = `.worktrees/reliable-retrieval-experiment-campaign/`。

---

## DE-1 · candidate-independent 512-D isotropic Gaussian under CSD（A2-CG）

**做了什么。** 在 frozen June/A1 retriever 上加一个 512 维 log-variance head，用
Contrastive-Symmetric-Distance (CSD) 目标训练（CSD 1.0 / grounding 0.05 / VIB 1e-4，
logvar clamp [−12, 4]，只训 263,426 个 sidecar 标量）。

**为什么死。** trusted-v1 val 上**每一个 acceptance gate 都失败**：
T2V ρ = −0.1472（负相关），T2V AUROC = 0.4064（**低于随机**），selective accuracy **非单调**
（acc@25/50/75/100 = 0.6788/0.7168/0.7312/0.7588，越去掉"高不确定"的 query 准确率越差），
prediction std = 2.5e-5。方差塌成全局常数：text var 2.602e-4 (std 2.49e-6)，
video var 2.209e-4 (std **3.37e-9**)。grounding 停在 0.1669，pairwise ranking 停在 `ln 2`。

**三条结构性根因（可复用的教训）：**
1. CSD 在**所有** pair 距离上加方差 → matching loss 全局奖励"降低方差"，query-specific 信号被全局收缩梯度淹没。
2. grounding target 是 O(1)，`mean(exp(logvar)) ≈ 2.6e-4` 是 O(1e-4)，再除以 512 → grounding 梯度被 matching 梯度碾压。
3. **结构性的：** retrieval failure 是 (query, candidate set) 这个**对**的属性；单个 pooled embedding 表达不了"是哪个 candidate 集合让它失败"。

**证据。** `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:35-39,56-69`；
`findings.md:55-66`；`PAPER.md:87,123-135`；ledger E-20~E-25。

**重开条件。** 仅当有人能给出一个**candidate-conditioned** 的 Gaussian 参数化，
且其 grounding 项与 matching 项在同一数量级（需在训练前用梯度范数实测证明）。
`PAPER.md:176` 提到的 I1（scalar vMF concentration `κ = softplus(raw)+ε`, `u = 1/(1+κ)`）
是唯一被 spec 化的候选，但**它仍是 candidate-independent 的**，因此上面的根因 3 依然适用——
在有新论据前，I1 应视为高风险而非"下一步"。

---

## DE-2 · A3 contextual risk v2（按 spec 原样）—— Stage 1 gate FAIL

**做了什么。** 在 A1 的 8 维 score 特征上追加 query mean、top-1 candidate mean、
softmax-attended top-K candidate context，128-dim MLP（186,113 个可训练标量），
保持 frozen-retriever 契约，从 A1 ckpt 初始化，3 epoch，seed 0，trusted-v1 split。

**为什么死。** 两个 gate 同时 FAIL：
- `a3_improves_strongest = false`：mean AUROC Δ = **−0.0287**，mean AURC Δ = **+0.0019**。
- `directional_rho_gte_0_20 = false`：V2T ρ = **0.19735** < 0.20（差 0.003）。

**最关键的一句：一个零参数的 entropy scorer 全面击败它。**

| scorer | mean AUROC | mean AURC | V2T AUROC | V2T ρ |
|---|---:|---:|---:|---:|
| **entropy（0 参数）** | **0.8300** | **0.0441** | **0.7883** | **0.2549** |
| one_minus_top1（0 参数） | 0.8200 | 0.0442 | 0.7581 | 0.2278 |
| A3（128-dim MLP） | 0.8013 | 0.0460 | 0.7230 | 0.1974 |
| score_variance | 0.4411 | 0.1733 | 0.4137 | −0.0786 |

**证据。** `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:11-34`；
`todo.md:14-17`；ledger E-31 / E-32。

**注意冲突（必须一起记住）：** A3 在 **2026-08-12/13 的 JSFUSION test 上曾报告 PASS**
（AUROC T2V 0.8162 / V2T 0.8290，且在 AUROC/AURC/coverage 上全面优于 A1），
见 `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:90-116`（ledger E-30）。
两者不矛盾但结论相反：那次 test **未开 DSL**、且 val 上 V2T ρ 只有 0.2007（勉强过线）；
trusted-v1 val 的 V2T 只有 35 failure / 500 query，是高方差 regime。
**要报 A3，必须两个都报。**

**重开条件。** 必须开**新的 pre-registered study**（`todo.md:45`）。且新 study 必须：
(a) 把 zero-parameter entropy/one_minus_top1 作为**强制 baseline**；
(b) 解决 V2T 的样本量问题（500 query / 35 failure 不足以支撑 ±0.03 的判定）。

---

## DE-3 · SAP encoder-internal uncertainty outputs —— 5 条断链，全部无信号

SAP = Semantic Anchor Probing（K=16 anchor + Dirichlet evidence head）。它的 4 个
uncertainty 输出（`u_mode`, `epistemic`, `logsigma`, `diversity`）**训练后全部塌成 per-video 常数**
（`logsigma_video` −1.5±0 触底、`epistemic_cont` 0.9995±5e-5、`diversity` 0.9996±5e-5、
`modal_entropy` 1.0000±0、`u_mode` 0.5794±2e-5）。
诊断出的 **5 条断链**（`todo.md:19-24`，机制细节见
`docs/superpowers/reports/2026-08-11_sap_collapse_investigation_and_fix.md:44+`）：

| # | 断链 | 为什么它杀死信号 |
|---|---|---|
| 1 | `α_dir → mu_agg → L2 归一化` | Σα（总证据量）的 magnitude 信息在 L2 归一化那一步被完全削掉，下游再也看不到"证据多少" |
| 2 | `orth_loss` 强制 anchor 正交 | anchor 被推成正交基 → anchor 间 diversity 变成常数 → "diversity as uncertainty" 这条设计自相矛盾 |
| 3 | `modal_entropy` 依赖孤儿 `α` | 那个 α 没有任何 loss 约束它（孤儿分支），于是输出恒为均匀分布（归一化熵 = 1.0000±0） |
| 4 | `logsigma` init 是常数 + MIL 只推整体变小 | 没有任何项鼓励 per-video 差异，结果全体触底 clamp min |
| 5 | `evidential_similarity` 的 `epistemic_penalty` 是 per-video 的，会整行缩放 sim_matrix | per-video 缩放不改变该行内部的排序 → 对 ranking 净效应 ≈ 0；同时梯度方向矛盾 |

**根源（第 6 条，最重要）：** SAP 的所有 uncertainty 输出**缺少直接的 failure-prediction 监督信号**
（`todo.md:24`）。更彻底的版本见下面的 DE-5。

**最终裁决（2026-08-21，study `sap-contextual-uncertainty-v1`，`status: falsified`, `do_not_resume: true`）：**
5 个 raw 描述子的 V2T direct AUROC 全部在随机附近，gate 阈值 0.55 全部 fail：
`u_mode` 0.5441 / `modal_entropy` 0.5397 / `epistemic_scale` 0.5138 / `aleatoric_scale` 0.4734 / `log_evidence` 0.4561
（后两个**弱反预测**）。把整个 `sap_uncertainty` 特征块接进 A3 sidecar，AUROC 变化
（0.0004）小于控制臂自己的 seed 间散布（0.0220）。
证据：`WT/experiments/sap-contextual-uncertainty-v1/study.yaml:282-455`。

**结论已证伪；代码已于 2026-08-24 合并进 master，SAP 在主线上彻底消失。**
2026-08-22 在分支 `feature/reliable-retrieval-experiment-campaign` 上，
经 6 个 commit（`3477da8`→`3ca9240`→`b9801a9`→`4020fc1`→`266a2b6`→`d3a3cd9`）删除：
`query_models/module_sap.py` 整文件、`UATVR.sap` / `sap_out` 分支、
`confidence_mlp` / `anchor_attn` / `anchor_ent` / `weighted_logits`，以及
MIL-NCE / evidential NLL / negative-evidence / anchor-orthogonality 四个辅助损失
和 `--w_mil --w_evidential --w_neg_reg --w_orth` 四个 CLI 开关。
**文本侧概率路径（`pie_net_text` / `prob_text` / text `logsigma`）与 `modules/modeling.py` 未动。**
替代实现：`UATVR._pooled_video_embedding(visual_output, video_mask)`（帧 token masked mean + L2 归一化，
全 padding 行安全），它**只喂 sidecar 与诊断项，不是排序组件**（`findings.md:264-266`）。

> ✅ **仓库现状（2026-08-24 更新，以此为准）：master 已 fast-forward 到 `d3a3cd9`，SAP 代码已不存在。**
> - 2026-08-24 研究者选择方案 (a)，`git merge --ff-only` 把 37 个 commit 全部并入 master
>   （master 原本 0 commit 领先于该分支，故为纯 fast-forward，无 merge commit，37 个 commit 线性保留）。
> - `query_models/module_sap.py` 已删除；全仓库（排除 `research/`）已无任何
>   `module_sap` / `SemanticAnchorProbing` 引用。
> - 曾经 import 它的 `scripts/evaluate_sap_uncertainty.py`（未跟踪的孤儿诊断脚本）
>   已移至 `research/legacy/dead-code/`——**不要试图运行它，它依赖已删除的模块。**
> - 合并后测试：**747 passed / 26 skipped**（`tvr` 环境，132.88s，全绿）。
> - 该分支在合并前已打 bundle 备份：`artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle`。
>
> **正确表述：** SAP 在 master 上已彻底删除。在任何分支上都不要试图 import 它。
> 历史缺陷 **D-06**（`findings.md:209` 与 master 实况矛盾）**已随本次合并自动消解**——
> 该文档的表述现在是对的，只是当时超前了两天。**ADR-10 已 RESOLVED**（见 `03-decisions.md`）。

**重开条件。** `do_not_resume: true`，且 study 的 conclusion 明确写了为什么不值得再试：
*"F3 shows the architecture gives sigma no path to the ranking function in the first place —
that path, not the uncertainty head, is the thing worth changing."*
（`WT/experiments/sap-contextual-uncertainty-v1/study.yaml:440-454`）
唯一未被排除的分支是"**联合训练**（unfrozen）的 SAP"，但那需要先让 σ 有一条通往 ranking function 的路，
是一个完全不同的架构改动，必须开新 study。

---

## DE-4 · `w_sap_failure_bce`（"Fix 1"）—— 一个 NO-OP 被当成了成功

**做了什么。** 在 SAP 的 `u_mode` 上加 BCE loss，标签 = 当前 batch 内该视频的 top-1 V2T 检索是否错
（`_sap_failure_bce_loss`，w=0.1）。v1 2-GPU PoC + v2 4-GPU 3-seed，全部 pre-registered gate PASS
（V2T A3 AUROC mean 0.7239，跨 seed sd 0.0013，R@1 V2T 93.0 不变）。

**为什么死。** `--contextual_risk_sidecar_only` 通过
`configure_contextual_risk_sidecar_training()` 把**所有**参数 `requires_grad=False`，
然后只重新打开 `contextual_risk_sidecar.*`。**SAP 也在冻结之列。**
所以 `w_sap_failure_bce` 这一项 loss 被计算、被加进总 loss，但它唯一的梯度通路指向
frozen SAP 参数 → **完全是空操作**。v1 和 v2 两个 campaign 里 SAP 一次梯度都没收到过。

**实证确认：** 三个 v2 winner 的 eval-only replay 返回**逐位相同**的 SAP direct AUROC
和逐位相同的 A1 failure AUROC (0.7294) —— 只有 SAP 从未移动过才可能。

**因此：** v1/v2 实际验证的是 "**score margin 能预测 retrieval error**" ——
一个标准的 selective-prediction baseline，**不是**论文所声称的 SAP / Dirichlet / anchor uncertainty 设计。
`todo.md:52,55` 记的 "V2T AUROC 0.5661 → 0.7244" **数字为真，但归因为假**，ledger 标 CAVEATED。

**另外两个已撤回的东西：**
- **"符号翻转"叙事已 RETRACTED**（2026-08-20）：ep0 那一行是**随机初始化的 contextual sidecar**
  （v2 三 seed 的 init 分别是 0.2296 / 0.6414 / 0.3461），不是"未加 Fix 1 的基线"。
  训练一个随机 head 从 0.23 涨到 0.72 是正常现象，不构成任何证据。（`findings.md:181,189`）
- **指标名已更正**（2026-08-20）：原写 "SAP u_mode AUROC"，实为
  **A3 contextual sidecar 的 `failure_risk = sigmoid(failure_logit)` 的 AUROC**
  （`main_task_retrieval.py:4637` / `prob_models/contextual_retrieval_risk.py:259`）。
  在 sap-contextual-uncertainty-v1 之前，全仓库**从未**对 `u_mode` 直接算过 AUROC。

**证据。** `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:240-258, 85-90`；
`findings.md:112-122, 181, 189`；`todo.md:57-58`；ledger E-50~E-56。

**重开条件。** 无。这条路在数学上就不存在。任何"给 SAP 加监督"的方案必须先解冻 SAP，
而那属于 DE-3 的重开条件。

---

## DE-5 · 通用教训：encoder-internal learned variance 不预测 retrieval failure

这不是一次实验，而是**四条独立证据线交汇出的结论**，应当作为本课题的默认先验：

1. **A2-CG**（CSD 512-D Gaussian）：AUROC 0.4064，低于随机。（DE-1）
2. **SAP**（Dirichlet evidence + mixture logsigma + anchor diversity）：5 个描述子 direct AUROC
   0.4561–0.5441，全在随机附近；接进 sidecar 后增量 0.0004。（DE-3）
3. **CRF 训练过程**主动抹掉了 u_video 与检索失败的相关性：ρ 从 +0.065 降到 +0.015，
   同期 u_video 的 std 反而从 0.160 涨到 0.197 —— **方差在变宽，但变宽的方向与"哪些视频难检索"无关**。
   （`docs/superpowers/tables/2026-08-11_r1_and_rho.md:7-13,27`）
4. **Stage 1 的 `score_variance` scorer**：mean AUROC **0.4411**，是六个 scorer 里唯一低于随机的。
   同一张表上零参数的 entropy 拿到 0.8300。（DE-2 的表）

**外部独立佐证：** DAB（ECCV 2026）站在相反立场，却在自己的 §4.4 里用
**top1−top2 KL margin**（一个纯 score-geometry 量）做 selective prediction，
并写下 *"DAB's uncertainty signal emerges from the bridge-induced distributional gap
rather than from text variance alone."*（`related_work/dab_eccv2026/NOTES.md:107-122`）

**机制解释（为什么会这样）：** σ 从来没有一条通往 ranking function 的路。
`modules/modeling.py:748` 计算 `retrieve_logits = logit_scale * wti_logits`，概率分支在推理时被丢弃。
唯一作用在 logsigma 上的常驻压力是 `vib_loss`（权重 1e-4），其 KL-to-N(0,I) 项
（`prob_models/probemb.py:157`）在 `logsigma = 0` 处取最小 —— **它把方差推向常数**。
（`WT/experiments/sap-contextual-uncertainty-v1/study.yaml:390-403`，finding F3）

**因此，任何新提案若属于"在 encoder 里学一个方差/浓度/证据量，然后期望它预测检索失败"，
默认视为 dead end，必须先给出 F3 的反例（即 σ 如何进入 ranking function）才允许立项。**

---

## DE-6（次要）· 把 crf_head 直接 fine-tune 进 retriever 以求 R@1 提升

CRF 训练与 R@1 是**互相拖累**的：在 HEAD arch 上 5 epoch CRF 训练把 T2V R@1 从 21.1 拉到 12.9
（`docs/superpowers/tables/2026-08-11_r1_and_rho.md:17-23,30`）。
正确的做法已经找到并验证：**权重移植**（transplant），0 训练成本、R@1 逐位不变、audit 1.0
（ledger E-63）。不要再走 fine-tune 路线来"顺便"提 R@1。
