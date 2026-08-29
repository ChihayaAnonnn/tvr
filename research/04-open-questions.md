# 04 · Open Questions —— 每条都带"什么结果能结案"

> 新实验从这里挑，不要从 `PAPER.md §7` 的 Future Work 挑（那份列表写于 2026-08-13，
> 其中 A3 与 I1 已经分别被 `02-dead-ends.md` DE-2 / DE-1 削弱或证伪）。
> 优先级按当前 north-star（`00-north-star.md`）排序。

---

## OQ-1 · Hubness 测量：概率方法的 R@1 增益有多少是隐式 hubness 校正？

**问题。** DAB 类方法（以及我们自己的 DSL rerank）带来的检索增益，是否可以完全由
"query-independent 的 per-candidate 修正"（加性偏置 $b_j$，或推广为仿射 $a_j s + b_j$）解释？

**为什么现在问。** 这是 north-star thesis 的诊断半部分。
已有的旁证：DAB 的增益形状是 R@1 +4.3 / R@5 +9.7 / R@10 +7.8 / MnR −64%
（ledger E-71），以及它的两个组件单独都会**降低** R@1（E-72）。
我们自己的 DSL rerank 贡献 +2.6 T2V R@1（E-08），DSL 本身就是一种 hubness 校正。

**⚠️ Scoping 事实（2026-08-28 核实）。** `artifacts/rescued_from_tmp/frozen_equiv/` 的
8 个 `sim_*.npy`（10000×500, float64）只有 **2 个不同矩阵**（fp32 一个 / bf16 一个）——
frozen-retriever 契约使 June/A1/A3 的相似度矩阵逐位相同。因此**仓库内不存在任何
"真正改变打分的概率检索器"**：F-A/F-B 的严格检验必须复现一个外部方法（见 Tier 1）。
原文所写 `/tmp/frozen_equiv/`（已转存）路径作废，以上为权威路径。

**怎么做——四级阶梯（成本递增，任一级都可能提前杀死 thesis）：**

- **Tier −1 · 增益形状普查（零 GPU，先做）。** 对 UATVR / T-MASS / ProTA / DUQ /
  DiffusionRet / DAB（可加 PAU），各自相对**它自己的确定性 baseline**（优先取论文自己
  消融表的同 backbone 行，如 DAB Table 5a），登记 (ΔR@1, ΔR@5, ΔR@10, ΔMnR/MnR_base)
  与不对称指数 `A = (ΔR@5+ΔR@10)/2 − ΔR@1`。每行必须同时登记四个混淆变量：
  backbone / split / 帧数 / **是否叠 DSL·QB-Norm 后处理**（叠了且 baseline 未叠 → 该行作废）。
  注意 DAB Table 5a 是 5 epoch、主表 10 epoch，不能混用。
  → 结案：签名（A>0 + MnR 崩塌）**家族普适** vs **仅 KL-cost 类（DAB）独有**；
  后者则 thesis 收窄，Tier 1 的 claim 随之改。

- **Tier 0 · 加性/仿射修正族的能力包络（无 GPU，val-only，PC-4 不涉及）。**
  在 fp32 矩阵上：(i) label-free 臂——DSL(T=2.0) / inverted softmax / QB-Norm / Sinkhorn，
  各算 Δ 指标与 `S(N_k)`（k=5,10）；(ii) **oracle 臂**——用标签 split-half 拟合最优
  加性 $b_j$ 与最优仿射 $(a_j, b_j)$，给出**两个族各自的增益上界**。
  开工前必须核实矩阵的 ground-truth 对齐规则与 MnR 口径（不要假设 query i → video ⌊i/20⌋）。
  → 若连 oracle 都造不出 "R@1 小涨 + R@5/R@10 大涨 + MnR 腰斩" 的形状，thesis 在
  零 GPU 成本下即被证伪。

- **Tier 1 · 只复现 DAB 的"概率头"（~1 GPU-day，关键实验）。** 不复现完整 DAB
  （无代码，`NOTES.md:6`），只复现 Table 5a 的 `w/ Prob. only` 行——该行已完整复刻签名
  （46.9→44.3 R@1↓，MnR 14.3→7.4）。头：L2-norm → LayerNorm+Linear ×2 出 μ / log σ²
  （bias init −2.0，clip [−6,2]，监控 clip 命中率），PAC loss 双向 softmax，τ 共享初始化。
  **打分实现必须把 KL 的四项分开加权**（(A) 纯候选 −½log σ_v²、(B) 纯 query、
  (C) 双线性、(D) Mahalanobis），消融只改 `lam_*` 配置：

  | 臂 | 配置 | 检验 |
  |---|---|---|
  | 1 | 全开 | 复刻签名（实现正确性锚点） |
  | 2 | lam_A=0 | **落点臂**：加性候选偏置消掉后签名是否消失 |
  | 3 | 只留 (D) | 纯匹配项 |
  | 4 | 臂3 + σ_t≡1 | 应≈确定性 baseline |
  | 5 | 对称 KL | 两个 log σ 项相消 → 加性偏置天然为零的 KL 变体 |
  | 6 | 同头换 ELK 打分 | Expected Likelihood Kernel：log(σ_t²+σ_v²) 双侧耦合，**不存在**可分离 candidate-only 项——概率打分的"无偏置"对照组 |

  注意 (B) 项是 V2T 方向的镜像偏置：T2V/V2T 必须分开报，臂 2 只预期杀 T2V 签名。
  每 epoch 记录三个机制指纹：$b_j=\tfrac12\sum_d\log\sigma^2_{v_j,d}$ 的分布、
  **corr(b_j, N_k^{baseline}(j))**（thesis 预测显著为负——方差头学到的就是 hub 名单）、
  各臂 `S(N_k)` 轨迹。评测期做增益分解 ΔS = b·1ᵀ + R，分别评 S+b 与 S+R。
  分布固定为**高斯 + 非对称 KL(Q_v‖P_t)**（分解最干净且为 DAB 原设）；vMF（κ = 乘性族
  代表）仅作可选扩展。1a-hybrid（WTI+KL 融合）**禁止**——归因会被污染。
  **Tier 0 归并项（2026-08-28）**：确定性 baseline 的相似度矩阵训完必须保存，并在其上
  重跑 Tier 0 的全套包络臂（DSL/逆softmax/QB-Norm/Sinkhorn + split-half 加性/仿射
  oracle，脚本 `scripts/oq1_tier0_correction_envelope.py` 可复用）——这是高 hubness
  regime 下的真正 F-A/F-B 检验，val 上因 regime 错配未能执行（E-88）；同时在此验证
  被移交的判据"S(N_k) 随增益操作下降且与增益单调相关"。

  **backbone 与 regime 的最终设计（2026-08-28，预注册稿）：** 原计划 backbone 取
  1b（X-Pool，因 Table 5a 的 46.9 是外部锚）。但该锚在 **1K-A test 上，PC-4 下够不着**，
  为一个 Stage 6 之前收益为零的锚引入未经本仓协议审计的外部代码库不划算。
  **改为：主线用本仓 June/WTI 确定性检索器，X-Pool 复现降为可选 Stage 7（不在 v1 预算内）。**
  regime 问题另解：hubness 差异主要来自**评测几何**而非 backbone（同 backbone 同 DSL，
  val −0.65 / test +2.6），故从 8500 个 train 视频中划出 **1000 视频 × 1 caption** 的
  `msrvtt_trusted_v1_val1k`（与 1K-A 结构同构、与 test 零交集，train 缩到 7500），
  Stage 1–5 全在这个 hubness-y regime 内做完，**test 一次不碰**。
  另有两条被写死的实现约束：主指标一律 **no-post-processing**（DSL/QB-Norm 会吸收掉
  待测的隐式修正，否则 F-A 假阴性）；**λ 权重固定 0/1 不扫描**（扫描 = 在 val 上选择，
  会污染 Stage 6 的一次性 test）。
  **PC-4**：预注册已落地并于 **2026-08-28 签署**：`experiments/oq1-kl-variance-hubness-v1/`
  （`PREREGISTRATION.md` + `study.yaml`，status: **registered**，commit `20a4a1d`，此后冻结）；
  其 stage4+stage5 双门通过是本 campaign 内 MSRVTT test 解封的唯一授权路径。
  签署前唯一修订：P5 阈值 60%→50%（见其 §12.5）。

- **Tier 2 · 完整 DAB（数 GPU-day，高实现风险）。** 仅当审稿要求。风险全在无代码细节
  （β schedule→[0.2,0.6]、N=1000 截断 T'=32、zero-init 输出头），且推理是
  per-pair×32 步的 cross-encoder 成本。

**结案判据。**
- **支持 thesis：** Tier 0 中 hubness skewness 随带来增益的操作显著下降且与增益单调相关；
  Tier 1 中签名在臂 2 消失、corr(b_j, N_k) 显著为负、增益分解落在 b 臂、臂 5/6 无签名。
- **证伪 thesis（F-A/F-B）：** 臂 2/3 仍保留增益签名；或臂 1 之上叠 QB-Norm 后仍有
  显著增量增益；或 Tier 0 的仿射 oracle 包络造不出 DAB 形状的增益。

**与 OQ-2 的联动。** Tier 1 的复现头正是 OQ-2 缺的那个"外部方法的方差头"——
训完后其 log σ² 直接进 OQ-2 的增量 AUROC 协议，一次训练服务两个问题。

**状态。**
- **Tier −1 ✅ 完成（2026-08-28）**，结果见 `research/oq1/tier-minus1-gain-shapes.md`
  与 ledger E-76…E-83。**结论：hubness 签名为 DAB 独有**（A ∈ [+2.1,+7.65]），
  UATVR/T-MASS/ProTA/DiffusionRet/PAU/DUQ 全部 R@1-forward（A ∈ [−2.5,−0.25]）。
  **Thesis 家族版收窄为"KL-cost 类"**；意外收获：DAB 表 5b（KL→W₂，R@1 增益归零、
  重排效应保留，E-82）就是论文自带的 F-A 近似实验，Tier 1 臂 5/臂 6 的先验大幅增强。
- **Tier 0 ✅ 完成（2026-08-28）**，结果见 `research/oq1/tier0-correction-envelope.md`
  与 ledger E-84…E-88。对齐规则（caption i → video ⌊i/20⌋）与 MnR 口径经数值复现验证。
  **结论：val 矩阵是低 hubness、近饱和 regime（raw 76.50/95.50/97.89，MnR 2.11，
  S(N10)=0.53），修正族的一切产出均为 R@1-forward**——DSL T=2 在 val 上 **−0.65**
  （与 test 上 +2.6 的 E-08 对照，增益 regime 依赖）；Sinkhorn 唯一正增益 label-free 臂
  （+1.72 @T=1，A=−1.55）；oracle split-half 可迁移增益 ≤+0.5、in-sample 加性 +2.30/
  仿射 +3.54 全为 A<0。**该结果既不证伪也不支持 thesis**（DAB 形状在此 regime 物理上
  不可表达），产出是 thesis 精化："hubness 签名 = candidate-only 修正 × 高 hubness
  分数几何，缺一不可"。⚠️ 结案判据修订：上方"支持 thesis"第一句（skewness 随增益
  下降且单调相关）在低 hubness regime 不适用（Sinkhorn 反例），移交 Tier 1 验证；
  "仿射 oracle 造不出 DAB 形状 → 证伪"一句以矩阵处于高 hubness regime 为前提，
  在本矩阵上不开火。
- **Tier 1 预注册已签署（2026-08-28，status: registered，commit `20a4a1d`），执行未开始。**
  `experiments/oq1-kl-variance-hubness-v1/{PREREGISTRATION.md,study.yaml}`，study.yaml 已冻结。
  八阶段（stage0 资格 → stage1 **regime 门 P0** → stage2 包络 → stage3 六臂 →
  stage4 多 seed **P1/P2/P3** → stage5 机制 **P4/P5/P6** → stage6 test 一次性确认 →
  stage7 可选 X-Pool）。Tier 0 的教训被制度化为：**第一道门是 regime 门不是效应门**——
  P0 不过则判 inconclusive、**不开火 F-A**、不解封 test（否则低 hubness 下"消掉 (A) 项
  也没变化"会让 F-A 假阳性触发）。证伪映射与 negative-result 强制归档条款见预注册 §6。
  阻塞项：PC-9 四卡资格测试（起草当日实测 GPU 0/1/4 各被外部进程占
  ~74.5 GiB，**当下起不了 4 卡**，故 study.yaml 把 PC-10 改为 reserve 期探测 + 写入
  manifest 的 fail-closed 策略，不再硬编码卡号）。Tier 2 搁置。
- **Stage 0 四门全部关闭（2026-08-29/30），Stage 1 解锁。** val1k 分区（b3d9f2f）、
  四卡资格（PC-9，摘要 `summaries/archive-qualification.md`）、source-tree sha 绑定、
  确定性契约（arm 训练栈 commit `1c0cc4e`；同 seed 两次 200 步 a/b 探针**逐位一致**，
  losses npy sha `7039f69f…`、params sha `6c1f9b10…`，摘要 `summaries/determinism-probe.md`）。
  arm 实现自由度的定死清单见控制面 README（clamp、维度均值、双向 CrossEn、
  precision 解释）。下一步 = Stage 1：arm0 × 3 seed 全程训练 + val1k 上测 P0
  regime 门（S(N₁₀)≥1.5 ∧ MnR≥8.0 ∧ R@1≤60，split=val1k, DSL off, t2v）。

---

## OQ-2 · variance vs geometry：跨方法的 failure-AUROC 增量比较

**问题。** 在**控制住 score geometry 之后**，是否还存在任何 encoder-internal variance 量能提升 failure AUROC？

**已有的部分答案（但不够）。**
- SAP 的 5 个描述子 direct AUROC 0.4561–0.5441（随机）；接进 sidecar 增量 0.0004（E-42/E-43）。
- A2-CG 的 512-D Gaussian：AUROC 0.4064，低于随机（E-20）。
- Stage 1 的 `score_variance` scorer：mean AUROC 0.4411（E-32）。
- **不够的原因：** 这三者都是**我们自己的**方差头。thesis 声称的是一个更普适的命题，
  需要至少一个**外部方法**的方差头（如 DAB 的 `log σ²`，或 T-MASS / ProTA）作为对照。

**怎么做。** 固定 A1 的 8 维 score 特征作为 baseline 预测器，逐一加入候选 variance 描述子，
测增量 failure AUROC（同一 split、同一 bootstrap 协议）。至少覆盖：
我们的 CSD logvar、我们的 vMF κ（若实现 I1）、一个复现的外部方差头。

**结案判据。**
- **支持：** 所有 variance 描述子的增量 AUROC ≤ +0.01（在 CI 内不显著）。
- **证伪（F-C）：** 任一 variance 描述子给出 ≥ +0.03 的增量且 CI 下界 > 0。

**⚠️ 功效约束。** trusted-v1 val 的 V2T 只有 35 failure / 500 query，
CI 半宽 ≈ ±0.09，**加 seed 不缩小**（E-57）。判 ±0.03 的效应必须先解决样本量，
否则只能在 T2V 方向（2348/10000，CI ±0.007）上判定。

**状态。** 未开始。依赖 OQ-5 的样本量方案。

---

## OQ-3 · Conformal coverage under CRF shift

**问题。** 用 A1 的 `failure_risk` 作为 nonconformity score 做 split-conformal
selective retrieval，能否得到 `P(top-1 correct | 不弃答) ≥ 1−α` 的无分布保证？
更关键的：在 **H.264 降质造成的分布偏移**下，marginal coverage 会不会崩？
用 CRF probe 的输出作为分层变量做 Mondrian/分层 conformal，能否恢复 conditional coverage？

**为什么可行。** 两个零件都已就绪且已验证：
- A1 `failure_risk` 已有 test 上的 risk-coverage 曲线（E-10a/E-10c）。
- CRF probe 在 6 级 ladder 上 15/15 pairwise、mean `u_video` 单调、
  per-video Δu 的 5th percentile > 0（E-60/E-61/E-62），是一个**可信的分层变量**。

**怎么做。**
1. 在 trusted-v1 val 上做 calibration，给定 α ∈ {0.1, 0.2}，求 `failure_risk` 阈值。
2. 在 clean JSFUSION test 上验 marginal coverage。
3. 构造 CRF-degraded 的 test 变体（用已有的 `scripts/build_msrvtt_crf_ladder.py`），
   验 (a) 不分层的 marginal coverage 是否失效、(b) 按 CRF probe 输出分层后的 conditional coverage。

**结案判据。**
- **成功：** clean 上 marginal coverage 落在 [1−α−0.02, 1] 内；shift 下不分层出现明显 undercoverage
  （证明 shift 是真的），而按 CRF probe 分层后每层 coverage 都恢复到 ≥ 1−α−0.02。
- **失败（F-D）：** 分层后仍有层系统性 undercoverage → 说明 CRF probe 不是有效的分层变量，
  需换 shift 轴。

**⚠️ 协议阻塞。** 这需要动 test 集。见 `07-protocol-constraints.md` PC-4：
**在新 study 的 Stages 4/5 通过之前，不授权任何 test 集评估。**
所以 OQ-3 必须先写成 pre-registered study。

**状态。** 未开始，被协议阻塞。这是 north-star 的 constructive half，**是论文的主贡献候选**。

---

## OQ-4 · MSVD / DiDeMo transfer

**问题。** A1 的 8 维 score-geometry 特征是否跨数据集泛化？CRF probe 是否跨数据集泛化？

**已有的部分答案。**
- CRF probe：donor ckpt 在 MSVD 上已验 `three_pair = 1.0`、`cross_domain_pass = true`、
  `spearman(u, −bpp) = −0.34`（E-65）。但 hybrid ckpt 的 MSVD 结论是 **by transitivity**，
  未直接重跑（rawvideo `.avi` decoder 有 `fps=0` bug）。
- A1：**完全未测**。`PAPER.md:174` 把它列为 Limitation。

**怎么做。** 需要先有 MSVD 的 June-arch base retriever。资产已存在：
`ckpts/ckpt_msvd_20260813_034603_msvd_june_baseline/`（bin.0–bin.4），
launcher `run_train_msvd_june_bg.sh` 与 `run_train_msvd_risk_sidecar_bg.sh`（两者均 untracked）。

**结案判据。**
- **强泛化：** A1 在 MSVD 上 failure AUROC ≥ 0.70 且 ρ ≥ 0.20（双向），**无需重调特征**。
- **弱泛化：** 需要重新拟合 head 但特征集不变即可达标 → 仍是可发表的结论。
- **不泛化：** 特征集本身必须改 → thesis 的 "score geometry 是通用信号" 部分被削弱。

**⚠️ 协议阻塞。** PC-3：MSVD launch 保持 **fail-closed**，直到 launcher 被 qualify（`todo.md:39` 的 Task 8）。
另需修 `.avi` decoder 的 `fps=0` bug。

**状态。** 未开始，被协议阻塞 + 有已知 decoder bug。

---

## OQ-5 · 1000 vs 1012 分母对账（以及 V2T 样本量）

**问题。** V2T 方向的 query 集在 risk 模块（1000）与检索指标（1012）之间不一致。
哪一个是正确的？两个 query 集的差集是什么？

**硬证据（本 session 直接从 artifact 核对）。**
`artifacts/contextual-risk-v2/msrvtt/feature_ablation/msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/checkpoints/final_test.json`：
- `selections/v2t/test_metrics/retrieval_risk/v2t/num_queries` = **1000**，`accuracy_at_1.00` = **0.526**
- `selections/v2t/test_metrics/v2t/cols` 长度 = **1012**，`R1` = **51.976...**（= 526/1012）
- 同分子 526，不同分母。T2V 侧两边都是 1000（`num_failures` 479 / `num_queries` 1000，`R1` = 52.1），**无此问题**。

**怎么做。** 定位 JSFUSION test 的 V2T query 构造代码，输出两个 query 集的 id 列表并求差集（12 条）。
判断是 caption 去重、还是多 caption 视频的展开方式不同。

**结案判据。** 能给出那 12 条 query 的具体来源，并给出一个**唯一的**、两处共用的口径；
重跑后 `accuracy_at_1.00` 与 `R@1` 数值一致。

**为什么必须做。** `PAPER.md:141` 自己标了 "Resolve before submission"。
在此之前，`PAPER.md:139` 的 "+35.0 pp"（V2T）**不能与 Table 1 的 V2T R@1 = 52.0 并列**。

**顺带要修的另外两处（见 ledger D-02 / D-03）：**
- `PAPER.md:95` 的 base ckpt SHA-256 写成了 A1 sidecar 的 SHA。正确值 `ee79f8a2…d030964`。
- `PAPER.md:51` 写 CLIP-ViT-B/32，实际两份 `training_args.json` 都是 **ViT-B/16**。

**状态。** 未开始。**成本最低、收益确定，建议与 OQ-1 并行做掉。**

---

## OQ-6（次要）· A1 的 vacuity 作为 open-set rejection 分数

**问题。** A1 的 `vacuity = 2/(α_ok+α_fail)` 是否能作为 out-of-domain query 的拒识分数？

**现状。** `PAPER.md:189` 把它列为 future work，无需重训。
但 val 上 vacuity 的分布很窄：T2V mean 0.2563 / std 0.0284，V2T mean 0.2572 / std 0.0206
（本 session 从 `final_test.json` 读取）。**std 只有 0.02–0.03，接近 acceptance gate 的 0.02 下限**，
说明它作为 rejection 分数的分辨率可能不足。

**结案判据。** 构造 negative query（Moment-of-Untruth 风格）后，
vacuity 的 ID/OOD AUROC ≥ 0.70 → 值得写；< 0.60 → 放弃这条 future work。

**状态。** 未开始，低优先级。先做 OQ-1/OQ-5。
