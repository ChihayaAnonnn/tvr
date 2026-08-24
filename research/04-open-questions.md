# 04 · Open Questions —— 每条都带"什么结果能结案"

> 新实验从这里挑，不要从 `PAPER.md §7` 的 Future Work 挑（那份列表写于 2026-08-13，
> 其中 A3 与 I1 已经分别被 `02-dead-ends.md` DE-2 / DE-1 削弱或证伪）。
> 优先级按当前 north-star（`00-north-star.md`）排序。

---

## OQ-1 · Hubness 测量：概率方法的 R@1 增益有多少是隐式 hubness 校正？

**问题。** DAB 类方法（以及我们自己的 DSL rerank）带来的检索增益，是否可以完全由
"query-independent 的 per-candidate 加性偏置"解释？

**为什么现在问。** 这是 north-star thesis 的诊断半部分，且**不需要训练**，成本最低。
已有的旁证：DAB 的增益形状是 R@1 +4.3 / R@5 +9.7 / R@10 +7.8 / MnR −64%
（ledger E-71），以及它的两个组件单独都会**降低** R@1（E-72）。
我们自己的 DSL rerank 贡献 +2.6 T2V R@1（E-08），DSL 本身就是一种 hubness 校正。

**怎么做。** 在 frozen June/A1 的 WTI 相似度矩阵上（`/tmp/frozen_equiv/sim_*.npy` 已有现成矩阵，
但**在 /tmp，先转存**），计算 k-occurrence 分布 `N_k` 与 hubness skewness `S(N_k)`（k=5,10），
分别在 (a) raw score、(b) DSL T=2.0 之后、(c) 人工 per-candidate 加性偏置之后。

**结案判据。**
- **支持 thesis：** hubness skewness 在带来 R@1 增益的操作前后**显著下降**，
  且下降幅度与 R@1/R@5/MnR 增益单调相关。
- **证伪 thesis（F-A/F-B）：** 消掉方差项后增益基本不变，或显式 hubness 校正之后
  概率方法仍有显著增量增益。

**状态。** 未开始。**这是推荐的下一个动作。**

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
