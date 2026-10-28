# 00 · North Star —— 当前研究命题与证伪判据

> 状态：**PROPOSED（提出于 2026-08-22 brainstorm），尚未验证。**
> **2026-08-28 修正（OQ-1 Tier −1，ledger E-76…E-83）：** thesis 中"概率化 embedding
> （UATVR / T-MASS / ProTA / DUQ / DAB 一系）的增益 = 隐式 hubness 校正"的**家族普适
> 表述已被证伪**——七方法普查显示 hubness 签名为 DAB（非对称 KL cost）独有，其余六个
> 方法的概率组件均为 R@1-forward。thesis 的诊断半部分**收窄为**："非对称 KL 打分中
> candidate-only 方差项是学习出来的 hubness 校正；该项的有无解释了 DAB 与其余概率方法
> 增益形状的分野。"（该分野本身成为新证据：DAB 表 5b KL→W₂ 后 R@1 增益归零，E-82。）
> uncertainty 半部分（usable signal 完全来自 score geometry）与 constructive half 不受影响。
> 来源：2026-08-22 与研究者的 brainstorm（跨 session 记忆，仓库内无对应文档，`[未溯源]`）；
> 其证据基础来自 `research/01-evidence-ledger.md` 中已标 SOLID 的条目
> 与 `related_work/dab_eccv2026/NOTES.md:87-97, 107-122`。

---

## Thesis（一段话）

在 text-video retrieval 中，**representation variance 与 predictive uncertainty 是两套
彼此独立的机制**：概率化 embedding（UATVR / T-MASS / ProTA / DUQ / DAB 一系）所带来的
accuracy 增益，很大程度上是一种**隐式的 hubness 校正**——即一个与 query 无关、只依赖
candidate 的 per-candidate 加性偏置（数学上与 inverted softmax / QB-Norm / Sinkhorn
同族）；而**真正可用的 uncertainty 信号完全来自 candidate-set 的 score geometry**
（top-1 margin、entropy、top-K 分布形状），与 encoder 内部学到的 variance 无关。

**建设性的另一半（constructive half）：** 在此认知上做出**视频检索领域第一个
distribution-free、risk-controlled 的 selective retrieval**——用 conformal risk control
给出 `P(top-1 correct | 不弃答) ≥ 1−α` 的无分布保证，并进一步给出**在 H.264 降质分布
偏移下的 conditional coverage**，以已验证的 **CRF probe 输出作为分层变量（stratifying
variable）**。压缩鲁棒探针在这里不再是"第二个卖点"，而是提供 shift 轴的工具。

## 为什么这个 thesis 现在可信度较高（但仍未验证）

- 我们从"内部方差"一侧证伪：encoder-internal evidential uncertainty（SAP 的 `u_mode` /
  Dirichlet evidence / mixture logsigma）对检索失败的预测力**等同随机**（AUROC 0.4561–0.5441），
  而同一模型上 score geometry 给到 0.72–0.88。
  见 `01-evidence-ledger.md` 行 E-11 / E-12。
- DAB（ECCV 2026, arXiv 2607.20984）从**相反立场**独立佐证同一件事：它自己承认
  *"DAB's uncertainty signal emerges from the bridge-induced distributional gap rather
  than from text variance alone."*（`related_work/dab_eccv2026/NOTES.md:115-117`）
- Hubness 假设的直接动机：DAB 的增益形状是 R@1 +4.3 / R@5 +9.7 / R@10 +7.8 / MnR −64%
  （`related_work/dab_eccv2026/NOTES.md:65-69`）——这是**全局 score 分布被改动**的签名，
  不是 per-pair matching 变好的签名。其 Gaussian KL cost 中的方差项
  (`σ_P²/σ_Q²`, `log σ_Q²/σ_P²`) 只依赖 candidate，因此是 query-independent 的
  per-candidate 偏置（`related_work/dab_eccv2026/NOTES.md:87-97`）。
- 我们自己也有 zero-parameter 对照：trusted-v1 val 上 **entropy scorer（0 参数）
  mean AUROC 0.8300 击败 128-dim A3 MLP 的 0.8013**
  （`.worktrees/reliable-retrieval-experiment-campaign/experiments/contextual-risk-v2/summaries/stage1-baselines.md:26-34`）。

## 证伪判据（Falsification criterion）

本 thesis 在**任一**下述结果出现时被证伪或需要重大修正：

| # | 若观察到 | 则 |
|---|---|---|
| F-A | 把概率方法（DAB 类 KL cost，或我们自己的 CSD/vMF 头）中**只依赖 candidate 的方差项消掉**后，R@1/R@5/MnR 的增益**基本不变** | "增益 = 隐式 hubness 校正" 被证伪 |
| F-B | 施加一个显式 hubness 校正（inverted softmax / QB-Norm / Sinkhorn）后，概率方法的**增量增益仍然显著存在** | 同上，被证伪 |
| F-C | 存在某个 encoder-internal variance 量，在**控制住 score geometry 之后**（例如以 8 维 score 特征为协变量做 partial AUROC / 增量 AUROC），仍能显著提升 failure AUROC | "usable uncertainty 完全来自 score geometry" 被证伪 |
| F-D | conformal selective retrieval 在 CRF 分层下**无法**达到 conditional coverage（即分层覆盖率系统性偏离目标 1−α 且不可通过 Mondrian/分层校准修复） | constructive half 不成立，需换 shift 轴或换方法 |

## 会验证/证伪它的 pilot（下一步该做的那一个实验）

**Pilot P0 — "hubness vs variance" 双臂测量（先做，成本最低，无需训练）：**

1. 在 frozen June/A1 retriever 的 MSRVTT val + JSFUSION test 的 WTI 相似度矩阵上，
   计算 **k-occurrence 分布 N_k 与 hubness skewness S(N_k)**（k=5,10），
   分别在 (a) 原始分数、(b) DSL T=2.0 重排后、(c) 一个人工加的 per-candidate 加性偏置后。
   → 量化"DSL/hubness 校正"在我们自己的 backbone 上贡献了多少 R@1（已知 DSL 贡献 +2.6 T2V，
   见 `01-evidence-ledger.md` E-08）。
2. **增量 AUROC 测试（对应 F-C）**：以 A1 的 8 维 score 特征为 baseline，逐一加入
   任一 encoder-internal variance 描述子，测 failure AUROC 的增量。已有的一次性证据
   （`sap_uncertainty` block 加入后 AUROC 变化 < 控制臂 seed 间散布，
   `01-evidence-ledger.md` E-12）需要在**非 SAP 的方差头**上复现才算通过。

判据：若 (1) 显示 hubness skewness 在带来 R@1 增益的操作前后显著下降，且 (2) 的增量
AUROC 在所有 variance 描述子上 ≤ +0.01，则 thesis 的诊断半部分成立，可以启动
constructive half（conformal + CRF 分层）。

**P0 第 1 项的 val 半（2026-08-28,OQ-1 Tier 0,E-84…E-88）已完成**：val 矩阵为低
hubness、近饱和 regime（S(N10)=0.53,MnR 2.11），DSL T=2 在 val 上 **−0.65** R@1
（test 上 +2.6,E-08）——增益 regime 依赖，佐证 hubness-校正解读；加性/仿射 oracle
split-half 可迁移增益 ≤+0.5，全部 R@1-forward。**判据修正**："偏度随增益操作下降且
单调相关"在低 hubness regime 不适用（Sinkhorn 反例,E-87），thesis 精化为
**"hubness 签名 = candidate-only 修正 × 高 hubness 分数几何，缺一不可"**。
test 半与 P0 第 2 项（增量 AUROC,对应 F-C/OQ-2）仍未开始；前者已归并入 OQ-1 Tier 1。

**F-A / F-B 的正式检验已预注册并签署（2026-08-28,status: registered,commit `20a4a1d`,已冻结）**：
`experiments/oq1-kl-variance-hubness-v1/{PREREGISTRATION.md,study.yaml}`。
其中 F-A = 预测 P2（arm2 即 λ_A=0 臂），F-B = 预测 P6（相对显式修正的增量形状）；
两者的开火条件均**以 P0 regime 门通过为前提**——低 hubness regime 下 F-A 会假阳性触发
（Tier 0 教训）。negative result 强制归档，不得撤销注册换配置重跑。
该 study 的 stage4+stage5 双门通过，也是 PC-4 下 MSRVTT test 解封的唯一授权路径。
