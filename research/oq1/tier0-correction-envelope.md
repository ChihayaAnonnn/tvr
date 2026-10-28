# OQ-1 Tier 0 · 加性/仿射修正族在冻结 val 矩阵上的能力包络

> 日期：2026-08-28。执行人：Claude（研究者授权"请你开始 Tier 0"）。
> 数据：`artifacts/rescued_from_tmp/frozen_equiv/sim_june_fp32.npy`
> （sha256 前缀 8ebf27ae4446，10000×500 float64，trusted-v1 val，分数范围 [15.8, 60.1]）。
> 脚本：`scripts/oq1_tier0_correction_envelope.py`；数值输出
> `artifacts/oq1_tier0/tier0_results.json`。无 GPU，val-only，PC-4 不涉及。

---

## 结论（一句话）

**在我们自己的 val 矩阵上，加性/仿射 per-candidate 修正族的全部产出都是
R@1-forward 形状（A<0），连 in-sample 过拟合的仿射 oracle 也造不出 DAB 签名
——但这不能当 F-A 证伪判据开火，因为该矩阵处于"低 hubness、近饱和"regime
（MnR=2.1，无处可崩），DAB 形状在此物理上不可表达。** Tier 0 的真正产出是把
thesis 精化为：**hubness 签名 = candidate-only 修正 × 高 hubness 分数几何，
两因子缺一不可**；以及一个新的内部证据：同为修正族的 DSL 在 test 上 +2.6
（E-08）而在 val 上 **−0.65**——增益 regime 依赖，正是 hubness-校正解读的预测。

---

## 0 · 预检：对齐规则与口径（全部通过）

- **GT 规则 caption i → video ⌊i/20⌋**：代码层面 val dataset 强制每视频恰 20 条
  caption（`dataloader_msrvtt_retrieval.py:276-283` 否则 raise），metric 走分组对角线；
  数值层面用仓库自己的 `_apply_dual_softmax` + `tensor_text_to_video_metrics` 从
  raw 矩阵复现了冻结契约 JSON 的全部 7 项 post-DSL 指标
  （T2V 75.8499984741/94.78/97.26/MdR 1；V2T 93.0/99.8/100.0），逐位一致。
- 细节：multi-sentence eval 的 V2T 指标走 `independent_directions=False` 分支，
  即从 **T2V 的 DSL 矩阵**按组取 max（`main_task_retrieval.py:4378-4384`），
  不是 v2t 方向的 DSL 矩阵——复现时踩过一次。
- **MnR 口径**：0-based rank 均值 +1，与 `compute_metrics`/ledger 一致。
- Flat 计算（gt=i//20 的逐行 rank）与仓库分组路径逐位等价 ✓。
- **训练选点不受本文 DSL 结论影响**：June `training_args.json` 无 DSL 键
  （默认 False），历史 val 选点用的是 raw R@1。

## 1 · Baseline 几何：低 hubness、近饱和

raw：**R@1 76.50 / R@5 95.50 / R@10 97.89 / MdR 1 / MnR 2.109**；
S(N5)=0.355，S(N10)=0.528，max N10=472（均匀期望 200）。
对照 DAB 的工作 regime（1K-A test，baseline R@1 46.9 / MnR 14.3）：
本矩阵 MnR 已贴地板，"MnR 腰斩"无从发生。这是解读下面一切结果的前提。

## 2 · Label-free 臂（全 10000 query，Δ 相对 raw）

| 臂 | ΔR@1 | ΔR@5 | ΔR@10 | ΔMnR% | **A** | S(N10) raw→后 |
|---|---|---|---|---|---|---|
| DSL T=2.0（生产设置） | **−0.65** | −0.72 | −0.63 | **+16.8%** | −0.02 | 0.53→0.70 |
| 逆 softmax T=2.0 | −0.98 | −0.80 | −0.71 | +19.1% | +0.23 | 0.53→0.73 |
| QB-Norm（self-bank） | ≡ 逆 softmax（见 §5 退化说明） | | | | | |
| Sinkhorn T=1.0 | **+1.72** | +0.25 | +0.09 | −1.4% | **−1.55** | 0.53→1.29 |
| Sinkhorn T=2.0 | +1.13 | +0.27 | −0.12 | 0.0% | −1.05 | 0.53→0.38 |

温度网格（0.5–20，val 上调参，仅作 supplementary）：DSL/IS 在**所有** T 下都
≤ raw；Sinkhorn 最优 T=1.0（78.22）。

要点：
- **DSL 在 val 上是负增益**，且把 hubness 偏度推高而非压低。与 E-08
  （JSFUSION test 上 +2.6，50.0→52.6）对照：同一修正、同一模型家族，
  增益完全由矩阵的 hubness regime 决定。
- **Sinkhorn 是唯一有效的 label-free 臂**，其对 T2V 排序的作用恰好就是
  加性 b_j = T·log v_j（行缩放不改行内排序）——族内成员，形状 R@1-forward。
- 偏度与增益**不单调相关**：Sinkhorn T=1 增益最大却把 S(N10) 推到 1.29。
  OQ-1"支持 thesis"判据的第一句在低 hubness regime 不适用（见 §6）。

## 3 · Oracle 包络（split-half：偶行拟合，奇行评测；raw@eval=76.22）

条件 logit MLE（对 (a_j,b_j) 凸），τ ∈ {0.5,1,2,5} 按拟合半 R@1 选择：

| 臂 | eval 半 R@1 | Δ | 备注 |
|---|---|---|---|
| oracle 加性（τ=0.5） | 76.14 | **−0.08** | 拟合半 in-sample 79.08（+2.30） |
| 　γ-收缩扫描最优（γ=0.3，直接看 eval 半，1 dof 乐观） | 76.76 | +0.54 | 收缩后可迁移上界 |
| oracle 仿射（τ=5.0） | 76.18 | −0.04 | 拟合半 in-sample 80.32（+3.54） |
| （对照）Sinkhorn T=2 同 eval 半 | 77.20 | **+0.98** | label-free 反而更好 |

In-sample 形状（拟合半上评拟合半，过拟合上界；raw@fit=76.78）：

| | ΔR@1 | ΔR@5 | ΔR@10 | ΔMnR | **A** | S(N10) |
|---|---|---|---|---|---|---|
| oracle 加性 | +2.30 | +0.54 | +0.12 | +0.03 | **−1.97** | 2.43 |
| oracle 仿射 | +3.54 | +1.00 | +0.58 | −0.31 | **−2.75** | 5.10 |

要点：
- **族的诚实可迁移增益 ≤ +0.5**，打不过 label-free 的 Sinkhorn（+0.98）。
  500 个自由度在 5000 样本上 MLE，in-sample 的 +2.3/+3.5 几乎全是过拟合
  ——这本身是一个可复用的方法论警示：**任何论文里"per-candidate 校正在
  小验证集上调出的增益"都应按此打折**。
- **即便 in-sample 过拟合 oracle，形状也全是 R@1-forward（A<0），MnR 几乎不动，
  偏度不降反升**（oracle 学到的是"标签先验"——押注常被命中的视频列，
  与反 hub 修正相反）。仿射并不比加性更接近 DAB 形状。

## 4 · 机制指纹

corr(b_j, N_k^baseline)：全部 |r| ≤ 0.12 且不显著
（oracle 加性 vs N5：Spearman −0.119, p>0.1；IS 的 b 甚至弱正相关 +0.08~+0.11）。
在低 hubness 矩阵上，无论 label-free 还是 oracle，学出的修正都**不是**反 hub 的
——没有 hub 可反。

## 5 · 限制与诚实性备注

- **regime 错配是根本限制**：Tier 0 原设想"oracle 造不出 DAB 形状 → F-A 证伪"，
  隐含前提是矩阵与 DAB 处于可比 regime。本矩阵近饱和（R@1 76.5、MnR 2.1），
  该前提不满足，故本结果**不构成对 thesis 的证伪，也不构成支持**——它把
  "高 hubness 几何"识别为签名的必要条件。高 hubness regime 的包络测量
  自然归并进 Tier 1 的 X-Pool/1K-A 设置（baseline MnR 14.3）。
- **QB-Norm 退化**：以 val query 自身为 bank 时，每个 query 的 top-1 必然在
  activation set 里（自己就是 bank 成员），QB-Norm 严格退化为逆 softmax。
  真 QB-Norm 需外部 querybank（train captions），本轮未做。
- oracle 的 τ 在拟合半上选；γ-收缩扫描直接读了 eval 半（1 dof 乐观），已标注。
- 单一 backbone（June/WTI）、单一 split（trusted-v1 val）；test 矩阵未保存
  且 PC-4 禁新增 test 评测，故 "test 矩阵 hubness 远高于 val" 目前是从
  E-08/E-86 增益差反推的推断，未直接测量。

## 6 · 对 OQ-1 的影响

1. **Thesis 精化（非收窄非证伪）**：hubness 签名 = candidate-only 修正 ×
   高 hubness 分数几何。Tier −1 的分野解释不受影响（七方法都在 1K-A test
   高 hubness regime 上比较，DAB 在同一 benchmark 造出签名而其他方法没有
   ——缺的是"修正"因子，不是"几何"因子）。
2. **结案判据需修订**："Tier 0 中 hubness skewness 随增益操作显著下降且与增益
   单调相关"这条支持判据在低 hubness regime 不适用（Sinkhorn 反例：增益最大、
   偏度上升）；应改为 regime-conditional，落到 Tier 1 上验证。
3. **新证据**：DSL 增益的 regime 依赖性（test +2.6 vs val −0.65）本身是
   "DSL=hubness 校正"解读的正面佐证——低 hubness 矩阵上无可校正，只剩扰动。
4. Tier 1 动机增强：真正的包络问题（高 hubness regime 下修正族能吃掉 DAB
   多少增益）只能在 X-Pool/1K-A 上回答，与 DAB 复现共用同一套设置。
