# Stage 1 · arm0 基线与 P0 regime 门 —— **不通过,study 按注册协议 HALT (inconclusive)**（2026-08-30）

## 判定

**P0 regime 门(study.yaml `stop_gates.stage1_baseline_regime`,冻结 20a4a1d)不通过:**

| 条件 | 阈值 | seed000 | seed001 | seed002 | 均值 | 判 |
|---|---|---|---|---|---|---|
| S(N₁₀) | ≥ 1.5 | 1.144 | 1.011 | 0.867 | 1.007 | **✗** |
| MnR | ≥ 8.0 | 21.50 | 23.70 | 23.21 | 22.80 | ✓ |
| R@1 | ≤ 60.0 | 45.20 | 45.30 | 45.20 | 45.23 | ✓ |

(全部数字口径:split=msrvtt_trusted_v1_val1k,DSL off,t2v,exact-fp32 评测,
最优 epoch checkpoint,冻结实现 `scripts/oq1_stage1_p0_metrics.py`,PC-12。)

三个 seed **逐一失败且方向一致**,故不存在聚合口径歧义(逐 seed / 均值 / 最优
任一约定下结论相同)。按冻结协议 `on_failure: halt_study_as_inconclusive`:

- **不开火 F-A**(这正是 P0 作为 regime 门的设计目的:低偏度 regime 下
  "消掉 (A) 项无变化"会是假阳性);
- **test 保持 sealed**(PC-4 的解封条件 stage4+stage5 永不满足);
- Stage 2–6 在本 study 下不再执行;
- 任何后续尝试 = 新 study 目录 + 新 protocol_version(PC-2/deviation_policy)。

## 完整记分板

| run | best epoch | t2v R@1/R@5/R@10 | t2v MdR/MnR | t2v S(N₅)/S(N₁₀) | v2t R@1 / MnR / S(N₁₀) |
|---|---|---|---|---|---|
| `…arm0-seed000-20260829T214100Z-4232ac5a7270` | 2 | 45.20/71.30/80.70 | 2/21.50 | 1.204/1.144 | 45.10/15.07/2.289 |
| `…arm0-seed001-20260829T214101Z-4232ac5a7270` | 1 | 45.30/70.90/79.70 | 2/23.70 | 1.141/1.011 | 45.50/16.30/1.863 |
| `…arm0-seed002-20260829T214102Z-4232ac5a7270` | 1 | 45.20/70.80/80.40 | 2/23.21 | 1.053/0.867 | 45.50/16.01/2.040 |

逐 seed 门判定证据:`p0-arm0-seed{000,001,002}-r2.json`(从归档矩阵摘要校验载入)。
registry:r2 phase terminal_counts {completed: 3}(`stage1-arm0-baseline-r2.json`)。
attempt-1(bf16 评测,已作废,见 README #9):{completed: 1, cancelled: 2}。

## 学到了什么(写给下一个 study 的设计者)

1. **预注册 §3.2 的 regime 假设只对了一半。** "hubness 差异主要来自评测几何"
   在 MnR 上成立得漂亮——候选池 500→1000、caption 20→1 使 MnR 2.11→22.8
   (×10.8);但 **t2v 偏度只从 0.53 涨到 ~1.0(×1.9),距 1K-A 推定的 ≥1.5
   有实距离**。偏度不只由几何决定,还依赖 backbone/训练分布/caption 选取。
2. **方向不对称是真实信号:v2t 的 S(N₁₀) = 1.86–2.29,全部过 1.5。** 文本侧
   hub(一段 caption 吸引多个视频 query)在本 backbone 上比视频侧 hub 强得多。
   注册协议把主方向锁在 t2v(PC-12,随 Tier −1 全表),故不得引 v2t 过门;
   但"KL cost 的 (B) 项 = v2t 镜像修正"这条线索对 v2t-regime 的后续设计有价值。
3. **backbone 极稳定:三 seed R@1 45.20/45.30/45.20**(极差 0.1),严格确定性
   模式下 seed 间方差几乎只剩数据顺序效应——stage4 想要的小差值检验功效在
   这套栈上是真实可得的(可惜本 study 到不了 stage4)。
4. P0 的 S(N₁₀)≥1.5 阈值当初就是七个阈值里证据第二薄弱的(两个实测 regime 的
   插值,见 PREREGISTRATION §5 起草记录);实测落在 1.0 说明插值高估。若开
   v2 study,阈值必须从**实测的 val1k 偏度分布**重新推导,或改造 regime
   (更大候选池 / v2t 主方向 / 换 caption 选取),而非降阈值迁就。

**结论:oq1-kl-variance-hubness-v1 于 Stage 1 终止,结果 = inconclusive
(negative result 已按 deviation_policy 归档,不可撤销注册)。PC-4 维持:
MSRVTT/MSVD test 仍 sealed。**
