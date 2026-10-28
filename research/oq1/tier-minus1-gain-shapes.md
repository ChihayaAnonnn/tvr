# OQ-1 Tier −1 · 概率化 TVR 方法的增益形状普查

> 日期：2026-08-28。执行人：Claude（研究者授权"完成Tier -1"）。
> 问题：**"R@1 小涨 + R@5/R@10 大涨 + MnR 崩塌"这个 hubness-校正签名，
> 是概率化 embedding 家族的普遍现象，还是 DAB 一家的现象？**
> 规则（见 `../04-open-questions.md` OQ-1）：每个方法只与**它自己的确定性 baseline** 比，
> 优先取论文自己消融表的同 backbone / 同训练预算行。

---

## 结论（一句话）

**签名是 DAB 独有的。** 七个方法中，只有 DAB（非对称 KL cost）的不对称指数 A > 0
（+3.55 ~ +7.65）；UATVR / T-MASS / ProTA / DiffusionRet / PAU / DUQ 全部 A < 0
（−0.25 ~ −2.5），即它们的概率组件是 **R@1-forward** 的。
North-star thesis 的"家族普适"版本**不成立**，必须收窄为：
**"隐式 hubness 校正"假设只适用于 KL-cost 类打分（DAB），不适用于概率化 embedding 全家族。**

额外的意外证据：DAB 自己的 cost 消融（表 5b）显示，把 directional KL 换成 W₂
（加性 log σ² 项 → (σ_t−σ_v)² 混合项）后 **R@1 完全回落到 baseline**（52.6 → 46.8，
baseline 46.9），而 R@5/R@10/MnR 的全局重排效应保留（A = +6.65）。
这是论文自带的、最接近我们 F-A 设想的实验。

---

## 汇总表（全部 MSRVTT-1K-A，T2V；Δ 相对各自 baseline）

`A = (ΔR@5 + ΔR@10)/2 − ΔR@1`（A>0 + MnR 大幅下降 = hubness 签名；A<0 = R@1-forward）

| 方法 | 对照（=各自 baseline） | backbone | ΔR@1 | ΔR@5 | ΔR@10 | ΔMnR% | **A** | 形状 |
|---|---|---|---|---|---|---|---|---|
| **DAB** w/ Prob. only (5ep) | X-Pool 46.9/72.8/82.2/14.3 | B/32* | **−2.6** | +3.8 | +5.0 | **−48%** | **+7.0** | 🔴 hubness |
| **DAB** w/ Bridge only (5ep) | 同上 | B/32* | −3.7 | +3.8 | +4.1 | −41% | **+7.65** | 🔴 hubness |
| **DAB** 完整 (5ep) | 同上 | B/32* | +5.7 | +9.9 | +8.6 | −63% | **+3.55** | 🔴 hubness |
| **DAB** KL→W₂ cost (5ep) | 同上 | B/32* | **−0.1** | +6.9 | +6.2 | −54% | **+6.65** | 🔴 hubness（R@1 增益消失） |
| **DAB** 完整 (10ep 主表) | 同上 | B/32 | +9.3 | +12.6 | +10.2 | −71% | **+2.1** | 🔴 hubness |
| UATVR 概率增量 (+DUA†+KL† vs TI+DSA) | TI+DSA 49.6/75.5/84.9/12.5 | **B/16** | +1.2 | +0.8 | +0.6 | −1% | −0.5 | 🟢 R@1-forward |
| UATVR 全栈 vs TI | TI 48.4/74.2/83.3/14.1 | **B/16** | +2.4 | +2.1 | +2.2 | −12% | −0.25 | 🟢 R@1-forward |
| T-MASS | X-Pool 46.9/72.8/82.2/14.3 | B/32 | +3.3 | +2.5 | +2.9 | −17% | −0.6 | 🟢 R@1-forward |
| T-MASS | X-Pool 48.2/73.7/82.6/12.7 | B/16 | +4.5 | +3.4 | +3.0 | −17% | −1.3 | 🟢 R@1-forward |
| ProTA 概率增量 (DPA§+Lkl+AC vs 确定性 DPA) | DPA 47.0/75.0/84.1 | B/32 | +1.1 | +0.4 | +0.2 | n/a† | −0.8 | 🟢 R@1-forward |
| DiffusionRet (Gen+Dis vs Dis) | Dis 46.8/73.5/82.9/13.0 | B/32 | +2.2 | +1.7 | −0.2 | −7% | −1.45 | 🟢 R@1-forward |
| PAU (全配置 vs baseline) | 45.6/72.9/81.5/14.5 | B/32 | +2.9 | −0.2 | +1.0 | −3% | −2.5 | 🟢 R@1-forward |
| DUQ (全配置 vs L_S baseline) | 46.9/74.5/82.2/14.3 | B/32‡ | +4.3 | +2.8 | +3.9 | −25% | −0.95 | 🟢 R@1-forward |
| DUQ V2T 同上 | 44.4/73.3/84.0/9.0 | B/32‡ | +6.0 | +5.9 | +3.5 | −29% | −1.3 | 🟢 R@1-forward |

\* DAB 表 5 未明示 backbone；baseline 行与主表 B/32 X-Pool 行逐数相同（46.9/72.8/82.2/14.3），推断为 B/32。
† ProTA 消融表（其 Table 1）无 MdR/MnR 列。
‡ DUQ 表 3 第 6 行与其主表 B/32 行逐数相同（51.2/77.3/86.1/1.0/10.8），推断为 B/32。

**参照系（我们自己的显式 hubness 校正,ledger E-08）：** DSL T=2.0 在 June backbone 上
+2.6 T2V R@1（50.0→52.6，**JSFUSION test**）。Tier 0 补齐（2026-08-28,E-86）：同一 DSL
在 trusted-v1 **val** 矩阵上为 **−0.65** R@1（A=−0.02,MnR +17%）——val 是低 hubness、
近饱和 regime（S(N10)=0.53,MnR 2.11），修正无的放矢；增益的 regime 依赖性本身佐证
hubness-校正解读。故本表的 DSL 参照行不能用 val 形状填充，完整 test 形状留待 Tier 1
设置（PC-4 解锁后）。详见 `tier0-correction-envelope.md`。

---

## 判读

1. **签名不是家族性质。** 六个非 DAB 方法全部 A<0。其中三个的对照特别锋利，
   因为拆出了"纯概率增量"：UATVR 的 +DUA+KL（+1.2/+0.8/+0.6，MnR 几乎不动）、
   ProTA 的 DPA§+Lkl+AC（+1.1/+0.4/+0.2）、DiffusionRet 的 +Gen（+2.2/+1.7/−0.2）。
   概率组件在这些方法里改善的主要就是 top-1 匹配。
2. **两个"乘性修正"方法也是 R@1-forward。** PAU 的 re-rank（exp(−uncertainty)⊙S）和
   DUQ 的推理重排（Eq.16：exp(−γ₁u_d)(1−d)∘exp(−γ₂u_s)s）都属于逐分量乘性修正族，
   但形状均为 R@1-forward——**"用了不确定性重排"并不自动产生 hubness 签名**。
3. **DAB 内部的机制证据反而最强。** 三条独立线索都指向其非对称 KL cost 的加性方差项：
   (a) 只加概率头 R@1 **降** 2.6 但 MnR 腰斩（A=+7.0）；
   (b) KL→W₂ 后 R@1 增益**归零**（46.8 vs baseline 46.9）而重排效应保留——
   W₂ 的方差项 (σ_t−σ_v)² 不含自由加性 log σ² 项，这正是 OQ-1 臂 5/臂 6 逻辑的
   论文内置版本；
   (c) 主表增益形状 A=+2.1 仍为正，且 MnR −71% 远超家族其他成员（−1% ~ −29%）。
4. **对 OQ-1 的影响。** Tier 1 的靶子和 claim 收窄为"KL-cost 类概率打分"；
   相应地，论文叙事从"概率方法=隐式 hubness 校正"（已证伪）变为
   **"非对称 KL 打分中的 candidate-only 方差项是一个学习出来的 hubness 校正；
   该项的有无解释了 DAB 与其余概率方法增益形状的分野"**——这是一个更强、
   更可防守的 claim，且 Tier −1 的这张表本身就是它的第一份证据。

## 混淆变量与诚实性备注

- **backbone 不齐**：UATVR 消融是 B/16（论文明示"ablations take ViT/B-16 by default"），
  其余为 B/32。A 指数是同方法内部的差值,跨行比较形状而非绝对值,受此影响有限,但正式成文时须标注。
- **训练预算**：DAB 表 5 是 5 epoch,其主表 10 epoch——两处分别与各自 baseline 配对,未跨用。
  DUQ 全家 5 epoch(论文 §4.1)。
- **后处理**：UATVR/T-MASS/ProTA/DiffusionRet 的消融数字均明示**无** DSL/QB-Norm;
  PAU/DUQ 的"全配置"含各自的不确定性重排(上表第 2 点已单独讨论);
  UATVR 主表的 ‡ 行(DSL/QB-Norm 后处理)未采用。
- **抄录方式与复核（2026-08-28 当日完成，债已还清）**：T-MASS/ProTA/UATVR/
  DiffusionRet/PAU 五篇初次经 WebFetch(arXiv HTML/ar5iv)由模型转录。同日已用
  `arxiv` 包下载五篇官方 PDF（存于 `related_work/{slug}/paper.pdf`，附 `SOURCE.txt`
  记版本号），`pdftotext -layout` 转文本后**逐数比对全部本表引用值：5/5 完全一致，
  零转录错误**（详见 ledger E-89）。DUQ 直接从 PDF 页面图像读取(本地存档);
  DAB 来自本地 `dab.txt`(逐行核对)。**本表数字现可直接用于投稿。**
  复核中补获的细节：UATVR 消融表另有 `+DUA† 50.1/75.8/84.6/1.5/12.8` 中间行
  （本表的概率增量取 TI+DSA → +DUA†+KL† 累积口径，不受影响）;ProTA 消融表确无
  MdR/MnR 列（其主表给 ProTA 12.5，但无 DPA-only 对应值，故 ΔMnR 仍为 n/a）。
- PAU 是 cross-modal(含图文)方法,此处只取其 MSRVTT 视频检索表。

## 来源

| 方法 | 论文 | 数据出处 |
|---|---|---|
| DAB | arXiv 2607.20984 (ECCV'26) | 本地 `related_work/dab_eccv2026/dab.txt:407-422,555-567`；`NOTES.md:63-97` |
| UATVR | arXiv 2301.06309**v2** (ICCV'23) | 其 Table 1(消融,B/16)、Table 6(主表)。本地 `related_work/uatvr_iccv2023/paper.{pdf,txt}` 已复核 ✓ |
| T-MASS | arXiv 2403.17998**v1** (CVPR'24) | 其 Table 6(a)(消融)、Table 1(主表)。本地 `related_work/tmass_cvpr2024/` 已复核 ✓ |
| ProTA | arXiv 2404.12216**v2** | 其 Table 1(消融含全部中间行)。本地 `related_work/prota_2024/` 已复核 ✓ |
| DiffusionRet | arXiv 2303.09867**v2** (ICCV'23) | 其 Table 3d(训练策略消融)、Table 1。本地 `related_work/diffusionret_iccv2023/` 已复核 ✓ |
| PAU | arXiv 2309.17093**v3** (NeurIPS'23) | 其 Table 6(消融)、Table 1(MnR)。本地 `related_work/pau_neurips2023/` 已复核 ✓ |
| DUQ | IJCAI 2025, DOI 10.24963/ijcai.2025/643 | 其 Table 3(消融)、Table 1；本地存档 `related_work/duq_ijcai2025/duq.pdf`(p.5-6) |
