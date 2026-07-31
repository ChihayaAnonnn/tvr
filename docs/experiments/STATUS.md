# STATUS — 当前实验状态

<!-- 覆盖写，≤40 行（exp_context.sh 会检查并提醒）。只描述"现在"，历史归 JOURNAL.md
     和 baseline-gap-diagnosis.md。一步实验收尾时更新；SessionStart / PostCompact 自动注入。 -->

**更新于**: 2026-07-31

## 在追的问题

在查询歧义下恢复**条件覆盖**：全局 conformal 阈值给边际 90% 覆盖，但按真实 |Rel|
分层后 83.3%(=1) → 98.3%(6+)，spread 15.0 pt；oracle 分层压到 0.7 pt，免费特征只到
13.3 pt。**这段差距就是开放问题**——第一个可测、有 ground truth、不是 R@1 的目标。

## 上一步结论

σ（A4 的 DUA log-variance）当分层变量是**负结果**：配对 +0.70±0.12 pt，方向是错的。
噪声 oracle 扫描把门槛钉在 **ρ ≈ 0.83** 才能把 gap 减半，而免费特征加 σ 只到 0.55。

## 正在跑：无

## 下一步（靶子：ρ 0.55 → 0.83）

1. **学习式歧义预测器** — 文本编码器在 |Rel| 上 fine-tune；四个标量特征的线性拟合
   不算认真尝试，而这个目标是有标注的。
2. **headline 改为固定*条件*覆盖下的平均集合大小**，R@1 并列仅为可比性。

## 已排除的死路（不要重开）

- **DUA 家族整体**（我们的和 UATVR 的）。八臂两协议没一个超过 +0.4，同配置复现误差
  就有 1.1；σ 的三种用法（重排/错误预测/分层）全负。别重跑，它们买的是可比性。
- **自适应 margin 集合**优于固定 top-k：四个 run 里三个持平或更大。
- **标签错误腐蚀 risk–coverage** 这个假设：AURC 相对降幅在两种标签下都是 45%。

## 长期约束

- 所有评分**同时**报原始标签和 FIRE 标签，加报 judged@k 让 pool 深度可见。
- 同一 split 上两个臂比较用**配对 McNemar**，不用跨 run 差值。per-query rank 在
  `final_test.json` → `selections.t2v.test_metrics.{t2v,v2t}.cols`。
- `any` 覆盖是下界：集合里未被判断的相关视频不计入。
- 预测器只按 spread 打分，绝不按 ρ。
