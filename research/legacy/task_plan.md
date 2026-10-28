# Task Plan — UATVR 论文初稿路径

**Goal（来自 /goal）:** 查看当前课题思路、创新点、实验结果；设计各种实验；出图；完成简易论文初稿放根目录；适当思考新的创新点；可以搜索并总结论文。

**Owner:** Claude (session 2026-08-13)
**Root artefacts:** `PAPER.md`（初稿）、`findings.md`（证据集）、`progress.md`（状态跟踪）、`paper/figs/`（图表）

---

## Milestone 1 · 现状盘点（本 session 完成）

- [x] 读通 `docs/deep-research-report.md`、`docs/superpowers/plans/*`、`specs/*`、`reports/*`
- [x] 定位 A1 sidecar 验收数值（`logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log`）
- [x] 汇总 A2-CG 失败证据（`reports/2026-08-12-csd-logvar-a2-results.md`）
- [x] 汇总 CRF 探针 audit（`tables/2026-08-11_r1_and_rho.md`）
- [ ] 写 `findings.md`：把所有可入论文的数字集中一份

## Milestone 2 · 图表制作（dataviz 规范）

- [ ] `paper/figs/fig1_architecture.svg` — pipeline 图：frozen retriever + A1 sidecar + CRF probe
- [ ] `paper/figs/fig2_a1_vs_a2cg.pdf` — 柱状图：ρ / AUROC / AURC 三指标对比
- [ ] `paper/figs/fig3_crf_ladder.pdf` — u_video vs CRF level (6 级 + 15 pairwise heatmap)
- [ ] `paper/figs/fig4_risk_coverage.pdf` — selective retrieval accuracy 曲线（T2V + V2T）
- [ ] `paper/figs/fig5_topk_score_features.pdf` — 8 维 score-distribution 特征分布（正确 vs 失败 query）
- [ ] 所有图统一 palette（`dataviz` 默认色）、白/深底可读、300dpi

## Milestone 3 · 论文初稿（PAPER.md）

结构：Abstract → Intro → Related Work → Method → Experiments → Discussion → Limitations → Conclusion。

- [ ] Abstract & Intro（问题 → 我们的三点贡献）
- [ ] Related Work（分四路：probabilistic embedding / evidential / rejection / interactive）
- [ ] Method（A1 sidecar 数学 + CRF probe + 与 A2-CG 反证）
- [ ] Experiments（表 1 主结果、表 2 消融、表 3 CRF ladder、表 4 selective retrieval）
- [ ] Discussion（为什么 candidate-independent variance 会 collapse；A3+I1 展望）
- [ ] Limitations（未做 MSVD/DiDeMo；单数据集；未做 open-set）

## Milestone 4 · 未来实验设计（写入 PAPER 的 Future Work + 独立追踪）

- [ ] A3 contextual retrieval risk v2（candidate-conditioned；已在 spec 阶段）
- [ ] I1 spherical concentration（1/(1+κ)；替代 512-dim diagonal Gaussian）
- [ ] A1 + I1 组合分数（是否达到 ρ > 0.6）
- [ ] MSVD / DiDeMo 迁移
- [ ] Open-set rejection 评测（Moment of Untruth 风格 negative queries）
- [ ] Conformal risk control 加持（把 sidecar 输出转成有覆盖率保证的 abstention）

## Milestone 5 · 新创新点头脑风暴（brainstorming 结果放 findings.md）

- [ ] 组合 A1 (contextual) + I1 (intrinsic) + CRF probe 为三通道 evidence fusion
- [ ] Sidecar 迁移到不同 retriever backbone 检验通用性（HEAD arch vs June arch）
- [ ] Sidecar 输出作为 conformal predictor 的 nonconformity score
- [ ] 与 UMIVR interactive query 结合：high-vacuity → 主动追问

## 进度总门槛

`PAPER.md` 出稿 + `paper/figs/*.pdf` 齐全 + `findings.md` 数字齐全 + `progress.md` 全部勾选 → goal 达成。
