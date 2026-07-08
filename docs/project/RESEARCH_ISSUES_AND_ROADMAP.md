# 当前研究问题清单与改进路线

> 面向后续 agent 的阅读入口。本文汇总 `report.md`、`report_SAP.md`、`report_uncertainty.md` 的当前结论，并以 2026-07-05 已修正文档为准。具体细节请按索引回到原文。

## 0. 阅读索引

| 索引 | 文档位置 | 主要内容 |
|------|----------|----------|
| R-main-1 | `report.md:9` | 最终检索分数仍基本是 `wti_logits` |
| R-main-2 | `report.md:42` | SAP/不确定性不是 pair-level retrieval uncertainty |
| R-main-3 | `report.md:79` | `logsigma` / uncertainty 塌缩或饱和风险 |
| R-main-fix-1 | `report.md:106` | `uncertainty_mode=none` 语义已修正 |
| R-main-4 | `report.md:120` | 当前 evidential loss 不是真正的 evidential retrieval |
| R-main-5 | `report.md:142` | `w_uncertainty_reg` 仍是 dead knob |
| R-main-6 | `report.md:170` | 文本概率分支 attention mask 已接入，不再是当前 bug |
| R-main-7 | `report.md:225` | 建议先修训练目标和评估分数不一致 |
| R-main-8 | `report.md:348` | 建议转向 pair-level uncertainty 和 soft false-negative |
| R-sap-1 | `report_SAP.md:4` | 当前代码核对：SAP 结构瓶颈仍成立，卫生项已部分修复 |
| R-sap-2 | `report_SAP.md:106` | SAP 是 query-agnostic，需要 query-conditioned anchors |
| R-sap-3 | `report_SAP.md:151` | SAP 尚未真正进入 final ranking score |
| R-sap-4 | `report_SAP.md:201` | `evidential_head(projected.detach())` 阻断 gate 对 SAP decoder 的反馈 |
| R-sap-5 | `report_SAP.md:231` | 当前 uncertainty 是 anchor 统计量，不是真正检索不确定性 |
| R-sap-6 | `report_SAP.md:278` | `logsigma` 是 scalar expand，表达力弱 |
| R-sap-7 | `report_SAP.md:419` | 最小改动版：SAP-ColBERT / AnchorWTI scoring |
| R-sap-8 | `report_SAP.md:473` | Query-conditioned SAP gate |
| R-sap-9 | `report_SAP.md:527` | Per-anchor Gaussian mixture |
| R-sap-10 | `report_SAP.md:569` | 用 SAP uncertainty 做 false-negative soft label |
| R-sap-11 | `report_SAP.md:699` | SAP 最小可行实验路线 |
| R-unc-1 | `report_uncertainty.md:132` | uncertainty 没有进入 final ranking score |
| R-unc-2 | `report_uncertainty.md:178` | 当前 uncertainty 是 sample/video-level，不是 pair-level |
| R-unc-3 | `report_uncertainty.md:204` | 视频侧 `logsigma` 太粗且塌缩 |
| R-unc-4 | `report_uncertainty.md:237` | MIL sampling 容易鼓励 variance shrinkage |
| R-unc-5 | `report_uncertainty.md:260` | UACL closest sample 进一步鼓励低方差 |
| R-unc-6 | `report_uncertainty.md:285` | `uncertainty_mode=none` 语义已修正 |
| R-unc-7 | `report_uncertainty.md:299` | 文本侧 padding mask 已接入 |
| R-unc-8 | `report_uncertainty.md:314` | evidential loss 未真正使用 Dirichlet evidence |
| R-unc-9 | `report_uncertainty.md:390` | Pair-level Retrieval Uncertainty |
| R-unc-10 | `report_uncertainty.md:459` | Closed-form Gaussian Score，少用 Monte Carlo |
| R-unc-11 | `report_uncertainty.md:496` | 真正的 Evidential Ranking Head |
| R-unc-12 | `report_uncertainty.md:542` | Uncertainty-aware Soft False Negative Learning |
| R-unc-13 | `report_uncertainty.md:573` | Calibration-aware uncertainty |
| R-unc-14 | `report_uncertainty.md:680` | 最小可行改法 E1-E6 |

## 1. 当前结论

当前研究的核心矛盾不是某个 loss 权重没调好，而是 **不确定性/SAP 分支没有成为最终检索决策的一部分**。当前主排序仍是 WTI，SAP、Gaussian sampling、evidential、MIL、UACL 更像辅助分支。辅助分支即使改变了诊断统计，也很难稳定改变 Recall@K。

已经修正或不应再重复处理的基础项：

- `uncertainty_mode=none/evidential/nig_mil` 语义已修正，`none` 真实关闭 evidential/neg_reg。见 R-main-fix-1、R-unc-6。
- 文本概率分支已接入 attention/padding mask。见 R-main-6、R-unc-7。
- `logsigma_v/t_min_ratio/max_ratio` 已进入日志，应作为后续判断方差塌缩的主要依据。见 R-main-3。
- `EXPERIMENT_PROFILE=hygiene` 已提供 WTI-only 干净归因入口。

仍然成立的核心问题：

- Final ranking 仍是 `wti_logits`。
- SAP 仍是 video-only，不是 query-conditioned。
- 视频侧 uncertainty 仍是 video/sample-level 统计量，不是 query-video pair-level 置信度。
- `logsigma` 仍然表达力弱，且 sampling/MIL/UACL 容易推动方差收缩。
- evidential loss 仍没有和 Dirichlet evidence / final logits 形成闭环。
- `w_uncertainty_reg` 仍没有进入主 loss，继续视为 dead knob。

## 2. 问题清单

### P0. Final Ranking 与概率/不确定性分支脱节

**现象**：训练和评估主检索分数仍是 `wti_logits`，概率分支只通过辅助 loss 间接影响共享参数。

**影响**：SAP、Gaussian、evidential 即使学到统计差异，也不一定改变最终排序。继续堆 `w_mil / w_evidential / w_neg_reg / w_orth / UACL` 的收益上限很低。

**索引**：R-main-1、R-main-7、R-sap-3、R-unc-1。

**建议**：第一阶段先验证 `final_logits = wti_logits + lambda * sap_or_prob_logits`。训练和评估必须使用同一 `final_logits`。

### P1. SAP 是 video-only anchor pooling，不是 query-conditioned matching

**现象**：SAP 只看视频生成 anchors、modal_probs、`mu_video/logsigma_video`。同一个视频面对不同 query 时，anchor 权重不会变。

**影响**：视频文本检索的相关性本质是 query-dependent。video-only 聚合无法表达“这个视频对 query A 确定、对 query B 不确定”。

**索引**：R-sap-2、R-sap-8、R-unc-2。

**建议**：保留 video anchors，但把聚合改成 pair-level gate：`text_i + anchor_jk -> gate_ijk -> query-conditioned video representation`。

### P2. 当前 uncertainty 不是检索不确定性

**现象**：视频侧 uncertainty 主要来自 anchor diversity 和 modal entropy；evidential similarity 用 video-level scalar confidence 对同一视频的所有文本统一折扣。

**影响**：它更像 video complexity，不是 `u(text_i, video_j)`。无法用于判断 Top-1 是否可信、负样本是否可能是 false negative。

**索引**：R-main-2、R-sap-5、R-unc-2、R-unc-9。

**建议**：定义 pair-level uncertainty head，输入可包含 `wti_ij`、`sap/prob_ij`、`|mu_t - mu_v|`、`mu_t * mu_v`、`var_t/var_v`，输出 `u_ij` 或 evidence/confidence。

### P3. 视频侧 `logsigma` 表达力弱且容易塌缩

**现象**：当前 `logsigma_video` 由 anchor 间方差压成 `[B,1]` 再 expand 到 `[B,D]`。MIL sampling 和 UACL closest sample 都倾向让采样更接近 mean。

**影响**：方差不能区分语义维度或 anchor 级不确定性，也容易退化成低噪声确定性分支。

**索引**：R-main-3、R-sap-6、R-unc-3、R-unc-4、R-unc-5、R-unc-10。

**建议**：短期用 closed-form score 代替或弱化 sampling；中期改为 per-anchor Gaussian mixture：每个 anchor 输出 `mu_k/logvar_k/evidence_k`，再按 mixture variance 聚合。

### P4. SAP Dirichlet gate 与 anchor decoder 耦合不足

**现象**：`evidential_head(projected.detach())` 仍在，evidence/gate 不能直接反向塑造 decoder anchors。

**影响**：Dirichlet head 更像旁路加权器，不是端到端语义选择器。

**索引**：R-sap-4、R-sap-11。

**建议**：做 detach 消融：`detach`、`no detach`、`partial detach`。推荐先试 `projected.detach() + grad_ratio * (projected - projected.detach())`，并控制 evidence loss 权重或 warmup。

### P5. Evidential 分支没有形成 ranking-level 证据学习闭环

**现象**：`_evidential_nll_loss(sim_matrix, alpha_dir)` 未真正使用 `alpha_dir`；`evidential_matrix_loss()` 存在但未接入主 loss；当前 evidential similarity 是旁路 `ev_sim`。

**影响**：evidential 分支更像 similarity regularizer，不是 batch-level / ranking-level evidence learning。

**索引**：R-main-4、R-sap-10、R-unc-8、R-unc-11。

**建议**：等 `final_logits` 建立后，把 evidence 作用到 final logits：`evidence_ij = softplus(final_logits_ij)`，用 Dirichlet MSE/NLL 或 calibration loss 训练。

### P6. MSR-VTT false negative / 多正例歧义没有被正面处理

**现象**：Hard negative 全链路已显示不稳定，很多高相似负样本可能是语义近邻或潜在正例。

**影响**：继续硬推远难负样本会伤害 Top-1 边界；UACL/HN 的负结果都指向当前数据歧义需要 soft treatment。

**索引**：R-sap-10、R-unc-12。

**建议**：用 SAP/prob score + pair uncertainty 发现 soft false negative，把 one-hot CE 部分替换或补充为 soft-label sigmoid contrastive。

### P7. 当前 dead knobs 与历史兼容参数会污染实验解释

**现象**：`w_uncertainty_reg`、`w_query_sim`、`fusion_mode` 等在当前主排序路径下不影响 final ranking，或仅作兼容保留。

**影响**：继续 sweep 这些参数会浪费实验预算，并污染因果解释。

**索引**：R-main-5、R-unc-14。

**建议**：实验表中标记为 inactive/compatibility；若保留 `w_uncertainty_reg`，必须明确“当前未接入 loss”。不要把它解释为有效 causal knob。

## 3. 建议改进路线

### Phase 0：实验卫生与归因基线

**目标**：确保后续任何提升都能归因到 backbone 或新 scoring 设计，而不是辅助项混杂。

**动作**：

1. 归档 UACL epoch-4 止损结果。
2. 跑 hygiene WTI-only baseline。
3. 在日志中确认 `score_source=wti_logits`、active loss 状态、`logsigma_v/t` clamp ratio。
4. 标记 `w_uncertainty_reg` 等 dead knobs，不再 sweep。

**通过标准**：hygiene baseline 有完整 eval；报告中明确 default/hygiene 的差异。

**索引**：R-main-6、R-unc-14。

### Phase 1：验证 SAP/概率均值是否有检索价值

**目标**：先回答 `mu_video` / SAP anchors 本身有没有排序价值。

**推荐实验**：

| 实验 | Final score | 目的 |
|------|-------------|------|
| E1.0 | `WTI` | 当前干净 baseline |
| E1.1 | `WTI + lambda * (mu_t @ mu_v)` | 验证概率均值是否有用 |
| E1.2 | `WTI + lambda * AnchorWTI(text_tokens, anchors)` | 验证 anchors 是否有 late-interaction 价值 |

**注意**：先关闭 MIL/evidential/neg/UACL/orth，只验证分数源。训练和评估必须使用同一个 final score。

**停止条件**：如果 E1.1/E1.2 都不超过 hygiene baseline，不要继续做 uncertainty；应先改善 anchor 表征或转 backbone。

**索引**：R-main-7、R-sap-7、R-sap-11、R-unc-14。

### Phase 2：Query-conditioned SAP

**目标**：把 SAP 从 video-level pooling 改成 query-conditioned matching。

**推荐设计**：

```text
(text_i, anchors_jk)
→ gate_ijk
→ v_ij = sum_k gate_ijk * anchor_jk
→ sap_logits_ij = cos(text_i, v_ij)
→ final_logits = wti_logits + lambda * sap_logits
```

**验证点**：

- query-anchor gate entropy；
- 正确 Top-1 与错误 Top-1 的 gate 分布差异；
- anchor pairwise cosine；
- 是否需要 allgather anchors。

**停止条件**：如果 query-conditioned SAP 不优于 global SAP，说明 query gate 没学到有效选择，不应继续加复杂 uncertainty head。

**索引**：R-sap-8、R-sap-11。

### Phase 3：Closed-form uncertainty penalty

**目标**：不用 Monte Carlo sampling 先验证 variance 是否有排序价值。

**推荐设计**：

```text
prob_logits = cos(mu_t, mu_v)
unc_ij = mean(exp(logvar_t_i)) + mean(exp(logvar_v_j))
final_logits = wti_logits + lambda * prob_logits - beta * unc_ij
```

**注意**：这一步不是最终方法，只是诊断当前 variance 是否有用。若 penalty 降低指标，说明当前 variance 没有校准，先不要把它写成正向贡献。

**索引**：R-main-3、R-unc-10、R-unc-14。

### Phase 4：Pair-level uncertainty head

**目标**：把不确定性从 `u_video` 升级为 `u(text_i, video_j)`。

**推荐输入**：

```text
wti_logits_ij
sap/prob_logits_ij
|mu_t_i - mu_v_j|
mu_t_i * mu_v_j
var_t_i
var_v_j
top1-top2 margin or row entropy
```

**输出用途**：

- 调整 final score：`final_logits = base_logits - beta * u_ij`；
- 识别 false negative；
- 做 calibration 分析。

**验证点**：

- wrong top1 的 uncertainty 是否高于 correct top1；
- uncertainty 与 GT rank / margin 的相关性；
- high similarity negative 是否被预测为低置信或 soft positive。

**索引**：R-main-8、R-unc-9、R-unc-13。

### Phase 5：Evidential ranking 与 calibration

**目标**：让 evidence 对应 batch/ranking 决策，而不是旁路 `ev_sim`。

**推荐设计**：

```text
evidence_ij = softplus(final_logits_ij)
alpha_ij = evidence_ij + 1
prob_ij = alpha_ij / sum_j alpha_ij
uncertainty_i = K / sum_j alpha_ij
```

训练目标可以先接 `evidential_matrix_loss(final_logits)`，再考虑 soft-label Dirichlet target。

**前置条件**：必须先有稳定的 `final_logits`，否则 evidence loss 仍是旁路正则。

**索引**：R-main-4、R-unc-11。

### Phase 6：Soft false-negative learning

**目标**：替代 hard negative 的硬推远策略，适配 MSR-VTT 语义近邻/多正例歧义。

**推荐规则**：

```text
Y_ii = 1
Y_ij = q_ij in (0, 1), if i != j but high SAP/prob similarity and low pair uncertainty
Y_ij = 0, otherwise
```

Loss 先用 `BCEWithLogitsLoss(final_logits / tau, Y_soft)` 做消融，不要一次性叠太多复杂 target。

**索引**：R-sap-10、R-unc-12。

## 4. 推荐执行顺序

1. **先做 Phase 0**：hygiene baseline 与 UACL 止损归档。
2. **再做 Phase 1**：`WTI + prob/global SAP score` 与 `WTI + AnchorWTI`，证明 SAP/概率均值是否有排序价值。
3. **若 Phase 1 有收益**：进入 Phase 2 的 query-conditioned SAP。
4. **若 Phase 2 有收益**：进入 Phase 3/4，逐步接 closed-form uncertainty 与 pair-level uncertainty。
5. **只有 pair-level uncertainty 对错例有校准性**：再接 Phase 5 evidential ranking。
6. **最后做 Phase 6**：soft false negative，作为 MSR-VTT 数据歧义的论文创新点。

## 5. 不建议继续投入的方向

- 不建议继续 hard negative 主线或同机制 repeat。
- 不建议继续 UACL sweep，除非 epoch-4 止损结果超过 B1-only v2 的 49.3。
- 不建议继续 sweep `w_uncertainty_reg`、`w_query_sim`、`fusion_mode` 等当前无效 causal knobs。
- 不建议在 `final score = wti_logits` 不变的情况下继续堆 MIL/evidential/UACL 辅助项。
- 不建议在当前 scalar-expanded `logsigma` 上直接写强不确定性贡献，除非先证明其校准性。

## 6. 给后续 agent 的最小任务包

如果只做一轮最小改进，建议执行：

1. 读 `report.md:225`、`report_SAP.md:419`、`report_uncertainty.md:680`。
2. 实现一个可开关的 `final_score_mode`：
   - `wti`
   - `wti_prob_mu`
   - `wti_anchor_wti`
3. 确保 train/eval 都使用同一个 final logits。
4. 增加日志：`score_source`、`lambda_prob/lambda_anchor`、prob/SAP logits 的 diag/offdiag gap。
5. 先跑 hygiene 条件下的 E1.0/E1.1/E1.2。

这一步的价值是判断“SAP/概率均值是否有直接排序价值”。如果答案是否定的，后续 pair-level uncertainty、evidential ranking、soft false-negative 都应该暂停，优先转 backbone 或重做 SAP 表征。
