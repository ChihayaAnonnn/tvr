# 当前研究问题清单与改进路线

> 面向后续 agent 的阅读入口。本文汇总 `report.md`、`report_SAP.md`、`report_uncertainty.md` 的当前结论，并以 2026-07-10 的决策口径为准。具体细节请按索引回到原文。

> 归档约定：本文是当前工作树唯一科研事实源。历史原始日志、诊断 TSV、已执行计划与过期 checkpoint 已移出工作树，需要复核时从 Git 历史恢复。

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
| R-main-8 | `report.md:348` | pair-level uncertainty 与历史 soft false-negative 建议；后者已被 trusted-v1 禁用 |
| R-sap-1 | `report_SAP.md:4` | 当前代码核对：SAP 结构瓶颈仍成立，卫生项已部分修复 |
| R-sap-2 | `report_SAP.md:106` | SAP 是 query-agnostic，需要 query-conditioned anchors |
| R-sap-3 | `report_SAP.md:151` | SAP 尚未真正进入 final ranking score |
| R-sap-4 | `report_SAP.md:201` | `evidential_head(projected.detach())` 阻断 gate 对 SAP decoder 的反馈 |
| R-sap-5 | `report_SAP.md:231` | 当前 uncertainty 是 anchor 统计量，不是真正检索不确定性 |
| R-sap-6 | `report_SAP.md:278` | `logsigma` 是 scalar expand，表达力弱 |
| R-sap-7 | `report_SAP.md:419` | 最小改动版：SAP-ColBERT / AnchorWTI scoring |
| R-sap-8 | `report_SAP.md:473` | Query-conditioned SAP gate |
| R-sap-9 | `report_SAP.md:527` | Per-anchor Gaussian mixture |
| R-sap-10 | `report_SAP.md:569` | 历史 false-negative soft label 建议；已被 trusted-v1 禁用 |
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
| R-unc-12 | `report_uncertainty.md:542` | 历史 uncertainty-aware soft target 建议；已被 trusted-v1 禁用 |
| R-unc-13 | `report_uncertainty.md:573` | Calibration-aware uncertainty |
| R-unc-14 | `report_uncertainty.md:680` | 最小可行改法 E1-E6 |

## 1. 当前结论

当前研究的核心矛盾不是某个 loss 权重没调好，而是 **不确定性/SAP 分支没有成为最终检索决策的一部分**。当前主排序仍是 WTI，SAP、Gaussian sampling、evidential、MIL、UACL 更像辅助分支。辅助分支即使改变了诊断统计，也很难稳定改变 Recall@K。

已经修正或不应再重复处理的基础项：

- `uncertainty_mode=none/evidential/nig_mil` 语义已修正，`none` 真实关闭 evidential/neg_reg。见 R-main-fix-1、R-unc-6。
- 文本概率分支已接入 attention/padding mask。见 R-main-6、R-unc-7。
- `logsigma_v/t_min_ratio/max_ratio` 已进入日志，应作为后续判断方差塌缩的主要依据。见 R-main-3。
- `EXPERIMENT_PROFILE=hygiene` 当前执行 trusted-v1 的 WTI-only forward；SAP、SpatialEnhancer 和概率辅助前向均应由代码路径绕过，并由测试持续验证。

### P0：可信实验基座与新基线（2026-07-10）

- 旧结果存在 JSFusion test 逐 epoch 选模、同视频描述被当作负例、WTI padding 最大池化三项混杂，只保留为历史档案。
- 新协议固定为 trusted-v1：8500 train / 500 internal val / JSFusion 1K blind test。
- 主损失按精确 video_id 使用双向多正例 InfoNCE。
- 下一次可解释实验必须先重跑 OpenAI CLIP hygiene WTI-only；未完成该基线前，不判断 EVA adapter、SAP 或不确定性模块收益。
- OpenAI hygiene 新基线建立后，EVA02-CLIP-B/16 只能在相同 split、global contrastive batch、optimizer steps 和 checkpoint-selection 指标下比较。

**停止条件**：trusted-v1 的固定拆分、独立 test 隔离、精确 `video_id` 多正例损失、WTI padding 修复和 hygiene WTI-only 前向绕过必须通过代码与测试验证；任一项未通过，不启动 backbone 对照或新的模型主线实验。

已实施、后续按停止条件运行验证的实验协议：

- MSRVTT 后续唯一有效协议为 `trusted-v1`：seed 42 固定拆分 8500 train / 500 internal val，val 每个视频恰好使用 20 条官方描述。
- JSFusion 1K test 只能在训练结束后通过独立、显式 test eval 使用；训练阶段不得构造或评估 test dataloader。
- 正例矩阵只能由精确相同 `video_id` 构造；双向多正例 InfoNCE 是唯一主检索损失。语义相似度只允许用于只读诊断，不得生成软正例、伪标签或改变训练 target。
- trusted-v1 hygiene WTI-only 必须在 forward 路径直接绕过 SpatialEnhancer、SAP、视频概率分支、文本 PIENet、不确定性头，以及概率采样和相关中间张量构造；仅把辅助 loss 权重置零不满足该约束。
- 上述纯 WTI forward 绕过、dataloader、loss 和训练/test 隔离已在代码中实施；历史 48.2/49.3 等结果只作历史参照，不能冒充 trusted-v1 基线。

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

**影响**：继续硬推远难负样本会伤害 Top-1 边界；但不能用语义相似度把跨视频样本改成软正例，否则会破坏 trusted-v1 的可审计正例定义。

**索引**：R-sap-10、R-unc-12。

**建议**：主训练只按精确 `video_id` 构造多正例矩阵，并使用双向多正例 InfoNCE。SAP/prob score 与 pair uncertainty 最多用于只读错误分析或校准报告，不得修改正例矩阵或主检索 target。

### P7. 当前 dead knobs 与历史兼容参数会污染实验解释

**现象**：`w_uncertainty_reg`、`w_query_sim`、`fusion_mode` 等在当前主排序路径下不影响 final ranking，或仅作兼容保留。

**影响**：继续 sweep 这些参数会浪费实验预算，并污染因果解释。

**索引**：R-main-5、R-unc-14。

**建议**：实验表中标记为 inactive/compatibility；若保留 `w_uncertainty_reg`，必须明确“当前未接入 loss”。不要把它解释为有效 causal knob。

## 3. 建议改进路线

### Phase 0：实验卫生与归因基线（已完成并止损）

**目标**：确保后续任何提升都能归因到 backbone 或新 scoring 设计，而不是辅助项混杂。

**已完成**：

1. legacy loss-zero hygiene 历史基线已完成，T2V R@1 = 48.2；该运行未验证概率辅助模块在 forward 中被绕过，不是 trusted-v1 的纯 WTI 基线。
2. UACL 第 4 epoch 四组结果已归档为 49.3、49.2、49.4、49.0；seed 43 的单次 49.4 仅比 49.3 高 0.1，同配置 seed 42 为 49.2，不构成跨配置、跨种子的稳定增益。
3. UACL 路线已冻结，不再追加 epoch、sweep 或同机制 repeat。
4. `w_uncertainty_reg` 等 dead knobs 已标记为 inactive/compatibility，不再 sweep。

**结论**：Phase 0 的旧路线排查已关闭；48.2 是 legacy hygiene 历史参考，不是 trusted-v1 基线。真正的 hygiene WTI-only forward 绕过已在当前 trusted-v1 P0 实施，须用 spy/mock 测试验证 SpatialEnhancer、SAP、PIENet、不确定性与采样路径均未调用。

**索引**：R-main-6、R-unc-14。

### Phase 1：验证 SAP/概率均值是否有检索价值（已完成并止损）

**目标**：先回答 `mu_video` / SAP anchors 本身有没有排序价值。

**已完成实验**：

| 实验 | Final score | 目的 |
|------|-------------|------|
| E1.0 | `WTI` | legacy hygiene baseline = 48.2 |
| E1.1 | `WTI + lambda * (mu_t @ mu_v)` | global `prob_mu` 正负分数 gap 接近 0，无稳定收益 |
| E1.2 | `WTI + lambda * AnchorWTI(text_tokens, anchors)` | AnchorWTI 正负分数 gap 接近 0，无稳定收益 |

**结论**：Phase 1 已满足停止条件并关闭；不再扩展 global score 融合或 AnchorWTI sweep。

**索引**：R-main-7、R-sap-7、R-sap-11、R-unc-14。

### Phase 2：Query-conditioned SAP（已完成并止损）

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

**2026-07-09 更新**：已完成 `FINAL_SCORE_MODE=wti_qc_sap, lambda_qc_sap=0.1` 的 hygiene 止损验证。
ckpt2 eval T2V R@1 = 47.9，低于 hygiene WTI-only 48.2 和 B1-only v2 49.3；训练诊断中
`qc_sap_gap` 长期接近 0，正负样本 gate entropy/top1 mass 无有效差异。结论：Phase 2 不继续扩展，
不做 hard top-k anchor selection，也不继续叠 query-conditioned uncertainty；后续先执行 trusted-v1 P0，
只有在 OpenAI hygiene 基线建立后才进行 backbone 对照。

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
- high-similarity cross-video 样本是否呈现低置信；该结果只作诊断，不得据此改写正例标签。

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

若后续恢复该方向，evidential/calibration 只能作为不改变正例矩阵的辅助诊断目标；主检索损失仍固定为按精确 `video_id` 构造的双向多正例 InfoNCE。

**前置条件**：必须先有稳定的 `final_logits`，否则 evidence loss 仍是旁路正则。

**索引**：R-main-4、R-unc-11。

### Phase 6：Semantic soft-target 路线（已终止）

**终止原因**：SAP/prob 相似度不能证明跨视频样本是正例；使用 `Y_soft`、soft positive、伪标签或语义 BCE target 会让正例定义不可审计，并与 trusted-v1 冲突。

**固定约束**：正例矩阵仅由相同 `video_id` 构造，主检索目标固定为双向多正例 InfoNCE。语义近邻和疑似 false negative 只进入只读诊断报告，不参与标签生成或 loss target。

**索引**：R-sap-10、R-unc-12。

## 4. 推荐执行顺序

1. **Phase 0 已完成**：legacy hygiene=48.2；UACL epoch-4 已归档并冻结。
2. **Phase 1 已完成**：global `prob_mu` 与 AnchorWTI 无稳定收益，停止继续 sweep。
3. **Phase 2 已完成**：query-conditioned SAP 未证明收益，停止 SAP gate/top-k/uncertainty 复杂化。
4. **当前先运行 trusted-v1 验证**：确认固定 8500/500 split、独立 test eval、多正例 InfoNCE、WTI padding 修复，以及真正绕过 SpatialEnhancer/SAP/概率分支/PIENet/不确定性/采样张量构造的 hygiene 前向；在代码与绕过测试完成前不启动新的可信主线实验。
5. **随后做匹配 backbone 对照**：用同一 seed、GPU 数、`batch_size`、`gradient_accumulation_steps` 和每次 forward 全局对比 batch，成对运行 OpenAI CLIP 与 EVA02-CLIP-B/16。
6. **首要判断依据**：EVA 相对同配置 OpenAI CLIP 的提升；legacy 48.2/49.3 只作历史参考。
7. **Phase 6 已终止**：不得恢复语义 soft target、伪标签或 soft-positive BCE。

## 5. 不建议继续投入的方向

- 不建议继续 hard negative 主线或同机制 repeat。
- 不建议继续 UACL sweep/repeat；epoch-4 已归档，单次 49.4 属噪声级结果，路线已冻结。
- 不建议继续 sweep `w_uncertainty_reg`、`w_query_sim`、`fusion_mode` 等当前无效 causal knobs。
- 不建议在 `final score = wti_logits` 不变的情况下继续堆 MIL/evidential/UACL 辅助项。
- 不建议在当前 scalar-expanded `logsigma` 上直接写强不确定性贡献，除非先证明其校准性。
- 禁止使用语义相似度构造 soft positive、伪标签、`Y_soft` 或 BCE/soft-label 主检索目标。

## 6. 给后续 agent 的最小任务包

如果只做一轮最小改进，建议执行：

1. 先按已实施设计验证 `trusted-v1`，包括 dataloader/loss、test 隔离和真正的 hygiene WTI-only 前向绕过。
2. 以 seed 42 和完全匹配的设备/batch 配置重跑 OpenAI CLIP trusted hygiene WTI-only 基线。
3. 仅在同一协议、同一设备/batch 配置下运行 EVA02-CLIP-B/16 配对实验。
4. 以 EVA 相对匹配 OpenAI CLIP 的差值作为首要判断依据，48.2/49.3 只作 legacy 历史参考。
5. 不恢复已关闭的 UACL、global score、AnchorWTI、QC-SAP 或 semantic soft-target sweep。
