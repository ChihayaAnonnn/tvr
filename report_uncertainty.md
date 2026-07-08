结论先说清楚：**当前不确定性模块的“研究方向”是合理的，但“当前实现闭环”不够合理。**
你现在同时有文本 Gaussian、视频 SAP 方差、MIL 采样、evidential loss、negative evidence regularization、UACL intra alignment、AdaNorm，但它们没有形成一个统一的检索不确定性目标。结果就是：模块很多，真正能稳定改变最终 Recall@K 的路径很少。

我会这样评价：

```text
不确定性研究方向：合理，值得保留
当前模块设计：偏散，闭环不强
当前提升困难的主要原因：uncertainty 没有直接控制 final ranking
建议路线：从 sample-level/video-level uncertainty 改成 pair-level retrieval uncertainty
```

---

# 1. 当前不确定性模块到底由哪些部分组成？

你现在的不确定性体系大概是 5 块。

## 1.1 文本侧 Gaussian uncertainty

模型初始化时，文本侧有 `PIENet`，然后根据参数选择 `UncertaintyModuleImage / UncertaintyModuleText / UncertaintyModuleTextMamba` 来输出 `logsigma`。

训练时，文本侧路径是：

```text
text_pooled + text_token
→ PIENet 得到 text mean
→ uncertain_net_text 得到 logsigma
→ 可选 AdaNorm
→ sample_gaussian_tensors
```

对应代码里，`probabilistic_text()` 先用 `pie_net_text` 得到 `out`，再用 `uncertain_net_text` 得到 `logsigma`，可选用 `ada_norm_text(out, logsigma)`，最后采样得到多个 text embeddings。

---

## 1.2 视频侧 SAP uncertainty

视频侧不再用独立 uncertainty head，而是由 SAP 输出：

```text
mu_video
logsigma_video
epistemic_video
u_mode
alpha_dir
modal_probs
```

SAP 的说明里写得很清楚：当前是“非学习不确定性版本”，不确定性来自 anchor 多样性和模态熵。

具体实现是：

```python
uncertainty = diversity * modal_entropy_norm
logsigma = log(anchor_dim_var)
logsigma = logsigma.expand(-1, d_model)
```

也就是视频侧 `logsigma` 由 anchor 间方差得到，并从 `[B,1]` 扩展成 `[B,D]`。

---

## 1.3 MIL probabilistic contrastive loss

训练时会从文本和视频 Gaussian 中采样：

```python
prob_video_embedding = sample_gaussian_tensors(mu_video, logsigma_video, n_video)
prob_text_embedding = sample_gaussian_tensors(out, logsigma, n_text)
```

`sample_gaussian_tensors()` 把 `logsigma` 当作 log-variance，采样时使用 `std = exp(0.5 * logsigma)`。

然后主模型用采样后的 text/video embeddings 计算 `MIL_loss`。

---

## 1.4 Evidential similarity / NLL / neg-reg

当前 evidential similarity 是：

```python
cosine_sim = mu_video @ text_pooled.T
confidence = exp(-mean(epistemic_video))
ev_sim = cosine_sim * confidence
```

也就是用视频侧 uncertainty 对视频和所有文本的相似度做统一折扣。

然后 `_evidential_nll_loss()` 对 diagonal score 做 `-log(relu(score))`，对 off-diagonal 正分数做惩罚。

---

## 1.5 UACL intra-modal uncertainty augmentation

你还加了 UACL-style intra-modal alignment：从 Gaussian samples 中选一个 augmentation，然后做 text-text / video-video intra-modal contrastive。当前默认策略支持 `closest` 或 `random`。

---

# 2. 哪些地方是合理的？

## 合理点 A：文本和视频都建模成分布，这是对的

视频文本检索天然存在一对多、多义文本、局部片段不完整、视觉噪声、caption 粗粒度等问题。把 embedding 从点向量扩展成 Gaussian 或概率分布，是合理路线。

---

## 合理点 B：你已经尝试把 uncertainty 分成 text-side 和 video-side

文本侧用 PIENet + uncertainty head，视频侧用 SAP anchor variance，这个方向比单一 sample-level variance 更细。

---

## 合理点 C：保留 WTI 作为强检索主干是对的

WTI 的 token-wise late interaction 是当前代码里最强、最稳定的排序信号。它对 text token 和 frame token 做 max interaction，再双向聚合。

所以不是说你应该删掉 WTI。相反，正确路线应该是：

```text
WTI 作为强主干
uncertainty 用来调节 WTI / 补充 WTI / 发现 false negative
```

而不是让 uncertainty 完全替代 WTI。

---

# 3. 当前设计中最不合理的地方

## 问题 1：不确定性没有进入最终 ranking score

这是最大问题。

训练时最终用于主检索 CE 的 logits 是：

```python
weighted_logits = wti_logits
```

注释也写明“WTI logits 直接作为检索分数”。

eval 时也直接返回 `wti_logits`。

所以当前真正决定 Recall@K 的路径是：

```text
text/frame token → WTI logits → CE / Eval
```

不确定性路径是：

```text
SAP / Gaussian / evidential / MIL / UACL
→ 辅助 loss
→ 间接影响共享参数
```

这意味着你的 uncertainty 即使学到了一些东西，也没有直接改变最终排序矩阵。对于检索任务来说，这是很致命的。

更合理的是：

```text
final_score = WTI_score
            + λ * probabilistic_score
            - β * uncertainty_penalty
```

或者：

```text
final_score = WTI_score adjusted by pair-level confidence
```

---

## 问题 2：当前 uncertainty 主要是 sample-level / video-level，不是 pair-level

你现在的视频侧 uncertainty 是 `epistemic_video`，本质上是每个视频一个不确定性标量。后续 `_evidential_similarity()` 用它对该视频和所有文本 query 统一折扣。

但视频文本检索需要的是：

```text
u(text_i, video_j)
```

不是：

```text
u(video_j)
```

同一个视频对不同 query 的不确定性应该不同。例如一个视频里既有 cooking 又有 dog running，那么它对 “a man cooking” 和 “a dog running” 都可能确定，但对 “a woman dancing” 不确定。当前 video-level uncertainty 表达不了这种关系。

这也是为什么我建议你改成：

```text
pair-level retrieval uncertainty
```

---

## 问题 3：视频侧 `logsigma` 太粗糙，且已经出现塌缩

SAP 当前把 anchor 间方差先平均成 `[B,1]`，再扩展成 `[B,D]`。

这意味着 512 维里每一维的不确定性完全一样。它不能表达：

```text
动作语义不确定
物体语义确定
场景语义确定
细节属性不确定
```

更严重的是，日志显示训练到 Epoch 3 后，`logsig_v` 基本卡在 `-1.5000`，也就是你设置的 `log_sigma_min` 下界；`epistemic_v` 也几乎固定在约 `0.2231`。

Epoch 3 summary 也显示：

```text
u_mode=0.4878±0.0030
epistemic_v=0.2232
var_t=0.2234
logsig_v=-1.4999
```

这说明当前不确定性分支没有学到很强的样本区分度，而是在被训练推向低方差、低噪声状态。换句话说：

```text
模型倾向于把 uncertainty 压小，
而不是让 uncertainty 反映检索困难度。
```

---

## 问题 4：MIL 采样目标天然容易鼓励 variance shrinkage

你现在的 Gaussian sampling 是：

```python
sample = mu + eps * exp(0.5 * logsigma)
```

对于 contrastive/MIL 来说，最容易降低 loss 的方式之一就是：

```text
让 logsigma 变小
→ sample 更接近 mean
→ 正样本更稳定
→ 对比学习更容易
```

所以如果没有额外的 calibration target，Gaussian variance 很容易塌缩。你日志里的 `logsig_v=-1.5000` 就是这个现象。

这不是你一个人的代码问题，而是概率检索里很常见的陷阱：**只用 sampling contrastive 不足以学出有意义的不确定性。**

---

## 问题 5：UACL 的 `closest` sample 策略进一步鼓励低方差

当前 UACL 会从 Gaussian samples 中选一个 augmentation。如果策略是 `closest`，它会选和 mean cosine similarity 最大的 sample。

这会产生一个隐含倾向：

```text
sample 越接近 mean，intra-modal alignment 越容易
variance 越小，sample 越接近 mean
```

所以 UACL + closest sample 很可能进一步鼓励 uncertainty 退化成小方差，而不是让模型表达真实不确定性。

如果继续用 UACL，我建议至少不要默认用 `closest`，可以改成：

```text
random sample
hard sample
uncertainty-preserving sample
```

或者只把 UACL 用于 mean 表征，不让它直接压 variance。

---

## 问题 6：`uncertainty_mode=none` 的语义和实现已修正

这一项在原始审计中确实会污染实验理解，但当前代码已经修正。现在 `modules/modeling_mulit.py` 通过 `resolve_loss_activations()` 控制辅助 loss：

```text
uncertainty_mode=none        关闭 evidential/neg_reg
uncertainty_mode=evidential  启用当前 evidential/neg_reg
nig_mil                      deprecated 兼容项
```

`train_msrvtt.sh` 当前 default profile 默认 `UNCERTAINTY_MODE=evidential`，用于保持历史训练行为；`EXPERIMENT_PROFILE=hygiene` 会强制 `uncertainty_mode=none` 并关闭 MIL/evidential/neg/orth/HN/UACL。因此这项已经不再是当前 bug，后续只需要在实验记录中明确 profile 和 active loss 状态。

---

## 问题 7：文本侧 uncertainty 的 padding mask 已接入

这一项在原始审计中是实现 bug，但当前代码已经修正。`_loose_similarity()` 现在调用 `probabilistic_text(text_pooled, text_token, attention_mask, sample_embeddings=...)`，内部会把 `attention_mask == 0` 转成 `pad_mask`。

当前 `PIENet`、`UncertaintyModuleText` 和 `UncertaintyModuleTextMamba` 统一采用 attention 习惯的 mask 语义：

```text
pad_mask=True  表示 padding
valid_mask     由 ~pad_mask 得到
```

因此，报告中“padding 会进入 GRU/attention 并污染 `logsigma`”的判断已经不适用于当前主模型路径。文本侧不确定性后续若仍异常，应结合 `logsigma_t_min_ratio/max_ratio`、attention 分布和错例校准来分析。

---

## 问题 8：evidential loss 名字像 evidential，但没有真正使用 Dirichlet evidence

`_evidential_nll_loss(sim_matrix, alpha_dir)` 虽然传入了 `alpha_dir`，但函数体没有使用它，只用 `sim_matrix` 做正对 NLL 和负对惩罚。

你确实实现了一个更接近 Dirichlet MSE 的 `evidential_matrix_loss()`，里面有：

```text
evidence = relu(sim_matrix)
alpha = evidence + 1
prob = alpha / strength
Dirichlet variance term
```

但它没有进入主 loss。当前主 loss 里返回的是：

```text
MIL_loss
evidential_loss
neg_reg_loss
orth_loss
hard_negative_loss
uacl_intra_loss
uacl_kl_loss
```

没有 `uncertainty_reg_loss`，也没有使用 `evidential_matrix_loss()`。

所以你现在的 evidential 分支理论闭环不完整。

---

## 问题 9：`w_uncertainty_reg` 基本是 dead knob

模型里定义了：

```python
self.w_uncertainty_reg = ...
```

参数说明也说它是 DUQ-style evidential uncertainty regularization。

但在主 loss 里没有对应项。

这意味着你调 `w_uncertainty_reg` 大概率不会改变训练目标。这也会让超参搜索结果变得不可信。

---

# 4. 总体判断：当前 uncertainty 模块为什么难提升？

我认为核心原因是这四个：

```text
1. uncertainty 没有直接参与 final ranking；
2. uncertainty 是 sample/video-level，不是 query-video pair-level；
3. sampling contrastive 会把 variance 压小，日志已经显示塌缩；
4. evidential / uncertainty loss 和 Dirichlet evidence 没有形成闭环。
```

所以当前结构更像：

```text
WTI 主检索模型
+
一堆概率/不确定性辅助正则
```

而不是：

```text
不确定性感知的视频文本检索模型
```

这就是为什么它很难稳定超过 baseline。

---

# 5. 更好的设计方向一：Pair-level Retrieval Uncertainty

这是我最推荐你做的。

把不确定性从：

```text
u_text_i
u_video_j
```

改成：

```text
u_ij = uncertainty(text_i, video_j)
```

输入可以用：

```text
WTI score s_wti_ij
SAP anchor score s_anchor_ij
text mean μ_t
video mean μ_v
text variance σ_t
video variance σ_v
|μ_t - μ_v|
μ_t ⊙ μ_v
top-k margin / entropy
```

输出：

```text
evidence_ij
uncertainty_ij
confidence_ij
```

例如：

```python
pair_feat_ij = [
    s_wti_ij,
    s_prob_ij,
    abs(mu_t_i - mu_v_j),
    mu_t_i * mu_v_j,
    var_t_i,
    var_v_j
]

evidence_ij = softplus(MLP(pair_feat_ij))
uncertainty_ij = 1.0 / (evidence_ij + 1.0)
```

最终检索分数：

```python
score_ij = s_wti_ij + λ * s_prob_ij - β * uncertainty_ij
```

这个设计比当前 `video confidence × cosine` 更合理，因为它能表达：

```text
同一个视频对不同文本 query 有不同不确定性。
```

---

# 6. 更好的设计方向二：Closed-form Gaussian Score，少用 Monte Carlo

当前 MIL 是采样式的，容易让 variance 塌缩。建议先改成 closed-form score：

```text
p_t = N(μ_t, σ_t²)
p_v = N(μ_v, σ_v²)
```

简单版：

```python
s_prob_ij = cos(mu_t_i, mu_v_j)
unc_ij = mean(exp(logvar_t_i)) + mean(exp(logvar_v_j))
score_ij = s_wti_ij + λ * s_prob_ij - β * unc_ij
```

更概率一点：

```python
dist_ij = ||mu_t_i - mu_v_j||² / (var_t_i + var_v_j)
logdet_ij = log(var_t_i + var_v_j).sum()
score_ij = -0.5 * dist_ij - 0.5 * logdet_ij
```

这样 uncertainty 直接影响分数，而不是只通过 sampling loss 间接影响。

短期我建议你先做简单版：

```text
WTI + μ cosine - β variance
```

这比继续堆 Monte Carlo MIL 更容易定位问题。

---

# 7. 更好的设计方向三：真正的 Evidential Ranking Head

你现在的 evidential loss 不够 evidential。建议重做成 batch-level Dirichlet ranking。

对于每个 query `i`，对所有 candidate videos `j` 生成 evidence：

```python
evidence_ij = softplus(score_ij)
alpha_ij = evidence_ij + 1
prob_ij = alpha_ij / alpha_i.sum()
uncertainty_i = K / alpha_i.sum()
```

训练目标：

```text
Dirichlet MSE / NLL:
target = one-hot 或 soft-label
```

核心是：

```text
正确 pair → 高 evidence
明显负样本 → 低 evidence
模糊/false negative → 不强行低 evidence，而是高 uncertainty 或 soft positive
```

你已经有 `evidential_matrix_loss()` 的雏形，但要把它接入主 loss，并且让它作用于最终 score，而不是一个旁路 `ev_sim`。

推荐：

```python
score = wti_logits + λ * prob_logits
evid_loss, batch_unc = evidential_matrix_loss(score)
loss += w_evid * evid_loss
```

而不是：

```python
ev_sim = cosine(mu_video, text_pooled) * video_confidence
```

---

# 8. 更好的设计方向四：Uncertainty-aware Soft False Negative Learning

你之前 hard negative 不稳定，我认为很正常。MSR-VTT 里很多负样本是语义近邻，硬推远会伤。

更好的做法是用 uncertainty 产生 soft labels：

```text
Y_ii = 1

如果 i != j 但：
    s_wti_ij 高
    s_anchor_ij 高
    uncertainty_ij 中等或偏低
    text/video 语义相近
则：
    Y_ij = q_ij ∈ (0, 1)

否则：
    Y_ij = 0
```

然后用 sigmoid contrastive：

```python
loss = BCEWithLogitsLoss(score_ij / τ, Y_ij)
```

这比 hard negative 更适合你当前方向，也更适合和 SigLIP/SigLIP2 backbone 结合。

---

# 9. 更好的设计方向五：Calibration-aware uncertainty

如果你想让“不确定性”真的有论文价值，建议加入 calibration 评价和训练目标。

你可以定义：

```text
confidence_i = 1 - uncertainty_i
correct_i = 1 if top1 video is GT else 0
```

加一个轻量 calibration loss：

```python
L_calib = BCE(confidence_i, correct_i.detach())
```

或 ranking-level calibration：

```text
如果 GT rank 高 → uncertainty 应低
如果 GT rank 低 → uncertainty 应高
```

训练中无法直接用验证 GT rank，但可以用 batch 内正负 margin 近似：

```python
margin_i = score_ii - max_j!=i score_ij
target_unc_i = sigmoid(-margin_i.detach())
```

然后：

```python
L_unc = MSE(uncertainty_i, target_unc_i)
```

这比单纯让 variance 参与 sampling 更能学出“预测错时更不确定”的信号。

---

# 10. 我建议你把当前不确定性重构成三层

## 第一层：Representation uncertainty

保留 Gaussian，但改成 closed-form，不要先主打 Monte Carlo。

```text
text/video embedding → μ, logvar
```

用途：

```text
补充分数
控制噪声
表示样本质量
```

---

## 第二层：Pair-level correspondence uncertainty

新增：

```text
u_ij = uncertainty(text_i, video_j)
```

用途：

```text
调整 final score
识别 false negatives
动态调节负样本权重
```

---

## 第三层：Decision uncertainty

针对每个 query 的 top-k 排序：

```text
ranking entropy
top1-top2 margin
Dirichlet strength
```

用途：

```text
calibration
selective retrieval
可信检索分析
```

这三层会比现在的：

```text
text logsigma + video logsigma + evidential regularizer
```

更清楚，也更适合作为论文方法。

---

# 11. 最小可行改法：别一次性大改

我建议你按下面 6 个实验改。

## E1：修 bug / 清理语义

当前基础卫生项已经基本完成：

```text
1. uncertainty_mode=none 的反逻辑已修正；
2. probabilistic_text 已接 attention mask；
3. 日志已增加 logsigma clamp ratio；
4. w_uncertainty_reg 仍未接入 loss，当前应继续视为 dead knob，后续要么删除/降级为兼容参数，要么接入真正的 final-logits evidential regularizer。
```

因此 E1 当前不再是“从零修 bug”，而是确认 hygiene profile 跑通，并在实验汇总中记录 active loss 与 score source。

---

## E2：验证概率均值有没有检索价值

新增：

```python
prob_logits = torch.matmul(prob_text["mean"], mu_video.t()) * logit_scale
final_logits = wti_logits + λ * prob_logits
```

训练和评估都用 `final_logits`。

如果 E2 不涨，说明当前概率分支的 `μ` 本身没有检索价值，不要继续做 uncertainty。

---

## E3：验证 variance 是否有价值

```python
var_t = prob_text_logsigma.exp().mean(dim=-1)
var_v = logsigma_video.exp().mean(dim=-1)
unc_ij = var_t[:, None] + var_v[None, :]
final_logits = wti_logits + λ * prob_logits - β * unc_ij
```

如果 E3 比 E2 差，说明当前 variance 学得不好。

---

## E4：把 video-level uncertainty 改成 pair-level

```python
pair_unc_ij = MLP([wti_ij, prob_ij, |mu_t_i - mu_v_j|, mu_t_i * mu_v_j])
final_logits = wti_logits + λ * prob_logits - β * pair_unc_ij
```

这是最关键版本。

---

## E5：接入 evidential_matrix_loss 到 final_logits

```python
evid_loss, unc = evidential_matrix_loss(final_logits)
loss += w_evid * evid_loss
```

不要再用旁路 `ev_sim`。

---

## E6：soft false negative

把 hard one-hot CE 改成 soft-label sigmoid contrastive。这个可以作为最终创新点。

---

# 12. 代码层面的优先修复建议

我会按这个顺序改：

```text
P0:
确认 hygiene profile 与 active loss 日志；
继续把 w_uncertainty_reg 标记为 dead knob，或删除/接入真实 loss；
用 logsigma clamp ratio 判断方差是否塌缩。

P1:
final_logits = wti_logits + λ * prob_logits；
训练/评估统一使用 final_logits。

P2:
closed-form uncertainty penalty；
替代或弱化 MIL sampling。

P3:
pair-level uncertainty head；
让 uncertainty 从 u_video 变成 u_text_video。

P4:
evidential_matrix_loss 接 final_logits；
做 calibration 评估。

P5:
soft false-negative sigmoid contrastive。
```

---

# 13. 最推荐的新版设计

如果你要继续写论文，我建议方法主线改成：

```text
Pair-level Evidential Uncertainty for Text-Video Retrieval
```

具体结构：

```text
WTI branch:
word-frame late interaction → s_wti

Probabilistic branch:
text/video/SAP → μ_t, μ_v, logvar_t, logvar_v → s_prob

Pair uncertainty branch:
[s_wti, s_prob, μ_t⊙μ_v, |μ_t-μ_v|, var_t, var_v]
→ evidence_ij
→ uncertainty_ij

Final score:
s_ij = s_wti + λ s_prob - β u_ij

Training:
CE or sigmoid contrastive
+ evidential ranking calibration
+ soft false-negative learning
```

这个设计有三个优点：

```text
1. uncertainty 直接影响最终排序；
2. uncertainty 是 pair-level，符合检索任务；
3. 可以自然处理 MSR-VTT 的语义近邻/false negative 问题。
```

---

## 最终判断

你当前不确定性模块**不是方向错了**，而是：

```text
不确定性定义太分散；
监督目标不闭环；
final ranking 没用它；
variance 被 sampling contrastive 压到下界；
evidential 分支没有真正使用 Dirichlet evidence。
```

所以我不建议继续在当前设计上调 `w_mil / w_evidential / w_neg_reg / w_orth`。更值得做的是把不确定性重构成：

```text
pair-level uncertainty
+
final score adjustment
+
soft false-negative learning
```

这会比当前“Gaussian sampling + auxiliary loss”的路径更有希望稳定提升视频文本检索效果。
