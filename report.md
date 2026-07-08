我看完了你仓库 `ChihayaAnonnn/tvr` 的主路径。先说结论：**我认为你现在效果一直上不去，主要不是“再调几个权重”的问题，而是当前不确定性模块和最终检索排序之间存在结构性脱节。** 方向本身是对的，但现在的实现更像是“在 WTI/CLIP 检索主干旁边挂了一个概率分支”，它没有稳定地进入最终 ranking 决策，所以很难带来 R@1 提升。

我没有实际跑你的训练，只做了**静态代码审计 + 已有日志分析**。下面是我认为最关键的问题。

> 2026-07-05 执行状态短注：backbone 替换前的实验卫生清理已按“先保留 WTI final ranking、不引入概率 score 融合”的原则接入。`EXPERIMENT_PROFILE=hygiene` 提供 WTI-only 干净基线；`uncertainty_mode=none/evidential/nig_mil` 语义已修正；文本概率分支已接入 mask；`logsigma` clamp ratio 已进入日志。本文仍保留原始审计判断，但“待修”项需按本短注理解。

---

## 1. 最大问题：最终检索分数基本还是 `wti_logits`

你的当前主模型在 `modules/modeling_mulit.py` 里确实构建了 SAP、文本概率头、MIL、evidential、orth、hard negative、UACL 等模块；训练时总 loss 也是 `sim_loss + MIL + evidential + neg_reg + orth + hard_negative + uacl`。

但是最关键的一行是：

```python
weighted_logits = wti_logits
```

也就是当前训练主 CE loss 用的是 WTI 分数，而不是概率匹配分数、evidential 分数或 SAP 分数。代码注释也写了“B1-only：置信度加权已移除，WTI logits 直接作为检索分数”。

评估时也是一样。`_loose_similarity()` 在 eval 分支最后直接返回 `wti_logits`。 而 `eval_epoch()` 把 `get_similarity_logits()` 返回的 `chunk_logits` 拼成最终 `sim_matrix`，再用这个矩阵计算 T2V/V2T 指标。

所以现在真实链路是：

```text
训练主排序：
text/video tokens → WTI logits → CrossEntropy

评估排序：
text/video tokens → WTI logits → Recall@K

SAP / Gaussian / Evidential：
主要作为辅助 loss 间接影响参数
```

这会导致一个很现实的问题：**你的不确定性分支即使学到了东西，也不一定能改变最终排序。** 它只能通过辅助 loss 间接拉动 CLIP token、SAP、PIENet、uncertainty head，但最终排名还是由 `wti_logits` 决定。

这也是为什么你会看到“loss 在变，概率统计在变，但 R@1 不涨”。

---

## 2. SAP 不确定性现在不是一个真正的 pair-level retrieval uncertainty

你的 SAP 模块设计目标很清楚：用可学习 anchor 从时空 token 中探测语义，再输出 `mu_raw / logsigma / alpha_dir / modal_probs / u_mode / epistemic_cont`。

但当前实现里有几个关键问题。

第一，Dirichlet evidence head 的输入被 `detach()` 了：

```python
ev = self.evidential_head(projected.detach())
```

这意味着 Dirichlet evidence / modal_probs 的学习不会反向推动 SAP decoder 和 anchor projection 去产生更适合 evidence 建模的特征。

第二，`logsigma` 不是 learnable uncertainty head 输出，而是直接从 anchor 间方差算出来：

```python
anchor_dim_var = torch.var(anchors, dim=1).mean(dim=-1, keepdim=True)
logsigma = torch.log(anchor_dim_var + 1e-8)
logsigma = logsigma.expand(-1, self.d_model)
```

也就是说视频侧每个维度的方差其实是同一个标量复制出来的。 这会让所谓的 Gaussian uncertainty 很粗糙，基本表达不了“哪些语义维度不确定、哪些 token 不确定、哪个 query-video pair 不确定”。

第三，`epistemic_cont` 是由 `diversity * modal_entropy` 直接算出来的非学习统计量。 这个东西更像视频复杂度指标，不是检索意义上的不确定性。它没有回答：

```text
query_i 和 video_j 这个 pair 是否可靠？
当前 Top-1 是不是可信？
这个负样本是不是 false negative？
query 对视频中哪些 clip 不确定？
```

所以它很难直接提高视频文本检索的排序效果。

---

## 3. 你的日志已经显示：不确定性在训练中塌缩/饱和了

这点非常关键。你的日志里，到了 Epoch 3，`logsigma_v` 基本卡在 `-1.5000`，也就是你设置的 `log_sigma_min` 下界；`epistemic_v` 也稳定在大约 `0.2231`。

同一段日志里，Epoch 3 summary 也显示：

```text
Evid : u_mode=0.4878±0.0030  epistemic_v=0.2232
Text : var_t=0.2234
Aux  : logsig_v=-1.4999
```

这说明当前概率分支学到的不是“有区分力的不确定性”，而是倾向于把方差压到下界。换句话说：

```text
模型不是在学习 uncertainty
而是在学习让 sampling 噪声尽可能小
```

这很常见。因为 Gaussian sampling + contrastive loss 往往会鼓励模型减小 variance，让采样 embedding 更接近 mean，这样 MIL loss 更容易下降。你现在的 UACL `closest` sample 策略也会加剧这个趋势，因为它选离 mean 最近的 sample 当正样本，本质上更鼓励小方差。

所以目前“不确定性模块没提升”的一个很直接解释是：**它已经退化成了接近确定性的辅助分支。**

2026-07-05 起，后续不再只看 `logsigma_v` 的均值判断塌缩，而是以日志中的 `logsigma_v_min_ratio/max_ratio` 与 `logsigma_t_min_ratio/max_ratio` 为准。如果 min ratio 长期接近 1，说明方差被下界夹住，概率辅助路径仍然缺少有效不确定性表达。

---

## 4. `uncertainty_mode=none` 的语义和代码实现已修正

这一项在原始审计中是一个很容易坑实验的实现问题，但当前代码已经修正。现在 `modules/modeling_mulit.py` 通过 `resolve_loss_activations()` 统一决定各辅助 loss 是否启用：

```text
none        = 真实关闭 evidential / neg_reg
evidential = 启用当前 Dirichlet/evidential regularizer
nig_mil    = deprecated 兼容选项，不作为新实验建议
```

当前 `train_msrvtt.sh` 的 default profile 默认使用 `uncertainty_mode=evidential` 来保持历史训练行为；`EXPERIMENT_PROFILE=hygiene` 会强制改为 `uncertainty_mode=none` 并关闭 MIL/evidential/neg/orth/HN/UACL。后续实验解释应以当前语义为准，不再把 `none` 解读为会意外启用 evidential/neg_reg。

---

## 5. evidential loss 的设计目前也不太像真正的 evidential retrieval

你定义了 `_evidential_similarity()`，用视频侧 uncertainty 对 cosine similarity 做折扣：

```python
cosine_sim = torch.mm(mu_video, text_pooled.t())
confidence = torch.exp(-epistemic_penalty)
sim_matrix = cosine_sim * confidence.unsqueeze(1)
```

这里有两个问题。

第一，它是**video-level scalar confidence**，不是 pair-level confidence。也就是说同一个视频对所有文本 query 用同一个不确定性折扣。这不太符合视频文本检索，因为一个视频对 query A 可能很确定，对 query B 可能很不确定。

第二，`_evidential_nll_loss(sim_matrix, alpha_dir)` 的参数里虽然传了 `alpha_dir`，但函数体其实没有使用 `alpha_dir`。它只是对 `sim_matrix` 的 diagonal 做 `-log(relu(score))`，再对 off-diagonal 的正分数做惩罚。

所以这个 loss 名字叫 evidential，但它并没有真正把 Dirichlet evidence、strength、uncertainty、target distribution 结合起来。你后面虽然实现了 `evidential_matrix_loss()`，里面有 Dirichlet MSE 和 uncertainty 计算，但当前主 loss 返回项里没有用它。

这会导致 evidential 分支变成一个额外的 similarity regularizer，而不是一个能稳定提升排序的证据学习模块。

---

## 6. `w_uncertainty_reg` 是一个“看起来在调，实际没进 loss”的参数

模型初始化里有：

```python
self.w_uncertainty_reg = getattr(self.task_config, "w_uncertainty_reg", 1e-3)
```

训练脚本和搜索脚本也都在调它。`hyperparam_search.py` 把 `w_uncertainty_reg` 放进了搜索空间。

但 `_loose_similarity()` 返回的 loss 字典里只有：

```python
MIL_loss
evidential_loss
neg_reg_loss
orth_loss
hard_negative_loss
uacl_intra_loss
uacl_kl_loss
```

没有 `uncertainty_reg_loss`。

所以这个参数当前基本是 dead knob。调它不会改变训练目标。这个会污染你的超参搜索结论。

---

## 7. 文本概率分支的 attention mask 已接入

这一项在原始审计中是实现 bug，但当前代码已经修正。`_loose_similarity()` 现在调用：

```python
prob_text = self.probabilistic_text(
    text_pooled,
    text_token[:, 0:word_num, :].contiguous(),
    attention_mask[:, 0:word_num].contiguous(),
    sample_embeddings=needs_text_samples,
)
```

`probabilistic_text()` 内部会把 `attention_mask == 0` 转成 `pad_mask`，并传给 `PIENet` 与 `UncertaintyModuleText/TextMamba`。当前 mask 语义统一为 `True = padding`，GRU/Mamba 侧再用 `valid_mask = ~pad_mask` 计算有效长度。

因此，报告中“padding 会进入 PIENet/GRU uncertainty”的判断不再适用于当前代码。后续若文本侧不确定性仍不稳定，应优先查看 `logsigma_t_min_ratio/max_ratio`、文本方差分布和 loss 设计，而不是把问题归因于 padding mask 未接入。

---

## 8. 当前模块太多，但真正有效路径太窄

日志里显示，有大量新增模块随机初始化，包括 `ada_norm_text`、`pie_net_text`、`sap`、`spatial_enhancer`、`text_weight_fc`、`uncertain_net_text`、`video_weight_fc`、`word_position_embeddings` 等。

同时 optimizer 里 CLIP 的学习率是 `1e-7`，新模块是 `1e-4`。

这意味着你的主干 CLIP 基本很小幅地动，而大量新模块在学。但最终评估分数又主要是 WTI logits。这个组合容易出现：

```text
新模块努力优化辅助 loss
↓
主排序分数没有真正使用新模块
↓
辅助分支扰动 token 表征
↓
验证集 R@1 不稳定甚至下降
```

你的 `search_results/best.json` 也印证了这一点：一次超参搜索最好也只有 T2V R@1 = 47.7。

---

# 我的判断：问题主要在设计，不是单纯调参

你现在的模块不是“完全错”，但它有三个核心设计短板：

```text
1. 不确定性没有直接进入最终 ranking score；
2. 当前 uncertainty 更像 video-level/sample-level 统计量，不是 query-video pair-level uncertainty；
3. Gaussian variance 在 contrastive/MIL 训练下塌缩到下界，失去区分力。
```

所以继续在当前结构上 sweep `w_evidential / w_neg_reg / w_orth / log_sigma_min`，大概率收益有限。

---

# 我建议你先做的修复优先级

## 优先级 1：先修正“训练目标和评估分数不一致”

现在最应该做的是把概率分支真正纳入最终检索分数，而不是只作为辅助 loss。

可以先做一个非常简单的版本：

```python
s_wti = wti_logits
s_prob = torch.matmul(prob_text["mean"], mu_video.t()) * logit_scale

score = s_wti + lambda_prob * s_prob
```

训练和评估都用这个 `score`。

更合理一点，可以加 uncertainty penalty：

```python
var_t = prob_text_logsigma.exp().mean(dim=-1)      # [B]
var_v = logsigma_video.exp().mean(dim=-1)          # [B]

unc_penalty = var_t[:, None] + var_v[None, :]
score = s_wti + lambda_prob * s_prob - beta * unc_penalty
```

这样不确定性才会影响 ranking。

你现在最不建议继续做的是：

```text
最终 score 仍然 = wti_logits
但继续堆 MIL / Evidential / UACL 辅助项
```

这条路很容易继续卡住。

---

## 优先级 2：`uncertainty_mode=none` 反逻辑已完成修复

当前实现已经通过 `loss activation resolver` 统一判断 MIL/evidential/neg/orth/HN/UACL 是否实际进 loss，并在日志输出 `[Chain-Hygiene]` 简报。后续不需要再围绕这一 bug 做修复，只需在实验记录里明确：

```text
uncertainty_mode=none        关闭 evidential/neg_reg
uncertainty_mode=evidential  启用当前 evidential/neg_reg
nig_mil                      deprecated 兼容项
```

---

## 优先级 3：先不要让 Gaussian variance 自由塌缩

你需要在日志里额外统计：

```text
ratio(logsigma_v == log_sigma_min)
ratio(logsigma_v == log_sigma_max)
ratio(logsigma_t == log_sigma_min)
ratio(logsigma_t == log_sigma_max)
```

如果大部分都卡在 `log_sigma_min=-1.5`，说明 uncertainty branch 没有学习价值。

2026-07-05 已接入上述 clamp ratio；后续判断 `logsigma` 塌缩时优先看 ratio，而不是只看均值。

短期修法：

```text
1. 先关掉 UACL intra alignment；
2. 降低或关闭 MIL sampling loss；
3. 不用 closest sample；
4. 把 Gaussian score 改为 closed-form distance，而不是依赖采样；
5. 用 uncertainty calibration loss 约束“不确定样本应该更容易错”，而不是只让采样 embedding 做对比。
```

例如先用 closed-form：

```python
mu_sim = torch.matmul(mu_t, mu_v.t())
unc = logvar_t.exp().mean(-1)[:, None] + logvar_v.exp().mean(-1)[None, :]
score_prob = mu_sim - beta * unc
```

这个比 Monte Carlo sampling 更稳定。

---

## 优先级 4：文本 probability head 的 mask 已完成接入

当前 `probabilistic_text(text_pooled, text_token, attention_mask)` 会把 `attention_mask == 0` 转为 `pad_mask`；`PIENet` 与 `UncertaintyModuleText/TextMamba` 统一采用 `True = padding` 的 mask 语义，内部再由 `valid_mask = ~pad_mask` 计算有效长度。

这一项已经不是待办。若后续还要做文本侧诊断，应重点看 mask 后的 attention 分布、`logsigma_t` clamp ratio 和错例上的文本不确定性，而不是继续重复“接 mask”修复。

---

## 优先级 5：移除 SAP 里的 `detach()`，或至少做消融

当前：

```python
ev = self.evidential_head(projected.detach())
```

我建议做三个版本：

```text
A. detach 保持现状
B. 去掉 detach
C. stop-gradient 只用于 uncertainty，不用于 modal_probs 聚合
```

如果去掉 detach 后不稳定，可以加：

```python
projected_for_ev = projected.detach() + 0.1 * (projected - projected.detach())
```

也就是只让一部分梯度回传。这样 Dirichlet gate 至少能告诉 SAP decoder：哪些 anchor 应该变得更有证据、更可区分。

---

# 更适合你论文方向的新设计

你现在不该继续主攻“video-level uncertainty scalar”。我建议改成：

```text
Pair-level Uncertainty-aware Text-Video Retrieval
```

也就是不确定性不再只属于 video 或 text，而属于：

```text
(query_i, video_j)
```

可以这样做：

```python
pair_feat = concat([
    text_emb_i,
    video_emb_j,
    text_emb_i * video_emb_j,
    abs(text_emb_i - video_emb_j),
    wti_score_ij
])

evidence_ij = softplus(MLP(pair_feat))
uncertainty_ij = 1 / (evidence_ij + 1)
```

然后用于两个地方：

### 1. 调整 ranking score

```python
score_ij = wti_ij + lambda_prob * prob_ij - beta * uncertainty_ij
```

### 2. 调整 contrastive loss

高置信负样本可以推远；低置信负样本可能是 false negative，应该少推甚至作为 soft positive：

```python
target_ij = 1                      if i == j
target_ij = q_ij in (0, 1)          if uncertain false negative
target_ij = 0                      otherwise
```

然后用 sigmoid / soft-label contrastive：

```python
loss = BCEWithLogitsLoss(score / tau, soft_target)
```

这和你后面想换 SigLIP/SigLIP2 backbone 也更契合。

---

# 我建议你下一轮实验这样排

不要继续大规模 sweep。先做一组非常干净的定位实验。

| 实验 | 改动                                   | 目的                   |
| -- | ------------------------------------ | -------------------- |
| E0 | 当前 `wti_logits` 原样                   | 复现当前结果               |
| E1 | 关闭 MIL/evidential/neg/orth，只保留 WTI   | 确认干净 baseline        |
| E2 | 只加 `s_prob = mu_t @ mu_v.T`，训练/评估都融合 | 验证概率分支是否有排序价值        |
| E3 | `s_prob - beta * uncertainty`        | 验证 uncertainty 是否真有用 |
| E4 | 修 mask 后重跑 E2/E3                     | 验证文本 padding 是否污染    |
| E5 | 去掉 SAP detach 后重跑 E3                 | 验证 SAP gate 是否需要端到端  |
| E6 | soft false-negative loss             | 验证 MSR-VTT 多正例歧义处理   |

如果 E2 都不涨，说明当前 SAP/prob embedding 本身没有形成有用检索空间，不要继续做 evidential。
如果 E2 涨、E3 不涨，说明概率均值有用，但 uncertainty 没学好。
如果 E3 涨，才值得继续写 uncertainty 论文主线。

---

# Backbone 前 hygiene 清理执行状态

2026-07-05 已完成：

```text
1. 修 uncertainty_mode=none 逻辑，并新增 uncertainty_mode=evidential；
2. 新增 EXPERIMENT_PROFILE=hygiene，形成 WTI-only 干净基线；
3. 给 probabilistic_text 接 attention mask；
4. 统计 logsigma clamp ratio，确认 uncertainty 是否塌缩；
5. 显式记录 loss activation 与 final score source = wti_logits。
```

本轮刻意暂缓：

```text
1. 不把 prob score 融入 train/eval final logits；
2. 不修改 backbone；
3. 不继续 HN/UACL sweep；
4. 不启动长期训练。
```

下一步先跑 hygiene baseline，再用同一实验卫生条件评估 SigLIP/EVA/InternVideo2 等 backbone 替换是否带来真实收益。

---

## 最后判断

你现在的问题不是“UATVR 方向不行”，而是当前实现中：

```text
不确定性模块 ≠ 最终排序模块
```

这导致它更像一个辅助正则分支。辅助分支如果设计得非常精细，可能有小幅收益；但你现在又叠了 SAP、Gaussian sampling、evidential、orth、UACL、RALA/SpatialEnhancer、hard negative 等很多随机初始化模块，收益信号被稀释了。

所以我的建议很明确：**先把不确定性从 auxiliary loss 拉回 retrieval score 本身。** 你的创新点应该从“我加了不确定性模块”改成：

```text
我用 pair-level uncertainty 直接控制 text-video matching score 和 false-negative contrastive learning。
```

这会比继续在当前 SAP 辅助分支上调权重更有希望。
