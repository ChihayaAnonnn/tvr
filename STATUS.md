# UATVR 项目现状

> 更新时间：2026-06-08

## 一、项目概述

UATVR（Uncertainty-Aware Text-Video Retrieval）基于 CLIP ViT-B/16，核心模块为 SAP（Semantic Anchor Probing）：K 个可学习 anchor 通过 TransformerDecoder 探测视频时空特征，输出门控聚合表征 + 双层不确定性估计。

- 数据集：MSRVTT（train 9k + test 1k）
- 评估指标：T2V R@1 / V2T R@1

## 二、当前最优结果

| 实验 | uncertainty_mode | w_evidential | w_neg_reg | Best T2V R@1 | Best Epoch |
|------|-----------------|-------------|-----------|-------------|-----------|
| **baseline_pure_sim** | none | **0** | **0** | **49.4** | 4 |
| NIG-MIL | nig_mil | 0（跳过） | 0（跳过） | 49.0 | 4 |
| baseline_no_evid | none | 0.01 | 0.05 | 48.1 | 3-4 |

最优 checkpoint：`ckpts/ckpt_msrvtt_20260607_102327/pytorch_model.bin.3`

## 三、不确定性机制分析结论

### 3.1 结论：不确定性机制零贡献

三组对照实验证明，SAP 的 EvidentialUncertaintyHead（NIG + Dirichlet）对检索精度无正向贡献。49.4 R@1 完全来自 sim_loss（WTI 对比学习），不确定性分支在训练中退化为常量。

### 3.2 根因：结构性失效（非超参问题）

**根因 1：MIL loss 天然惩罚方差**

MILNCELoss_BoF 是对比学习 loss。方差越大 → 采样越分散 → 正负样本混淆越严重 → loss 越高。梯度方向永远是减小方差，没有任何对冲力量。`logsig_v` 在 200 步内撞到下界 -1.5，之后 93% 训练全程钉死。

**根因 2：sim_loss 完全绕过不确定性路径**

```
sim_loss: text_token + visual_output → WTI → CrossEn
                                    ↑
                          不经过 SAP evidential_head
```

主检索 loss 的梯度只更新 CLIP encoder + WTI 层，不确定性参数（NIG layer）零梯度。唯一训练信号 MIL_loss 在 Epoch 1 后饱和在 0.16（退化为 vanilla InfoNCE）。

**根因 3：anchors 被迫同时服务两个冲突目标**

SAP 的 decoder 输出 `anchors` 同时被两个 head 消费：
- `dirichlet_layer`：驱动语义聚合（modal_probs）
- `nig_layer`：驱动不确定性估计（v, alpha, beta）

两个梯度流共享 decoder 作为瓶颈，互相干扰。evidential_loss（~5.9）的梯度量级远大于 sim_loss（~0.2-0.8），严重干扰语义表征学习。

### 3.3 关键数据

| 指标 | baseline_no_evid | NIG-MIL / baseline_pure_sim |
|------|------------------|---------------------------|
| logsig_v 坍缩 | step 20 撞 -1.5 | step 200 撞 -1.5 |
| MIL_loss 饱和 | 0.16（Epoch 1 后） | 0.16（Epoch 1 后） |
| evidential_loss | 5.9（有害梯度） | 0（跳过或 w=0） |
| neg_reg_loss | 0.18 | 0 |
| uncertainty_reg_loss | 1.0（detached，无梯度） | 同左 |
| sim_loss 驱动 R@1 提升 | ✅ 唯一有效项 | ✅ 唯一有效项 |

## 四、当前活跃 Loss 项

| Loss | 权重 | 状态 | 说明 |
|------|------|------|------|
| sim_loss (WTI CrossEn) | 1.0 | ✅ 唯一驱动 | 主检索 loss，不经过不确定性路径 |
| MIL_loss | 0.01 | ⚠️ 饱和 | Epoch 1 后退化为 vanilla InfoNCE |
| orth_loss | 0.1 | ✅ 有效 | Epoch 1 自行消解，锚点正交性 |
| uncertainty_reg | 0.001 | ❌ 无梯度 | detached 输入，纯监控 |
| evidential_loss | 0（已关闭） | ❌ 有害 | 干扰语义表征学习 |
| neg_reg_loss | 0（已关闭） | ❌ 有害 | 同上 |

## 五、当前超参（最优配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| `w_evidential` | 0 | 已关闭 |
| `w_neg_reg` | 0 | 已关闭 |
| `w_mil` | 0.01 | 保留但实际不贡献 |
| `w_orth` | 0.1 | 有效 |
| `w_uncertainty_reg` | 0.001 | 无梯度，可考虑移除 |
| `log_sigma_min` | -1.5 | logsig_v 下界 |
| `anneal_warmup_epochs` | 0 | 退火无效（evidential 已关闭） |
| `uncertainty_mode` | none | 使用默认模式 |
| `epochs` | 5 | Epoch 5 过拟合，最佳 Epoch 4 |

## 六、待执行方向

### 6.1 清理（低成本）

- [ ] 移除 NIG-MIL 相关死代码（`uncertainty_mode=nig_mil` 分支、`EvidentialUncertaintyHead`）
- [ ] 移除或标记 `uncertainty_reg_loss`（detached，无梯度）
- [ ] 训练减到 4 epochs（避免 Epoch 5 过拟合）
- [ ] 清理 argparse 中 `--w_evidential`、`--w_neg_reg` 参数（或保留但默认 0）

### 6.2 架构改进（中成本）

- [ ] 考虑移除 SAP 的 evidential_head，用简单 sigmoid 门控替代（降低参数量，消除冲突）
- [ ] 保留 SAP 的 decoder + anchor_tokens（已证明有效的语义聚合结构）
- [ ] 保留 Dirichlet modal_probs 聚合（如果它对 mu_raw 质量有贡献）

### 6.3 新方向（见 plan.md）

- [ ] E：不确定性课程学习 — 当前 evidential 已关闭，此方向需重新定义
- [ ] F：Top-K 锚点过滤 — 仍然有效，可在简化后的 SAP 上实验
- [ ] A：多粒度视频特征 — 与不确定性机制正交，可独立推进

## 七、关键文件

| 文件 | 说明 |
|------|------|
| `modules/modeling_mulit.py` | 主模型，~1018 行 |
| `query_models/module_sap.py` | SAP + EvidentialUncertaintyHead |
| `prob_models/tensor_utils.py` | `sample_gaussian_tensors` |
| `prob_models/uncertainty_module.py` | 文本侧不确定性头（仍活跃） |
| `main_task_retrieval.py` | 训练/评估入口 |
| `train_msrvtt.sh` | MSRVTT 训练脚本 |
| `hyperparam_search.py` | 超参搜索脚本 |

## 八、已知问题

1. **不确定性机制结构性失效**：见第三节分析，非超参问题
2. **Epoch 5 过拟合**：所有实验在 Epoch 4 达峰，R@1 下降 0.8
3. **MIL_loss 饱和**：退化为 vanilla InfoNCE，无额外贡献
4. **modeling_mulit.py 文件名拼写错误**：历史债务，非 bug

## 九、不确定性层优化方案

> 基于 2026-06-08 完整代码审查，见 `query_models/module_sap.py`、`modules/modeling_mulit.py`、`prob_models/uncertainty_module.py`、`modules/until_module.py`。

### 9.1 当前数据流（诊断用）

```
                    ┌─→ dirichlet_layer → alpha_dir → modal_probs ───┐
anchors [B,16,512] ─┤                                                ├─→ mu_raw [B,512] (检索用)
                    └─→ nig_layer → γ,ν,α,β → epistemic_cont         │
                                         → per_anchor_logsigma ──────┘

主检索路径: text_token + frame_token → WTI → CrossEntropy (sim_loss)
           ↑ 完全不经过 SAP 不确定性输出

不确定性梯度来源（仅 3 条）:
  ① MIL_loss → logsigma_video → per_anchor_var/modal_probs → nig_layer/dirichlet_layer
  ② evidential_loss → epistemic_cont/alpha_dir → 同上
  ③ orth_loss → anchors → decoder → anchor_tokens

  ❌ uncertainty_reg_loss: wti_logits.detach() → 零梯度
```

### 9.2 补充设计问题

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | NIG 过参数化：每 anchor 每维度独立 ν,α,β，共 24,576 个不确定性参数 | `module_sap.py:66` | 训练困难，梯度噪声大 |
| 2 | `_evidential_similarity` 不对称：仅视频侧有置信度折扣，文本侧无 | `modeling_mulit.py:730-746` | 相似度矩阵不对称 |
| 3 | `epistemic_cont [B,16,512]` → `.mean(dim=(1,2))` 坍缩为标量，丢弃全部结构化信息 | `modeling_mulit.py:743` | 无法区分锚点/维度的不确定性 |
| 4 | 两个 `EvidentialUncertaintyHead` 重复定义 | `module_sap.py:28` vs `uncertainty_module.py:228` | 死代码混淆 |
| 5 | 文本侧不确定性独立运作，与视频侧无交叉约束 | `modeling_mulit.py:848-873` | 两侧分布无对齐 |

### 9.3 方案 A：不确定性直接参与检索评分 🔴 高优先

**动机**：当前不确定性输出不进入主检索路径，梯度完全断裂。

**改动**（`modeling_mulit.py:_loose_similarity`）：

```python
# 当前：纯 WTI logits 作为检索分数
retrieve_logits = wti_logits  # [B, B]

# 改为：不确定性置信度加权
confidence = 1.0 / (1.0 + epistemic_video.mean(dim=(1, 2)))  # [B]
retrieve_logits = wti_logits * confidence.unsqueeze(1)         # [B, B]
```

**优点**：
- 不确定性首次获得与检索目标的直接梯度连接
- 改动极小（~3 行），不引入新模块
- 高不确定性样本自动降权，天然实现困难样本软过滤
- 使用 `1/(1+x)` 而非 `exp(-x)`，梯度更稳定（`exp(-x)` 在 x 大时梯度消失）

**风险与缓解**：
- 初期不确定性未校准可能拖累检索 → 加 linear warmup（前 500 step 纯 WTI，之后线性混入）
- 需要监控 confidence 值的分布，防止全部坍缩到 0 或 1

---

### 9.4 方案 B：解耦不确定性与语义路径 🔴 高优先

**动机**：STATUS.md 根因 3 — decoder 输出的 anchors 同时服务语义聚合和不确定性估计，两路梯度冲突（evidential 梯度 ~5.9 vs sim_loss ~0.2-0.8）。

**方案 B1（最小改动）**：stop_gradient 隔离。

```python
# module_sap.py:140 — 在 EvidentialUncertaintyHead 前插入
uncertainty_input = anchors.detach()  # 阻断语义梯度
ev = self.evidential_head(uncertainty_input)
```

**方案 B2（彻底）**：独立不确定性编码器，不共享 decoder。

```python
self.uncertainty_probe = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
# 从 frame features 直接估计不确定性，绕过 anchor decoder
```

**优点**：消除双头梯度冲突。方案 B1 仅 1 行改动。

**代价**：方案 B2 需新增模块，增加 ~1M 参数。

**建议**：先用 B1 验证效果，若不确定性开始产生正向贡献再考虑 B2。

---

### 9.5 方案 C：标量化不确定性 🔶 中优先

**动机**：当前 NIG 为每个 anchor 的每个维度估计独立的 ν, α, β（3×16×512=24,576 参数），过参数化导致训练困难。实际有用的信号只有一个标量：这个 anchor 有多可靠。

**改动**（`module_sap.py:EvidentialUncertaintyHead`）：

```python
# 替换 nig_layer: Linear(512, 2048) → Linear(512, 1)
self.uncertainty_fc = nn.Linear(d_model, 1)  # 每 anchor 一个标量 log-variance

# 前向
logsigma_per_anchor = self.uncertainty_fc(anchor_features).squeeze(-1)  # [B, K]
# 低不确定 → 高聚合权重
uncertainty_weight = F.softmax(-logsigma_per_anchor, dim=-1)  # [B, K]
mu_raw = (uncertainty_weight.unsqueeze(-1) * anchors).sum(dim=1)  # [B, D]
# 聚合不确定性
logsigma = torch.logsumexp(logsigma_per_anchor - math.log(K), dim=-1)  # [B]
```

**优点**：
- 参数量从 24K → 512（减少 98%）
- 标量不确定性更易训练和校准
- 保持 dirichlet_layer 用于模态概率（语义聚合），仅简化 NIG 连续层
- 不确定性降权的聚合方式比模态概率加权更直观

---

### 9.6 方案 D：不确定性驱动的难负样本挖掘 🔶 中优先

**动机**：不确定性可用于识别难样本（高不确定性 = 模型不确定 = 可能在决策边界附近），对这些样本加权可提升对比学习的判别力。

**改动**（`modeling_mulit.py:forward`）：

```python
# 高不确定性 → 难样本 → 在 contrastive loss 中提权
difficulty_weight = 1.0 + epistemic_video.mean(dim=(1, 2))  # [B]
# 对 sim_loss 每样本加权
sim_loss_per_sample = ...  # 提取 per-sample loss
sim_loss = (sim_loss_per_sample * difficulty_weight).mean()
```

**优点**：不确定性服务于主 loss（sim_loss），梯度路径直接。

**注意**：CrossEn 不直接支持 per-sample 权重，需修改 `loss_fct` 或手动展开计算。

---

### 9.7 方案 E：MIL 方差正则化反转 🔷 低优先

**动机**：MIL_loss 天然惩罚方差，logsig_v 在 200 步内撞击 -1.5 下界。需要一个对冲力维持最小方差。

**改动**（`modeling_mulit.py:forward`）：

```python
variance_floor = -1.0  # target log-variance floor
variance_penalty = F.relu(variance_floor - logsigma_video).mean()
loss += 0.01 * variance_penalty
```

**缺点**：
- 人为设定方差目标不自然
- 即使方差不坍缩，MIL_loss 本身在 Epoch 1 后已饱和（0.16），维持方差无实际收益
- 仅当方案 A 使 MIL 采样重新产生价值时才有意义

---

### 9.8 方案 F：清理死代码 🟢 清理

| 项 | 内容 | 位置 |
|----|------|------|
| F1 | 删除 `uncertainty_reg_loss`（`.detach()` 零梯度，纯噪音） | `modeling_mulit.py:669,723` |
| F2 | 删除重复的 `EvidentialUncertaintyHead` | `prob_models/uncertainty_module.py:228-269` |
| F3 | 删除 `uncertainty_mode=nig_mil` 分支（方案 A 直接替代） | `modeling_mulit.py:594-597` |
| F4 | `evidential_loss` / `neg_reg_loss` 保留关闭状态（`w=0`），暂不删代码 | `modeling_mulit.py:655-660,721-722` |

---

### 9.9 推荐执行路径

```
第1步 (清理):   方案 F — 删除 uncertainty_reg_loss、重复类、nig_mil 分支
第2步 (核心):   方案 A — 不确定性置信度融入 WTI 检索分数（~3 行，加 warmup）
第3步 (加固):   方案 B1 — stop_gradient 隔离双头梯度冲突（~1 行）
第4步 (简化):   方案 C — 标量化 NIG，减少 98% 不确定性参数
第5步 (可选):   方案 D — 若方案 A 有效，叠加难负样本挖掘
第6步 (暂缓):   方案 E — 仅当方案 A 已成功且 MIL 重新生效时考虑
```

**预期**：方案 A+B 组合应能让不确定性对 R@1 产生首次正向贡献。当前 baseline 49.4 来自纯 sim_loss，不确定性完全闲置 — 任何正向利用都是净收益。
