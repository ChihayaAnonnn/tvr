# UATVR 项目现状

> 更新时间：2026-06-14（此次会话完整重构）

## 一、项目概述

UATVR (Uncertainty-Aware Text-Video Retrieval) 基于 CLIP ViT-B/16。
核心模块：SAP（Semantic Anchor Probing）— 16 个可学习 anchor token 通过 2 层
TransformerDecoder 探测视频时空特征，输出概率化视频表征。

- **数据集**：MSRVTT（train 9k + test 1k），MSVD（备用）
- **主指标**：T2V R@1 / V2T R@1（取 T2V R@1 为优化目标）
- **Git 分支**：`master`（当前所在）
- **活跃模型**：`modules/modeling_mulit.py`（命名 typo "mulit" 是历史债务）

> 2026-06-08 到 2026-06-14 期间，执行了从方案 F 到方向 1 的完整迭代。
> 下文使用**当前状态**，不重复历史中间态。

---

## 二、当前最优结果与关键实验

| 实验 | commit | 关键改动 | Best T2V R@1 |
|------|--------|---------|-------------|
| Baseline pure sim | `169ba95` | uncertainty_mode=none, w_evid=0, w_neg=0 | **49.4** @ Epoch 4 |
| Exp 1 (Plan A+B1, NIG) | ~ | unsqueeze(1), detach, NIG uncertainty | **50.0** @ Epoch 4 |
| Dir1_colwise_warmup500 | `23c3065` | Dir1 + unsqueeze(0) + warmup=500 | **49.3** @ Epoch 3 |
| Dir1_zscore_warmup500 | `6ec88b9` | Dir1 + unsqueeze(0) + batch z-score → sigmoid | ~46.5（实验中断，已废弃） |

**Exp 1 的 50.0 是历史最高，但其置信度加权对 T2V 实际贡献为零**（见第四节分析）。
当前实际可复现的最优为 baseline 49.4。

最优 checkpoint：`ckpts/ckpt_msrvtt_20260607_102327/pytorch_model.bin.3`（baseline）

---

## 三、当前架构（Direction 1 — 非学习不确定性）

### 3.1 数据流

```
视频帧 → CLIP ViT-B/16 → frame tokens [B, T×S, 512]
                                │
            ┌───────────────────┘
            ▼
   SAP.decoder (TransformerDecoder)
      anchor_tokens [B, 16] × frame tokens
            │
            ▼
      anchors [B, 16, 512]  ←─── LayerNorm 后
            │
   ┌───────┼───────────────────────┐
   │       │                       │
   │   anchor_proj (Linear)    [B1] detached
   │       │                       │
   │   projected [B, 16, 512]      ▼
   │       │              EvidentialUncertaintyHead
   │   modal_probs              (仅 Dirichlet layer)
   │   加权聚合                      │
   │       │                  alpha_dir, u_mode
   │   mu_raw [B, 512]              │
   │   (L2 归一化)             modal_probs
   │                              （detached，用于熵计算）
   │                                  │
   │   非学习不确定性 ◄────────────────┘
   │   - anchor 多样性 (1−cos_sim均值)
   │   - 模态熵 (Dirichlet熵/K的log)
   │   - e_c = diversity × entropy_norm
   │     epistemic_cont [B, 1, 1]
   │
   │   logsigma: anchor 维度方差取 log
   │
   ▼
mu_raw [B, 512] + logsigma [B, 512] + epistemic_cont [B, 1, 1]
   │
   ▼
modeling_mulit.py: WTI 相似度 + confidence 加权
```

### 3.2 关键代码位置

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| EvidentialUncertaintyHead | `query_models/module_sap.py` | 27–60 | 仅含 Dirichlet layer，输出 alpha_dir + u_mode |
| SAP.__init__ | 同上 | 63–91 | anchor_tokens + decoder + evidential_head + anchor_proj |
| SAP.forward | 同上 | 93–156 | decoder → anchor_proj → detach → head + 非学习不确定性 |
| B1 梯度隔离 | 同上 | 116 | `ev = self.evidential_head(projected.detach())` |
| 非学习不确定性 | 同上 | 121–134 | diversity × modal_entropy_norm |
| logsigma 计算 | 同上 | 141–145 | `torch.var(anchors, dim=1).mean(dim=-1)` → log |
| 置信度加权 | `modules/modeling_mulit.py` | 665–675 | `1/(1+x)` + warmup + `unsqueeze(0)` |
| T2V 列方向折扣 | 同上 | 675 | `weighted_logits = wti_logits * confidence.unsqueeze(0)` |
| V2T 路径 | 同上 | — | `confidence.unsqueeze(1)` 行方向缩放（转置交叉熵中抵消） |
| diag 链 | 同上 | 677–734 | pos/neg/gap + epistemic + confidence + warmup_alpha |
| MIL loss + 采样 | 同上 | — | 使用 logsigma 做高斯采样 |
| orth_loss | 同上 | — | 锚点正交性正则 |

### 3.3 已删除/废弃的组件

- **NIG 层**（γ/ν/α/β）：方案 C 标量化后完全移除
- **EvidentialUncertaintyHead（重复副本）**：`prob_models/uncertainty_module.py:228-269` 已删除（方案 F2）
- **uncertainty_reg_loss**：detached，无梯度，已删除（方案 F1）
- **nig_mil 分支**：已删除（方案 F3）
- **视频侧 PIENet**：更早前已移除
- **工作二/QueryFormer/Qwen3-VL**：未接入，文档中已移除描述

---

## 四、核心发现：置信度加权对 R@1 不贡献

### 4.1 为什么 Exp 1 达到 50.0

Exp 1 (50.0) 的收益来自 **B1 梯度隔离（anchors.detach()）**，而非置信度加权。

Exp 1 使用 `unsqueeze(1)` 行方向缩放：

```python
weighted_logits = wti_logits * confidence.unsqueeze(1)  # [B, 1] × [B, B] → 行缩放
```

T2V cross_entropy 对每行做 softmax：

```
softmax_i = exp(w_i × c) / Σ_j exp(w_j × c)   =   exp(w_i) / Σ_j exp(w_j)
// c 因子在分子分母同时出现 → 约掉 → 无 T2V 效果
```

**结论**：50.0 的 T2V R@1 收益来自 B1 梯度隔离改善了表征质量（ret_gap 从 ~15 → ~18），
置信度加权对 T2V R@1 贡献为零。

### 4.2 当加权真正生效时发生了什么

改用 `unsqueeze(0)` 后加权终于进入了 T2V softmax：

| 实验 | 加权方向 | T2V 实际生效? | Best T2V R@1 |
|------|---------|-------------|-------------|
| Exp 1 | unsqueeze(1) | ❌ 行方向抵消 | 50.0 |
| Dir1_colwise | unsqueeze(0) | ✅ 列方向 | 49.3 |
| Dir1_zscore | unsqueeze(0) | ✅ 列方向 | ~46.5 |

真正生效后，最佳仅 49.3（低于 baseline 49.4）。
Z-score 放大区分度后进一步恶化到 46.5。

### 4.3 根本原因

per-video 置信度缩放是对称的——一个视频作为正确答案和作为干扰项时
被同等地缩放。给定视频 V：

- 当 V 是正确答案（对角线）：confidence[V] 缩放了正确分数
- 当 V 是干扰项（非对角线）：confidence[V] 缩放了干扰分数

**正负影响对称抵消。** 除非不确定性有 **query-dependent** 信号
（"视频 V 对当前查询有多不确定"），否则均匀的 per-video 缩放只能在
batch 内重新分配分数水位，不能系统性地让正确答案浮上来。

---

## 五、什么确实有效：B1 梯度隔离

```
+0.6 R@1（49.4 → 50.0）
```

`projected.detach()` 阻止了 evidential_head 的梯度回传至 decoder
和 anchor_tokens。原来两路梯度（语义聚合 vs 不确定性估计）在 decoder
瓶颈处冲突：语义聚合需要 anchor 专注于模态建模，不确定性估计需要
anchor 捕捉噪声/变化。

分离后 decoder 只接收语义聚合的梯度，表征质量显著提升。

**这是唯一被严格验证的正向收益。**

---

## 六、非学习不确定性（Direction 1）状态

### 6.1 设计

```
epistemic_video = diversity × modal_entropy_norm
  diversity      = 1 − mean(cos_sim(anchor_i, anchor_j))  // [B]
  entropy_norm   = H(modal_probs) / log(K)                // [B] ∈ [0,1]
```

0 个可学习参数，直接从 anchor 统计量计算。

### 6.2 行为

- **epistemic_v_mean**：从 ~0.6 (Epoch 1) 稳定增长到 ~0.9 (Epoch 3+)
- **u_mode_std**：仅 0.001-0.004 —— batch 内 anchor 几乎无差异化
- **ret_gap**：27.5（历史最高，表征质量强）
- **不坍缩**：与 NIG/标量不确定性不同，非学习不确定性不收敛到常量

### 6.3 问题

batch 内 per-video 的 epistemic 方差极低（u_mode_std ~0.002），
意味着所有视频的不确定性几乎相同，`1/(1+x)` 映射后 confidence
差异仅 ~6%，softmax 几乎无感。

Diversity 计算（anchor 间 cosine 相似度的均值）在 batch 内区分度很低，
因为同一个模型的所有视频使用同一组 anchor_tokens。

---

## 七、当前 Loss 结构

| Loss | 权重 | 状态 |
|------|------|------|
| sim_loss（WTI CrossEn） | 1.0 | ✅ 唯一驱动 R@1 的 loss |
| MIL_loss | 0.01 | ⚠️ 饱和，退化为 vanilla InfoNCE |
| orth_loss | 0.1 | ✅ Epoch 1 自行消解 |
| evidential_loss | 0 | ❌ 已关闭（w_evid=0），干扰语义学习 |
| neg_reg_loss | 0 | ❌ 已关闭（w_neg=0） |

训练命令模板：
```bash
EXPERIMENT_DESC="<desc>" CUDA_VISIBLE_DEVICES=<gpus> bash run_train_msrvtt_bg.sh \
  --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 \
  --batch_size 64 --gradient_accumulation_steps 1
```

有效 batch = 64 × GPU数 × 1。4 GPU → eff_batch = 256。

---

## 八、Git 里程碑（master 分支）

| Commit | 描述 | T2V R@1 |
|--------|------|---------|
| `169ba95` | 添加 log-analysis skill 与实验分析归档 | 49.4 |
| `4d64c9b` | 方案 C：SAP 不确定性标量化 | — |
| `21348d1` | 方案 F(清理) + 方案 A(置信度加权) + 方案 B1(梯度隔离) | 50.0 |
| `61f91c2` | 方向 1：非学习不确定性（anchor 统计量） | — |
| `23c3065` | unsqueeze(0) 修正 T2V 列方向折扣 | 49.3 |
| `6ec88b9` | 文档精简（当前 HEAD） | — |

---

## 九、未解决问题与下步方向

### 9.1 优先级排序

| 优先级 | 方向 | 说明 |
|--------|------|------|
| 🟢 P0 | **去掉置信度加权** | 已验证不贡献 R@1，保留增加复杂度 |
| 🟢 P0 | **保留 B1 梯度隔离** | 唯一正向收益（+0.6），已验证 |
| 🟡 P1 | **优化表征质量** | ret_gap 27.5 说明正负分离强，但 R@1 未充分受益 |
| 🟡 P1 | **提升 anchor 多样性区分度** | 当前 batch 内几乎无差异化 |
| ⚪ P2 | **MIL loss 改革** | 当前饱和，可考虑新的方差利用方式 |

### 9.2 可以考虑但需谨慎的

1. **query-dependent 不确定性**：不是 per-video 均匀权重，而是 per (query, video) pair 的置信度。这样正负影响不再对称，可能真正提升 R@1。需要新的架构设计。
2. **不确定性用作检索后校准**：不参与排序，而是输出每个结果的置信度。对 R@1 无帮助但对下游有用。
3. **超参自动搜索**：当架构稳定 2+ 轮无结构变更后再启动。当前不建议。

### 9.3 不建议的

- ❌ **batch 标准化 / z-score**：破坏绝对参照，batch 内方差太低时放大噪声
- ❌ **β sweep on 1/(1+βx)**：输入区间 [0.8,1.0] 太窄，任何绝对单调函数区分度都有限
- ❌ **重新引入 NIG 层**：已验证坍缩到常量

---

## 十、文件速查

| 文件 | 行数 | 用途 |
|------|------|------|
| `modules/modeling_mulit.py` | ~900 | 主模型 UATVR，forward + loss + diag |
| `query_models/module_sap.py` | 157 | SAP + EvidentialUncertaintyHead（仅 Dirichlet） |
| `prob_models/uncertainty_module.py` | ~200 | 文本侧不确定性头（attention/GRU/Mamba） |
| `main_task_retrieval.py` | ~ | 训练/评估入口，argparse |
| `train_msrvtt.sh` | ~ | MSRVTT 训练脚本（bash 包装） |
| `run_train_msrvtt_bg.sh` | ~ | nohup 后台训练 |
| `hyperparam_search.py` | ~ | 超参搜索（Bayesian + Grid） |
| `tests/test_modeling_mulit_losses.py` | ~ | 单元测试 |
| `AGENTS.md` | — | 项目架构文档 |
| `CLAUDE.md` | — | 命令速查 |
| `STATUS.md` | — | 本文件 |

### 活跃实验日志

```
logs/20260608/   — Exp 1 (50.0, 首个 B1 + A)
logs/20260609/   — 对称折扣实验
logs/20260610/   — 列方向 + Plan C 实验
logs/20260611/   — Plan C fixed
logs/20260612/   — 方向 1 unsqueeze(1)
logs/20260613/   — 方向 1 unsqueeze(0) (49.3)
logs/20260614/   — z-score 实验（失败，已中断）
```

每个实验目录下 `analysis_*.md` 为详细分析文件。

---

## 十一、给下一个会话的摘要

**一句话**：B1 梯度隔离是唯一被验证的收益来源（+0.6 → 50.0），
置信度加权无论用 `1/(1+x)` 还是 z-score 都无法在 T2V softmax 中
产生正贡献——per-video 均匀权重正负对称抵消。当前代码处于
Direction 1 + `1/(1+x)` + `unsqueeze(0)` 状态（commit `6ec88b9`）。

**推荐起点**：去掉 confidence 加权（仅保留 B1 detach），
在 49.9-50.0 的基础上优化表征质量或尝试 query-dependent 不确定性。
