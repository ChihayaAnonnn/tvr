# UATVR 项目现状

> 更新时间：2026-06-16（per-pair 置信度探索完结）

## 一、项目概述

UATVR (Uncertainty-Aware Text-Video Retrieval) 基于 CLIP ViT-B/16。
核心模块：SAP（Semantic Anchor Probing）— 16 个可学习 anchor token 通过 2 层
TransformerDecoder 探测视频时空特征，输出概率化视频表征。

- **数据集**：MSRVTT（train 9k + test 1k），MSVD（备用）
- **主指标**：T2V R@1 / V2T R@1（取 T2V R@1 为优化目标）
- **Git 分支**：`master`
- **活跃模型**：`modules/modeling_mulit.py`（命名 typo "mulit" 是历史债务）
- **总参数量**：176.70M（CLIP 149.62M + SAP 8.68M + 文本侧 3.81M + 其他 ~14M）
- **CLIP LR**：base_lr × coef_lr = 1e-4 × 1e-3 = 1e-7（差分学习率，近似冻结）

> 2026-06-08 到 2026-06-16 期间，从方案 F 到 per-pair 置信度 MLP，共执行 8 轮实验。
> 下文使用**当前状态**，不重复历史中间态。

---

## 二、所有关键实验结果

| 实验 | commit | 关键改动 | Best T2V R@1 | ret_gap |
|------|--------|---------|:---:|:---:|
| Baseline pure sim | `169ba95` | uncertainty_mode=none, w_evid=0, w_neg=0 | **49.4** | — |
| Exp 1 (Plan A+B1, NIG) | ~ | unsqueeze(1), detach, NIG uncertainty | **50.0** | 18.0 |
| Dir1_colwise_warmup500 | `23c3065` | Dir1 + unsqueeze(0) + warmup=500 | **49.3** | 27.5 |
| Dir1_zscore_warmup500 | `6ec88b9` | Dir1 + unsqueeze(0) + batch z-score | ~46.5 | ~24 |
| PairConf (per-pair) | `a6b32ef` | per-pair anchor entropy, 无 detach | **48.6** | 16.7 |
| PairConfMLP (per-pair+MLP) | `fe5e744` | per-pair MLP + detach 梯度隔离 | **48.6** | 26.3 |

**Exp 1 的 50.0 是历史最高，但收益来自 B1 梯度隔离，非置信度加权**（见第四节）。
最优 checkpoint：`ckpts/ckpt_msrvtt_20260607_102327/pytorch_model.bin.3`（baseline）

---

## 三、当前架构

### 3.1 数据流

```
视频帧 → CLIP ViT-B/16 → frame tokens [B, T×S, 512]
            │
            ▼
   SAP.decoder (TransformerDecoder)
      anchor_tokens [B, 16] × frame tokens
            │
            ▼
      anchors [B, 16, 512]
            │
   ┌───────┼───────────────────────┐
   │   anchor_proj             detach (B1)
   │       │                       │
   │   projected [B,16,512]   EvidentialUncertaintyHead
   │       │                  (仅 Dirichlet layer)
   │   modal_probs                 │
   │   加权聚合              alpha_dir, u_mode
   │       │
   │   mu_raw [B, 512]      非学习不确定性
   │   (L2 norm)            diversity × entropy_norm
   │                        epistemic_cont [B,1,1]
   │
   │   logsigma: anchor方差取log
   │
   ▼
mu_raw [B,512] + logsigma [B,512] + epistemic_cont [B,1,1]
   │
   ▼
modeling_mulit.py: WTI + confidence 加权 (当前: per-pair MLP + detach)
```

### 3.2 关键代码位置

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| EvidentialUncertaintyHead | `query_models/module_sap.py` | 27–60 | 仅含 Dirichlet layer |
| SAP.__init__ | 同上 | 63–91 | anchor_tokens + decoder + head + anchor_proj |
| SAP.forward | 同上 | 93–156 | B1 detach + 非学习不确定性 |
| anchor_attn + MLP 置信度 | `modules/modeling_mulit.py` | 676–693 | detach 输入 + confidence_mlp |
| confidence_mlp 定义 | 同上 | ~332 | nn.Sequential(16→8→1, Sigmoid) |
| WTI weighted_logits | 同上 | 693 | `wti_logits * confidence` [B,B] |
| diag 链 | 同上 | 695–720 | pos/neg/gap + confidence_diag/off/gap |
| Chain-Prob 日志 | `main_task_retrieval.py` | 737–745 | 含 cnf/cnf_d/cnf_off/gap |

### 3.3 已删除/废弃的组件

- **NIG 层**（γ/ν/α/β）：方案 C 标量化后移除
- **EvidentialUncertaintyHead（重复副本）**：`prob_models/uncertainty_module.py` 已删除
- **uncertainty_reg_loss / nig_mil 分支**：已删除
- **视频侧 PIENet**：已移除

---

## 四、核心发现：置信度加权对 R@1 的贡献为零

### 4.1 Exp 1 的 50.0 来自什么

Exp 1 的 50.0 收益 = **B1 梯度隔离（detach）+0.6**，置信度加权贡献为零。

Exp 1 使用 `unsqueeze(1)` 行方向缩放，在 T2V cross_entropy 中每行的公共因子被 softmax 约掉，无任何效果。

### 4.2 per-video 加权（unsqueeze(0)）

真正生效后最佳仅 49.3（< baseline 49.4）。根因：per-video 标量在 softmax 中正负对称抵消。

### 4.3 per-pair 加权（[B,B] 矩阵）

解决了正负对称问题，但引发新问题：

| 变体 | detach | T2V R@1 | ret_gap | 置信度行为 |
|------|--------|:---:|:---:|------|
| PairConf | ❌ | 48.6 | 16.7 | 梯度冲突，表征退化 |
| PairConfMLP | ✅ | 48.6 | **26.3** | MLP 坍缩为常量 |

**detach 成功恢复了表征质量**（ret_gap 16.7 → 26.3），但 **MLP 始终输出相同值**（cnf_d = cnf_off, gap = 0.000 贯穿 5 个 epoch）。

根因：`anchor_attn`（text_i 对 video_j 的 16 路 softmax 注意力）在正负对上无法区分——16 路里总有 1-2 个 anchor 得分突出，熵始终很低。MLP 收到的正负对输入几乎相同，无法学习判别。

### 4.4 确定性结论

**置信度加权路径（per-video / per-pair / 熵 / MLP）已彻底探索完毕，均不能提升 T2V R@1。**

原因：不确定性/置信度信号中包含的"噪声/校准"信息与 softmax 排序所需的"判别"信息是正交的。在对比学习框架中将不确定性用于排序权重，本质上是让一个没有正负区分度的信号去调节正负分数——无论怎么设计映射函数，都无法让无区分度的输入产生有区分度的输出。

---

## 五、唯一正向收益：B1 梯度隔离

```
+0.6 R@1（49.4 → 50.0）
```

`projected.detach()` 阻止 evidential_head 梯度回传至 decoder 和 anchor_tokens，
消除两路梯度（语义聚合 vs 不确定性估计）在 decoder 瓶颈处的冲突。

**这是整个不确定性优化工作中唯一被严格验证的正向收益。**

---

## 六、非学习不确定性（Direction 1）状态

### 6.1 设计

```
epistemic_video = diversity × modal_entropy_norm
  diversity      = 1 − mean(cos_sim(anchor_i, anchor_j))         // [B]
  entropy_norm   = H(modal_probs) / log(K)                       // [B] ∈ [0,1]
```

0 个可学习参数。不坍缩，但不贡献 R@1。

### 6.2 行为

- epistemic_v_mean：~0.6 → ~0.9 (Epoch 1 → 3)
- u_mode_std：0.001-0.004，batch 内几乎无差异
- 在 `1/(1+x)` 映射后 confidence 差异仅 ~6%

---

## 七、当前 Loss 结构

| Loss | 权重 | 状态 |
|------|------|------|
| sim_loss（WTI CrossEn） | 1.0 | ✅ 唯一驱动 R@1 |
| MIL_loss | 0.01 | ⚠️ 饱和，退化 |
| orth_loss | 0.1 | ✅ Epoch 1 消解 |
| evidential_loss | 0 | ❌ 已关闭 |
| neg_reg_loss | 0 | ❌ 已关闭 |

训练命令模板：
```bash
EXPERIMENT_DESC="<desc>" CUDA_VISIBLE_DEVICES=<gpus> bash run_train_msrvtt_bg.sh \
  --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 \
  --batch_size 64 --gradient_accumulation_steps 1
```
有效 batch = 64 × 4GPU × 1 = 256。

---

## 八、Git 里程碑

| Commit | 描述 | T2V R@1 |
|--------|------|:---:|
| `169ba95` | 添加 log-analysis skill | 49.4 |
| `21348d1` | 方案 F+A+B1 | 50.0 |
| `61f91c2` | 方向 1：非学习不确定性 | — |
| `23c3065` | unsqueeze(0) 列方向修正 | 49.3 |
| `6ec88b9` | 文档精简 + z-score 回退 | — |
| `a6b32ef` | STATUS.md 重构，per-pair 总结 | — |
| `fe5e744` | per-pair confidence_mlp + detach（HEAD） | 48.6 |

---

## 九、下一步建议

### 9.1 推荐（优先级排序）

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 🟢 P0 | **去掉置信度加权，仅保留 B1 detach** | 已验证 50.0，唯一正向收益 |
| 🟢 P0 | **关闭/移除 MIL loss** | 已饱和撞 -1.5 下界，保留无意义 |
| 🟡 P1 | **优化 WTI 表征质量** | ret_gap 27.5 但 R@1 49.4，有提升空间 |
| 🟡 P1 | **简化 SAP** | 去掉 evidential_head + non-learned uncertainty，仅保留 anchor_proj |
| ⚪ P2 | **超参自动搜索** | 架构稳定后启动 |

### 9.2 不建议

- ❌ 任何形式的置信度加权（per-video / per-pair / entropy / MLP）
- ❌ batch 标准化 / z-score / β sweep
- ❌ 重新引入 NIG 层
- ❌ 大改 batch_size（破坏实验对照）

### 9.3 如果仍要探索不确定性方向

不确定性在检索中的价值不在排序精度，而在：
- **检索后校准**：输出每个结果的不确定性分数
- **OOD/开集检测**：区分分布内/外查询
- **主动学习**：选择高不确定性的未标注样本

这些不与 R@1 直接挂钩，但是不确定性机制的自然应用场景。

---

## 十、文件速查

| 文件 | 用途 |
|------|------|
| `modules/modeling_mulit.py` (~920 行) | 主模型：forward + loss + diag + confidence_mlp |
| `query_models/module_sap.py` (157 行) | SAP + EvidentialUncertaintyHead（仅 Dirichlet） |
| `prob_models/uncertainty_module.py` | 文本侧不确定性头 |
| `main_task_retrieval.py` | 训练/评估入口 |
| `train_msrvtt.sh` / `run_train_msrvtt_bg.sh` | 训练脚本 |
| `AGENTS.md` / `CLAUDE.md` | 架构文档 / 命令速查 |
| `STATUS.md` | 本文件 |

### 实验日志与 Analysis

```
logs/20260608/  — Exp 1 (50.0, B1+A)
logs/20260609/  — 对称折扣实验
logs/20260610/  — 列方向 + Plan C
logs/20260611/  — Plan C fixed
logs/20260612/  — 方向 1 unsqueeze(1)
logs/20260613/  — 方向 1 unsqueeze(0) (49.3)
logs/20260614/  — z-score (废弃) + PairConf (48.6)
logs/20260615/  — PairConfMLP (48.6)
logs/20260616/  — analysis 文件归档
```

---

## 十一、给下一个会话的摘要

**一句话**：不确定性置信度加权（per-video、per-pair、entropy、MLP 全路径）
对 T2V R@1 贡献为零，已彻底验证。唯一正向收益是 B1 梯度隔离（detach, +0.6 → 50.0）。

**当前代码状态**：per-pair confidence_mlp + detach（commit `fe5e744`）。
MLP 坍缩为常量 0.59，gap=0.000，实际等价于 B1-only。

**推荐起点**：
1. 去掉 confidence 加权路径，仅保留 B1 detach
2. 可选择去掉 non-learned uncertainty 计算（diagnostics only）
3. 在 ~50.0 的干净起点上优化 WTI 表征或探索其他方向
