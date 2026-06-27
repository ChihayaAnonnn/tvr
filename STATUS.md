# UATVR 项目现状

> 更新时间：2026-06-22（raw hard-negative packing 已完成但未提升；clean map 已生成，等待 GPU 空闲后重跑）

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

> 2026-06-08 到 2026-06-22 期间，从方案 F 到 per-pair 置信度 MLP，再到 B1-only v2 复现实验，
> 已完成主要置信度加权路径排查。当前工作树为 **B1-only（无 confidence 加权）**，并已接入
> hard-negative batch packing 的代码开关，默认关闭。raw hard-negative 映射实验未提升，当前等待 clean 映射实验。

---

## 二、所有关键实验结果

| 实验 | commit | 关键改动 | Best T2V R@1 | ret_gap |
|------|--------|---------|:---:|:---:|
| Baseline pure sim | `169ba95` | uncertainty_mode=none, w_evid=0, w_neg=0 | **49.4** | — |
| Exp 1 (Plan A+B1, NIG) | ~ | unsqueeze(1), detach, NIG uncertainty | **50.0** | 18.0 |
| Baseline pure sim repeat2 | `169ba95` worktree | 同配置复现 | **48.7** | — |
| Exp 1 repro repeat2 | `21348d1` worktree | 历史 50.0 路径复现 | **49.1** | — |
| Dir1_colwise_warmup500 | `23c3065` | Dir1 + unsqueeze(0) + warmup=500 | **49.3** | 27.5 |
| Dir1_zscore_warmup500 | `6ec88b9` | Dir1 + unsqueeze(0) + batch z-score | ~46.5 | ~24 |
| PairConf (per-pair) | `a6b32ef` | per-pair anchor entropy, 无 detach | **48.6** | 16.7 |
| PairConfMLP (per-pair+MLP) | `fe5e744` | per-pair MLP + detach 梯度隔离 | **48.6** | 26.3 |
| B1only_v2 repeat1 | 未提交 | 移除 confidence_mlp，WTI logits 直接检索 | **49.3** | 16.7 |

**Exp 1 的 50.0 是历史最高，但收益只能归因到当时的 NIG gamma + detach 组合，不能直接外推到当前 anchor_proj + Dirichlet-only 架构。**
历史最高 checkpoint：`ckpts/ckpt_msrvtt_20260608_180208/pytorch_model.bin.4`（Exp 1）。
当前强 baseline checkpoint：`ckpts/ckpt_msrvtt_20260607_102327/pytorch_model.bin.3`（49.4）。

### 2.1 2026-06-19 三组复现实验最终结果

| 实验 | 日志 | 最佳 checkpoint | Best T2V R@1 | 对应 V2T R@1 | 结论 |
|------|------|----------------|:---:|:---:|------|
| B1only_v2_repeat1 | `logs/20260617/b1only_v2_repeat1_train_msrvtt.log` | `pytorch_model.bin.4` | **49.3** | 47.6 | 当前三组最优，但未到 50 |
| exp1_repro_repeat2 | `../UATVR_exp1_21348d1/logs/20260617/exp1_repro_repeat2_train_msrvtt.log` | `pytorch_model.bin.4` | **49.1** | 46.4 | 接近 B1，但未复现历史 50.0 |
| baseline_pure_sim_repeat2 | `../UATVR_baseline_169ba95/logs/20260617/baseline_pure_sim_repeat2_train_msrvtt.log` | `pytorch_model.bin.4` | **48.7** | 47.3 | 低于历史 baseline 49.4 |

关键判断：

- 三组复现实验均已正常结束，没有训练中断或 OOM 迹象。
- 历史最高 50.0 没有被复现；当前最可靠的新结果是 B1only_v2 repeat1 的 49.3。
- Exp1 repro 相比 baseline repeat 在 T2V R@1 上高 0.4，但 V2T R@1 更低，不能证明全面优于 baseline。
- 后续不建议继续堆同配置 repeat；应转向作用于主 WTI/CrossEn 链路的 hard-negative batch packing。

### 2.2 2026-06-19 / 2026-06-22 Hard-Negative 结果与数据清洗

| 项目 | 路径 / 日志 | 关键结果 | 结论 |
|------|-------------|---------|------|
| raw HN packing 训练 | `logs/20260619/hn_pack_wmil0_repeat1_4gpu_b64_train_msrvtt.log` | Best T2V R@1 = **48.1**，最佳 ckpt=`pytorch_model.bin.3`；V2T R@1 最高 48.7 | 机制跑通，但 T2V top-1 明显低于 B1-only v2 的 49.3 |
| raw HN 映射 | `cache_dir/hard_negatives/msrvtt_train_hardneg.json` | 180000/180000 links | 可追溯，但不再作为默认训练路径 |
| audit 报告 | `cache_dir/hard_negatives/msrvtt_train_hardneg_audit.md` | exact caption pairs=4059，高风险 pairs=17637，max hard caption/video 复用=504/547 | raw map 有明显假负例和通用 caption 吸附问题 |
| clean HN 映射 | `cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json` | 163592/180000 links；exact caption pairs=0；max hard caption/video 复用=39/93 | 当前默认 HN 路径，等待训练验证 |
| audit/clean 脚本 | `scripts/audit_msrvtt_hard_negatives.py` | 生成 clean JSON、audit markdown、manual review CSV | 已有单测 `tests/test_audit_msrvtt_hard_negatives.py` |

raw HN packing 的训练设置：

```bash
RUN_DATE=20260619 RUN_TIME=hn_pack_wmil0_repeat1_4gpu_b64 CUDA_VISIBLE_DEVICES=1,2,3,4 EXPERIMENT_DESC="HN packing, 4GPU global batch=64, w_mil=0" bash run_train_msrvtt_bg.sh --use_hard_negative_packing --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg.json --w_mil 0 --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1
```

raw HN 的失败不能完全否定 hard-negative 思路，因为数据审计确认 raw map 中有大量跨视频重复/近重复 caption，会把语义近似正例当成强负例。当前更合理的判定实验是只跑一组 clean-HN packing：

```bash
RUN_DATE=20260622 RUN_TIME=hn_pack_clean_wmil0_repeat1_4gpu_b64 CUDA_VISIBLE_DEVICES=1,2,3,4 EXPERIMENT_DESC="Clean HN packing, 4GPU global batch=64, w_mil=0" bash run_train_msrvtt_bg.sh --use_hard_negative_packing --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json --w_mil 0 --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1
```

截至 2026-06-22，GPU 被占满，clean-HN 实验尚未启动。

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
modeling_mulit.py: WTI logits 直接作为检索分数 (当前工作树: B1-only，无 confidence 加权)
```

### 3.2 关键代码位置

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| EvidentialUncertaintyHead | `query_models/module_sap.py` | 27–60 | 仅含 Dirichlet layer |
| SAP.__init__ | 同上 | 63–91 | anchor_tokens + decoder + head + anchor_proj |
| SAP.forward | 同上 | 93–156 | B1 detach + 非学习不确定性 |
| WTI retrieve_logits | `modules/modeling_mulit.py` | ~660–662 | `weighted_logits = wti_logits`，无 confidence 加权 |
| diag 链 | 同上 | ~663–696 | pos/neg/gap + u_mode/epistemic/logsigma |
| Chain-Prob 日志 | `main_task_retrieval.py` | ~737–745 | 当前仅记录 u_mode / epistemic / var_t / kl_t |

### 3.3 已删除/废弃的组件

- **NIG 层**（γ/ν/α/β）：方案 C 标量化后移除
- **EvidentialUncertaintyHead（重复副本）**：`prob_models/uncertainty_module.py` 已删除
- **uncertainty_reg_loss / nig_mil 分支**：已删除
- **per-pair confidence_mlp**：`fe5e744` 中验证失败，2026-06-17 工作树已移除
- **视频侧 PIENet**：已移除

---

## 四、核心发现：置信度加权对 R@1 的贡献为零

### 4.1 Exp 1 的 50.0 来自什么

Exp 1 的 50.0 不能归因于置信度加权；更合理的归因是当时的 **NIG gamma + detach** 表征路径带来了 +0.6。

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

### 4.4 B1-only v2（当前工作树）

2026-06-17 移除 per-pair `confidence_mlp` 后，当前架构退回到 **WTI logits 直接检索**：

| 实验 | 架构 | confidence | T2V R@1 | ret_gap |
|------|------|------------|:---:|:---:|
| Exp 1 | NIG gamma + anchors.detach() | per-video unsq(1)，T2V 实际抵消 | **50.0** | ~18 |
| Baseline pure sim | NIG gamma，无 detach | 无 | **49.4** | — |
| B1only_v2 repeat1 | anchor_proj + projected.detach() | 无 | **49.3** | 16.7 |

这说明 **B1 detach 的收益依赖具体 SAP 表征路径**。Exp 1 中 `NIG gamma + anchors.detach()` 的组合曾带来 +0.6，
但当前 `anchor_proj + projected.detach()` 复现到 49.3，仍未达到历史 50.0，也没有超过历史 baseline 49.4。

### 4.5 确定性结论

**置信度加权路径（per-video / per-pair / 熵 / MLP）已彻底探索完毕，均不能提升 T2V R@1。**

原因：不确定性/置信度信号中包含的"噪声/校准"信息与 softmax 排序所需的"判别"信息是正交的。在对比学习框架中将不确定性用于排序权重，本质上是让一个没有正负区分度的信号去调节正负分数——无论怎么设计映射函数，都无法让无区分度的输入产生有区分度的输出。

---

## 五、B1 梯度隔离：收益需限定

Exp 1 中的现象：

```
NIG gamma + anchors.detach(): 49.4 → 50.0（+0.6）
```

当前架构中的复验：

```
anchor_proj + projected.detach(): 49.4 → 49.3（-0.1，相对历史 baseline）
```

因此不能再把 B1 detach 写成跨架构稳定收益。更严谨的结论是：

- detach 可以切断不确定性头到 SAP decoder 的梯度冲突；
- 但它是否提升 R@1，取决于均值表征路径是 `NIG gamma` 还是当前 `anchor_proj`；
- 2026-06-19 的 `21348d1` repeat2 只到 49.1，说明历史 50.0 至少不是稳定复现结果；
- 继续追 50.0 的收益不如转向能直接增强主检索分母的 hard-negative batch packing。

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
| MIL_loss | 0.01（clean-HN 判定实验中设为 0） | ⚠️ 饱和，退化；clean-HN 首轮关闭以保持归因干净 |
| orth_loss | 0.1 | ✅ Epoch 1 消解 |
| evidential_loss | 0 | ❌ 已关闭 |
| neg_reg_loss | 0 | ❌ 已关闭 |

训练命令模板：
```bash
EXPERIMENT_DESC="<desc>" CUDA_VISIBLE_DEVICES=<gpus> bash run_train_msrvtt_bg.sh --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1
```
`--batch_size=64` 表示全局有效 batch=64；4GPU 时每卡 micro-batch=16。

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
| `fe5e744` | per-pair confidence_mlp + detach | 48.6 |
| 未提交 | B1-only v2 repeat1：移除 confidence_mlp，WTI 直接检索 | 49.3 |

---

## 九、下一步建议

### 9.1 推荐（优先级排序）

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 🟢 P0 | **clean hard-negative batch packing, `w_mil=0`** | raw HN 因映射质量问题失败；clean map 是下一组唯一应跑的判定实验 |
| 🟡 P0/P1 | **clean hard-negative batch packing, `w_mil=0.01`** | 仅当 clean `w_mil=0` 明显回升到约 49.0 或更高时再跑 |
| 🟡 P1 | **显式 hard negative loss** | 仅在 batch packing 提升不明显时启动，代价是额外编码与显存 |
| 🟡 P1 | **优化 WTI 表征质量** | R@1 当前主要由 WTI CrossEn 驱动，优先提升主判别路径 |
| ⚪ P2 | **不确定性感知数据增强** | 工程风险更高，放在 hard-negative 路径之后 |
| ⚪ P2 | **超参自动搜索** | 架构稳定后启动 |

### 9.2 不建议

- ❌ 任何形式的置信度加权（per-video / per-pair / entropy / MLP）
- ❌ batch 标准化 / z-score / β sweep
- ❌ 在当前 anchor_proj 主线上盲目重引入 NIG；`21348d1` repeat2 未复现 50.0 后，该路径降级
- ❌ 大改 batch_size（破坏实验对照）
- ❌ 继续使用 raw hard-negative 映射直接训练；raw map 已发现明显假负例/近重复问题

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
| `modules/modeling_mulit.py` (~920 行) | 主模型：forward + loss + diag；当前工作树为 B1-only |
| `query_models/module_sap.py` (157 行) | SAP + EvidentialUncertaintyHead（仅 Dirichlet） |
| `prob_models/uncertainty_module.py` | 文本侧不确定性头 |
| `main_task_retrieval.py` | 训练/评估入口 |
| `scripts/build_msrvtt_hard_negatives.py` | 构建 raw MSRVTT query hard-negative 映射 |
| `scripts/audit_msrvtt_hard_negatives.py` | 审计 raw HN 映射并生成 clean HN 映射 |
| `dataloaders/hard_negative_sampler.py` | hard-negative batch packing sampler |
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
logs/20260617/  — B1only_v2 repeat1 (49.3) + baseline/Exp1 repeat2（均未到 50）
logs/20260619/  — raw HN packing w_mil=0 (Best T2V R@1=48.1，失败)
logs/20260622/  — clean-HN 实验尚未启动；若有日志需先确认是否为真实训练日志
```

---

## 十一、给下一个会话的摘要

**一句话**：不确定性置信度加权（per-video、per-pair、entropy、MLP 全路径）
对 T2V R@1 贡献为零，已彻底验证。B1 detach 曾在 Exp 1 的 NIG gamma 路径中达到 50.0，
但 2026-06-19 三组复现实验均未到 50；当前最好结果是 B1only_v2 repeat1 的 49.3。raw HN packing 跑到 48.1，失败；raw map 已审计并清洗，clean-HN 判定实验等待 GPU 空闲。

**当前代码状态**：工作树未提交改动已移除 per-pair `confidence_mlp`，`modules/modeling_mulit.py` 中 `weighted_logits = wti_logits`，即 B1-only v2。hard-negative mapping 脚本、audit/clean 脚本与 batch packing sampler 已实现；`--hard_negative_path` 默认值已切到 `cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json`，但 packing 默认关闭，仍需显式传 `--use_hard_negative_packing`。

**推荐起点**：
1. GPU 空闲后启动 clean hard-negative batch packing 首轮实验，验证 `w_mil=0`，避免概率采样支路干扰归因
2. 若 clean-HN `w_mil=0` 明显回升到约 49.0 或更高，再测试 `w_mil=0.01`
3. 若 batch packing 无明显提升，再考虑显式 hard negative loss
