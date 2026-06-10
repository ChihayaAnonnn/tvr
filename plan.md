# UATVR 待验证方向

## 一、当前状态（已验证）

### 架构

- 双分支设计：Base Branch（视频帧→SAP→WTI）+ Query Branch（属性→QueryFormer→text-text WTI）→ MoE 融合
- SAP 为 Evidential 版本（Dirichlet + NIG 双层不确定性），无旧变体残留
- 视频侧 PIENet 已移除，由 SAP 的 Dirichlet 模态概率聚合替代

### 已实现的 Loss 项

| Loss | 权重参数 | 默认值 | 作用 |
|------|----------|--------|------|
| sim_loss (CrossEn) | — | 基础 | 对称对比学习 |
| MIL_loss | `--w_mil` | 1.0 | 多实例学习，正则化锚点分布 |
| evidential_loss | `--w_evidential` | 1e-2 | Evidential NLL，驱动不确定性学习 |
| neg_reg_loss | `--w_neg_reg` | 5e-2 | 负证据正则化，防止平凡解 |
| uncertainty_reg_loss | `--w_uncertainty_reg` | 1e-3 | DUQ 风格 evidential 矩阵正则化（detached，仅监控） |
| orth_loss | `--w_orth` | 0.1 | 锚点正交性，防止退化 |

### 已验证的实验结论

- TQFS（帧质量采样）有效
- Evidential uncertainty 优于 LogNormal（理论分析支持）
- `w_evidential=1e-2` + `w_neg_reg=5e-2` 是合理区间
- 锚点数量 16 是当前默认值

### 已知问题

- 最近一次 eval（2026-06-05）崩溃：`uncertainty_mode=wti_confidence` 的 eval 路径有 tensor shape mismatch bug
- `--num_queries` 在 logging 中被引用但未定义为 argparse 参数，硬编码默认值 16

---

## 二、待验证方向

### 方向 A：多粒度视频特征

**动机**：当前仅用 CLIP 单一尺度特征（ViT-B/16, 8 帧），时序信息不足。

**方案**：
- 3D Conv 后处理：在 CLIP visual encoder 之后加 3D 卷积捕捉帧间运动
- SlowFast 双路径：低帧率语义流 + 高帧率运动流

**验证方式**：对比仅 SAP baseline 的 T2V R@1 变化

**预期收益**：R@1 +1~2（时序建模是视频检索核心瓶颈之一）

**优先级**：中。实现成本较高，但与 SAP 正交，可独立验证。

---

### 方向 B：文本侧语义锚点

**动机**：当前 Query Branch 对文本的建模依赖 CLIP text encoder + QueryFormer，未引入类似 SAP 的结构化锚点机制。

**方案**：为文本端增加可学习语义锚点（动作锚点、实体锚点），与 SAP 视频锚点对称。

**验证方式**：对比 text-only branch 的 WTI 质量变化

**预期收益**：提升 text-video 对齐精度，尤其在长文本/复杂描述场景

**优先级**：中。需设计文本锚点的语义空间，避免与 CLIP text encoder 功能重叠。

---

### 方向 C：跨视角 KL 散度约束

**动机**：Query Branch 和 Base Branch 各自产生分布，但缺乏一致性约束。

**方案**：添加 `--w_cross_view_kl` 参数，对两个分支的锚点分布施加 KL 散度惩罚。

**验证方式**：控制其他 loss 不变，仅加 KL 项，观察 R@1 和 gate score 变化

**预期收益**：改善双分支对齐，减少 MoE 融合的学习负担

**优先级**：低。需先确保两个分支各自稳定。

---

### 方向 D：不确定性驱动的关键帧选择

**动机**：当前 8 帧等间隔采样，未利用 SAP 输出的帧级不确定性。

**方案**：训练后用不确定性排序帧，选择 top-k 低不确定性帧重新计算相似度。

**验证方式**：在 eval 阶段实现，对比 uniform vs uncertainty-weighted 的 R@1

**预期收益**：R@1 +0.5~1（过滤噪声帧）

**优先级**：低。实现简单但依赖不确定性校准质量。

---

### 方向 E：不确定性课程学习

**动机**：不确定性正则项从 epoch 0 就参与训练，可能在早期干扰主 loss 优化。

**方案**：添加 `--anneal_warmup_epochs N` 参数，前 N 个 epoch 不启用 evidential/uncertainty loss。

**验证方式**：对比 N=0 vs N=2 的训练曲线和最终 R@1

**预期收益**：训练更稳定，可能提升最终精度

**优先级**：高。参数已存在（`--anneal_warmup_epochs`，当前默认 0），只需跑对比实验。

---

### 方向 F：Top-K 锚点过滤

**动机**：SAP 输出 16 个锚点，但不是所有锚点都对检索有贡献。

**方案**：添加 `--sap_top_k` 参数，在 WTI 计算时仅使用 gate score 最高的 top-k 锚点。

**验证方式**：sweep k ∈ {4, 8, 12, 16}，观察 R@1 变化

**预期收益**：过滤噪声锚点，可能提升精度并加速推理

**优先级**：高。实现简单，直接在 `get_similarity_logits()` 中加 top-k masking。

---

## 三、建议执行顺序

1. **先修 bug**：修复 `wti_confidence` eval 路径的 shape mismatch
2. **E（课程学习）**：成本最低，`--anneal_warmup_epochs` 已有参数，跑 2 组对比即可
3. **F（Top-K 过滤）**：实现简单，快速验证锚点质量
4. **A（多粒度特征）**：收益潜力大，但实现成本高
5. **B（文本锚点）**：与 A 有交互，建议 A 完成后再做
6. **C、D**：作为补充实验，在上述方向有结论后执行
