# SAP 不确定性坍缩 —— 排查与路 1+3 修复

**日期**：2026-08-11
**范围**：诊断 SAP (Semantic Anchor Probing) per-video 不确定性信号的坍缩现象，在代码层面定位根因，实施最小侵入式的结构性修复。
**用途**：论文写作参考（方法 / 局限 / 消融章节可直接引用）。

---

## 1. 执行摘要

- **症状**：SAP 输出的 4 个不确定性信号（`logsigma_video`, `epistemic_cont`, `modal_probs`, `u_mode`）在训练后全部坍缩为 per-video 常数。`u_video` 与真实检索失败的相关系数 ρ ∈ [−0.08, +0.02]（所有 ckpt 测得），远低于论文自身的决策阈值 ρ ≥ 0.2。
- **根因**：`query_models/module_sap.py` 与其 loss 组合中的 4 条机制彼此叠加造成坍缩（详见 §3）。
- **修复（路 1 + 路 3）**：(a) 移除 `evidential_head` 前的 `.detach()`，让 `modal_probs` 获得 per-video 梯度；(b) 用独立的 `logsigma_head(mu_agg)` MLP 替代 `log(var(anchors))` 这个结构性瓶颈；(c) 把 `w_orth` 从 0.1 降到 0.03、把 `log_sigma_min` 从 −1.5 拓宽到 −6.0。
- **成本**：单文件约 20 行代码改动 + 1 次训练（约 3h，从 June ckpt warm-start）。
- **状态**：代码已提交并通过 smoke test，训练已于 2026-08-11 19:44 启动（日志 `logs/20260811/194405_train_msrvtt.log`）。

---

## 2. 修复前的诊断证据

测量对象：`ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4`（June UATVR 架构，MSRVTT JSFUSION test R@1 = 50.0）以及 CRF 训练变体。所有数值在 MSRVTT JSFUSION 1000 视频测试集上聚合。

### 2.1 每视频的不确定性信号全部退化为常数

| 信号 | 实测分布 | 设计意图 |
|---|---|---|
| `logsigma_video` | `−1.5 ± 0`（触底 clamp min） | per-video Gaussian 方差，供 MIL 采样 |
| `epistemic_cont` | `0.9995 ± 5e-5`（接近上限） | per-video epistemic 不确定性，用于检索时置信折扣 |
| `diversity`（anchor 空间） | `0.9996 ± 5e-5`（饱和） | 视频内容复杂度代理 |
| `modal_entropy`（归一化） | `1.0000 ± 0`（完全均匀） | 模态数量代理 |
| `u_mode` | `0.5794 ± 2e-5` | Dirichlet 离散不确定性 |
| `exp(-epistemic)` 折扣 | `0.368 ± 1.8e-5` | 检索得分调制系数 |

**解读**：模型对每个视频输出相同的不确定性。所有信号都不携带"哪些视频难检索"的信息。

### 2.2 CRF ckpt 演进过程中 ρ(不确定性, 检索难度)

| ckpt | T2V R@1 | ρ(u, T2V rank) | ρ(u, V2T rank) | ρ(u, max rank) |
|---|---:|---:|---:|---:|
| bin.0（audit 最佳） | 8.8 | +0.083 | +0.039 | +0.065 |
| bin.1 | 11.9 | +0.052 | +0.021 | +0.044 |
| bin.2 | 12.0 | +0.022 | +0.010 | +0.016 |
| bin.3 | 12.7 | +0.003 | +0.023 | +0.013 |
| bin.4 | 12.9 | +0.004 | +0.013 | +0.015 |

**解读**：CRF 训练**主动抹掉**了初始阶段仅存的一点相关性。u_video 标准差在增大（0.16 → 0.20），但 ρ 反而降到零——方差的增长方向与"检索难度"无关。

所有数值均在论文自身决策规则的阈值以下（`ρ ≥ 0.2` 才允许保留 uncertainty claim）。

---

## 3. 根因分析（代码级）

数据流：`video_features [B, T×S, D] → SAP → { anchors, mu_raw, logsigma, epistemic_cont, modal_probs, u_mode, alpha_dir }`

4 条机制彼此叠加，共同造成坍缩。

### 3.1 根因 A：`.detach()` 阻断 Dirichlet head 学习 per-video 权重

位置：`query_models/module_sap.py` line 116（修复前）

```python
# 修复前
projected = self.anchor_proj(anchors)          # [B, K, D]
ev = self.evidential_head(projected.detach())  # ← 梯度被阻断
```

Dirichlet 层（`Linear(D, 1)` 输出 `alpha_dir`）拿到的是 detached 输入，通过 `alpha_dir` 反传到 decoder 和 anchor tokens 的梯度通路被截断。结果：`alpha_dir` 收敛到 anchor 特征的 batch-independent 均值 → `modal_probs = 1/K`（每个视频都 uniform）→ `u_mode` 和 `modal_entropy` 都变常数。

### 3.2 根因 B：`log(var(anchors))` 让 logsigma 与 orth_loss 几何耦合

位置：`query_models/module_sap.py` line 141–145（修复前）

```python
# 修复前
anchor_dim_var = torch.var(anchors, dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
logsigma = torch.log(anchor_dim_var + 1e-8)  # [B, 1]
logsigma = torch.clamp(logsigma, min=-1.5, max=4.0)
logsigma = logsigma.expand(-1, self.d_model)                              # [B, D]
```

`logsigma` 完全由 16 个 anchor 向量沿 K 轴的方差决定。但 `w_orth=0.1` 损失（`||QQ^T − I||²`）显式地把 anchors 推向每个视频都单位正交。维度为 512 的正交单位向量满足 `var(dim=K) ≈ 1/512 ≈ 1.95e-3`，因此 `log(var) ≈ −6.24`，被 `log_sigma_min = −1.5` clamp 到 −1.5。不确定性信号变成 orth_loss 几何的确定性函数，与视频无关。

### 3.3 根因 C：`diversity = mean(1 − cos(anchor_i, anchor_j))` 饱和

位置：`query_models/module_sap.py` line 123–126

`diversity` 设计上作为 anchor 分散度代理。但 `w_orth=0.1` 会把 `cos(anchor_i, anchor_j) → 0`（i ≠ j），于是 `diversity → 1.0` 对所有视频都成立。与根因 B 是同一个几何机制，只是换成 cosine 表达。

### 3.4 根因 D：不确定性的训练信号事实上为零

位置：June ckpt 的 `training_args.json` + `modules/modeling_mulit.py` 中的 loss 组合

- `w_evidential = 0.0`（config）：evidential NLL 损失（作用在 `alpha_dir` 上）乘以 0 → Dirichlet head 从这条通路拿不到任何梯度。
- `w_mil = 0.01`（config）：MIL 损失从 `N(mu_raw, exp(logsigma))` 采样做 batch 内对比。这是 `logsigma` **唯一**的梯度通路，且系数极小。
- 代码库中没有任何 loss 告诉模型"这个视频难检索，所以它的 `logsigma` 应该更大"。

结果：即使去掉根因 A/B/C，per-video 不确定性的训练信号本身就弱到不足以起作用。

### 3.5 坍缩信号的下游消费者

作为完整性参考（源自 `modules/modeling_mulit.py:697–861`）：

| SAP 输出 | 下游消费者 | Loss 系数 |
|---|---|---:|
| `anchors` | orth_loss（自消费） + `confidence_mlp`（detach 后消费） | `w_orth = 0.1` |
| `mu_raw` | MIL Gaussian 采样；evidential 相似度 `ev_sim` | `w_mil = 0.01` |
| `logsigma` | MIL Gaussian 采样；diagnostic 日志 | `w_mil = 0.01` |
| `epistemic_cont` | `ev_sim = cos(mu, text) × exp(-epistemic)` → evidential NLL | `w_evidential = 0.0` |
| `alpha_dir` | evidential NLL loss | `w_evidential = 0.0` |
| `modal_probs` | 内部（加权聚合 `mu_raw`） | — |
| `u_mode` | 仅 diagnostic 日志 | — |

**SAP 不确定性的 4 条 loss 通路中，有 2 条（`w_evidential`）被 config 显式关闭。**MIL 是唯一活跃的信号，且权重很小。

---

## 4. 修复方案 —— 路 1 + 路 3

### 4.1 路 1 —— 结构性修改（`query_models/module_sap.py`）

**修改 1a**：移除 `.detach()`，让 evidential head 通过正常梯度流训练。

```python
# 修改前
ev = self.evidential_head(projected.detach())
# 修改后
ev = self.evidential_head(projected)
```

**修改 1b**：用独立的 per-video MLP head 替代 `log(var(anchors))`。

```python
# __init__ 中新增模块
self.logsigma_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.SiLU(),
    nn.Linear(d_model // 2, d_model),
)
nn.init.zeros_(self.logsigma_head[-1].weight)
nn.init.constant_(self.logsigma_head[-1].bias, -1.5)
```

**修改 1c**：替换 logsigma 计算逻辑。

```python
# 修改前
anchor_dim_var = torch.var(anchors, dim=1).mean(dim=-1, keepdim=True)
logsigma = torch.log(anchor_dim_var + 1e-8)
logsigma = torch.clamp(logsigma, ...).expand(-1, self.d_model)
# 修改后
mu_agg = (modal_probs.unsqueeze(-1) * projected).sum(dim=1)  # [B, D] 未归一
mu_raw = mu_agg / (mu_agg.norm(dim=-1, keepdim=True) + 1e-9)
logsigma = self.logsigma_head(mu_agg)                        # [B, D] per-video per-dim
logsigma = torch.clamp(logsigma, min=self.log_sigma_min, max=self.log_sigma_max)
```

**设计决策**：
- 喂给 `logsigma_head` 的是 `mu_agg`（**未归一**）——L2 归一后所有视频都在单位球上，per-video 幅度信息丢失。
- 最后一层 Linear 的 weight 初始化为 0、bias 初始化为 −1.5 → 初始阶段 `logsigma ≡ −1.5`，与修复前的 clamp 值完全对齐。MIL 训练动态无冲击，per-video 差异会随 MLP 权重适应而逐步 emerge。
- `logsigma_head` 增加约 197K 参数（Linear 512→256 → SiLU → Linear 256→512），< 0.05% 模型总参数，未触及"新模块 > 100 行"审查门槛。

### 4.2 路 3 —— 超参调整（CLI 传入，无代码改动）

- `--w_orth 0.03`（原 0.1）：松绑正交约束，anchor 方差不再被几何 pinning。
- `--log_sigma_min -6.0`（原 −1.5）：拓宽 clamp 下界，允许 `logsigma_head` 在需要时表达小方差。

保持不变：
- `--log_sigma_max 4.0`
- `--w_mil 0.01`（仍是 `logsigma` 的主要训练信号）
- `--w_evidential 0.01` 默认（未显式传 0，隐式开启）

### 4.3 被否决的方案

| 路线 | 否决理由 |
|---|---|
| 路 2 —— 加显式检索难度监督（MSE(logsigma, margin)） | 需要约 30 行代码 + 存在 loss 冲突风险；作为路 1+3 若未把 ρ 抬到 ≥ 0.2 的兜底方案。 |
| 路 B2 —— 用 CRF ladder 做监督 | 已通过 `crf_head` 试过；产出 audit = 1.0 但 ρ(u_video, retrieval_failure) = −0.08。压缩退化与检索难度是正交信号。 |
| 路 C1 —— Ensemble variance | 单次实验 5–10 倍算力，超出预算。 |
| 路 D —— 接受坍缩、focus 于 `crf_head` | 用户明确否决："不能承认 collapse，我要想办法改善"。 |

---

## 5. 验证方案

### 5.1 Smoke test 结果（2026-08-11 19:20）

用随机输入 `[B=4, T×S=1576, D=512]`、`log_sigma_min=-6, log_sigma_max=4` 做前向传播：

- 输出 shape：全部符合设计（`logsigma: [B, D]`、`modal_probs: [B, K]` 等）
- 初始 `logsigma` = −1.5，所有视频一致（bias init 生效）
- `loss.backward()` 后梯度流：
  - `logsigma_head` grad norm = 128（新 head 收到梯度）
  - `evidential_head` grad norm = 164（修复前因 `.detach()` 为 0）✅
  - `anchor_proj`、decoder、anchor_tokens 均非零

### 5.2 训练期需监控的指标

| 指标 | 位置 | 目标 |
|---|---|---|
| `logsigma_v_mean` | diag log `_aux_chain` | 随 epoch 从 −1.5 漂移 |
| `logsigma std_across_B` | 需在 diag 加一行 | epoch 3 前 > 0 |
| `orth_loss × w_orth` | diag `_aux_chain` | 收敛且不爆炸 |
| per video `modal_probs` 熵 | 需在 diag 加一行 | epoch 3 前 < log(16) = 2.77 |
| val R@1 (trusted-v1) | eval log | ≥ 48（baseline 50，容忍下降 2 分） |

### 5.3 训练后评估

| 测试 | 命令 / 脚本 | 通过标准 |
|---|---|---|
| JSFUSION T2V R@1（无 DSL） | `main_task_retrieval.py --do_eval` + 最终 ckpt | ≥ 48 |
| JSFUSION T2V R@1（DSL T=2.0） | 同上 + DSL 重排 | ≥ 52 |
| ρ(u_video, retrieval_hardness) | `scripts/diagnose_uncertainty_retrieval_correlation.py` | ≥ 0.2 |
| CRF audit（若 crf_head 另行训练） | `scripts/evaluate_crf_ladder.py` | `three_pair_accuracy ≥ 0.9` |

兜底：若本次 ρ < 0.2，升级到路 2（加显式难度监督）。

---

## 6. 修复前 baseline ckpt 对比表

用于论文消融章节。

| Ckpt | 架构 | 训练 | T2V R@1 | V2T R@1 | audit `three_pair_accuracy` | 耗时 |
|---|---|---|---:|---:|---:|---:|
| June `bin.4` | 完整 (sap+spatial+ada_norm) | 5 ep, 无 CRF | 50.0 | 46.3 | 不适用（无 crf_head） | reference |
| June `bin.4` + DSL T=2.0 | 同上 | 仅 eval | **52.6** | **52.5** | 不适用 | 0h |
| Hybrid ckpt | June arch + 移植 crf_head | 0 ep（权重手术） | **52.6** | 52.5 | **1.0000** | 0h |
| Step 5 v2 `bin.4` | June arch（fresh restore） | 5 ep, 无 CRF, fresh | 51.x | 51.x | 不适用 | 5h |
| Step 6 `bin.x` | June arch + CRF | 1.4h CRF 训练 | 46.x | 48.x | 1.0000 | 1.4h |
| HEAD arch + CRF `bin.4` | 简化架构 | 5 ep CRF | 12.9 | 16.5 | 1.0000 | 5h |

**目前最佳交付物**：Hybrid ckpt（权重手术）。同时满足两项目标（R@1 ≥ 52、audit ≥ 0.9），训练时间 0。

**路 1+3 本次运行的目的**：不是为了提升 R@1（已达标），而是测试 SAP 能否产生 ρ ≥ 0.2 的 per-video 不确定性信号——这才能解锁论文的**核心** uncertainty claim。

---

## 7. 论文叙述路径

### 7.1 不需要本次修复即可成立的 claim（已验证）

- SAP 作为**多锚点多样池化**用于检索 → R@1 = 50 baseline，DSL +2.6，MSRVTT 上与 UATVR-official 有竞争力。
- 通过 `crf_head` 实现**压缩鲁棒探测**（compression-robust probing） → `three_pair_accuracy = 1.0000`、`challenge_delta_ci_low = 0.044`、MSVD 跨域迁移成立。
- SpatialEnhancer + UncertaintyAdaNorm 作为架构贡献 → 消融待补。

### 7.2 依赖本次路 1+3 成功的 claim

- SAP 作为**有原则的 per-video 不确定性估计器**、有 retrieval-difficulty grounding（ρ ≥ 0.2）。这正是当前实验要验证的东西。

### 7.3 若路 1+3 成功（目标场景）

- 论文可以呈现 SAP 产生 3 个互补的不确定性信号，各有独立解读：
  1. **`modal_probs`** —— 对语义锚点的 soft attention 分布，per-video，非 uniform。
  2. **`logsigma_video`** —— MLP head 输出的 per-video 方差，通过 MIL（可选加 evidential NLL）训练。
  3. **`epistemic_cont`** —— anchor-space 多样性 × 模态熵，现在有 per-video 变化。
- 消融矩阵：(a) 完整修复、(b) 仅去 `.detach()`、(c) 仅加独立 `logsigma_head`、(d) 仅松 `w_orth`。分离每个机制的贡献。

### 7.4 若路 1+3 失败（兜底场景）

- SAP 不确定性挪到 *Limitations* / *Analysis* 节：*"我们设计了有原则的 per-video 不确定性 pipeline；消融揭示 anchor 正交正则与 detached evidential 梯度共同饱和了不确定性信号（§3）。路 1+3 修复恢复了梯度流并独立参数化了 logsigma，但未能把 ρ 抬到 0.2 以上，说明 anchor-space 几何不是 retrieval-difficulty uncertainty 的合适信号源。我们把直接的难度监督（路 2）留作 future work。"*
- `crf_head` 保持为主要不确定性故事（compression-robust probing）。
- R@1 故事依然成立（SAP 作为池化）。

---

## 8. 可复现性

### 8.1 代码状态

- 分支：`199`
- 修改文件：`query_models/module_sap.py`（单文件，约 15 行改动 + 8 行新增）
- 未改动：其他所有文件（June arch、CRF pipeline、训练循环）

### 8.2 训练命令

```bash
EXPERIMENT_DESC="sap_route1_plus_3_no_crf_warm_from_june" \
CUDA_VISIBLE_DEVICES=0,2,3,4 \
MASTER_PORT=29551 \
bash run_train_msrvtt_bg.sh \
  --use_uatvr_full --use_ada_norm \
  --strategy 2 \
  --uncertainty_text_head text \
  --log_sigma_min -6.0 --log_sigma_max 4.0 \
  --w_mil 0.01 --w_orth 0.03 \
  --init_model /data2/hxj/project/UATVR/ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4
```

Effective batch = 256（4 GPU × bs=64），5 epoch，预计 wall time ~3h。

### 8.3 日志与 ckpt 位置

- 训练日志：`logs/20260811/194405_train_msrvtt.log`
- Ckpt 目录：`ckpts/ckpt_msrvtt_<RUN_ID>/`（自动加时间戳）
- 诊断脚本：
  - `scripts/evaluate_crf_ladder.py`（CRF audit）
  - `scripts/diagnose_uncertainty_retrieval_correlation.py`（ρ 测量）

### 8.4 已有产物

- Hybrid ckpt（权重手术）：`/tmp/hybrid_ckpt/pytorch_model.bin.hybrid`
- Hybrid audit report：`/tmp/hybrid_audit/crf_audit_report.json`
- June 参考 ckpt：`ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4`
- CRF 供体 ckpt：原始位置 `ckpts/ckpt_msrvtt_20260808_.../pytorch_model.bin.0`（供 `crf_head` 移植）

---

## 9. 待跟进的开放问题

1. 是否要往 diag 日志加一行 per-batch `logsigma std_across_B`，实时观察 per-video 差异 emerge？（`modeling_mulit.py:820–832` 一行改动。）
2. 是否要跑一个并行的"路 1+3 + CRF"联合训练，让 SAP 不确定性和 CRF audit 同时训好？（额外约 3h，能统一两条不确定性叙事。）
3. 若 ρ 只涨到约 0.15（边界值），是否上路 2（retrieval-hardness supervision）做第三层修复，还是就此打住、用边界数值 + 诚实讨论写论文？
4. 单独消融 1a（去 `.detach()`）vs 单独消融 1b（独立 logsigma_head）—— 到底哪一个才是关键变量？

以上问题等路 1+3 结果落地后再决定。
