# RSPR：重参数化随机原型排序模块设计

## 1. 文档定位

本文定义第一阶段概率嵌入与不确定性模块 **RSPR（Reparameterized
Stochastic Prototype Ranking）**。模块以 UATVR 的 CLIP—DSA—词/帧匹配链路
为确定性主干，通过重参数化概率嵌入、双向随机原型匹配和随机排序损失，提高
文本—视频表征及困难候选区分能力。

本阶段优先回答两个问题：

1. 重参数化概率表示能否稳定提高标准文本—视频检索性能；
2. 从随机原型匹配中得到的配对不确定性，能否帮助模型区分可信困难负样本和
   语义歧义候选。

设计依据包括：

- [UATVR 模块设计报告](../../reference/uatvr_uncertainty_adaptive_text_video_retrieval_design_report.md)；
- [DUQ 模块设计报告](../../reference/duq_dual_uncertainty_quantification_text_video_retrieval_design_report.md)；
- [概率嵌入与不确定性研究报告](../../deep-research-report.md)。

## 2. 背景与设计缺口

UATVR 使用高斯概率嵌入和多实例对比损失增强训练，但实际推理仍以 DSA
确定性分数为主。DUQ 同时建模 intra-pair similarity uncertainty 和 inter-pair
distance uncertainty，但仓库实现主要使用三路分数平均，证据量没有形成独立、
经过校准的推理置信度。

直接串接两套完整模型存在以下问题：

- 两边的概率参数、采样数、特征粒度和最终分数语义不统一；
- UATVR 将同源的全部随机样本组合作为正集合，可能强制对齐不对应的语义模式；
- DUQ 的硬 min/max 边界容易被极端随机样本主导；
- 模态自身的固定方差不能完整表达查询—候选特有的映射不确定性；
- 一次随机采样得到的分数可能造成评估排名抖动；
- 若同时加入 PCM、三路匹配、证据损失、一致性 KL 和新概率模块，将难以形成
  可解释的消融闭环。

RSPR 因此采用“稳定确定性主干 + 可独立验证的概率旁路”。第一版只修改概率
表示和排序监督，不改数据协议、CLIP 主干类型、DSA 结构及标准检索指标。

## 3. 目标与非目标

### 3.1 目标

- 在 DSA 更新后的文本 token 和视频帧 token 上学习对角高斯概率嵌入；
- 将重参数化采样直接接入跨模态匹配与排序损失，而非只作为辅助数据增强；
- 用平滑双向匹配替代“全部样本强制对齐”与“单个极值决定边界”；
- 输出查询—候选级概率分数和配对不确定性矩阵；
- 用随机排序损失优化正候选压过困难负候选的概率；
- 在核心版本稳定后，用 Pairwise Beta evidence 识别可信困难负样本；
- 保持均值向量可用于 ANN 召回，概率计算只作用于训练 batch 或 Top-R 精排。

### 3.2 非目标

第一阶段不包含：

- DUQ Patch Compression Module；
- 多模型 ensemble 或严格 epistemic uncertainty 分解；
- conformal prediction、动态候选集合或拒绝策略；
- 主动追问、Video-LLM 决策与开放集训练；
- 视频时刻边界预测；
- 全库逐候选多次随机采样；
- 对现有 UATVR/DUQ 全部损失的并集式融合。

这些内容只有在 RSPR 核心版证明有效后，才进入独立规格和消融。

## 4. 总体架构

~~~text
文本 token / 视频帧
        │
        ▼
CLIP + UATVR DSA
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
确定性双向词—帧匹配                Mask-aware 统计聚合
        │                         ├─ 中心语义 h
        │                         ├─ token 离散度 d
        │                         └─ 注意力熵 H(a)
        │                               │
        │                         μ head / logvar head
        │                               │
        │                   z = normalize(μ + σ ⊙ ε)
        │                               │
        │                      K 个文本/视频随机原型
        │                               │
        │                  双向 Soft Prototype Matching
        │                         ├─ S_prob
        │                         └─ U_pair
        │                               │
        └──────────────┬────────────────┘
                       ▼
                 Stochastic Rank Loss
                       │
              Pairwise Beta Evidence
                 （核心稳定后启用）
                       │
                       ▼
        S_final = S_det + λ_p S_prob + λ_e S_evid
~~~

## 5. 输入输出与张量契约

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| <code>T</code> | $B_t\times L\times D$ | DSA 更新后的文本 token |
| <code>T_mask</code> | $B_t\times L$ | 有效文本 token mask |
| <code>V</code> | $B_v\times F\times D$ | DSA 更新后的视频帧 token |
| <code>V_mask</code> | $B_v\times F$ | 有效视频帧 mask |
| <code>S_det</code> | $B_t\times B_v$ | UATVR 确定性匹配分数 |
| <code>pair_id</code> | $B_t,B_v$ | 显式多正样本关系标识 |
| <code>mu_t/logvar_t</code> | $B_t\times D$ | 文本分布参数 |
| <code>mu_v/logvar_v</code> | $B_v\times D$ | 视频分布参数 |
| <code>Z_t</code> | $B_t\times K\times D$ | 文本随机原型 |
| <code>Z_v</code> | $B_v\times K\times D$ | 视频随机原型 |
| <code>C</code> | $B_t\times B_v\times K\times K$ | 原型两两余弦相似度 |
| <code>S_prob</code> | $B_t\times B_v$ | 双向概率匹配分数 |
| <code>U_pair</code> | $B_t\times B_v$ | 查询—候选配对不确定性 |
| <code>alpha/beta</code> | $B_t\times B_v$ | 可选 Beta evidence 参数 |

所有检索监督必须依据 <code>pair_id</code> 构造正集合，不永久假设 batch 对角线
是唯一正配对。

## 6. 核心模块

### 6.1 Mask-aware 统计聚合

给定文本或视频序列 $X=\{x_n\}_{n=1}^{N}$，首先产生注意力权重：

$$
a_n=
\operatorname{softmax}
\left(
w^\top\operatorname{MLP}_{pool}(x_n)+m_n
\right),
$$

其中无效位置的 $m_n=-\infty$。中心表示为：

$$
h=\sum_n a_nx_n.
$$

同时计算 token 加权离散度：

$$
d=\sum_n a_n(x_n-h)^2,
$$

以及归一化注意力熵：

$$
H(a)=
-\frac{\sum_n a_n\log(a_n+\epsilon)}
{\log N_{\mathrm{valid}}}.
$$

三类统计量职责分别为：

- $h$：样本的中心语义；
- $d$：词、帧或局部语义在特征维度上的离散程度；
- $H(a)$：聚合过程是否存在明确的关键位置。

### 6.2 概率参数头

均值采用对确定性语义的残差参数化：

$$
\tilde\mu
=
h+
\operatorname{MLP}_{\mu}
\left(
\operatorname{LN}([h;d])
\right).
$$

方差头统一输出 <code>logvar</code>：

$$
\ell
=
\log\sigma^2
=
\operatorname{clip}
\left(
\operatorname{MLP}_{\sigma}
\left(
\operatorname{LN}([h;d;H(a)])
\right),
\ell_{\min},
\ell_{\max}
\right).
$$

第一版建议：

$$
\ell_{\min}=-8,\qquad \ell_{\max}=2.
$$

文本与视频使用结构相同、参数不共享的概率头。不得将同一个张量同时解释为
<code>logsigma</code> 和 <code>logvar</code>。

### 6.3 重参数化随机原型

标准差和随机原型定义为：

$$
\sigma=\exp(0.5\ell),
$$

$$
\tilde z^{(k)}
=
\tilde\mu
+
\sigma\odot\epsilon^{(k)},
\qquad
\epsilon^{(k)}\sim\mathcal N(0,I),
$$

$$
z^{(k)}
=
\frac{\tilde z^{(k)}}
{\|\tilde z^{(k)}\|_2+\epsilon}.
$$

训练默认 $K=4$，使用 antithetic sampling：

$$
\{
\epsilon_1,
-\epsilon_1,
\epsilon_2,
-\epsilon_2
\}.
$$

重参数化必须保留以下梯度链：

~~~text
概率匹配损失 / 随机排序损失
        ↓
随机原型 z(k)
        ↓
μ + exp(0.5 logvar) ⊙ ε(k)
        ↓
μ head / logvar head
        ↓
DSA token / CLIP 特征
~~~

禁止在训练主路径中对随机原型、均值或方差执行无意的 <code>detach</code>。采样
分离版本只作为消融。

### 6.4 双向 Soft Prototype Matching

文本 $i$ 和视频 $j$ 的随机原型相似度为：

$$
C_{ij}^{ab}
=
\cos
\left(
z_{t,i}^{(a)},
z_{v,j}^{(b)}
\right).
$$

文本原型到视频原型的平滑匹配为：

$$
g_{ij}^{a}
=
\tau_m
\log
\left[
\frac1K
\sum_b
\exp
\left(
\frac{C_{ij}^{ab}}{\tau_m}
\right)
\right].
$$

视频原型到文本原型为：

$$
q_{ij}^{b}
=
\tau_m
\log
\left[
\frac1K
\sum_a
\exp
\left(
\frac{C_{ij}^{ab}}{\tau_m}
\right)
\right].
$$

最终概率匹配分数为：

$$
S_{ij}^{prob}
=
\frac12
\left[
\frac1K\sum_a g_{ij}^{a}
+
\frac1K\sum_b q_{ij}^{b}
\right].
$$

这里使用 <code>logmeanexp</code> 而不是普通 <code>logsumexp</code>，避免采样数
$K$ 改变时产生额外的常数偏移。初始匹配温度建议
$\tau_m\in[0.05,0.1]$。

该设计同时避免：

- 将全部 $K^2$ 组合无差别作为强正例；
- 使用不可平滑的单个 hard max；
- 使用一个极端样本决定整对文本—视频的关系。

### 6.5 配对不确定性

随机原型的双向匹配分歧定义为：

$$
U_{ij}^{pair}
=
\frac12
\left[
\operatorname{Var}_a(g_{ij}^{a})
+
\operatorname{Var}_b(q_{ij}^{b})
\right].
$$

可选地计算全部原型匹配权重：

$$
\pi_{ij}^{ab}
=
\operatorname{softmax}_{ab}
\left(
C_{ij}^{ab}/\tau_m
\right),
$$

并加入归一化映射熵：

$$
U_{ij}^{map}
=
U_{ij}^{pair}
+
\lambda_H
\frac{
-\sum_{a,b}\pi_{ij}^{ab}
\log(\pi_{ij}^{ab}+\epsilon)
}{
\log K^2
}.
$$

$U^{pair}$ 和 $U^{map}$ 是查询—候选级量。同一个视频面对不同文本时，应能
产生不同数值，这是区别于模态固定方差的关键设计。

### 6.6 随机排序损失

对查询 $i$、正视频 $p$ 和困难负视频 $n$，构造第 $k$ 次随机配对分数
$r_{ip}^{(k)}$ 与 $r_{in}^{(k)}$。实现时可使用相同索引的独立文本/视频
随机原型，或从双向匹配结果中形成 $K$ 个随机得分。

单个三元组的随机排序损失为：

$$
L_{rank}^{i,p,n}
=
\frac1K
\sum_k
\log
\left[
1+
\exp
\left(
\frac{
r_{in}^{(k)}
-
r_{ip}^{(k)}
+
\delta
}{
T_r
}
\right)
\right].
$$

同时定义负候选反超正候选的概率：

$$
P_{ipn}^{inv}
=
\frac1K
\sum_k
\operatorname{sigmoid}
\left(
\frac{
r_{in}^{(k)}
-
r_{ip}^{(k)}
}{
T_r
}
\right).
$$

第一版每个查询从 $S^{det}$ 或 $S^{prob}$ 中选取 8 或 16 个困难负样本。
禁止把其他有效 caption 对应的同一视频当作负样本。

随机排序损失使梯度同时作用于均值、方差和 DSA 表示。方差不再只是附属输出，
而是决定正负排序在随机嵌入空间中的稳定性。

### 6.7 Pairwise Beta Evidence 增强

Beta evidence 不属于最小核心版。只有在“概率嵌入 + 双向 soft matching +
随机排序”稳定后才启用。

证据头输入：

$$
\phi_{ij}
=
\left[
S_{ij}^{det},
S_{ij}^{prob},
U_{ij}^{pair},
\overline{\sigma_{t,i}^2},
\overline{\sigma_{v,j}^2},
\operatorname{localGap}_{ij}
\right].
$$

输出正、负证据：

$$
e_{ij}^{+},
e_{ij}^{-}
=
\operatorname{softplus}
\left(
\operatorname{MLP}_{e}(\phi_{ij})
\right),
$$

$$
\alpha_{ij}=e_{ij}^{+}+1,
\qquad
\beta_{ij}=e_{ij}^{-}+1.
$$

候选相关概率和证据不足度为：

$$
p_{ij}^{rel}
=
\frac{\alpha_{ij}}
{\alpha_{ij}+\beta_{ij}},
$$

$$
u_{ij}^{evid}
=
\frac{2}
{\alpha_{ij}+\beta_{ij}}.
$$

使用 Beta 期望 Brier 损失：

$$
L_{Beta}^{ij}
=
\left(
y_{ij}
-
\frac{\alpha_{ij}}
{\alpha_{ij}+\beta_{ij}}
\right)^2
+
\frac{
\alpha_{ij}\beta_{ij}
}{
(\alpha_{ij}+\beta_{ij})^2
(\alpha_{ij}+\beta_{ij}+1)
}.
$$

可信困难负样本权重定义为：

$$
w_{in}
=
\operatorname{stopgrad}
\left[
P_{ipn}^{inv}
\cdot
\frac{\beta_{in}}
{\alpha_{in}+\beta_{in}}
\cdot
(1-u_{in}^{evid})
\right].
$$

权重在当前 query 的困难负样本集合内归一化，使其均值接近 1，避免改变总损失
量级。训练开始时令 $w=1$，之后线性启用证据权重。

### 6.8 锚定分布先验

不直接把概率均值强制拉向标准高斯零均值。CLIP/DSA 已经形成有效语义空间，
零均值先验可能破坏预训练表示。

使用以确定性聚合特征为中心的锚定先验：

$$
q(z|x)
=
\mathcal N
\left(
\tilde\mu,
\operatorname{diag}\sigma^2
\right),
$$

$$
p(z|x)
=
\mathcal N
\left(
\operatorname{sg}[h],
\sigma_0^2I
\right).
$$

对应 KL：

$$
L_{anchor}
=
\frac12
\sum_d
\left[
\frac{
\sigma_d^2+
(\tilde\mu_d-\operatorname{sg}[h_d])^2
}{
\sigma_0^2
}
-1
+
\log
\frac{\sigma_0^2}{\sigma_d^2}
\right].
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。该先验限制均值漂移和方差膨胀，
但允许每个样本学习不同语义范围。

## 7. 训练目标

### 7.1 核心版本

核心版本总损失：

$$
L
=
L_{DSA}
+
\lambda_pL_{prob}
+
\lambda_rL_{rank}
+
\lambda_aL_{anchor}.
$$

概率检索损失为支持多正样本的双向对比损失：

$$
L_{prob}
=
\frac12
\left[
\operatorname{MultiPosCE}
\left(
\tau_p S^{prob},
\operatorname{pairId}
\right)
+
\operatorname{MultiPosCE}
\left(
\tau_p(S^{prob})^\top,
\operatorname{pairId}
\right)
\right].
$$

### 7.2 Beta 增强版本

启用证据头后：

$$
L
=
L_{DSA}
+
\lambda_pL_{prob}
+
\lambda_rL_{rank}^{weighted}
+
\lambda_eL_{Beta}
+
\lambda_aL_{anchor}.
$$

初始搜索范围：

| 参数 | 建议范围 |
| --- | --- |
| $\lambda_p$ | 0.05–0.2 |
| $\lambda_r$ | 0.05–0.2 |
| $\lambda_e$ | 0.01–0.05 |
| $\lambda_a$ | $10^{-5}$–$10^{-4}$ |
| $K_{train}$ | 4 |
| $K_{eval}$ | 8 |
| hard negatives/query | 8 或 16 |
| $\tau_m$ | 0.05–0.1 |

这些范围是实验起点，不是固定论文超参数。各损失应先记录未经加权的实际量级，
再确定最终系数。

## 8. 训练流程

### 8.1 单次端到端核心训练

- 从 OpenAI CLIP <code>ViT-B/16</code> 权重初始化，不要求 UATVR 全模型
  checkpoint；
- 在同一个作业和 optimizer 中连续训练 5 epochs；
- <code>FREEZE_LAYER_NUM=8</code>，CLIP 后 4 个 block 与 DSA、WTI、RSPR
  从第一步联合训练；
- $L_{DSA}$ 与 $L_{prob}$ 从第一步使用完整权重；
- $\lambda_r$ 与 $\lambda_a$ 在第一个 epoch 内线性 warm-up；
- 每个 query 使用显式 <code>pair_id</code> 排除多正样本后选择困难负样本；
- 以验证集 R@1 为主选择 checkpoint，同时记录方差与错误的相关性。

概率 mean head 以确定性中心为残差起点，<code>logvar</code> 从先验值开始并受
固定上下界约束；配合 CLIP 低学习率和损失 warm-up，无需拆分概率头预热与联合
排序两个训练作业。A0–A8 使用相同 CLIP 起点、trusted split、5 epochs 和优化
日程，只改变消融矩阵定义的参数。

### 8.2 后续扩展：可信负样本

- 初始化 Pairwise Beta evidence；
- 先令随机排序权重 $w=1$，单独训练证据头；
- 证据头稳定后，将 $w$ 从 1 线性过渡到证据加权值；
- 比较无证据、只用反超概率、完整可信困难负样本三种设置；
- 不在同一阶段加入 PCM、ensemble 或 conformal。

## 9. 推理与检索

### 9.1 全库召回

使用概率均值归一化向量：

$$
S_{ij}^{ann}
=
\cos
\left(
\tilde\mu_{t,i},
\tilde\mu_{v,j}
\right).
$$

该分数用于全库 ANN 或初始 Top-R 召回，不进行全库 $K^2$ 原型匹配。

### 9.2 Top-R 精排

对 Top-R 候选计算 DSA 和概率匹配：

$$
S_{ij}^{final}
=
\frac{S_{ij}^{det}}{T_d}
+
\lambda_p
\frac{S_{ij}^{prob}}{T_p}.
$$

启用且验证 Beta evidence 有效后，再加入：

$$
S_{ij}^{final}
=
\frac{S_{ij}^{det}}{T_d}
+
\lambda_p
\frac{S_{ij}^{prob}}{T_p}
+
\lambda_e
\operatorname{logit}
\left(
p_{ij}^{rel}
\right).
$$

$T_d,T_p,\lambda_p,\lambda_e$ 在验证集确定。不得未经尺度校准直接相加
CLIP 温度 logits、余弦分数和 evidence logits。

推理噪声使用固定的 antithetic 或 Sobol 正态样本，保证同一 checkpoint 与数据集
重复评估得到相同排名。

T2V 与 V2T 的精排输出矩阵分别只保留各自召回候选，方向内未召回位置填
<code>-inf</code>，不得用两个方向的候选并集补全另一方向。该稀疏矩阵用于表达
实际精排范围和导出结果，不得直接传给会把 <code>-inf</code> 真值当作无效样本的
历史指标函数。

计算标准检索指标时，为每个方向单独构造完整且确定的指标排名：Top-R 候选按
$S^{final}$ 排在前面，未进入 Top-R 的尾部候选继续按 $S^{ann}$ 排序。所有排序
使用稳定降序；分数完全相同时，原始候选索引较小者优先。指标表示必须保证每个
真实 query 恰好计入一次，Top-R 漏召回记为失败；多 caption padding 仍单独使用
<code>-inf</code> 排除。指标用完整排名不得回写或污染方向独立的精排输出矩阵。

### 9.3 不确定性输出

第一阶段输出：

- 模态方差摘要 $\overline{\sigma_t^2},\overline{\sigma_v^2}$；
- 查询—候选配对不确定性 $U^{pair}$；
- Beta evidence 启用后的 $p^{rel},u^{evid}$；
- Top-1 与 Top-2 的随机交换概率。

这些量首先作为可分析输出。未经验证集校准前，不将其命名为“检索正确概率”或
直接用于拒绝决策。

## 10. 代码组织建议

建议新增独立模块，不将所有逻辑继续写入现有 <code>UATVR</code> 类：

~~~text
prob_models/
  reparameterized_distribution.py
    ├─ MaskedStatPool
    ├─ ReparameterizedDistributionHead
    └─ antithetic_sample

modules/
  stochastic_prototype_ranking.py
    ├─ BidirectionalSoftPrototypeMatcher
    ├─ PairwiseUncertainty
    ├─ StochasticRankLoss
    └─ PairwiseBetaEvidence
~~~

主模型只负责：

1. 从 DSA 获得 $T,V,S^{det}$；
2. 调用 RSPR 得到分布、样本、$S^{prob}$ 和 $U^{pair}$；
3. 按训练阶段组装损失；
4. 在评估阶段选择 mean-only 召回或 Top-R 概率精排。

建议 RSPR 返回具名结构：

~~~text
RSPROutput
  mu_text
  logvar_text
  mu_video
  logvar_video
  samples_text
  samples_video
  probabilistic_logits
  pair_uncertainty
  relevance_probability      # optional
  evidence_uncertainty       # optional
  anchor_kl
~~~

## 11. 数值稳定性与复杂度

### 11.1 数值规则

- 统一使用 <code>logvar</code>，采样标准差必须是
  <code>exp(0.5 * logvar)</code>；
- <code>logvar</code> 与指数运算使用 FP32，即使其他路径启用 AMP；
- 对 <code>logvar</code> 执行范围限制；
- 随机原型进入余弦相似度前执行 L2 normalize，并保留有限的
  <code>eps</code>；
- <code>logmeanexp</code> 使用稳定的 max-shift 实现；
- mask 全空时立即报错，不返回伪造池化向量；
- anchor KL 按 batch 和维度归一化后再乘权重；
- 监控方差最小值、最大值、均值及分位数，识别方差塌缩和膨胀；
- Beta evidence 使用 <code>softplus</code>，不使用会截断负分梯度的
  <code>ReLU(score)</code>；
- evidence 概率进入 <code>logit</code> 前限制到
  $[\epsilon,1-\epsilon]$。

### 11.2 复杂度

概率原型相似度复杂度为：

$$
O(B_tB_vK^2D).
$$

当 $K=4$ 时，batch 内概率分支通常小于词—帧交互
$O(B_tB_vLFD)$ 的成本。全库推理不运行该复杂度，只对 Top-R 精排。

若显存不足，按候选维度分块计算 $C$，不得降低为无记录的随机子采样。

## 12. 实验与消融设计

### 12.1 主实验

首轮数据集：

- MSR-VTT 1K-A；
- DiDeMo；
- VATEX。

主指标：

- Text-to-Video R@1/R@5/R@10、MdR、MnR；
- Video-to-Text R@1/R@5/R@10、MdR、MnR；
- 训练显存、训练吞吐和 Top-R 精排耗时。

辅助指标：

- $U^{pair}$ 与 Top-1 错误的 AUROC；
- 不确定性与正负随机排序交换率的相关性；
- 输入模糊、帧丢失和文本截断下的不确定性变化；
- 推理重复运行的排名一致性。

### 12.2 必须完成的消融

| 编号 | 设置 | 验证问题 |
| --- | --- | --- |
| A0 | UATVR DSA baseline | 确定性起点 |
| A1 | $K=1$，只使用概率均值 | 参数量增加是否已经解释收益 |
| A2 | $K=4$，随机样本 detach | 随机增强但无重参数化梯度的效果 |
| A3 | $K=4$，完整重参数化 | 重参数化梯度的独立贡献 |
| A4 | UATVR 原始多实例监督 | 全部随机组合强制对齐的效果 |
| A5 | hard max prototype matching | 极值匹配的效果与稳定性 |
| A6 | 双向 soft prototype matching | 平滑多原型匹配的贡献 |
| A7 | A6 去掉 stochastic rank | 随机排序监督的贡献 |
| A8 | A6 去掉 anchor KL | 方差膨胀/塌缩及性能变化 |
| A9 | 核心版 + Beta evidence | 可信困难负样本的贡献 |
| A10 | $K=1/2/4/8$ | 精度、成本与稳定性的关系 |
| A11 | random / antithetic sampling | 采样方差控制的作用 |

必须优先展示以下递进关系：

~~~text
确定性均值
  < 重参数化概率匹配
  < 重参数化概率匹配 + 随机排序
  < 随机排序 + 可信困难负样本
~~~

若该关系不能在至少两个数据集上稳定出现，不继续堆叠更复杂的不确定性模块。

## 13. 验收条件

RSPR 核心版完成需同时满足：

1. DSA 确定性基线在相同数据协议下可复现；
2. 概率头明确输出 <code>mu/logvar</code>，采样公式与 KL 参数语义一致；
3. $K=4$ 重参数化样本能对均值头、方差头和上游 DSA 产生有限梯度；
4. 双向 soft prototype matching 支持 $B_t\ne B_v$；
5. 多正样本和 DDP gather 后的正例由 <code>pair_id</code> 正确构造；
6. 固定推理噪声下重复评估得到相同分数和排名；
7. 核心损失无 NaN/Inf，方差未出现全维塌缩或持续触顶；
8. 完成 A0–A8 消融后再决定是否启用 Beta evidence；
9. 至少在两个标准数据集上验证检索性能，并报告不确定性与错误的关系；
10. 不以未经校准的方差或证据量宣称系统已经获得可靠置信度。

## 14. 后续研究边界

若第一阶段主要提高 R@1，则下一步可将 $U^{pair}$、Beta evidence、分支分歧和
Top-1/Top-2 随机交换概率送入校准模块，发展来源归因的 selective retrieval 或
Mondrian conformal candidate set。

若第一阶段 Recall 提升有限，但 $U^{pair}$ 对错误检测明显有效，则停止继续增加
采样数或分布距离，将研究主线转向风险—覆盖、动态候选集合和开放集决策。

若概率原型存在明显单峰表达不足，再单独研究由 DSA 聚合 token 或 DUQ patch
cluster 锚定的高斯混合表示；该方向不与 RSPR 第一阶段同时实施。
