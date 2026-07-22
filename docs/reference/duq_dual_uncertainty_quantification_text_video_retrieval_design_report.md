# 方法设计报告：DUQ（Dual Uncertainty Quantification for Text-Video Retrieval）

## 1. 目标问题

- **任务与输入输出**：给定文本查询和候选视频库，模型输出一张以文本为行、视频为列的检索分数矩阵；Text-to-Video 按行评估，Video-to-Text 由其转置评估。输入文本由词元序列表示，输入视频由均匀采样帧表示；训练时一个 batch 内的同索引文本—视频为正配对，其余组合为负配对。
- **具体缺口**：低质量、多场景视频或稀疏文本会使正配对内部的语义交互缺少可信度，形成 **intra-pair interaction uncertainty**；相似场景、相似画面或近似描述会使负候选难以排除，形成 **inter-pair exclusion uncertainty**。单一余弦相似度只能表达接近程度，不能同时回答“这次匹配是否可信”和“相似负样本是否已经拉开”。
- **设计目标**：DUQ 以 Intra-pair Similarity Uncertainty Module（ISUM）将相似度矩阵转为 Dirichlet 证据分布，约束正配对的可信度；以 Inter-pair Distance Uncertainty Module（IDUM）构造多样化概率嵌入和边界距离，增强负配对区分。两者对应“Pull 正配对”和“Push 负配对”，共同改善最终检索分数。

## 2. 总体解决思路

~~~text
文本 token ──→ CLIP Text Encoder ──→ 句子特征 s / 词特征 w ─────────────┐
                                                                        │
视频帧 ────→ CLIP Vision Encoder ─→ 帧特征 f / 视觉 token p（含帧 CLS） │
                                         │                              │
                        时序聚合 + Patch 压缩                           │
                                         │                              │
                 ┌───────────────────────┴──────────────────────────────┤
                 │                                                      │
        句子—帧 S_sf / 词—Patch S_wp                           局部特征自注意力聚合
                 │                                                      │
        ISUM：Dirichlet 证据损失                              μ、σ → K 个概率嵌入
                 │                                                      │
                 └────────────────→ 概率集合 → 论文 IDUM：D,u_d / 源码：S_tv
                                                        │
训练：三路双向匹配 + 三路证据约束 + Gaussian KL + 分数一致性 KL
推理：S_raw=(S_sf+S_wp+S_tv)/3 → 全库评估矩阵 → R@K、MdR、MnR
~~~

论文把基础相似度 \(s\) 交给 ISUM，把局部—全局融合后的概率嵌入交给 IDUM，并在训练中联合 \(L_S,L_S^U,L_D,L_D^U,L_{KL}\)。仓库训练与评估入口则直接组织三张矩阵：句子—帧 \(S_{sf}\)、词—视觉 token \(S_{wp}\) 和概率集合 \(S_{tv}\)；下文分别说明论文机制与这些源码计算。[Model.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:159) 计算训练目标，[get_similarity_logits](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:398) 形成评估阶段的原始分数。

## 3. 核心模块设计

### 3.1 CLIP 多粒度特征提取与视频时序编码

- **作用**：同时保留可用于全局语义匹配的句子/帧表示，以及可用于局部对应和概率建模的词元/视觉 token 表示，避免在编码阶段过早压缩多场景视频和稀疏文本。
- **输入与输出**：文本输入为 \(B\times L\)：\(B\) 条文本的 \(L\) 个 token；输出句子 EOT 特征 \(s\in\mathbb{R}^{B\times D}\) 和词元特征 \(w\in\mathbb{R}^{B\times L\times D}\)。视频输入为 \(B\times F\times3\times224\times224\)：每个视频 \(F\) 帧；输出帧 CLS 特征 \(f\in\mathbb{R}^{B\times F\times D}\)，以及跨帧拼接的完整视觉 token 序列 \(p\in\mathbb{R}^{B\times(FP_{\mathrm{frame}})\times D}\)，其中每帧序列包含 CLS 和 Patch token。
- **核心计算**：文本 Transformer 取 EOT 位置作为全局表示并保留全部 hidden token；视觉 Transformer 对每帧输出 CLS 和完整 Patch 序列。默认 <code>seqTransf</code> 为帧特征加入位置编码，经轻量 Transformer 更新后与原帧特征残差相加，保持 \(B\times F\times D\) 形状。
- **阶段与依赖**：训练和推理均执行；\(s,f\) 进入句子—帧匹配，\(w,p\) 进入细粒度匹配，四类表示共同服务于后续概率嵌入。
- **核心代码**：[get_text_feat](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:333) 和 [get_video_feat](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:344) 整理四类特征；[CLIP.encode_text](/data2/hxj/project/tvr/research_refs/DUQ/models/module_clip.py:468) 与 [CLIP.encode_image](/data2/hxj/project/tvr/research_refs/DUQ/models/module_clip.py:457) 返回全局/局部表示；[agg_f_feat](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:276) 完成视频时序编码。
- **模块结果**：该模块是三条相似度路径的共同输入，主文没有给出编码器的独立消融。论文 Table 3 中 conventional feature-similarity baseline \(L_S\) 的 Text-to-Video/Video-to-Text R@1 为 46.9/44.4，该数值只作为完整方法的总体比较起点。

### 3.2 Patch Compression 与确定性双粒度匹配

- **作用**：既让一句文本能够从多帧中选择重要时刻，又让每个词元能够寻找最相关的局部视觉 token；Patch Compression Module（PCM）先合并重复视觉 token，再执行细粒度匹配。
- **输入与输出**：输入句子 \(s\)、词序列 \(w\)、帧序列 \(f\) 与跨帧视觉 token \(p\)。三级 PCM 将 \(p:B\times(FP_{\mathrm{frame}})\times D\) 逐级压缩为 \(p':B\times P\times D\)；两条匹配路径分别输出 \(S_{sf},S_{wp}\in\mathbb{R}^{B_t\times B_v}\)。
- **核心计算**：
  1. PCM 用 DPC-KNN 计算 token 间距离、局部密度和“密度 × 到更高密度点距离”，选择聚类中心；簇内 token 按学习到的指数权重合并。三个阶段的 <code>sample_ratio=0.5</code>，压缩 token 再以自身为 query、压缩前 token 为 key/value 做注意力残差更新。
  2. 句子—帧路径先计算 \(C^{sf}_{ijr}=\cos(s_i,f_{jr})\)，再用视频侧 softmax 帧权重聚合：

     \[
     S_{sf}(i,j)=\sum_r a^f_{jr}C^{sf}_{ijr}.
     \]

  3. 词—Patch 路径计算 \(C^{wp}_{ij\ell p}=\cos(w_{i\ell},p'_{jp})\)，分别执行“每词找最佳 Patch”和“每 Patch 找最佳词”，再以学习权重聚合：

     \[
     S_{wp}(i,j)=\frac12\left[
     \sum_\ell a^w_{i\ell}\max_p C^{wp}_{ij\ell p}
     +\sum_p a^p_{jp}\max_\ell C^{wp}_{ij\ell p}
     \right].
     \]
- **阶段与依赖**：训练和推理均保留。两张矩阵各自接受对称检索监督和 ISUM 证据约束，并与概率分支共同形成最终分数。
- **核心代码**：[get_less_patch_feat](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:382) 串联三级压缩；[PCM](/data2/hxj/project/tvr/research_refs/DUQ/models/cluster.py:508)、[cluster_dpc_knn](/data2/hxj/project/tvr/research_refs/DUQ/models/cluster.py:324) 与 [Att_Block_Patch](/data2/hxj/project/tvr/research_refs/DUQ/models/cluster.py:640) 完成聚类和注意力更新；[s_and_f](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:305) 与 [w_and_p](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:311) 产生两张确定性矩阵。
- **模块结果**：PCM 和双粒度实现没有在主文中独立拆分数值；它们属于代码中的确定性特征分支，并在完整模型中与 ISUM、IDUM 联合工作。

### 3.3 ISUM：Intra-pair Similarity Uncertainty

- **作用**：使正配对不仅拥有较高相似度，还能提供足够证据支撑该判断；对低质量视频或稀疏文本造成的不可信交互进行显式约束。
- **输入与输出**：输入为 batch 内相似度矩阵 \(S\in\mathbb{R}^{B\times B}\)，监督目标为单位矩阵 \(Y=I_B\)。论文机制由此定义 Dirichlet 参数、期望配对概率、相似度不确定度 \(u_s\) 和训练损失 \(L_S^U\)。源码 <code>UM</code> 在内部构造 \(\alpha,\hat p\)，直接返回证据损失与 \(1-B/A\) 的置信质量；三个训练调用只使用损失，并分别作用于 \(S_{sf},S_{wp},S_{tv}\)。
- **核心计算**：先把非负相似度视为证据，

  \[
  e_{ij}=\operatorname{ReLU}(S_{ij}),\quad
  \alpha_{ij}=e_{ij}+1,\quad
  A_i=\sum_j\alpha_{ij},\quad
  \hat p_{ij}=\frac{\alpha_{ij}}{A_i}.
  \]

  论文以 \(u_i=B/A_i\) 表示总不确定度。每个方向的广义 MSE 同时惩罚预测误差和 Dirichlet 方差：

  \[
  L_i^U=\sum_j\left[
  (Y_{ij}-\hat p_{ij})^2+
  \frac{\hat p_{ij}(1-\hat p_{ij})}{A_i+1}
  \right],
  \]

  论文 Eq. (7) 以 \(B^{-1}(\sum_iL_i^U+\sum_jL_j^U)\) 汇合两个方向；代码 <code>UM</code> 则对行方向和转置方向的均值再取算术平均。
- **阶段与依赖**：证据损失仅参与训练并直接约束三张相似度矩阵。论文的推理公式使用 \(u_s=B/A\) 调节基础相似度；源码第二返回值是其补量 \(1-B/A\)，训练调用未保留该值，仓库评估路径也不调用 <code>UM</code>。
- **核心代码**：[UM.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/uncertainty.py:27) 建立单位目标、ReLU 证据、Dirichlet strength 和双向 MSE；三个调用点位于 [Model.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:193)。
- **模块结果**：MSRVTT 消融中，\(L_S\) 加入 \(L_S^U\) 后，Text-to-Video R@1 从 46.9 提升到 47.5，Video-to-Text R@1 从 44.4 提升到 46.9；这是“基础相似度 + ISUM”组合的结果。

### 3.4 Local Feature Aggregation 与高斯概率嵌入

- **作用**：将一个文本或视频从单个确定性向量扩展为多个语义实例，使多场景视频和多义文本能够以分布形式参与候选区分。
- **输入与输出**：文本侧拼接 \(t=[s;w]\in\mathbb{R}^{B\times(1+L)\times D}\)，视频侧拼接 \(v=[f;p']\in\mathbb{R}^{B\times(F+P)\times D}\)。每个模态输出 \(\mu,\sigma\in\mathbb{R}^{B\times D}\) 和 \(K\) 个样本 \(z\in\mathbb{R}^{B\times K\times D}\)。
- **核心计算**：<code>module_agg</code> 先以多头自注意力、MLP 和 LayerNorm 汇聚局部 token；<code>prob_embed</code> 将汇聚结果归一化为 \(\mu\)，并以另一条注意力—MLP 路径预测代码中命名为 \(\sigma\) 的 log-scale。重参数采样为

  \[
  z^k=\mu+\exp(\sigma)\odot\epsilon_k,\qquad
  \epsilon_k\sim\mathcal N(0,I).
  \]

  论文主实验设 \(K=7\)；仓库模型初始化将文本与视频采样数均设为 10。
- **阶段与依赖**：训练和推理均生成概率样本；论文将样本交给 IDUM 边界距离，源码将样本交给概率集合相似度。训练时另以标准高斯先验 KL 约束分布。
- **核心代码**：[module_agg](/data2/hxj/project/tvr/research_refs/DUQ/models/module_agg.py:69) 完成局部聚合，[prob_embed](/data2/hxj/project/tvr/research_refs/DUQ/models/module_prob.py:6) 产生 \(\mu,\sigma\)；[create_text_prob/create_video_prob](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:256) 和 [sample_gaussian](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:251) 形成样本集合。
- **模块结果**：主文把概率表示与距离目标作为 \(L_D\) 组合评估；在仅有 \(L_S\) 的基础上加入 \(L_D\)，Text-to-Video/Video-to-Text R@1 从 46.9/44.4 变为 48.2/45.2。

### 3.5 论文 IDUM 与源码概率集合匹配

- **作用**：针对多个高相似负候选，通过概率嵌入的边界关系强化排斥，而不是只比较两个单点向量。
- **输入与输出**：论文输入文本、视频各 \(K\) 个概率嵌入，形成样本两两关系 \(B_t\times B_v\times K\times K\)，输出边界距离矩阵 \(D\in\mathbb{R}^{B_t\times B_v}\)、距离不确定度 \(u_d\)、距离匹配损失 \(L_D\) 和证据损失 \(L_D^U\)。仓库概率分支输出同尺寸的概率集合相似度 \(S_{tv}\)。
- **核心计算**：
  - 论文先定义 \(d(z_t^{i,m},z_v^{j,n})=1-\cos(z_t^{i,m},z_v^{j,n})\)，再按配对类型选边界：

    \[
    D_{ij}=
    \begin{cases}
    \min_{m,n}d(z_t^{i,m},z_v^{j,n}), & i=j,\\
    \max_{m,n}d(z_t^{i,m},z_v^{j,n}), & i\ne j.
    \end{cases}
    \]

    正配对保留最吻合的实例，负配对保留最外侧边界；\(D\) 使用 \(1-I_B\) 作为证据目标，产生 \(L_D^U\)。
  - 仓库中的概率集合落点 <code>t_and_v</code> 先计算归一化样本余弦 \(C^{tv}_{ijmn}\)，再做两个方向的 max-match 与学习权重聚合：

    \[
    S_{tv}=\frac12\left[
    \sum_m a^t_{im}\max_n C^{tv}_{ijmn}
    +\sum_n a^v_{jn}\max_m C^{tv}_{ijmn}
    \right].
    \]

    \(S_{tv}\) 接受双向交叉熵，并以单位矩阵目标调用 <code>UM</code>；该函数没有显式构造论文的 \(D,L_D,L_D^U\)，而是把 \(S_{tv}\) 直接用于源码的最终检索分数。
- **阶段与依赖**：论文 IDUM 在训练中优化距离与距离证据、在推理中提供 \(d,u_d\)；源码概率分支在训练中优化 \(S_{tv}\) 的匹配和证据损失，推理时重新采样并计算 \(S_{tv}\)。
- **核心代码**：[t_and_v](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:322) 实现概率样本的双向 max-match；[Model.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:202) 将其接入概率匹配和证据损失。
- **模块结果**：论文消融中，\(L_S+L_S^U+L_D\) 的 Text-to-Video/Video-to-Text R@1 为 49.2/49.2；加入距离证据约束后的完整 DUQ 为 51.2/50.4。该数值体现论文 ISUM、概率距离和 IDUM 的组合效果。

### 3.6 双不确定性联合目标与检索分数融合

- **作用**：让确定性相似度、概率关系和证据约束在同一训练目标中协同，并在推理时形成供评估转换的原始分数矩阵。
- **输入与输出**：训练输入为多张 \(B\times B\) 匹配矩阵及概率分布参数，输出标量总损失；推理输入为缓存的文本/视频多粒度特征，输出全库 \(N_t\times N_v\) 原始分数矩阵。
- **核心计算**：论文总目标为

  \[
  L_{\mathrm{paper}}
  =L_S+L_S^U+\alpha(L_D+L_D^U)+\beta L_{KL},
  \]

  其中 \(L_{KL}\) 将文本、视频概率分布约束到标准高斯。论文推理将相似度、距离及两类不确定度组合为

  \[
  s'=\left[e^{-\gamma_1u_d}(1-d)\right]
  \circ\left[e^{-\gamma_2u_s}s\right].
  \]

  仓库 <code>forward</code> 独立组织三路匹配。对 \(x\in\{sf,wp,tv\}\)，每路监督为

  \[
  L_x=\frac12\left[
  \operatorname{CrossEn}(\tau S_x)+
  \operatorname{CrossEn}(\tau S_x^\top)
  \right],\qquad
  \tau=\exp(\mathrm{clip.logit\_scale}).
  \]

  实际总目标为

  \[
  \begin{aligned}
  L_{\mathrm{code}}={}&L_{sf}+L_{wp}+L_{tv}\\
  &+\alpha\left[U(S_{sf})+U(S_{wp})+U(S_{tv})\right]\\
  &+\beta L_{\mathrm{priorKL}}+L_{\mathrm{cons}},
  \end{aligned}
  \]

  其中 \(L_{\mathrm{priorKL}}\) 以 \(K\) 个样本的均值 \(\bar z\) 和预测 \(\sigma\) 计算 \(-\frac12\sum(1+\sigma-\bar z^2-e^\sigma)\)，并累加文本、视频两项；\(L_{\mathrm{cons}}\) 是三张矩阵两两、双方向的 softmax KL 均值。仓库评估的模型原始分数为

  \[
  S_{\mathrm{raw}}=\frac{S_{sf}+S_{wp}+S_{tv}}{3}.
  \]
- **阶段与依赖**：交叉熵、证据损失、高斯 KL 与一致性 KL 仅在训练时存在；推理只保留 CLIP、多粒度匹配、概率采样和三路分数融合。
- **核心代码**：三路匹配及总损失组装位于 [Model.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:193)；[CrossEn、KLdivergence 与 KL](/data2/hxj/project/tvr/research_refs/DUQ/models/until_module.py:62) 实现基础损失；[get_similarity_logits](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:398) 返回最终三路均值。
- **模块结果**：完整 DUQ 在 MSRVTT、ViT-B/32 上达到 Text-to-Video R@1/R@5/R@10 51.2/77.3/86.1，Video-to-Text 为 50.4/79.2/87.5；相对 \(L_S\) 基线的两方向 R@1 分别提高 4.3 和 6.0 个百分点。

## 4. 端到端训练与推理

### 训练

1. DataLoader 输出文本 id、文本 mask、视频帧、视频 mask 和正配对索引；默认主文设置为每视频 12 帧、分辨率 \(224\times224\)。多卡训练先聚合各卡特征，以全局 batch 建立配对矩阵。[build_dataloader](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:125) [Model.forward](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:175)
2. CLIP 生成 \(s,w,f,p\)；Patch 压缩后计算 \(S_{sf},S_{wp}\)，局部聚合和高斯采样后计算 \(S_{tv}\)。
3. 三张矩阵分别计算双向对角正配对交叉熵与 ISUM 证据损失；概率分布计算标准高斯 KL，三张矩阵之间计算一致性 KL。
4. 代码总损失为 \(L_{sf}+L_{wp}+L_{tv}+\alpha(U_{sf}+U_{wp}+U_{tv})+\beta L_{\mathrm{priorKL}}+L_{\mathrm{cons}}\)。训练循环执行反向传播、梯度裁剪和 BertAdam 更新。[train_epoch](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:207)

### 推理

1. <code>eval_epoch</code> 在 <code>model.eval()</code> 和无梯度环境中缓存全部文本的 \(s,w\) 与全部视频的 \(f,p\)，多句对应一个视频时只保留一次视频特征。[eval_epoch](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:311)
2. 文本块与视频块两两调用 <code>get_similarity_logits</code>，重新计算 Patch 压缩、概率样本及 \(S_{sf},S_{wp},S_{tv}\)，返回三路均值并拼接为完整原始矩阵 \(S_{\mathrm{raw}}\in\mathbb{R}^{N_t\times N_v}\)。[_run_on_single_gpu](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:288)
3. 常规单句设置先分别对 \(S_{\mathrm{raw}}\) 及其转置调用 <code>np_softmax</code>（内部将分数乘 100，并按默认 axis 0 归一化），再计算两个方向的对角正配对名次；多句对应单视频时改用 tensor 排名函数。最终输出 R@1/R@5/R@10、Median Rank 和 Mean Rank。[eval_epoch](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:436) [compute_metrics](/data2/hxj/project/tvr/research_refs/DUQ/utils/metrics.py:10)

## 5. 核心代码实现映射

| 设计模块 | 代码入口 | 核心对象/函数 | 输入 → 输出 | 直接职责 |
| --- | --- | --- | --- | --- |
| 参数、模型与数据入口 | [main_retrieval.py](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:36) | <code>get_args</code>、<code>build_model</code>、<code>build_dataloader</code> | 运行参数 → Model 与 DataLoader | 建立训练/评估调用链 |
| CLIP 多粒度编码 | [modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:333) | <code>get_text_feat</code>、<code>get_video_feat</code> | token/视频帧 → \(s,w,f,p\) | 提供全局与局部表示 |
| 视频时序聚合 | [modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:276) | <code>agg_f_feat</code>、<code>TransformerClip</code> | \(B\times F\times D\) → 上下文化帧序列 | 建模帧间时序关系 |
| Patch Compression | [cluster.py](/data2/hxj/project/tvr/research_refs/DUQ/models/cluster.py:508) | <code>PCM</code>、<code>cluster_dpc_knn</code> | \(B\times(FP_{\mathrm{frame}})\times D\) → \(B\times P\times D\) | 合并跨帧视觉 token |
| 确定性匹配 | [modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:305) | <code>s_and_f</code>、<code>w_and_p</code> | \(s,w,f,p'\) → \(S_{sf},S_{wp}\) | 产生全局/局部检索矩阵 |
| ISUM | [uncertainty.py](/data2/hxj/project/tvr/research_refs/DUQ/models/uncertainty.py:5) | <code>UM</code> | \(B\times B\) 分数 → 证据损失与置信质量 | 约束 intra-pair 可信度 |
| 概率嵌入 | [module_agg.py](/data2/hxj/project/tvr/research_refs/DUQ/models/module_agg.py:69)、[module_prob.py](/data2/hxj/project/tvr/research_refs/DUQ/models/module_prob.py:6)、[modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:251) | <code>module_agg</code>、<code>prob_embed</code>、<code>sample_gaussian</code> | token 集 → \(\mu,\sigma,z^{1:K}\) | 建立多实例分布表示 |
| 概率集合匹配（源码） | [modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:322) | <code>t_and_v</code> | 文本/视频概率样本 → \(S_{tv}\) | 以双向 max-match 形成第三路分数 |
| 总损失 | [modeling.py](/data2/hxj/project/tvr/research_refs/DUQ/models/modeling.py:193) | <code>CrossEn</code>、<code>UM</code>、<code>KLdivergence</code>、<code>KL</code> | 三路矩阵与分布参数 → 标量损失 | 联合优化匹配、证据与分布 |
| 全库检索评估 | [main_retrieval.py](/data2/hxj/project/tvr/research_refs/DUQ/main_retrieval.py:288) | <code>get_similarity_logits</code>、<code>compute_metrics</code> | 缓存特征 → 全库分数与双向指标 | 形成最终排名 |

## 6. 实现结果

### 6.1 功能性结果

- **多粒度表示**：每条文本同时产生句子特征和词元序列，每个视频同时产生时序帧特征和压缩 Patch 序列；它们构成确定性 \(S_{sf},S_{wp}\) 两张分数矩阵。
- **概率与不确定性表示**：局部—全局聚合结果被参数化为 \(\mu,\sigma\) 并采样 \(K\) 个文本/视频嵌入；论文 ISUM/IDUM 定义相似度与距离证据，源码则以 <code>UM</code> 训练三路相似度分布。
- **最终检索分数**：仓库模型输出 \(S_{\mathrm{raw}}=(S_{sf}+S_{wp}+S_{tv})/3\)。证据损失、高斯 KL 和一致性 KL 只参与训练；常规评估再对 \(S_{\mathrm{raw}}\) 及其转置分别做 softmax 后计算双向排名。

### 6.2 实验结果

- **MSRVTT-1K 主结果**：

  | Backbone | 方向 | R@1 | R@5 | R@10 | MdR | MnR |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | ViT-B/32 | Text-to-Video | 51.2 | 77.3 | 86.1 | 1.0 | 10.8 |
  | ViT-B/32 | Video-to-Text | 50.4 | 79.2 | 87.5 | 1.0 | 6.4 |
  | ViT-B/16 | Text-to-Video | 55.9 | 81.0 | 88.6 | 1.0 | 8.4 |
  | ViT-B/16 | Video-to-Text | 54.6 | 82.4 | 89.9 | 1.0 | 5.3 |

- **其他数据集 Text-to-Video 结果**：

  | 数据集 | R@1 | R@5 | R@10 | MdR | MnR |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | DiDeMo | 51.8 | 77.9 | 86.5 | 1.0 | 10.6 |
  | LSMDC | 28.5 | 48.2 | 58.0 | 6.0 | 41.2 |
  | Charades | 28.5 | 55.0 | 66.8 | 4.0 | 22.9 |
  | VATEX | 80.0 | 97.4 | 99.0 | 1.0 | 1.6 |

- **组件消融（MSRVTT，ViT-B/32）**：

  | 组合 | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | V2T R@5 | V2T R@10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | \(L_S\) | 46.9 | 74.5 | 82.2 | 44.4 | 73.3 | 84.0 |
  | \(L_S+L_S^U\) | 47.5 | 73.9 | 83.5 | 46.9 | 73.8 | 83.8 |
  | \(L_S+L_D\) | 48.2 | 74.6 | 83.4 | 45.2 | 73.7 | 84.8 |
  | \(L_S+L_S^U+L_D\) | 49.2 | 77.6 | 85.4 | 49.2 | 77.7 | 84.8 |
  | \(L_S+L_S^U+L_D^U\) | 48.9 | 76.3 | 84.6 | 48.5 | 74.5 | 84.3 |
  | 完整 DUQ | **51.2** | **77.3** | **86.1** | **50.4** | **79.2** | **87.5** |

- **关键设计选择**：主文采用 CLIP、batch size 32、5 epochs、每视频 12 帧、\(224\times224\) 输入、\(\alpha=0.1\)、\(\beta=10^{-4}\)、\(\gamma_1=\gamma_2=0.1\) 和 \(K=7\)。在域外 Text-to-Video 结果中，MSRVTT→DiDeMo/LSMDC 的 R@1 为 43.0/21.4。

## 7. 设计闭环总结

- **正配对相似但依据不足** → **ISUM 将相似度转为 Dirichlet 证据并最小化误差与方差** → “相似”同时受到可信度监督，加入 \(L_S^U\) 后 MSRVTT 的 T2V/V2T R@1 达到 47.5/46.9。
- **相似负候选难以由单点表示排除** → **局部特征聚合、高斯多实例与 IDUM 边界关系** → 概率语义实例扩展候选差异，\(L_S+L_S^U+L_D\) 的两方向 R@1 达到 49.2/49.2。
- **单一路径不能同时完成 Pull 与 Push** → **双不确定性联合目标和三路推理分数融合** → 完整 DUQ 在 MSRVTT、ViT-B/32 上取得 T2V/V2T R@1 51.2/50.4，ViT-B/16 下进一步达到 55.9/54.6。
