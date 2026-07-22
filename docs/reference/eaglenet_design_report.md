# 方法设计报告：EagleNet

## 1. 目标问题

- **任务与输入输出**：EagleNet 面向双向文本—视频检索。输入是一条文本与由多个均匀采样帧组成的视频，输出每个文本—视频候选对的匹配分数，并据此完成文本检索视频或视频检索文本。
- **具体缺口**：视频描述通常很短，原始文本嵌入不足以覆盖视频的丰富语义。已有文本增强方法主要利用文本—帧交互来扩展文本，未将视频内部的帧—帧关系纳入增强过程，因此生成的文本表示难以反映视频的全局与时序上下文。
- **设计目标**：围绕候选视频生成上下文感知的增强文本表示；同时在帧级学习文本—视频配对关系，使最终的全局检索对齐由细粒度关系支撑。

## 2. 总体解决思路

```text
文本、视频帧
    │
    ├─ CLIP 文本/图像编码 ──► 原始文本 t、帧序列 F
    │                              │
    │                         随机文本候选
    │                              │
    └────────────► FRL 文本—帧图与 RGAT ──► 增强文本 t_gen
                                                  │
                                  文本条件帧聚合（XPool 风格）
                                                  │
                                         视频表示 v 与检索 logits
                                                  │
训练期：Sigmoid 匹配损失 + support 损失 + EAM 帧级能量损失
```

EagleNet 先用 CLIP 编码文本和视频帧，再针对每个文本—视频候选对采样多个文本候选。细粒度关系学习（FRL）将帧、原文本和候选文本组成多关系图，通过 RGAT 计算候选文本与视频上下文的关联权重，融合为增强文本 `t_gen`。`t_gen` 作为 Query 聚合视频帧，得到条件化视频表示；二者的归一化点积构成检索分数。能量感知匹配（EAM）在训练时以逐文本—帧能量提供细粒度监督，sigmoid 损失则对检索矩阵的每个文本—视频对独立施加匹配目标。

## 3. 核心模块设计

### 3.1 CLIP 表征与随机文本候选

- **作用**：建立共同的文本/视觉表征空间，并从原始短文本周围生成多个可由视频上下文选择的语义候选。
- **输入与输出**：CLIP 产生文本表示 `t ∈ R^(B×D)` 和帧表示 `F ∈ R^(B×M×D)`；随机文本模块输出 `S` 个候选，形状为 `S×B×D`。实现中 `D=512`，默认候选数 `S=20`。
- **核心计算**：`LinearCosRadius` 以文本和各帧的余弦相似度预测每个维度的 `log_var`，再通过重参数化生成候选：

  \[
  t^{\mathrm{sto}}=t+\epsilon\odot\exp(\mathrm{log\_var}) .
  \]

  原始文本与所有候选一同进入 FRL；原始文本还参与后续 support 文本的构造。
- **阶段与依赖**：训练与推理都生成候选。其输出与原始文本、帧表示共同构成关系图节点。
- **核心代码**：[`StochasticText.stochastic_ntimes`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/stochastic_module.py:92) 生成候选；[`EagleNet.get_max_similarity_logits`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:353) 将每个文本—视频组合展开后调用该模块。
- **模块结果**：组件消融中的基线（未启用 FRL、EAM 与 sigmoid 损失）在 MSRVTT/DiDeMo 的文本到视频 R@1 为 48.5/42.1；候选文本由 FRL 进一步筛选、融合后形成完整的增强表示。

### 3.2 FRL：细粒度关系学习

- **作用**：将视频帧之间的上下文关系与文本—帧关联共同用于选择、融合文本候选。
- **输入与输出**：对一个文本—视频对，FRL 使用 `M+1+S` 个节点：`M` 个加入可学习时间位置编码的帧、1 个原始文本与 `S` 个随机文本候选。输出是增强文本 `t_gen ∈ R^D`。
- **核心计算**：图中包含文本—文本、帧—帧、文本—帧三种关系。RGAT 为每种关系维护独立的线性投影与注意力投影，先在关系内归一化邻居注意力并聚合消息，再通过残差投影更新节点。最终层的帧—文本关系分数依次跨注意力头、跨帧平均，再在原始文本及 `S` 个候选之间 softmax，得到融合权重：

  \[
  t_{\mathrm{gen}}=\sum_{i=1}^{S+1}w_i t_i .
  \]

- **阶段与依赖**：训练和推理均参与。`t_gen` 同时作为帧聚合的条件与检索损失中的文本表示。
- **核心代码**：[`EagleNet.get_max_similarity_logits`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:380) 构建三张邻接矩阵、抽取末层关系分数并聚合候选；[`RelationGraphAttention`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/gnn.py:139) 定义关系特定的注意力；[`RGAT`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/gnn.py:209) 堆叠多层关系图注意力。
- **模块结果**：仅加入 FRL 后，MSRVTT/DiDeMo 的文本到视频 R@1 为 48.8/47.9；FRL 与 EAM 组合后为 50.5/49.2。

### 3.3 文本条件帧聚合

- **作用**：将增强文本转化为帧选择条件，形成与当前候选文本—视频对相对应的全局视频表示。
- **输入与输出**：输入是 `t_gen ∈ R^(B_t×B_v×D)` 与帧表示 `F ∈ R^(B_v×M×D)`；输出为 `v ∈ R^(B_v×B_t×D)`，即每个文本—视频组合对应一个聚合后的视频向量。
- **核心计算**：`TransformerXPool` 以增强文本为 Query、视频帧为 Key/Value 进行多头交叉注意力。文本与聚合后的视频向量均做 L2 归一化，再以 `einsum` 计算 `B_t×B_v` 的检索 logits。
- **阶段与依赖**：训练与推理均参与，承接 FRL 的 `t_gen`，直接产生用于排序的全局视频表示。
- **核心代码**：[`Transformer.forward`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/transformer_eaglenet.py:91) 实现帧聚合；[`EagleNet.get_max_similarity_logits`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:426) 完成帧池化、归一化和检索 logits 计算。
- **模块结果**：该模块是完整检索路径的一部分。完整 EagleNet 在 MSRVTT ViT-B/16 上达到文本到视频 R@1/R@5/R@10 为 51.0/76.2/85.6。

### 3.4 EAM：能量感知匹配

- **作用**：在全局检索目标之外，对逐文本—帧互动建立能量监督，使 FRL 的关系学习同时感知匹配对分布。
- **输入与输出**：输入为帧表示 `F ∈ R^(B×M×D)` 与原始文本或 support 文本 `t ∈ R^(B×D)`；输出为训练期能量损失 `L_eam`，不产生推理期 logits。
- **核心计算**：EAM 先计算逐帧能量，再沿帧维度池化。实现提供三类函数：负余弦、带可学习映射的双线性函数、MLP。双线性形式为：

  \[
  E(t,F)=-\frac{1}{M}\sum_{m=1}^{M}
  \frac{(Wt)^\top f_m}{\lVert Wt\rVert\lVert f_m\rVert} .
  \]

  `loss_compute` 将批内对角线配对作为正能量项、非对角组合与 Langevin/MCMC 生成样本作为负能量项，并通过能量差与正则项形成训练损失。
- **阶段与依赖**：仅训练期参与。它分别作用于原始文本和 support 文本，与检索损失相加；推理只保留 FRL 与文本条件帧聚合。
- **核心代码**：[`EBM.energy` 与 `EBM.energy_diag`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/ebm.py:53) 计算帧级及成对能量；[`EBM.loss_compute`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/ebm.py:94) 组合批内与 MCMC 样本；[`EagleNet.get_max_similarity_logits`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:444) 接入 EAM 损失。
- **模块结果**：仅加入 EAM 时，MSRVTT/DiDeMo 的文本到视频 R@1 为 49.0/43.4；与 FRL 组合后提升至 50.5/49.2。能量函数比较中，Bilinear 在 MSRVTT 达到 51.0 R@1，MLP 在 DiDeMo 达到 51.5 R@1。

### 3.5 support 文本与 Sigmoid 匹配目标

- **作用**：support 文本为随机候选范围提供辅助对齐分支；sigmoid 损失对每一个文本—视频对独立施加正负匹配目标。
- **输入与输出**：由原始文本、其 `log_var` 和平均池化视频表示形成 `t_sup`，并产生 support 检索矩阵。`SigLoss` 接收方阵 logits，输出单个匹配损失。
- **核心计算**：源码以视频表示与原始文本的差向量作为方向，按 `exp(log_var)` 缩放并加回原文本得到 `t_sup`。`SigLoss` 对相似度施加可学习温度 `τ` 与偏置 `b`，并以对角线正标签、其余负标签计算：

  \[
  L_{\mathrm{sig}}=-\frac{1}{B}\sum_{i,j}
  \log\sigma\bigl(y_{ij}(\tau s_{ij}+b)\bigr) .
  \]

- **阶段与依赖**：support 分支与 sigmoid 损失仅训练期参与；推理输出仅使用 `t_gen` 和条件帧聚合得到的主检索 logits。
- **核心代码**：[`EagleNet.forward`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:300) 组合主、support 与能量损失；[`EagleNet.get_max_similarity_logits`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:433) 计算 support 文本及其 logits；[`SigLoss`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/until_module.py:194) 实现逐对 sigmoid 目标。
- **模块结果**：FRL+EAM 的平均 R@1 为 49.9；加入 sigmoid 匹配后，完整 EagleNet 的平均 R@1 为 51.3。

## 4. 端到端训练与推理

### 训练

1. [`EagleNet.forward`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:300) 将文本 token 编码为文本向量，将视频张量编码为帧序列表示。
2. `get_max_similarity_logits` 将批内每个文本与每个视频组合展开，采样文本候选，构建 FRL 图并得到 `t_gen`。
3. 文本条件帧聚合为每个组合生成视频表示；归一化点积形成主检索矩阵。随后构造 `t_sup` 及其辅助检索矩阵。
4. 源码中的总目标为：

   \[
   L=L_{\mathrm{sig}}(t_{\mathrm{gen}},v)
   +\lambda_{\mathrm{sup}}L_{\mathrm{sig}}(t_{\mathrm{sup}},v)
   +\lambda_{\mathrm{eam}}L_{\mathrm{eam}}(t,F)
   +\lambda_{\mathrm{eam,sup}}L_{\mathrm{eam}}(t_{\mathrm{sup}},F).
   \]

   训练循环在 [`train_epoch`](/data2/hxj/project/tvr/research_refs/EagleNet/train_and_eval.py:14) 中调用模型并执行反向传播与参数更新。

### 推理

1. [`eval_epoch`](/data2/hxj/project/tvr/research_refs/EagleNet/train_and_eval.py:90) 先缓存所有文本与视频帧特征。
2. [`_run_on_single_gpu`](/data2/hxj/project/tvr/research_refs/EagleNet/train_and_eval.py:67) 对文本块和视频块调用 `get_max_similarity_logits`；此时该函数返回主检索矩阵及其转置。
3. 推理保留候选文本、FRL 与文本条件帧聚合，不计算 support 和 EAM 损失。拼接得到的相似度矩阵后，按分数进行双向排序。

## 5. 核心代码实现映射

| 设计模块 | 代码入口 | 核心对象/函数 | 输入 → 输出 | 直接职责 |
| --- | --- | --- | --- | --- |
| 模型总调度 | [`modules/modeling.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:161) | `EagleNet` | tokens、视频 → loss 或 logits | 组装 CLIP、候选、FRL、池化和 EAM |
| 文本/帧编码 | [`modules/modeling.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:506) | `get_sequence_visual_output` | tokens、图像帧 → `B×D`、`B×M×D` | 形成初始跨模态表示 |
| 随机文本候选 | [`modules/stochastic_module.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/stochastic_module.py:92) | `stochastic_ntimes` | 文本、帧 → `S×B×D` | 重参数化采样候选文本 |
| FRL/RGAT | [`modules/modeling.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/modeling.py:380) | `get_max_similarity_logits` | 图节点 → `t_gen` | 建三类关系并融合文本候选 |
| 关系注意力 | [`modules/gnn.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/gnn.py:169) | `RelationGraphAttention.forward` | 节点、关系邻接 → 节点/边分数 | 对不同关系独立注意力聚合 |
| 条件帧聚合 | [`modules/transformer_eaglenet.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/transformer_eaglenet.py:91) | `Transformer.forward` | `t_gen`、帧 → 视频表示 | 文本条件化帧池化 |
| 帧级能量 | [`modules/ebm.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/ebm.py:94) | `EBM.loss_compute` | 帧、文本 → `L_eam` | 细粒度能量监督与采样负例 |
| 成对匹配 | [`modules/until_module.py`](/data2/hxj/project/tvr/research_refs/EagleNet/modules/until_module.py:194) | `SigLoss` | logits → `L_sig` | 独立优化文本—视频对 |
| 批量评估 | [`train_and_eval.py`](/data2/hxj/project/tvr/research_refs/EagleNet/train_and_eval.py:67) | `_run_on_single_gpu` | 特征块 → 相似度矩阵 | 拼接候选对检索分数 |

## 6. 实现结果

### 6.1 功能性结果

- 最终检索产物是每个文本—视频组合的增强文本 `t_gen`、由其条件化的全局视频表示 `v`，以及二者归一化点积组成的相似度矩阵。
- FRL 改变进入最终相似度计算的文本表示；文本条件帧聚合相应地产生组合特定的视频表示。
- support 分支、Sigmoid 匹配损失和 EAM 损失共同塑造训练目标；推理阶段输出不包含这些辅助损失。
- 训练脚本的 ViT-B/16 配置使用 12 帧、20 个文本候选、RGAT、Bilinear 能量函数与平均帧能量池化：[`run_train_msrvtt9k-vit_b_16.sh`](/data2/hxj/project/tvr/research_refs/EagleNet/run_train_msrvtt9k-vit_b_16.sh:7)。

### 6.2 实验结果

- **主结果**：ViT-B/16 下，MSRVTT 的文本到视频 R@1/R@5/R@10 为 51.0/76.2/85.6，视频到文本 R@1/R@5/R@10 为 49.2/77.1/86.6，Rsum 为 425.7。DiDeMo、MSVD、VATEX 的文本到视频 R@1 分别为 51.5、50.9、63.6。
- **组件消融**：基线、FRL、EAM、FRL+EAM 的 MSRVTT/DiDeMo R@1 分别为 48.5/42.1、48.8/47.9、49.0/43.4、50.5/49.2；完整 EagleNet 为 51.0/51.5。
- **关键设计选择**：能量函数比较中，CosSim、Bilinear、MLP 在 MSRVTT 的 R@1 为 50.0、51.0、50.3；在 DiDeMo 的 R@1 为 50.2、51.3、51.5。帧级能量的 Avgpool 在 MSRVTT/DiDeMo 达到 51.0/51.5 R@1。

## 7. 设计闭环总结

- 短文本无法充分表达视频上下文 → **随机候选文本与 FRL 的文本—文本、帧—帧、文本—帧关系学习** → 生成视频条件化的增强文本；FRL 将 DiDeMo R@1 从 42.1 提升至 47.9。
- 全局检索匹配缺少逐帧对齐信号 → **EAM 的逐文本—帧能量建模** → FRL+EAM 达到 MSRVTT/DiDeMo R@1 为 50.5/49.2。
- 检索矩阵需要逐对建立对齐目标 → **support 分支与 sigmoid 匹配损失** → 完整 EagleNet 达到 MSRVTT/DiDeMo R@1 为 51.0/51.5，并在 MSVD、VATEX 分别达到 50.9、63.6。
