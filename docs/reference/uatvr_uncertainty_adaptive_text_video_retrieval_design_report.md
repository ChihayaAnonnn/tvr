# 方法设计报告：UATVR（Uncertainty-Adaptive Text-Video Retrieval）

## 1. 目标问题

- **任务与输入输出**：给定文本查询集合与视频库，学习文本—视频相似度函数 `s(t, v)`，分别完成 Text→Video 与 Video→Text 排序。每个文本由词级 token 构成，每个视频由采样帧构成；模型最终输出文本数 × 视频数的检索分数矩阵。
- **具体缺口**：一个视频往往同时含有多个事件、对象和时间片段，而一句描述只覆盖其中的部分语义。固定的全局表示或单一粒度的词—帧配对难以决定当前查询应对应哪些词、帧或高层语义组合；一个视频与多个合理描述之间也不是单个确定特征点能够充分表达的关系。
- **设计目标**：UATVR 以互补的两条路径处理该问题：Dynamic Semantic Adaptation（DSA）在确定性 token 空间中动态汇聚多粒度语义，Distribution-based Uncertainty Adaptation（DUA）把每个模态表示为概率分布，以多实例对比学习对齐同一文本—视频对的多种采样表示。

## 2. 总体解决思路

```text
文本 token / 视频 M 帧
        │
        ▼
CLIP 文本编码器 / CLIP 帧编码器
        │  词表示 B × N × D，帧表示 B × M × D
        ▼
DSA：追加 C_t / C_v 个可学习聚合 token，并经 SeqTransf 更新
        ├─────────────────────────────────────────────────────┐
        ▼                                                     ▼
加权双向 word–frame max matching                         池化非额外 token
        │                                                     │
        ▼                                                     ▼
最终检索 logits S                                      DUA：μ、logsigma、K 个高斯样本
        │                                                     │
        └────── CrossEn 双向检索损失 ───────┬──── MIL 多实例对比 + KL（训练期）
                                             ▼
                                            总损失
```

实现采用 CLIP 产生逐词和逐帧表示；`seqTransf` 让追加的聚合 token 与原有序列双向交互。随后，DSA 路径通过“每个词找最佳帧、每帧找最佳词”的双向匹配产生最终检索分数。DUA 路径从同一批经 DSA 更新的词/帧表示中得到高斯均值和尺度，采样多个原型，并把同一原始文本—视频对的全部跨模态样本作为正集合。官方启动命令选择 `--loose_type --sim_header seqTransf`，因此上述流程是实际训练与评估主线。[`train.sh`](/data2/hxj/project/tvr/research_refs/UATVR_official/train.sh:1)

## 3. 核心模块设计

### 3.1 CLIP 双塔编码与词—帧表示

- **作用**：保留文本中的词级语义和视频中的帧级语义，为后续按查询选择匹配粒度提供最细粒度的共同表示空间。
- **输入与输出**：文本 token 与 attention mask 经文本编码器得到全局 EOT 表示及 `B × N × D` 的词表示；`B` 个视频的 `M` 帧图像经视觉编码器得到 `B × M × D` 的帧表示。官方 ViT-B/16 启动配置中 `D=512`、`N≤32`、`M≤12`。
- **核心计算**：每帧由视觉 Transformer 的 `[CLS]` 输出投影为一条帧表示；文本 Transformer 返回末端全局表示和全部词位置的隐藏表示。UATVR 以这些词和帧，而非单一视频平均向量，作为 DSA 与 DUA 的共同输入。
- **阶段与依赖**：训练和推理均执行；其词/帧序列同时交给 DSA 的 `seqTransf`，再由更新后的非额外 token 被池化供 DUA 使用。
- **核心代码**：[`UATVR.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:297) 调度两侧编码；[`get_sequence_output`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:352) 和 [`get_visual_output`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:368) 整理输出；每帧 `[CLS]` 表示的形成位于 [`VisualTransformer.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/module_clip.py:318)。
- **模块结果**：论文将词—帧 Token-Wise Interaction（TI）作为基线；在 MSR-VTT、ViT-B/16 的 Text→Video 消融中，TI 得到 R@1 48.4、R@5 74.2、R@10 83.3、MdR 2.0、MnR 14.1。

### 3.2 DSA：可学习聚合 token 与加权 token-wise matching

- **作用**：根据具体跨模态查询，在原始词/帧之外形成可吸收不同粒度语义的聚合位置，使“整段视频—句子”“局部帧—词”等候选对应可以共存于一次匹配中。
- **输入与输出**：输入为 `B × M × D` 帧序列、`B × N × D` 词序列及其 mask；输出为扩展后的 `B × (M+C_v) × D`、`B × (N+C_t) × D` 序列，以及任意文本批与视频批之间的确定性分数矩阵 `S ∈ R^(B_t × B_v)`。
- **核心计算**：
  1. `seqTransf` 分别把 `C_v` 个视频 token、`C_t` 个文本 token 追加到序列尾部；这些额外位置从可学习的位置嵌入开始，并与原始 token 一起通过轻量 `TransformerClip`。Transformer 输出把原始词/帧表示残差加回，因此原 token 保留基础编码并获得序列上下文。
  2. 所有有效 token 归一化后，文本和视频各用一个两层 MLP 产生 softmax 权重。对任意查询 `i` 与候选 `j`，代码先计算 `R_ij,n,m = <w_i,n, f_j,m>`，再取双向最大匹配并加权：

     \[
     S_{ij}=\frac{1}{2}\left(\sum_n a_{i,n}\max_m R_{ij,n,m}
     +\sum_m b_{j,m}\max_n R_{ij,n,m}\right).
     \]

     最终将 `S` 乘以可学习温度 `exp(clip.logit_scale)`。这实现了论文中在“原始 token ∪ 额外 token”上进行双向匹配的 DSA 思路，并在代码中显式加入词/帧显著性加权。
- **阶段与依赖**：训练与推理均保留；它依赖双塔 token 序列，直接生成最终排名所用的 logits，并向 DUA 提供经上下文更新的原始词/帧表示。
- **核心代码**：扩展 token、mask、`TransformerClip` 和残差融合在 [`_loose_similarity`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:450)；双向加权最大匹配在 [`weighted_token_wise_intersection`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:577)；分数缩放与返回在 [`_loose_similarity`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:553)。
- **模块结果**：MSR-VTT 消融中，TI 加 DSA 的 R@1 从 48.4 提升至 49.6，R@5/R@10 为 75.5/84.9，MnR 从 14.1 降至 12.5。额外 token 数由命令行参数控制；论文该组消融的最佳组合为 `C_v=3, C_t=2`，而随附启动脚本给出了 `C_v=2, C_t=2` 的可执行配置。[`main_task_retrieval.py`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:106) [`train.sh`](/data2/hxj/project/tvr/research_refs/UATVR_official/train.sh:14)

### 3.3 DUA：概率均值、尺度与可微原型采样

- **作用**：把一个文本或视频从单一确定性点扩展为可采样的分布表示，使同一语义可通过多个原型参与跨模态对齐。
- **输入与输出**：输入是 DSA 更新后、去除额外 token 的帧/词序列及其平均池化向量；对每个模态输出归一化均值 `μ ∈ R^(B × D)`、`logsigma ∈ R^(B × D)` 和 `K` 个样本 `z ∈ R^(B × K × D)`。官方命令中视频与文本均取 `K=7`。
- **核心计算**：`PIENet` 先对帧或词进行注意力汇聚，经过门控残差和 LayerNorm 调整池化表示，得到均值基础并做 L2 归一化；`UncertaintyModuleImage` 以注意力汇聚结果和池化表示预测 `logsigma`。采样使用重参数化形式 `z^(k)=μ+exp(logsigma)⊙ε^(k)`，其中 `ε^(k)` 为标准高斯噪声，因而梯度可回传至均值与尺度网络。
- **阶段与依赖**：该分支依赖 DSA 后的词/帧表示，在训练时以样本集合生成概率对齐损失；它不直接构成推理阶段返回的 `retrieve_logits`，其作用是通过训练优化塑造 DSA 所用表示。
- **核心代码**：视频和文本的概率封装分别位于 [`probabilistic_video`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:602) 与 [`probabilistic_text`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:622)；均值调整由 [`PIENet`](/data2/hxj/project/tvr/research_refs/UATVR_official/prob_models/pie_model.py:42) 完成，尺度预测由 [`UncertaintyModuleImage`](/data2/hxj/project/tvr/research_refs/UATVR_official/prob_models/uncertainty_module.py:12) 完成，可微采样实现在 [`sample_gaussian_tensors`](/data2/hxj/project/tvr/research_refs/UATVR_official/prob_models/tensor_utils.py:35)。
- **模块结果**：在已有 DSA 的分支上加入 DUA 的 Multi-Instance 对比后，MSR-VTT R@1 为 50.1、MdR 为 1.5；再加入 KL 项形成完整 UATVR 后，R@1 达到 50.8、MdR 为 1.0。概率样本数消融显示，`K=1/3/5/7` 的 R@1 分别为 49.6/49.8/50.5/50.8。

### 3.4 双向检索、多实例对比与 KL 总目标

- **作用**：让确定性 DSA 分数完成批内正负配对，同时让同一原始文本—视频对的全部概率样本共同成为正集合，并以 KL 项约束分布表示。
- **输入与输出**：DSA 输入产生的分数矩阵为 `B × B`；DUA 样本展平后形成 `(B·K_v) × (B·K_t)` 点积矩阵。输出为一个反向传播的标量总损失。
- **核心计算**：
  - `CrossEn(S)` 和 `CrossEn(S^T)` 分别计算 Text→Video、Video→Text 的对角线正配对交叉熵，二者平均得到 `L_DSA`。
  - `MILNCELoss_BoF` 将同一个原始样本索引对应的全部 `K_v × K_t` 跨模态样本标为正组，分别计算两个方向并平均得到 `L_DUA`。
  - `KLdivergence` 根据两个模态的样本均值和 `logsigma` 计算相对于单位高斯的 `L_KL`。

  官方实现的总目标为：

  \[
  \mathcal L = \mathcal L_{\mathrm{DSA}}
  + 10^{-2}\mathcal L_{\mathrm{DUA}}
  + 10^{-4}\mathcal L_{\mathrm{KL}}.
  \]
- **阶段与依赖**：仅训练阶段计入总损失；`L_DSA` 直接监督最终 logits，`L_DUA` 与 `L_KL` 是分布分支的辅助监督。
- **核心代码**：总损失组装在 [`UATVR.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:313)；样本点积、双向 MIL 和加权返回在 [`_loose_similarity`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:526)；两种损失的实现位于 [`MILNCELoss_BoF`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/until_module.py:225) 和 [`KLdivergence`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/until_module.py:248)。
- **模块结果**：该模块的结果应与 DUA 组合解读：DSA + DUA + KL 的完整组合达到消融表中的最高 R@1 50.8，而非把完整成绩归因于某一个损失项。

## 4. 端到端训练与推理

### 训练

1. 数据加载器输出文本 id、文本 mask、视频帧张量和帧 mask；`UATVR.forward` 将文本展开为 `B × N`、将视频帧整理为编码器可处理的图像批。[`train_epoch`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:259) [`UATVR.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:297)
2. CLIP 生成词表示和帧表示；DSA 为两侧追加聚合 token，经 `seqTransf` 交互、归一化并进行加权双向词—帧最大匹配，得到批内 `S`。
3. 同时，DSA 更新后的非额外 token 被池化，经 PIE/不确定性头产生 `μ`、`logsigma` 和 `K` 个概率样本；展平的全部样本对输入多实例对比损失。
4. 以 `L_DSA + 0.01L_DUA + 0.0001L_KL` 反向传播。训练循环对该标量执行梯度裁剪、优化器更新，并限制 CLIP 的 `logit_scale` 上界。[`train_epoch`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:282)

### 推理

1. `eval_epoch` 先缓存所有文本的词表示和所有视频的帧表示，避免为每一个文本—视频组合重复运行 CLIP 编码器。[`eval_epoch`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:330)
2. `_run_on_single_gpu` 按文本块和视频块调用 `get_similarity_logits`，经 DSA 的 `weighted_token_wise_intersection` 拼接完整的 `N_text × N_video` 分数矩阵。[`_run_on_single_gpu`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:313) [`get_similarity_logits`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:642)
3. 每行按分数排序得到 Text→Video 结果，每列的相应排序得到 Video→Text 结果；评估函数据此计算 R@1、R@5、R@10、Median Rank 和 Mean Rank。[`compute_metrics`](/data2/hxj/project/tvr/research_refs/UATVR_official/metrics.py:9)

## 5. 核心代码实现映射

| 设计模块 | 代码入口 | 核心对象/函数 | 输入 → 输出 | 直接职责 |
| --- | --- | --- | --- | --- |
| 双塔 token/帧编码 | [`UATVR.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:297) | `get_sequence_output`、`get_visual_output` | 文本与帧图像 → `B × N × D` 词、`B × M × D` 帧 | 提供细粒度共同嵌入 |
| DSA 序列聚合 | [`_loose_similarity`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:432) | `TransformerClip`、位置嵌入、残差融合 | 词/帧 → 扩展 token 序列 | 让额外 token 汇聚多粒度语义 |
| DSA 检索评分 | [`weighted_token_wise_intersection`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:577) | `text_weight_fc`、`video_weight_fc` | 两个 token 序列 → `B_t × B_v` logits | 产生最终可排序相似度 |
| DUA 分布参数化 | [`probabilistic_video`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:602) | `PIENet`、`UncertaintyModuleImage` | 池化表示与序列 → `μ`、`logsigma` | 建立每个模态的概率表示 |
| DUA 重参数采样 | [`sample_gaussian_tensors`](/data2/hxj/project/tvr/research_refs/UATVR_official/prob_models/tensor_utils.py:35) | `sample_gaussian_tensors` | `B × D` 参数 → `B × K × D` 样本 | 提供多原型跨模态配对 |
| 训练目标 | [`UATVR.forward`](/data2/hxj/project/tvr/research_refs/UATVR_official/modules/modeling.py:313) | `CrossEn`、`MILNCELoss_BoF`、`KLdivergence` | logits/概率样本 → 标量损失 | 同时优化确定性检索与分布对齐 |
| 批量检索评估 | [`eval_epoch`](/data2/hxj/project/tvr/research_refs/UATVR_official/main_task_retrieval.py:330) | `_run_on_single_gpu`、`compute_metrics` | 缓存表示 → 全量分数矩阵与指标 | 执行两种检索方向的排名 |

## 6. 实现结果

### 6.1 功能性结果

- **最终表示**：模型保留经 DSA 更新的词和帧表示；DUA 为两侧表示学习 `μ` 与 `logsigma`，并在训练中产生 `K` 个概率原型。
- **最终检索 logits**：实际排名分数是 `exp(clip.logit_scale)` 缩放后的加权双向 token-wise 最大相似度 `S`。它由 DSA 路径直接输出，形状为文本候选数 × 视频候选数。
- **训练期辅助结果**：概率样本的 `MILNCELoss_BoF` 和 `KLdivergence` 仅作为训练总损失的两项；它们通过参数学习影响最终 DSA 表示，但不作为推理阶段返回的检索分数。

### 6.2 实验结果

- **主结果**：论文在 MSR-VTT 1K-A、ViT-B/16、未使用表中标记的后处理时报告 Text→Video R@1/R@5/R@10 为 **50.8/76.3/85.5**，MdR/MnR 为 **1.0/12.4**；Video→Text 为 **48.1/76.3/85.4**，MdR/MnR 为 **2.0/8.0**。同一论文在 VATEX、MSVD、DiDeMo 的 ViT-B/16 Text→Video R@1 分别为 **64.5、49.7、45.8**。

- **组件消融**：下表为论文在 MSR-VTT、ViT-B/16 上的 Text→Video 消融；`†` 表示 DUA 使用 Multi-Instance InfoNCE。

  | 组件组合 | R@1 | R@5 | R@10 | MdR↓ | MnR↓ |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | TI（词—帧基线） | 48.4 | 74.2 | 83.3 | 2.0 | 14.1 |
  | TI + DSA | 49.6 | 75.5 | 84.9 | 2.0 | 12.5 |
  | TI + DSA + DUA† | 50.1 | 75.8 | 84.6 | 1.5 | 12.8 |
  | TI + DSA + DUA† + KL（UATVR） | 50.8 | 76.3 | 85.5 | 1.0 | 12.4 |

- **关键设计选择**：额外 token 消融中，`C_v=3, C_t=2` 的配置对应完整 DSA 分支；概率样本数从 `K=1` 增至 `K=7` 时，R@1 从 49.6 提升至 50.8。帧数消融中，12 帧配置得到 50.8 R@1，论文将其作为默认帧数。

## 7. 设计闭环总结

- **视频内容与文本描述的粒度不确定** → **CLIP 词/帧表示 + DSA 额外聚合 token + 双向最大匹配** → 在同一相似度函数中保留局部与高层语义，并把 TI 的 R@1 从 48.4 提升至 49.6。
- **同一视频—文本关系具有多种合理对应** → **DUA 的高斯均值、尺度、可微多原型采样与多实例正集合** → 用全部同源样本共同对齐，在 DSA 之上把 R@1 提升至 50.1。
- **概率表示需要与最终检索目标协同优化** → **双向 CrossEn + 加权 MIL + KL 总目标** → 形成完整 UATVR，MSR-VTT Text→Video 达到 R@1 50.8、MdR 1.0。
