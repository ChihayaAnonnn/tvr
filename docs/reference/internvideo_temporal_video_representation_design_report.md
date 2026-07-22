# 方法设计报告：InternVideo 时序视频表征模块（Video-Text-Retrieval）

## 1. 目标问题

- **任务与输入输出**：在文本—视频检索中，模型接收一段由 `T` 帧组成的视频和一条文本描述，分别编码为共享的 `D` 维语义向量，并以向量相似度作为文本检索视频或视频检索文本的排序依据。本文聚焦视频侧的时序表征：`B × 3 × T × H × W` 的视频张量如何变为 `B × D` 的视频级表示。
- **具体缺口**：原始 CLIP ViT 擅长单帧图像—文本对齐，但其空间 Transformer 将每帧视为独立图像，无法直接表达动作顺序、跨帧状态变化和长于单帧的事件语义。
- **设计目标**：保留 CLIP 已有的空间语义与图文对齐空间，仅在高层引入轻量、分阶段的时序汇聚，使视频表示既包含局部运动线索，也能从全部帧与空间位置选择检索相关证据。

## 2. 总体解决思路

```text
视频 B×3×T×H×W
  → CLIP ViT 的逐帧空间 token
  → 末四层 DPE 深度可分离 3D 卷积
  → temporal_cls 对 T×L 时空 token 的 Cross-Attention
  → 与帧级 CLS 均值进行门控残差融合
  → LayerNorm + CLIP visual projection → 视频向量 B×D
                                                  │
文本 token → CLIP 文本向量 B×D ───────────────────┴→ 相似度矩阵与双向对比训练
```

InternVideo 的通用框架包含掩码视频编码器和多模态视频编码器；本检索子项目实际使用后者，即以 CLIP 为基础、在 ViT 高层插入 global UniBlocks 的 UniFormerV2 风格时序视觉编码器。视频—文本对比学习将视频和文本映射到同一空间；字幕解码器作为上游多模态预训练辅助，检索推理直接使用视频 encoder 的输出。[InternVideo 论文](https://arxiv.org/abs/2212.03191)

## 3. 核心模块设计

### 3.1 CLIP 空间 token 骨干

- **作用**：复用 CLIP 已学习的对象、场景和语言语义对齐，作为时序模块的空间证据来源。
- **输入与输出**：输入为 `B × 3 × T × H × W`。Patch 投影后，每一帧得到 `L=1+H'W'` 个 `C` 维 token；实现将帧维并入 batch，得到 `L × (B·T) × C`。
- **核心计算**：Patch 投影卷积的核为 `(1, patch, patch)`，先对每帧做空间 token 化和 CLIP ViT 编码；时间关系由后续的 DPE 与 `Extractor` 引入，而不是在输入 patch 层混合相邻帧。
- **阶段与依赖**：训练与推理均保留；输出的高层空间 token 依次交给四个时序融合阶段。
- **核心代码**：[`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:275>) 中的 `VisionTransformer.forward` 构造帧级 token 并调用 `Transformer`。
- **模块结果**：为每个视频提供跨层时序汇聚所需的每帧 `cls` token 与空间 patch token；该骨干与后续模块共同构成论文报告的检索结果。

### 3.2 多阶段 DPE：局部时空增强

- **作用**：让高层空间 token 感知相邻帧和相邻空间位置的局部变化，例如运动方向、姿态变化和物体状态转换。
- **输入与输出**：在被选中的 ViT 层，非 `cls` token 被还原为 `B × C × T × H' × W'`；DPE 输出同形状的局部时空增量，再被加回空间 token。
- **核心计算**：每个阶段使用一层逐通道 `3×3×3` `Conv3d`：每个通道独立进行局部时空滤波，保持 CLIP 通道语义，同时为全局注意力准备带运动线索的 Key/Value。
- **阶段与依赖**：仅在选中的高层 ViT block 后执行。ViT-B/16 选择第 `8–11` 层，ViT-L/14 选择第 `20–23` 层；DPE 的输出随即送入该阶段的 `Extractor`。
- **核心代码**：[`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:178>) 中的 `self.dpe` 定义逐通道 3D 卷积，[`Transformer.forward`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:219>) 完成 token 重排、DPE 与残差相加；层选择在 [`model.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/model.py:351>) 配置。
- **模块结果**：产生包含局部时空变化的 `T × L` 个证据 token，供视频级查询从全片选择信息。

### 3.3 `temporal_cls_token` 与 Extractor：全局视频摘要

- **作用**：将所有帧、所有空间位置以及多个高层语义阶段的信息汇集为一个视频级语义向量。
- **输入与输出**：每段视频从可学习的 `temporal_cls_token ∈ R^(1×B×C)` 开始。当前层的 DPE 增强 token 被展平为 `y ∈ R^((T·L)×B×C)`；输出为更新后的 `temporal_cls_token`。
- **核心计算**：每个 `Extractor` 先用视频摘要 token 作 Query、以全部时空 token 作 Key/Value，执行跨注意力，再接残差 MLP：

  \[
  q' = q + \operatorname{MHA}(\operatorname{LN}(q),\operatorname{LN}(y),\operatorname{LN}(y)),\qquad
  q'' = q' + \operatorname{MLP}(\operatorname{LN}(q')).
  \]

  四个高层阶段顺序更新同一个 `q`，因而早期阶段的局部运动线索和后期阶段的语义线索都会写入视频摘要。
- **阶段与依赖**：训练与推理均使用；依赖 DPE 产生的时空 token，输出交给下一阶段的 `Extractor` 或最终残差融合。
- **核心代码**：[`Extractor.forward`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:146>) 实现 Cross-Attention 与 MLP 更新；[`Transformer.forward`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:203>) 创建 `temporal_cls_token`、展平 `T×L` token 并逐阶段调用 `self.dec[j]`。
- **模块结果**：得到以全视频为感受域的时序摘要 `q`，而不是独立帧特征的简单堆叠。

### 3.4 CLIP 语义残差、门控融合与投影

- **作用**：在学习时序信息的同时保留原 CLIP 帧级语义及其图文对齐方向。
- **输入与输出**：输入为时序摘要 `q` 与末层所有帧 `cls` token 的平均表示 `r ∈ R^(B×C)`；输出为共享检索空间中的视频向量 `f_v ∈ R^(B×D)`。
- **核心计算**：先计算帧级残差 `r`，再以逐通道门控 `w=σ(balance)` 融合：

  \[
  f=(1-w)q+w r.
  \]

  随后经过 `visual_ln_post` 和原 CLIP 的 `visual_proj` 映射到 `D` 维。新增 `Extractor` 的注意力输出投影与 MLP 输出投影采用零初始化，初始时融合方向保持接近原 CLIP，随后在视频预训练和检索微调中学习时序偏移。
- **阶段与依赖**：训练与推理均使用；输出直接成为视频—文本相似度计算中的视频表示。
- **核心代码**：[`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:233>) 计算残差和门控融合；[`model.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/model.py:201>) 中的 `encode_video` 完成 LayerNorm 与 projection。
- **模块结果**：生成一个包含时序动态、仍可与 CLIP 文本向量直接点积的 `B × D` 视频表示。

## 4. 端到端训练与推理

### 训练

1. **上游视频表征学习**：论文先以 CLIP 初始化多模态视觉编码器，用视频/图像—文本对比学习建立共享嵌入空间，并以字幕生成的跨注意力解码器辅助预训练；之后以动作识别进行视频后训练。时序全局 UniBlocks 被插入 ViT-B/L 的最后四层。
2. **检索微调编码**：`CLIP4Clip.from_pretrained` 根据 `--clip_evl` 加载时序视觉模型；`get_visual_output` 将帧序列重排为 `B × C × T × H × W`，调用 `encode_video`。默认路径返回一个视频级向量，再形状化为 `B × 1 × D`。见 [`modeling.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/modeling.py:395>)。
3. **相似度计算**：视频与文本向量分别归一化后，以 CLIP 的可学习温度计算点积相似度矩阵 `S = exp(logit_scale) · T V^⊤`。默认 `meanP` 路径见 [`_loose_similarity`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/modeling.py:473>)。
4. **总损失**：训练对 `S` 和 `S^⊤` 分别使用对角正例交叉熵，并取二者平均，从而同时优化文本→视频与视频→文本检索。模型训练 `forward` 位于 [`modeling.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/modeling.py:338>)；外部训练循环位于 [`main_task_retrieval.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/main_task_retrieval.py:339>)。

### 推理

1. 保留 CLIP 空间编码、DPE、四阶段 `Extractor` 与门控残差融合，分别缓存文本表示和每段视频的 `B × D` 表示。
2. 计算全体文本—视频点积矩阵后，评估代码可使用 Dual Softmax 进行检索分数重加权：`S_dsl = S * softmax(S, axis=0)`；反向检索在转置矩阵上执行。见 [`eval_epoch`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/main_task_retrieval.py:408>)。

## 5. 核心代码实现映射

| 设计模块 | 代码入口 | 核心对象/函数 | 输入 → 输出 | 直接职责 |
| --- | --- | --- | --- | --- |
| 时序模型装载 | [`modules/modeling.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/modeling.py:254>) | `clip_evl.load` | 预训练权重、`max_frames` → 时序 CLIP | 选择 InternVideo 时序视觉路径 |
| 空间 token 化 | [`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:275>) | `VisionTransformer.forward` | `B×3×T×H×W` → `L×(B·T)×C` | 帧级 CLIP 空间表示 |
| 局部时空增强 | [`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:181>) | `self.dpe` | `B×C×T×H'×W'` → 同形状增量 | 提取局部时间和空间变化 |
| 全局时序汇聚 | [`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:146>) | `Extractor` | `q`、`T×L` token → 更新的 `q` | 从全片选择并写入证据 |
| 视频级表示 | [`clip_vit_only_global.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/evl_utils/clip_vit_only_global.py:233>) | `balance`、残差融合 | `q`、帧 CLS 均值 → `B×C` | 保留 CLIP 语义并注入时序语义 |
| 共享检索空间 | [`model.py`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/modules/clip_evl/model.py:201>) | `encode_video` | `B×C` → `B×D` | 投影为可与文本匹配的视频向量 |

## 6. 实现结果

### 6.1 功能性结果

- 时序视觉 encoder 对每段视频输出一个 `D` 维向量；默认检索路径中，这个向量已汇聚 `T` 帧的时序信息，而非在检索头中再对独立帧向量进行主要建模。
- 文本 encoder 输出同一语义空间的向量；二者经归一化点积形成检索 logits。
- 字幕解码器只服务于上游多模态预训练；Dual Softmax 只服务于检索分数的评估期重加权。

### 6.2 实验结果

- **主结果**：论文在 MSR-VTT 的全量微调检索 R@1 达到 Text→Video `55.2`、Video→Text `57.9`；同表的 CLIP4Clip 为 `45.6`、`45.9`。在零样本 MSR-VTT 上，InternVideo 为 `40.7`、`39.6`。复制目录的 [`README.md`](</data2/hxj/project/tvr/research_refs/Video-Text-Retrieval/README.md:7>) 还给出了六个检索基准的完整 R@K、MedR 和 MeanR。
- **组件组合**：上述数值对应“CLIP 初始化的时序视觉 encoder、视频—文本对比学习、检索集微调与 Dual Softmax 后处理”的完整组合；论文的检索主表未逐项分离 DPE、`Extractor` 与残差门控的检索 R@1。
- **关键设计选择**：将 global UniBlocks 放在 ViT 最后四层、多阶段累积时序摘要、以稀疏采样的 224 分辨率视频训练，并以保持原 CLIP 输出方向的初始化作为时序模块接入方式。

## 7. 设计闭环总结

- **逐帧 CLIP 缺少局部运动信息** → DPE 在高层空间 token 上执行逐通道 `3×3×3` 时空滤波 → 为动作和状态变化提供局部证据。
- **单个帧或单层特征无法概括视频事件** → `temporal_cls_token` 在四个高层阶段对全部 `T×L` token 做 Cross-Attention → 形成全视频时序摘要。
- **直接改造 CLIP 可能破坏既有图文对齐** → 帧级 CLS 残差、逐通道门控与零初始化输出投影 → 让时序信息进入共享检索空间，同时保留 CLIP 的语义方向。
