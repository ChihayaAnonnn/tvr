# 多模态文本—视频检索研究综合分析

> 本文记录外部论文事实、复算和公平性审计，不是 UATVR 科研决策的单一事实源。
> 当前优先级、停止条件和实验顺序只见 [科研路线图](../project/RESEARCH_ISSUES_AND_ROADMAP.md)。

## 1. 分析范围与证据规则

### 1.1 纳入论文与版本

本文核验五份本地 PDF 和三份本地参考实现。PDF 页码一律指阅读器显示的物理页，不把论文印刷页码或 `pdftotext` 行号当作稳定定位。GraviAlign 未找到作者公开实现，因此只能核验论文公式和报告数字，不能核对真实数据流。

| 论文与证据源 | 版本 | PDF 页数 | 补充材料 | 本地代码 | 复核状态 |
|---|---:|---:|---|---|---|
| EagleNet | CVPR 2026 | 11 | 主文附录 | `research_refs/EagleNet/` | 已核验 |
| GARE | NeurIPS 2025 | 34 | 同一 PDF | `research_refs/GARE-text-video-retrieval/` | 已核验 |
| TempMe | ICLR 2025 | 22 | 同一 PDF | `research_refs/TempMe/` | 已核验 |
| GraviAlign | CVPR 2026 | 11 | 单独 1 页 | 未找到 | 论文已核验、实现不可复算 |

### 1.2 证据等级

全文区分四类陈述，不能互相替代：

- **论文事实**：可在公式、表格、图注或实验设置中直接定位；只说明作者实际写了什么、报告了什么。
- **作者主张**：论文对机制、理论或泛化的解释；若没有相应推导、对照或统计证据，不能升级为事实。
- **本地代码事实**：本地公开实现中真实执行的数据流、split、选模和打分逻辑，使用 `path:line` 定位。
- **本项目推论**：面向 UATVR 的判断或新假设，必须经过本项目协议下的独立实验才能成立。

证据标签含义如下：“已核验”表示原始载体与本文一致；“弱证据”表示有单次结果或消融但缺少统计闭环；“未验证”表示没有直接证据；“协议不兼容”表示 split、选模、正例定义或 test 使用方式不能与 `trusted-v1` 直接比较；“不可复算”表示缺少实现、日志或必要细节。

### 1.3 数值复算与协议比较规则

绝对差按 `方法值 - 基准值` 计算，相对差按 `绝对差 / 基准值` 计算；论文把绝对百分点写成百分比时，本文会显式区分。只在同一表、同一 backbone、同一方向、同一 split 内复算。四篇论文的绝对 Recall 均不能与本项目 `trusted-v1` 结果直接归因，原因至少包括训练集组成、选模集合、正例矩阵和 forward contrastive batch 不同。

“达到最佳”不是本文的唯一判据。更重要的证据依次是：问题是否真实存在、机制是否直接作用于该问题、单变量消融是否闭环、训练与推理是否一致、结论是否跨 seed/数据集稳定、代价是否完整报告。未在每个数据集和每个指标领先，不等于研究失败；反之，单个最好数字也不能替代机制证据。

## 2. EagleNet

### 2.1 研究矛盾

**论文事实**：EagleNet 认为短文本表达不足，而已有文本扩展方法主要使用文本—视频/帧关系，没有把视频内部帧—帧上下文用于选择扩展语义（PDF 物理页 1–3）。其目标不是再换视觉编码器，而是让“同一句文本面对不同候选视频”形成不同的富化表示。

**本项目推论**：这是一个候选条件化的错配问题。若固定文本表示已经能在候选间稳定排序，复杂关系图没有必要；若错误集中在语义相近但时序/上下文不同的候选，才有引入候选交互的理由。

### 2.2 核心创新与数据流

**论文事实**：数据流是 CLIP 文本/帧特征 → 随机生成 `S=20` 个文本候选 → 将原文本、候选文本和候选视频的帧构成含 text-text、frame-frame、text-frame 三类边的图 → RGAT 学习关系权重 → 用 text-frame 权重聚合候选文本形成富化文本 → 文本条件的帧池化得到视频表示（公式 4–9，PDF 物理页 3–4）。EAM 另外用逐帧能量的平均、最大或最小池化约束真实配对分布（公式 11–16，PDF 物理页 4–5）。

**本地代码事实**：实现先把每个文本复制到所有候选视频，形成 `bst × bsv` 对，再对每一对采样文本并构图；见 `research_refs/EagleNet/modules/modeling.py:370`、`research_refs/EagleNet/modules/modeling.py:377`、`research_refs/EagleNet/modules/modeling.py:393`。RGAT 的三类邻接矩阵与聚合位于 `research_refs/EagleNet/modules/modeling.py:380`、`research_refs/EagleNet/modules/modeling.py:411`、`research_refs/EagleNet/modules/modeling.py:417`。这证明 FRL 不是独立编码后的一次矩阵乘法。

### 2.3 与常规检索表示的实质差异

普通双塔为每个文本和视频各产生一个固定向量，文本向量与候选集合无关。EagleNet 的富化文本 `t'_{ij}` 依赖候选视频 `v_j`，其视频池化也以 `t'_{ij}` 为条件；代码最终张量形状为 `(bst, bsv, d)`，见 `research_refs/EagleNet/modules/modeling.py:423`、`research_refs/EagleNet/modules/modeling.py:426`。因此可以缓存 CLIP 帧特征，却不能把完整 EagleNet score 直接转换为固定向量 ANN 检索。

随机候选表示“可能的文本语义”，但论文没有把其方差校准为错误概率。它是一种随机表示扩展，不等同于可用于风险决策的不确定性估计。

### 2.4 训练目标与最终检索分数

**论文事实**：最终训练目标由富化文本匹配、原文本 support 匹配和 EAM 组成；主文用 pair-wise sigmoid loss 取代 batch-softmax InfoNCE（公式 18–19，PDF 物理页 5）。作者理由是 sigmoid 对每个配对独立建模，较少依赖 batch 内负样本组成。

**本地代码事实**：`loss_fn=sig` 时使用 `SigLoss`，并叠加 support/EAM loss，见 `research_refs/EagleNet/modules/modeling.py:263`、`research_refs/EagleNet/modules/modeling.py:330`、`research_refs/EagleNet/modules/modeling.py:341`。真正用于排序的是归一化后的 pair-conditioned 富化文本和池化视频的余弦值，见 `research_refs/EagleNet/modules/modeling.py:428`、`research_refs/EagleNet/modules/modeling.py:431`；EAM 在公开实现中是训练正则，而不是额外加到最终检索 logit 的独立分数。

### 2.5 实验设计与协议

**论文事实**：实验覆盖 MSR-VTT、DiDeMo、MSVD、VATEX；使用 CLIP ViT-B/32 与 ViT-B/16，MSR-VTT/MSVD/VATEX 采 12 帧，DiDeMo 采 64 帧；训练 5 epoch、batch 64、两张 A100，CLIP/新增模块学习率分别为 `1e-7/1e-4`（PDF 物理页 6）。

**本地代码事实**：MSR-VTT 脚本以 `MSRVTT_train.9k.csv` 训练，以 `MSRVTT_JSFUSION_test.csv` 作为 `val_csv`，设置 `--eval_in_train 1` 与 seed 0，见 `research_refs/EagleNet/run_train_msrvtt9k-vit_b_16.sh:8`、`research_refs/EagleNet/run_train_msrvtt9k-vit_b_16.sh:11`、`research_refs/EagleNet/run_train_msrvtt9k-vit_b_16.sh:12`。训练循环直接在该 loader 上算 R@1 并选择 checkpoint，见 `research_refs/EagleNet/main_my.py:287`、`research_refs/EagleNet/main_my.py:291`。所以 MSR-VTT headline 是 test-aware 选模口径，对本项目属于协议不兼容。

### 2.6 主结果、消融、效率与鲁棒性

**论文事实**：

- Table 1（PDF 物理页 6）中 ViT-B/16 的 MSR-VTT T2V R@1 为 51.0，高于表内 Video-ColBERT 的 50.0；但 T2V R@5 76.2 低于 GLSCL/Video-ColBERT 的 76.3，V2T R@1 49.2 与 UCoFiA 的 49.0 接近，不能概括为所有指标领先。
- Table 2（物理页 7）中 ViT-B/16 的 DiDeMo/MSVD/VATEX T2V R@1 分别为 51.5/50.9/63.6；VATEX 的 63.6 低于表内 TV-ProxyNet 64.0，进一步否定“所有数据集所有指标最好”的强读法。
- Table 3（物理页 8）完整模型在 MSR-VTT/DiDeMo 的 T2V R@1 为 51.0/51.5。仅 FRL 相对无组件基线为 48.5→48.8 和 42.1→47.9，说明 FRL 的边际作用高度依赖数据集；仅 EAM 为 49.0/43.4，也不是跨数据集等幅收益。FRL+EAM+sigmoid 才形成 headline。
- Tables 4–5（物理页 8）显示能量函数的最优选择随数据集变化：MSR-VTT 是 bilinear 51.0，DiDeMo 是 MLP 51.5；平均逐帧能量在两者上最好，而先融合视频再算能量降为 48.8/47.8。

论文用 Figure 1 报告性能—测试时间关系，但没有在正文表格给出候选规模、分块策略和完整数值。代码会分别计时特征缓存与相似度计算（`research_refs/EagleNet/main_my.py:299`），但这不足以证明大库部署效率。多 seed、均值方差和显著性均未报告。

### 2.7 未获支持的强主张、复现缺口与理论风险

- **作者主张**：“捕获真实文本—视频对分布”超出了当前证据。EAM 使用可学习能量和 Langevin 负样本，但没有密度拟合检验、校准或分布外验证。
- **作者主张**：sigmoid 天然适合多匹配并不等于公开训练构造了精确多正例标签；若同批多个 caption 指向同一视频却仍按对角标正例，false negative 问题依然存在。
- **本项目推论**：所有 `bst × bsv` 候选对都运行文本采样、RGAT 和文本条件池化，复杂度随候选库线性乘上 pair scorer 成本；小型 1K test 的时间不能外推到大库。
- **复现缺口**：公开脚本用 test 选模，且只显示 seed 0；论文重实现多个 baseline，虽然有助于同代码公平比较，但也使 headline 同原论文数字的可比性更复杂。

### 2.8 对 UATVR 可迁移与不可照搬部分

可迁移的是“错误候选需要候选条件证据”“帧—帧上下文可辅助区分语义近邻”和“最终机制必须进入真实排序”的问题意识。不可照搬的是在全库上运行随机候选 + RGAT，也不能把 EAM 的训练正则称为已校准的不确定性。

若未来 P1 证明固定 WTI 在 top-k 中存在系统性的上下文错配，可测试一个有候选上限、可关闭、报告 latency/throughput/memory 的轻量 reranker；若诊断没有这种错误模式，EagleNet 路线停止。

## 3. GARE

### 3.1 研究矛盾

**论文事实**：GARE 从 InfoNCE 梯度出发，认为强模态间隙下正负梯度近共线且相反，使 anchor 更新相互抵消；hard/fuzzy negatives 又会把噪声直接施加到固定 anchor（PDF 物理页 2–4）。它提出把部分优化压力转移给每个文本—视频对自己的残差 `Δ_{ij}`。

**本项目推论**：这是“全局固定表示是否足以处理局部候选差异”的可证伪问题，而不是只要存在 modality gap 就必然需要残差。首先应定位错误对和梯度/排序症状，再决定是否增加 pair-conditioned correction。

### 3.2 核心创新与数据流

**论文事实**：对每个 `(t_i,v_j)`，以语义差 `v_j-t_i` 为 query，以候选视频帧序列（或文本 token）为 key/value，经单层 cross-attention `ψ` 输出 `Δ_{ij}`，再形成 `t_i+Δ_{ij}`（公式 7，PDF 物理页 5）。不同数据集可选择修正文本或视频侧。

**本地代码事实**：实现对所有 batch 文本和视频构造 `video_mean - text.detach()`，经 `psi` 后做 `cls = _cls + delta`，见 `research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:199`、`research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:203`、`research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:207`。`ψ` 是一层 cross-attention 加均值/对数标准差投影，见 `research_refs/GARE-text-video-retrieval/tvr/models/psi_module.py:111`、`research_refs/GARE-text-video-retrieval/tvr/models/psi_module.py:128`；调用方只使用第一个输出，公开实现中的残差是确定性的。

### 3.3 与常规检索表示的实质差异

`t_i+Δ_{ij}` 随候选 `v_j` 改变，所以同一 query 没有唯一的最终检索向量。基础文本、视频和帧特征可离线缓存，但完整 GARE score 仍须对候选对执行 `ψ`。论文附录也明确承认条件距离破坏共享度量/三角不等式，不能直接使用固定向量 ANN；建议先用基础双塔取 top-k，再用 GARE 重排（PDF 物理页 26–27）。

因此，称其“兼容双塔”只对特征缓存成立，不对完整索引几何成立。它更准确的定位是轻量 pair-conditioned scorer。

### 3.4 训练目标与最终检索分数

**论文事实**：最终 score 是归一化后的 `t_i+Δ_{ij}` 与文本条件视频池化表示的余弦。训练目标为 InfoNCE、relaxed IB、残差范数下界和方向多样性之和（公式 28，PDF 物理页 21）。

为规避 deterministic Dirac posterior 对连续先验的无限 KL，论文在一个 batch 内按视频 anchor 聚合 `{Δ_{ij}}_i`，拟合高斯后与 `N(0,I)` 做 KL（公式 15，PDF 物理页 6）。**本地代码事实**：`mu=delta.mean(dim=0)`、`sigma=delta.std(dim=0)` 和 KL 位于 `research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:220`–`230`，方向/范数项位于 `research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:234`–`239`。它是 batch 统计正则，不是对单个 pair 输出预测分布。

### 3.5 实验设计与协议

**论文事实**：覆盖 MSR-VTT、DiDeMo、ActivityNet Captions、MSVD；CLIP ViT-B/32 后接 4 层 temporal Transformer；短视频 12 帧/32 words，长视频 64 帧/64 words；batch 128，5 或 10 epoch，4–8 张 4090/A100/V100；没有报告 seed 数（PDF 物理页 7）。

**本地代码事实**：MSR-VTT loader 将 `val` 与 `test` 都映射到 `MSRVTT_JSFUSION_test.csv`，见 `research_refs/GARE-text-video-retrieval/tvr/dataloaders/dataloader_msrvtt_retrieval.py:28`–`31`。训练每 epoch 在该 `val_dataloader` 上算 R@1 并保存 best，见 `research_refs/GARE-text-video-retrieval/main_retrieval.py:583`、`research_refs/GARE-text-video-retrieval/main_retrieval.py:590`。所以其 “1K-A validation” 实际承担 test-aware 选模，对本项目属于协议不兼容。

### 3.6 主结果、消融、效率与鲁棒性

**论文事实**：

- Tables 1–2（PDF 物理页 7）显示相对作者自有 baseline，MSR-VTT 46.6→49.1、DiDeMo 45.4→47.6、ActivityNet 40.2→42.6、MSVD 45.0→46.4；四项绝对增益为 2.5/2.2/2.4/1.4 个百分点。
- Table 3（物理页 8）中只加 `Δ` 为 47.4；`Δ+IB` 为 48.3；完整正则为 49.1。范数或方向项在没有 IB 时为 47.2/47.0，不能单独支持这些结构正则有效。
- Table 4 显示修正哪一侧依数据集而变；Table 5 中显式 gap query 为 49.1，而无 gap variant 为 46.1；Table 6 中未归一化残差配标准高斯最好，但先验尺度敏感。
- Tables 7–9（物理页 15–16）进一步比较 context/gap 方向、方向温度和 IB anchor；最优参数均来自同一个 1K-A 集合，不能当作独立验证。
- Tables 10–11（物理页 25–26）在 4×RTX 4090、batch 128 上报告训练 1h34m vs baseline 1h30m，推理 6.9s vs 7.6s，训练/推理 FLOPs 均约 +0.3%，`ψ` 为 36.35 GFLOPs、约占 forward 0.31%。

效率表证明 1K 候选上的 batch-parallel 实现开销小，却没有证明百万级全库 pair scoring 可扩展。论文给出的 top-256 覆盖 GARE top-10 是单数据集观察，尚未报告不同 k 的 recall—cost 曲线。

### 3.7 未获支持的强主张、复现缺口与理论风险

- 一阶 Taylor/trust-region 推导给出局部更新直觉，但神经网络 `ψ` 是数据驱动近似；“复现解析下降方向”没有逐样本误差、边界或泛化保证。
- relaxed IB 使用当前 batch 中多个文本对同一视频的残差拟合高斯，估计质量依赖 batch 大小与样本组成；batch 128 下的结论不能直接迁移到本项目其他 forward batch。
- 残差的 `log_sigma` 输出被调用方丢弃，所谓 VIB 不包含单 pair uncertainty sampling；把其当作概率置信度是错误解释。
- 代码选择 checkpoint 的集合与最终 test 相同，没有 multi-seed、方差或显著性。其“抵抗 hard negatives”主要来自结果与梯度可视化，没有专门的受控 false-negative benchmark。

### 3.8 对 UATVR 可迁移与不可照搬部分

可迁移的是：以 pair residual 表达“这个候选相对这个 query 缺什么”、对残差做有界/结构约束、并把候选规模视为方法定义的一部分。不可照搬的是全库两两 cross-attention、batch 高斯即 pair uncertainty、以及 test-aware 参数搜索。

P3 只应在 P1 证明系统性固定表示错配后解锁；首个版本应限定 top-k、保留原始 WTI score、报告不同 k 的质量与开销，并设置残差幅值和退化保护。若增益只来自增加 pair scorer 容量，或在至少三 seed 下不稳定，则停止。

## 4. TempMe

### 4.1 研究矛盾

**论文事实**：基于 CLIP 图像编码器的视频检索通常把每帧独立展开为 patch tokens。参数高效微调虽然减少了可训练参数，却没有减少推理 token；图像压缩方法只消除单帧空间冗余，又忽略相邻帧之间的时间冗余（PDF 物理页 1–4）。

TempMe 因而把问题拆成两个维度：用 LoRA 控制可训练参数，用跨帧渐进 token merging 同时做时序建模和计算压缩。这个问题定义本身比“换一个更强 backbone”更可检验，因为精度、FLOPs、吞吐和显存都能独立测量。

### 4.2 核心创新与数据流

**论文事实**：Progressive Multi-Granularity 框架前期用 ImgMe 在每帧内部合并相似 patch，后期用 ClipMe 把相邻 clips 逐级合并。短视频采用 `12→6→3→1`，长视频采用 `64→16→4→1`；跨 clip 合并前加入可训练 clip positional embeddings，随后让原 CLIP self-attention 在跨帧 token 上工作（PDF 物理页 4–7）。

ClipMe 不是简单均值池化：它先把多个 clip 的 token 展开，以 bipartite soft matching 找相似 token 对并按 token size 加权合并，再执行 attention 和 intra-clip merging。**本地代码事实**：层号、合并帧数和保留比例在 `research_refs/TempMe/tvr/models/modeling.py:72`–`99` 解析；跨帧 reshape、位置编码、合并与 attention 在 `research_refs/TempMe/tvr/models/module_tome_patch.py:25`–`84`。

### 4.3 与常规检索表示的实质差异

TempMe 仍是独立编码：视频 token 的合并不看当前文本候选，最终每个视频形成一个固定向量，能与固定文本向量做矩阵乘法和 ANN。它改变的是视频编码器内部 token 路径，而不是跨候选 score 函数。

与普通逐帧平均相比，后几层 self-attention 能在多个帧的 token 之间传播信息；与 ToMe 相比，它显式合并跨帧 token。代码最终把合并后的多个 CLS token reshape 回原帧数并平均，见 `research_refs/TempMe/tvr/models/modeling.py:270`–`280`。因此“时序能力”来自跨帧 attention 的可见域，“效率”来自 token 数减少，两者必须分开评价。

### 4.4 训练目标与最终检索分数

**论文事实**：默认冻结 CLIP 基础权重，只训练 rank-8 LoRA 与约 0.01M clip positional embeddings；LoRA 权重可在推理前并入基础矩阵。最终 score 仍是文本向量与视频向量的余弦，训练使用双向对比损失（附录 A，PDF 物理页 16）。

**本地代码事实**：模型构造 LoRA 与合并层后使用 `CrossEn`，见 `research_refs/TempMe/tvr/models/modeling.py:44`、`research_refs/TempMe/tvr/models/modeling.py:72`、`research_refs/TempMe/tvr/models/modeling.py:85`。合并路径在训练和推理都执行，不存在只训练不用的额外 branch，因而训练—推理机制一致。

### 4.5 实验设计与协议

**论文事实**：覆盖 MSR-VTT、ActivityNet、DiDeMo、LSMDC；以 CLIP ViT-B/32 为默认并补充 ViT-B/16、LaCLIP、MetaCLIP 和 UMT；batch 128、AdamW、5 epoch、学习率 `6e-4`。短视频 12 帧/32 words，长视频 64 帧/64 words；吞吐在 A100 上测量（PDF 物理页 7）。

MSR-VTT 使用 9000 train+val 视频训练并在 1K-A test 报告（物理页 16）。**本地代码事实**：`val` 与 `test` 均映射到 `MSRVTT_JSFUSION_test.csv`（`research_refs/TempMe/tvr/dataloaders/dataloader_msrvtt_retrieval.py:28`–`31`），每 epoch 用该 loader 的 R@1 保存 best（`research_refs/TempMe/main.py:487`–`500`）。因此 headline 的 checkpoint selection 与本项目盲测边界协议不兼容。

### 4.6 主结果、消融、效率与鲁棒性

**论文事实**，按 Tables 1–17 分组核验：

- Table 1（PDF 物理页 6）给出最清晰的精度—复杂度对照。ViT-B/16、12 帧时，LoRA 为 211.3 GFLOPs、2364 tokens、R-Sum 201.4；TempMe 为 121.4 GFLOPs、127 tokens、R-Sum 206.7，即 token 减少 94.6%、GFLOPs 减少 42.5%，精度反而增加 5.3。
- Tables 2–3（物理页 7）覆盖四数据集 T2V。MSR-VTT ViT-B/32 为 46.1/71.8/80.7，ActivityNet/DiDeMo/LSMDC R@1 为 44.9/48.0/23.5；这些是同表强结果，但协议与 baseline 来源并不完全统一。
- Table 4（物理页 8）给出 ViT-B/16 真实吞吐：TempMe 45.1 videos/s、121.4 GFLOPs、127 tokens；相对 VoP 的 25.0 videos/s、246.2 GFLOPs、2368 tokens，为 1.804× 吞吐、50.7% GFLOPs 降幅和 +4.4 R-Sum，摘要 headline 复算成立。
- Table 5（物理页 8）说明 full fine-tuning 下 TempMe 相对 CLIP4Clip 的 R-Sum 202.3→210.2（+7.9），训练速度 1.57×，训练显存 70.1→52.7GB；表注称五个随机 seed，但只给单个汇总数，没有均值±方差。
- Tables 6–7（物理页 8–9）将机制移植到 UMT4Clip 的检索/问答：检索可同时降计算并恢复性能，问答则 44.4→44.6，仍低于原 UMT 44.9，说明迁移收益不是所有任务都领先。
- Table 8（物理页 9）从 LoRA 43.7/193.0 出发，ImgMe 单独为 42.9/191.4，ClipMe 完整为 46.1/198.6；空间压缩本身没有带来精度。
- Table 9（物理页 9）是核心因果证据：只做 Temporal Modeling 为 54.3 GFLOPs、R-Sum 199.7；只做 Token Reduction 为 34.7 GFLOPs、R-Sum 188.8；完整 TempMe 为 34.8/198.6。精度主要来自跨帧建模，压缩主要负责效率且可损伤精度。
- Table 10（物理页 10）表明渐进、最终合为一个 clip 的 `12→6→3→1` 最好；过早合并虽降至 18.1 GFLOPs，却把 R@1 降到 40.8，存在明确精度—压缩边界。
- Tables 11–12（物理页 17）补充 V2T；Tables 13–14（物理页 18）显示 Prompt/Adapter/LoRA 和 LaCLIP/MetaCLIP 的兼容性；Tables 15–16（物理页 18）分别验证位置编码 +0.4 R@1、8 帧相对 12 帧少 1.4 R-Sum；Table 17（物理页 19）给出 4×A100、每卡 batch 32 下显存，TempMe 为训练/推理 12381/4735MB，18 帧变体则为 16410/5879MB。

这些结果支持 TempMe 是四篇中效率证据最完整的一篇，也同时提供了“压得更多可能更差”的失败边界。

### 4.7 未获支持的强主张、复现缺口与理论风险

- “精度提升来自 token merging”过于粗糙；Table 9 已证明真正正向变量是跨帧 attention，纯 reduction 明显降精度。
- headline 的 GFLOPs 只计算视频 backbone，吞吐比较还省略 VoP/DGL 的 prompt generation；这些限定必须随数字一起报告。
- Table 5 虽称五 seed，却无方差、置信区间或每 seed 值；其余 headline 没有多 seed 证据。
- 公开 MSR-VTT 实现用 1K test 选模；效率结论可复算，不代表 Recall 能直接迁移到本项目。
- 合并是离散的相似 token 贪心结构，早层细节丢失不可逆；Figure 5/Table 10 的敏感性说明 start layer 和保留率不是无代价默认值。

### 4.8 对 UATVR 可迁移与不可照搬部分

可迁移的是将“时序建模”和“压缩效率”拆成两个独立因素，报告准确率、GFLOPs、吞吐、训练/推理显存，并把 start layer/候选 token 数写入方法定义。不可照搬的是在不诊断当前 WTI 错误模式前直接替换视频路径，也不能把 token 数减少解释成 uncertainty 增益。

TempMe 适合 P4 独立支线：先固定 backbone、loss、split 与 batch，仅加入无压缩的 temporal modeling；确认跨 seed 的语义收益后，再加入 reduction 并验证 Pareto 前沿。首轮不能与 pair uncertainty 或 candidate-conditioned alignment 同时改变。

## 5. GraviAlign

### 5.1 研究矛盾

**论文事实**：GraviAlign 认为确定性点向量不能表达一对多/多对多语义，已有概率嵌入又常把均值距离与方差惩罚简单相加，使距离梯度与 uncertainty 脱钩。它希望用一个同时依赖中心距离和联合方差的 closed-form score 替代采样估计（PDF 物理页 1–3）。

这个研究矛盾与本项目未来方向高度相关：真正需要解释的不是某个样本“总体有多不确定”，而是 query 与某个候选之间为何可能错配。不过 GraviAlign 的高斯参数仍由每个模态独立生成；pair-level 只体现在两者组合后的 score，不等于直接监督 `u(t_i,v_j)`。

### 5.2 核心创新与数据流

**论文事实**：CLIP-ViP 产生序列/池化特征，轻量 uncertainty heads 分别输出文本和视频的均值 `μ` 与对角协方差 `Σ`。作者从 Semantic Gravitational Interaction 积分出发，把不可积的交互拆成：

1. Term A：`(G/T) m_v m_t / d_M(μ_v,μ_t)`，以熵函数定义“semantic mass”；
2. Term B：`-1/2 (μ_v-μ_t)^T(Σ_v+Σ_t)^{-1}(μ_v-μ_t)`；
3. Term C：`-1/2 log|Σ_v+Σ_t|`。

最终定义 `S_align=A+B+C`（公式 4–8，PDF 物理页 4–5）。Term B 与 C 共同衡量高斯重叠；Term A 是以 Mahalanobis 距离代入重力类比的额外启发式项。

### 5.3 与常规检索表示的实质差异

每个 item 的 `(μ,Σ)` 可独立缓存，但 `Σ_v+Σ_t` 让 score 依赖具体 pair，不能直接化为标准余弦 ANN。理论上每对是 `O(D)`，全库仍需 `O(N_qN_vD)` 的精确评分，除非另行设计两阶段索引或近似。

与 EagleNet/GARE 不同，GraviAlign 不用候选内容重新生成 query 表示；它是“独立概率表示 + pair-wise closed-form scorer”。与普通概率模型不同，它把联合方差放进 Mahalanobis 距离，确实让均值距离的影响随 pair uncertainty 改变。

### 5.4 训练目标与最终检索分数

**论文事实**：训练为确定性 CLIP-ViP 的双向 InfoNCE 加正配对上的 `L_SGI=-(A+B+C)`，总损失 `L=L_InfoNCE+αL_SGI`（公式 9–10，PDF 物理页 5）。论文把 `A+B+C` 称为 alignment/ranking score，并在 Figure 3 展示三个分量。

**证据边界**：主文没有给出足以复现的推理伪代码，也没有说明 headline 检索矩阵是否只用 `S_align`、确定性 score，或二者组合；没有实现就无法核对最终 logits、数值稳定项、variance parameterization 和分块策略。因此“进入最终排序”的具体方式标记为不可复算。

### 5.5 实验设计与协议

**论文事实**：覆盖 MSR-VTT、DiDeMo、ActivityNet；baseline 为 CLIP-ViP ViT-B/32/B/16；MSR-VTT/DiDeMo 12 帧，ActivityNet 32 帧，224×224，最大文本 50；batch 128；MSR-VTT 5 epoch，另外两个 60 epoch；AdamW、weight decay 0.2、H800。`α` 按数据集分别设 0.1/0.1/0.05（PDF 物理页 5–6）。

MSR-VTT 用 9k train 和 official 1k test；论文没有说明独立 validation、checkpoint selection、seed 或每数据集如何调出 `α`。Table 4–5 的组件和超参数分析也在 MSR-VTT 上报告，存在 test 被用于机制选择的风险，但在缺少代码时只能标为“未说明”，不能断言其实际流程。

### 5.6 主结果、消融、效率与鲁棒性

**论文事实**：

- Table 1（PDF 物理页 7）中 ViT-B/32 的 MSR-VTT T2V 为 52.4/75.7/85.4，相对 CLIP-ViP 50.1/74.8/84.6 提升 2.3/0.9/0.8；ViT-B/16 为 55.8/77.8/85.8，相对 54.2/77.2/84.8 提升 1.6/0.6/1.0。它在若干 R@1 最好，但 V2T ViT-B/16 R@5=76.2 低于表内 ProTA 77.0，不能表述为所有指标领先。
- Table 2（物理页 8）中 DiDeMo ViT-B/32 R@5=76.9 低于 baseline 77.1，尽管 R@1/R@10 提升；ViT-B/16 52.3/78.6/87.3 相对 baseline 50.5/78.4/87.1 增幅集中在 R@1。
- Table 3（物理页 8）中 ActivityNet R@1 提升 B/32 +1.3、B/16 +2.6；这支持相对 baseline 的增益，但未提供多 seed 稳定性。
- Table 4（物理页 8）完整模型 R@1 52.4；去 A/B/C 分别为 50.5/51.9/52.1。A 的消融差最大，但单表不能证明三个分量“正交”或“不可缺少”，也没有相同参数量的替代函数对照。
- Table 5（物理页 8）只在少量 `T/λ/G` 点上单变量搜索；其结果显示局部不敏感，不等于全范围鲁棒。

论文只给理论 `O(D)`，未报告参数量、FLOPs、吞吐、延迟、训练/推理显存或候选规模。故“efficient”只有复杂度阶数证据，没有实测闭环。

### 5.7 论文陈述、公式事实、未获支持主张与复现风险

这里必须把证明强度分开：

- **公式事实（已核验）**：单页 supplemental 从 `Z_diff=Z_v-Z_t` 的零点 log-likelihood 和高斯乘积 overlap integral 两条路径，都推出 `Term B+Term C+常数`。所以 B+C 是对角高斯 log-overlap 的严格形式。
- **作者主张（弱证据）**：Term A 不是从 SGI 积分推导出的 closed-form 等价项，而是“在分布中心代入重力势”的近似/类比；supplemental 没有给出误差界或与原积分的关系。
- **作者主张（未验证）**：semantic mass `exp[-λH_norm/(1+H_norm)]` 被称为位于 `(0,1]` 且随 uncertainty 单调下降。但预测 `log σ²` 未见保证 `H_norm≥0`；当 `H_norm` 为负或接近 `-1` 时，该有界性与稳定性并不由公式自动成立。
- **作者主张（未验证）**：加法 `A+B+C` 本身不赋予每项“独立 veto power”；若没有每项范围、权重和阈值约束，一个高分量可以补偿另一个低分量。
- **作者主张（未验证）**：Term C 惩罚大方差，不等于整体目标自然防止方差向零塌缩；B、C、A 在小方差极限的竞争需给出完整梯度、数值下界和稳定性实验。
- **复现风险（不可复算）**：未找到公开代码；无多 seed、方差、显著性、校准、risk-coverage、分布外或实测效率。只用正配对训练 `L_SGI` 也不足以证明该 score 对困难负配对经过校准。

### 5.8 对 UATVR 可迁移与不可照搬部分

可迁移的是 B+C 的高斯重叠视角、让距离与联合 uncertainty 相互作用、以及把 ambiguity 纳入最终 pair score。不可照搬的是无界/奇异 Term A、未经约束的 semantic mass、“加法即 veto”解释和没有负候选监督的正配对结构损失。

P2 可从稳定的 log-overlap baseline 出发，但必须重新定义有界 log-variance、`ε` 稳定项和正负 pair 监督，并明确它如何改变匹配 score、证据聚合或训练梯度。主检验是能否稳定改善 Recall、GT rank 与目标错配切片；校准、错误检测及与 WTI score/margin 的去冗余只用于验证 uncertainty 语义。若只改善风险预测而不改善检索，不能作为本项目核心路线继续堆叠解释性 loss。

## 6. 横向比较

### 6.1 创新发生层级

四篇论文的价值不在于换了哪个 backbone，而在于修改了检索系统的不同层级：EagleNet 改候选条件表示，GARE 改 pair 优化与 scorer，TempMe 改视频编码器的时序可见域/计算图，GraviAlign 改概率几何 score。它们都说明 backbone 应是匹配控制变量，而不是方法贡献的替代物。

| 方法机制矩阵 | 表示层 | 交互层 | 时序层 | 概率层 | 最终 score | 训练 loss | candidate-conditioned | 可固定索引 | 训练—推理一致 |
|---|---|---|---|---|---|---|---|---|---|
| EagleNet | 候选视频条件的富化文本与文本条件视频 | 文本候选/帧三关系 RGAT | frame-frame 边与 frame PE | 随机文本候选；未做校准 | pair 富化文本—池化视频余弦 | sigmoid matching + support + EAM | 是，表示和 score 均依赖候选 | 只能缓存基础帧特征，完整 score 不可固定 ANN | 部分一致：FRL 用于推理，EAM 仅训练 |
| GARE | `t_i+Δ_ij` pair residual | gap-query 单层 cross-attention | 4 层 temporal Transformer | batch Gaussian relaxed IB；pair 残差本身确定 | 修正文本—条件池化视频余弦 | InfoNCE + relaxed IB + norm + direction | 是，修正表示和 score 均依赖候选 | 可缓存基础特征，完整 score 不可固定 ANN | 已核验：`Δ` 训练和推理均保留 |
| TempMe | 独立固定文本/视频向量 | 无跨模态 candidate 交互 | 后层跨帧 attention + 渐进合并 | 论文未使用概率表示 | 固定向量余弦 | 双向对比损失 | 否 | 是，可标准矩阵检索/ANN | 已核验：合并路径训练推理一致 |
| GraviAlign | 独立高斯 `(μ,Σ)` | 联合方差 Mahalanobis/overlap pair scorer | 继承 CLIP-ViP | 对角高斯；B+C 为 log-overlap | 论文定义 `A+B+C`，真实 logits 不可复算 | deterministic InfoNCE + 正配对 `L_SGI` | score 是，单 item 表示否 | 可缓存 `μ,Σ`，score 非标准内积 | 不可复算：缺少推理实现 |

### 6.2 Candidate-conditioned 与独立编码边界

要区分三件事：

1. **独立表示与独立 score**：TempMe 可为每个 item 预先生成固定向量，适合全库检索。
2. **独立表示与 pair score**：GraviAlign 可缓存高斯参数，但联合方差 score 仍需逐 pair 计算。
3. **pair-conditioned 表示与 pair score**：EagleNet/GARE 会因候选改变 query/video 表示，完整方法天然是 reranker。

因此 EagleNet/GARE 的实验报告若没有候选数 `k`，方法定义就是不完整的。对 UATVR 的第一版迁移上限应是固定 top-k，而非把 1K 全对全测试的速度当成大库可部署性。GraviAlign 即便每对只有 `O(D)`，也同样需要说明全库分块或近似索引。

### 6.3 训练—推理一致性

EagleNet 的 FRL 直接产生推理 logits，但 EAM 只通过训练改变参数；若要声称能量训练改善匹配，需要说明其梯度路径，并用移除 EAM 的直接消融验证最终排序变化，不要求训练机制必须在推理时继续运行。GARE 的 `Δ` 同时进入训练和推理，机制闭环最清楚，但其 relaxed IB 只在训练使用且不是预测 uncertainty。TempMe 的 token 路径两阶段一致，因果解释也最干净。GraviAlign 把 `A+B+C` 称为 ranking score，却未给可复现推理数据流，是当前最大的闭环缺口。

对本项目，任何新增 uncertainty/alignment 分支都必须回答三个问题：训练时监督什么、它如何改变梯度/表示/匹配 score、关闭后是否严格回到 P0。纯训练机制可以在推理时回到 WTI，但必须用梯度分析、直接消融和最终检索收益证明因果作用；只增加 confidence/risk head、却不改善学习或排序，不能作为核心创新。

### 6.4 协议、额外数据与后处理

| 实验协议对照 | 数据集/任务 | split 与选模 | backbone/预训练 | 输入帧与尺寸 | batch/seed | test 使用 | 后处理/额外数据 | 证据定位 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| EagleNet | MSR-VTT、DiDeMo、MSVD、VATEX；T2V/V2T | MSR-VTT 9k train；公开脚本用 JSFusion 1K R@1 选模 | CLIP ViT-B/32、B/16 | 12 帧；DiDeMo 64；CLIP 224 输入 | 64；脚本 seed 0 | val_csv 即 JSFusion test，逐 epoch 使用 | 无检索后处理；重实现多项 baseline | PDF 物理页 6 Tables 1；`research_refs/EagleNet/run_train_msrvtt9k-vit_b_16.sh:8`–`12` | 协议不兼容 |
| GARE | MSR-VTT、DiDeMo、ActivityNet、MSVD；T2V/V2T | 1K-A 称 validation；代码 val/test 同 CSV 并以 R@1 选模 | CLIP ViT-B/32 + 4 层 temporal Transformer | 短视频 12 帧，长视频 64；CLIP 224 输入 | 128；seed 论文未报告 | JSFusion 1K 同时承担 val/test | 无 QB-Norm；无额外数据报告 | PDF 物理页 7 Tables 1–2；`research_refs/GARE-text-video-retrieval/tvr/dataloaders/dataloader_msrvtt_retrieval.py:28`–`31` | 协议不兼容 |
| TempMe | MSR-VTT、ActivityNet、DiDeMo、LSMDC；T2V/V2T | MSR-VTT 9k train/1K test；代码每 epoch 在该 1K 选模 | CLIP ViT-B/32、B/16；另测 LaCLIP/MetaCLIP/UMT | 短视频 12 帧，长视频 64；CLIP 224 输入 | 128；Table 5 称五 seed，主结果未报告 | val/test 映射同 JSFusion CSV | 无检索后处理；吞吐省略部分 prompt 生成 | PDF 物理页 7–8 Tables 2–5；`research_refs/TempMe/main.py:487`–`500` | 协议不兼容 |
| GraviAlign | MSR-VTT、DiDeMo、ActivityNet；T2V/V2T | 9k/official 1k；独立 val 与 checkpoint selection 未说明 | CLIP-ViP ViT-B/32、B/16 | 12 帧；ActivityNet 32；224×224 | 128；seed 未报告 | 在 MSR-VTT 报组件/超参分析，是否 test 调参未验证 | 明示不用 QB-Norm/DSL；无额外数据报告 | PDF 物理页 5–8，Tables 1–5 | 未验证、不可复算 |

四篇的 headline 都不能替代 `trusted-v1` 实验。尤其 MSR-VTT 公开传统 9k/1K 流程常把 test 当作 validation 使用，而本项目要求 internal val 选模、JSFusion 1K 只做冻结后的显式盲测。

### 6.5 效率、候选规模与扩展性

| 效率与候选规模 | 论文报告指标 | 报告值 | 是否计入候选对数 | 全库扩展成本 | 延迟/吞吐 | 显存 | 参数/FLOPs | 证据强度 |
|---|---|---|---|---|---|---|---|---|
| EagleNet | MSR-VTT 性能—test time 图；代码分 cache/sim time | 正文图示但无可抄录完整数值表 | 1K 全矩阵隐含计入，未按候选数归一 | 每个候选对运行采样、RGAT、条件池化，约随 `N_qN_v` 增长 | 论文未报告可审计吞吐 | 论文未报告 | 论文未报告参数/FLOPs | 弱证据 |
| GARE | 训练/推理时间、显存、per-batch FLOPs、模块占比 | 1h34m、6.9s；4×12561MB/4216MB；训练/推理约 +0.3%；`ψ` 36.35 GFLOPs | per-batch 报告，没有给不同候选库大小曲线 | 完整 score 两两计算；论文建议 base ANN + top-k GARE | 1K 推理 6.9s；未报告吞吐随 k 曲线 | 已报告 4090 reserved memory | `ψ` 1.58M；Table 10–11 给 FLOPs | 已核验小库、全库弱证据 |
| TempMe | video-backbone GFLOPs、tokens、videos/s、训练/推理显存 | B/16 121.4 GFLOPs、127 tokens、45.1 videos/s；B/32 12381/4735MB | 独立视频编码，不产生候选对模块 | 视频一次编码后可固定索引；扩展成本接近标准双塔 | A100 实测 45.1 videos/s | Table 17 完整报告 | 0.50M trainable；多表报告 GFLOPs | 已核验、四篇最强 |
| GraviAlign | 理论 per-pair 复杂度 | `O(D)`，无实测值 | 没有报告候选数量或全矩阵成本 | 缓存高斯后仍需逐 pair 联合方差 score | 论文未报告 | 论文未报告 | 参数/FLOPs 论文未报告 | 不可复算 |

方法学结论不是“pair-conditioned 一定不可用”，而是候选规模属于方法定义：双塔负责召回，pair scorer 负责有限候选重排；同时报告 k、候选 recall 上限、额外延迟、吞吐、显存和检索收益。这里的资源指标用于公平比较和界定适用范围，不把产品部署作为 P2/P3 的研究目标。TempMe 可独立作用于第一阶段，但精度变量与压缩变量不能混在一个消融里。

### 6.6 证据强度与不可支持的强主张

| Headline 复算 | 作者主张 | 实际比较基准 | 基准值 | 方法值 | 绝对差 | 相对差 | 是否成立 | 证据定位 |
|---|---|---|---:|---:|---:|---:|---|---|
| EagleNet | “各数据集 T2V R@1 最好”的强读法 | VATEX ViT-B/16 同表 TV-ProxyNet | 64.0 | 63.6 | -0.4 | -0.63% | 不成立；MSR-VTT/DiDeMo/MSVD 的局部 headline 可成立 | PDF 物理页 7 Table 2 |
| GARE | 相对自有 baseline 在四数据集一致提升 | 同表自有 baseline，按 MSR-VTT/DiDeMo/ActivityNet/MSVD | 46.6/45.4/40.2/45.0 | 49.1/47.6/42.6/46.4 | +2.5/+2.2/+2.4/+1.4 | +5.36%/+4.85%/+5.97%/+3.11% | 对自有 baseline 成立；不证明跨协议普适 | PDF 物理页 7 Tables 1–2 |
| TempMe | 相对 VoP，+4.4 R-Sum、约 1.8× 吞吐、约 51% GFLOPs 降幅 | ViT-B/16 同表 VoP | R-Sum 202.3；25.0 videos/s；246.2 GFLOPs | 206.7；45.1；121.4 | +4.4；+20.1；-124.8 | +2.18%；+80.4%；-50.7% | 数值成立；4.4 是绝对点而非相对 4.4% | PDF 物理页 8 Table 4 |
| GraviAlign | ActivityNet ViT-B/16 R@1 相对 CLIP-ViP +2.6 | 同表 CLIP-ViP | 53.4 | 56.0 | +2.6 | +4.87% | 表内成立；无多 seed/显著性 | PDF 物理页 8 Table 3 |

| 证据质量 | 多 seed | 方差 | 显著性 | 独立数据集 | 校准 | 失败案例 | 实现可复算性 | 综合等级 |
|---|---|---|---|---|---|---|---|---|
| EagleNet | 未报告；公开脚本 seed 0 | 未报告 | 未报告 | 4 个数据集，但机制最优配置随数据集变化 | 未报告 | 有定性检索案例；有单模块退化 | 本地代码可读，test-aware 选模降低协议可信度 | 弱证据、协议不兼容 |
| GARE | 未报告 | 未报告 | 未报告 | 4 个数据集；修正侧依数据集变化 | 未报告 risk/calibration | 有几何/梯度图；报告无 IB 时正则无效 | 本地代码可读；数据与选模协议可核验 | 弱证据、协议不兼容 |
| TempMe | Table 5 称五 seed；其余未报告 | 未报告 | 未报告 | 4 数据集、多 PEFT/backbone/UMT | 不适用且未报告 | 明确报告纯压缩、过早合并和少帧退化 | 本地代码可读；效率设置较完整 | 已核验边界、Recall 协议不兼容 |
| GraviAlign | 未报告 | 未报告 | 未报告 | 3 数据集 | 未报告；没有 AURC/ECE/Brier | 4 个定性案例；无数值稳定失败分析 | 无代码，最终 logits 和效率不可复算 | 弱证据、不可复算 |

总体上，TempMe 对“效率与时序因素”的证据最完整；GARE 对 pair residual 的实现闭环清楚但协议与大库扩展证据不足；EagleNet 对候选上下文的消融有启发但系统较重；GraviAlign 的 B+C 理论最干净，而整体实现和校准证据最弱。任何“全数据集都领先”“理论保证泛化”“自然防塌缩”“小库低延迟即可扩展全库”的表述都不被现有证据支持。

## 7. 对本项目的综合判断

### 7.1 已采纳原则

本项目采纳的是可证伪研究原则，而不是四套模块的拼装：

- 固定 `trusted-v1`、OpenAI CLIP ViT-B/16、主损失、forward batch 和选模规则；backbone 只作为匹配控制变量。
- 先诊断“何时、为何错配”，再决定是 pair uncertainty、candidate alignment 还是 temporal modeling。
- 新 uncertainty 必须是 query-video pair-level，并进入匹配表示、token/frame 证据聚合、similarity/logits 或可验证的训练目标，形成新的文本—视频匹配/学习机制。
- 新机制一次只改变一个因果变量，至少三 seed，报告 T2V/V2T、均值方差和失败条件。
- 不要求所有数据集和指标都达到 SOTA，但核心方法仍须相对同协议 P0 稳定改善检索；新颖问题、因果闭环、机制消融、可复现性和复杂度边界用于解释该收益，而不能替代它。

| UATVR 采纳矩阵 | 目标问题 | 机制 | 证据等级 | 采用/拒绝/延后 | Roadmap 阶段 | 前置证据 | 停止条件 | 综合分析锚点 |
|---|---|---|---|---|---|---|---|---|
| EagleNet | 固定 query 对上下文近邻区分不足 | 候选文本采样 + frame/text RGAT | 协议不兼容、部分已核验 | 延后，不照搬全库图 | P3 | P1 证明 top-k 内存在帧上下文型系统错配 | 三 seed 无稳定 Recall/GT-rank 收益，或收益仅来自容量即停；复杂度单独报告 | `2.2`、`2.6`、`6.2` |
| GARE | modality gap 下 pair-specific 局部修正 | 有界 `Δ_ij` 候选残差/reranker | 协议不兼容、部分已核验 | 延后，仅考虑 top-k | P3 | P1 给出固定 WTI 错配及候选召回上限 | 三 seed 无稳定排序收益、目标切片未修复或收益仅来自容量即停；k-cost 界定适用范围 | `3.2`、`3.6`、`6.5` |
| TempMe | 视频时序理解与编码效率 | 先无压缩 temporal modeling，再 token reduction | 已核验边界 | 独立支线采用研究方法，不直接合并 | P4 | P0 稳定；P4 单独规格与效率基线 | 时序本身无收益，或压缩不形成 Pareto 改善即停 | `4.6`、`4.8`、`6.5` |
| GraviAlign | pair ambiguity 与距离—uncertainty 耦合 | GraviAlign 仅提供有界高斯 B+C log-overlap 启发；pair-level 双源分解与 uncertainty-aware matching/training 是 UATVR 新假设，不是论文事实 | 弱证据、实现不可复算 | 只采用 B+C 启发，拒绝原样照搬 | P2 | P1 给出可纠正错配结构；稳定公式、监督与容量控制 | 三 seed 无稳定检索收益即停；若分量复制 score/margin 或彼此不可区分，则不能声称双源 uncertainty 创新 | `5.7`、`5.8`、`6.6` |

### 7.2 延后机制

EagleNet/GARE 的共同洞见是“同一 query 对不同候选需要不同证据”，但二者都把完整库变成 pair computation。只有 P1 证据显示固定 WTI 的错误集中在 top-k 且可由候选上下文纠正，P3 才解锁。首版应在固定 top-k 上比较最简单的 bounded residual 或 pair interaction/reranker，直接检验候选排序修复，不加入 risk gate，也不能叠加随机采样、图网络、能量模型和多项正则。

P2 与 P3 首先作为两条独立检索机制验证：P2 检验双源 uncertainty 如何改善 matching/training，P3 检验 candidate-conditioned evidence 如何改善排序。只有两者分别相对严格基线成立后，才研究 uncertainty-conditioned alignment，让 uncertainty 调节证据或 residual，而不是触发拒答、人工审核或产品风险路由。

TempMe 的 temporal modeling 独立延后到 P4。它与 P2/P3 同时启用会让“是时序表示变好，还是 uncertainty/alignment 有效”无法归因。backbone 升级同理：可在核心方法稳定后作为外部有效性测试，但不应成为当前盲目 sweep 变量。

### 7.3 明确拒绝或不可直接迁移机制

当前明确拒绝：

- 用不同 backbone 的绝对 Recall 代替方法创新；这会同时改变预训练数据、容量和表征几何，无法回答当前科研问题。
- 在全库执行未报告候选上限的 pair-conditioned 图/残差 scorer。
- 把随机表示、batch KL 或单样本 variance 直接命名为 pair uncertainty。
- 只加辅助 loss、却不说明它如何改变训练梯度、学习表示或最终检索排序。
- 原样使用无界 `1/d` attraction、未经约束的 semantic mass 或“加法自动 veto”解释。
- 继续为已终止的旧 sample/video-level 辅助不确定性路线补消融；该路线已停止，不再构成未来方法基线。
- 把 abstention、selective retrieval、人工审核或产品可信路由作为 P2/P3 核心目标；这些最多是附加应用实验，不能替代匹配/训练创新与检索收益。
- 以“所有数据集 SOTA”为主要目标。跨协议排行榜不足以证明因果贡献，追逐它会鼓励同时换 backbone、数据处理和模块。

对“盲目换 backbone 没什么收益”的判断是：**作为当前研究动作，基本同意；作为最终外部验证，不能绝对排除。** 当前它不回答 pair uncertainty 或 multimodal alignment 的机制问题，且制造最大混杂；等方法在固定 backbone 上通过后，再用第二 backbone 检验可迁移性才有意义。

### 7.4 需要独立规格的问题

以下方向值得深入，但每一项都需要单独设计规格，不能在一次实验中混合：

1. **Pair-level aleatoric–epistemic uncertainty-aware matching/training**：分别定义并估计 `u^ale(t_i,v_j)` 与 `u^epi(t_i,v_j)`，选择 matching score、token/frame evidence aggregation 或 contrastive objective 中的一个首轮入口。主要评价 Recall、GT rank 和目标错配切片；AURC、Brier/NLL、分桶校准和去 score/margin 冗余只作语义验证。
2. **Candidate-conditioned multimodal alignment**：基础 WTI 保持可回退；首版固定 top-k，使用 bounded residual 或轻量 pair interaction 直接纠正候选排序。报告 candidate recall ceiling、Recall/GT-rank/rank-flip、同参数量控制，以及作为二级边界的延迟、吞吐和显存；不加入风险 gate。
3. **稳定概率几何**：以已证明的 Gaussian B+C log-overlap 为最小 baseline，使用有界 log-variance、`ε`、score clipping/normalization，并给出方差塌缩与爆炸测试。Term A 若保留，必须重新推导并与同参数量 MLP/有界核函数公平比较。
4. **错配来源分解**：分别量化 query ambiguity、video ambiguity、pair incompatibility；利用 token/frame alignment、top-1 margin、近邻密度与多正例结构建立诊断集，而不是先生成伪标签。
5. **时序作为证据而非混杂**：先评估无压缩跨帧 attention 是否修复特定时序错误；再研究帧间分歧能否成为 video ambiguity 的输入。该方向始终与 P2/P3 首轮实验隔离。
6. **机制优先的成功标准**：主张必须对应可观测量和失败条件。不要求 Recall 在所有数据集和指标上领先，但核心方法至少要在同协议主设置中相对 P0 跨 seed 稳定改善预注册检索指标，并由错误切片、消融和容量控制解释来源；仅风险校准、错误预测或效率改善不足以支撑 P2/P3 核心贡献。

## 附录 A：PDF 页码、表号与关键数字索引

- **EagleNet（PDF 物理页 6–8）**：Table 1 MSR-VTT；Table 2 DiDeMo/MSVD/VATEX；Table 3 组件 51.0/51.5；Tables 4–5 能量函数/池化。关键公式 4–19 在物理页 3–5。pair-conditioned 代码见 `research_refs/EagleNet/modules/modeling.py:370`–`431`；MSR-VTT test-aware 选模见 `research_refs/EagleNet/main_my.py:287`–`299`。
- **GARE（PDF 物理页 7–8、15–16、25–26）**：Tables 1–2 主结果；Tables 3–6 主消融；Tables 7–9 附加消融；Tables 10–11 效率。`t_i+Δ_ij` 见公式 7，relaxed IB 见公式 15，总目标见公式 28。实现见 `research_refs/GARE-text-video-retrieval/tvr/models/modeling.py:193`–`245`。
- **TempMe（PDF 物理页 6–10、17–19）**：Table 1 complexity；Tables 2–7 主结果/效率/迁移；Tables 8–10 组件、功能和合并策略；Tables 11–17 V2T、泛化、位置编码、帧数与显存。跨帧合并实现见 `research_refs/TempMe/tvr/models/module_tome_patch.py:25`–`100`。
- **GraviAlign（PDF 物理页 7–8；supplemental 物理页 1）**：Tables 1–3 主结果；Table 4 分量消融；Table 5 超参数。主文公式 4–10 定义 mass、A/B/C 与训练目标；supplemental 公式 12–17 只证明 B+C 的 Gaussian log-overlap。未找到实现，最终 logits 与实测效率不可复算。
