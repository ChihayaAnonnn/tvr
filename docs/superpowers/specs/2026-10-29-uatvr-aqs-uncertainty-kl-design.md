# UATVR + AQS：不确定性参与匹配的 KL 融合设计

日期：2026-10-29（北京时间）。代码基点：`5886cd1`，分支 `codex/uatvr-aqs-training-v1`。
状态：依据用户已讨论方案撰写的设计，尚未实现、验证或冻结科学超参数；本文不授权实验启动。

## 1. 目标与明确选择

在语义适配之后、AQS 选负之前，将 UATVR 已有文本/视频概率分布用于跨模态匹配。融合后的分数同时进入完整 batch 对比损失、AQS 困难度计算、AQS 辅助损失和推理排序，使均值与方差具有通往检索目标的直接梯度路径。

第一版复用现有分布头，不新增 CLIP 编码器、交叉注意力网络、假负例识别器或 Laya 模块。不把方差自动解释为检索失败概率，也不承诺 Recall 或校准收益。

需要检验的机制假设是：在相同训练条件下，分布匹配进入排序是否优于现有 AQS；若有收益，是否来自可学习方差，而不是均值匹配、额外分数尺度或候选偏置。

| 位置 | 取舍 | 本版决定 |
|---|---|---|
| 编码器后再增加均值/方差头 | 重复现有概率分支，仍可能与排序脱节 | 不采用 |
| 语义适配后，概率匹配与 WTI 融合 | 可复用现有头，训练和推理共享明确评分函数 | 采用 |
| AQS 选负后，仅增加局部概率损失 | 成本较小，但不能直接改善选样依据，且推理可能仍丢弃分布 | 不作为主设计 |

这是对原 AQS 设计“推理保持 WTI”的显式扩展：旧 AQS 行为保持为兼容模式；新模型使用版本化的融合评分，不能在旧身份下替换推理结果。

## 2. 当前代码事实与研究依据

| 事实 | 依据 |
|---|---|
| 时序/语义适配后计算 WTI，目前检索分数仍为 `exp(logit_scale) * WTI` | `src/semantic_retrieval/aqs/core.py`，`AqsCore._loose_similarity` |
| 已有概率头输出归一化均值、`logsigma` 和采样 embedding | `full_data/upstream_uatvr/modeling.py`，`probabilistic_text` / `probabilistic_video` |
| 原训练已包含加权 MIL 与 KL/VIB 项 | `aqs/core.py` 返回 `0.01 * MIL_loss`、`0.0001 * vib_loss` |
| AQS 难度和多样性选择为无梯度计算 | `aqs/selection.py`，`select_negatives` / `pooled_frames` |
| 当前 AQS loss 对选中分数保留梯度 | `aqs/model.py` / `aqs/loss.py` |
| 仅给内部方差增加正则，不保证获得检索风险信号 | `research/02-dead-ends.md`，DE-3 / DE-4 / DE-5 |

研究前置依据包括 `research/README.md`、`00-north-star.md`、`01-evidence-ledger.md`、`02-dead-ends.md` 和 `07-protocol-constraints.md`。历史研究命题不是已验证结论；旧 study 的失败和终止状态不因本文改变。

外部方法背景：[UATVR 原论文](https://arxiv.org/html/2301.06309v2)；本文的 KL 融合是本次新设计，不声称它就是原论文推理算法。

## 3. 架构与数据流

```text
文本 → CLIP 文本编码 ─┐
                     ├→ 原有语义/时序适配与全局特征对齐
视频 → CLIP 视觉编码 ─┘                  │
                         ┌──────────────┴───────────────┐
                         │                              │
                    原有 WTI 匹配                原有概率分布头
                         │                      文本/视频 μ、logsigma
                         │                              │
                         │                    显式转换、有效性校验
                         │                              │
                         │                      对称 Gaussian KL
                         └──────────────┬───────────────┘
                                   融合分数 S'
                                        │
                ┌───────────────────────┼────────────────────┐
                │                       │                    │
          全 batch 双向 CE       detach 后 AQS 选负      推理融合评分
                │                       │                    │
                │                 固定离散列索引           全库排序
                │                       │
                │                原 S' 提取选中 logits
                │                       │
                └───────────────── AQS 辅助 CE
                                        │
                         加原有 MIL / legacy KL-VIB → 总损失
```

AQS 的视频级多样性仍来自原有真实帧 masked pooling，保持 canonical video ID 去重、全局唯一视频 batch、确定性 tie-break 和全 caption 覆盖。不增加跨 batch 队列。

第一版的概率头仍各自依赖单模态特征；“跨模态匹配”指每对文本/视频的分布比较，不宣称已经有 query-conditioned 视频方差或候选集合条件 posterior。

## 4. 分布参数合同：先解决 logsigma 语义

现有实现存在必须显式保留的约定差异：

1. `probability/tensor_utils.py::sample_gaussian_tensors` 使用 `mu + exp(logsigma) * eps`，因此采样路径的 `logsigma` 是 **log 标准差**。
2. `nn_components/until_module.py::KLdivergence` 使用 `1 + logsigma - mu^2 - exp(logsigma)`，对应的是 **log 方差**形式，且输入均值取自采样 embedding 的均值。

因此，旧 `vib_loss` 在本设计中命名为 **legacy KL/VIB**；不能把它未经转换地说成与采样分布完全一致的解析 KL。本文不修改该历史路径，也不将历史 checkpoint 的训练语义重新解释为正确的解析先验 KL。纠正旧采样/正则关系属于独立改动，需要单独设计和对照。

新匹配分支使用明确的评分分布。设旧头输出为 $a=\mathrm{logsigma}$，先验证 $a$ 和 $\mu$ 有限，再定义

$$
\ell=2\operatorname{clip}(a,\ell_{\min}/2,\ell_{\max}/2),\qquad
q^{score}=\mathcal N(\mu,\operatorname{diag}(\exp\ell)).
$$

这里 $\ell$ 才是新匹配计算的 log 方差。先裁剪再乘 2，避免有限极端输入在转换中溢出。

- 均值复用现有概率头的 L2 归一化输出，不用采样 embedding 的经验均值。
- 评分 log 方差边界设计为 `[-8, 4]`，作为显式、版本化的数值策略记录；不得依赖隐式默认或失败后静默放宽。
- 新评分分布在边界外与旧采样分布不同，这是局限而非等价变换；旧 MIL/VIB 仍接收原始输出。裁剪饱和会切断边界外评分梯度，必须记录上下界命中比例，不能用裁剪掩盖分布退化。
- 文本和视频使用相同维度、相同参数转换及边界，第一版 $d=512$。
- 不增加额外随机采样来计算新评分；Gaussian KL 使用解析参数，FP32 计算。

## 5. Gaussian KL 与分数融合

设第 $i$ 条文本和第 $j$ 段视频的评分分布分别为 $q_i^t$、$q_j^v$。对角 Gaussian KL 为

$$
K(q_i^t\|q_j^v)=\frac12\sum_{k=1}^{d}\left[
\ell^v_{jk}-\ell^t_{ik}
+\exp(\ell^t_{ik}-\ell^v_{jk})
+(\mu^t_{ik}-\mu^v_{jk})^2\exp(-\ell^v_{jk})-1
\right].
$$

本版采用每维归一化的对称 KL：

$$
D_{ij}=\frac{K(q_i^t\|q_j^v)+K(q_j^v\|q_i^t)}{2d}.
$$

$D$ 越小表示这两个评分分布越接近。对称 KL 不是有界距离，也不保证满足三角不等式；仍包含方差比例和候选相关效应，不能宣称对称化就消除了 hubness 或候选偏置。

记原未缩放 WTI 为 $S$、既有可学习 logit scale 为 $\alpha=\exp(\mathrm{logit\_scale})$：

$$
S'_{ij}=S_{ij}-\eta D_{ij},\qquad Z'_{ij}=\alpha S'_{ij}.
$$

融合系数 $\eta$ 要求显式提供，有限且非负；启用融合时须大于 0。第一版使用固定超参数，不新增可学习融合门，避免通过门归零绕开分布路径。本文不指定已经验证的 $\eta$ 默认值；其候选和选择规则需在未来研究配置中预先冻结，只用 validation 选择。

两种 KL 的用途必须区分：$D$ 是匹配代价，进入检索 logits；legacy KL/VIB 是原有辅助正则。不能对所有文本/视频对直接最小化 $D$，否则会同时拉近负例；不能另外重复加入同一个正则权重。

## 6. AQS 与总训练目标

AQS 选择器继续接收方阵分数，但改为 `detach(S').float()`。保留原有负例 min-max 难度、多样性余弦距离、候选屏蔽和 tie-break；难度归一化只针对负例。输入非有限时拒绝执行，不静默清洗。

选择完成后，用未 detach 的 $Z'$ 提取正例和所选负例：

$$
L_{AQS}=-\frac1B\sum_i\log
\frac{\exp Z'_{ii}}{\exp Z'_{ii}+\sum_{j\in A_i}\exp Z'_{ij}}.
$$

总损失为

$$
L=\tfrac12\left(L_{T2V}(Z')+L_{V2T}(Z')\right)
+\lambda L_{AQS}(Z',A)
+0.01L_{MIL}+10^{-4}L_{legacy\,VIB}.
$$

代码中若已有 `mil`、`kl` 是加权返回值，总损失只累加一次。保持完整 batch 双向分母，不用所选负例替代主损失。AQS 仍只有 T2V 辅助项。

当前已训练 AQS 的 `M=16`、`lambda=0.2` 可作为同协议比较的保持项；它们不是新融合结构的有效性证据。原 seed、全局 batch64、5 epochs、学习率5e-5、优化器和初始化必须在比较中一致并绑定实际配置。

梯度合同：

- `CE/AQS → Z' → D → μ/logsigma → 原有可训练语义/编码参数` 必须可达。
- WTI 主路径同样从 $Z'$ 接收梯度；不冻结整个 backbone 后只训练一个无法影响排序的旁支。
- 离散选择索引、多样性表示和困难度计算不反传；不通过 straight-through 估计器修改 AQS。
- 不用 `D.detach()` 生成训练 logits。只记录诊断时允许 detach。
- 不将大方差直接作为“假负例”标签、删除条件或负样本权重。不同 ID 的语义假负例问题仍然存在。

## 7. 推理、分块与兼容模式

融合启用时，推理也使用 $Z'$，不执行 AQS，不依赖训练 batch 的对角正例或 video ID。既有文本/视频编码接口可保留，评分扩展须取得语义适配后的概率参数；不能错误地从语义适配前 token 另算一套分布。

计算新分布评分不采样；相同输入与状态下不依赖 MC seed。保留原有评估对训练 RNG 的保护，不以此偷偷改变旧 forward 的随机数消耗。

评分适用于矩形 `[query_count, video_count]`，支持完整图库分块。逐模态分布不应随候选集合、候选顺序和分块大小变化；有效 token/frame mask 必须正确，纯评分不得原地修改输入。

已有 UATVR 池化/mask 路径存在 singleton 别名风险，新路径不得调用 Laya 的 monkey-patch 绕过。融合评分验收须检查重复调用、输入不变性及 singleton/不同分块一致性；若发现旧共享 helper 会破坏新分布语义，使用显式隔离适配并记录差异。不得顺手改旧禁用路径来掩盖不等价。

融合 `enabled=false` 或 $\eta=0$ 时，在进入新分布评分前直接走当前 AQS 旧路径。必须保持旧 logits、选择、损失、梯度、状态键和 RNG 行为的等价性，不能“算完再乘 0”。AQS 的 $\lambda=0$ 只关闭 AQS；当融合启用时，分布评分仍应参加主损失和推理。

DSL 第一版保持显式关闭，以 raw 融合评分进行验证；若将来改变 DSL，必须形成独立配置。新分数不能混用旧 WTI 缓存或旧阈值。

## 8. 实现边界与计算开销

以下是拟议边界，不表示已创建实现：

| 组件 | 职责 |
|---|---|
| 独立 `aqs` 融合配置 | schema、enabled、eta、维度归一化、log 方差转换/边界、评分模式校验 |
| 纯概率匹配函数 | 接收两个模态的均值/log 方差，返回矩形 KL 代价；无采样、无 ID、无 I/O |
| AQS core 显式扩展 | 复用语义特征和已有概率参数，提供 WTI/KL/融合分数；vendor 源码保持不变 |
| 训练/评分适配器 | 使用统一评分函数，接入完整 CE、AQS 和原有 MIL/VIB |
| checkpoint / 启动校验 | 绑定融合语义、源码、输入和恢复身份，拒绝缺失配置 |

当前 observer 在概率分支之前捕获裸 WTI；新适配器须在取得概率参数并完成融合后，再向新模式的选择器暴露融合分数。不能仅修改旧 observer 的名字或记录值，却仍用旧 logits 训练。

训练复用现有一次特征 all-gather，分布在对齐后的全局特征上计算，不再重复 gather 或额外平均全局损失。rank 之间必须核对完全相同的融合配置和 ID 映射。

不构造完整 `[B,B,d]` 或 `[Q,V,d]` 常驻张量。对称 KL 的方差比与加权平方均值差可用矩阵乘法和逐模态向量和计算，或按 query/video 分块；允许 $O(QVd)$ 运算，但额外常驻空间控制在 $O(QV+(Q+V)d)$。避免逐对重复运行概率头。方差倒数、指数和 reduction 使用 FP32，有限性校验先于选择与 loss。

新函数须验证形状、设备、浮点类型、有限性和参数边界。只允许预先明确的浮点舍入容差；非有限值或显著负的 KL 是错误，不能无条件 `nan_to_num` 或大幅 clamp。实现应核对解析 KL 与可信分布实现的一致性及接近分布时的消减误差。

## 9. 状态、恢复与初始化

采用独立评分身份，例如 `uatvr-aqs-kl-v1`；保留 `uatvr-aqs-v1` 的历史含义。不直接改写 `AqsConfig` 旧 recipe 常量以伪装兼容。

recipe/checkpoint 至少绑定：融合 schema、eta、对称 KL 定义、按维归一化、`logsigma` 转换、上下界、精度、legacy VIB 版本、AQS 配置、model recipe、可训练参数、sampler/ID 映射、source SHA、相关源文件哈希、输入哈希和 world size。新增评分源文件须进入完整性绑定，不能只沿用当前六个 AQS 文件的哈希清单。

即使复用旧头而没有新增参数键，相同 `state_dict` 也不代表相同评分语义：

- 正式恢复要求新评分身份和全配置一致，并恢复优化器、进度和各 rank RNG。
- 旧 AQS checkpoint 不允许自动当作融合 checkpoint 续训；需要 warm-start 时必须显式记录父权重、初始化规则、优化器重置和新训练进度。
- 主要比较应使用相同原始 CLIP 初始化和训练预算，避免把“旧 AQS 再训 5 轮”与“基线只训 5 轮”比较后归因于结构。
- 不通过 `strict=False` 绕过状态缺失或不匹配；显式禁用融合的历史回放应继续走原接口。

## 10. 诊断与软件验收

必要标量诊断：WTI、$D$、$\eta D$ 的均值/范围/离散程度，正负例代价差，均值/方差头梯度与更新量，log 方差分位数及边界命中率，融合前后排名/所选 ID 改变比例，主损失/AQS/MIL/legacy VIB 各项，显存与耗时。按 rank 核对，不能只以 rank0 的有限 loss 判断整个训练健康。

软件验收必须覆盖：

1. **分布语义**：已知一维分布手算，非单位标准差案例，`logsigma → 2*logsigma` 转换，对称性、相同分布的零代价、按维归一化，FP32 有限性；用独立参考实现交叉核对。
2. **真实梯度**：隔离原有 MIL/VIB，仅从融合 CE 或 AQS 回传，确认方差头、均值头和允许训练的 encoder 获得非零梯度；避免原有辅助梯度掩盖新路径断开。
3. **语义作用**：保持 WTI 和均值不变，只改变某个分布方差，在构造样本中确认相应 KL、融合排序/选样能够改变。候选无关的全行常数不能当作有效排序改变。
4. **兼容退化**：关闭融合或 eta0 时与 `5886cd1` 同输入/状态的旧路径对比；日志之外的数值、梯度、选择与 RNG 均核对。AQS 关闭但融合开启也须独立验证。
5. **评分合同**：方形训练与矩形推理使用同一对样本的评分公式；验证分布计算本身的 train/eval 一致性，测试 eval 重复调用、分块/排列/singleton、输入 mask 不变与无随机采样。不要求整个含 dropout 的训练网络和 eval 网络逐位相同。
6. **分布式与恢复**：单进程参考与多 rank 的全局 loss/梯度比例、候选 ID 顺序、rank 配置不一致的集体拒绝；新模型实例精确恢复后比较下一步 loss、参数、优化器与 RNG。容差须预先明确。
7. **依赖与身份**：Laya 不可访问时路径仍成立；错误评分版本、转换规则、eta、边界、源码、输入或 checkpoint 身份在修改模型前拒绝。

大于 1 MiB 的逐查询结果、分数矩阵和诊断数组外置 NPY/NPZ，并绑定哈希；JSON/日志只存标量摘要和引用。

## 11. 机制判断与后续比较原则

本节规定论证需要的证据，不构成实验预留、运行矩阵、预算或启动命令：

- 当前 AQS、同均值但固定单位方差的 KL 融合、可学习方差融合需要同协议比较，区分额外均值距离与方差贡献。
- 比较固定/共享方差、简单温度/尺度控制，排除只靠重新缩放得分获得的收益。
- 对照“融合打分但仍按 WTI 选负”与“融合打分且按融合分数选负”，区分匹配改动和选样改动。主版本仍是后者，不能评估后再改口定义主版本。
- 检查候选偏置和 hubness；对称 KL 本身不能证明它们已消除。
- Recall 改善只能支持检索效用。若要声称预测不确定性有效，需在 validation 上与 margin/entropy 等明确基线比较 failure discrimination、risk-coverage 或校准指标，不能将原始方差直接报告成正确概率。
- 新旧分数的评估数值必须来自同一数据认证、精度与评分协议；当前单 seed AQS 验证结果不等于新模型成绩，也不应与历史不同协议的高 Recall 直接相减。

融合可能只学到均值距离、方差可能趋同或饱和、分布尺度可能压过 WTI、未知假负例仍可能被强化；这些都是需如实报告的否定结果。仅有 Recall 上升而无方差增量证据，不支持“不确定性学习有效”的因果表述。

## 12. Laya 隔离与研究生命周期

不 import `semantic_retrieval.laya_rerank`，不加载 Laya 权重/processor/环境叠加，不使用其缓存、概率、排名、伪标签或教师信号；不读写、改造其 study/registry/产物，不移植 Laya 运行时 monkey-patch。继续通过独立 AQS 适配层与严格依赖检查实现隔离。

本文只新增设计文档，不修改训练实现，不新建研究控制面、不预留或启动 run、不访问 test，不修订旧 gate。旧已终止 study 不复活。

未来执行前须重新核对当时生效的 AGENTS、研究前置文档、`experiments/contextual-risk-v2/study.yaml` / `registry.jsonl` 和实际适用的新控制面，明确归档合同、GPU 资格与研究授权。用户此前为当前 AQS 训练选择双卡，不等同于当前规则下新评分架构自动取得双卡科学实验授权；其他 study 的双卡例外不继承。

训练使用 train，选择/消融使用 validation，MSRVTT/MSVD test 继续 sealed。必须按适用协议 reserve-phase、提交 reserved 记录、冻结源码/配置/命令/输入/manifest 后再启动；运行中不修改受跟踪文件；终态按规则核验与归档。旧 AQS 运行存在完成记录不代表本文已经审计或认可其生命周期满足当前规则，不能据此绕过新链路要求。

## 13. 自审结论

本版确定了复用已有概率头、在 WTI 后/AQS 前融合、解析对称 KL、显式 log 标准差转换、legacy VIB 不变、完整双向 CE 与 T2V AQS、训练/推理共用评分、detach 边界、关闭模式退化、checkpoint 评分身份和 Laya 隔离。

eta 无科学默认值，必须在后续注册中明确；这不影响接口或公式定义。log 方差边界是本文明确的数值设计，不是效果验证。文中 Gaussian KL 公式已在 CPU 上以非单位方差、矩形文本/视频集合，对照 `torch.distributions.kl_divergence` 通过核对；这不是新模型的软件或 GPU 资格验证。本文没有新模型实现、Recall 收益、校准收益、资源资格或实验成功声明。
