# UATVR 科研问题与路线图

> 更新时间：2026-07-12。本文是科研决策、证据、停止条件和实验顺序的唯一事实源。
> 外部论文事实与复算见 [多模态检索研究综合分析](../analysis/multimodal_retrieval_research_synthesis.md)。
> 历史结构、旧报告、日志与 checkpoint 仅从 Git 历史追溯。

## 1. 当前决策摘要

1. 当前唯一 P0 是建立可信的 OpenAI CLIP WTI-only 基线；在其完成前，不判断其他 backbone、uncertainty 或 alignment 机制的收益。
2. MSR-VTT 唯一有效协议是 `trusted-v1`。JSFusion 1K 不参与训练、调参或 checkpoint selection，只在方法和超参数冻结后显式盲测。
3. 不再把“所有数据集达到 SOTA”作为主要成功标准，但核心方法仍必须相对同协议 P0 稳定改善检索。研究优先设计新的文本—视频匹配或训练机制，并用受控实验验证排序收益、因果来源和可复现性。
4. 不进行盲目 backbone sweep。backbone 在核心研究中是匹配控制变量；EVA 只可作为后续外部有效性对照，不包装成方法贡献。
5. 后续执行顺序为 P1 只读跨模态错配机制诊断 → P2 pair-level 双源不确定性感知匹配/训练 → P3 candidate-conditioned alignment；P2/P3 是由 P1 分别支持的独立假设，P3 不以 P2 成败作为解锁条件。二者独立成立后才研究融合；P4 时序/效率始终独立。
6. Hard negative 与 UACL 主线均已终止，不再 sweep 或 repeat。语义相似度只能用于诊断，不能改变正例定义。

## 2. P0：可信 WTI-only 基线

### 2.1 固定协议

| 字段 | 固定值 |
|---|---|
| 数据协议 | `trusted-v1` |
| split seed | 42 |
| train / internal val | 8500 / 500 |
| test | JSFusion 1K，仅在方法、超参数和 checkpoint selection 冻结后显式盲测 |
| 主损失 | 按精确 `video_id` 构造的双向多正例 InfoNCE |
| backbone | OpenAI CLIP ViT-B/16 |
| global forward batch | 256 |
| GPU / micro / accumulation | 4 / 64 / 1 |
| LayerNorm | native FP16，保留 FP32 master affine；`CLIP_LAYER_NORM_PRECISION=fp32` 只作回退 |
| 激活显存 | OpenAI CLIP 视觉 Transformer 前 4 层启用 activation checkpointing，不改变 forward batch |
| checkpoint selection | internal-val T2V R@1 |

forward contrastive batch 与 optimizer effective batch 必须分别记录。梯度累积不会合并不同 forward 的 in-batch negatives；任何对照都必须保持 global forward batch、GPU 数、每卡 micro-batch、accumulation 和 optimizer steps 一致。

### 2.2 稳定实现事实

- 主入口固定为 `run_train_msrvtt_bg.sh` → `train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_retrieval.py`。
- hygiene profile 的主分数固定为 `weighted_logits = wti_logits`，并在 forward 中真实绕过旧空间增强、概率表示、不确定性头和辅助 loss 路径；仅把 loss 权重设为 0 不满足要求。
- WTI padding mask、精确 `video_id` 多正例矩阵以及 train/internal-val/test 隔离必须由测试持续保护。
- OpenAI CLIP 自定义 LayerNorm 默认执行 native FP16，FP32 master affine 参与参数更新；环境变量只提供显式回退，不代表全模型 AMP。
- OpenAI CLIP 视觉 Transformer 默认只对前 4 层启用 activation checkpointing，在 40GB A100 上平衡显存与重计算；该工程开关及层数必须写入实验 manifest，且不改变 forward contrastive batch。
- TQFS 对退化视频按实际不同帧特征数聚类，不允许依赖缺失时静默退化；预处理帧按 `video_id` 写入带配置契约的共享原子缓存，缓存命中与在线路径必须保持张量一致。
- `--batch_size` 表示目标有效 batch。accumulation=1 时，4 卡、`batch_size=256` 对应 global forward batch 256、每卡 micro-batch 64。
- 历史 checkpoint 与已删除的旧分支参数不属于兼容接口；P0 只接受从当前 WTI-only 配置重新训练的 checkpoint。

### 2.3 尚未完成的结果

- 尚无符合上述全部条件的可信 OpenAI CLIP WTI-only 训练结果。
- 历史 48.2、49.x 等结果存在旧 split、test-aware selection、正例定义或活动分支混杂，只能从 Git 历史追溯，不能作为 `trusted-v1` baseline。
- 在 baseline 完成前，不启动长期 uncertainty/alignment 训练，不根据一次验证波动改变路线。

### 2.4 P0 完成门槛

P0 同时满足以下条件才算完成：

1. split manifest 精确为 seed 42、8500 train / 500 internal val，且 JSFusion 1K 不被训练流程加载。
2. 主损失只使用精确 `video_id` 双向多正例矩阵，T2V/V2T 均有单元测试。
3. hygiene WTI-only 的禁用分支通过 spy/mock 测试证明没有被调用，final score 与 WTI logit 数值一致。
4. 运行清单记录 Git SHA、split、seed、backbone、数据路径、global forward batch、micro/accumulation 与 hard-negative 关闭状态。
5. 用户手动完成训练；以 internal-val T2V R@1 选择 checkpoint，并保存完整 T2V/V2T R@1/R@5/R@10、MdR/MnR。
6. 在方法、超参数和 checkpoint 固定前不读取 JSFusion 1K；盲测结果只报告一次，不反向调参。

## 3. 外部研究证据采纳表

| 论文机制 | 证据等级 | 决定 | 对应阶段 | 理由 | 综合分析证据 |
|---|---|---|---|---|---|
| EagleNet candidate interaction | 协议不兼容、部分已核验 | 延后 | P3 | 先证明固定 embedding 存在系统性候选错配 | [EagleNet](../analysis/multimodal_retrieval_research_synthesis.md#2-eaglenet) |
| GARE candidate-conditioned correction | 协议不兼容、部分已核验 | 延后 | P3 | 必须限制为 top-k 并报告候选规模与重排成本 | [GARE](../analysis/multimodal_retrieval_research_synthesis.md#3-gare) |
| TempMe temporal merging | 已核验边界 | 独立支线 | P4 | 不与 P2/P3 首轮实验混合 | [TempMe](../analysis/multimodal_retrieval_research_synthesis.md#4-tempme) |
| GraviAlign Gaussian overlap | 弱证据 | 仅采纳研究启发 | P2 | 需要重新有界化、推导并验证其对匹配学习与最终排序的作用；校准只作辅助证据 | [GraviAlign](../analysis/multimodal_retrieval_research_synthesis.md#5-gravialign) |

外部论文的传统 9k/1K Recall 不能冒充本项目基线。采纳某个机制只表示它形成可测试假设，不表示接受作者的全部理论解释、协议或工程代价。

## 4. 后续研究问题

### P1：跨模态错配机制诊断

**目标**：先回答固定 WTI “何时错、为何错、哪些错配可能由新机制纠正”，为匹配/训练创新建立可证伪假设；不改模型、不生成伪标签。

固定 backbone、主损失、正例矩阵和 checkpoint 后，分别对 T2V/V2T 统计 correct/incorrect Top-1 的 score、top1-top2 margin、GT rank、row entropy、token/frame alignment 和近邻密度。重点比较：

- high-similarity hard/fuzzy negatives 与按精确 `video_id` 定义的真实多正例；
- query ambiguity、video ambiguity 与 pair mismatch；
- 短文本语义不足、视觉近邻、动作/顺序错误和帧质量错误；
- WTI score/margin 是否已足以解释错误，是否存在可被 pair evidence、概率几何或候选条件交互修复的额外错配结构。

P1 只输出聚合统计、预定义错误切片和可视化，不修改 label、loss 或训练数据。只有错误模式在至少三个 seed/checkpoint 或预定义数据切片上方向一致，才作为 P2/P3 的前置证据。

### P2：Pair-level 双源不确定性感知匹配/训练

**研究对象**：对每个 `(text_i, video_j)` 分解 pair-level aleatoric 与 epistemic uncertainty，并将其用于新的文本—视频匹配或训练机制；不是为 query/video 单独输出 scalar，也不是构建拒答、人工审核或部署可信路由。

**核心假设**：两类不确定性描述不同的 pair 错配来源。若将这种差异纳入表示、token/frame 证据聚合、相似度几何或对比学习过程，应能改善困难候选排序。检索收益是主结果；校准和错误检测只验证 uncertainty 的语义是否成立。

独立规格必须明确：

1. `u^ale_{ij}` 与 `u^epi_{ij}` 的统计含义、估计方式、可辨识边界，以及为何二者都是 pair-level；epistemic 必须体现随模型/数据证据增加而可约减的模型无知，aleatoric 必须对应在模型分歧降低后仍持续的数据歧义。若不能验证该区分，只能称为 data-ambiguity/model-disagreement signal；内容/帧分歧不能仅凭命名冒充 epistemic uncertainty；
2. uncertainty 只选择一个首轮机制入口：pair score、token/frame evidence aggregation，或对比学习中的 temperature、margin、weight/gradient allocation；首轮不得同时修改 score、表示和多项 loss；
3. 正配对、精确多正例和所有按 `video_id` 定义的非匹配 pair 如何沿用既定双向多正例 InfoNCE；high-similarity negative 只作预定义评价切片，不得单独采样、加权、修改标签或恢复 hard-negative 主线；
4. 若机制仅在训练期生效，必须说明它如何改变梯度与最终学习表示；若机制进入推理，必须定义最终 similarity/logits 及严格关闭后的 P0 回退；
5. 方差塌缩/爆炸、无界 score、数值奇异、分数尺度和双分量退化为同一信号如何防护；
6. 如何证明 `u^ale/u^epi` 不是 WTI score、margin、row entropy 的简单复制，如何用参数量/计算量匹配对照排除额外 stochastic pass 或模型容量带来的收益，并分别消融 aleatoric、epistemic 与二者组合。

最小方法从稳定、可有界的 pair 特征、Gaussian log-overlap 或 uncertainty-aware contrastive objective 开始。主评价为 T2V/V2T Recall、GT rank、MdR/MnR 及预定义错配切片；AURC、错误检测、Brier/NLL 和分桶校准只作附加语义验证。若至少三个 seed 下不改善检索，不能靠追加辅助 loss、校准或拒答实验挽救其核心研究地位。

### P3：Candidate-conditioned multimodal alignment

P3 处于锁定状态。只有 P1 提供“固定 embedding/WTI 在 top-k 内存在系统性、可纠正的候选错配”证据后才解锁。

首个规格应固定 top-k，使用 bounded residual 或轻量 pair interaction/reranker 直接纠正候选排序；第一版不使用 risk gate。必须同时报告：

- 第一阶段候选规模 `k` 与 candidate recall ceiling；
- 最终 score 与关闭 reranker 后的严格基线回退；
- T2V/V2T Recall、GT rank、rank flip、预定义错配切片和失败样本；
- residual/correction 范数及其与实际排序修复的关系；
- 与同参数量非候选条件 MLP 的容量对照。

`k` 对延迟、吞吐和显存的曲线是方法复杂度与公平比较边界，不是项目的部署目标。不允许从 1K 全对全时间直接推断大库扩展性，也不允许在首轮同时加入图网络、随机采样、能量目标和多项正则。

P2 与 P3 首先独立验证。只有二者分别相对 P0 和各自容量控制获得稳定检索收益，才研究 uncertainty 如何调节 pair evidence、交互强度或 residual 幅度；该融合仍以改善检索为目标，不引入 abstention、人工审核或产品风险路由。

### P4：时序建模与效率支线

P4 永远作为独立支线。第一步固定 P0 全部条件，只加入**无 token 压缩**的跨帧 temporal modeling，判断是否修复预定义时序错误；第二步才加入 token reduction，分别报告准确率与效率 Pareto。

首轮 P4 不得与 P2/P3 同时改变。评价至少包括 video-backbone GFLOPs、tokens、videos/s、训练/推理显存和检索指标；若压缩带来效率但损失不可接受，应如实保留负结果。

## 5. 成功标准与停止条件

### 5.1 不再采用的成功标准

- 不以“所有数据集达到 SOTA”作为唯一或主要目标。
- 不以单 seed 的最高 R@1、传统 test-aware 协议数字或跨论文绝对 Recall 判断成败。
- 不把更强 backbone、更多参数、更多输入帧或额外后处理本身包装成方法创新。
- 不以 loss 下降、variance 非零、校准改善或可视化好看替代最终检索排序证据。
- “不追逐全数据集 SOTA”不等于检索收益可有可无；核心方法必须相对同协议 P0 在预注册主指标上形成稳定改善。

### 5.2 机制筛选标准

- 单 seed 只作机制筛选。新增推理 scorer 必须保证训练—推理定义一致；纯训练机制可以在推理时回到 WTI，但必须通过梯度路径、直接消融和最终排序证明因果作用。
- 一次直接消融只改变一个 causal variable，并提供严格关闭后的 baseline 回退。
- 主张必须有对应观测量：匹配/alignment/uncertainty-aware training 首先对应 Recall、GT rank 与错误切片修复；aleatoric/epistemic 语义另对应去冗余、校准或错误相关性；效率对应实测资源。
- 只有 test 提升、没有 internal-val 机制信号的实验视为无效，不得继续调 test。
- 没有机制信号、收益只随换 backbone 出现，或必须同时换标签/多项 loss 才出现时立即停止。

### 5.3 稳定性与泛化标准

- 进入主线前，基线与新方法使用同一固定 split 和同一组**至少三个训练随机种子**。
- backbone、split、forward contrastive batch、optimizer steps、checkpoint selection 和数据预处理完全匹配。
- 常规检索报告 R@1/R@5/R@10、MdR/MnR，分别给 T2V/V2T 的均值与离散度。
- 声称 uncertainty 时额外报告 AURC、错误检测 AUROC 或相关性、Brier/NLL、分桶校准与离散度；这些指标只验证不确定性语义，不能替代检索收益。
- 至少在一个独立数据集或预定义错误切片上验证方向一致；结果不要求每个数据集、方向和指标都最好，但核心方法必须相对同协议 P0 稳定改善至少一个预注册检索主指标，且不能以另一方向的明显退化换取局部数字。

### 5.4 盲测边界

JSFusion 1K 是显式盲测，只能在以下内容全部冻结后读取：方法结构、loss、超参数、训练 seed 集、checkpoint-selection 指标、停止条件和报告模板。盲测之后不得根据其结果继续选 checkpoint 或调参；需要新假设时必须回到 internal val，并将下一次 test 视为新的预注册阶段。

### 5.5 逐阶段停止条件

- **P0**：任一协议/隔离/主损失/forward hygiene 契约失败就停止训练请求，先修实现。
- **P1**：若错误可由现有 WTI score/margin 充分解释，且没有稳定、可纠正的新错配结构，则停止新增复杂分支。
- **P2**：若至少三 seed 不改善预注册检索主指标，或收益只来自参数量/额外计算，则停止核心路线；若双分量退化为基线分数的复制或彼此不可区分，则不能声称 aleatoric–epistemic 创新。仅改善校准、错误预测或拒答效果不足以保留其核心研究地位。
- **P3**：若 P1 无固定表示错配证据、candidate recall ceiling 太低、至少三 seed 无稳定排序收益、目标错误切片未被修复，或收益仅来自参数量，则不解锁/立即停止；k-cost 只界定方法适用范围，不替代检索结论。
- **P4**：若无压缩 temporal modeling 没有机制收益，则不把 token reduction 的速度收益解释为检索创新；若效率不形成 Pareto 改善则停止。

## 6. 固定实验顺序

1. 完成 P0 代码契约和可信 OpenAI CLIP WTI-only baseline。
2. 冻结 P0 checkpoint，只做 P1 只读诊断。
3. 根据 P1 预注册一个 P2 机制入口（matching score、evidence aggregation 或 training objective 三选一），先单 seed 筛选，再以同一组至少三 seed 复核；双源分量必须分别消融。
4. 只有 P1 明确支持 candidate-conditioned correction 时才进入独立 P3；否则保持锁定。
5. 只有 P2、P3 分别成立后，才预注册 uncertainty-conditioned alignment 融合实验；不得用融合结果替代两个独立基线。
6. P4 可在 P0 后单独开展，但不与 P2/P3 共用首轮实验。
7. EVA 只作匹配 backbone control：在方法稳定后，以相同 split、batch、steps 和 selection 检查外部有效性，不作为创新点。
8. 所有方法与超参数冻结后，才执行一次 JSFusion 1K 显式盲测。

## 7. 已关闭路线

| 已关闭路线 | 当前状态 | 追溯方式 |
|---|---|---|
| SAP 及其依赖链 | 已删除、不得恢复 | 历史结构、实验和止损证据仅从 Git 历史追溯 |
| Hard negative 主线 | 已终止 | 只保留独立诊断入口，不再 sweep 或 repeat |
| UACL 主线及活动接口 | 活动接口已删除，不恢复 | 历史结构、实验和止损证据仅从 Git 历史追溯 |
| Semantic soft target / 伪标签 | 禁止 | 正例只由精确 `video_id` 定义 |
