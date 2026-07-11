# SAP 退役与多模态检索科研路线转向设计

> 日期：2026-07-11
>
> 状态：对话设计已获用户批准，等待书面规格复核
>
> 适用范围：论文综合分析、科研 SSOT 更新、活跃主训练链路中的 SAP 及其依赖分支清理

## 1. 背景与决策依据

UATVR 当前可信主入口为：

```text
run_train_msrvtt_bg.sh
→ train_msrvtt.sh
→ main_task_retrieval.py
→ modules/modeling_mulit.py
```

现有科研证据已经说明：

- SAP 是 video-only semantic anchor pooling，同一视频面对不同 query 时不能表达不同的相关性或置信度。
- global probabilistic mean、AnchorWTI 和 query-conditioned SAP 已完成止损实验，没有形成稳定、可解释的排序增益。
- SAP 输出的 `mu_raw/logsigma` 同时承载视频概率表示，进一步连接 MIL、evidential、negative regularization、UACL 和多组诊断字段；这些分支与最终 WTI 排序长期脱节。
- 当前可信基线仍未完成正式训练验证，因此更换 backbone 不能替代实验卫生，也不能证明新方法贡献。
- EagleNet、GARE、TempMe、GraviAlign 的共同启示不是“每个数据集都必须 SOTA”，而是以清晰问题、受控对照和机制证据支撑创新主张。论文中也存在只在部分数据集或部分指标领先的情况。

因此，本设计采用已经确认的“完整退役”方案：SAP 不再保留为主线模块、兼容入口或消融对象；同时清理活跃主训练路径中只为 SAP 概率表示服务的旧概率/不确定性链。项目转向固定可信基线上的 pair-level uncertainty 与 multimodal alignment 研究。

## 2. 目标与非目标

### 2.1 目标

1. 形成一份可长期引用的四篇论文综合分析，区分论文事实、代码事实、作者主张和本项目推论。
2. 更新 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`，使其成为新决策下唯一有效的科研事实源。
3. 从活跃主训练链路中彻底删除 SAP 及依赖其输出的旧概率、不确定性、打分、日志和 CLI 路径。
4. 删除 SAP 专属文档，以及混合文档中围绕 SAP 架构、实验、消融和未来改进的段落。
5. 保留并强化 trusted-v1、双向多正例 InfoNCE、WTI 与 test 隔离构成的可信实验基座。
6. 把后续创新目标改为可证伪的研究假设，而不是无条件追逐所有数据集 SOTA。

### 2.2 非目标

- 本次不实现新的 pair-level uncertainty head、Gaussian score、candidate-conditioned reranker 或 temporal merging 模块。
- 本次不启动训练，不生成新 checkpoint，不在 JSFusion 1K test 上调参。
- 本次不做新的 backbone sweep，也不把 EVA/OpenAI CLIP 的差值写成方法创新。
- 本次不恢复 hard negative、UACL、semantic soft target 或 SAP 消融。
- 本次不清理 `research_refs/`，也不把本地参考论文、第三方代码或权重纳入 Git。
- 不因名称相似而删除与 SAP 无依赖关系的旧文件；不相关遗留代码需要另行授权和规格。

## 3. 文档架构

本次形成三层文档，职责不得混用：

| 文档 | 职责 | 是否是科研 SSOT |
|---|---|---|
| `docs/analysis/multimodal_retrieval_research_synthesis.md` | 四篇论文的详细创新、实验、结果、局限与可迁移方向 | 否 |
| `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` | 项目已经接受的结论、当前问题、停止条件和实验顺序 | 是 |
| `docs/README.md` | 提供入口并解释上述两份文档的关系 | 否 |

`AGENTS.md` 也必须同步当前稳定事实：删除项目描述、当前决策、代码入口和稳定事实中的全部 SAP 引用，改为 WTI-only 可信基线与未来 pair-level uncertainty/multimodal alignment 的中性表述。`AGENTS.md` 只保留执行约束和稳定实现事实，不承载详细论文讨论。

文档迁移还必须处理当前已发现的结构问题：

- 删除 `report_SAP.md`。它完全服务于已经终止的 SAP 架构、消融和 v2 设计，不迁移正文。
- 删除 `report.md` 和 `report_uncertainty.md`。两者描述的是已经废弃的 SAP/旧概率耦合实现；其中仍成立的 SAP-independent 原则只能重新提炼为“pair-level、进入最终排序、需要校准”等中性结论，写入新综合分析或 roadmap，不能复制旧实现段落。
- Roadmap 删除 SAP 的结构说明、问题清单、实验表、消融建议和未来扩展方案，只在“已关闭路线”表中保留一条墓碑式记录：`SAP 及其依赖链已删除，不再恢复；历史细节见 Git`。这是当前文档允许保留的唯一 SAP 决策记录。
- 消除 roadmap 中两个“P0”的编号冲突；P0 只表示可信 WTI-only 基线，其余研究问题使用独立编号。
- legacy hygiene 的协议教训可用中性语言保留；global probabilistic score、AnchorWTI、QC-SAP 等所有 SAP 派生变体统一收进同一条墓碑记录，不再单列名称、结果或未来建议。
- `docs/analysis/query_branch_analysis.md` 保留为历史结构分析，但必须标明其中旧指标和候选改法不是当前决策。
- `docs/README.md` 不继续链接当前工作树中已经不存在的文档，也不把临时设计规格列为长期科研入口。

新论文综合分析不复述 SAP 的网络结构、实现细节或消融路线；它只从项目决策角度说明“旧的 sample/video-level 辅助不确定性路线已停止”，并直接讨论新的研究要求。设计规格在实施完成后按既有惯例从当前工作树归档，由 Git 历史保留，因此不会成为长期 SAP 文档入口。

## 4. 四篇论文综合分析设计

### 4.1 单篇论文统一模板

EagleNet、GARE、TempMe、GraviAlign 分别使用同一结构：

1. 论文要解决的具体矛盾。
2. 核心创新与数据流。
3. 与普通 embedding similarity、token-wise matching 或概率表示的实质差异。
4. 训练目标与推理最终分数是否一致。
5. 实验数据集、split、backbone、训练设置、比较对象和指标。
6. 主结果与消融实际支持到什么程度。
7. 没有被实验支持的强主张、复现缺口或理论风险。
8. 对 UATVR 可迁移与不可直接照搬的部分。

正文中的关键结论必须就近给出 PDF 页码、表号、公式号或本地代码位置。没有代码的 GraviAlign 必须把“论文明确说明”和“根据公式推断”分开书写。

### 4.2 横向比较

综合文档至少包含以下比较维度：

- 创新发生在表示、交互、时间建模、概率建模、打分还是训练目标。
- 方法是独立编码、pair-conditioned 交互还是候选重排。
- 新模块是否直接进入最终检索分数。
- 主要增益是否来自同 backbone 受控对照。
- 是否报告多 seed、显著性、效率和校准。
- 是否在所有数据集、所有方向和所有 Recall 指标领先。
- 方法复杂度、推理扩展性以及大规模候选库的可用性。

每条关键结论标记为“已核验”“弱证据”“未验证”“协议不兼容”或“不可复算”。Roadmap 只保留一张外部证据采纳表，记录论文机制、证据等级、采用/拒绝/延后决定、理由和综合文档锚点。

不制作简单的“SOTA 排名榜”。表格应突出证据强度和适用边界，避免把不同 split、额外数据、后处理或 backbone 的数字直接混为一谈。

### 4.3 对项目的结论

综合分析应明确形成以下判断：

- 盲目换 backbone 只能建立模型能力上限或控制变量，不能替代方法假设与机制验证。
- 不在所有数据集达到 SOTA 不等于研究失败；更重要的是贡献是否新颖、因果链是否闭合、实验是否能反驳自身假设。
- EagleNet 与 GARE 的核心交互会产生 candidate-conditioned 表示，部署语义更接近全库 pairwise scorer 或 top-k reranker；不能只按网络 FLOPs 忽略候选对数量和固定向量索引失效。
- TempMe 主要改变视频 backbone 内的 token 流和时序计算，应作为可离线编码的独立效率/时序支线评价。
- GraviAlign 可借鉴的是经过有界化和数值稳定处理的 Gaussian overlap 思路；其引力项、semantic mass、自防塌缩和 independent-veto 叙事不能未经重新推导直接采用。
- 四篇论文在 MSR-VTT 上采用的公开协议或代码选模方式均不满足本项目的 trusted-v1 盲测要求，绝对指标只能作为外部证据，不能充当本项目基线。
- 新的不确定性必须是 query-video pair-level，并对最终排序或检索风险提供可测作用。
- 新的多模态对齐应优先解决“何时、为何发生错配”，而不是继续叠加无解释辅助 loss。
- TempMe 式时序效率、GARE 式候选条件修正、分布式对齐等方向只能作为相互独立的研究变量，不能一次实验同时改变。

## 5. 科研 SSOT 的新结构

路线图将从“修补 SAP 概率链”改写为以下结构：

### P0：可信 WTI-only 基线

- 固定 MSRVTT `trusted-v1`：split seed 42，8500 train / 500 internal val，JSFusion 1K 仅作显式盲测。
- 主损失固定为按精确 `video_id` 构造的双向多正例 InfoNCE。
- OpenAI CLIP ViT-B/16、全局 forward batch 256、四卡每卡 micro-batch 64、accumulation 1 是当前可信基线口径。
- OpenAI CLIP 自定义 LayerNorm 默认 native FP16、保留 FP32 master affine 参数；`CLIP_LAYER_NORM_PRECISION=fp32` 只作回退。该实现已经合并，但可信训练结果尚未产生。
- P0 的目的不是追 SOTA，而是建立所有后续假设共享的可解释参照。
- P0 未完成前，不评价新对齐或新不确定性机制。

### P1：跨模态错配与检索风险诊断

在不改 backbone 和主损失的前提下，先定义并统计：

- correct/incorrect Top-1 的分数、margin、rank 和 token alignment 差异；
- high-similarity hard/fuzzy negatives 与真实多正例的差异；
- query ambiguity、video ambiguity 和 pair mismatch 的可区分性；
- T2V 与 V2T 是否呈现不同的错误模式。

P1 只做诊断，不改变正例矩阵，不生成伪标签。

### P2：Pair-level uncertainty

后续独立规格必须回答：

- 不确定性变量 `u(text_i, video_j)` 的操作性定义；
- 它如何接受正、负候选监督；
- 它如何直接进入最终 logits、排序决策或 selective retrieval；
- 如何防止方差塌缩、方差爆炸、分数无界和只学到样本难度；
- 如何证明其不是 WTI 分数或 margin 的单调复制。

### P3：Candidate-conditioned multimodal alignment

只有 P1 证明固定 embedding/WTI 存在系统性候选错配后，才考虑轻量 pair-conditioned residual、candidate-conditioned alignment 或两阶段 reranking。必须同时报告召回收益和候选规模、延迟、显存等代价。

### P4：时序建模与效率支线

TempMe 式 temporal modeling 或 token merging 作为独立支线。它不能与 P2/P3 在同一首轮实验中同时启用；否则无法区分收益来自时间表示、压缩还是不确定性/对齐机制。

## 6. 科研成功标准与停止条件

### 6.1 不再采用的成功标准

- “所有数据集都达到 SOTA”。
- “换更强 backbone 后绝对 Recall 更高”。
- “加入模块后单次 seed 提高 0.1–0.3”。
- “辅助 uncertainty loss 下降，但最终排序不使用 uncertainty”。

### 6.2 新成功标准

一个新方向只有同时满足下列条件，才可进入主线：

1. 在完全匹配的 backbone、split、forward contrastive batch、optimizer steps 和选模指标下优于基线。
2. 训练和推理使用同一最终分数定义。
3. 核心模块有直接消融，且消融只改变一个 causal variable。
4. MSRVTT 筛选阶段使用固定 split seed 42 和 internal val；冻结方法后才允许一次盲测。
5. 通过单 seed 筛选后，基线与新方法都使用固定 split、同一组至少三个训练随机种子报告均值和离散度。
6. 至少在一个独立数据集或独立错误切片上验证方向一致性；不要求每项 Recall 都第一。
7. 若声称 uncertainty，有机制指标证明它能预测检索错误或风险，而不只报告 Recall。

建议的 uncertainty/风险指标包括：

- correct 与 incorrect Top-1 的 uncertainty 分布差异；
- risk-coverage curve 与 AURC；
- uncertainty 与 GT rank、margin、错误事件的相关性或 AUROC；
- 分桶后的置信度—正确率关系；
- T2V/V2T 分别报告，避免方向平均掩盖问题。

任一方向若在受控筛选中没有机制信号，或收益只在换 backbone 后出现，应按预设停止条件终止，不继续 sweep 来追结果。

## 7. SAP 与依赖链清理设计

### 7.1 支持边界

本次“完整清理”指支持入口 `main_task_retrieval.py → modules/modeling_mulit.py` 中的完整依赖链，不只是删除一个类。历史文件 `modules/modeling.py` 不属于 AGENTS 声明的主入口，且其旧概率实现不依赖 SAP，因此本设计不授权顺带删除该文件；如需归档整个历史模型，应另立清理任务。

### 7.2 删除内容

| 区域 | 删除内容 |
|---|---|
| SAP 模块 | `query_models/module_sap.py`、`SemanticAnchorProbing`、SAP 内部 evidential head、anchor decoder 和相关初始化 |
| 主模型结构 | `self.sap`、QC-SAP projections、只服务 SAP 概率链的文本 PIENet/uncertainty/AdaNorm、采样状态和权重 |
| 主模型前向 | SAP spatial-token 构造、anchors、`mu_video/logsigma_video`、epistemic/Dirichlet 输出、probabilistic text、Gaussian sampling |
| 最终打分 | `wti_prob_mu`、`wti_anchor_wti`、`wti_qc_sap`、所有对应 compose/helper/gate diagnostics |
| 旧辅助目标 | 活跃主路径中的 MIL、evidential、negative evidence regularization、orthogonality、UACL intra/KL 及 annealing |
| CLI/脚本 | `final_score_mode` 及三个非 WTI mode、lambda 参数、QC-SAP 温度、旧概率/不确定性/UACL 参数和环境变量 |
| 日志 | prob/SAP/QC/evidential/UACL chain 字段、TSV 列和控制台摘要 |
| 测试 | 只验证上述退役行为的测试；替换为“这些接口不存在”和 WTI-only 数据流测试 |
| 文档 | 删除 `report_SAP.md`、`report.md`、`report_uncertainty.md`；删除 Roadmap、AGENTS 和其他当前文档中的 SAP 结构/实验/消融/未来方案；Roadmap 只留一条“已删除、不得恢复”的墓碑记录 |

`final_score_mode` 不降级为只有一个 `wti` choice，而是从支持 CLI 中删除。最终分数在代码中固定为 WTI logits，避免一个无实际选择空间的兼容参数继续污染实验记录。

### 7.3 保留内容

- OpenAI CLIP/EVA backbone 载入基础设施；EVA 仅保留为受控对照能力，不是当前科研主线。
- WTI token weighting、mask 处理和 deterministic retrieval logits。
- trusted-v1 dataloader、精确 `video_id` positive mask、双向多正例 InfoNCE。
- 训练阶段与 JSFusion 1K test 的严格隔离。
- 与 SAP 无关的 SpatialEnhancer 和 hard-negative 诊断代码；两者继续默认关闭，且不恢复为科研主线。
- `experiment_profile=hygiene` 作为可信协议和配置约束；在旧辅助链删除后，它不再需要通过旁路证明 SAP 未调用。
- `prob_models/` 不在本任务中整目录删除，因为历史 `modules/modeling.py` 仍依赖其中实现；活跃主入口在清理后不得再导入它。归档历史模型及其概率工具需要独立任务。

### 7.4 清理后的数据流

```text
text/video inputs
→ CLIP token features
→ deterministic token processing
→ WTI logits
→ exact-video-id bidirectional multi-positive InfoNCE
→ internal-val retrieval metrics
```

训练返回结构只保留主检索损失真正消费的字段，以及仍独立启用的诊断字段。不得继续返回恒为零的 `MIL_loss`、`evidential_loss` 或 UACL 字段来模拟兼容。

### 7.5 旧参数与 checkpoint 的错误语义

- 旧 SAP/概率 CLI 参数应由 `argparse` 直接判为未知参数，不静默映射到 WTI。
- 含 `sap.`、QC-SAP 或已删除概率头权重的旧 checkpoint 不属于新主线支持格式。
- 若当前 checkpoint loader 使用非严格加载，必须在检测到这些旧键时明确拒绝或报错，不能打印普通 missing/unexpected keys 后继续声称完整恢复。
- 不增加兼容 stub、deprecated alias 或自动权重迁移；需要复现实验时使用对应 Git 历史版本。

## 8. 实验设计原则

### 8.1 Backbone 的角色

固定 backbone 是新方法首轮研究的前提。一个匹配的 EVA 对照仍可用于回答“表示能力上限是否变化”，但必须满足：

- 在 P0 OpenAI hygiene 基线完成之后；
- 使用相同 split、batch 语义、优化步数和 checkpoint selection；
- 与新 uncertainty/alignment 方法实验分开；
- 结果只作为 backbone control，不包装成方法贡献。

因此，“盲目换 backbone 没什么收益”的判断在科研归因层面成立：它可能提高绝对指标，却通常不能回答当前 SAP/uncertainty 为什么失效，也不能构成新的多模态对齐机制。

### 8.2 建议的实验层级

1. **可信基线**：固定 OpenAI CLIP + WTI-only。
2. **机制筛选**：固定训练随机种子，仅判断机制指标和 val 是否同时出现信号。
3. **关键消融**：一次只关闭新方法中的一个必要组件。
4. **稳定性确认**：固定数据 split，至少三个训练随机种子。
5. **泛化检查**：一个独立数据集或预先定义的错误切片。
6. **冻结后盲测**：只在方法、权重和 checkpoint selection 全部冻结后评估 test。

论文表格仍应报告 R@1/R@5/R@10、MdR/MnR，但结论优先依据受控增量、稳定性和机制指标。若某数据集没有领先，应如实分析错误类型和适用边界，不以隐藏结果或继续换 backbone 处理。

## 9. 验证策略

代码清理采用“先写失败测试、再删除实现”的顺序。至少覆盖：

1. 主模型构造不再创建 SAP、PIENet text probability、uncertainty 或 QC-SAP 参数。
2. 训练与评估最终 logits 恒来自 WTI。
3. hygiene/trusted-v1 仍要求精确 `video_id` 并使用多正例 loss。
4. WTI 的 padding mask、T2V/V2T 形状和 DDP gather 语义保持不变。
5. 旧 CLI 参数被拒绝，旧 SAP checkpoint 不被静默接受。
6. 日志和 TSV 不再包含已删除链路字段。
7. 训练阶段仍不能构造或评估 JSFusion 1K test dataloader。

验证命令按项目约束执行：

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests
/home/xujie/miniconda3/envs/ret/bin/ruff check <本次修改的 Python 文件>
```

同时使用限定范围的 `rg` 检查支持代码、脚本和当前长期文档。验收时：

- 活跃代码、脚本、测试、`AGENTS.md`、`docs/README.md` 和分析文档不得出现 SAP/QC-SAP/AnchorWTI 及旧概率参数引用；
- Roadmap 只允许一条 SAP 墓碑记录，不允许保留结构、实验、消融或未来改进段落；
- 本设计规格在实施期间可出现 SAP，实施完成后按项目既有归档惯例从当前工作树删除，以 Git 历史追溯；
- `research_refs/` 不纳入扫描和提交。

不得运行根目录无范围的 `pytest -q`，不得启动长期训练。

## 10. 实施顺序

1. 编写四篇论文综合分析，核对 PDF、补充材料和已有代码。
2. 把仍成立的 SAP-independent 科研原则重新表述到新综合分析和 SSOT，不复制旧实现段落。
3. 删除三个根级历史报告，清理 Roadmap、`AGENTS.md`、`docs/README.md` 和其他当前文档中的 SAP 内容，仅在 Roadmap 留一条墓碑记录。
4. 为清理后的模型接口、CLI、checkpoint 和日志语义添加失败测试。
5. 删除 SAP 及活跃主路径的依赖链，收敛为 WTI 数据流。
6. 删除或改写旧测试、脚本参数和跟踪字段。
7. 运行 targeted tests、完整 `tests/`、ruff 和引用扫描。
8. 归档本设计规格，使当前工作树不再保留 SAP 专属设计文档。
9. 只向用户提供 P0 基线的单行训练命令，由用户手动启动；本任务本身不运行训练。
10. P0 结果写回 SSOT 后，再为新的 pair-level uncertainty 单独进行 brainstorming、设计和实施计划。

## 11. 验收条件

### 文档

- 四篇论文均覆盖创新、实验设计、结果、局限和可研究点。
- 论文数字能追溯到具体表格；代码结论能追溯到具体实现。
- `report_SAP.md`、`report.md`、`report_uncertainty.md` 已从当前工作树删除，Git 历史承担追溯职责。
- Roadmap 仅用一条墓碑记录声明 SAP 已删除且不得恢复；除此之外不再保留 SAP 结构、实验、消融或扩展段落。
- `AGENTS.md`、`docs/README.md` 和长期分析文档不再把 SAP 作为当前组件、历史研究入口或候选方向。
- Roadmap 明确 backbone 只作控制变量、SOTA 不是唯一目标，并给出新阶段、停止条件和评价指标。

### 代码

- 支持入口不再导入、实例化或执行 SAP 及其依赖概率链。
- 非 WTI final-score modes、相关 CLI/脚本/日志接口消失。
- 旧 SAP checkpoint 不会被静默当作兼容 checkpoint。
- trusted-v1、WTI、多正例 InfoNCE 和 test 隔离测试继续通过。
- 完整限定测试与静态检查通过。

### 科研边界

- 本次不宣称新方法增益，因为没有实现或训练新方法。
- 后续论文创新必须在固定可信基线上验证，并同时提供排序结果与机制证据。
- 对没有达到 SOTA 的数据集如实报告，不把换 backbone 当作补救策略。

## 12. 已接受的代价与风险

- **旧 checkpoint 失去主线兼容性**：这是主动选择；Git 历史承担复现职责。
- **短期绝对指标可能不提升**：清理目标是获得可信、可解释的研究起点，而不是制造一次性数字。
- **删除旧实验入口降低消融便利性**：SAP 已完成止损，其未来消融不再具有决策价值，因此不为此保留复杂度。
- **新 uncertainty 方向暂时为空**：这是有意隔离。没有操作性定义、监督和评价闭环前，不提前实现新头部。
- **文档与代码可能短暂不同步**：实施必须在同一计划内完成文档、代码和测试更新，最终提交前以引用扫描消除差异。

## 13. 后续独立设计议题

P0 基线完成后，新的 pair-level uncertainty 规格至少需要重新决策：

- uncertainty 的预测对象是 pair correctness、rank risk、semantic ambiguity 还是 distributional mismatch；
- 单阶段最终打分与两阶段 reranking 的取舍；
- 是否需要候选条件交互，以及候选规模上限；
- bounded score、variance floor/ceiling 和稳定训练约束；
- calibration 与 retrieval metrics 的联合选择标准；
- 与 EagleNet、GARE、TempMe、GraviAlign 的差异化创新主张。

这些问题不在本设计中预设答案，以免把论文启发直接等同于本项目的新方法。
