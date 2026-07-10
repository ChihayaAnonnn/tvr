# 可信实验基座设计规格

**日期**：2026-07-10
**状态**：已确认，待实施
**适用主线**：MSRVTT `train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_mulit.py`

## 1. 背景与决策

当前实验结果不足以继续直接判断 SAP、backbone adapter 或更大模型是否有效，原因不是单一模型结构问题，而是实验协议同时存在三类基础风险：

1. MSRVTT 训练期间直接在 JSFusion 1K test 上逐 epoch 评估并选择 checkpoint，产生测试集选择偏差。
2. `expand_msrvtt_sentences` 将同一视频的多条描述展开为独立样本，但主损失使用对角 CrossEn；同一全局 batch 中重复出现的视频会被错误地互当负例。
3. WTI 对 padding 位置先乘零再取最大值；当所有有效相似度为负时，padding 的零值会成为错误最大值。与此同时，hygiene WTI-only 配置仍计算 SAP、概率分支和视觉 patch hidden states，造成显存与归因污染。

因此，下一步不应先扩大模型设计，而应先重建唯一、强约束、可追溯的 `trusted-v1` 实验协议，再重跑 hygiene baseline。历史结果继续归档，但旧协议结果不得与新协议结果直接等价比较。

本设计不保留 legacy 训练或评估模式。旧行为仅存在于 Git 历史中，不再通过 CLI 开关兼容。

## 2. 目标与非目标

### 2.1 目标

- 建立固定、可版本化的 MSRVTT 8500/500 train/val 拆分，并将 JSFusion 1K 作为独立盲测集。
- 使用精确 `video_id` 定义双向多正例 InfoNCE，消除同视频描述之间的错误负例。
- 修复 WTI padding 最大池化错误。
- 使 hygiene 配置真正只执行 WTI 所需路径。
- 完整记录数据、代码、backbone、batch 和损失口径，使实验可审计、可复现。
- 在不启动长期训练的前提下，用单元测试和小型 smoke test验证协议与代码路径。

### 2.2 非目标

- 不引入语义相似度软正例、伪标签或新的 hard-negative 机制。
- 不重构为新的 Trainer 框架，不改写与本问题无关的 SAP 内部结构。
- 不改变 OpenAI CLIP 的默认 backbone 地位。
- 不代替用户启动长期 GPU 训练。
- 不为历史的 test-per-epoch 或 diagonal CE 行为保留兼容开关。

## 3. 方案选择

评估过三种范围：

1. **最小修补**：只隔离 test、增加多正例损失、修复 WTI mask。改动小，但不能消除 hygiene 无效计算与日志口径混乱。
2. **完整可信基座**：同时修复数据协议、损失、WTI、hygiene 执行路径、元数据和测试。该方案覆盖当前科研判断的主要混杂因素，且不要求大规模框架重构。
3. **训练框架重构**：新增独立 Trainer/Protocol 层。长期边界更整洁，但当前实施面和回归风险过大，会推迟 backbone 主线。

采用方案 2。

## 4. 唯一可信数据与评估协议

### 4.1 固定拆分

新增版本化 split 生成器，以官方 9000 个训练视频及其 JSON 描述为唯一输入：

1. 对唯一 `video_id` 按字符串排序。
2. 使用独立的 `random.Random(42)` 打乱，不依赖全局随机状态。
3. 前 500 个视频组成内部 val，其余 8500 个组成 train。
4. val 为每个视频保留官方 JSON 中全部 20 条描述，并保持 manifest 视频顺序和 JSON 内描述顺序。
5. JSFusion 1K test 不参与拆分、调参或 checkpoint 选择。

版本库提交 `dataloaders/splits/msrvtt_trusted_v1_seed42.json`。manifest 至少记录：

- `protocol_version=trusted-v1`；
- seed 和确定性算法说明；
- 源 train CSV、描述 JSON、JSFusion test CSV 的 SHA-256；
- 8500 个 train ID、500 个 val ID；
- test ID 摘要；
- train/val/test 数量及两两交集检查结果。

生成器在本地派生：

- train CSV：每个 train 视频一行；
- val CSV：每条描述一行，即 500 个视频、10000 条描述；
- split 校验摘要。

派生文件默认写入 `data/generated/msrvtt_trusted_v1/` 并加入 `.gitignore`，不作为事实来源提交。manifest 才是版本化事实来源。

### 4.2 dataloader 契约

MSRVTT trusted-v1 训练强制展开 8500 个视频的全部 20 条官方描述，即每个 epoch 的训练数据集包含 170000 条文本—视频样本。每条样本额外返回由 manifest 顺序映射得到的稳定整数 `video_group_id`。禁止使用 Python 进程随机 hash 生成该 ID，也不保留单描述随机采样作为同协议下的另一种训练行为。

内部 val dataloader 读取 10000 行描述，并显式暴露现有评估代码需要的：

- `multi_sentence_per_video=True`；
- `cut_off_points`；
- `sentence_num=10000`；
- `video_num=500`。

val CSV 必须按视频连续分组；loader 在初始化时校验每个视频恰有 20 条描述。评估时文本保留全部 10000 条，视频特征每个视频只编码和计分一次。

### 4.3 CLI 与执行边界

- `--eval_split` 仅允许 `val` 或 `test`，默认 `val`。
- `--val_csv` 明确指向内部 500-video val；新增或明确独立的 test CSV 参数供 JSFusion 1K 使用。
- MSRVTT 训练要求描述展开；缺少 `--expand_msrvtt_sentences` 时直接报错。
- `--do_train` 时只构造 train 和 val dataloader，并强制按 val T2V R@1 选择 checkpoint。
- JSFusion 1K 只允许通过 `--do_eval --eval_split test --init_model ...` 独立运行。
- `--do_train` 与 `--eval_split test` 同时出现时立即报错。
- 不再把 MSRVTT val alias 到 JSFusion test，也不保留 test-per-epoch 的兼容分支。

训练启动前必须验证 train、val、test ID 两两无交集。manifest 缺失、哈希不匹配、数量不符、ID 重复或描述数不符均直接终止。

## 5. 双向多正例主损失

### 5.1 正例定义

正例仅由精确相同的官方 `video_id` 定义。不给语义近邻、同主题视频或属性相似样本赋予软正例权重。

模型在分布式训练中将检索特征与 `video_group_id` 一并 all-gather。特征继续使用现有可回传梯度的聚合路径；group ID 使用无梯度聚合。全局正例掩码为：

\[
P_{ij}=\mathbb{1}[g_i=g_j]
\]

对相似度矩阵 \(S\)，每个 query 的损失为：

\[
L_i=-\log\frac{\sum_{j:P_{ij}=1}\exp(S_{ij})}
{\sum_j\exp(S_{ij})}
\]

实现使用 `logsumexp` 保证数值稳定。T2V 使用正例掩码 \(P\)，V2T 使用 \(P^T\)，最终主损失取两个方向均值。

### 5.2 强制语义

- 多正例损失是唯一主检索损失，不提供 diagonal CE CLI 开关。
- 当全局 forward batch 中所有 `video_group_id` 唯一时，损失必须数值等价于现有对角 CrossEn。
- 每个 query 必须至少有一个正例，否则抛出包含 query 位置和 group ID 的错误。
- 日志记录全局 batch 的唯一视频数、重复视频数和平均正例数，用于判断描述碰撞分布。
- gradient accumulation 不跨 forward 合并对比矩阵；日志必须把单次 forward 的全局对比 batch 与优化器有效 batch 分开表述。

## 6. WTI 正确性修复

WTI token-frame 相似度在执行最大池化前应用布尔 mask：无效文本 token 和无效视频帧置为当前 dtype 可表示的极小值，而不是乘零。池化后的求和或平均只使用有效位置，并以实际有效数量归一化。

若任一样本没有有效文本 token 或有效视频帧，立即抛出带 batch 位置的错误，不用 clamp 或空张量默认值掩盖数据问题。

核心回归场景为：全部有效相似度均为负、padding 相似度原始值为零。修复后最大值必须来自有效位置。

## 7. 真正的 hygiene WTI-only 执行路径

模型在进入编码和相似度计算前，根据 profile 与实际启用的损失生成内部特征需求。hygiene 配置要求所有概率辅助权重为零，并只申请 WTI 必需特征。

hygiene 下：

- 视觉编码器返回逐帧检索特征，不请求 patch hidden states；
- 文本编码器只返回 WTI 所需 token/global 特征；
- 不调用 SpatialEnhancer、SAP、视频概率采样、文本 PIENet、不确定性头；
- 不创建 MIL、evidential、negative regularization、orthogonality 等辅助中间量；
- 最终排序仍为 `weighted_logits = wti_logits`。

default profile 保留现有 SAP 研究路径，但其主检索损失和数据协议同样切换到 `trusted-v1`。如果 hygiene 启用了任何概率辅助权重，启动时直接报错，而不是静默覆盖。

当前 backbone adapter 工作树建立在该特征契约之上。OpenAI CLIP 继续作为默认 backbone；adapter 必须明确声明输出维度以及是否支持 token/patch hidden states，契约不符时在模型初始化阶段失败。

## 8. 实验可追溯信息

每次训练由 rank 0 在输出目录原子写入 `experiment_manifest.json`。checkpoint 权重格式保持不变，元数据作为 sidecar，避免破坏已有加载方式。

manifest 与日志首部至少包含：

- `protocol_version=trusted-v1`；
- Git commit、工作树是否 dirty，以及已修改路径列表；
- train/val/test 路径、split seed、版本化 manifest SHA-256、视频与描述数量；
- backbone 类型、权重来源或标识、adapter 配置；
- experiment profile、最终排序方式、启用损失及其权重；
- world size、每卡 micro-batch、单次 forward 全局对比 batch、gradient accumulation、有效优化 batch；
- 每 epoch forward 数、optimizer steps 和预计总 optimizer steps；
- 随机种子和关键数据采样配置。

日志中不再用单一 `batch_size` 混称上述四种 batch 口径。

## 9. 错误处理

以下情况必须失败，不允许警告后回退：

- 训练尝试读取 test split；
- train/val/test ID 有交集；
- split manifest 缺失、重复、数量或源文件哈希不符；
- val 视频不是恰好 20 条官方描述；
- 多正例掩码存在无正例 query；
- WTI 样本没有有效 token 或帧；
- hygiene 实际启用概率辅助损失；
- backbone/adapter 输出维度或隐藏特征能力与调用方契约不符。

错误信息必须包含冲突参数、路径或样本位置，使问题可直接定位。

## 10. 测试策略

实施遵循 TDD，先增加失败测试，再做最小实现。测试至少覆盖：

### 10.1 数据协议

- seed 42 拆分确定且重复生成字节一致；
- train/val 为 8500/500 个视频且两两无交集；
- val 恰有 10000 条描述，每视频 20 条；
- train/val/test 交叉时生成器或启动校验失败；
- multi-sentence 元数据与 CSV 行顺序一致。

### 10.2 多正例损失

- group ID 唯一时等价于 diagonal CE；
- 重复视频样本在 T2V、V2T 两个方向均为正例；
- 无正例 query 明确失败；
- 模拟跨 rank 聚合后特征与 group ID 对齐；
- 极端 logits 下结果有限且梯度有限。

### 10.3 WTI 与 hygiene

- 有效相似度全为负时 padding 不会成为最大值；
- 文本和视频 padding 均正确排除；
- 全无效 token/帧立即失败；
- hygiene 下通过 mock/spy 证明 SAP、SpatialEnhancer、PIENet 和不确定性模块未被调用；
- default profile 仍能进入 SAP 路径。

### 10.4 协议入口与 adapter

- `--do_train --eval_split test` 被拒绝；
- 训练构造 val 而不构造 test，并按 val T2V R@1 选 checkpoint；
- 显式 test eval 能读取指定 checkpoint；
- OpenAI CLIP 默认路径及现有 backbone adapter 契约测试通过；
- 用小型合成输入完成 CPU smoke test，不下载权重、不启动长期训练。

## 11. 文档与提交边界

- `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 继续作为科研问题、决策门槛和后续实验路线的唯一主文档。
- `docs/project/STATUS.md` 只保留当前状态摘要和主文档链接。
- 不扩张 `docs/project/plan.md`；重复内容应删除或并入 roadmap。
- 开题报告大纲保持删除，不恢复相关索引。
- 在用户已确认的当前 dirty adapter 工作树上实施，保留并识别所有既有修改。
- adapter 与可信实验基座按职责分开暂存和提交；不得把无关工作树修改夹带进任一提交。
- 本设计规格单独提交。

## 12. 验收标准

实现完成需同时满足：

1. MSRVTT 训练无法在 JSFusion test 上逐 epoch 选择 checkpoint。
2. 固定 8500/500 manifest 可重复验证，内部 val 使用全部 20 条描述。
3. 训练主损失只按精确 `video_id` 构造双向多正例。
4. WTI padding 回归测试通过。
5. hygiene WTI-only 测试证明所有非必要概率分支未执行。
6. 日志和 `experiment_manifest.json` 能区分并追溯数据、代码、backbone 与 batch 口径。
7. 相关单元测试、CPU smoke test、格式检查和 `git diff --check` 通过。
8. `RESEARCH_ISSUES_AND_ROADMAP.md` 明确：先重跑新协议 hygiene baseline，再决策 backbone adapter 或更大模型；历史旧协议结果只作档案参照。
9. 不由自动化流程启动长期训练；最终向用户提供单行训练与显式 test 评估命令。
