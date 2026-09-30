# Retrieval Rebuild B: Semantic Retrieval System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 A 验收通过后构建可训练、可检索、可评测的多正例与候选存在判断系统，交付合成 CPU 闭环及有条件的语义分组/对齐扩展。

**Architecture:** 单一 `SemanticRetrievalModel` 组合 CLIP、时序表示、粗检索投影、PairMatcher 和存在头。监督与推理输入分离，canonical 去重先于 top-K；推理使用冻结校准/阈值决定返回集合、扩检、澄清或弃答。训练、评估和索引产物接入 A 的归档合同。

**Tech Stack:** A 的 PyTorch / NumPy / OpenCV 基础组件，标准 AdamW、DDP；合成 CPU fixture 优先。首版精确分块检索，不引入 ANN、flow 或 diffusion 依赖。

**Spec:** [已获批设计](../specs/2026-10-01-uatvr-audit-cleanup-and-semantic-retrieval-design.md)。**Prerequisite:** [计划 A](2026-10-01-retrieval-rebuild-a-foundation.md)验收报告，不能用旧 UATVR 代替。

## Global Constraints

- A 的全部保护规则继续生效；新包无旧模型/入口/开关，历史数据面与冻结控制面只读。
- “标签字段只进入监督/评测，不进入推理特征。”unknown 的张量值为 −1，不能当负例。
- “canonical 去重应在粗检索索引及 top-K 截断前完成”；语义相近不等于答案等价。
- “候选内容必须用当前编码器重新计算”；最终索引/缓存绑定 model SHA、库版本、内容身份与预处理版本。
- 所有选择、校准、K、分组参数仅使用 validation；test sealed。不能通过删 GT 伪造库无答案标签。
- 本计划的所有数值只用于 CPU fixture/算子验证；没有冻结科学配置、方法成功数值 gate、样本量或新实验 phase。真实科学实验需独立 study + Stage 0 + 注册流程，后续另行制定。
- OT 辅助项和 learned grouping 默认关闭；必须有明确标注/验证才启用。无冲突信号时不自动澄清。

## Review Focus

1. 两个 alias 给出相反标签或不同视频内容：拒绝构建 episode，不能静默择一；B1。
2. 无正例/全 unknown 的 batch：跳过不可定义目标，有限 loss 与明确计数，不能 NaN 或伪造正例；B4。
3. 索引和查询模型/预处理版本失配：检索前拒绝，不能复用旧特征；B5/B8。
4. 概率低且预算耗尽：有限次扩检后弃答，不输出“全库不存在”；B6。
5. 有梯度但 optimizer 漏参数或输出不参与排序：用实际 step 与推理分数改变验证，不能只检查 grad 非空；B3/B7。

## 文件结构与接口复用

使用 A 定义的 `QueryInputs/QueryFeatures`、`VideoFeatures`、`EncoderInputs/EncodedFeatures`、`PairScores`、`Targets`、`RuntimeConfig`、`RunContext`、`MaskedWTI` 和归档 API，不重新定义同名合同。所有新模块位于 `src/semantic_retrieval/`，测试位于 `tests/semantic_retrieval/`。

新增 `config.py` 中的 `ModelConfig/TrainConfig/PolicyConfig`，通过 `SystemConfig(runtime, model, train, policy)` 装配；拒绝未知键。ModelConfig 包含 `clip: ClipConfig`、`clip_weights/clip_sha256/dim/num_layers/num_heads/max_frames/coarse_dim`；TrainConfig 包含 `learning_rate/weight_decay/seed/scheduler_name/scheduler_steps/trainable_prefixes/loss_weights`；PolicyConfig 包含 `pair_threshold/presence_threshold/k_sequence/max_budget`。容量、学习率、预算等生产配置必填，不由旧 argparse 默认值补齐。测试 D=8、最多 4 帧、2 query，不是实验推荐参数。

验证沿用 A 的 `$SR_PY` / `PYTHONPATH=src:.` / CPU 环境，pytest 禁用 cacheprovider。训练/eval 合成 fixture 写入 tmp_path 的临时仓库，不能写真实 `logs/`、`ckpts/` 或 registry。
断言示例 fixture 在各测试文件按任务合同构造；不是另一个生产模型或新运行接口。

## Task B1: Episode loader 与三态监督

**Files:** Create `data/{episodes,collate}.py`、`tests/semantic_retrieval/test_episodes.py`；Modify `types.py`。

**Interfaces:**
- `Episode` 字段按 spec §6；`read_episodes(path: Path, *, gallery_version: str, split: str) -> list[Episode]`，校验 annotation provenance。
- `collate_episodes(episodes: Sequence[Episode]) -> EpisodeBatch`，字段 `inputs/targets/query_ids/canonical_ids/pair_indices/query_offsets`；`pair_indices` `[N,2]` 映射每个 query 的真实 C，`query_offsets` `[Q+1]` 对 packed pair 分段。候选视频可全局去重，Q 与 V 不要求相等。`inference_queries() -> QueryInputs` 必须剥离监督；跨 rank 的 ID 编码使用同一已记录词典。
- `audit_episode_splits(episodes: Sequence[Episode]) -> None`：canonical/same-source/video-caption group 跨 split 泄漏拒绝；库不存在声明必须提供该库审核 provenance，否则降为 unknown 或明确失败，不自动标 no。

- [ ] 写失败测试：单/多正例、粗检索 miss、已审核库无答案、unknown、空候选、alias 相反标签/内容拒绝；删 GT 保持 gallery existence unknown；split 泄漏拒绝。
  ```python
  def test_episode_inference_has_no_supervision(episode_batch):
      query = episode_batch.inference_queries()
      assert not hasattr(query, "targets")
      assert not hasattr(query, "relevance")
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_episodes.py`；Expected loader 不存在失败。
- [ ] 实现 loader/collate，canonical 稳定 ID 排序；unknown 与 known alias 合并仅使用可靠已知标注，positive/negative 矛盾拒绝。candidate presence 标签必须与可靠 pair 标签一致。
- [ ] 同命令通过；测试 `inference_queries()` 无标签属性，模型无需 Episode 真值构造输入；空候选不伪造 padding 视频的有效帧。
- [ ] 提交数据合同与测试；commit `feat: add audited retrieval episodes and unknown-aware supervision`。

## Task B2: 新时序表示与模型组合

**Files:** Create `models/{temporal,coarse,model}.py`、`tests/semantic_retrieval/test_model.py`；Modify `config.py`。

**Interfaces:**
- `TemporalEncoder(dim: int, num_layers: int, num_heads: int, max_frames: int).forward(frames: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]`；按有效位置输出逐帧与 masked mean pooled 表示。
- `CoarseProjector(dim: int, output_dim: int)` 的 `project_text(pooled: Tensor) -> Tensor`、`project_video(pooled: Tensor) -> Tensor` 做独立投影/L2；`forward(text_pool, video_pool) -> tuple[Tensor, Tensor]` 组合这两方法；zero norm 拒绝而非静默 NaN。
- `SemanticRetrievalModel.encode(inputs: EncoderInputs) -> EncodedFeatures`；`encode_query(inputs: QueryInputs) -> QueryFeatures`；`encode_gallery(video: Tensor, frame_mask: Tensor) -> VideoFeatures`；`project_query(features: QueryFeatures) -> Tensor`、`project_gallery(features: VideoFeatures) -> Tensor`；`coarse_embeddings(features: EncodedFeatures) -> tuple[Tensor, Tensor]` 组合两侧投影。在线查询不需要伪视频；V=0 集合不调用视频 encoder。

- [ ] 写失败测试：A 小 CLIP 与标准 Transformer 组合 shape/finite；padding 内容不改变有效输出；全空、frame count 超 max_frames、zero norm 拒绝；时序关闭的基础配置可运行。
  ```python
  def test_query_and_candidate_counts_are_independent(model, inputs_q2_v3):
      features = model.encode(inputs_q2_v3)
      assert features.pooled_text.shape == (2, 8)
      assert features.pooled_video.shape == (3, 8)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_model.py`；Expected 缺组合类失败。
- [ ] 实现新模型组件，不继承旧类；标准 Transformer 使用 padding mask、显式位置参数和配置。训练/推理具名方法返回类型不随 root training flag 改变。
- [ ] 同命令通过；trace 新包 import 无旧模型依赖，coarse embedding 单位范数。
- [ ] 提交组合模型与配置；commit `feat: compose a new retrieval encoder and coarse representation`。

## Task B3: 共用排序表征与候选存在头

**Files:** Create `models/{pair_matcher,answerability}.py`、`tests/semantic_retrieval/test_matching_gradients.py`；Modify `models/model.py`。

**Interfaces:**
- `PairMatcher(dim: int).forward(features: EncodedFeatures, pair_indices: Tensor, *, return_alignment: bool = False) -> PairScores`，`pair_indices` 为 `[N,2]` query/video indices；仅计算被请求 pair。对齐监督启用时输出 attention 与共享投影后的 token/frame cosine，其他时刻不保留辅助输出。
- 使用投影后的 query/frame 构造带 dustbin 的 token→frame attention；`h` 聚合 query/video pooled、逐元素交互、对齐后的 pooled 表示、dustbin mass 与 WTI。`logit=softplus(a_raw)*WTI+b+MLP(h)`。
- `AnswerabilityHead(feature_dim: int).forward(pair_scores: PairScores, query_offsets: Tensor) -> Tensor`：canonical 候选有序聚合后 mean/max，输出 `[Q]` presence logits；空集合语义明确为不存在，不在 pooling 上产生 NaN。
- `model.score_pairs(features, pair_indices, *, return_alignment: bool = False) -> PairScores`；`model.score_presence(pair_scores, query_offsets) -> Tensor`。

- [ ] 写失败测试：pair 分数与输入候选顺序无关；alias 去重后存在预测稳定；单独 pair/existence loss 的梯度进入共享投影；optimizer step 后共享参数及实际推理 match logit 改变。
  ```python
  def test_presence_update_changes_actual_ranking_logits(model, features, pair_indices, offsets, optimizer):
      before = model.score_pairs(features, pair_indices).logits.detach().clone()
      model.score_presence(model.score_pairs(features, pair_indices), offsets).sum().backward()
      optimizer.step()
      after = model.score_pairs(features, pair_indices).logits.detach()
      assert not torch.equal(before, after)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_matching_gradients.py`；Expected 新头缺失失败。
- [ ] 实现并接线，同一 match logit 用于训练和排序；去掉所有无消费诊断/采样。存在头不消费 SemanticSet 的预测组。
- [ ] 同命令通过，测试冻结 CLIP 时其 grad=None、解冻指定层后 grad 非零；默认 dropout=0 的 fixture 保证变化来自 step。
- [ ] 提交头与接线；commit `feat: jointly connect matching and candidate answerability representations`。

## Task B4: 显式多正例与存在 objective

**Files:** Create `objectives/{__init__,retrieval}.py`、`tests/semantic_retrieval/test_objectives.py`。

**Interfaces:**
- `compute_objective(coarse_logits: Tensor, pair_scores: PairScores, presence_logits: Tensor, targets: Targets, pair_indices: Tensor, query_offsets: Tensor, *, weights: Mapping[str,float], coarse_relevance: Tensor | None = None, auxiliary_losses: Mapping[str,Tensor] | None = None) -> dict[str,Tensor]`，keys `total/coarse/match/pair/exist/align/group` 与 `valid_coarse_count/valid_match_count/valid_pair_count/valid_exist_count`。pair 标签由 `targets.relevance[pair_indices[:,0], pair_indices[:,1]]` 提取；coarse_relevance 默认用本地 targets，跨 rank 时显式传全局已审核关系。auxiliary_losses 仅接受 `align/group`，未启用返回 finite zero。
- 粗检索与匹配用已知正例均匀 soft target，分母只含已知正/负；无已知正或无负行跳过对比项。pair/exist 只在各自标签非 −1 上做 BCE；全 unknown 总项为与输入图相连的 finite zero，Trainer 不执行空监督 step。

- [ ] 写失败测试：多正例均有梯度；unknown logit 任意改变不影响监督 loss；无答案只负 pair/exist BCE；全 unknown 得 zero/计数0；一 query 及不等候选数正确；手算损失一致。
  ```python
  def test_all_unknown_loss_is_finite_zero(all_unknown_objective_inputs):
      terms = compute_objective(**all_unknown_objective_inputs)
      assert terms["total"].item() == 0.
      assert terms["valid_pair_count"].item() == 0
      assert torch.isfinite(terms["total"])
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_objectives.py`；Expected objective 不存在失败。
- [ ] 实现显式各项与计数，无对角 GT 假设，无旧 frozen-sidecar objective；单 rank 和需要 gather 的粗检索 mask 分开构建。
- [ ] 同命令通过；B3 的共享表征梯度测试同时通过。
- [ ] 提交目标及测试；commit `feat: implement multi-positive and unknown-aware presence objectives`。

## Task B5: canonical 索引与版本化特征缓存

**Files:** Create `retrieval/{__init__,index,cache}.py`、`tests/semantic_retrieval/test_index.py`。

**Interfaces:**
- `IndexIdentity(model_sha256: str, gallery_version: str, preprocessing_sha256: str)`；每个 cache 另含内容 hash。
- `CanonicalIndex(ids: Sequence[str], embeddings: Tensor, identity: IndexIdentity)`；`search(query: Tensor, k: int, *, expected_identity: IndexIdentity) -> list[list[str]]`，exact 分块 dot product，分数并列 ID 升序。
- `FeatureCache.read(canonical_id: str, *, expected_identity: IndexIdentity, content_sha256: str) -> VideoFeatures`；帧表示、pooled 表示与 mask 一并缓存，`write` 只进 run-local array store，读取校验每个数组 hash。

- [ ] 写失败测试：alias 复制不能挤占 top-K；k 大于库大小返回全部，k<=0 拒绝；空库返回空集合；chunk 一致；错 model/gallery/preprocess/content/array hash 拒绝。
  ```python
  def test_index_rejects_wrong_model(index, query, wrong_model_identity):
      with pytest.raises(ValueError):
          index.search(query, 2, expected_identity=wrong_model_identity)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_index.py`；Expected 新索引/缓存缺失失败。
- [ ] 实现去重前建索引，aliases 单独保存；冻结模型才构建可部署索引，feature schema 与身份一起校验。不实现 ANN 或适应性训练挖掘。
- [ ] 同命令通过；确保读缓存不实例化旧模型，也不在训练解冻路径中复用旧表征。
- [ ] 提交检索/缓存；commit `feat: build canonical retrieval index with strict feature identity`。

## Task B6: canonical 集合与动作 policy

**Files:** Create `retrieval/{semantic_set,policy}.py`、`types.py` 中 Decision、`tests/semantic_retrieval/test_policy.py`。

**Interfaces:**
- `SemanticSet.build(candidate_ids: Sequence[str], pair_scores: PairScores, *, group_predictor=None) -> SemanticSetOutput`；字段 `canonical_ids/groups/supports/entropy/conflict_signal`；无 predictor 仅 canonical 分组。首版 `conflict_signal=False`，不接受在线请求中的任意 GT/冲突标志；policy 测试可用合成信号验证未来有条件路径，不表示已实现冲突 classifier。
- `Decision(action: str, answer_ids: tuple[str,...], reason: str, next_k: int | None)`，action 只 `return/expand/clarify/abstain`。
- `RetrievalPolicy.decide(pair_probabilities: Tensor, presence_probability: float, semantic_set: SemanticSetOutput, *, attempted_k: Sequence[int], remaining_budget: int) -> Decision`；构造时传入冻结 `PolicyConfig`。

- [ ] 写失败测试：无支持且有预算 expand；无预算 abstain；两个同时成立的答案 return 集合；无冲突信号不 clarify；已验证冲突 clarify；NaN/非法概率抛错；空集合和循环有限，reason 不声明全库不存在。
  ```python
  def test_budget_exhaustion_abstains_without_global_absence_claim(policy, unsupported_set):
      decision = policy.decide(torch.tensor([0.1]), 0.1, unsupported_set, attempted_k=[2, 4], remaining_budget=0)
      assert decision.action == "abstain"
      assert decision.next_k is None
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_policy.py`；Expected policy 不存在失败。
- [ ] 实现 spec 优先级，严格递增 K 序列/已尝试集合及预算扣减；group entropy 仅为描述性值。默认没有语义冲突 classifier，clarify 路径只接受来自已配置且验证的信号。
- [ ] 同命令通过；置换/alias 后输出 canonical 集合一致，存在头保持独立。
- [ ] 提交决策与测试；commit `feat: add bounded expansion and semantic retrieval decision policy`。

## Task B7: Trainer 与新 checkpoint

**Files:** Create `runtime/{trainer,checkpoint}.py`、`scripts/train_semantic_retrieval.py`、`tests/semantic_retrieval/test_trainer.py`；Modify `config.py`。

**Interfaces:**
- `Trainer(model, optimizer, scheduler, context: RunContext).train_step(batch: EpisodeBatch) -> dict[str,float | int]`；空监督不 optimizer/scheduler step，明确计数。
- `save_checkpoint(context: RunContext, model, optimizer, scheduler, config: SystemConfig, step: int) -> Path`；`load_checkpoint(path: Path, model, *, expected_config_sha256: str) -> TrainingState`，恢复模型参数并返回 `TrainingState(step: int, optimizer_state: Mapping, scheduler_state: Mapping, rng_state: Mapping)`，由 Trainer 恢复其余状态。新 schema、精确键/shape、全部状态与哈希校验。
- CLI 仅 `--config` 与明确的 Scheme A 身份参数，拒绝旧开关/任意 output-dir 覆盖；完整新模型配置禁止未知键。

- [ ] 写失败测试：CPU fixture 两步训练更新排序和存在共享参数；冻结参数无 step；resume 新 checkpoint 的下一步一致；config/state/key/hash 失配失败；训练返回 type 不随 eval 模式变化。
  ```python
  def test_unknown_batch_skips_optimizer(trainer, unknown_batch):
      before = [p.detach().clone() for group in trainer.optimizer.param_groups for p in group["params"]]
      trainer.train_step(unknown_batch)
      after = [p for group in trainer.optimizer.param_groups for p in group["params"]]
      assert all(torch.equal(a, b) for a, b in zip(before, after))
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_trainer.py`；Expected Trainer 缺失失败。
- [ ] 标准 AdamW，scheduler 策略与参数在 config 必填；显式优化器可训练集合，检查每个要求更新的参数确实在 optimizer。fp32 前向/排序验证，不启用旧 autocast 默认值。运行生命周期调用 A 桥接，信号失败保留证据。
- [ ] 同命令通过；两 rank CPU/Gloo fixture 检查粗检索梯度与本地 episode loss 归一化。全局 coarse relevance 只由已审核 `(query_id, canonical_id, label)` 关系构建，未审核的跨 rank pair 保持 unknown，不能假设 off-diagonal 为负。逐 rank 信息一致；不以此代替四卡 GPU 资格。
- [ ] 提交 Trainer、薄入口及测试；commit `feat: train new retrieval model with explicit state and archive lifecycle`。

## Task B8: 推理评估、validation 校准与闭环

**Files:** Create `evaluation/{calibration,episodes}.py`、`runtime/inference.py`、`scripts/evaluate_semantic_retrieval.py`、`tests/semantic_retrieval/test_end_to_end.py`。

**Interfaces:**
- `CalibrationIdentity(model_sha256: str, preprocessing_sha256: str, gallery_version: str, candidate_policy_sha256: str, calibration_split_sha256: str)`；`CalibrationState(identity: CalibrationIdentity, pair_temperature: float, pair_bias: float, presence_temperature: float, presence_bias: float)`。
- `fit_calibration(pair_logits, pair_labels, presence_logits, presence_labels, *, split: str, identity: CalibrationIdentity) -> CalibrationState`；positive temperature + bias 的 sigmoid 映射，unknown 排除，单类则拒绝拟合；仅 `val_calibration`。policy 阈值不由这一步隐式选择。
- `evaluate_episodes(predictions, targets, *, split: str) -> dict[str,object]`：按 episode 类型给 retrieval/set quality、误接受/弃答、扩检找回率、预算与计数；单类/缺完整标签指标明确不可定义。
- `RetrievalService.retrieve(inputs: QueryInputs, *, budget: int) -> Decision`：encode_query→search→读取版本匹配 VideoFeatures→组装无监督 EncodedFeatures→score_pairs/presence→calibration→SemanticSet→policy，循环只重算新增 pair 与整个存在集合。首版一次在线请求 Q=1，多 query 用逐请求服务，不跨查询合并存在集合。

- [ ] 写失败测试：临时库内单/多正例、miss 扩检、已审核 no-answer、unknown，全链路无 GT 输入；剥离/改变标签不影响同一推理输出；alias/置换稳定、版本失配拒绝、校准 split 错误拒绝、test sealed 拒绝。
  ```python
  def test_calibration_refuses_test_labels(calibration_arrays, calibration_identity):
      with pytest.raises(ValueError):
          fit_calibration(*calibration_arrays, split="test", identity=calibration_identity)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_end_to_end.py`；Expected 服务缺失失败。
- [ ] 实现只消费冻结模型/索引/校准/PolicyConfig；校准模型 SHA 与候选生成配置绑定，应用时验证。eval 输出写 run-local array store + 小 JSON；在统计中给各事件的分子/分母，不能用全弃答掩盖 coverage。
- [ ] 同命令通过；CPU 临时仓库闭环可从新 checkpoint 恢复，训练/评测薄入口 `--help` 正常，真实 test 未访问。
- [ ] 提交服务、评测及测试；commit `feat: evaluate calibrated retrieval decisions end to end`。

## Task B9: 有条件的语义分组扩展

**Files:** Create `models/grouping.py`、`tests/semantic_retrieval/test_grouping.py`；Modify `retrieval/semantic_set.py`、`runtime/trainer.py`。

**Prerequisite:** 存在 query-conditioned 答案等价 pair 标签和 provenance；没有标注时不启用生产路径，基础系统可独立验收。

**Interfaces:** `EquivalencePredictor.forward(pair_features_a: Tensor, pair_features_b: Tensor) -> Tensor`；对称输入交互确保交换顺序结果一致；`group_loss(logits, labels) -> Tensor` 排除 −1；`conservative_groups(ids, probabilities, threshold: float) -> tuple[tuple[str,...],...]` 使用 canonical 顺序 complete-link。

- [ ] 写失败测试：同查询两等价/不等价候选正确消费 labels；unknown 不监督；非传递概率链不能把未达阈值端点并成同组；置换稳定；阈值 fixture=0.8 不代表 validation 最终值。
  ```python
  def test_complete_link_does_not_merge_a_nontransitive_chain():
      p = np.array([[1., .9, .1], [.9, 1., .9], [.1, .9, 1.]])
      assert conservative_groups(["a", "b", "c"], p, .8) == (("a", "b"), ("c",))
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_grouping.py`；Expected 新模块不存在失败。
- [ ] 实现 predictor、loss 与保守 grouping，监督由 `Targets.group_pairs/group_labels` 提取，训练 loss 经 `auxiliary_losses['group']` 接入 B4；不推导等价即冲突，不输入 group GT 做在线预测、不把全局视频聚类当语义等价。
- [ ] 同命令通过；B6/B8 覆盖存在判断与此组预测独立，无码/未审核标签时只能 canonical grouping。
- [ ] 提交扩展；commit `feat: add supervised query-conditioned equivalence grouping`。生产启用另需 validation 选择及对应 study 记录。

## Task B10: 有条件的局部 OT 辅助监督

**Files:** Create `objectives/alignment.py`、`tests/semantic_retrieval/test_alignment.py`；Modify `runtime/trainer.py`、`config.py`。

**Interfaces:** `dustbin_ot_target(similarity: Tensor, text_mask: Tensor, frame_mask: Tensor, *, config: AlignmentConfig) -> Tensor`，返回 detached 逐 token 分布；`alignment_loss(attention: Tensor, target: Tensor, text_mask: Tensor) -> Tensor` 为有效 token 的 KL/cross entropy，明确 target 0 项处理。

算法决定：采用带双侧 dustbin 的平衡增广 transport。n 个有效 token、m 个有效 frame，真实行/列单位质量，额外行质量 m、额外列质量 n，均除以 n+m；真实 cost=`1-cos`，真实↔dustbin cost=c，bin↔bin cost=0。log-domain Sinkhorn 满足上述边际，再将真实 token 行归一化为 m+1 attention target。teacher 全部 detach，padding 排除；不收敛则明确失败，不悄悄使用最后一次迭代。

- [ ] 写失败测试：fixture ε=0.1、c=0.5、max_iter=500、tol=1e-5，边际在容差内；目标行和=1、非负/finite、padding 不影响、teacher 无梯度、student/shared projection 有梯度、低匹配更多 dustbin；无效/不收敛输入拒绝。
  ```python
  def test_ot_teacher_is_detached_and_row_normalized(similarity, text_mask, frame_mask, alignment_config):
      target = dustbin_ot_target(similarity, text_mask, frame_mask, config=alignment_config)
      assert not target.requires_grad
      torch.testing.assert_close(target.sum(-1)[text_mask], torch.ones_like(target.sum(-1)[text_mask]))
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_alignment.py`；Expected 新模块不存在失败。
- [ ] 实现算法与 `AlignmentConfig(epsilon, dustbin_cost, max_iterations, tolerance, warmup_steps)`；启用时作为 ModelConfig 的可选 alignment 字段校验。Trainer 请求 `return_alignment=True`，用 `PairScores.alignment_similarity` 构造 teacher、`attention` 做 student，loss 经 `auxiliary_losses['align']` 接入 B4。参数在生产显式配置，fixture 数值不是论文默认值或最终实验值。最小闭环 `λ_align=0`；仅可靠正例弱对齐监督，负例由 pair/exist loss 监督。
- [ ] 同命令通过；B3 梯度接线测试和 B4 loss 屏蔽测试通过，teacher 的 detach 不切断 student→推理表征链路。
- [ ] 提交辅助项；commit `feat: add masked dustbin transport supervision as optional alignment loss`。生产启用需要新 study 中固定选择流程和独立消融。

## 完成条件与尚不执行的工作

- B1–B8 是基础系统闭环；B9/B10 为明确的可选研究扩展，不启用也必须在交付中报告状态，不能称全部研究扩展已验证。
- 最终 Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval` 和 A 的受影响治理测试；结构无旧架构、CPU 闭环、归档保护通过后交付代码与验证报告。
- 后续真实数据审核、独立 study 的科学配置/样本量/成功 gate、四卡资格、注册训练与 validation 方法比较是独立实验计划；本次不预留或启动。
- flow/diffusion、大库 ANN、视频到文本、冲突解释 classifier、开放世界无答案保证不在首版实现范围。
- 方法报告必须包含 raw score/margin/entropy/energy、校准 pair 阈值、去重、存在头、联合/冻结表征、局部对齐等对应消融及同预算对照；这些是真实实验前必须冻结的比较合同，不是本计划 CPU fixture 的结论。
