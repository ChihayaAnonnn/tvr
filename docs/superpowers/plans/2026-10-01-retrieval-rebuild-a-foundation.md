# Retrieval Rebuild A: Foundation Migration and Legacy Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 迁移经验证的基础组件和通用运行合同，完整清除旧 UATVR 架构，交付可独立测试、可被计划 B 使用的基础包。

**Architecture:** 新代码位于 `src/semantic_retrieval/`，不继承或包装旧模型。按组件验证后迁移，先迁出归档运行逻辑，再删除旧入口、分支、配置、专用测试与依赖。历史源码由 Git 保存，历史归档原样保留。

**Tech Stack:** 当前项目 Python / PyTorch / NumPy / OpenCV / pytest；复用既有 JSON schema 与 Scheme A 工具。本计划不安装新的模型库，也不启动 GPU 科学实验。

**Spec:** [已获批设计](../specs/2026-10-01-uatvr-audit-cleanup-and-semantic-retrieval-design.md)，含提交 `40f5803` 的端到端图。

## Global Constraints

- “HEAD / June 模型、旧训练入口及无用分支最终全部移除”；“新系统不继承旧 UATVR 类，不包装旧模型，也不保留旧架构运行选项。”
- “`logs/`、`ckpts/` 及历史数据面只读。”旧 schema、manifest、registry、输入与已发布报告不改写。
- “在 Stage 0 四卡资格测试通过并记录前，不启动 GPU 科学实验。”本计划只运行合成 CPU fixture 与临时仓库归档测试。
- “所有模型选择、K、分组阈值与校准使用 validation，test 保持 sealed。”不读取真实 test，也不创建新 study/run。
- 保留 BPE 词表、许可证和迁移来源；不保留无用途旧代码。禁止泛用 `strict=False`。
- 先用 using-git-worktrees 流程建立隔离工作区，源基线为获批文档所在提交；不得覆盖主工作区已有修改或批量删除未跟踪文件。
- 除删除审计外，任务提交只包含其明确文件；worker 运行期间禁止修改受跟踪文件。每个实现任务使用 TDD，再做相关窄验证；不重复已通过且未受后续修改影响的验证。

## Review Focus

1. 重复 canonical ID 带矛盾标注或不一致内容：拒绝合并，不静默覆盖。A1 定义身份合同，B1 验证 episode。
2. 分数并列、全 padding、非有限值：每查询一个 rank；无效 token 不进入 max。A2/A5 覆盖。
3. 编码配置与本地权重键/shape 不匹配：加载明确失败，不能变成随机初始化。A3 覆盖。
4. 未提交预留、脏源码、路径逃逸或终态复用：启动拒绝且证据保留。A6 覆盖。
5. 删除旧入口导致独立工具或保护测试失效：迁移通用断言，并验证实际依赖闭包。A6/A7 覆盖。

## 执行约定与文件边界

验证命令统一前缀（下文 `$SR_PY` 使用此解释器）：

```bash
export SR_PY=/home/xujie/.conda/envs/tvr/bin/python
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
```

所有 pytest 命令带 `-p no:cacheprovider`。这里的合成测试参数不代表科学实验配置。
断言示例中的 fixture 由各测试文件按本任务合同构造，不是额外产品 API；其余边界断言写在对应失败测试步骤中。

| 新文件 | 责任 |
|---|---|
| `config.py`, `types.py` | 通用运行配置、具名输入/特征/监督类型 |
| `evaluation/ranking.py`, `binary.py`, `baselines.py` | 修正指标与需要的数学工具 |
| `models/clip.py`, `models/weights.py`, `data/tokenizer.py` | 独立 CLIP 编码与本地权重合同、tokenizer |
| `data/video.py`, `data/splits.py` | 必要解码与 split 只读校验 |
| `models/wti.py`, `runtime/distributed.py` | 正确 mask 的评分算子与显式 gather |
| `runtime/artifacts.py`, `runtime/qualification.py` | 通用运行身份、生命周期、CPU 资格 fixture |
| `docs/maintenance/2026-10-01-code-rebuild-audit.md` | 逐项迁移/删除清单与验证证据 |

表中路径除报告外均位于 `src/semantic_retrieval/`。包 init 文件随使用它的任务创建，不做独立 scaffolding 任务。

## Task A1: 具名合同与迁移清单

**Files:** Create `src/semantic_retrieval/{__init__,config,types}.py`、`tests/semantic_retrieval/test_contracts.py`、上述审计报告；Modify `pyproject.toml` 的 pytest pythonpath 加入 `src`，保留现有根路径。

**Interfaces:**
- `RuntimeConfig(study_path: Path, registry_path: Path, run_dir: Path, run_id: str, manifest_sha256: str, source_sha: str, eval_split: str)`；`load_runtime_config(path: Path) -> RuntimeConfig`，拒绝未知键。
- `QueryInputs(input_ids: Tensor, text_mask: Tensor)` 与 `EncoderInputs(input_ids: Tensor, text_mask: Tensor, video: Tensor, frame_mask: Tensor)`：ID `[Q,L]`，video `[V,T,3,H,W]`，Q 与 V 独立；mask bool，每条实际样本至少一个有效位置，V=0 可表示空候选。
- `QueryFeatures(text_tokens: Tensor, pooled_text: Tensor, text_mask: Tensor)`、`VideoFeatures(frame_tokens: Tensor, pooled_video: Tensor, frame_mask: Tensor)`；`EncodedFeatures` 合并六个字段，末维同 D，query/video batch 独立；类型不可含监督标签。
- `PairScores(logits: Tensor, features: Tensor, dustbin_mass: Tensor | None, attention: Tensor | None = None, alignment_similarity: Tensor | None = None)`：logits `[N]`、features `[N,F]`，可选 attention `[N,L,T+1]`、similarity `[N,L,T]`；未请求对齐监督时后两者不保留。`Targets(relevance: Tensor, candidate_presence: Tensor, group_pairs: Tensor | None = None, group_labels: Tensor | None = None)`：relevance `[Q,V]`、presence `[Q]`，group_pairs 为 packed pair 之间的等价监督索引 `[M,2]`，labels `[M]`；编码 `1=positive/yes/equivalent, 0=negative/no/not-equivalent, -1=unknown`。

- [ ] 写失败测试 `test_contracts_reject_shape_mask_and_unknown_config_keys`：空 mask、错 shape、非法标签值、旧 `use_uatvr_full` 键均拒绝；`test_features_have_no_target_fields` 检查具名边界。
  ```python
  def test_features_have_no_target_fields(features):
      assert not hasattr(features, "relevance")
      assert not hasattr(features, "candidate_presence")
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_contracts.py`；Expected 初次因新包缺失失败。
- [ ] 实现以上 dataclass 与 `validate_inputs(inputs: EncoderInputs) -> None`、`validate_targets(targets: Targets) -> None`，固定错误类别为 `ValueError`。配置 loader 不设置旧模型默认值。
- [ ] 同命令通过；报告建立每个受跟踪可执行文件的处置表（迁移/删除/治理工具保留），记录延迟/相对 import、现有未跟踪脚本与 dirty diff。只读取源码/config/shell，不扫描数据面。
- [ ] 提交该任务的五类文件与必要 init；commit `feat: establish new retrieval contracts and migration inventory`。

## Task A2: 指标与基础评分工具

**Files:** Create `evaluation/{__init__,ranking,binary,baselines}.py`、`tests/semantic_retrieval/test_metrics.py`；Update 审计报告。

**Interfaces:**
- `positive_ranks(scores: ndarray, positive_mask: ndarray, candidate_ids: Sequence[str]) -> ndarray[int64]`：分数降序、ID 升序，每查询恰好一 rank；无正例行由调用者过滤，否则拒绝。
- `retrieval_metrics(scores, relevance, candidate_ids) -> dict[str, float | int]`：返回 `R1/R5/R10/MeanR/MedianR/num_eligible/num_no_answer/num_unknown`，Recall 数值范围 0–100；有 unknown 的非完整审核行不冒充精确全库评测。
- 迁移必要 `binary_calibration_metrics`、`paired_bootstrap_delta`、`score_risk_baselines` 的既有签名；内部依赖闭包随函数迁移。新增 `binary_metrics(labels: ndarray, probabilities: ndarray) -> dict[str,float | int | None]` 统一报告计数与 AUROC；单类 AUROC=None，不伪造 0.5 或输出 JSON NaN。评分 baseline 不命名为已校准概率。

- [ ] 写失败测试：矩阵 `[[1,1],[0,1]]`、ID `['a','b']` 得 ranks `[0,0]` 和 R1=100，分母=2；多正例 best rank、无答案/unknown 分母、NaN/Inf、重复 ID 拒绝、两方向分别验证、单类 calibration 可定义项正确。
  ```python
  def test_ties_keep_one_rank_per_query():
      ranks = positive_ranks(np.array([[1., 1.], [0., 1.]]), np.eye(2, dtype=bool), ["a", "b"])
      assert ranks.tolist() == [0, 0]
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_metrics.py`；Expected 新函数不存在而失败。
- [ ] 实现 `ranking.py`；从 `prob_models/retrieval_risk_{metrics,baselines}.py` 只迁移上面数学工具，记录来源。不复制 Gaussian/sidecar 汇总或旧并列算法。
- [ ] 同命令通过；迁移现有数学测试到新路径并保留其实际断言；单候选时 baseline 结果省略不可定义 margin key、entropy=0，报告不可定义原因，不能除以 log(1)。
- [ ] 提交新指标/测试/报告；commit `fix: define deterministic retrieval and binary metric contracts`。

## Task A3: CLIP 与 tokenizer 的最小迁移

**Files:** Create `models/{__init__,clip,weights}.py`、`data/{__init__,tokenizer}.py`、`data/bpe_simple_vocab_16e6.txt.gz`、`tests/semantic_retrieval/test_clip_components.py`；Update 报告。

**Interfaces:**
- `ClipConfig(embed_dim, image_size, vision_width, vision_layers, patch_size, context_length, vocab_size, text_width, text_heads, text_layers)` 参数均为 int；`ClipBackbone(config: ClipConfig)` 提供 `encode_text(input_ids: Tensor, text_mask: Tensor) -> tuple[Tensor, Tensor]` 和 `encode_frames(video: Tensor, frame_mask: Tensor) -> Tensor`。text tokens 投影到 embed_dim，视频每帧 CLS 投影到同维；不继承 UATVR，输出 fp32。首版只声明 ViT-B/32，测试用同结构缩小维度，其他模型族拒绝；B 的 dim 必须等于 embed_dim。
- `load_clip_state(path: Path, expected_sha256: str, model: ClipBackbone) -> None`：本地权重，校验哈希、完整键集合和 shape，错误即失败。
- `ClipTokenizer.encode_batch(texts: Sequence[str], max_length: int) -> tuple[Tensor, Tensor]`：pad 与截断规则显式，保留 BPE 内容哈希与原许可证。

- [ ] 写失败测试：小配置随机 CLIP CPU fixture 的 tokens/pooled/frames shape、finite、EOS/截断；完全匹配 state 加载；删除键、额外键、错 shape/哈希均失败；不访问网络或加载实际 ckpt。
  ```python
  def test_clip_rejects_missing_weight_key(model, missing_key_path, file_sha256):
      with pytest.raises(ValueError):
          load_clip_state(missing_key_path, file_sha256, model)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_clip_components.py`；Expected 新组件缺失失败。
- [ ] 从 `modules/module_clip.py` 迁移所需 ViT/text encoder 算子，去掉未用模型族、自动下载和旧 config 包装；tokenizer 与词表从已有源迁移。本地权重格式明确支持 CLIP TorchScript 的 state_dict 和裸 state_dict，原模型的非参数元数据须显式校验/剥离，不接受 UATVR `clip.*` 权重格式。前向结构按合同验证。
- [ ] 同命令通过；用相同小配置/state/tokens 验证迁移保留的编码计算，明确 dtype 与 shape，禁止旧完整模型参与此对照。
- [ ] 提交组件、许可证/来源记录及测试；commit `refactor: migrate standalone verified CLIP and tokenizer components`。

## Task A4: 视频预处理与 split 校验

**Files:** Create `data/{video,splits}.py`、`tests/semantic_retrieval/test_video_and_splits.py`；Modify `scripts/build_msrvtt_trusted_split.py` 的 import；Update 报告。

**Interfaces:**
- `decode_video(path: Path, frame_indices: Sequence[int], image_size: int) -> tuple[Tensor, Tensor]`，输出 `[T,3,H,W]` 与有效 mask；OpenCV 为首版唯一后端，不静默切换解码器。
- `sample_frame_indices(frame_count: int, max_frames: int) -> list[int]`：确定性均匀采样，不引入 TQFS/CRF。
- 从 `dataloaders/msrvtt_protocol.py` 迁移实际 split 构建/验证函数及其被调用的最小依赖，签名保持；构建脚本改用新模块，旧 schema 不重写。

- [ ] 写失败测试：CPU 合成短视频逐帧内容/采样可重复；fps=0 情况按实际帧数索引；空/坏视频、缺失文件明确失败；读取不足只标有效帧，不能伪造有效 padding；split 输入哈希改变则拒绝。
  ```python
  def test_missing_video_is_not_valid_padding(tmp_path):
      with pytest.raises(FileNotFoundError):
          decode_video(tmp_path / "missing.mp4", [0], 32)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_video_and_splits.py`；Expected 新函数缺失失败。
- [ ] 从 `rawvideo_util.py` 仅迁移必要解码与预处理；保留 CLIP normalize/resize/crop 精确参数及版本记录。处理短片/超长索引边界，不访问真实数据集。
- [ ] 同命令通过，确认构建脚本 `--help` 正常且不加载模型；读旧 split 只做 schema/哈希验证，不读取 sealed test 内容。
- [ ] 提交组件、工具 import 与测试；commit `refactor: migrate deterministic video and split utilities`。

## Task A5: masked WTI 与 gather 的梯度合同

**Files:** Create `models/wti.py`、`runtime/{__init__,distributed}.py`、`tests/semantic_retrieval/test_wti.py`、`test_distributed.py`。

**Interfaces:**
- `MaskedWTI(dim: int).forward(text: Tensor, frames: Tensor, text_mask: Tensor, frame_mask: Tensor) -> Tensor`，输入 `[Q,L,D]` / `[V,T,D]`，输出 `[Q,V]`；可学习 token/frame 权重，仅在有效位置归一化。
- `gather_with_grad(values: Tensor) -> Tensor`；`gather_metadata(values: Tensor) -> Tensor`，首版各 rank 相同 batch，变化长度明确拒绝；metadata 无梯度。

- [ ] 写失败测试：有效相似度 −1，加 masked padding 后仍 −1；全空/NaN 拒绝，padding 内容/长度不影响结果；有效输入可反传；不同视频 chunk 拼接等于一次计算。
  ```python
  def test_padding_does_not_beat_negative_similarity():
      score = MaskedWTI(2)(torch.tensor([[[1., 0.]]]), torch.tensor([[[-1., 0.], [1., 0.]]]),
                           torch.tensor([[True]]), torch.tensor([[True, False]]))
      assert score.item() == pytest.approx(-1.)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_wti.py tests/semantic_retrieval/test_distributed.py`；Expected 缺实现失败。
- [ ] 实现 max 前用有效 pair mask 排除位置，正确处理方向聚合且不引入 `0 * -inf` NaN；gather 优先采用标准 PyTorch autograd-aware collective，不拷贝未验证的旧 AllGather。
- [ ] 同命令通过，包括两 rank CPU/Gloo 合成 backward 对单进程参考的梯度/缩放比较、metadata 对齐、错 batch 拒绝；这不是 GPU 科学实验。
- [ ] 提交算子及梯度测试；commit `fix: implement padding-safe matching and explicit gather semantics`。

## Task A6: 归档桥接迁出旧主程序

**Files:** Create `runtime/{artifacts,qualification}.py`、`tests/semantic_retrieval/test_runtime_artifacts.py`；Modify `scripts/manage_experiment_run.py`、`experiment_tracking.py`、`tests/test_{manage_experiment_run,experiment_tracking,experiment_archive_e2e}.py`；通用 `experiment_artifacts.py` 仅有必要改动时修改。

**Interfaces:**
- `RunContext` 包含既有 `RunPaths`、manifest、`ArtifactArrayStore`。
- `prepare_run(config: RuntimeConfig, *, repo_root: Path) -> RunContext`；`start_run(context)`、`complete_run(context)`、`fail_run(context, *, exit_code: int)`、`publish_metrics(context, payload: Mapping[str, object]) -> Path`；沿用既有生命周期和 hash 校验。
- `run_cpu_qualification(context: RunContext, *, scenario: str) -> None`，scenario 仅 `success/array_tamper/signal`；用测试生成的临时仓库 manifest，不绕过真实启动 gate。

- [ ] 迁移失败测试：未提交 reservation、脏源码、命令/config/hash 不一致、路径越界、sealed test、重复 terminal、数组篡改、单个可修复 pending transition；无模型 import 即可验证。
  ```python
  def test_uncommitted_reservation_cannot_start(repo, uncommitted_config):
      with pytest.raises(ValueError):
          prepare_run(uncommitted_config, repo_root=repo)
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_runtime_artifacts.py`；Expected 新桥接缺失失败。
- [ ] 抽取旧 `_prepare/_start/_complete/_fail/_publish_artifact_*` 必要通用行为，移除 Namespace 模型开关耦合；新 config 与 manifest 显式比较，环境/精度/身份检查不得 no-op。Scheme A manager 保持现有命令与旧 run 只读校验，去掉生成旧 A3 启动 argv 的活动入口。旧 schema 校验数据可保留，不保留旧模型构造。
- [ ] 改现有归档/追踪测试为新桥接，保留通用断言；旧 A3 专项测试移除的原因记报告。现有 `experiment_tracking.py` 仅保留仍被治理工具需要的通用函数/历史数据解释，删除模型训练配置构造分支。
- [ ] Run 上述新测试及 `tests/test_experiment_artifacts.py tests/test_manage_experiment_run.py tests/test_experiment_tracking.py tests/test_experiment_archive_e2e.py`，全部相关 CPU 用例通过；若测试包含 GPU 子集，显式区分未运行项，不把 skip 当资格通过。
- [ ] 提交桥接/工具/保护测试；commit `refactor: decouple experiment lifecycle from retired training entrypoint`。

## Task A7: 删除旧架构及其完整依赖闭包

**Files:** Delete 以下明确范围；Update `requirements.txt`、`README.md`、`docs/agent/README.md`、审计报告；Create `tests/semantic_retrieval/test_no_legacy_dependencies.py`。

删除清单以 A1 报告逐文件展开，以下目录模式禁止未经核对直接 `rm -rf`：

- `main_task_retrieval.py`、`metrics.py`、`modules/modeling.py`、`modules/modeling_mulit.py`。
- 迁移完必要组件后清除旧 `modules/` 源码/词表/config；旧 `query_models/`、`prob_models/`、`uncertainty_modules/`、`crf_ladder/` 可执行代码；`dataloaders/` 的旧 dataset/CRF/TQFS/属性路径与已迁移源码。冻结 split JSON 等受跟踪研究输入保留在原位置。
- 旧 `run_train*.sh`、`eval.sh`、`train_msvd.sh`、`smoke_test_msrvtt.sh`；两个未跟踪 MSVD 旧启动脚本在主工作区只读核对/记录 hash，不复制进隔离 worktree；集成时按批准清单和匹配 hash 清除，未知或已变化内容不得批量删除。
- `scripts/oq1_*`、CRF builder/evaluator、旧模型诊断、旧 reliability campaign、旧 frozen-equivalence verifier、旧 ckpt args dump；`build_idf_weights.py`、hard-negative 工具等若新系统无消费也清除。
- 旧 A1/A2/A3/OQ1/CRF/uncertainty/launcher 专用测试和无用 bench。数学/归档/split 保护已迁移的测试由新任务保留，而不是删断言后照旧宣称通过。
- `util.py` 与旧 file/config/optimizer 包装若无新消费删除。独立 `inventory_legacy_artifacts.py`、`migrate_critical_artifacts.py` 及对应证据校验测试保留，必要通用 hash 调用迁出旧模块。

**Interfaces:** 新 package + archive tools 是唯一执行面；不存在旧 import 或转发入口。

- [ ] 写失败测试 `test_active_sources_do_not_import_or_select_legacy_architectures`、`test_supported_cli_help_without_legacy_imports`：覆盖 Python 静态/相对 import、动态 import 的 literal、shell 中旧入口与开关。historical docs/schema/registry 不按代码检查；保证测试确实能捕获注入的坏引用。
  ```python
  def test_old_model_entrypoints_are_removed(repo_root):
      for name in ("main_task_retrieval.py", "modules/modeling.py", "modules/modeling_mulit.py"):
          assert not (repo_root / name).exists()
  ```
- [ ] Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval/test_no_legacy_dependencies.py`；Expected 当前遗留入口存在而失败。
- [ ] 按报告显式 `git rm -- <逐项路径>`，删除依赖闭包和仅为旧分支存在的 requirement；不删除 JSON 冻结输入或任何 artifacts/ckpts/logs。必要数学工具进一步迁移到 A2 新模块，不能留下旧包转发壳。
- [ ] 同命令通过；Run `$SR_PY -m pytest -q -p no:cacheprovider tests/semantic_retrieval tests/test_experiment_artifacts.py tests/test_experiment_artifact_docs.py tests/test_manage_experiment_run.py tests/test_experiment_tracking.py tests/test_experiment_archive_e2e.py tests/test_legacy_artifact_migration.py`；保留的通用文档/迁移/治理测试若引用旧执行入口，改为新入口合同再验证。相关 CPU 测试通过，新源码无旧依赖。记录退出码和未执行资格范围。
- [ ] 比较受保护控制面与数据面已有 metadata 的 hash 清单（不重读大权重）；检查 git diff 只包含计划代码/文档，治理命令 `--help` 正常。更新根 README 为新开发路径，历史文档明确仅 source SHA 复现，不把新系统描述为已训练。
- [ ] 提交实际删除清单与验收证据；commit `refactor: remove legacy UATVR architectures and obsolete execution paths`。

## A 阶段完成与 B 的交接

- 新组件和 CPU 验证是可用交付；此时尚无研究模型或真实数据训练结果。
- A7 的结构、保护与归档验证全部通过后，才能执行计划 B。失败必须修复或调整获批设计，不能恢复旧模型作为 fallback。
- 审计报告列出迁移来源、删除文件、断言去向、依赖变更、精度/梯度证据及保护文件未修改证据。
- 实施计划待用户审阅并选择执行方式；本文件不是实验预留或 GPU 启动授权。
