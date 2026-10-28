# UATVR + AQS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划按单 agent 顺序执行；只有用户另行要求委派时才采用 subagent-driven-development。

**Goal:** 实现与 Laya 无依赖的 UATVR 视频级 AQS 训练扩展，交付可验证的选择器、辅助损失、真实模型适配和恢复接口。

**Architecture:** 保留完整 batch 双向检索与现有概率辅助损失；仅增加 T2V 的所选负视频辅助损失。停止梯度的原始 WTI 分数决定难度，停止梯度的时序帧池化表示决定多样性；辅助梯度来自原始缩放 logits。推理沿用 UATVR，不执行选择。

**Tech Stack:** 现有 Python/PyTorch、pytest、CPU Gloo、UATVR/CLIP、现有 checkpoint/ArtifactArrayStore；不增加第三方运行依赖。

**Design:** `docs/superpowers/specs/2026-10-29-uatvr-aqs-design.md`，设计提交 `19cf89fbfae37c450dcfaaeaac2a2e5d0a3401e6`。

## Global Constraints

- 本次交付是实施文档。下列任务均未执行；命令及 PASS 是未来验收预期，不是当前测试记录。
- 实施范围为 CPU 软件与合成数据验证。不创建/运行科学 GPU 实验，不访问 test，不提供绕过 manifest/gate 的科学训练 CLI。
- 新逻辑位于 `src/semantic_retrieval/aqs/`；只选择整段负视频；不引入局部帧损失、负文本选择、额外 query 或新温度。
- 保留完整 batch 双向 CrossEn；MIL/KL 使用现有返回的已加权标量，不能再次乘 0.01/0.0001。
- 启用时每个全局 batch 的 canonical video ID 唯一，使用 `unique_video_all_captions_v1`。每个 epoch 遍历全部 caption，不能为“一视频一描述/step”而丢弃其他描述。
- `M>=1`、`lambda>=0`；启用值没有科学默认值。下文测试中的 `M=2, weight=0.2` 仅是合成软件测试常量。
- 难度范围 `<=1e-8` 时有效负例难度为 1；余弦夹到 `[-1,1]`；无有效负例的辅助损失为 0。
- AQS 关闭或权重为 0 时直接执行原有路径；保持随机数、损失、logits 和梯度，不先计算新分支再乘零。
- 不 import Laya、不加载其权重/processor、不读取其缓存或输出、不复用其适配器/资格/授权。保留其代码和历史产物。
- 不修改 `src/semantic_retrieval/full_data/upstream_uatvr/` 及其 PROVENANCE；生产路径不使用运行时 monkey-patch、forward hook、`__class__` 替换或 `strict=False`；测试可用受控拦截器验证隔离。
- 正式训练实施前需独立 study 和实际适用资格；已有 Laya 或 full-data 双卡/test 授权不外溢。此计划不创建新 study，也不确定实验预算、拓扑或运行矩阵。

---

## 0. 已核对的仓库事实与实施准备

已核对入口：

| 现有文件 | 复用内容 / 限制 |
|---|---|
| `src/semantic_retrieval/full_data/models.py` | `BaselineModel`，已收集全局 ID 并检查唯一性；`_logits` 只返回 logits/MIL/KL |
| `src/semantic_retrieval/full_data/upstream_uatvr/modeling.py` | `_loose_similarity` 内部完成时序处理、all-gather、WTI 和概率路径；暂未暴露 WTI 输入 |
| `src/semantic_retrieval/full_data/training.py` | `unique_video_batches`、`make_optimizer`、`save_state`、`restore_state` |
| `src/semantic_retrieval/full_data/loader.py` | `CaptionDataset` 提供由排序视频表映射的整数 ID；现有 batch 可直接传入模型 |
| `src/semantic_retrieval/full_data/qualification.py` | `step` 固定提取四种旧损失，不能拿它冒充已有 AQS 日志 |
| `src/semantic_retrieval/runtime/distributed.py` | `gather_metadata` 等长、rank-order 聚合；禁止重复归约梯度 |
| `tests/semantic_retrieval/full_data/test_models_training.py` | 真实 tiny UATVR、双 rank Gloo 梯度及恢复测试模式 |

- [ ] 实施开始时读取适用 `AGENTS.md`，用 using-git-worktrees 技能建立独立 `codex/uatvr-aqs-v1` worktree；不切换、重置或清理当前用户目录。
- [ ] 核对所选 UATVR 基点是否包含上述文件与设计提交。使用 `git log --oneline -- <file>` 和 PROVENANCE 哈希验证，记录实际基点；不依赖分支名推断身份。
- [ ] 只迁移设计/计划文档和本任务所需代码，不把未跟踪的 `QueryMatch/`、`MANIFEST.md`、`idea-stage/`、`figures/` 加入提交。
- [ ] 验证 CPU 环境：`conda run --no-capture-output -n tvr python -c "import torch, pytest; print(torch.__version__)"`。若环境不存在，选择已有兼容环境并在报告记录实际命令；不在 Laya 专属 overlay 安装依赖。

以下 CPU 命令使用 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest ...`。不会生成 GPU 资格回执。

## 文件结构

| 动作 | 文件 | 责任 |
|---|---|---|
| Create | `src/semantic_retrieval/aqs/__init__.py` | 包说明，保持轻量，不 eager import 模型 |
| Create | `src/semantic_retrieval/aqs/config.py` | 严格配置与 canonical recipe |
| Create | `src/semantic_retrieval/aqs/selection.py` | 身份校验、纯池化、贪心选择 |
| Create | `src/semantic_retrieval/aqs/loss.py` | 稳定的 T2V 辅助损失 |
| Create | `src/semantic_retrieval/aqs/core.py` | 显式、版本绑定的训练组件接口 |
| Create | `src/semantic_retrieval/aqs/model.py` | `AqsBaselineModel`，训练组合与原有推理 |
| Create | `src/semantic_retrieval/aqs/training.py` | 软件训练 step、recipe 身份、checkpoint 包装 |
| Modify | `src/semantic_retrieval/full_data/models.py` | 仅提取 UATVR core 类型工厂；基线默认类型不变 |
| Create | `tests/semantic_retrieval/aqs/conftest.py` | tiny CLIP、batch、模型工厂 |
| Create | `tests/semantic_retrieval/aqs/test_config_selection.py` | 配置、身份、选择、池化行为 |
| Create | `tests/semantic_retrieval/aqs/test_loss.py` | logits 梯度及边界 |
| Create | `tests/semantic_retrieval/aqs/test_core_model.py` | 真模型组件等价、开关与参数梯度 |
| Create | `tests/semantic_retrieval/aqs/test_distributed.py` | 双 rank CPU 集合与梯度 |
| Create | `tests/semantic_retrieval/aqs/test_training_resume.py` | 新 recipe 身份、恢复及辅助日志 |
| Create | `tests/semantic_retrieval/aqs/test_isolation.py` | Laya/QueryMatch 依赖与写入隔离 |
| Create | `docs/agent/uatvr-aqs.md` | 软件接口与资格边界说明 |

## Task 1: 配置、身份和纯池化

**Files:** Create `aqs/__init__.py`、`aqs/config.py`、`aqs/selection.py`、`tests/semantic_retrieval/aqs/conftest.py`、`test_config_selection.py`（均使用上表完整目录）。

**Interfaces:**

```python
@dataclass(frozen=True)
class AqsConfig:
    enabled: bool
    negatives: int
    weight: float
    @property
    def active(self) -> bool: ...
    def recipe(self) -> dict: ...

def validate_ids(video_ids: torch.Tensor, batch: int) -> None: ...
def pooled_frames(tokens: torch.Tensor, mask: torch.Tensor,
                  frame_count: int) -> torch.Tensor: ...  # [B,D], FP32, detached
```

- [ ] 写配置及池化失败测试，包含 bool 混入整数、NaN/负权重、重复 ID、负 ID、全 padding 和零范数。关键行为测试：

```python
def test_pool_ignores_padding_and_aggregate_tokens_without_mutation():
    from semantic_retrieval.aqs.selection import pooled_frames
    x = torch.tensor([[[3., 0.], [0., 4.], [90., 90.]]], requires_grad=True)
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    before, before_mask = x.detach().clone(), mask.clone()
    actual = pooled_frames(x, mask, frame_count=2)
    torch.testing.assert_close(actual, torch.tensor([[1., 0.]]), rtol=0, atol=0)
    assert not actual.requires_grad
    assert torch.equal(x.detach(), before) and torch.equal(mask, before_mask)
```

- [ ] 运行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_config_selection.py -q`，预期因新接口未实现失败。
- [ ] 实现配置：严格 `type(enabled) is bool`、`type(negatives) is int`，要求整数至少 1；weight 为非 bool 有限数且非负。`active = enabled and weight > 0`。`recipe()` 固定包含 schema `uatvr-aqs-v1`、三个字段、`difficulty_epsilon=1e-8`、`selection_dtype=fp32`、`tie_break=canonical_video_id`、`aux_direction=t2v`、`batching=unique_video_all_captions_v1`，拒绝未知外部字段。
- [ ] 池化核心使用下列运算，先验证维度、device、掩码为 0/1、`1<=frame_count<=T`、有效帧有限，再只读取前 frame_count 槽位；不能用 `NaN * 0` 掩盖无效值：

```python
with torch.no_grad():
    frames = tokens.detach()[:, :frame_count].float()
    valid = mask[:, :frame_count].bool()
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError('all-padding video')
    selected = torch.where(valid[..., None], frames, torch.zeros_like(frames))
    mean = selected.sum(1) / counts[:, None]
    norms = mean.norm(dim=-1, keepdim=True)
    if not torch.isfinite(mean).all() or (norms <= 1e-8).any():
        raise ValueError('invalid pooled video representation')
    return mean / norms
```

- [ ] 从既有 `test_models_training.py` 的 fixture 构造 tiny CLIP/batch：`ClipConfig(64,16,64,1,4,32,128,64,1,1,fixture=True)`；新目录自有 fixture，不从其他测试文件 import 私有 fixture。`video_ids` 是按全局有序字符串表映射的 long，映射哈希在 Task 7 绑定。
- [ ] 重跑本任务测试，预期通过；提交 `feat: add strict AQS configuration and pure frame pooling`，仅 add 本任务文件。

## Task 2: 确定性难度与多样性选择

**Files:** Modify `src/semantic_retrieval/aqs/selection.py`、`tests/semantic_retrieval/aqs/test_config_selection.py`。

**Interfaces:**

```python
@dataclass(frozen=True)
class Selection:
    indices: torch.Tensor  # long[B,m]；B=1 时 m=0
    selected_ids: torch.Tensor  # long[B,m]
    difficulty: torch.Tensor  # detached FP32[B,m]
    uniqueness: torch.Tensor  # detached FP32[B,m]，每项选中时的值

def select_negatives(scores: torch.Tensor, video_repr: torch.Tensor,
                     video_ids: torch.Tensor, negatives: int) -> Selection: ...
```

- [ ] 写以下行为测试，另覆盖 `B=1`、`M>B-1`、全相同难度、全相同表示、非有限输入和顺序置换：

```python
def test_hard_first_then_diverse():
    from semantic_retrieval.aqs.selection import select_negatives
    ids = torch.tensor([10, 20, 30, 40, 50])
    scores = torch.zeros(5, 5)
    scores[0] = torch.tensor([0., .9, .8, .7, .1])
    reps = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [-1., 0.], [0., -1.]])
    out = select_negatives(scores, reps, ids, negatives=2)
    assert out.selected_ids[0].tolist() == [20, 40]
    assert not out.difficulty.requires_grad
```

- [ ] 运行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_config_selection.py -q`，先确认新测试失败。
- [ ] 实现：输入必须方阵、同 device、行列共享同一唯一 ID 表；要求非空 B、`D>=1`、正 M，拒绝非有限值。选择全程 `@torch.no_grad()`，FP32。代表向量归一化，先按 `torch.argsort(video_ids, stable=True)` 重排候选列，保存返回原列映射。
- [ ] 使用以下循环主体；`available` 初始排除同 ID，`h` 为仅有效负例的 min-max 难度，范围 `<=1e-8` 令 h=1；`distance` 是重排后的归一化代表向量两两距离。B=1 在 min/max 前返回 `[1,0]`：

```python
uniqueness = torch.ones_like(h)
chosen, chosen_h, chosen_u = [], [], []
for _ in range(min(negatives, scores.shape[0]-1)):
    quality = (h * uniqueness).masked_fill(~available, -torch.inf)
    j = quality.argmax(dim=1)  # 有序候选中首个最大值：canonical ID tie-break
    rows = torch.arange(len(scores), device=scores.device)
    chosen.append(candidate_order[j])
    chosen_h.append(h[rows, j])
    chosen_u.append(uniqueness[rows, j])
    available[rows, j] = False
    uniqueness = torch.minimum(uniqueness, distance[:, j].T)
```

- [ ] 构造 Selection 时沿第二维 stack；`selected_ids=video_ids[indices]`。禁止重复选择。诊断不读取 GT、caption、Laya 分数；不增加随机数消耗。
- [ ] 运行本任务测试，确认置换后按 ID 比较的选择一致；提交 `feat: select diverse hard negative videos deterministically`。

## Task 3: 稳定辅助损失和真实 logits 梯度

**Files:** Create `src/semantic_retrieval/aqs/loss.py`、`tests/semantic_retrieval/aqs/test_loss.py`。

**Interfaces:** `selected_t2v_loss(logits: Tensor, selection: Selection) -> Tensor`。logits 已包含原有 logit scale，不接受 temperature 参数。

- [ ] 写梯度行为测试，不只比较两份相同公式：

```python
def test_gradient_reaches_positive_and_selected_negative_only():
    from semantic_retrieval.aqs.selection import select_negatives
    from semantic_retrieval.aqs.loss import selected_t2v_loss
    raw = torch.tensor([[.2,.9,.4],[.8,.3,.6],[.5,.7,.1]], requires_grad=True)
    reps = torch.eye(3, requires_grad=True)
    selected = select_negatives(raw, reps, torch.arange(3), 1)
    logits = raw.clone().detach().requires_grad_()
    loss = selected_t2v_loss(logits, selected)
    loss.backward()
    rows = torch.arange(3)
    assert (logits.grad.diag() < 0).all()
    assert (logits.grad[rows, selected.indices[:, 0]] > 0).all()
    used = torch.eye(3, dtype=torch.bool)
    used[rows, selected.indices[:, 0]] = True
    assert (logits.grad[~used] == 0).all()
    assert raw.grad is None and reps.grad is None
```

- [ ] 运行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_loss.py -q`，预期新接口缺失；再实现以下主体：

```python
def selected_t2v_loss(logits, selection):
    # 先验证方阵、有限 logits；indices long、同 device、B 对齐、合法列、
    # 不含自身、每行无重复。不能从传入的诊断值重建 loss。
    if selection.indices.shape[1] == 0:
        return logits.sum() * 0.0
    chosen = logits.gather(1, selection.indices)
    columns = torch.cat((logits.diagonal()[:, None], chosen), dim=1)
    targets = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(columns, targets)
```

- [ ] 补充大 logits 不溢出、空负集为有限零且可 backward、伪造负索引/对角列/重复列被拒绝；跑 config/selection/loss 测试，预期通过。
- [ ] 提交 `feat: add selected-video T2V auxiliary contrastive loss`。

## Task 4: 显式取得真实 WTI 分数和时序帧表示

**Files:** Create `src/semantic_retrieval/aqs/core.py`；Modify `src/semantic_retrieval/full_data/models.py`；Create `tests/semantic_retrieval/aqs/test_core_model.py`。

**决定：** 当前 upstream 不暴露必要中间量。为保持 vendor/provenance 不变，在新包维护一个有源码等价约束的 `_loose_similarity` 覆盖版本，只增加可选 observer 形参及一个调用点；其他代码原样继承/复制，不重写编码、概率或梯度聚合算法。复制的局部方法是显式版本化扩展，不是运行时抽取/exec。

**Interfaces:**

```python
class AqsCore(UATVR):
    def _loose_similarity(self, sequence_output, text_token, visual_output,
                          attention_mask, video_mask, sim_header='seqTransf',
                          *, training_observer=None): ...

    def training_components(self, text: dict, video: dict) -> tuple:
        # (logits, weighted_mil, weighted_kl, detached_raw_wti, detached_video_repr)
        ...
```

- [ ] 首先在 `test_core_model.py` 写失败测试：同一个严格加载的 state、输入和 RNG 下，基线与新 core 返回的 logits/MIL/KL、RNG 一致；observer 恰好调用一次；raw WTI×scale 等于 logits；聚合 token 不进入多样性池化；state-dict keys/shape 不变。
- [ ] 为 `BaselineModel` 提取下列默认工厂，并把原构造语句 `UATVR(cross, dict(clip_state), task)` 改成 `self._uatvr_core_class()(cross, dict(clip_state), task)`。不修改权重初始化、参数顺序或其他方法：

```python
@staticmethod
def _uatvr_core_class():
    from .upstream_uatvr.modeling import UATVR
    return UATVR
```

- [ ] 新 core 复制当前 `_loose_similarity` 方法，保留注释及来源说明。将其签名增加 keyword-only observer；紧接原有 `wti_logits = self.weighted_token_wise_intersection(...)` 后插入且仅插入：

```python
if training_observer is not None:
    if not self.training:
        raise ValueError('AQS observer is training-only')
    training_observer(wti_logits, visual_output, video_mask, frame_num)
```

文件只需显式 import `torch`、`UATVR`、原模块 `allgather`、`pack_padded_sequence` / `pad_packed_sequence`，以及新的 `pooled_frames`；不 wildcard import。编写 AST 测试移除 observer 形参和插入的 if 后，与来源方法 AST（忽略位置信息）完全相等；不只比文件名。这样上游漂移不能无声通过。

- [ ] 实现组件接口，observer 使用函数局部容器，不把中间量写到模块属性，避免跨 batch 持有计算图：

```python
def training_components(self, text, video):
    if not self.training or self.sim_header != 'seqTransf':
        raise ValueError('AQS components require UATVR seqTransf training')
    observed = []
    def observe(raw, frames, mask, count):
        observed.append((raw.detach().float(), pooled_frames(frames, mask, count)))
    logits, mil, kl = self._loose_similarity(
        text['pooled'], text['tokens'], video['tokens'],
        text['mask'].long(), video['mask'].long(),
        sim_header=self.sim_header, training_observer=observe)
    if len(observed) != 1:
        raise ValueError('unexpected WTI observation count')
    return logits, mil, kl, *observed[0]
```

- [ ] 组件已有原 upstream all-gather：不要再 gather 一次 frames 或 raw WTI。ID 由后续 adapter 以同一 rank 顺序 gather。
- [ ] 跑 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_core_model.py tests/semantic_retrieval/full_data/test_models_training.py -q`，预期新旧 core 等价及已有初始化/梯度/推理回归通过。
- [ ] 提交 `feat: expose version-bound UATVR training components for AQS`。

## Task 5: 接入训练组合，严格保留禁用和推理路径

**Files:** Create `src/semantic_retrieval/aqs/model.py`；Modify `tests/semantic_retrieval/aqs/test_core_model.py`、`conftest.py`。

**Interfaces:** `AqsBaselineModel(clip_config, clip_state, *, aqs: AqsConfig, id_mapping_sha256: str, **baseline_kwargs)`；`forward` 与 BaselineModel 的五个输入完全一致。继承 `encode_text`、`encode_video`、`score`；新增输出只在 active 训练路径提供 `aqs` 及 `aqs_stats`。

- [ ] 先写失败测试：disabled 与 weight=0 的逐项损失/梯度/RNG 相同；active 时 `loss == contrastive+mil+kl+weight*aqs`；`autograd.grad(output['aqs'], ...)` 对文本投影、视觉投影、时序参数非零且有限；固定 state 的 eval score 与基线相同。
- [ ] 子类通过工厂返回 AqsCore；`id_mapping_sha256` 必须是实际排序视频表的 canonical SHA-256（64 位小写十六进制），软件 fixture 使用自己的合成排序表。模型只保存普通字符串和配置，不把身份映射输入编码器；构造只允许 UATVR 和唯一视频 batching，拒绝 `caption_shuffle_v1`。配置是普通 immutable 数据，不注册额外参数或 buffer，不改变原有 state-dict key。
- [ ] 构造时（若 dist 已初始化）用一次 `dist.all_gather_object` 比较所有 rank 的 `aqs.recipe()` 与 `id_mapping_sha256`；包括 disabled，避免不同开关的 rank 进入不同 collective 序列。若模型早于 process group 创建，则在外部 DDP 包装前显式调用一次相同的构造检查，不把检查偷偷放入 disabled 前向。active 分支在任何编码前校验本地输入。局部校验和编码使用现有 `_collective_call` 将异常同步传播，避免一个 rank 失败后其他 rank 卡在下一次 gather；所有 rank 调用顺序一致。
- [ ] forward 采用下列组合，变量 `global_ids` 为 `gather_metadata(video_ids)` 的结果；先 `validate_ids`，再编码；验证组件/logits 与全局 B 对齐：

```python
if not self.training or not self.aqs_config.active:
    return super().forward(input_ids, text_mask, video, frame_mask, video_ids)
text, visual = _collective_call(lambda: (
    self.encode_text(input_ids, text_mask), self.encode_video(video, frame_mask)))
logits, mil, kl, raw, representation = self.core.training_components(text, visual)
target = torch.arange(len(global_ids), device=logits.device)
contrastive = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
selection = select_negatives(raw, representation, global_ids, self.aqs_config.negatives)
aqs = selected_t2v_loss(logits, selection)
loss = contrastive + mil + kl + self.aqs_config.weight * aqs
if not torch.isfinite(loss):
    raise ValueError('nonfinite AQS training loss')
return dict(loss=loss, contrastive=contrastive, mil=mil, kl=kl,
            logits=logits, aqs=aqs, aqs_stats=summarize_selection(selection))
```

`summarize_selection(selection) -> dict[str,float]` 在 `selection.py` 实现，只返回平均有效负例数、平均所选难度、平均选中时 uniqueness、空负集行数，空张量均值显式为 0；不将完整索引或逐查询数组嵌入日志。形状/类型校验不在此片段省略的调用中失效。

- [ ] disabled 使用 inherited core 默认 observer=None，不构造 ID 新映射、不执行池化、选择或新 RNG 操作。推理通过 inherited `score`，不调用 active forward，保持旧 eval 前向语义；不能悄悄修复旧 singleton mask 问题，见 Task 8。
- [ ] 跑本任务及旧 `test_models_training.py`，预期全部通过；提交 `feat: integrate AQS auxiliary learning without changing UATVR inference`。

## Task 6: 全局身份、双 rank 梯度及失败传播

**Files:** Create `tests/semantic_retrieval/aqs/test_distributed.py`；必要实现收敛在 `aqs/model.py`、`selection.py`。

**Interfaces:** 沿用 Task 5 的五输入模型；rank-order ID gather 与现有 upstream 特征 gather 对齐。通过训练构造时的 recipe 和 ID 映射 digest 检查 rank 一致性。

- [ ] 以现有 `_ddp_worker` 测试模式创建独立 Gloo worker，`timeout=45s`、`torch.set_num_threads(1)`、每 rank 两个样本、全局四个唯一 ID；每个 rank 输出选择 ID、辅助损失及命名参数梯度到 pytest tmp_path。测试文件自身不得加载真实数据/权重。
- [ ] 覆盖：相同全局合成输入下串行与 DDP 的 AQS 辅助梯度；行列整体置换后的 ID 选择一致；跨 rank 重复 ID 被一致拒绝；ID 映射摘要不一致、配置不一致和一个 rank 非法本地输入均在 gather 前同步失败。
- [ ] 使用显式断言：选择 ID 必须精确相等；FP32 梯度 `atol=2e-6, rtol=2e-4` 与既有 CPU Gloo 测试一致。浮点近似 tie 用构造的非近似值验证；CPU 不能替代 CUDA 资格。
- [ ] 概率辅助分支有随机采样，因此先用单独 `output['aqs']` 验证辅助全局梯度，再保留既有全损失回归；不能将不同 RNG 拓扑产生的 MIL 差异归咎于 AQS，或为通过测试去掉概率分支。
- [ ] 运行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_distributed.py -q`。失败时核对已 gather 的全局 logits 不再额外 all-reduce loss；不能手工乘 world_size 掩盖梯度倍数问题。
- [ ] 提交 `test: verify AQS global identities and distributed gradients`。

## Task 7: 软件 step、日志与严格恢复

**Files:** Create `src/semantic_retrieval/aqs/training.py`、`tests/semantic_retrieval/aqs/test_training_resume.py`。

**Interfaces:**

```python
def aqs_step(model, network, optimizer, batch) -> dict[str, float]: ...
def bound_identity(base_identity: dict, base_recipe: dict, aqs: AqsConfig,
                   id_mapping_sha256: str) -> dict: ...
@dataclass(frozen=True)
class CheckpointBinding:
    base_identity: dict
    base_recipe: dict
    id_mapping_sha256: str

def save_aqs_state(path, model, optimizer, binding: CheckpointBinding, progress, store): ...
def restore_aqs_state(path, model, optimizer, binding: CheckpointBinding): ...
```

- [ ] 使用现有 `make_optimizer`，保持冻结策略、参数分组及学习率倍率；不增加选择器参数。`aqs_step` 将现有 `qualification.step` 的 backward、非有限检查、梯度裁剪和 logit_scale clamp 语义保留在独立实现，额外记录 aqs/aqs_stats，不能调用一次旧 step 后再重复更新。disabled 输出没有新键时，日志层将辅助项及选择计数显式记零，但不能反向触发选择计算；日志记录 active 标志以区分“未运行”与“运行后为零”。
- [ ] 写失败测试：一次合成模型更新确实改变原允许训练参数；冻结参数保持不变；日志含 `loss/contrastive/mil/kl/aqs` 和四个选择诊断。检查 aqs 原始损失与加权贡献分开命名，不能把 weight 乘两次。
- [ ] 用现有 `unique_video_batches` 生成每 epoch 顺序，不新写采样器；测试所有 caption 一次覆盖、全局唯一性、最后等长分片以及由 epoch/cursor 重建下一个 batch。保存排序视频 ID 表的 canonical SHA-256，不能跨数据集重新编号后继续恢复。
- [ ] `bound_identity` 保留既有 schema 要求的五个键和 `method='uatvr'`，但重算 recipe 哈希绑定 `architecture='uatvr-aqs-v1'`、base_recipe、`aqs.recipe()`、ID 映射哈希和核心扩展源身份。科学输入哈希仍由认证输入产生；不伪造已注册 recipe：

```python
payload = dict(architecture='uatvr-aqs-v1', base_recipe=base_recipe,
               aqs=aqs.recipe(), id_mapping_sha256=id_mapping_sha256)
identity = dict(base_identity)
identity['recipe_sha256'] = hashlib.sha256(artifact_json_bytes(payload)).hexdigest()
```

核心扩展绑定到 base_recipe 中的源码/版本指针，source_sha 仍为实际干净源码 SHA。先验证 base_identity、映射 digest 格式和 canonical recipe，禁用/启用均使用 AQS 包装时显式绑定其配置，不用基线 recipe 冒名。

- [ ] `save_aqs_state/restore_aqs_state` 先检查 `binding.id_mapping_sha256 == model.id_mapping_sha256`，再用 `bound_identity(binding.base_identity, binding.base_recipe, model.aqs_config, binding.id_mapping_sha256)` 重建 identity，交给既有 `save_state/restore_state`。`CheckpointBinding` 在创建时对字典做深复制并拒绝非 canonical JSON 值，序列化时再次校验；不接受调用方直接覆盖已派生的 recipe 哈希。复用旧完整 checkpoint schema 和 array store，不宽松加载。
- [ ] 合成两步恢复测试：保存 step1，连续执行 step2，再新建模型/优化器，恢复 step1 后执行同一 step2；严格比较模型、optimizer、RNG、选择索引、损失及下一 sampler cursor。篡改 M、weight、ID 映射、source、world_size 或 recipe 均须在模型/优化器/RNG 被修改前拒绝。
- [ ] `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs/test_training_resume.py -q` 预期通过；提交 `feat: bind AQS training diagnostics and exact recovery identity`。

## Task 8: Laya 隔离、回归验收与交付

**Files:** Create `tests/semantic_retrieval/aqs/test_isolation.py`、`docs/agent/uatvr-aqs.md`；Create `docs/superpowers/reports/2026-10-29-uatvr-aqs-implementation.md`（实际完成日期不同则使用完成日）。

- [ ] 依赖隔离测试在 fresh CPU subprocess 安装 import blocker，拒绝 `semantic_retrieval.laya_rerank`、任何 `QueryMatch` 包以及 Laya 外部模型模块。对子进程的 `open`/`Path.open`/目录枚举访问拦截 Laya 权重/缓存/产物根路径，再执行 tiny 构造、active forward/backward、disabled forward 和 eval score。只静态搜索 import 不足以完成验收。
- [ ] 同时用 AST 检查新包无 Laya/QueryMatch import；测试写入只允许 pytest tmp_path，不以删除真实目录证明隔离。提交前确认 Laya 代码/控制面/数据面和 vendor 源文件均无 diff。
- [ ] 固定相同权重、输入、精度和批处理路径，验证 active/inactive 模型在 eval 的 logits 和排序相同。额外记录已有单查询/多查询掩码别名限制：若原基线本身不等价，报告已有问题，不把 Laya measurement monkey-patch 接进来或宣称已修复；独立基线修正需要另行明确范围。
- [ ] 按最窄到相关回归运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py tests/semantic_retrieval/full_data/test_factors.py tests/semantic_retrieval/full_data/test_audit.py -q
git diff --check
```

预期新测试和受影响回归通过，`git diff --check` 无输出；不能事先填入通过数量。若工厂改动影响其他旧测试，仅扩展直接相关回归，避免无理由全仓扫描。

- [ ] 在 `docs/agent/uatvr-aqs.md` 写接口、软件测试常量与科学配置的区别、strict checkpoint 使用、全局 ID 映射、禁用等价及 Laya 无依赖说明；说明当前只提供软件 API，不提供无 gate 的真实训练入口。
- [ ] 完成报告包含实际 commit、文件改动、执行命令/退出码/测试数量、未执行 CUDA 项、源码复制等价检查、梯度证据、DDP 归约、恢复与 Laya 隔离结果。软件 CPU 成功不能写成科学提升或 GPU 资格通过。
- [ ] 提交 `docs: document isolated UATVR AQS integration and validation`。按照 verification-before-completion 技能，仅在看到实际输出后声明软件交付完成。

## 实施之后的研究执行边界

软件 API 完成后，正式 manifest 解析、controller/worker、生产容量 CUDA 资格及科学训练属于后续独立研究执行阶段，不能直接给现有 `full-data-retrieval-v1` 或 Laya worker 添加一个绕过冻结配置的 AQS 开关。

在开始该阶段的实验规划前，必须完整读取研究 README/00/01/02/07、通用 `experiments/contextual-risk-v2/study.yaml` 与 `registry.jsonl`，再建立适用的新 study；其配置须冻结 M、lambda、数据、基线、消融、预算与失败策略，并通过现行资格/研究 gate。按 reserve→提交→verify→运行→终态归档执行，test 仍 sealed。本文不给出可误用的生产启动命令，不继承旧 study 的例外授权。

## 覆盖自审与完成定义

| 设计要求 | 任务 |
|---|---|
| 整段视频、全局唯一身份、全部 caption 覆盖 | 1、2、5、6、7 |
| 难度×多样性、稳定 tie、边界值与 FP32 | 1、2 |
| 原有温度、完整 batch、MIL/KL、单方向辅助 | 3、4、5 |
| 停止选择梯度、真实参数更新、DDP 归约 | 3、5、6、7 |
| 无重复编码、时序有效帧、无 mask 副作用 | 1、4、5 |
| disabled/零权重及固定权重推理等价 | 4、5、8 |
| 严格配置、ID 映射和断点恢复 | 1、6、7 |
| Laya 隔离、vendor 保留、资格不外溢 | 0、4、8、后续边界 |

完成定义：本计划的 CPU 软件任务逐项有实际验证证据；Laya、vendor 和历史 study 未被修改；实现报告清楚区分软件可用与科学有效。任何未通过项目应如实列出，不将未运行测试标成已通过。
