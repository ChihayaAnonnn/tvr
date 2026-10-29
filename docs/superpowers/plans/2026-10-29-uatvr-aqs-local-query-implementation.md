# UATVR + AQS Local Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现每帧 9 个重叠窗口的局部语义 query，将局部匹配加入 UATVR + AQS 的训练与检索评分，保留原路径。

**Architecture:** 显式提取同一次 CLIP 前向的 CLS 和 patch；9 个 query 分别读取自己的 3×3 窗口，形成局部 token。复用原 AQS core 的全局 logits 和 legacy 损失，在新模型适配层加入局部分数；新增独立配置、软件训练 step 和精确恢复绑定。

**Tech Stack:** 现有 Python、PyTorch、pytest，CPU tiny CLIP 与双 rank Gloo；不安装新依赖。

依据：[局部 Query 设计](../specs/2026-10-29-uatvr-aqs-local-query-design.md)，提交 `6abf6f8`。实施基点同为 `6abf6f8`，后续仅文档提交不改变该代码基点。Task 1–6 已完成，代码及审查修复提交为 `0f4ec65`；最终 CPU 回归 203 passed。详见 [交付报告](../reports/2026-10-29-uatvr-aqs-local-query-implementation.md)。软件实现不表示实验获准。

## Global Constraints

- 默认单 agent 顺序实施，复用当前隔离 worktree；执行前核对分支、适用 AGENTS 和用户改动，不自动派生开发 agent。
- 生产骨干固定 CLIP ViT-B/32：224×224、7×7 patch、维度 512，最多 12 帧。窗口中心 `{1,3,5}²`、尺寸 3×3、步长 2；9 个 query，每帧最多 9 个局部 token。
- 水平/垂直相邻窗口共享 3 个 patch，对角相邻共享 1 个；覆盖全部 49 个 patch。不增加邻接 query 一致性损失，不声称保证对象或跨帧语义连续。
- 一层 8-head cross-attention，dropout=0，残差与 LayerNorm，无 FFN；query 和 attention 参数跨帧共享。没有新时间模块、检测器、动态 mask、额外文本 encoder。
- 局部文本取全局适配前的 CLIP token，排除 BOS/EOS/padding，均匀词权重；β、τ_l 显式给出，无科学默认值。
- 保留原 WTI、时间建模、MIL、legacy VIB 公式和系数。`fusion.enabled=false` 且 `eta=0`；不修改已有 `fusion_*`，不依赖 Laya。
- 不修改 vendor、旧 AQS core/model/config/training/launcher 的行为。关闭局部功能或 β=0，在新参数创建前直接走旧模型工厂，保持旧随机状态与 checkpoint 结构。
- AQS 单位为负视频；不把正视频的未选窗口当负例。融合分数用于完整双向 CE、AQS hardness/辅助损失及推理。
- 本计划只交付 CPU 可验证的软件接口，不开发科学实验启动器，不创建实验预留，不执行 GPU、不读取 test。后续科学实验另行遵守研究 gate 与归档合同。
- 每项先写失败测试、确认失败，再实施、运行窄范围测试，验证通过后提交并勾选。示例代码说明关键实现，接口契约与验收条件共同约束最终代码。

## 0. 实施前核对与文件布局

执行前读取设计，检查 `git status --short`、当前 HEAD、适用 AGENTS。涉及新的研究方案或实验操作前，按 [research/README](../../../research/README.md) 完成强制前置阅读；涉及具体实验再读对应 study/registry。本计划不能继承其他 study 的双卡豁免。

以下均为拟新增文件，不改写旧模式。测试根目录统一为 `tests/semantic_retrieval/aqs/local_query/`。

| 文件 | 职责 |
|---|---|
| `src/semantic_retrieval/aqs/local_config.py` | 不可变配置、固定规范和严格 recipe |
| `src/semantic_retrieval/aqs/local_visual.py` | 无新骨干参数的 CLS/patch 双输出函数 |
| `src/semantic_retrieval/aqs/local_tokens.py` | 窗口索引、9-query attention、局部词 mask |
| `src/semantic_retrieval/aqs/local_scoring.py` | 分块词到局部 token 软匹配及融合 |
| `src/semantic_retrieval/aqs/local_model.py` | 新模型、旧路径工厂、训练/评估公共评分契约 |
| `src/semantic_retrieval/aqs/local_training.py` | step、parent 初始化、checkpoint 身份和恢复 |
| `src/semantic_retrieval/aqs/local_runtime.py` | 严格离线配方解析；不启动 worker |
| `tests/semantic_retrieval/aqs/local_query/conftest.py` | 7×7 网格的小型 CPU fixture |
| 同测试目录 `test_config_tokens.py`、`test_visual.py`、`test_scoring.py` | 配置、几何、视觉一致性、分数与梯度 |
| 同测试目录 `test_model.py`、`test_distributed.py`、`test_resume_runtime.py` | 模型集成、分布式、恢复与隔离 |
| `docs/agent/uatvr-aqs-local-query.md` | 开发完成后的实际 API、使用限制与恢复说明 |
| `docs/superpowers/reports/2026-10-29-uatvr-aqs-local-query-implementation.md` | 实际改动、验证证据和未验证边界 |

依赖顺序：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6。Task 6 完成前不得称为可恢复训练交付。

所有测试从仓库根目录执行。后文使用的命令前缀展开为：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest
```

各任务的首次测试预期因缺失新模块或断言失败而失败；只接受与本任务相关的失败原因。完成实现后对应命令须退出 0。测试中 `beta=0.2`、`tau=0.1` 仅为软件数值 fixture，不是训练推荐值。

## Task 1：严格配置、重叠窗口与局部 token

**Files:** 新建 `local_config.py`、`local_tokens.py`、`test_config_tokens.py`、`conftest.py`，路径按上表展开。

**Interfaces:**

```python
@dataclass(frozen=True)
class LocalQueryConfig:
    enabled: bool
    beta: float
    tau: float
    # active = enabled and beta > 0
    # recipe()/from_recipe() 记录并校验固定规范；未知键拒绝。

def window_indices(device=None) -> torch.Tensor:  # long [9,9]
    return torch.tensor([
        [7*r+c for r in range(a-1, a+2) for c in range(b-1, b+2)]
        for a in (1, 3, 5) for b in (1, 3, 5)
    ], dtype=torch.long, device=device)

def lexical_mask(text_mask: torch.Tensor) -> torch.Tensor:  # bool [B,L]
    positions = torch.arange(text_mask.shape[1], device=text_mask.device)
    lengths = text_mask.sum(-1)
    return text_mask & (positions[None] > 0) & (positions[None] < lengths[:, None]-1)

class LocalQueryTokens(nn.Module):
    def __init__(self, dim: int = 512): ...
    def forward(self, patches, frame_mask): ...  # [B,F,49,d], bool[B,F] -> [B,F,9,d]
```

`dim` 仅支持正整数且能被 8 整除；生产由模型强制为 512，fixture 可为 64。配置拒绝 bool 冒充数值、NaN/Inf、负 β、非正 τ、非 bool 开关。`recipe()` 必须含固定窗口版本 `grid7-window3-stride2-v1`、query数9、head数8、`uniform-lexical` 聚合和 epsilon `1e-6`；不可变常量被修改则拒绝。

- [x] **1.1 写失败测试并运行** `test_config_tokens.py -q`。覆盖完整覆盖率、精确 row-major 顺序、非相邻交集为零、相邻交集和配置严格往返。最小几何断言：

```python
w = window_indices()
assert w.shape == (9, 9)
assert set(w.flatten().tolist()) == set(range(49))
assert len(set(w[0].tolist()) & set(w[1].tolist())) == 3
assert len(set(w[0].tolist()) & set(w[3].tolist())) == 3
assert len(set(w[0].tolist()) & set(w[4].tolist())) == 1
assert not (set(w[0].tolist()) & set(w[2].tolist()))
```

- [x] **1.2 实现配置与 token 模块。** `LocalQueryTokens` 创建 `query: Parameter[9,d]`（正态初始化 std=`d**-0.5`）、`MultiheadAttention(d,8,dropout=0,batch_first=True)` 与 `LayerNorm(d)`。索引用 nonpersistent buffer，避免将确定性索引变成 checkpoint 可变状态。attention 核心如下：

```python
b, f, _, d = patches.shape
kv = patches[:, :, self.indices, :].reshape(b*f*9, 9, d)
q = self.query[None, None].expand(b, f, -1, -1).reshape(b*f*9, 1, d)
u, _ = self.attention(q, kv, kv, need_weights=False)
out = F.normalize(self.norm(q+u), dim=-1, eps=1e-6)
out = out.reshape(b, f, 9, d)
return out.masked_fill(~frame_mask[..., None, None], 0)
```

入口检查浮点、有限值、shape/device、bool mask、每视频至少一帧；`lexical_mask` 使用已经通过原文本 BOS/EOS/连续 padding 校验的 mask，并拒绝空词集合。不得添加相邻 query 相似性目标。

- [x] **1.3 增加可辨识行为测试。** 在 attention 输入层改变窗口0以外的 patch，窗口0输出保持一致；改变共享 patch 时允许两个相邻窗口改变。复制同一帧到两个时间位置，输出一致。对非退化随机输入反传，query 与投影均有有限、非零梯度；padding 帧输出为零。
- [x] **1.4 重跑窄测试并提交** `feat: add overlapping local query tokens`。fixture 使用 `ClipConfig(64,28,64,1,4,32,128,64,1,1,fixture=True)`，维持 7×7 网格；新 paired batch 图像改为28×28，不改父目录旧 fixture。

## Task 2：单次视觉前向暴露 CLS 和 patch

**Files:** 新建 `local_visual.py`、`test_visual.py`。

**Interfaces:** `visual_tokens(visual, images, *, video_frame) -> (cls, patches)`，输入 `[B*F,3,H,W]`，输出 `[B*F,d]` 与 `[B*F,49,d]`。函数只使用传入的原视觉模块参数，不持有、复制或重新注册骨干。

- [x] **2.1 写失败测试并运行** `test_visual.py -q`。在同一份 tiny 视觉权重上比较原 `visual(images,video_frame=F)` 与双输出 CLS 的逐元素一致性、梯度和 state_dict keys；检查49个patch的顺序，不允许仅复制CLS填充。
- [x] **2.2 实现双输出。** 显式按原 `VisualTransformer.forward` 的 2D 分支执行 conv1、展平、CLS拼接、位置嵌入、ln_pre、transformer、ln_post/proj；最后返回整序列切片，不再调用原 forward 第二次。核心末段：

```python
x = visual.transformer(x.permute(1, 0, 2), video_frame=video_frame)
x = visual.ln_post(x.permute(1, 0, 2)) @ visual.proj
return x[:, 0, :].float(), x[:, 1:, :].float()
```

显式拒绝 `linear_patch!='2d'`、非7×7网格、错误帧数/空间维度。输入 dtype 转换和原 `encode_image` 保持一致。所有 CLS、位置和投影操作保持原顺序；不使用 monkey-patch 或 hook 实现功能。
- [x] **2.3 验证调用次数与输入独立性。** 测试允许使用临时 spy 统计 conv1/transformer 各一次；非均匀 patch 输入产生不同 patch 输出；公共参数梯度与原 CLS-only 前向在相同 CLS 标量目标下相等。检查底层冻结参数集合未变。
- [x] **2.4 重跑本项及 Task1 测试并提交** `feat: expose CLIP patches in a single visual pass`。

## Task 3：局部评分与可微分分块

**Files:** 新建 `local_scoring.py`、`test_scoring.py`。

**Interfaces:**

```python
def local_scores(text_tokens, word_mask, local_tokens, frame_mask, *, tau,
                 query_chunk: int, video_chunk: int) -> torch.Tensor:
    # [Q,L,d], [Q,L], [V,F,9,d], [V,F] -> FP32[Q,V]
    # 训练使用 non-reentrant checkpoint 重计算每个分块；eval 无需。
    ...

def add_local_logits(global_logits, local_raw, logit_scale, *, beta):
    return global_logits + logit_scale.exp() * beta * local_raw
```

最后一个函数接收**已经缩放**的旧 logits；不得再对旧 logits 乘 scale。这样保留其梯度，无需除以 logit scale 再乘回。

- [x] **3.1 写失败测试并运行** `test_scoring.py -q`。手算 oracle 用 Python 循环逐文本、逐视频、逐有效词计算 `softmax(c/tau)·c`，对有效词取平均；验证矩形 Q≠V、mask、负相似度及温度变化。
- [x] **3.2 实现每块核心。** 先按有效 mask 置零无效特征并用 epsilon归一化，避免 padding 内容影响结果，再计算：

```python
# t:[q,L,d], o:[v,F*9,d], valid:[v,F*9]
c = torch.einsum('qld,vud->qvlu', t, o)
attn = (c/tau).masked_fill(~valid[None, :, None, :], -torch.inf).softmax(-1)
word_score = (attn*c).sum(-1)
score = (word_score*word_mask[:, None, :]).sum(-1) / word_mask.sum(-1)[:, None]
```

入口在 softmax 前拒绝空词、空视频、错维度、非有限输入和无效 τ/chunk。展平 frame mask 时每帧重复9次；FP32算分。双重循环分块，行列使用 `torch.cat` 连接，禁止 detached/NumPy拼接。启用梯度时用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 重算每块，避免所有 `[q,v,L,F*9]` 中间量同时被反向图保留；接受必须保留的最终 `[Q,V]` 分数矩阵。
- [x] **3.3 验证全部梯度。** 比较未分块 oracle 与多种chunk边界的输出、text/local/logit_scale梯度，FP64 oracle 建议误差 `atol=1e-6,rtol=1e-5`；CPU FP32按实测明确阈值，不放宽掩盖错误。padding值扰动、增加padding帧/词不改变结果。用 saved-tensor检查确认没有保存完整跨批四维相似度中间量。
- [x] **3.4 重跑窄测试并提交** `feat: add chunked local semantic matching`。

## Task 4：训练、AQS 与推理集成

**Files:** 新建 `local_model.py`、`test_model.py`。

**Interfaces:**

```python
def build_local_model(clip_config, clip_state, *, local, fusion, aqs,
                      id_mapping_sha256, **baseline_kwargs): ...

class LocalQueryAqsModel(AqsBaselineModel):
    # 仅由 active local 工厂创建；core 仍为原 AqsCore。
    def encode_text(self, input_ids, mask): ...
    def encode_video(self, video, mask): ...
    def forward(self, input_ids, text_mask, video, frame_mask, video_ids): ...
    def _logits(self, text, video): ...
    def validate_rank_context(self): ...
```

- [x] **4.1 写失败测试并运行** `test_model.py -q`。同一 seed 下新工厂 disabled/β0 与直接旧构造比较参数 keys、参数值、构造后的 RNG、前向损失及梯度；active模式即使 AQS关闭也必须保持局部分数。测试 fusion enabled 或 eta非零时拒绝，避免只检查 `fusion.active` 而放过冲突组合。
- [x] **4.2 实现工厂与编码。** 配置校验后，inactive直接构造 `AqsBaselineModel`；active先完整构造旧骨干，再创建 `local_queries`，确保新参数不改变骨干初始化。扩展 `encode_text` 字典加入 `local_tokens`（原tokens）和 `local_mask`；扩展 `encode_video` 使用 Task2单次前向得到原CLS `tokens` 和 Task1 `local_tokens`，原 `mask`不变。保留 `EncoderInputs` 与原空间/帧数校验，不跳过旧输入约束。
- [x] **4.3 实现训练合流。** 复用 `AqsCore.training_components` 返回的 `global_logits,mil,kl,global_raw,representation`；其 `global_raw` 已 detached，只能用于选样。新增特征独立通过 `gather_with_grad` 汇集一次，mask通过 `gather_metadata`一次，与旧 core 和全局 IDs 同 rank顺序。禁止对已经汇集的旧输入二次 gather。关键合流：

```python
local_raw = local_scores(global_text, global_words, global_local, global_frames,
                         tau=cfg.tau, query_chunk=qchunk, video_chunk=vchunk)
logits = add_local_logits(global_logits, local_raw, self.clip.logit_scale, beta=cfg.beta)
raw_for_selection = global_raw + cfg.beta * local_raw.detach()
target = torch.arange(len(logits), device=logits.device)
contrastive = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))/2
# 仅当 aqs_config.active 时调用原 select_negatives 和 selected_t2v_loss。
loss = contrastive + mil + kl + self.aqs_config.weight * aqs_loss
```

局部全局训练分支不能经旧 `AqsBaselineModel.forward` 的 AQS关闭早退漏掉新功能。未启用AQS时 `aqs_loss=logits.sum()*0`；视频唯一性及形状仍检查。selection沿用 `select_negatives(raw_for_selection, representation, global_ids, M)`，保持原表示多样性等既有策略；这里只替换困难程度用的分数。
- [x] **4.4 实现评估。** `_logits` 在eval时取原全局已缩放logits，加同一个局部算子输出，返回 `(logits,None,None)`；继承 `score` 的 RNG保护和有限值检查。不在eval执行AQS或gather。支持独立文本/视频缓存和矩形评分，不改变输入字典。
- [x] **4.5 增加集成行为测试。** 隔离局部CE反传，检查query/attention及允许训练的视觉参数有限非零梯度；仅对hardness选样无法反传。构造分数使局部项改变负视频顺序，验证损失选中的是同一融合矩阵。扰动padding不改变全模型分数；不把正视频窗口制造为负例；公共参数相同且前向开始前恢复相同RNG时，legacy损失保持旧值，不能把新增模块初始化消耗的RNG误当作公式变化。
- [x] **4.6 核对rank上下文并提交** `feat: integrate local scoring into UATVR AQS`。调用父 `validate_rank_context` 保持原 `_aqs_context` 结构不变；另存 `_local_context`，包含local/fusion/AQS/ID摘要和chunk规范，并在每次训练前检查，防止初始化后修改配置。父构造会调用 `validate_rank_context`，因此必须在 `super().__init__` 前设置无参数配置，不能提前创建新nn.Module。

## Task 5：分布式数值与失败行为

**Files:** 新建 `test_distributed.py`；仅修正 Task4新增文件中被测试证明有误的行为。

**Interfaces:** 复用现有 `runtime.distributed.gather_with_grad/gather_metadata`，生产全局 batch 等于各rank本地batch之和；不得新增自定义 backward 绕过数值核验。

- [x] **5.1 写CPU双rank Gloo失败测试并运行** `test_distributed.py -q`。在临时目录使用进程组和超时，统一微型权重/数据，比较分布式每rank loss与单进程global-batch参考，以及DDP平均后的局部参数和骨干梯度。明确全局loss被每rank计算的缩放语义，不能仅比较两rank相互一致。
- [x] **5.2 补充分支与拒绝用例。** 验证跨rank负视频可被选中、全局ID顺序和mask一致。重复ID按现有unique-video合同拒绝，不将其误选为负样本。β/τ/ID或rank shape失配、单rank空词/非有限输入要通过 `_collective_call` 等统一边界使所有rank有界失败，不得另一个rank挂在gather。
- [x] **5.3 验证并提交** `test: verify distributed local query gradients`。若数值未通过，不进入Task6，也不以CPU测试替代未来所需CUDA资格。

## Task 6：软件训练、身份绑定、恢复与交付

**Files:** 新建 `local_training.py`、`local_runtime.py`、`test_resume_runtime.py`、上表两份开发完成文档。

**Interfaces:**

```python
class LocalBinding(CheckpointBinding): ...
def local_recipe(model) -> dict: ...
def local_source_identity() -> dict: ...
def initialize_local_parent(path, model, *, expected_sha256) -> dict: ...
def local_step(model, network, optimizer, batch) -> dict: ...
def save_local_state(path, model, optimizer, binding, progress, store): ...
def restore_local_state(path, model, optimizer, binding): ...
def local_spec_from_recipe(recipe): ...  # 返回不可变LocalRuntimeSpec，不创建模型/目录/任务
```

- [x] **6.1 写恢复与配置失败测试并运行** `test_resume_runtime.py -q`。测试精确恢复与不中断两步训练的模型、优化器、进度和RNG相等；拒绝错误β/τ、窗口规范、架构ID、源码、ID表、优化器、损坏NPY引用、KL checkpoint，以及未知/缺失配方字段。拒绝后模型、优化器与RNG必须完全未改变。
- [x] **6.2 实现纯配方。** `LocalRuntimeSpec` 记录模型/base recipe、local/AQS/fusion、ID摘要、chunk尺寸、父初始化来源或resume身份，互斥 parent/resume。固定架构 `uatvr-aqs-local-query-v1`。生产必须512维224/32，fixture仅显式测试入口允许64维28/4；不允许科学配方带fixture标志。禁用/β0配方返回旧模型语义，不伪造新架构checkpoint。
- [x] **6.3 实现parent初始化。** 使用`weights_only=True`读取并校验内容哈希、已声明旧架构、全部公共keys/shape/dtype/finite；仅允许缺失精确的 `local_queries.*` 新参数，其他缺失/多余均拒绝。全量校验后一次加载公共状态，不恢复旧优化器/RNG/步数。返回来源哈希与公共参数摘要；不得使用无审计的 `strict=False`。新query保留本次显式初始化。
- [x] **6.4 实现新step。** 参考 `fusion_step` 的错误同步、有限loss/梯度与step前检查，但不导入KL评分实现；复用原 `make_optimizer` 的参数组规则，确认query/MHA/LayerNorm被纳入优化器。输出标量loss/CE/AQS/legacy损失、局部分数统计与有效token计数；不把逐query数组塞进JSON。
- [x] **6.5 实现绑定与恢复。** 参考现有训练/恢复事务的验证顺序，使用独立 `.local.json` sidecar，绑定新架构、全部配置、实际涉及的源码字节哈希、ID摘要、输入身份和可训练参数列表。新旧共有源码及新视觉/评分/runtime文件均纳入，不能只绑定local文件。复用底层安全路径、独占发布、NPY外置与引用哈希；不得复制旧架构字符串后冒充新恢复。

恢复阶段先只读校验sidecar、checkpoint内容、所有数组、全模型/优化器/RNG结构和rank状态；所有rank验证通过才修改内存。失败保留原状态；历史产物不覆盖。底层工具若需复用内部校验函数，应先核对其输入合同，不以导入`fusion_*`建立KL运行依赖。

- [x] **6.6 加隔离与依赖验证。** AST/导入测试确认新增模型链不import Laya或QueryMatch；离线解析不访问文件、创建目录或启动进程。disabled路径在禁止调用视觉双输出和query构造的spy下仍可运行。旧KL测试继续通过，禁止为通过测试改旧断言。
- [x] **6.7 更新实际API说明与交付报告并提交** `feat: add local query training and strict recovery`。报告记录命令、输出、源码提交、测试数量、CPU局限；仅勾选有实际验证证据的任务。

## 最终验证与停止条件

先运行各任务对应测试，再执行以下一次回归；失败时定位到新增行为，不扩展到无关仓库测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest tests/semantic_retrieval/aqs tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py -q --tb=short
git diff --check
git status --short
```

必须同时满足：几何重叠正确、单次视觉前向、局部梯度有效、融合scale只用一次、AQS关闭不漏局部分支、旧路径随机/数值兼容、单进程与Gloo梯度一致、恢复失败不修改状态、无Laya依赖。

设计覆盖映射：窗口/语义范围→Task1；单次编码→Task2；词mask/软匹配/内存→Task1/3；全局融合/负视频/推理→Task4；跨rank→Task5；KL隔离/旧路径→Task1/4/6；checkpoint与配方→Task6。未加入邻窗一致性损失或局部时序模块。

完成上述软件验证、提交代码与报告后停止。不自动启动科学训练、不选择β/τ的科学值、不开放test；正式容量CUDA数值、显存、吞吐与恢复资格需在另行授权的控制面归档，CPU验证不能证明效果提升或GPU训练资格。
