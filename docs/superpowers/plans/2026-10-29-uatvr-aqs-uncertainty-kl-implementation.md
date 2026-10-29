# UATVR + AQS Uncertainty KL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复用 UATVR 概率头，将解析对称 Gaussian KL 融入 WTI，使不确定性参与完整检索损失、AQS 选择与推理评分。

**Architecture:** 在原有 AQS 外新增显式融合配置、纯 KL 算子、独立 core/model 和训练恢复适配器。关闭融合直接执行旧路径；启用时统一使用 `S'=WTI-eta*D`、`Z'=exp(logit_scale)*S'`，不修改 vendor 或引入 Laya。

**Tech Stack:** 现有 Python3.10、PyTorch2.1.2、NumPy、pytest；CPU tiny CLIP 和双 rank Gloo；不安装新依赖。

设计依据：[不确定性 KL 设计](../specs/2026-10-29-uatvr-aqs-uncertainty-kl-design.md)，提交 `7223d26`。旧训练代码基点 `5886cd1`。实施状态：Task1–7 已完成，验证代码提交为 `aa96c45`；最终结果见同日交付报告。示例用于说明接口，实际实现以源文件为准。

## Global Constraints

- 默认单 agent 按 `executing-plans` 顺序执行，未经用户选择不派生开发子 agent。已有隔离 worktree 可复用；执行前核对分支和本地改动，不覆盖用户文件。
- 实现范围是可验证的模型、训练 step、checkpoint 和离线配置接口；不直接运行或改造现有双卡 launcher 来启动新科学实验。
- 不修改 `full_data/upstream_uatvr/`、原有 `aqs/core.py`、`aqs/model.py`、`aqs/config.py`、`aqs/training.py` 和旧核心 AST 等价测试。旧模块作为不可变兼容路径；新能力用新文件承载。
- 复用现有均值/方差头，生产维度512；tiny fixture 可用其实际维度64。新增配置不创建可学习门、额外 encoder 或 pair-conditioned 方差头。
- 旧 `logsigma` 按采样路径解释为 log 标准差。新评分用 `2*clip(logsigma,-4,2)`，即 log 方差边界 `[-8,4]`；旧采样、MIL 和 legacy KL/VIB 公式与权重不变。
- `D=(KL(t||v)+KL(v||t))/(2*d)`，FP32 解析计算；`eta` 必须显式给出，没有科学默认值。`enabled=false` 或 `eta=0` 提前返回旧路径。
- 保留全 batch 双向 CE、原有已加权 MIL/KL、T2V AQS、唯一视频 ID、全 caption sampler 和一次全局特征 gather；融合与 AQS 两个开关相互独立。
- 新模式推理使用融合分数，不执行 AQS，不采样，不修改输入；DSL 显式关闭。旧模式保持旧随机数消耗和评分语义。
- 不 import 或读取 Laya 模型、processor、缓存、排名、教师信号及 monkey-patch；不写旧 Laya study。
- 不访问 test，不新增研究预留或 GPU 执行；不得把旧 AQS 完成记录或其他 study 双卡例外当作新评分结构的资格。
- 每个任务先写能识别错误行为的失败测试，再实施、窄范围验证并提交。仅验证通过后勾选；任务完成依据为实际代码、测试输出和交付报告。

## 0. 开始实施前的核对

执行工程师先读设计文档，核对 `git status --short` 与 `git log -3 --oneline`。检查适用 AGENTS、`research/README.md`、`00-north-star.md`、`01-evidence-ledger.md`、`02-dead-ends.md`、`07-protocol-constraints.md`；涉及实验规划或启动时，再读取对应 study/registry，不能依据本计划跳过 gate。

现有关键入口：

| 文件/符号 | 可复用内容 | 不能直接继承的假设 |
|---|---|---|
| `aqs/core.py::AqsCore` | 已核对的适配顺序、gather 与连续梯度保护 | observer 在概率分支之前，仅暴露 WTI |
| `aqs/model.py::AqsBaselineModel` | 构造、ID 与原 AQS 兼容分支 | `aqs.active=false` 会直接走旧 forward，不适用于融合仍开启时 |
| `aqs/selection.py` / `aqs/loss.py` | 无梯度选择、稳定选中分数 CE | 选择不能使新方差头获得梯度 |
| `full_data/models.py::BaselineModel` | CLIP 初始化与原有冻结策略 | `score` 当前只返回 WTI；新模式必须覆盖评分路径 |
| `aqs/training.py` | `video_id_mapping`、`model_recipe`、step 安全检查的行为参考 | 旧 checkpoint wrapper 固定 `uatvr-aqs-v1` 和旧源码列表 |
| `full_data/training.py` | `make_optimizer`、`save_state` / `restore_state` | 底层身份本身不认识新评分语义，需上层绑定 |
| `aqs/launch.py` / `scripts/prepare_aqs_training.py` | 旧运行的事实依据 | 固定双卡、旧 manifest，不得直接增加 eta 冒充合规新启动器 |

所有 Python 验证从仓库根目录执行，使用以下前缀。示例 `eta=0.1` 只用于软件 fixture，不构成科学配置：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest
```

## 文件结构与依赖顺序

以下路径均相对于仓库根目录；代码目录简称 `A=src/semantic_retrieval/aqs`，测试目录简称 `T=tests/semantic_retrieval/aqs/uncertainty`。表格列出完整路径，任务中的简称始终按此展开。

| 操作 | 路径 | 职责 |
|---|---|---|
| Create | `src/semantic_retrieval/aqs/fusion_config.py` | 不可变配置、严格 recipe |
| Create | `src/semantic_retrieval/aqs/fusion_math.py` | 分布转换、矩形对称 KL、融合算子 |
| Create | `src/semantic_retrieval/aqs/fusion_core.py` | 独立语义适配、概率参数、训练组件与无采样推理 |
| Create | `src/semantic_retrieval/aqs/fusion_model.py` | 新训练 forward、开关与 rank 上下文 |
| Create | `src/semantic_retrieval/aqs/fusion_training.py` | 新 step、身份、checkpoint 和恢复 |
| Create | `src/semantic_retrieval/aqs/fusion_runtime.py` | 纯配置解析、显式拒绝旧 manifest，不启动 worker |
| Create | `tests/semantic_retrieval/aqs/uncertainty/conftest.py` | 复用父目录 tiny_clip/paired_batch，提供融合模型工厂 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_math.py` | 配置、真实分布 oracle、数值/内存/梯度 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_core.py` | 适配特征、mask、推理、矩形分块与兼容 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_model.py` | 完整损失、梯度隔离、方差影响选样 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_resume.py` | step、身份、精确恢复、拒绝先于修改 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_distributed.py` | CPU Gloo 全局梯度、配置失配与恢复 |
| Create | `tests/semantic_retrieval/aqs/uncertainty/test_runtime_isolation.py` | 离线配置拒绝、依赖/写入隔离 |
| Create | `docs/agent/uatvr-aqs-uncertainty-kl.md` | 软件接口、配置、恢复和执行边界 |

依赖：Task1 → Task2 → Task3 → Task4 → Task5 → Task6 → Task7。每个任务的测试进入自己的 commit，不能先提交完整功能、最后补测试。

## Task 1：显式配置与正确的 Gaussian 匹配算子

**Files:** Create `A/fusion_config.py`、`A/fusion_math.py`、`T/test_math.py`。

**Interfaces:**

```python
@dataclass(frozen=True)
class FusionConfig:
    enabled: bool
    eta: float
    # active = enabled and eta > 0；两项构造参数都无默认值。
    # recipe()/from_recipe() 严格往返，未知键和常量变化拒绝。

@dataclass(frozen=True)
class ScoreGaussian:
    mean: torch.Tensor       # [N,d] FP32，有梯度
    logvar: torch.Tensor     # [N,d] FP32，有梯度

def score_gaussian(mean, logsigma) -> ScoreGaussian:
    """finite 2D 浮点、同 shape/device；logvar=2*logsigma.clamp(-4,2)。"""

def symmetric_kl(text: ScoreGaussian, video: ScoreGaussian) -> torch.Tensor:
    """[Q,V] FP32；Q/V 可不相等，d 与 device 必须一致。"""

def fuse_scores(wti, distance, logit_scale, config: FusionConfig):
    """返回 raw=S' 和 logits=exp(logit_scale)*S'；不负责 AQS。"""
```

接口块规定签名与合同，不是允许提交空函数的脚手架。生产维度由 model recipe 绑定，数学函数允许不同 d 的合法软件 fixture。

- [x] **1.1 写失败测试并运行** `T/test_math.py`：覆盖 bool 伪装数值、负/NaN/Inf eta、未知 recipe 字段、eta0、非单位方差、矩形集合和只改变方差造成的候选重排。关键独立 oracle：

```python
def test_kl_matches_distribution_oracle_and_keeps_gradients():
    import torch
    from torch.distributions import Independent, Normal, kl_divergence
    from semantic_retrieval.aqs.fusion_math import score_gaussian, symmetric_kl
    mu_t = torch.tensor([[0.2, -0.1]], requires_grad=True)
    mu_v = torch.tensor([[0.1, 0.4], [0.3, -0.5]], requires_grad=True)
    a_t = torch.tensor([[0.4, -0.3]], requires_grad=True)
    a_v = torch.tensor([[0.2, 0.1], [-0.2, 0.3]], requires_grad=True)
    t, v = score_gaussian(mu_t, a_t), score_gaussian(mu_v, a_v)
    qt = Independent(Normal(mu_t.double()[:, None], a_t.double()[:, None].exp()), 1)
    qv = Independent(Normal(mu_v.double()[None], a_v.double()[None].exp()), 1)
    oracle = (kl_divergence(qt, qv) + kl_divergence(qv, qt)) / 4
    actual = symmetric_kl(t, v)
    torch.testing.assert_close(actual.double(), oracle, atol=1e-5, rtol=2e-4)
    grads = torch.autograd.grad(actual.sum(), (mu_t, mu_v, a_t, a_v))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
```

初次应 FAIL 于新模块不存在或错误参数转换，而不是 fixture/环境错误。

- [x] **1.2 实现配置**，recipe 固定字段：`schema=uatvr-aqs-kl-v1`、enabled、eta、`metric=symmetric_diagonal_gaussian_kl`、`normalization=per_dimension`、`parameterization=legacy_logstd_x2`、`logvar_min=-8.0`、`logvar_max=4.0`、`dtype=fp32`、`legacy_vib=sample_mean_logsigma_unmodified_v1`。构造不接受 schema/边界的任意覆盖；from_recipe 逐项验证。eta0 是明确的兼容退化模式。

- [x] **1.3 实现算子**，输入先校验再转 FP32；不在函数内重新归一化旧头均值，不读取 ID，不采样，不 detach。核心计算采用矩阵分解，避免 autograd 隐含保存 `[Q,V,d]`：

```python
# t/v 已验证，mean/logvar 为 [N,d]；该分解与设计的对称 KL 等价。
mt, mv, lt, lv = t.mean, v.mean, t.logvar, v.logvar
wt, wv = (-lt).exp(), (-lv).exp()
d = mt.shape[1]
ratio = lt.exp() @ wv.T + wt @ lv.exp().T - 2*d
mean_t = ((mt.square()*wt).sum(1)[:, None]
          + wt @ mv.square().T - 2*(mt*wt) @ mv.T)
mean_v = ((mv.square()*wv).sum(1)[None, :]
          + mt.square() @ wv.T - 2*mt @ (mv*wv).T)
distance = (ratio + mean_t + mean_v) / (4*d)
if not torch.isfinite(distance).all() or (distance < -1e-5).any():
    raise ValueError('invalid symmetric Gaussian KL')
distance = distance.clamp_min(0)  # 只吸收预先界定的 FP32 负舍入误差。
```

边界附近与近相同分布要单独测试。若此分解不能满足既定容差，改进运算顺序/稳定计算，不能调大容差掩盖问题。显著错误拒绝，不用 `nan_to_num`。

- [x] **1.4 运行并通过** `T/test_math.py`：增加相同分布/转置对称、裁剪边界与命中梯度、错误 shape/dtype/device、有限极端值与 NaN/Inf；使用 `torch.autograd.graph.saved_tensors_hooks` 检查不会保存全量三维 pair 特征。数学比较 `atol=1e-5, rtol=2e-4` 预先固定。
- [x] **1.5 提交** `feat: add explicit Gaussian KL fusion configuration and scoring`，只暂存本任务文件。

## Task 2：独立 core，复用概率头与语义适配

**Files:** Create `A/fusion_core.py`、`T/conftest.py`、`T/test_core.py`。

**Interfaces:**

```python
@dataclass(frozen=True)
class PreparedModal:
    pooled: torch.Tensor     # 语义适配后的原有池化特征
    tokens: torch.Tensor     # 含原有聚合 token
    mask: torch.Tensor
    real_count: int          # 排除聚合 token 的原始槽位数

@dataclass(frozen=True)
class FusionComponents:
    wti: torch.Tensor        # [Q,V]，保留梯度
    distance: torch.Tensor   # [Q,V]，保留梯度
    raw: torch.Tensor        # [Q,V]，保留梯度
    logits: torch.Tensor     # [Q,V]，保留梯度
    mil: torch.Tensor        # 已加权；训练时有效
    kl: torch.Tensor         # legacy VIB 已加权；训练时有效
    representation: torch.Tensor  # [V,d]，detached，用于 AQS
    stats: dict              # 只含 detached 标量诊断

class FusionCore(AqsCore):
    # 复用 inherited 参数和初始化，不注册额外网络权重。
    def prepare_modalities(self, text, video, *, gather: bool):
        """返回 PreparedModal 文本与视频；仅训练可 gather。"""
    def distribution_parameters(self, modal: PreparedModal, *, modality: str):
        """返回 {'mean': tensor, 'logsigma': tensor}，不采样。"""
    def score_prepared(self, text: PreparedModal, video: PreparedModal):
        """返回 wti、distance、raw、logits 的 dict。"""
    def fusion_components(self, text, video) -> FusionComponents:
        """仅训练：完成一次适配/gather、评分、旧采样与 MIL/VIB。"""
```

- [x] **2.1 写失败测试**：对同一 tiny 模型复制状态，比较原 core 与新 core 的语义 token、pool 和裸 WTI；检验输入 mask/token 未修改。概率参数应与原头在相同 prepared 特征上的均值/logsigma 一致，不要求改变过 mask 的旧 singleton 错误成为正确 oracle。
- [x] **2.2 实现显式特征准备**：从 `AqsCore._loose_similarity` 的 seqTransf 前半段移植原有 extra token、位置编码、Transformer、残差、特征 all-gather 和归一化顺序到新类。训练仅 gather 一次，保留 `_ContiguousBackward` 的梯度布局保护。只支持新模式 `seqTransf`，其他 header 显式拒绝。

文本 pooling 传入掩码副本；若发现 singleton helper 原地写入，使用本地纯实现，不改 vendor、不导入 Laya。真实帧表示继续调用旧 `pooled_frames`，不混入聚合 token。

- [x] **2.3 实现无采样参数计算**：视频使用 `pie_net_video` / `uncertain_net_video`；文本使用 `pie_net_text` / `uncertain_net_text`，调用参数顺序完全按现有 `probabilistic_video/text`，均值继续 `l2_normalize`。不调用会自动采样的旧概率包装函数来做 eval 参数提取。
- [x] **2.4 实现共享评分与训练组件**：`score_prepared` 将两模态参数交给 Task1 算子。`fusion_components` 复用这些原始参数，仅在训练中调用旧 `sample_gaussian_tensors`，按原顺序计算 MIL/VIB 并只乘权一次。选择器获得融合 raw，不再使用旧的 pre-probability WTI observer 值。
- [x] **2.5 实现 eval 的 `_loose_similarity` 适配**：调用相同 prepare/score，返回 tensor logits；沿用 upstream `get_similarity_logits` 的 eval `(logits,{})` 约定。训练旧入口若被误调用新 core 也须得到合法训练三元组；正式新模型使用有类型的 `fusion_components`。

此方法的参数和 `training_observer` 关键字须兼容 `AqsCore`。融合未启用时，直接向 `super()._loose_similarity` 转发全部原参数和 observer，不进入新 prepare、pooling 或概率计算；否则父类的旧 `training_components` 仍会间接执行新算法，破坏禁用等价性。
- [x] **2.6 运行并通过** `T/test_core.py`：独立参数 oracle、旧 MIL/VIB 在相同输入/参数/RNG 下的数值、无新增参数键、eval monkeypatch 采样函数后仍成功、连续调用不污染输入。初期不实例化生产容量模型，不占用 GPU。
- [x] **2.7 提交** `feat: add explicit uncertainty fusion core without changing legacy UATVR`。

## Task 3：新模型训练路径与兼容退化

**Files:** Create `A/fusion_model.py`、`T/test_model.py`；Modify `T/conftest.py`。

**Interfaces:**

```python
class FusionAqsModel(AqsBaselineModel):
    """额外要求显式 fusion 配置，保留旧模型输入/输出合同。"""
    # 构造参数与旧模型相同，额外要求 fusion: FusionConfig，无默认值。
    # _uatvr_core_class() 返回 FusionCore。
    # forward 输入仍是 input_ids/text_mask/video/frame_mask/video_ids。
    # score(text, video) 返回矩形 logits tensor。
    # validate_rank_context() 同步旧 AQS 身份及完整 fusion.recipe()。
```

`T/conftest.py` 提供 `fusion_factory(tiny_clip, *, aqs, fusion, mapping='a'*64)`，构造 temporal_layers1 的新模型；父目录 `tiny_clip`、`paired_batch` 直接复用，不创建第二份 CLIP fixture。

- [x] **3.1 写失败测试**：分别测试四组开关：融合关闭/开启 × AQS 关闭/开启。特别覆盖 `FusionConfig(True,0)` 兼容路径，以及融合开启而 `aqs.weight=0` 时主损失/推理仍含 KL。
- [x] **3.2 实现构造与 rank 上下文**：在父构造可能调用虚方法前设置并集体验证普通配置；完成父构造后把配置绑定到 core。不要给旧模型写 monkey-patch。同步上下文中加入完整融合 recipe，防止仅比较 eta 漏掉转换/边界语义。支持建模后才初始化 process group 的再次验证，配置改变必须拒绝。
- [x] **3.3 实现 forward**，关闭融合直接调用父类旧路径；启用则先取得 Task2 `FusionComponents`，验证全局 ID、shape 和有限性。复用旧选择器及 loss，核心组合为：

```python
target = torch.arange(len(global_ids), device=components.logits.device)
contrastive = (F.cross_entropy(components.logits, target)
               + F.cross_entropy(components.logits.T, target))/2
if self.aqs_config.active:
    selected = select_negatives(components.raw.detach(), components.representation,
                               global_ids, self.aqs_config.negatives)
    auxiliary = selected_t2v_loss(components.logits, selected)
else:
    auxiliary = components.logits.sum()*0
loss = contrastive + components.mil + components.kl + self.aqs_config.weight*auxiliary
```

返回旧字段 `loss/contrastive/mil/kl/logits`，以及 `aqs/aqs_stats/fusion_stats`；仅新启用路径增加这些诊断，旧兼容路径的返回结构不变。diagnostics 不保存有梯度的大张量。

- [x] **3.4 验证真实梯度**：测试仅对 `out['aqs']` 或 `out['contrastive']` 调用 `autograd.grad`，不反传 MIL/VIB，检查 `uncertain_net_text.fc`、`uncertain_net_video.fc`、两侧 PIE 均值头、CLIP projection 和 temporal 参数的有限非零梯度；原有冻结参数继续无梯度。测试 fixture 避免使方差落到裁剪区外。
- [x] **3.5 构造只改变方差的行为测试**：保持 WTI 和所有均值固定，交换某个候选分布的方差，证明 `distance`、融合分数、对应 AQS 第一项能够改变。固定候选 ID，用明确非平局分数，不能只检验输出 shape 或全行常量改变。
- [x] **3.6 验证旧路径等价**：新/旧模型严格复制相同 state，重置同一 RNG，比较 logits、所有损失、选择、参数梯度、返回键和前后 RNG，要求逐位一致。用 monkeypatch 禁止进入新 KL 算子证明 eta0 的提前返回；不修改旧 AST regression。
- [x] **3.7 运行并通过** `T/test_model.py` 与旧 `tests/semantic_retrieval/aqs/test_core_model.py`，再提交 `feat: connect fused KL scores to AQS and retrieval gradients`。

## Task 4：推理合同与矩形分块一致性

**Files:** Modify `A/fusion_core.py`、`A/fusion_model.py`、`T/test_core.py`。

**Consumes:** Task2 的 `prepare_modalities` / `score_prepared` 和 Task3 模型。
**Produces:** 新模式可直接使用 `model.eval(); model.score(text,video)` 的全图库评分，不依赖 ID/对角配对；旧模式仍执行旧 `score`。

- [x] **4.1 写失败测试**：构造 Q3/V2 的异长文本和不同 frame mask。比较一次评分、逐条 query/video 的 singleton 分块、候选排列还原、连续调用两次与输入深拷贝。另测 Q1/V1 和相同模态作为不同候选块时的分布参数一致性。
- [x] **4.2 实现新模式 score**：取适配后的参数解析计算，不请求 AQS，不调用任何随机采样函数。保留原来的 RNG 保护约定，拒绝 training 状态下的 cross-product score。评分公式必须复用 Task2，不能在 eval 另写一个不同的 KL。
- [x] **4.3 核对分块开销**：一次 prepare 中每个模态头调用一次，不逐 pair 重复编码；chunk 之间数值一致。当前 evaluator 会重复准备部分 gallery，第一版不新增缓存来优化；需记录此成本，不能将未实现缓存写成已优化性能。
- [x] **4.4 运行并通过** `T/test_core.py`、`T/test_model.py`。eval 分块允许 `atol=2e-6, rtol=2e-4`，旧禁用等价仍用0容差。所有 CPU/CUDA RNG（可用时）、Python/NumPy RNG 都不得被新 eval 消耗。
- [x] **4.5 提交** `feat: ensure deterministic rectangular KL fusion inference`。

## Task 5：训练 step、版本化身份与精确恢复

**Files:** Create `A/fusion_training.py`、`T/test_resume.py`。

**Interfaces:**

```python
@dataclass(frozen=True, init=False)
class FusionBinding:
    """构造参数：base_identity, base_recipe, id_mapping_sha256。"""
    # 与旧 CheckpointBinding 一样保存 canonical bytes，避免嵌套可变字典。
    # 构造：base_identity, base_recipe, id_mapping_sha256。

def fusion_recipe(model) -> dict:
    """原 model_recipe + fusion.recipe()；含 AQS 配置和评分身份。"""

def fusion_step(model, network, optimizer, batch) -> dict:
    """有限标量：原 loss 项/AQS 项 + fusion diagnostics。"""

def save_fusion_state(path, model, optimizer, binding, progress, store):
    """调用 full_data.training.save_state；专用 .fusion.json sidecar。"""

def restore_fusion_state(path, model, optimizer, binding) -> dict:
    """集体校验完整载荷后，才恢复模型、优化器和各 rank RNG。"""
```

- [x] **5.1 写 step 与恢复失败测试**：两次 tiny 更新，保存 checkpoint，再创建新模型/新优化器进行恢复。比较下一步完整标量、模型参数、优化器 fingerprint 与 RNG；包含 AQS 开启和 AQS 关闭但融合开启。
- [x] **5.2 实现专用 step**：沿用旧 `aqs_step` 的安全行为，在新文件中显式执行网络归属/optimizer 参数身份核对、zero_grad、forward、backward、集体梯度检查、clip_norm1、optimizer.step、logit_scale 上限与更新后有限性检查。不可复用旧返回结构而丢弃融合诊断；不得只存 `last_stats` 这种隐藏可变状态。

原有 loss 分解与 diagnostic 返回口径：

```python
# output 来自同一次网络 forward；诊断均在新模型中明确提供。
result = {k: float(output[k].detach())
          for k in ('loss', 'contrastive', 'mil', 'kl')}
result['aqs_active'] = float(model.aqs_config.active)
result['aqs'] = float(output['aqs'].detach()) if model.aqs_config.active else 0.0
result['aqs_weighted'] = model.aqs_config.weight*result['aqs']
result['fusion_active'] = float(model.fusion_config.active)
if model.aqs_config.active:
    result.update(output['aqs_stats'])
else:
    result.update(mean_negatives=0.0, mean_difficulty=0.0,
                  mean_uniqueness=0.0, empty_rows=0.0)
if model.fusion_config.active:
    result.update(output['fusion_stats'])
```

原 intended-gradient 检查保留；融合 variance head 的新梯度链由 Task3 隔离测试确认，不能用混合总 loss 的非零梯度替代。

- [x] **5.3 实现新身份**：完整 recipe 绑定 `architecture=uatvr-aqs-kl-v1`、Task1 常量、eta、模型/可训练参数、ID/sampler、精度、world size、source SHA 和输入身份。源码列表显式覆盖本计划所有新 A 文件，以及实际依赖的旧 AQS/基线/概率函数；不得只哈希原有六个文件。可利用准备时的已跟踪 source inventory，但 checkpoint wrapper 也须校验当前源字节。
- [x] **5.4 实现独立 sidecar**：使用原底层 checkpoint schema 和 `ArtifactArrayStore` 外置机制，专用 `.fusion.json` 记录评分配置/派生身份/源文件哈希。先比对所有 rank 身份，再进行独占发布。JSON 不嵌入参数或逐查询数组；绑定 canonical bytes 时保持有限 JSON 值约束。
- [x] **5.5 实现严格恢复**：eta、转换、边界、旧 VIB 语义、ID、源码、输入、world size 或 sidecar 任一不符，在参数/优化器/RNG 修改前拒绝。旧 `.aqs.json` 不自动视为新 `.fusion.json`。本计划不增加 warm-start 导入功能，只明确拒绝误用；未来显式 warm-start 必须另记录 lineage 和重置语义。
- [x] **5.6 运行并通过** `T/test_resume.py` 与旧 `tests/semantic_retrieval/aqs/test_training_resume.py`。覆盖两个开关的四种组合，包括均关闭时旧 forward 不返回 aqs/fusion_stats 的情况；测试外来 optimizer、sidecar 缺失/篡改、配方嵌套修改、旧 checkpoint、eta 同键异值。失败前后模型/optimizer/RNG fingerprint 必须一致。
- [x] **5.7 提交** `feat: bind KL fusion training checkpoints to scoring semantics`。

## Task 6：CPU 双 rank 全局梯度与失败传播

**Files:** Create `T/test_distributed.py`；Modify `A/fusion_core.py`、`A/fusion_model.py`、`A/fusion_training.py` 中被本任务发现的必要错误。

**Consumes:** Tasks1–5。使用 `torch.multiprocessing.spawn`、Gloo、临时 FileStore，复用旧 `tests/semantic_retrieval/aqs/test_distributed.py` 的组织方式，不修改它，也不将软件 rank 数当作 GPU 科学资格。

- [x] **6.1 写全局梯度失败测试**：两 rank 各取同一全局 batch 的一半；参考模型单进程处理同一全局 batch。fixture 固定参数与输入，并把实际存在的 dropout 设为0，排除不同 batch 分片的随机掩码干扰。仅对 `contrastive + lambda*aqs` 反传，隔离 legacy 采样损失。

```python
# 两种执行都以同一批完整候选和同一评分配置计算。
auxiliary_objective = out['contrastive'] + model.aqs_config.weight*out['aqs']
auxiliary_objective.backward()
# rank0 保存 CPU 梯度、候选 ID 与分数到 pytest 临时目录。
# 父进程对照 serial oracle，包含两侧 variance/mean 头和 encoder projection。
```

- [x] **6.2 检验同步与布局**：分数和 ID 的全局顺序一致；不重复 all-gather；DDP 不额外乘除 world size。旧 `_ContiguousBackward` 保护要覆盖新路径，不能因串行 loss 正确而遗漏跨 rank 梯度错误。梯度比较使用 `atol=2e-6, rtol=2e-4`。
- [x] **6.3 写并实现集体拒绝**：一 rank 配置 eta 不同、配置字段变化、ID 重复、坏本地 mask、非有限概率参数、外来 optimizer。局部失败必须在其他 rank 进入下一次内部 gather 前通过 `_collective_call` 同步；不得把整个含不同数量 collectives 的 forward 外面简单包一层来冒充安全。为 spawn 测试设置有限超时，所有子进程退出，不留等待进程。
- [x] **6.4 写并通过双 rank 新模型恢复测试**：各 rank 的 RNG 独立保存/恢复，下一步全模型和优化器状态一致。关闭融合模式仍运行旧 DDP 回归。
- [x] **6.5 运行并通过** `T/test_distributed.py`，再运行旧 `tests/semantic_retrieval/aqs/test_distributed.py`；提交 `test: verify KL fusion global gradients and collective recovery`。

## Task 7：离线配置入口、隔离与交付

**Files:** Create `A/fusion_runtime.py`、`T/test_runtime_isolation.py`、`docs/agent/uatvr-aqs-uncertainty-kl.md`。

Modify `A/fusion_core.py`、`A/fusion_training.py`、`T/test_resume.py`，仅补全本任务 scalar 诊断和对应验证；不改动旧训练模块。

**Interfaces:**

```python
@dataclass(frozen=True)
class FusionRuntimeSpec:
    fusion: FusionConfig
    aqs: AqsConfig
    model_recipe_json: bytes    # 已验证且 canonical 序列化的 model recipe
    sampler_recipe_json: bytes  # 已验证且 canonical 序列化的 sampler recipe
    source_sha: str
    input_sha256: str
    world_size: int
    dsl: bool
    test_authorized: bool

def fusion_spec_from_recipe(recipe: dict) -> FusionRuntimeSpec:
    """纯解析；要求新 schema、显式 eta、DSL/test_authorized false，无启动副作用。"""
```

此接口只能声明配置有效，不能声明研究授权或 GPU gate 通过。旧 manifest、缺少 eta、新 schema 内携带旧 scoring 身份、未知字段和 test 请求均拒绝。

输入对象顶层字段固定为 `schema/fusion/aqs/model/sampler/identity/dsl/test_authorized`，schema 为 `uatvr-aqs-kl-software-recipe-v1`。fusion 与 aqs 分别调用严格 from_recipe；model 字段沿用旧 `model_recipe` 的准确键集与维度约束。sampler 精确包含 `policy/global_batch/seed/epochs`，policy 固定 `unique_video_all_captions_v1`，global_batch/epochs 为正整数、seed 为非负整数。identity 精确包含 `source_sha/input_sha256/world_size`，分别要求40位十六进制、64位十六进制和正整数，且 global_batch 可被 world_size 整除。dsl/test_authorized 都必须是布尔 false；不把字符串 false 当作布尔值。解析器逐层拒绝未知键，model/sampler 验证后存 canonical bytes，防止调用者之后改写输入字典。

- [x] **7.1 写失败测试**：原 manifest 不能通过新解析器；显式新配置可往返；eta 和评分语义变化导致新身份变化。monkeypatch `torch.cuda.init`、数据加载和写入函数为抛错，证明解析无副作用。
- [x] **7.2 实现纯解析与文档示例**：示例只展示 fixture `FusionConfig(enabled=True,eta=0.1)` 的模型构建和 step/checkpoint API，不提供绕过研究 gate 的生产 torchrun 命令。列出如何显式关闭融合、错误 recipe 的拒绝信息、旧 checkpoint 的不兼容原因和新评分缓存身份。
- [x] **7.3 实现新模式依赖隔离测试**：新进程屏蔽所有 Laya/外部 reranker import，拒绝打开其模型/缓存；验证 tiny 模型 forward、eval、step 和 checkpoint 仍成功，写入仅在 pytest 临时目录。测试必须实际执行新 active 分支，不能仅沿用旧隔离测试对六个旧文件的静态扫描。
- [x] **7.4 完成 scalar 诊断**：两侧 log 方差上下界命中率/分位数、WTI/KL/etaKL 的离散程度及正负例差、排名改变率/selected ID 改变率、variance/mean 参数梯度与更新量。诊断对比使用 detached 数据和同一批候选，不参与训练目标；额外逐查询数组需要外置。各 rank 的 scalar 以 rank 标识保留，不能仅输出 rank0 后宣称全局正常。
- [x] **7.5 按窄到宽运行最终回归**：先 `T/test_runtime_isolation.py`，通过后运行下节一次全套相关验证。修复仅限新能力所需范围；不以修改历史测试的预期来让禁用兼容通过。
- [x] **7.6 写交付报告并提交** `docs: document verified KL fusion software APIs and boundaries`。报告记录实际测试数、命令、时间、源码 commit、未验证的生产 GPU/科学项；不得提前写“训练成功”或“Recall 提升”。

## 最终验证命令与预期

执行者使用上述统一 CPU 前缀，依次运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest tests/semantic_retrieval/aqs/uncertainty -q --tb=short
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest tests/semantic_retrieval/aqs tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py -q --tb=short
git diff --check
git diff 7223d26 -- src/semantic_retrieval/full_data/upstream_uatvr src/semantic_retrieval/aqs/core.py src/semantic_retrieval/aqs/model.py src/semantic_retrieval/aqs/config.py src/semantic_retrieval/aqs/training.py scripts/prepare_aqs_training.py src/semantic_retrieval/aqs/launch.py
```

预期：新增与旧相关测试全部通过；diff 检查无空白错误；最后的旧代码 diff 为空。测试数由实际输出填写交付报告，不预填虚构数量。若无变动或未解决失败，不额外反复运行全仓库测试。

## 设计覆盖与完成定义

| 设计要求 | 实施任务 |
|---|---|
| 两类 KL、log 标准差转换、FP32、内存边界 | Task1、Task2 |
| WTI 后/AQS 前融合、选样与 loss 使用相同分数 | Task2、Task3 |
| 原 MIL/VIB 不变，冻结策略和真实梯度 | Task2、Task3、Task5 |
| 新推理、singleton/mask、矩形和分块不变性 | Task2、Task4 |
| eta0 和禁用退化，AQS 独立开关 | Task3、Task4 |
| 评分身份、全部源码绑定、新模型恢复 | Task5、Task6 |
| rank 配置、梯度布局、失败传播 | Task3、Task6 |
| 离线启动配置拒绝、Laya 隔离、诊断 | Task7 |
| 对照归因、研究资格、test sealed | 设计文档§11–12及下述执行边界；本轮不实施科学实验 |

完成意味着新评分、训练和恢复 API 的软件合同通过上述验证并有独立提交；不意味着新结构取得 GPU 资格、研究授权、性能收益或校准能力。所有实现 checkbox 只有对应代码与验证完成才可勾选。

## 软件完成后的研究执行边界

本计划不修改研究生命周期 schema，不创建新 study，不 reserve-phase，不自动续训，不重用旧 AQS 双卡 controller 启动新架构，不触碰旧产物或 test。生产 launcher、科学 eta 候选与选择规则、匹配对照、GPU 拓扑/资格、预算、恢复策略和研究 gate 必须在未来适用控制面中形成独立、可审阅的执行方案。

未来启动须先读当时适用的研究文档与 study/registry，完成归档预留并提交 reserved 记录，绑定干净 source SHA、完整输入/命令/配置/manifest；worker 期间不改受跟踪文件，终态核验后按规则追加 registry 和受跟踪摘要。当前规则的四卡要求与各 study 的双卡例外保持各自适用范围，本计划不授予新例外。

## 本文档自审

任务覆盖设计§1–13；接口名在相邻任务中一致；新类与旧评分身份分离；没有将构造概率头等同于不确定性有效，也没有只测试 legacy 辅助梯度。eta 的科学选择明确留给后续注册，软件 fixture 的0.1不作为生产默认。实现文件、验收对象、拒绝条件、测试命令和提交边界均已列出。


## 实施修订记录（2026-10-29）

- 新增测试目录 `__init__.py`，解决历史 `test_model` 模块同名的 pytest 收集冲突。
- 数值验证发现矩阵展开在小方差、近相同分布时会消减。保留快速矩阵路径，使用基于 FP32 舍入界的判据选择近零 pair，以有界二维 scratch 重算，并采用解析一阶 backward；不保存完整 pair×dimension 张量、不放宽原容差。该稳定回退不支持高阶导数。
- 独立审查指出底层 restore_state 的本地验证后立即修改状态不满足多 rank 拒绝合同。因此新模块显式复用其校验算法，将完整载荷校验与应用分成两阶段；所有 rank 通过只读阶段后再恢复，旧通用 loader 和 schema 保持不变。
- 添加 legacy 采样和 MIL/VIB 有限性检查、集体最终 loss 检查，避免评分已裁剪但旧采样仍溢出时只让单 rank 退出。
- 补充外层 autocast 下 KL 的 FP32 验证、低维到512维的近零/梯度 oracle，以及 rank-local checkpoint 读取失败后所有 rank 状态不变的测试。
- 相关回归最终结果：147 passed，1 项已有 TypedStorage 弃用 warning。未运行 GPU 资格或新科学训练。
