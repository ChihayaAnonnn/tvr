# Evidence-Conditioned Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现观测隔离、概率对比匹配、补帧收益预测与预算执行的可恢复软件闭环。

**Architecture:** 在现有CLIP、TemporalEncoder和PairMatcher组件上新增evidence子系统。匹配训练、转移编译和收益训练按固定快照分阶段执行；原pilot配置及行为保持不变。通用归档继续复用Scheme A，只增加显式的新配置分派和拒绝守卫。

**Tech Stack:** 仓库现有Python、PyTorch、NumPy、OpenCV、pytest、Gloo与Scheme A工具；不新增产品依赖。

**Spec:** [已批准设计v1](../specs/2026-10-10-evidence-conditioned-retrieval-design.md)，用户于2026-10-10回复“通过”。

状态更新（2026-10-10）：用户已确认计划并选择 Native 执行。任务1–11与独立审查修复已实现，实际验证见 [开发与资格审查记录](../reviews/2026-10-10-evidence-conditioned-retrieval-review.md)。下文复选框和代码块保留原实施步骤，不作为运行结果。独立科学授权、来源修订、预注册、分阶段 gate 和实际运行记录位于 `experiments/evidence-conditioned-retrieval-v1/`；软件实现成功不代表研究假设成立。

## Global Constraints

- 第一版固定候选集合C。其生成使用冻结粗检索器和所有方法共同可用的初始帧，不根据相关性标签插入答案。候选漏召回进入总体失败统计。
- 收益类别显式编码：`0=worsened, 1=unchanged, 2=corrected`。unknown由单独bool valid_mask表示。
- position_id始终是该帧在固定时间顺序帧池中的槽位。增加帧不改变旧帧的position_id。
- 冻结CLIP的text/frame编码；训练时序编码、PairMatcher及概率头。
- S2与S4不联合回传。未来联合训练需另行设计目标重编译及策略分布合同。
- 所有新路径训练和受保护运行采用FP32，保留禁用TF32和autocast的合同。
- 严格加载新路径checkpoint，不放宽为strict=False，不实例化历史UATVR架构兼容旧权重。
- 第一版不实现增量Transformer缓存，避免在双向时序注意力中错误复用旧上下文。
- 新研究不恢复SAP/CSD/A3，不改旧LOSS_NAMES、pilot arm或冻结study；不创建真实study，不预留真实run，不启动GPU科学工作。
- 所有测试只用CPU合成fixture和tmp_path中的临时归档；不得访问真实数据、历史checkpoint或sealed test。
- 研究选择、阈值、K与校准仅用validation；MSRVTT/MSVD test保持sealed。
- 超过1 MiB数值数组外置为已校验引用；不写遗留logs/ckpts。新科学路径的study与四卡资格授权不由本计划产生。

## Review Focus

1. 同canonical视频在两查询中具有不同观测：不得union观测或共享错误上下文；任务2的跨查询隔离测试。
2. 重排候选及分数并列：噪声按身份稳定，最终按canonical ID打破并列；任务3与任务8。
3. 短视频、重复帧和追加帧全部解码失败：有限停止、失败请求计费、初始全失败拒绝；任务2与任务7。
4. 某rank无标签而其他rank有标签：所有rank同步参与，按全局有效权重归一化；任务9的两rank Gloo测试。
5. 特征已落盘但动作未commit时崩溃：可核验恢复不能重复扣费或把未获取帧视为已观测；任务7与任务9。

## Execution Setup

执行获准后使用隔离worktree，按Superpowers using-git-worktrees创建 `codex/evidence-conditioned-retrieval` 分支。保留当前未跟踪MANIFEST.md、idea-stage/及其他用户改动；只提交每项任务列明的文件。

测试命令统一前缀：
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider
```
下文每项验证均给出完整前缀。CPU测试不代表四卡资格通过。

## File Map and Dependency Order

新增产品目录 `src/semantic_retrieval/evidence/`，新增测试目录 `tests/semantic_retrieval/evidence/`。任务1→2→3→4→5→6→7→8顺序推进；任务9接分布式与恢复，任务10接归档/CLI，任务11做CPU保护闭环与回归。不得在中间任务启用真实科学入口。

| 设计交付 | 实施任务 | 主要文件 |
|---|---|---|
| D1 观测合同 | 1–2 | types/config/views/frame_cache/encoding，现有temporal.py |
| D2 概率匹配 | 3–4 | sampling/matcher/objectives/model/training |
| D3 转移编译 | 5 | transitions/data |
| D4 收益与对照 | 6、8 | value/calibration/baselines/evaluation |
| D5 获取闭环 | 7 | policy/journal |
| D6 运行接线 | 9–11 | checkpoint/runtime/authorization/qualification，通用归档及CLI |

因sampling、encoding、checkpoint和journal具有独立状态/数值合同，从设计的较大模块中单独建文件；职责不增加。

---

### Task 1: 建立类型、严格配置与合成fixture（D1）

**Files:**

- Create: `src/semantic_retrieval/evidence/__init__.py`, `src/semantic_retrieval/evidence/types.py`, `src/semantic_retrieval/evidence/config.py`
- Create: `tests/semantic_retrieval/evidence/conftest.py`, `tests/semantic_retrieval/evidence/test_contracts.py`

**Interfaces:**

在types.py定义冻结dataclass：ResourceCounts(frame_requests, clip_frames, temporal_tokens, mc_elements)，四项为非负int；BudgetLedger(limits, spent, reserved)；CandidateObservation(canonical_id, content_sha256, frame_count, pool_frame_ids, observed_frame_ids, attempted_frame_ids)；ExecutionProfile(mc_samples, cost_weights)；ObservationState(query_id, gallery_version, candidates, view_levels, profile, budget, version)，提供state_sha256属性；EvidenceAction(action_id, state_sha256, candidate_id, frame_ids, position_ids, reservation, estimated_cost)；ActionOutcome(action_id, succeeded_ids, failed_ids, actual_counts, elapsed_seconds, feature_references)。
定义ObservedFeatures(features: EncodedFeatures, position_ids: Tensor, pair_indices: Tensor, query_offsets: Tensor, pair_ids: tuple[tuple[str,str],...], observation_keys: tuple)，PairPredictiveScores(loc, scale, features, ranking_log_probability, pair_ids, sampling_sha256)，EvidenceTargets(relevance, rich_scores, rich_valid, reference_sha256)，TransitionTargets(outcome_class, valid_mask, producer_sha256)，LossTerm(numerator: Tensor, denominator: Tensor)，ActionValueDistribution(logits)，AcquireDecision(action, ledger)和StopDecision(reason)。

config.py定义ViewConfig、SamplingConfig、StageTrainConfig、SplitBinding、EvidenceModelConfig(base: ModelConfig, variant, scale_floor)、EvidenceSystemConfig；实现load_evidence_config(path: Path) -> EvidenceSystemConfig和evidence_manifest_config(config, *, eval_split: str) -> dict。stage仅允许cpu_fixture/reference/matcher/transitions/value/calibration/evaluation；schema_version=1，evaluation.config_kind=evidence-retrieval-v1显式识别新路径；runtime可在纯单元fixture中为None，CLI拒绝。角色仅T_match/T_value/V_development/V_calibration/V_locked。科学配置无隐藏默认；父模型按阶段强制绑定。

新增fixture config_factory(stage="matcher", **overrides)，复用父目录model_config，把max_frames显式改为16，clip.fixture=True，不下载权重。state_factory(query_id="q0", canonical_ids=("a","b"), observed=(0,15), frame_count=16)使用hashlib固定合成哈希、levels=(4,8,16)、M=4、成本系数全1、limits=(32,32,1024,4096)，spent/reserved全0。fixture数值只属于测试。
配置字段固定：SamplingConfig(train_samples, eval_samples, temperature, seed, protocol)；StageTrainConfig(learning_rate, weight_decay, seed, scheduler_name, scheduler_steps, checkpoint_interval, trainable_prefixes, loss_weights)。EvidenceSystemConfig含schema_version/runtime/stage/synthetic/model/views/sampling/train/splits/index/parents/transitions/value/policy/resume，提供to_payload() -> dict。输入引用统一InputReference(path: Path, sha256: str)，未使用父项显式None，不能缺字段。config_factory中的父文件仅为合成身份fixture，不用于模型加载；加载测试由任务4创建真实模型快照。

- [ ] **Step 1: 编写失败测试**

```python
import pytest
import torch
from dataclasses import replace
from semantic_retrieval.evidence.types import ResourceCounts, TransitionTargets
from semantic_retrieval.evidence.config import load_evidence_config

def test_negative_gain_class_is_not_unknown():
    t = TransitionTargets(torch.tensor([0, 1, 2]),
                          torch.tensor([True, False, True]), "a" * 64)
    assert t.outcome_class[0].item() == 0 and t.valid_mask[0].item()
    assert not t.valid_mask[1].item()
    with pytest.raises(ValueError):
        ResourceCounts(-1, 0, 0, 0)

def test_config_rejects_unknown_fields_and_boolean_counts(config_factory, tmp_path):
    cfg = config_factory()
    payload = cfg.to_payload()
    payload["extra"] = 1
    p = tmp_path / "config.json"
    import json
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_evidence_config(p)
    with pytest.raises(ValueError):
        replace(cfg.sampling, train_samples=True)
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_contracts.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

类型构造器验证mask、shape、canonical唯一性、帧池/观测包含关系、有限数值及预算spent+reserved≤limits；state_sha256采用artifact_json_bytes的规范内容，不包含执行主机时间。ObservationState不含标签、rich目标或收益。ActionValueDistribution.expected_gain返回softmax(logits)[:,2]−softmax(logits)[:,0]。

config.to_payload()使用与现有config_payload一致的Path规范化，但不调用旧TrainConfig的LOSS_NAMES校验。新LOSS_KEYS固定pc/score/value，未使用项权重必须显式0。reference阶段绑定CLIP，matcher绑定D_ref，transitions/value绑定P_ref及转移身份；evaluation绑定模型/校准/冻结策略。拒绝非有限、bool充整数、错误阶段父模型。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: define evidence state and strict configuration contracts'
```

---

### Task 2: 实现嵌套视图、逐帧缓存与观测编码（D1）

**Files:**

- Create: `src/semantic_retrieval/evidence/views.py`, `src/semantic_retrieval/evidence/frame_cache.py`, `src/semantic_retrieval/evidence/encoding.py`
- Modify: `src/semantic_retrieval/models/temporal.py:TemporalEncoder.forward`
- Create: `tests/semantic_retrieval/evidence/test_views.py`, `tests/semantic_retrieval/evidence/test_frame_cache.py`, `tests/semantic_retrieval/evidence/test_encoding.py`

**Interfaces:**

views.build_frame_pool(frame_count: int, pool_size: int) -> tuple[int,...]；priority_order(pool: tuple[int,...]) -> tuple[int,...]；view_frame_ids(pool, count: int) -> tuple[int,...]；propose_actions(state: ObservationState) -> tuple[EvidenceAction,...]。按设计最远覆盖顺序选择，打包时按时间排序。动作必须完整返回资源上界和正成本：frame_requests与clip_frames为未尝试帧块大小，temporal_tokens为新增后全C观测帧数上界，mc_elements=profile.mc_samples*len(C)，estimated_cost为正cost_weights点乘；不能先返回占位动作。

FrameCacheIdentity(content_sha256, clip_sha256, clip_config_sha256, preprocessing_sha256)；FrameFeatureCache(store: ArtifactArrayStore).write(candidate_id, frame_id, feature: Tensor, *, identity) -> ArrayReference；read(...) -> Tensor。索引只写store.run_root/provenance/frame-cache，数组走现有store。

encoding.pack_observations(states: tuple[ObservationState,...], queries: QueryFeatures, frame_reader: Callable) -> ObservedFeatures。frame_reader(candidate: CandidateObservation, frame_ids: tuple[int,...]) -> Tensor[T,D]；仅接收已成功观测帧，按完整观测身份去重，不能按canonical单独去重。高层encode_observations(state, frame_cache, *, query_features)委托该函数。

fixture query_features复用父目录model与inputs的encode_query结果；frame_reader用固定8维值表示candidate/frame并记录requests，不读真实视频。

- [ ] **Step 1: 编写失败测试**

```python
import torch
from semantic_retrieval.evidence.views import build_frame_pool, view_frame_ids
from semantic_retrieval.evidence.encoding import pack_observations
from semantic_retrieval.models.temporal import TemporalEncoder

def test_nested_and_single_frame():
    pool = build_frame_pool(16, 16)
    assert set(view_frame_ids(pool, 4)) < set(view_frame_ids(pool, 8))
    assert view_frame_ids(build_frame_pool(1, 16), 16) == (0,)

def test_same_video_different_queries_stays_separate(state_factory, query_features, frame_reader):
    a = state_factory(query_id="q0", canonical_ids=("a",), observed=(0,))
    b = state_factory(query_id="q1", canonical_ids=("a",), observed=(15,))
    out = pack_observations((a, b), query_features, frame_reader)
    assert len(out.observation_keys) == 2
    assert out.pair_indices[:, 1].tolist() == [0, 1]
    assert out.position_ids[:, 0].tolist() == [0, 15]

def test_hidden_frames_cannot_change_temporal_output():
    t = TemporalEncoder(8, 1, 2, 16).eval()
    x = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, False, True]])
    ids = torch.tensor([[0, 7, 15]])
    before = t(x, mask, position_ids=ids)
    x[:, 1] = 999
    after = t(x, mask, position_ids=ids)
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_views.py tests/semantic_retrieval/evidence/test_frame_cache.py tests/semantic_retrieval/evidence/test_encoding.py tests/semantic_retrieval/test_model.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

扩展forward(self, frames, mask, *, position_ids: Tensor | None = None)，省略时沿用旧槽位行为。显式ID仅校验有效mask位置，必须long、范围合法且每行有效ID不重复；padding ID先替换安全索引，避免无效位置触发gather。遮罩在Transformer之前生效。

缓存拒绝requires_grad、非有限向量、哈希不符、同键内容冲突与symlink；引用每读必验。补充test_cache_rejects_wrong_identity_and_tamper：写入8维向量、改变身份或NPY bytes均必须ValueError；使用tmp_path/"arrays"构造ArtifactArrayStore。

观测集合与已尝试集合分离，成功帧属于已尝试。后续帧块跳过已尝试失败帧；无新帧不产生动作。追加测试有效position重复拒绝、顺序重打包对应ID不变，以及两查询相同观测允许共享而不同观测不共享。
追加coverage_vector(candidate: CandidateObservation) -> Tensor[4]，依次为成功帧数/帧池大小、首帧位置/(F−1)、末帧位置/(F−1)、含首尾边界的最大未覆盖间隙/(F−1)；F=1时后三项为0。作为matcher/value共享的固定可观测描述。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: isolate nested observations and stable temporal positions'
```

---

### Task 3: 实现身份稳定的MC概率匹配和对比目标（D2）

**Files:**

- Create: `src/semantic_retrieval/evidence/sampling.py`, `src/semantic_retrieval/evidence/matcher.py`, `src/semantic_retrieval/evidence/objectives.py`
- Create: `tests/semantic_retrieval/evidence/test_sampling.py`, `tests/semantic_retrieval/evidence/test_matcher.py`, `tests/semantic_retrieval/evidence/test_objectives.py`

**Interfaces:**

sampling.SamplingState(config: SamplingConfig, *, mode: str, model_sha256: str).noise(pair_ids, *, device, dtype) -> Tensor[M,P]；state_dict()/load_state_dict()保存训练Generator，eval按SHA256(pair身份、model、seed、sample序号)派生稳定噪声。

matcher.mc_log_probabilities(loc: Tensor, scale: Tensor, offsets: Tensor, noise: Tensor, *, temperature: float) -> Tensor[P]。GaussianPairMatcher(dim: int, *, scale_floor: float)包装PairMatcher为base，用h加帧覆盖描述预测scale；forward(features: EncodedFeatures, pairs, offsets, *, pair_ids, coverage, sampling) -> PairPredictiveScores。

objectives.probabilistic_contrastive(loc, scale, offsets, labels, noise, *, temperature) -> LossTerm；gaussian_score_nll(loc, scale, target, valid_mask) -> LossTerm。LossTerm保存sum与有效数，不在这里做跨rank归一化。

- [ ] **Step 1: 编写失败测试**

```python
import math
import torch
from semantic_retrieval.evidence.matcher import mc_log_probabilities
from semantic_retrieval.evidence.objectives import probabilistic_contrastive

def test_mc_averages_probabilities_before_log():
    loc = torch.zeros(3, requires_grad=True)
    scale = torch.ones(3, requires_grad=True)
    noise = torch.tensor([[0., 0., 0.], [3., -3., 0.]])
    offsets = torch.tensor([0, 3])
    actual = mc_log_probabilities(loc, scale, offsets, noise, temperature=1.)
    expected = torch.logsumexp(torch.log_softmax(loc[None] + scale[None] * noise, -1), 0) - math.log(2)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, torch.log_softmax(loc[None] + scale[None] * noise, -1).mean(0))
    (-actual[0]).backward()
    assert scale.grad.abs().sum() > 0 and torch.isfinite(scale.grad).all()

def test_unknown_never_enters_contrastive_denominator():
    loc = torch.tensor([1., 0., 1000.], requires_grad=True)
    result = probabilistic_contrastive(
        loc, torch.ones(3), torch.tensor([0, 3]),
        torch.tensor([1, 0, -1]), torch.zeros(2, 3), temperature=1.)
    expected = -torch.log_softmax(loc[:2], 0)[0]
    torch.testing.assert_close(result.numerator, expected)
    assert result.denominator.item() == 1

def test_eval_noise_is_candidate_permutation_invariant(config_factory):
    from semantic_retrieval.evidence.sampling import SamplingState
    s = SamplingState(config_factory().sampling, mode="eval", model_sha256="a"*64)
    ids = (("q0", "a"), ("q0", "b"))
    original = s.noise(ids, device=torch.device("cpu"), dtype=torch.float32)
    permuted = s.noise(ids[::-1], device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(original.flip(1), permuted, rtol=0, atol=0)
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_sampling.py tests/semantic_retrieval/evidence/test_matcher.py tests/semantic_retrieval/evidence/test_objectives.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

每个offset段独立log_softmax，再logsumexp跨M减log(M)，不得全P一起softmax。验证offsets、shape、σ>0与finite；允许空pair段返回空结果。unknown筛选发生在对比目标内部，整行缺已知正例或负例返回与loc/scale连图的零sum和0 count。

测试eval噪声候选重排不变、训练Generator保存/恢复得到相同下一噪声；pair_ids包括query与canonical避免别名泄漏。log_probability保持log域，派生概率下溢不影响排序。

增加test_small_scale_matches_point_softmax与test_scale_head_changes_inference_and_receives_gradient：用3候选人工输入、固定噪声，干预scale头改变预测分布；不要求每次干预都改变top1。覆盖多正例均匀目标、极端logits、padding、空pair和reference目标无梯度。
scale_head精确输入为PairMatcher.feature_dim+4，Linear→dim→GELU→Linear(1)，覆盖4维来自任务2。不得输入完整视图、frame-reader或目标。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: implement probabilistic contrastive scores with stable sampling'
```

---

### Task 4: 组合模型、固定参考目标与阶段优化器（D2/D3）

**Files:**

- Create: `src/semantic_retrieval/evidence/model.py`, `src/semantic_retrieval/evidence/training.py`
- Create: `tests/semantic_retrieval/evidence/test_training.py`
- Modify: `tests/semantic_retrieval/evidence/conftest.py`

**Interfaces:**

EvidenceMatcherModel(config: EvidenceModelConfig)复用ClipBackbone、TemporalEncoder与GaussianPairMatcher；forward(observed: ObservedFeatures, *, sampling: SamplingState) -> PairPredictiveScores，在frame tokens进入matcher前按position_ids运行temporal。CLIP始终冻结，raw编码通过encode_inputs(query: QueryInputs, states, frame_source, cache) -> ObservedFeatures。

training.build_stage_optimizer(model: nn.Module, config: StageTrainConfig) -> tuple[Optimizer, LRScheduler]；freeze_reference(model) -> nn.Module；rich_score_targets(reference, rich_observed, *, current_observed) -> tuple[Tensor, Tensor]；matcher_loss(student_scores, targets: EvidenceTargets, *, noise, config) -> dict[str,LossTerm]。

reference/matcher模式共享精确命名clip/temporal/matcher.base；显式load_reference_parts(path, student, *, expected_sha256)逐子模块strict加载，拒绝多余/缺失/shape不符，绝不全模型strict=False。reference的scale头冻结且不参与reference点分数。

追加observed_features fixture：任务2的两个查询、每查询a/b两候选、16容量、dim=8；rich_features使用相同候选更丰富视图。
MatcherTrainingBatch(observed: ObservedFeatures, rich: ObservedFeatures, targets: EvidenceTargets)定义在training.py；deterministic_contrastive(logits, offsets, labels, *, temperature) -> LossTerm供reference点匹配训练，不能向要求正scale的MC函数传0来冒充确定性分支。

- [ ] **Step 1: 编写失败测试**

```python
import torch
from semantic_retrieval.evidence.model import EvidenceMatcherModel
from semantic_retrieval.evidence.training import freeze_reference, build_stage_optimizer

def test_stage_allowlist_and_reference_freeze(config_factory):
    cfg = config_factory()
    model = EvidenceMatcherModel(cfg.model)
    reference = freeze_reference(EvidenceMatcherModel(cfg.model))
    opt, _ = build_stage_optimizer(model, cfg.train)
    desired = {id(p) for n, p in model.named_parameters()
               if n.startswith(cfg.train.trainable_prefixes)}
    actual = {id(p) for g in opt.param_groups for p in g["params"]}
    assert actual == desired
    assert all(not p.requires_grad for p in model.clip.parameters())
    assert all(not p.requires_grad for p in reference.parameters())
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_training.py tests/semantic_retrieval/evidence/test_objectives.py tests/semantic_retrieval/test_matching_gradients.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

阶段allowlist固定：reference更新temporal./matcher.base.，matcher追加matcher.scale_head.，value仅value_head.；配置不能将clip或reference加入allowlist。scheduler只按实际step推进。

rich目标在no_grad和eval下生成，raw参考logit尺度固定；当前与rich有效帧相同的配对rich_valid=False。目标及参考哈希进入EvidenceTargets，学生forward签名不接受targets。测试teacher权重/梯度未变、rich输入不能进入学生分支、score NLL和PC各自能给loc/scale有效梯度且count分别为配对/查询。

reference点排序、确定性/固定方差/学习温度/standard_probabilistic模式在任务8统一工厂接入，不为每种模式复制模型。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: train evidence matcher against a frozen rich-view reference'
```

---

### Task 5: 编译来源隔离、固定快照的补帧转移（D3）

**Files:**

- Create: `src/semantic_retrieval/evidence/data.py`, `src/semantic_retrieval/evidence/transitions.py`
- Create: `tests/semantic_retrieval/evidence/test_data.py`, `tests/semantic_retrieval/evidence/test_transitions.py`

**Interfaces:**

data.read_split_binding(binding: SplitBinding) -> tuple[EvidenceEpisode,...]；audit_component_roles(bindings) -> None；EvidenceEpisode包装已验证Episode、角色和source_component_ids，不重写原episode标签协议。校验split、manifest SHA、角色与所有query/candidate来源。

transitions.label_transition(before_id: str, after_id: str, relevance: Mapping[str,int], *, producer_sha256: str) -> TransitionTargets；sample_states(initial_states, *, seed, max_steps) -> Iterator[ObservationState]采用固定均匀合法动作脚本；compile_transitions(states, actions, frozen_matcher, targets, *, store: ArtifactArrayStore, producer_identity) -> TransitionReferences；read_transitions(references, *, expected_producer) -> TransitionBatch。

TransitionProducerIdentity含matcher/config/index/label/视图/MC/状态采样哈希；TransitionRecord含state/action IDs、before/after top1、targets引用与source component。TransitionReferences为索引及外置数组引用；不嵌完整矩阵。

在transitions.py定义TransitionReferences(index_path: Path, index_sha256: str, producer: TransitionProducerIdentity, arrays: tuple)与TransitionBatch(features: Tensor, targets: TransitionTargets, component_ids: tuple[str,...])，并保持value特征schema独立绑定。

- [ ] **Step 1: 编写失败测试**

```python
from semantic_retrieval.evidence.transitions import label_transition

def test_three_outcomes_and_unknown_are_distinct():
    truth = {"good": 1, "bad": 0, "unknown": -1}
    cases = [("good", "bad", 0), ("bad", "bad", 1), ("bad", "good", 2)]
    for before, after, expected in cases:
        t = label_transition(before, after, truth, producer_sha256="a" * 64)
        assert t.valid_mask.item() and t.outcome_class.item() == expected
    t = label_transition("unknown", "good", truth, producer_sha256="a" * 64)
    assert not t.valid_mask.item()
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_data.py tests/semantic_retrieval/evidence/test_transitions.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

每个抽样状态枚举全部合法单步动作，保持C固定；不枚举所有组合状态，不按预测收益筛动作。before/after使用同一P_ref和共同积分噪声。可达脚本只用于离线状态采样，其种子和步数写入producer。

补充test_components_cannot_cross_match_and_value：临时两个来源manifest即使query不同但candidate source重复也拒绝。test_all_negative_candidates_are_retained：标签完整、没有正例时保留class1，统计漏召回而非删episode。若不完整标签不能声称确无正例。

编译记录按状态/动作身份排序，producer不同拒绝读取；损坏任一外置NPY必须拒绝。参考与概率匹配器均eval且冻结；转移labels/完整帧不会进入收益输入。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: compile provenance-bound evidence acquisition transitions'
```

---

### Task 6: 实现收益分布、均值对照和三分类校准（D4）

**Files:**

- Create: `src/semantic_retrieval/evidence/value.py`, `src/semantic_retrieval/evidence/calibration.py`
- Modify: `src/semantic_retrieval/evidence/objectives.py`
- Create: `tests/semantic_retrieval/evidence/test_value.py`, `tests/semantic_retrieval/evidence/test_calibration.py`

**Interfaces:**

value.build_value_features(state, actions, pair_scores) -> Tensor[A,H]只用当前交互、geometry、概率描述子、覆盖/动作元数据；ValueHead(input_dim, hidden_dim).forward(features) -> ActionValueDistribution；MeanRegressor(input_dim, hidden_dim).forward(features) -> Tensor[A]；GeometryValue使用同结构且显式feature mask去掉额外概率描述子。

objectives.value_nll(logits, targets: TransitionTargets, *, importance_weights: Tensor | None=None) -> LossTerm；mean_gain_mse(predictions, targets, *, importance_weights=None) -> LossTerm。类别到g映射为class−1，只对valid取值。

calibration.fit_value_temperature(logits, targets, *, split: str, identity: dict) -> ValueCalibration；ValueCalibration.apply(logits, *, expected_identity) -> Tensor[A,3]；split只允许V_calibration，字段包含producer/model/角色/label scope/特征schema哈希。
固定value特征顺序：h_j；loc/log_scale/log_probability（3）；query margin/entropy/mean_log_probability/population_std（4）；coverage_vector（4）；动作新增帧数/F、最早/最晚位置归一化、estimated_cost（4）；四类剩余硬预算比例（4），总维度PairMatcher.feature_dim+19。单候选margin/std为0。GeometryValue仅屏蔽显式log_scale列，保留共同score geometry；确定性匹配器对照另负责区分打分路径的概率作用。
定义EvidenceValueModel(matcher, value_head)，冻结matcher，优化器仅允许value_head.；ValueTrainingBatch(features, targets, importance_weights)定义在value.py。

- [ ] **Step 1: 编写失败测试**

```python
import torch
from semantic_retrieval.evidence.types import ActionValueDistribution, TransitionTargets
from semantic_retrieval.evidence.objectives import value_nll

def test_value_retains_harm_and_uses_weight_sum():
    probs = torch.tensor([[.6, .3, .1], [.1, .2, .7]])
    out = ActionValueDistribution(probs.log())
    torch.testing.assert_close(out.expected_gain, torch.tensor([-.5, .6]))
    target = TransitionTargets(torch.tensor([0, 2]),
                               torch.tensor([True, True]), "a" * 64)
    loss = value_nll(probs.log(), target, importance_weights=torch.tensor([1., 3.]))
    assert loss.denominator.item() == 4
    torch.testing.assert_close(loss.numerator, -probs[0,0].log() - 3*probs[1,2].log())
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_value.py tests/semantic_retrieval/evidence/test_calibration.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

value头训练时所有匹配特征detach，冻结整个matcher；新增test_value_step_cannot_update_matcher，反向后仅value_head参数有梯度。默认无类权重；分层采样只能通过已知采样比率importance correction，拒绝负/非有限权重。

多分类温度只优化正标量log_temperature，拟合原动作总体NLL；数据有效数0拒绝。测试V_development/V_locked/test均拒绝fit，身份变化拒绝apply，proper score不把类别0当unknown。

MeanRegressor与分布头使用相同完整特征及预先声明的参数预算，报告实际参数数；GeometryValue另作输入消融，禁止让两个对照混用问题。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: learn calibrated acquisition outcomes and mean-value baselines'
```

---

### Task 7: 实现预算决策、获取执行与动作事务（D5）

**Files:**

- Create: `src/semantic_retrieval/evidence/policy.py`, `src/semantic_retrieval/evidence/journal.py`
- Modify: `src/semantic_retrieval/evidence/views.py`
- Create: `tests/semantic_retrieval/evidence/test_policy.py`, `tests/semantic_retrieval/evidence/test_journal.py`
- Modify: `tests/semantic_retrieval/evidence/conftest.py`

**Interfaces:**

policy.reserve_action(state, action, *, predicted_counts: ResourceCounts) -> AcquireDecision；decide(state, actions, values, budget, *, stop_threshold: float) -> AcquireDecision | StopDecision；acquire(action, state, frame_source) -> ActionOutcome；apply_outcome(state, action, outcome) -> ObservationState。

使用任务2已完整定义的views.propose_actions及资源上界，不另造cost估计。BudgetLedger.reserve(counts) -> BudgetLedger和settle(reservation, actual) -> BudgetLedger负责整数向量运算与上界拒绝。初始预测的temporal/MC操作由runtime先预留结算。

journal.ActionJournal(root: Path, store: ArtifactArrayStore).prepare(state, decision) -> str；record_execution(action_id, outcome) -> None；commit(action_id, next_state) -> None；recover(state) -> ObservationState。root固定为store.run_root/provenance/evidence-actions；按action ID写不可变事件，引用哈希核验。
FrameSource为Callable[[EvidenceAction, ObservationState], ActionOutcome]；生产adapter在此执行decode、冻结CLIP编码和缓存写入，acquire负责先核验合法性再调用，不把原始完整视频特征交给策略。
新增journal_case fixture，包含state/decision/outcome/next_state/journal：初始spent为0，合法2帧动作，成功特征真实写入tmp_path ArtifactArrayStore，next_state由apply_outcome生成。

- [ ] **Step 1: 编写失败测试**

```python
import torch
from semantic_retrieval.evidence.types import ActionValueDistribution, StopDecision
from semantic_retrieval.evidence.policy import decide

def test_harmful_actions_stop(state_factory):
    state = state_factory()
    from semantic_retrieval.evidence.views import propose_actions
    actions = propose_actions(state)
    probs = torch.tensor([[.7, .2, .1]]).repeat(len(actions), 1)
    result = decide(state, actions, ActionValueDistribution(probs.log()),
                    state.budget, stop_threshold=0.)
    assert isinstance(result, StopDecision)

def test_failed_acquisition_charges_requests_and_exhausts_attempts(state_factory):
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.types import ActionOutcome, ResourceCounts
    from semantic_retrieval.evidence.policy import acquire, apply_outcome
    state = state_factory()
    action = propose_actions(state)[0]
    def fail_source(a, s):
        return ActionOutcome(a.action_id, (), a.frame_ids,
                             ResourceCounts(len(a.frame_ids), 0, 0, 0), 0., ())
    outcome = acquire(action, state, fail_source)
    updated = apply_outcome(state, action, outcome)
    assert updated.budget.spent.frame_requests == len(action.frame_ids)
    assert updated.budget.spent.clip_frames == 0
    for other in propose_actions(updated):
        if other.candidate_id == action.candidate_id:
            assert not set(other.frame_ids) & set(action.frame_ids)

def test_execution_without_commit_reconciles_exactly_once(journal_case):
    c = journal_case
    c.journal.prepare(c.state, c.decision)
    c.journal.record_execution(c.decision.action.action_id, c.outcome)
    once = c.journal.recover(c.state)
    twice = c.journal.recover(once)
    assert once == twice == c.next_state
    assert once.budget.spent == c.outcome.actual_counts
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_policy.py tests/semantic_retrieval/evidence/test_journal.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

选择最大gain/cost，只考虑gain>stop_threshold且硬资源可预留动作；平局canonical ID/action ID排序。decide不接相关性Targets。禁止预留超过limits、非正成本和state哈希过期的动作。执行后失败请求计费、编码成功帧计费，失败帧加入attempted，未来不重复请求。

补充test_cost_changes_action_choice：两个合法动作分别gain .6/.3、cost 1/.1，第二个被选中。test_all_decode_failures_stop_without_free_frames：两个失败请求消耗2 frame_requests，0 clip_frames，state不新增成功帧、动作不重复。

journal测试prepare后崩溃、record_execution后崩溃、commit后恢复三种窗口。已验输出恢复仅结算一次，若引用缺失/损坏或无法确认实际操作计数则拒绝恢复；不能把未执行预留自动释放再跑而绕过累计预算。每条事件只含身份/计数/引用，不写张量。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: execute budgeted evidence actions with recoverable transactions'
```

---

### Task 8: 实现评价与公平基线工厂（D4/D5）

**Files:**

- Create: `src/semantic_retrieval/evidence/evaluation.py`, `src/semantic_retrieval/evidence/baselines.py`
- Create: `tests/semantic_retrieval/evidence/test_evaluation.py`, `tests/semantic_retrieval/evidence/test_baselines.py`

**Interfaces:**

evaluation.rank_candidates(log_probability: Tensor, canonical_ids: tuple[str,...]) -> tuple[str,...]；score_outcomes(logits, targets: TransitionTargets) -> dict[str,float]；evaluate_evidence(records, *, relevance_space, source_components) -> EvidenceReport；paired_component_interval(before, after, component_ids, *, seed, samples, confidence_level) -> dict。

baselines.build_matcher_variant(config, *, variant: str) -> EvidenceMatcherModel；variant允许gaussian/deterministic/fixed_scale/learned_temperature/standard_probabilistic；build_acquisition_baseline(name, *, seed, config) -> callable，name允许fixed_frames/random/margin/entropy/mean_regressor/distribution。随机规则只读合法动作与预算。

EvidenceReport小标量单列score NLL/coverage、动作NLL/Brier、overall top1、initial_miss、unknown count、corrections/worsenings、预算/耗时；逐查询及bootstrap载荷外置。

在evaluation.py定义EvidenceReport(summary_metrics: dict[str,float], array_references: tuple, undefined_reasons: dict[str,str], identity: dict)。转移与收益数据消费任务5定义的类型，不重声明。
追加build_initial_index(states, frame_reader, *, identity: IndexIdentity) -> CanonicalIndex和search_initial_candidates(query_features, index, k, *, expected_identity) -> tuple[tuple[str,...],...]，复用现有CanonicalIndex。粗检索直接使用冻结CLIP的归一化pooled text及初始帧均值，不引入随机冻结的CoarseProjector。仅允许初始层级视图，identity将CLIP与视图协议共同绑定；不能借完整帧均值构建稀疏索引。

- [ ] **Step 1: 编写失败测试**

```python
import torch
from semantic_retrieval.evidence.evaluation import rank_candidates, score_outcomes
from semantic_retrieval.evidence.types import TransitionTargets

def test_ties_and_permutations_have_same_canonical_order():
    assert rank_candidates(torch.zeros(3), ("b", "c", "a")) == ("a", "b", "c")
    assert rank_candidates(torch.zeros(3), ("c", "a", "b")) == ("a", "b", "c")

def test_perfect_outcomes_have_zero_brier():
    logits = torch.tensor([[30., -30., -30.], [-30., -30., 30.]])
    targets = TransitionTargets(torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool), "a"*64)
    assert score_outcomes(logits, targets)["brier"] < 1e-12
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_evaluation.py tests/semantic_retrieval/evidence/test_baselines.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

总体排序失败包括初始候选漏召回；unknown比例单列，标签不全不发布精确全库Recall。无有效动作指标返回显式undefined原因与count，不写NaN或假0。Gaussian分数覆盖只针对rich-score，动作校准分总体与策略选中子集。

paired_component_interval以source component整组重采样，不能按caption/action/MC条目采样。加test_duplicate_captions_do_not_add_bootstrap_units；源组件数不足时输出不足说明，不能把seed数当样本量。

baseline共享C、初始索引、动作机会及标签监督范围，固定帧/随机动作也走同BudgetLedger。standard_probabilistic令score权重0而保留MC对比；fixed_scale冻结scale；learned_temperature使用确定性score加正温度；variant配置进入hash，不能复用不同variant checkpoint。报告每种实际参数、编码与MC成本。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: evaluate evidence utility against information-matched baselines'
```

---

### Task 9: 分布式优化、精确进度与checkpoint恢复（D6）

**Files:**

- Create: `src/semantic_retrieval/evidence/checkpoint.py`
- Modify: `src/semantic_retrieval/evidence/training.py`
- Create: `tests/semantic_retrieval/evidence/test_distributed.py`, `tests/semantic_retrieval/evidence/test_checkpoint.py`

**Interfaces:**

training.global_loss(terms: dict[str,LossTerm], weights: dict[str,float]) -> tuple[Tensor,bool]；EvidenceStepper(model, optimizer, scheduler, *, weights).step(batch) -> StepMetrics。EvidenceStepper是可测试优化组件，生产调用必须置于任务10保护runtime内；不自行发布任何run。

checkpoint.EvidenceProgress(stage, step, epoch, cursor, sampler_state, mc_state, producer_sha256, observation_state, pending_action)；save_evidence_checkpoint(context, models, optimizer, scheduler, config, progress) -> Path；load_evidence_checkpoint(path, models, *, expected_config_sha256, expected_parents) -> EvidenceTrainingState。

复用现有model_sha256、rng_state、restore_rng与ArtifactArrayStore；原checkpoint schema1/2不增加兼容分支。新evidence-checkpoint-v1明确marker，发布.pt与.json sidecar，后者为完成标记；所有大数组外置。

在training.py定义StepMetrics(loss_sums: dict, weight_sums: dict, step: int, stepped: bool)；在checkpoint.py定义EvidenceTrainingState(progress: EvidenceProgress, optimizer_state: dict, scheduler_state: dict, rng_states: tuple)。所有恢复消费方使用这些字段，不借用旧TrainingState.progress的pilot键。

- [ ] **Step 1: 编写失败测试**

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _loss_worker(rank, rendezvous):
    from semantic_retrieval.evidence.types import LossTerm
    from semantic_retrieval.evidence.training import global_loss
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        x = torch.tensor(2., requires_grad=True)
        count = torch.tensor(float(rank == 1))
        loss, active = global_loss({"pc": LossTerm(count*x*x, count)}, {"pc": 1.})
        assert active
        loss.backward()
        # DDP平均前的本地梯度：rank0为0，rank1为8；平均为单进程4。
        assert x.grad.item() == (0. if rank == 0 else 8.)
    finally:
        dist.destroy_process_group()

def test_one_rank_unknown_does_not_skip_collectives(tmp_path):
    mp.spawn(_loss_worker, args=("file://" + str(tmp_path/"rendezvous"),), nprocs=2, join=True)
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_distributed.py tests/semantic_retrieval/evidence/test_checkpoint.py tests/semantic_retrieval/test_distributed.py tests/semantic_retrieval/test_training_resume.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

global_loss对每项all_reduce denominator；本地numerator乘world/global_denominator，交给DDP梯度平均。全局有有效项时每rank均backward，即使本地全unknown；全局全0时skip optimizer及scheduler。importance权重按浮点和归一化。拒绝rank间stage/keys/权重不一致。

对denominator.detach().clone()进行collective，不原地修改LossTerm中的count，避免破坏numerator反向图保存的张量版本。

checkpoint每rank收集MC/数据采样器/RNG；签名绑定阶段、来源划分、父模型、转移producer和参数allowlist。load先校验sidecar、模型/配置/数组/父链哈希，再strict加载。禁止根据恢复时运行状态推测缺失cursor。

增加test_resume_next_noise_sampler_and_parameters_exact：同CPU模型连续两步与一步→保存→恢复→第二步逐张量equal，下一采样与动作一致；test_all_unknown_does_not_step；test_partial_sidecar_and_wrong_producer_rejected。policy恢复委托任务7journal，不重复实现事务。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: restore evidence training and acquisition state exactly'
```

---

### Task 10: 接入受保护归档、运行权限与薄CLI（D6）

**Files:**

- Create: `src/semantic_retrieval/evidence/runtime.py`, `src/semantic_retrieval/evidence/authorization.py`
- Create: `scripts/train_evidence_retrieval.py`, `scripts/compile_evidence_transitions.py`, `scripts/evaluate_evidence_retrieval.py`
- Modify: `src/semantic_retrieval/runtime/artifacts.py:prepare_run/start_run; scripts/manage_experiment_run.py:reserve policy dispatch`
- Create: `tests/semantic_retrieval/evidence/test_runtime.py`, `tests/semantic_retrieval/evidence/test_authorization.py`, `tests/semantic_retrieval/evidence/test_cli.py`
- Create: `src/semantic_retrieval/evidence/schemas.py`

**Interfaces:**

authorization.assert_evidence_authorized(*, repo_root: Path, registry_path: Path, manifest: dict, operation: str) -> None；runtime.prepare_evidence_run(config: EvidenceSystemConfig, *, repo_root) -> RunContext；run_evidence_stage(context, config) -> dict；run_policy_episode(context, state, models, frame_source, *, journal) -> EpisodeResult。

每个新CLI main(argv=None) -> int只接受--config与可核验的--run-id/--manifest-sha256/--source-sha；无任意输出目录、绕过资格或test开关。train接reference/matcher/value，compile接transitions，evaluate接calibration/evaluation；cpu_fixture经任务11的qualification入口调用同runtime。

在runtime.py定义EpisodeResult(ranking: tuple[str,...], action_ids: tuple[str,...], final_state: ObservationState, costs: ResourceCounts, timings: dict[str,float])。

- [ ] **Step 1: 编写失败测试**

```python
import pytest
from semantic_retrieval.evidence.authorization import assert_evidence_authorized

def test_evidence_never_inherits_old_study_authorization():
    manifest = {
        "study": "semantic-answerability-pilot-v1",
        "registry_path": "experiments/semantic-answerability-pilot-v1/registry.jsonl",
        "config": {"evaluation": {"config_kind": "evidence-retrieval-v1"}},
        "test_authorized": False,
    }
    with pytest.raises(ValueError):
        assert_evidence_authorized(repo_root=None, registry_path=None,
                                   manifest=manifest, operation="start")
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_runtime.py tests/semantic_retrieval/evidence/test_authorization.py tests/semantic_retrieval/evidence/test_cli.py tests/semantic_retrieval/test_runtime_artifacts.py tests/semantic_retrieval/test_pilot_policy.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

在prepare_run对EvidenceSystemConfig做显式类型分派，比较evidence_manifest_config；原SystemConfig路径逐项保持现有逻辑。提取公共身份/环境/输入/生命周期核验供两路使用，不能让新路径绕过verify_committed_reservation或通过system_config=None进入空配置模式。start_run重校验沿用同一分派。新resume验证使用evidence父链，原_validate_resume/_validate_pilot仅作用旧SystemConfig。

reserve与worker/start两端都调用新authorization。marker、study kind、registry及artifact namespace必须一致，禁止旧study借marker接入。真实stage要求新study显式科学授权及已提交新链路四卡资格引用；资格必须绑定源码/配置/输入、4个CUDA rank、FP32、有效梯度及恢复检查。缺失/错误资格全部拒绝。S0仅允许绑定tiny随机权重、合成数据、CPU的临时归档fixture；无新增GPU资格授权。

测试新marker移除但CLI/配置为evidence仍拒绝、科学CPU阶段也不能绕gate、test即使manifest true也因本路径无新研究gate而拒绝。temporary reservation fixture复用test_runtime_artifacts.reservation/commit_reservation的临时Git方式，不能触碰真实registry。

manifest有效配置比较覆盖温度/M、全部split、成本/预算、父模型、value特征schema及积分seed。新checkpoint/transition/校准JSON采用显式新schema，调用artifact_schema_validator，拒绝未知字段；不放宽旧metrics schema。checkpoint与CLI退出失败证据沿用现有RunContext生命周期。
schemas.py提供schema_for(kind: str) -> dict和validate_evidence_document(kind, payload) -> None，kind固定checkpoint_sidecar/transition_index/calibration/action_event/report；使用experiment_artifacts.artifact_schema_validator和strict JSON writer。selected validation沿用现有小标量metrics envelope(split/summary_metrics/array_references)，详细EvidenceReport为额外已引用文档；不在旧schema增加字段。
新配置manifest仍使用既有顶层model/loss/optimizer/scheduler/evaluation，stage写在evaluation.evidence_stage，不能把S0–S6写进归档ALLOWED_STAGES。新checkpoint进度与父链核验在evidence路径分派，不能误调用旧LOSS_NAMES绑定的progress检查。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'feat: bind evidence workers to protected experiment identities'
```

---

### Task 11: 完成受保护CPU闭环、覆盖审查和交付文档（D6）

**Files:**

- Create: `src/semantic_retrieval/evidence/qualification.py; scripts/qualify_evidence_retrieval.py`
- Create: `tests/semantic_retrieval/evidence/test_end_to_end.py`
- Create: `docs/agent/evidence-retrieval.md`
- Modify: `docs/superpowers/plans/2026-10-10-evidence-conditioned-retrieval-implementation.md（仅执行进度/实际验证结果）`

**Interfaces:**

qualification.run_cpu_evidence_qualification(context, config, *, interrupt_at: str | None=None) -> QualificationReport；脚本main与任务10一样绑定config/identity，不接受真实数据替换参数。QualificationReport含软件check bool、输入/模型/状态hash与成本计数，不写科学Recall改善结论。

在test_end_to_end.py提供cpu_qualification_runs fixture：tmp_path临时Git、合成权重与2查询/3候选视频、已提交S0预留；通过真实CLI进入RunContext，执行少量reference/matcher/value优化、固定转移、一次获取/停止、checkpoint与动作恢复。fixture不得假装产生四卡合格记录。

- [ ] **Step 1: 编写失败测试**

```python
def test_protected_cpu_pipeline_and_resume(cpu_qualification_runs):
    continuous, resumed = cpu_qualification_runs
    assert continuous["status"] == resumed["status"] == "completed"
    assert continuous["model_sha256"] == resumed["model_sha256"]
    assert continuous["action_ids"] == resumed["action_ids"]
    assert continuous["cost_counts"] == resumed["cost_counts"]
    assert continuous["checks"]["hidden_evidence_isolated"]
    assert continuous["checks"]["scale_affects_prediction"]
    assert continuous["checks"]["teacher_frozen"]
    assert continuous["scientific_authorized"] is False
    assert continuous["test_authorized"] is False
```

- [ ] **Step 2: 运行下列命令，确认新增接口/行为尚未实现时失败**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence/test_end_to_end.py -q
```

- [ ] **Step 3: 实现接口并补全列明的验收用例**

通过实际worker prepare/start/complete路径和控制器terminal追加验证闭环；在临时repo测试失败证据，不修改真实受跟踪协议。中断注入包括训练checkpoint后和动作执行后commit前，比较精确重放。用toy干预产生三类转移，只声明软件正确性。

docs/agent/evidence-retrieval.md列出配置角色、CLI接口、成本/校准口径、输入哈希检查、恢复与科学gate拒绝原因；没有已授权新study不得给出可直接启动科学训练的伪run命令。

完成spec逐节覆盖表；对每项实际跑过测试记录命令与结果，失败如实保留。收尾审阅整条分支，确认无真实数据/旧checkpoint访问、无GPU实验、无被悄悄放宽的test/registry/schema守卫。
CLI临时仓库fixture复制受测src/semantic_retrieval包、三个evidence入口及必要experiment_artifacts.py/experiment_tracking.py等归档依赖后提交，worker从该临时源码SHA启动；实际运行入口的repo_root来自其__file__。不能仅伪造一个Git目录却在未绑定的外部源码上宣称验证了源码身份。
完整回归命令见收尾清单；若现有测试失败，先判断是否本改动引入并修复相关回归，不用skip或放宽合同消除失败。

- [ ] **Step 4: 重跑Step 2同一命令，所有本任务测试与列明回归应通过；不新增skip或访问GPU**

- [ ] **Step 5: 仅暂存Files列出的精确路径并提交**

```bash
git diff --cached --check
git commit -m 'test: qualify the protected evidence retrieval pipeline on CPU'
```

---

## Spec Coverage and Final Verification

| 设计章节 | 对应任务 |
|---|---|
| §1–3 目标、范围、方案选择 | 1、4、8、11；只实现固定C/预定义帧块路径 |
| §4 架构与数据流 | 2、4、5、6、7、10 |
| §5 数据/身份/稳定位置 | 1、2、5 |
| §6 特征与缓存 | 2、8 |
| §7 概率匹配、分数监督、MC重放 | 3、4、9 |
| §8 转移编译 | 5 |
| §9 收益/校准/对照 | 6、8 |
| §10 策略/成本/故障 | 7、9 |
| §11 模块、接口与配置 | 1、10 |
| §12 分阶段训练 | 4、5、6、10、11 |
| §13 分布式与恢复 | 9、10、11 |
| §14 验收矩阵 | 1–11 |
| §15 评价与贡献判断 | 8、11 |
| §16 协议与授权 | 10、11 |
| §17 书面阶段审阅 | 本计划审阅与执行方式选择 |

- [ ] 全部新证据链CPU测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence -q
```

- [ ] 由于本计划修改共享TemporalEncoder、RunContext身份分派和reserve守卫，完成相关子系统回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval -q
```

- [ ] 查找并执行仓库实际存在的归档管理相关测试：先 `rg --files tests | rg 'artifact|experiment_run|registry'`，仅扩展到修改分派实际影响的测试，不扫描大日志/权重。
- [ ] 确认新schema引用哈希、父链、预算及恢复记录；检查原test授权和旧study状态没有变化。
- [ ] 汇总每项实际测试证据、已知限制与未完成工作；只有对应测试执行并通过才勾选，不根据计划写“全绿”。

## Handoff

执行前请用户审阅本计划并选择执行方式。推荐**Native（本会话执行）**：11项任务对观测/预测/进度类型有连续依赖，在同一上下文按任务验收便于保持合同一致；最后做一次独立整分支审阅。

也可选**Subagent-driven**：每任务新执行agent与新reviewer，按Superpowers相应技能执行。当前未选择执行方式，不启动agent或实现。本计划批准不授权科学运行，新链路四卡资格、独立study与研究gate仍分别生效。

## 2026-10-10 Native 执行记录

用户已明确选择本会话 Native 并确认设计与计划；最新执行指令另外授权新 study、全机 GPU 软件资格，以及通过研究 gate 后的训练/validation。原文中的 CPU-only、不得建立 study 与未选择执行方式是批准前的历史边界，由该明确指令替代；旧 study、sealed test 和研究定义保持不变。

任务 1–10 已按依赖顺序提交；任务 11 已实现真实临时源码 CLI 闭环、成功及失败归档、精确 checkpoint 和动作日志恢复，以及独立四卡资格入口。CPU 闭环覆盖 tiny 随机 CLIP、合成视频、教师/概率匹配/收益训练和采集/停止。新 study 控制面明确保持 science gates 关闭；四卡资格和来源完整性审计分别验收，不以 CPU 成功替代。

实际验证：任务 10 指定新旧路径回归 53 passed；扩展相关集 64 passed；新 evidence 集在任务 10 结束前 73 passed；CPU 真实 CLI 成功及失败闭环 2 passed。更晚的验证以执行报告及归档摘要为准。
