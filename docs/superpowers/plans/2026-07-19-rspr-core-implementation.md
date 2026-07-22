# RSPR Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 UATVR 的 CLIP–DSA 确定性主干上实现第一阶段 RSPR（重参数化随机原型排序）核心版，使 Mask-aware 概率嵌入、双向 soft prototype matching、配对不确定性和随机排序损失形成可训练、可复现、可消融的完整链路。

**Architecture:** 保留 `modules/modeling.py` 中的确定性 DSA/WTI 主干，把概率分布参数化放入独立的 `prob_models/reparameterized_distribution.py`，把原型匹配、随机排序和核心编排放入 `modules/stochastic_prototype_ranking.py`。训练阶段用显式 `pair_id` 构造多正样本目标和困难负样本；评估阶段先用概率均值召回 Top-R，再用固定 antithetic 噪声进行概率精排。旧 UATVR 概率分支仅作为 A4 消融保留，不与新核心损失叠加。Pairwise Beta Evidence 不在本计划中实现。

**Tech Stack:** Python 3、PyTorch、torch.distributed、NumPy、pytest、Ruff、Bash、Git

## Global Constraints

- 规格事实源为 `docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md`。
- 活动模型路径固定为 `modules/modeling.py`；工作树中的该文件目前来自用户正在进行的 rename，实施时不得改回历史路径。
- 保留工作树中已有的无关修改、删除和未跟踪文件；每次提交只暂存当前任务明确列出的文件。
- `rspr_mode=legacy` 必须保持当前 UATVR 概率分支和损失语义，但与其他消融一样从 OpenAI CLIP 权重起训，不要求历史 UATVR 全模型 checkpoint；新模块只在 `mean` 或 `stochastic` 模式实例化和执行。
- `rspr_mode=off` 是 A0 确定性 DSA 基线；`mean` 是 A1；`stochastic` 是 A2/A3/A5–A8；`legacy` 是 A4。
- 第一阶段只实现核心损失 `L_DSA + λp L_prob + λr L_rank + λa L_anchor`；不得新增 Beta evidence、PCM、ensemble、conformal 或 Video-LLM 路径。
- 文本与视频分布头结构相同但参数不共享；全项目统一使用 `logvar` 语义，标准差只能由 `exp(0.5 * logvar)` 得到。
- `K=1` 只允许 mean-only；随机采样的 `K` 必须为正偶数。训练默认 `K=4`，评估默认 `K=8`。
- `logvar`、指数、anchor KL 和稳定 `logmeanexp` 在 FP32 中执行；所有随机原型在相似度计算前 L2 normalize。
- 空 mask 立即抛出 `ValueError`；只有一个有效 token/frame 时归一化注意力熵固定为 0。
- 训练主路径不得 detach `mu`、`logvar`、随机原型或 DSA token；`--rspr_detach_samples` 仅用于 A2。
- 所有正负关系均由显式 `pair_id` 构造；同一视频的其他 caption 不得进入困难负样本。
- 全库不执行 `Bt × Bv × K²` 概率匹配；只用概率均值召回，并在 Top-R 候选上运行随机原型精排。
- 固定评估噪声必须保证同一 checkpoint、数据和参数重复评估得到逐元素相同的分数与排名。
- 每个任务遵循 RED → GREEN → REFACTOR；没有看到预期失败，不得直接进入实现步骤。
- 训练和评估入口统一使用 `/home/xujie/.conda/envs/tvr/bin/python` 与 `/home/xujie/.conda/envs/tvr/bin/torchrun`；科研验收优先真实 smoke，不为低风险改动扩展测试矩阵。

---

### Task 1: 实现 Mask-aware 统计聚合与重参数化分布头

**Files:**
- Create: `prob_models/reparameterized_distribution.py`
- Create: `tests/test_reparameterized_distribution.py`

**Interfaces:**

```text
DistributionOutput(center, dispersion, attention_entropy, mean, logvar, samples, anchor_kl)
MaskedStatPool.forward(tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]
antithetic_standard_normal(batch_size, sample_count, dim, *, device, dtype, generator=None) -> Tensor
ReparameterizedDistributionHead.forward(
    tokens: Tensor,
    mask: Tensor,
    *,
    sample_count: int,
    noise: Tensor | None = None,
    mean_only: bool = False,
    detach_samples: bool = False,
) -> DistributionOutput
```

- `tokens` 为 `[B,N,D]`，`mask` 为 `[B,N]`；输出 `center/dispersion/mean/logvar` 为 `[B,D]`，`attention_entropy` 为 `[B,1]`，`samples` 为 `[B,K,D]`。
- `anchor_kl` 已按 batch 和维度取均值，是可直接乘 `λa` 的标量。
- 当传入 `noise` 时必须严格验证形状 `[B,K,D]`；`mean_only=True` 时忽略随机噪声并返回 `normalize(mean).unsqueeze(1)`。

- [ ] **Step 1: 写入 Mask-aware 聚合、采样和梯度契约测试**

测试至少包含以下函数：

```python
def test_masked_stat_pool_ignores_padding_and_handles_single_valid_token():
    pool = MaskedStatPool(dim=3, hidden_dim=4)
    tokens = torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[1, 0]])
    center, dispersion, entropy = pool(tokens, mask)
    torch.testing.assert_close(center, tokens[:, :1].squeeze(1))
    torch.testing.assert_close(dispersion, torch.zeros_like(dispersion))
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))


def test_masked_stat_pool_rejects_all_invalid_rows():
    pool = MaskedStatPool(dim=3, hidden_dim=4)
    with pytest.raises(ValueError, match="at least one valid position"):
        pool(torch.randn(2, 4, 3), torch.zeros(2, 4, dtype=torch.long))


def test_antithetic_noise_contains_exact_positive_negative_pairs():
    noise = antithetic_standard_normal(
        2, 4, 5, device=torch.device("cpu"), dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(noise[:, 0], -noise[:, 1])
    torch.testing.assert_close(noise[:, 2], -noise[:, 3])


def test_distribution_head_reparameterization_reaches_mean_variance_and_tokens():
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8, requires_grad=True)
    output = head(tokens, torch.ones(3, 5), sample_count=4)
    output.samples.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert head.mean_head[-1].weight.grad is not None
    assert head.logvar_head[-1].weight.grad is not None
```

同时覆盖：`logvar` 位于 `[-8,2]`、样本范数为 1、`K=3` 抛错、给定固定 `noise` 两次输出完全一致、`detach_samples=True` 阻断概率匹配梯度但不影响单独的 anchor KL 梯度。

- [ ] **Step 2: 运行新测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_reparameterized_distribution.py
```

Expected: collection error，提示 `prob_models.reparameterized_distribution` 不存在。

- [ ] **Step 3: 实现稳定的 Mask-aware 统计池化**

核心计算必须等价于：

```python
valid = mask.to(device=tokens.device, dtype=torch.bool)
if valid.ndim != 2 or valid.shape != tokens.shape[:2]:
    raise ValueError("mask must have shape [batch, sequence]")
if (~valid.any(dim=1)).any():
    raise ValueError("every sample must contain at least one valid position")

work = tokens.float()
logits = self.score(self.features(work)).squeeze(-1)
logits = logits.masked_fill(~valid, float("-inf"))
attention = torch.softmax(logits, dim=-1)
center = torch.einsum("bn,bnd->bd", attention, work)
dispersion = torch.einsum(
    "bn,bnd->bd", attention, (work - center.unsqueeze(1)).square()
)
valid_count = valid.sum(dim=1, keepdim=True)
raw_entropy = -(attention * attention.clamp_min(self.eps).log()).sum(
    dim=1, keepdim=True
)
entropy = torch.where(
    valid_count > 1,
    raw_entropy / valid_count.float().log().clamp_min(self.eps),
    torch.zeros_like(raw_entropy),
)
```

`features` 使用 `Linear(D,H) → GELU → LayerNorm(H)`，`score` 使用 `Linear(H,1,bias=False)`；padding 位置不得参与中心、离散度或熵。

- [ ] **Step 4: 实现均值/方差头、antithetic sampling 和 anchor KL**

使用以下参数化：

```python
mean_delta = self.mean_head(self.mean_norm(torch.cat([center, dispersion], dim=-1)))
mean = center + mean_delta
logvar_input = torch.cat([center, dispersion, entropy], dim=-1)
logvar = self.logvar_head(self.logvar_norm(logvar_input)).float()
logvar = logvar.clamp(self.logvar_min, self.logvar_max)
variance = logvar.exp()

prior_variance = self.prior_std ** 2
anchor_kl = 0.5 * (
    (variance + (mean.float() - center.detach()).square()) / prior_variance
    - 1.0
    + math.log(prior_variance)
    - logvar
).mean()
```

随机路径使用：

```python
raw_samples = mean.float().unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * noise
samples = F.normalize(raw_samples, dim=-1, eps=self.eps)
if detach_samples:
    samples = samples.detach()
```

antithetic 顺序固定为 `[ε1, -ε1, ε2, -ε2]`；更大的 K 继续按正负对交错，使任意正偶数前缀仍是完整 antithetic pair。

- [ ] **Step 5: 运行单元测试并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_reparameterized_distribution.py
python3 -m ruff check prob_models/reparameterized_distribution.py tests/test_reparameterized_distribution.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 6: 提交分布头**

```bash
git add prob_models/reparameterized_distribution.py tests/test_reparameterized_distribution.py
git commit -m "feat: add reparameterized distribution head"
```

### Task 2: 实现双向随机原型匹配与配对不确定性

**Files:**
- Create: `modules/stochastic_prototype_ranking.py`
- Create: `tests/test_stochastic_prototype_ranking.py`

**Interfaces:**

```text
PrototypeMatchOutput(
    logits,
    pair_uncertainty,
    text_prototype_scores,
    video_prototype_scores,
    stochastic_pair_scores,
)
BidirectionalSoftPrototypeMatcher.forward(
    text_samples: Tensor, video_samples: Tensor
) -> PrototypeMatchOutput
BidirectionalSoftPrototypeMatcher.score_pairs(
    text_samples: Tensor, video_samples: Tensor
) -> PrototypeMatchOutput
```

- `forward` 接受 `[Bt,K,D]` 和 `[Bv,K,D]`，支持 `Bt != Bv`，返回矩阵输出 `[Bt,Bv]`。
- `score_pairs` 接受相同首维的 `[P,K,D]` 与 `[P,K,D]`，只计算 P 个已对齐 query-candidate pair，供 Top-R 精排使用。
- `stochastic_pair_scores` 为 `0.5 * (g + q)`，全矩阵模式形状 `[Bt,Bv,K]`，对齐模式形状 `[P,K]`；文本、视频采样数必须相同。

- [ ] **Step 1: 写入形状、稳定性和数值等价测试**

测试覆盖：

```python
def test_soft_matcher_supports_rectangular_batches():
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)
    output = matcher(torch.randn(2, 4, 8), torch.randn(3, 4, 8))
    assert output.logits.shape == (2, 3)
    assert output.pair_uncertainty.shape == (2, 3)
    assert output.stochastic_pair_scores.shape == (2, 3, 4)


def test_soft_matcher_is_invariant_to_duplicate_identical_prototypes():
    text = F.normalize(torch.randn(2, 1, 8), dim=-1)
    video = F.normalize(torch.randn(3, 1, 8), dim=-1)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)
    one = matcher(text, video).logits
    four = matcher(text.expand(-1, 4, -1), video.expand(-1, 4, -1)).logits
    torch.testing.assert_close(one, four)


def test_score_pairs_matches_diagonal_of_full_matcher():
    text = torch.randn(3, 4, 8)
    video = torch.randn(3, 4, 8)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)
    paired = matcher.score_pairs(text, video)
    full = matcher(text, video)
    torch.testing.assert_close(paired.logits, full.logits.diagonal())
```

另加：归一化输入下输出有限；`pair_uncertainty >= 0`；全部原型相同则不确定性为 0；不相等的 K 抛错；小温度下不产生 NaN/Inf；soft 和 `hard_max=True` 两种模式均可反向传播。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_stochastic_prototype_ranking.py
```

Expected: collection error，提示模块或 matcher 不存在。

- [ ] **Step 3: 实现稳定的双向 logmeanexp**

全矩阵核心计算固定为：

```python
text = F.normalize(text_samples.float(), dim=-1, eps=self.eps)
video = F.normalize(video_samples.float(), dim=-1, eps=self.eps)
cosine = torch.einsum("tad,vbd->tvab", text, video)
log_k = math.log(text.size(1))
text_scores = self.temperature * (
    torch.logsumexp(cosine / self.temperature, dim=-1) - log_k
)
video_scores = self.temperature * (
    torch.logsumexp(cosine / self.temperature, dim=-2) - log_k
)
logits = 0.5 * (text_scores.mean(dim=-1) + video_scores.mean(dim=-1))
pair_uncertainty = 0.5 * (
    text_scores.var(dim=-1, unbiased=False)
    + video_scores.var(dim=-1, unbiased=False)
)
stochastic_pair_scores = 0.5 * (text_scores + video_scores)
```

`hard_max=True` 只用于 A5：分别将两个 `logsumexp-log(K)` 替换为原型维度 `max`，其他输出契约不变。

- [ ] **Step 4: 实现 Top-R 所需的对齐 pair 评分**

`score_pairs` 使用 `torch.einsum("pad,pbd->pab", text, video)`，不得通过先构造 `[P,P,K,K]` 再取对角线实现。

- [ ] **Step 5: 运行测试和静态检查并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_stochastic_prototype_ranking.py
python3 -m ruff check modules/stochastic_prototype_ranking.py tests/test_stochastic_prototype_ranking.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 6: 提交 matcher**

```bash
git add modules/stochastic_prototype_ranking.py tests/test_stochastic_prototype_ranking.py
git commit -m "feat: add stochastic prototype matcher"
```

### Task 3: 实现多正样本感知的随机排序损失

**Files:**
- Modify: `modules/stochastic_prototype_ranking.py`
- Modify: `tests/test_stochastic_prototype_ranking.py`

**Interfaces:**

```text
StochasticRankOutput(loss, inversion_probability, negative_indices)
StochasticRankLoss.forward(
    stochastic_scores: Tensor,
    group_ids: Tensor,
    mining_logits: Tensor,
) -> StochasticRankOutput
StochasticRankLoss.bidirectional(
    stochastic_scores: Tensor,
    group_ids: Tensor,
    mining_logits: Tensor,
) -> tuple[Tensor, StochasticRankOutput, StochasticRankOutput]
```

- 单向输入 `stochastic_scores` 为 `[B,B,K]`，`group_ids` 为 `[B]`，`mining_logits` 为 `[B,B]`。
- 每个 query 的正分数为同 `group_id` 候选随机分数的算术平均；hard negative 只能从不同 `group_id` 候选中按 `mining_logits` 选取。
- `hard_negative_count` 超过可用负例数时取全部有效负例；batch 内没有任何负例时抛出 `ValueError`。

- [ ] **Step 1: 写入多正样本排除和公式测试**

构造 `group_ids=[10,10,20,30]`，让候选 1 的 mining score 最大；断言 query 0 的负样本索引不含 0 或 1。再用手算张量验证：

```python
expected = F.softplus(
    (negative_scores - positive_scores + margin) / temperature
).mean()
```

测试还要覆盖：`inversion_probability` 落在 `[0,1]`；转置方向独立选择 hard negatives；损失可传回 `stochastic_scores`；group dtype 非整数、形状不匹配和无负例均抛出可读错误。

- [ ] **Step 2: 运行定向测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_stochastic_prototype_ranking.py -k "rank"
```

Expected: FAIL，提示 `StochasticRankLoss` 未实现。

- [ ] **Step 3: 实现单向 hard-negative mining 和随机排序**

实现顺序固定为：

```python
positive_mask = group_ids[:, None].eq(group_ids[None, :])
negative_mask = ~positive_mask
positive_count = positive_mask.sum(dim=1, keepdim=True)
positive_scores = (
    stochastic_scores * positive_mask.unsqueeze(-1)
).sum(dim=1) / positive_count

candidate_count = int(negative_mask.sum(dim=1).min().item())
negative_count = min(self.hard_negative_count, candidate_count)
negative_indices = mining_logits.detach().masked_fill(
    ~negative_mask, float("-inf")
).topk(negative_count, dim=1).indices
gather_index = negative_indices.unsqueeze(-1).expand(-1, -1, stochastic_scores.size(-1))
negative_scores = stochastic_scores.gather(dim=1, index=gather_index)
difference = negative_scores - positive_scores.unsqueeze(1)
loss = F.softplus((difference + self.margin) / self.temperature).mean()
inversion_probability = torch.sigmoid(difference / self.temperature).mean(dim=-1)
```

`bidirectional` 对 `[B,B,K]` 和 `[B,B]` 的前两维同时转置，返回两个方向损失均值及诊断。

- [ ] **Step 4: 运行完整 matcher/rank 测试并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_stochastic_prototype_ranking.py
python3 -m ruff check modules/stochastic_prototype_ranking.py tests/test_stochastic_prototype_ranking.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 5: 提交随机排序**

```bash
git add modules/stochastic_prototype_ranking.py tests/test_stochastic_prototype_ranking.py
git commit -m "feat: add multi-positive stochastic rank loss"
```

### Task 4: 组合 RSPR 核心并验证完整重参数化梯度链

**Files:**
- Modify: `modules/stochastic_prototype_ranking.py`
- Create: `tests/test_rspr_core.py`

**Interfaces:**

```text
RSPROutput(
    text_distribution,
    video_distribution,
    probabilistic_logits,
    pair_uncertainty,
    stochastic_pair_scores,
    anchor_kl,
)
RSPRCore.forward(
    text_tokens: Tensor,
    text_mask: Tensor,
    video_tokens: Tensor,
    video_mask: Tensor,
    *,
    sample_count: int,
    mean_only: bool = False,
    detach_samples: bool = False,
    text_noise: Tensor | None = None,
    video_noise: Tensor | None = None,
) -> RSPROutput
```

- RSPRCore 拥有独立的 `text_distribution` 和 `video_distribution`；二者不得共享参数对象。
- `anchor_kl = 0.5 * (text.anchor_kl + video.anchor_kl)`。
- `mean_only=True` 时 matcher 收到 K=1 的归一化均值原型。
- 构造时注册固定评估噪声 buffer：`fixed_text_noise` 和 `fixed_video_noise`，形状 `[K_eval,D]`，分别由 `eval_seed` 与 `eval_seed+1` 的局部 generator 生成，不修改全局 RNG。

- [ ] **Step 1: 写入端到端梯度与可复现测试**

测试必须分别断言 `probabilistic_logits.mean()` 与 stochastic rank loss 能产生以下有限梯度：

- 文本 mean head；
- 文本 logvar head；
- 视频 mean head；
- 视频 logvar head；
- 输入 `text_tokens` 和 `video_tokens`。

另测 `detach_samples=True` 时概率匹配不再回传上述分布梯度；固定 buffer 在两次 `.eval()` 调用中产生逐元素相同的 `samples/logits/pair_uncertainty`；保存再加载 `state_dict` 后输出仍相同。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_rspr_core.py
```

Expected: FAIL，提示 `RSPRCore` 或 `RSPROutput` 不存在。

- [ ] **Step 3: 实现 RSPRCore 和固定评估噪声**

训练时 `noise=None`，由两个分布头各自采样；评估时扩展已注册 buffer：

```python
text_noise = self.fixed_text_noise[:sample_count].unsqueeze(0).expand(text_tokens.size(0), -1, -1)
video_noise = self.fixed_video_noise[:sample_count].unsqueeze(0).expand(video_tokens.size(0), -1, -1)
```

buffer 按 Task 1 的 antithetic 顺序生成，`sample_count > eval_sample_count` 立即报错。

- [ ] **Step 4: 运行测试和静态检查并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py
python3 -m ruff check prob_models/reparameterized_distribution.py modules/stochastic_prototype_ranking.py tests/test_rspr_core.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 5: 提交核心编排**

```bash
git add modules/stochastic_prototype_ranking.py tests/test_rspr_core.py
git commit -m "feat: compose rspr probability core"
```

### Task 5: 将 pair_id 贯通训练 batch、DDP gather 和多正样本损失

**Files:**
- Modify: `main_task_retrieval.py:665-724`
- Modify: `modules/modeling.py:327-364,489-633,722-738`
- Modify: `modules/until_module.py:236-332`
- Create: `tests/test_rspr_pair_ids.py`

**Interfaces:**

- `_unpack_train_batch(batch)` 改为返回六项：`input_ids, input_mask, segment_ids, video, video_mask, group_ids`。
- 5/8 项旧 batch 返回 `group_ids=None`；6/9 项 batch 返回 loader 提供的整数 group ID。
- `UATVR.forward(input_ids, token_type_ids, attention_mask, video, video_mask=None, group_ids=None, rspr_rank_scale=1.0, rspr_anchor_scale=1.0)`。
- `MultiPositiveCrossEn.bidirectional(logits, text_group_ids, video_group_ids=None)` 支持方形同 ID 和矩形双 ID 两种情况；训练 DDP 方形路径仍只传一个全局 ID 张量。

- [ ] **Step 1: 写入 batch 解包和多正样本契约测试**

覆盖 5/6/8/9 项 batch，断言 group ID 不再被丢弃。为 `MultiPositiveCrossEn` 增加矩形 logits 测试，并构造 `group_ids=[1,1,2]` 证明同组两个位置都属于正集合。

增加一个纯函数：

```python
def gather_group_ids(group_ids, task_config):
    if group_ids is None:
        raise ValueError("RSPR training requires explicit group_ids")
    ids = torch.as_tensor(group_ids)
    if ids.dtype not in MultiPositiveCrossEn._INTEGER_DTYPES:
        raise ValueError("group_ids must use an integer dtype")
    return allgather(ids.reshape(-1).long(), task_config)
```

测试用 monkeypatch 的 `allgather` 返回两个 rank 的拼接结果，断言 ID 顺序与 feature gather 的 rank-major 顺序相同。

- [ ] **Step 2: 运行定向测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_rspr_pair_ids.py
```

Expected: 至少 `_unpack_train_batch` 返回值数量和矩形 `bidirectional` 契约失败。

- [ ] **Step 3: 修改 batch 解包和训练调用**

训练循环改为：

```python
input_ids, input_mask, segment_ids, video, video_mask, group_ids = _unpack_train_batch(batch)
warmup_epochs = max(args.rspr_warmup_epochs, 0.0)
progress_epoch = epoch + step / max(num_steps, 1)
rspr_warmup_scale = 1.0 if warmup_epochs == 0 else min(1.0, progress_epoch / warmup_epochs)
loss = model(
    input_ids,
    segment_ids,
    input_mask,
    video,
    video_mask,
    group_ids=group_ids,
    rspr_rank_scale=rspr_warmup_scale,
    rspr_anchor_scale=rspr_warmup_scale,
)
```

- [ ] **Step 4: 在模型中只 gather 一次 group IDs 并复用正例 mask**

当 `self.training` 且模式为 `off/mean/stochastic` 时，要求显式 group IDs；在进入 `_loose_similarity` 前 gather。特征仍由现有 `_loose_similarity` 按相同 rank 顺序 gather。`legacy` 模式允许无 group ID，以保持旧 checkpoint 的诊断兼容；有 ID 时同样使用多正样本确定性损失。

确定性损失改为：

```python
if global_group_ids is None:
    sim_loss = 0.5 * (self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T))
else:
    sim_loss, positive_mask = self.multi_positive_loss.bidirectional(
        sim_matrix, global_group_ids
    )
```

- [ ] **Step 5: 运行 pair ID 和既有损失测试并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_rspr_pair_ids.py
python3 -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py
python3 -m ruff check main_task_retrieval.py modules/modeling.py modules/until_module.py tests/test_rspr_pair_ids.py
```

Expected: 全部 PASS；Ruff 无新增诊断。

- [ ] **Step 6: 提交 pair ID 链路**

```bash
git add main_task_retrieval.py modules/modeling.py modules/until_module.py tests/test_rspr_pair_ids.py
git commit -m "feat: propagate retrieval pair ids"
```

### Task 6: 将 RSPR 核心接入 UATVR 训练目标和配置

**Files:**
- Modify: `main_task_retrieval.py:60-350,410-470,698-796`
- Modify: `modules/modeling.py:1-20,164-364,489-738`
- Modify: `experiment_tracking.py:114-165`
- Create: `tests/test_rspr_model_integration.py`
- Create: `tests/test_rspr_cli.py`

**CLI contract:**

```text
--rspr_mode {legacy,off,mean,stochastic}       default=legacy
--rspr_sample_count INT                        default=4
--rspr_eval_sample_count INT                   default=8
--rspr_match_mode {soft,hard}                  default=soft
--rspr_detach_samples                          default=false
--rspr_match_temperature FLOAT                 default=0.07
--rspr_prob_temperature FLOAT                  default=0.07
--rspr_rank_temperature FLOAT                  default=0.07
--rspr_hard_negatives INT                      default=8
--rspr_prior_std FLOAT                         default=0.1
--rspr_prob_weight FLOAT                       default=0.1
--rspr_rank_weight FLOAT                       default=0.1
--rspr_anchor_weight FLOAT                     default=1e-4
--rspr_warmup_epochs FLOAT                     default=1.0
--rspr_eval_seed INT                           default=0
--rspr_top_r INT                               default=100
--rspr_det_temperature FLOAT                   default=1.0
--rspr_rerank_temperature FLOAT                default=1.0
--rspr_rerank_weight FLOAT                     default=0.1
--rspr_pair_chunk_size INT                     default=4096
--rspr_freeze_clip                             default=false
--rspr_freeze_dsa                              default=false
```

验证规则：`mean` 强制 `sample_count=1`；`stochastic` 的训练/评估 K 必须为正偶数；三个温度、prior std 和 chunk size 必须大于 0；loss weights、warmup epochs 和 Top-R 不得小于 0；两个 freeze 开关只允许用于 `mean/stochastic`。`legacy` 记录但忽略其余 RSPR 专用数值参数，以便默认命令保持兼容。

- [ ] **Step 1: 写 CLI 默认值和非法组合测试**

将参数校验抽成 `validate_rspr_cli(args)`，在 `get_args()` 完成解析后调用。测试直接构造 `argparse.Namespace`，覆盖默认 legacy、mean K、奇数 stochastic K、非正温度、负权重、legacy 与 freeze 开关冲突，以及 `top_r=0`（允许，表示仅评估均值召回）。

- [ ] **Step 2: 写最小模型集成测试并确认 RED**

不加载完整 CLIP checkpoint；使用 `UATVR.__new__` + `nn.Module.__init__` 构造轻量 harness，注入伪 DSA token、`RSPRCore`、`MultiPositiveCrossEn` 和 rank loss。验证总损失严格等于：

```python
expected = (
    dsa_loss
    + args.rspr_prob_weight * probability_loss
    + args.rspr_rank_weight * rank_scale * rank_loss
    + args.rspr_anchor_weight * anchor_scale * anchor_kl
)
```

同时断言：`off` 不调用旧概率分支；`legacy` 只返回旧 MIL/KL；`mean` 不计算随机排序；`stochastic` 计算所有核心项；任何模式都不把 legacy MIL/KL 与 RSPR 核心相加。

Run:

```bash
python3 -m pytest -q tests/test_rspr_cli.py tests/test_rspr_model_integration.py
```

Expected: FAIL，提示参数、RSPR 初始化或 loss assembly 尚不存在。

- [ ] **Step 3: 在 UATVR 中按模式初始化概率路径**

导入 `MultiPositiveCrossEn` 和 `RSPRCore`。初始化规则：

```python
self.rspr_mode = task_config.rspr_mode
self.multi_positive_loss = MultiPositiveCrossEn()
if self.rspr_mode in {"mean", "stochastic"}:
    self.rspr = RSPRCore(
        dim=embed_dim,
        sample_count=task_config.rspr_sample_count,
        eval_sample_count=task_config.rspr_eval_sample_count,
        match_temperature=task_config.rspr_match_temperature,
        rank_temperature=task_config.rspr_rank_temperature,
        hard_negative_count=task_config.rspr_hard_negatives,
        prior_std=task_config.rspr_prior_std,
        hard_max=task_config.rspr_match_mode == "hard",
        eval_seed=task_config.rspr_eval_seed,
    )
else:
    self.rspr = None
```

旧 `PIENet/UncertaintyModuleImage/MILNCELoss_BoF/KLdivergence` 只在 `legacy` 模式初始化，避免 A0 和新核心携带未使用参数。

- [ ] **Step 4: 拆出 DSA 模态内 refinement helper**

把 `_loose_similarity` 中 `seqTransf` 的视频和文本处理分别移动到：

```text
UATVR._refine_video_tokens(
    visual_output: Tensor, video_mask: Tensor
) -> tuple[Tensor, Tensor]
UATVR._refine_text_tokens(
    text_token: Tensor, attention_mask: Tensor
) -> tuple[Tensor, Tensor]
```

两者保留现有 position embedding、extra token、transformer 和 residual 的逐行行为。新增回归测试：固定输入和权重下，refactor 前保存的张量 fixture 与 helper 输出 `assert_close(rtol=1e-6, atol=1e-6)`；该步骤不得顺便修改 WTI 算法。

- [ ] **Step 5: 在真实 token 区间调用 RSPR 并组装损失**

RSPR 输入只取 DSA 更新后原始长度部分：

```python
rspr_output = self.rspr(
    text_token[:, :word_num],
    attention_mask[:, :word_num],
    visual_output[:, :frame_num],
    video_mask[:, :frame_num],
    sample_count=(1 if self.rspr_mode == "mean" else self.task_config.rspr_sample_count),
    mean_only=self.rspr_mode == "mean",
    detach_samples=self.task_config.rspr_detach_samples,
)
probability_loss, _ = self.multi_positive_loss.bidirectional(
    rspr_output.probabilistic_logits / self.task_config.rspr_prob_temperature,
    global_group_ids,
)
```

stochastic 模式再以 `wti_logits.detach()` 选 hard negatives，调用 rank loss的 `bidirectional`。保留概率/排序梯度；只对 mining logits detach。

总损失组装和诊断放入独立方法 `_assemble_training_loss`，并将未经加权的标量保存到：

```python
self.last_loss_diagnostics = {
    "dsa": dsa_loss.detach(),
    "prob": probability_loss.detach(),
    "rank": rank_loss.detach(),
    "anchor": rspr_output.anchor_kl.detach(),
    "pair_uncertainty_mean": rspr_output.pair_uncertainty.mean().detach(),
    "text_variance_mean": rspr_output.text_distribution.logvar.exp().mean().detach(),
    "video_variance_mean": rspr_output.video_distribution.logvar.exp().mean().detach(),
}
```

- [ ] **Step 6: 在训练日志和 experiment manifest 记录 RSPR 配置**

有效参数日志增加 `RSPR` 分组；训练每 `n_display` 步在现有 loss/LR 后追加未加权的 `dsa/prob/rank/anchor` 和两个 variance mean。`experiment_tracking.build_experiment_manifest` 增加顶层 `rspr` 字典，显式记录上述所有 CLI 字段，禁止只依赖命令行日志。

在 `main()` 完成模型加载、创建 optimizer 之前应用冻结契约：

```python
if args.rspr_mode in {"mean", "stochastic"} and args.rspr_freeze_clip:
    for parameter in model.clip.parameters():
        parameter.requires_grad = False
if args.rspr_mode in {"mean", "stochastic"} and args.rspr_freeze_dsa:
    dsa_modules = (
        model.transformerClip,
        model.frame_position_embeddings,
        model.word_position_embeddings,
        model.text_weight_fc,
        model.video_weight_fc,
    )
    for module in dsa_modules:
        for parameter in module.parameters():
            parameter.requires_grad = False
```

集成测试断言：只冻结 CLIP 时 DSA 与 RSPR 仍可训练；同时冻结 CLIP/DSA 时 optimizer 的非空参数组中只剩 RSPR 参数。规范端到端训练固定 `freeze_layer_num=8`，CLIP 后 4 个 block 与 DSA、WTI、RSPR 从第一步联合训练。

- [ ] **Step 7: 运行集成测试并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_rspr_cli.py tests/test_rspr_model_integration.py tests/test_rspr_pair_ids.py
python3 -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py
python3 -m ruff check main_task_retrieval.py modules/modeling.py experiment_tracking.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py
```

Expected: 全部 PASS；Ruff 无新增诊断。

- [ ] **Step 8: 提交训练集成**

```bash
git add main_task_retrieval.py modules/modeling.py experiment_tracking.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py
git commit -m "feat: integrate rspr training objective"
```

### Task 7: 实现概率均值召回、固定噪声 Top-R 精排和方向独立评估

**Files:**
- Create: `modules/rspr_rerank.py`
- Create: `tests/test_rspr_rerank.py`
- Modify: `modules/modeling.py`
- Modify: `main_task_retrieval.py:833-1120`

**Interfaces:**

```text
TopRRetrievalOutput(
    text_to_video_logits,
    video_to_text_logits,
    mean_logits,
    pair_uncertainty,
)
rerank_top_r(
    deterministic_logits: Tensor,
    text_mean: Tensor,
    video_mean: Tensor,
    text_samples: Tensor,
    video_samples: Tensor,
    matcher: BidirectionalSoftPrototypeMatcher,
    *,
    top_r: int,
    deterministic_temperature: float,
    probabilistic_temperature: float,
    probabilistic_weight: float,
    pair_chunk_size: int,
) -> TopRRetrievalOutput
```

- 全库候选矩阵只能由 `normalize(text_mean) @ normalize(video_mean).T` 产生。
- T2V 对每个文本取 Top-R 视频；V2T 对每个视频取 Top-R 文本。两方向分别生成稀疏精排输出矩阵，未召回位置填 `-inf`，避免一个方向的候选并集污染另一个方向排名。该稀疏输出不得直接传入历史指标函数；标准指标使用下文定义的方向独立完整排名。
- 对 Top-R pair 使用 `matcher.score_pairs` 分块评分；不得构造全库 `[Bt,Bv,K,K]`。
- `top_r=0` 返回 mean-only 两方向分数，用于 A1；`top_r >= gallery_size` 只用于小规模单元测试和诊断。

- [ ] **Step 1: 写 Top-R 选择、方向隔离和复杂度契约测试**

测试构造 `Bt=3,Bv=5,K=4,D=8`，用 spy matcher 记录 `score_pairs` 的首维，断言总评分 pair 数不超过 `Bt*R + Bv*R`，且从未调用全矩阵 `forward`。分别验证 T2V/V2T 的非候选位置为 `-inf`，固定输入两次输出逐元素相同。另构造均值完全并列的候选，断言稳定排序按原始候选索引升序打破并列。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_rspr_rerank.py
```

Expected: collection error，提示 `modules.rspr_rerank` 不存在。

- [ ] **Step 3: 实现分块对齐 pair 评分和双方向 scatter**

T2V 和 V2T 都对 `mean_logits` 执行稳定降序排序后截取 Top-R；完全并列时保留原始候选索引升序。不得依赖未定义并列顺序的裸 `topk`。flatten 后按 `pair_chunk_size` gather 文本/视频样本并调用 `score_pairs`，最终分数严格使用：

```python
selected_final = (
    selected_deterministic / deterministic_temperature
    + probabilistic_weight * selected_probability / probabilistic_temperature
)
```

`pair_uncertainty` 只在已实际评分位置写数值，其他位置为 `NaN`，防止把“未计算”误解释为零不确定性。

- [ ] **Step 4: 为模型增加独立分布编码接口**

新增：

```text
UATVR.get_rspr_text_distribution(
    text_token: Tensor, attention_mask: Tensor
) -> DistributionOutput
UATVR.get_rspr_video_distribution(
    visual_output: Tensor, video_mask: Tensor
) -> DistributionOutput
```

两者分别调用 Task 6 的模态 refinement helper 和对应分布头，评估时自动使用固定 buffer。输出只包含真实 word/frame 区间，不让 extra DSA token 进入统计池化。

- [ ] **Step 5: 改造评估为两阶段输出**

`_run_on_single_gpu` 在 RSPR 模式下：

1. 复用当前 chunk 路径计算完整 `S_det`；
2. 每个文本 batch 只编码一次 text distribution；每个视频 chunk 只编码一次 video distribution；
3. 拼接 mean 与固定噪声 samples；
4. 调用 `rerank_top_r`；
5. 返回包含 `t2v/v2t/mean/uncertainty` 的 NumPy 字典。

`eval_epoch` 不得把含 `-inf` 非候选的稀疏精排输出直接传给历史指标函数。先为两个方向分别构造仅用于指标的完整排名矩阵：

1. 已召回的 Top-R 候选按该方向的 `selected_final` 稳定降序排列；
2. 未召回候选排在全部 Top-R 候选之后，并按同方向的 `mean_logits` 稳定降序排列；
3. 任意完全并列均由原始候选索引升序打破；
4. 指标矩阵中的每个真实 query 必须恰好计入一次，Top-R 漏召回必须记为失败，不得与多 caption padding 混淆。

随后对方向独立的完整指标矩阵分别计算：

```python
tv_metrics = compute_metrics(t2v_metric_matrix)
vt_metrics = compute_metrics(v2t_metric_matrix.T)
```

`t2v/v2t` 稀疏输出仍保留 `-inf`，完整指标矩阵不得回写或用候选并集污染它们。多 caption 评估对两个方向的完整指标矩阵分别执行相同的 sentence-to-video reshape，只有 reshape 新增的 padding 使用 `-inf`。DSL 与 RSPR Top-R 同时启用时立即报错，因为 DSL 会再次改变分数尺度和候选集合。历史 `legacy/off` 路径保持现有单矩阵行为。

- [ ] **Step 6: 增加固定推理噪声和重复排名集成测试**

使用 4 个文本、5 个视频的小张量连续调用评估 helper 两次，断言：mean logits、T2V logits、V2T logits、有限位置的不确定性和 `argsort` 排名完全相同。改变 `rspr_eval_seed` 后允许概率分数变化，但均值召回矩阵必须不变。

增加指标回归测试：当部分或全部真实 pair 未进入 Top-R 时，指标仍以全部真实 query 为分母且不抛异常；多 caption 下真实漏召回计为失败、padding 才被排除。增加完全并列的 mean/final 分数测试，断言候选选择、完整指标排名和重复评估排名均按原始索引稳定复现。

- [ ] **Step 7: 运行评估测试并确认 GREEN**

Run:

```bash
python3 -m pytest -q tests/test_rspr_rerank.py tests/test_rspr_model_integration.py
python3 -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py tests/test_rspr_pair_ids.py tests/test_rspr_cli.py
python3 -m ruff check modules/rspr_rerank.py modules/modeling.py main_task_retrieval.py tests/test_rspr_rerank.py
```

Expected: 全部 PASS；重复评估逐元素一致；Ruff 无新增诊断。

- [ ] **Step 8: 提交推理与精排**

```bash
git add modules/rspr_rerank.py modules/modeling.py main_task_retrieval.py tests/test_rspr_rerank.py
git commit -m "feat: add deterministic rspr top-r reranking"
```

### Task 8: 固化训练入口、A0–A8 消融矩阵和最终验收

**Files:**
- Modify: `run_train_msrvtt_bg.sh`
- Modify: `eval.sh`
- Create: `scripts/rspr_ablation_matrix.py`
- Create: `tests/test_rspr_ablation_matrix.py`
- Create: `docs/experiments/rspr-core-stage1.md`

**Interfaces:**

`scripts/rspr_ablation_matrix.py` 输出 A0–A8 的唯一参数映射：

```python
ABLATIONS = {
    "A0": ["--rspr_mode", "off"],
    "A1": ["--rspr_mode", "mean", "--rspr_sample_count", "1", "--rspr_top_r", "0"],
    "A2": ["--rspr_mode", "stochastic", "--rspr_detach_samples"],
    "A3": ["--rspr_mode", "stochastic"],
    "A4": ["--rspr_mode", "legacy"],
    "A5": ["--rspr_mode", "stochastic", "--rspr_match_mode", "hard"],
    "A6": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft"],
    "A7": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft", "--rspr_rank_weight", "0"],
    "A8": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft", "--rspr_anchor_weight", "0"],
}
```

脚本提供 `--ablation A3 --print-shell-args`，只输出 shell-safe 参数，不直接启动训练。

- [ ] **Step 1: 写消融映射和 shell 配置测试**

测试断言 A0–A8 无缺项，A2 与 A3 只有 detach 差异，A5 与 A6 只有 matcher 差异，A7 仅去 rank，A8 仅去 anchor，A4 只能走 legacy。测试还执行 `bash -n run_train_msrvtt_bg.sh eval.sh`。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python3 -m pytest -q tests/test_rspr_ablation_matrix.py
```

Expected: collection error，提示 `scripts.rspr_ablation_matrix` 不存在。

- [ ] **Step 3: 在训练/评估 shell 暴露显式 RSPR 环境变量**

`run_train_msrvtt_bg.sh` 增加默认值但保持 `RSPR_MODE=legacy`，并将所有 RSPR 参数显式传给 Python。`RSPR_FREEZE_CLIP` 与 `RSPR_FREEZE_DSA` 使用 0/1 环境变量并转换为可选 flag。对 stochastic 的 K、温度和 Top-R 在 shell 入口做同样的快速校验。启动日志打印模式、K、三项 loss weight、两个 freeze 开关、Top-R 和 eval seed。

规范 A3 使用单次端到端命令：

```bash
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=stochastic \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
FREEZE_LAYER_NUM=8 \
RUN_ID=rspr_a3_seed0 \
./run_train_msrvtt_bg.sh
```

该命令从 OpenAI CLIP `ViT-B/16` 权重初始化，在同一个 optimizer 和学习率计划内连续训练 5 epochs。DSA/WTI/RSPR 与 CLIP 后 4 个 block 从第一步联合训练，`L_DSA`、`L_prob` 立即生效，`L_rank`、`L_anchor` 在第一个 epoch 内线性 warm-up。A0–A8 共享相同 CLIP 起点、trusted split、总 epoch 数和优化日程，只改变消融矩阵定义的参数。

- [ ] **Step 4: 编写实验记录模板和停止条件**

`docs/experiments/rspr-core-stage1.md` 必须有以下表格列：数据协议 hash、git commit、A0–A8、seed、K、参数量、R@1/5/10、MdR、MnR、峰值显存、吞吐、Top-R 延迟、`logvar` min/mean/p50/p95/max、`U_pair` 错误 AUROC、重复评估排名一致率。

写明停止条件：任何 loss 出现 NaN/Inf、`logvar` 连续一个 epoch 全维触及 -8 或 2、A3 无法对 DSA/mean/logvar 产生有限梯度、固定噪声重复排名不一致时，停止主实验并先修复实现。

- [ ] **Step 5: 运行 CPU 完整测试和引用扫描**

Run:

```bash
python3 -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py tests/test_rspr_pair_ids.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py tests/test_rspr_rerank.py tests/test_rspr_ablation_matrix.py
python3 -m ruff check prob_models/reparameterized_distribution.py modules/stochastic_prototype_ranking.py modules/rspr_rerank.py modules/modeling.py modules/until_module.py main_task_retrieval.py experiment_tracking.py scripts/rspr_ablation_matrix.py tests
bash -n run_train_msrvtt_bg.sh
bash -n eval.sh
rg -n "TODO|FIXME|pass$|logsigma|PairwiseBetaEvidence|beta_evidence" prob_models/reparameterized_distribution.py modules/stochastic_prototype_ranking.py modules/rspr_rerank.py modules/modeling.py tests/test_rspr_*.py
```

Expected: pytest 全部 PASS；Ruff 与 Bash 语法检查通过；扫描不得在新 RSPR 路径命中占位符、`logsigma` 或 Beta evidence。`modules/modeling.py` 中允许 `logsigma` 仅存在于明确标记的 legacy 方法，检查时逐条确认命中均位于 legacy 分支。

- [ ] **Step 6: 运行单卡最小 GPU smoke test**

先打印命令而不执行训练：

```bash
python3 scripts/rspr_ablation_matrix.py --ablation A3 --print-shell-args
```

在可用 GPU 上用独立输出目录执行 20 个 optimizer step 的小规模 smoke run，验证：

- 总损失和四个未加权 loss 均有限；
- mean/logvar/DSA 梯度均有限；
- 样本范数接近 1；
- 显存没有随 step 单调增长；
- checkpoint 保存/加载后固定噪声评估完全复现。

Expected: 进程退出码 0；日志无 NaN/Inf；重复评估的分数最大绝对差为 0，排名一致率为 1.0。

- [ ] **Step 7: 按 A0–A8 顺序运行 MSR-VTT 消融**

先运行 A0、A1、A3、A6、A7、A8，确认核心递进关系；再补 A2、A4、A5。每个实验至少 3 个 seed，并使用同一 trusted split manifest。只有 A0–A8 结果、数值诊断和成本指标填写完成后，才允许创建 Beta evidence 的独立设计/实施计划。

- [ ] **Step 8: 建立第二数据集研究门槛**

当前仓库没有 DiDeMo/VATEX loader。本计划的代码完成不以临时拼接新数据协议为条件；完成 MSR-VTT A0–A8 后，创建独立的 DiDeMo 数据协议计划，复用相同模型 commit 和消融子集 A0/A1/A3/A6/A7/A8。论文结论必须至少在 MSR-VTT 与 DiDeMo 两个标准数据集上成立，才能宣称核心模块有效。

- [ ] **Step 9: 提交实验入口和文档**

```bash
git add run_train_msrvtt_bg.sh eval.sh scripts/rspr_ablation_matrix.py tests/test_rspr_ablation_matrix.py docs/experiments/rspr-core-stage1.md
git commit -m "experiments: add rspr core ablation protocol"
```

## Final Verification Checklist

- [ ] `rspr_mode=legacy` 保持旧概率分支及 MIL/KL 语义，并从与其他消融相同的 CLIP 权重起点训练。
- [ ] A0 不实例化或执行概率分支；A4 只执行旧概率分支；其他模式不混入 legacy MIL/KL。
- [ ] 分布头输出字段统一命名为 `mean/logvar`，且采样公式为 `mean + exp(0.5*logvar)*epsilon`。
- [ ] K=4 的 probability loss 和 rank loss 都能对文本/视频 mean head、logvar head 和 DSA token 产生有限非零梯度。
- [ ] matcher 在 `Bt != Bv`、K=1 和 K=4 下均通过形状与数值测试。
- [ ] DDP gather 后 positive mask 由全局 pair IDs 构造，同组 caption 永不作为负例。
- [ ] Top-R 只对召回 pair 调用 `score_pairs`，没有全库 K² 原型张量。
- [ ] 固定 eval seed 下两次评估分数逐元素相同、排名一致率 1.0。
- [ ] step 日志包含未加权 loss、variance mean 和 pair uncertainty；完整分位数与推理成本按需进入离线实验表。
- [ ] A0–A8 完成前代码中不存在 Beta evidence 实现。
- [ ] MSR-VTT 核心实验完成后，以独立计划接入 DiDeMo；未经两个数据集验证不进入论文结论。
