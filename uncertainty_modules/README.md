# 不确定性模块

这是一个只依赖 PyTorch 的独立模块包。当前阶段实现 Aleatoric（数据固有）与 Epistemic（模型知识）不确定性，不包含 encoder、数据退化、训练循环、检索重排序或项目主流程接线。

## Aleatoric 输入与输出

`AleatoricUncertaintyModule` 接收：

- `features`：浮点张量 `[B, N, D]`；`N` 可以表示视频帧或文本 token。
- `mask`：可选布尔张量 `[B, N]`；`True` 表示有效元素。

`forward` 信任固定训练链路提供的输入，不执行 shape、有限值或显式 mask 校验；调用方需要保证输入满足上述约定。只有在 `mask=None` 时，模块才会创建全有效 mask。

返回 `AleatoricOutput`：

- `embedding.mean [B,D]`：不确定性感知聚合后的语义表示。
- `embedding.variance [B,D]`：有限、为正的对角方差。
- `embedding.uncertainty_score [B]`：最终方差的维度均值。
- `local_uncertainty [B,N]`：帧/token 级不确定性。
- `global_uncertainty [B]`：样本级不确定性。
- `relevance_weights [B,N]`：只由语义相关性分支产生的权重。
- `aggregation_weights [B,N]`：使用可靠性修正后的最终聚合权重。

视频和文本应分别创建实例，避免跨模态共享不确定性参数：

```python
from uncertainty_modules import AleatoricUncertaintyModule

video_uncertainty = AleatoricUncertaintyModule(
    input_dim=512,
    hidden_dim=256,
)
text_uncertainty = AleatoricUncertaintyModule(
    input_dim=512,
    hidden_dim=256,
)

video_output = video_uncertainty(video_tokens, video_mask)
text_output = text_uncertainty(text_tokens, text_mask)
```

## 双分支与梯度隔离

模块使用相互独立的 relevance 和 uncertainty 分支。最终聚合权重正比于：

```text
relevance_weight × exp(-local_uncertainty)
```

可靠性进入 pooling 前会执行 `detach`。因此，只作用于 `embedding.mean` 的语义检索损失不会通过可靠性路径把 uncertainty 分支训练成第二套 attention。uncertainty 分支需要由专门的不确定性损失训练。

## 可选训练损失

```python
from uncertainty_modules import (
    semantic_consistency_loss,
    uncertainty_ranking_loss,
    variance_prior_loss,
)

ranking = uncertainty_ranking_loss(
    clean_output.embedding.uncertainty_score,
    degraded_output.embedding.uncertainty_score,
    margin=0.1,
)
consistency = semantic_consistency_loss(
    clean_output.embedding.mean,
    degraded_output.embedding.mean,
)
prior = variance_prior_loss(
    degraded_output.embedding.variance,
    target_log_variance=-2.0,
)
```

- ranking loss 要求退化样本的不确定性高于干净样本。
- semantic consistency loss 约束退化前后的语义嵌入保持稳定。
- variance prior loss 对 log variance 施加弱先验，辅助避免长期贴住方差边界。

这些 loss 的组合权重需要在后续训练接入阶段根据验证实验确定，本独立包不预设权重。

## Epistemic 输入与输出

`EpistemicUncertaintyModule` 不执行 encoder 或随机前向过程，只消费调用方已经得到的表示。它支持三种知识不确定性来源：

- Ensemble disagreement：`ensemble_embeddings [M,B,D]`
- MC Dropout disagreement：`mc_dropout_embeddings [M,B,D]`
- Prototype distance：`features [B,D]` 与 `prototypes [K,D]`

Ensemble 和 MC Dropout 至少需要两次表示。模块沿 `M` 维计算总体方差（`unbiased=False`），返回 `diagonal_variance [B,D]` 及其维度均值 `uncertainty_score [B]`。

Prototype 默认使用最近余弦距离，也可以指定 `metric="squared_euclidean"`。输出包含最近 prototype 距离和索引。

```python
from uncertainty_modules import EpistemicUncertaintyModule

epistemic = EpistemicUncertaintyModule()

ensemble_output = epistemic.from_ensemble(
    ensemble_embeddings,
)  # [M,B,D]
dropout_output = epistemic.from_mc_dropout(
    mc_dropout_embeddings,
)  # [M,B,D]
prototype_output = epistemic.from_prototypes(
    features,
    prototypes,
)  # [B,D], [K,D]
```

这里的 `ensemble_embeddings`、`mc_dropout_embeddings` 和 `prototypes` 都由调用方产生；本模块不会持有或调用检索模型。

## Epistemic 校准组合与 OOD

不同来源的数值尺度通常不同。`combine` 使用调用方提供的验证集均值和标准差做 z-score，再按显式非负权重求加权均值：

```python
combined = epistemic.combine(
    {
        "ensemble": ensemble_output.uncertainty_score,
        "prototype": prototype_output.uncertainty_score,
    },
    {
        "ensemble": 1.0,
        "prototype": 1.0,
    },
    {
        "ensemble": (ensemble_mean, ensemble_std),
        "prototype": (prototype_mean, prototype_std),
    },
    threshold=ood_threshold,
)
```

- 权重、校准统计和 OOD 阈值均由调用方显式提供，模块不学习或硬编码这些值。
- `threshold=None` 时，`is_ood` 为 `None`。
- 提供有限阈值时，`is_ood` 等于 `uncertainty_score > threshold`。
- 所有分量必须具有相同的 `[B]` shape、device 和 dtype。

当前独立包不负责校准集选择、阈值拟合、检索重排序或主流程接线。

## CRF Degradation Ladder（`u_frame` / `u_video`）

`AleatoricUncertaintyModule` 的输出还被 CRF 退化阶梯（见 [`docs/superpowers/plans/2026-08-09-crf-degradation-ladder.md`](../docs/superpowers/plans/2026-08-09-crf-degradation-ladder.md)) 直接消费。术语约定：

- **`u_frame ≡ local_uncertainty`**：每帧的 aleatoric 不确定性 `[B, T]`。
- **`u_video ≡ Σ_t relevance_weights_t × local_uncertainty_t`**：按 relevance 加权的样本级 aleatoric 分数 `[B]`。CRF 训练/评估**只**监督这两个量。

其他张量在 CRF 计划下的语义：

- `embedding.variance` 仍然是模型输出的概率方差，本计划**不**监督它。任何"probabilistic variance 已校准"的声明都超出本计划的证据范围。
- `global_uncertainty`：由 `global_uncertainty_head` 计算；CRF 训练**冻结**这个 head（`.requires_grad_(False)`），使其不进入优化器。
- `propagated_variance`：从 aggregation weights 传播的方差；同样**不**由本计划监督。

### 训练契约

- 使用 `AleatoricUncertaintyModule(..., pooling_temperature=0.0)`。`pooling_temperature=0` 走 Python `if` 短路 `aggregation_logits = relevance_logits`，避免 `0 * -inf = NaN`。
- 排序损失见 `uncertainty_modules.losses.crf_ranking_loss`：clean vs. primary + 可选 order pair，四项 (`cf`, `cv`, `of`, `ov`)。`base_margin=0.05` 是启发式的单调间隙——**不是** CRF 损伤线性可分的声明。
- CRF 视频对由 `dataloaders/crf_pairing.py::CRFDistributedBatchSampler` 保证：每个本地 batch 的 order-pair 数固定，全局 batch 内 rank 计数最多相差 1。
- 视频编码路径按当前 CLIP 视觉编码器 + 冻结的 x264 recipe（见 plan Appendix A）范围内成立；换 encoder 或改 x264 参数都需要重新证伪。

### 推理契约

- `modules.crf_head.CRFTrainingHead.score_features(features, mask) -> (u_frame, u_video, relevance)` 和顶层 `UATVR.score_video_risk(video, video_mask)` 是唯一被诊断/评估 CLI（`scripts/evaluate_crf_ladder.py`、`scripts/diagnose_crf_training_contract.py`）调用的 CRF 推理入口。
- 标准检索评估**不**接收成对参数；`get_similarity_logits(...)` 的返回契约不变。

### 可接受阈值（预注册）

- CRF-audit `u_video` 三对准确率 ≥ 0.90；CRF50 vs CRF42 ≥ 0.80；否则报告用 plan `§`"extrapolation unreadable" 措辞而不是"blur detector"或"true image quality"。
- 挑战对 `(34,36)` 与 `(42,44)` 的 `(u − Laplacian)` 视频 bootstrap 95% CI low > 0，否则整体证伪。
- MSVD 三对准确率 ≥ 0.80，95% CI low > 0.5 才计为 cross-domain 成立；否则收窄为"in-domain 成立；cross-domain 未验证"。
