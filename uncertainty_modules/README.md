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
