# 独立不确定性模块设计

## 目标

为视频-文本检索实现独立的、具备不确定性感知能力的 PyTorch 表征模块。首版先实现偶然不确定性（Aleatoric Uncertainty）模块，再实现认知不确定性（Epistemic Uncertainty）模块。本次交付不查看、不导入、不修改现有检索实现，也不接入现有主流程。

模块需要帮助区分以下三个问题：

1. 哪些 token 或帧在语义上重要？
2. 哪些已观测证据不可靠？
3. 模型何时缺乏相关知识？

## 范围

本次交付将在项目顶层新增独立的 `uncertainty_modules` Python 包，并包含单元测试和使用文档。模块只接收预先提取的特征张量，不包含 encoder、数据集、退化流程、检索损失、训练循环或主流程接线。

视频和文本通过不同实例使用同一个模态无关的 Aleatoric 类。因此，两种模态拥有独立参数，但 API 和行为保持一致。

## 包结构

```text
uncertainty_modules/
├── __init__.py
├── types.py
├── aleatoric.py
├── epistemic.py
├── losses.py
├── README.md
└── tests/
    ├── test_aleatoric.py
    ├── test_epistemic.py
    └── test_losses.py
```

该包只依赖 PyTorch 和 Python 标准库。测试使用仓库当前可用的测试运行器，但只测试这个独立包。

## 通用结果类型

`ProbabilisticEmbedding` 是不可变的结果容器，包含：

- `mean`：形状为 `[B, D]` 的语义嵌入。
- `variance`：形状为 `[B, D]` 的正对角方差。
- `uncertainty_score`：形状为 `[B]` 的方差汇总分数。

`AleatoricOutput` 包含概率嵌入，以及：

- `local_uncertainty`：形状为 `[B, N]` 的 token/帧级不确定性。
- `global_uncertainty`：形状为 `[B]` 的样本级不确定性。
- `relevance_weights`：形状为 `[B, N]`、只表达语义重要性的归一化权重。
- `aggregation_weights`：形状为 `[B, N]`、使用可靠性修正后的聚合权重。

`EpistemicOutput` 包含：

- `uncertainty_score`：形状为 `[B]` 的最终或单来源知识不确定性。
- 命名后的各来源分数；适用时还包含对角 disagreement variance。
- 使用 prototype 估计时，包含最近 prototype 的距离和索引。
- 只有在显式提供已校准阈值时，才包含可选的 OOD 判定。

## Aleatoric 模块

### 接口

`AleatoricUncertaintyModule(input_dim, hidden_dim, min_variance, max_variance)` 接收 `features: [B, N, D]` 和可选的布尔张量 `mask: [B, N]`，其中 `True` 表示有效元素。

构造函数拒绝非正的维度以及非法的方差范围。训练链路负责保证 `features` 与显式传入的 `mask` 满足上述接口约定；`forward` 不执行运行时输入校验。若 `mask=None`，`forward` 创建形状为 `[B, N]` 的全有效布尔 mask。

### 分离的双分支

输入特征分别进入两个相互独立的 MLP：

- relevance 分支输出标量 token logits。经过 masked softmax 后得到 `relevance_weights`，只表达语义重要性。
- uncertainty 分支输出形状为 `[B, N, D]` 的 token 级对角方差。通过正值参数化和配置的上下界保证方差有限且数值稳定。沿 `D` 维取均值得到 `local_uncertainty`。

可靠性定义为 `exp(-local_uncertainty)`。聚合权重是语义相关性权重与可靠性乘积的归一化结果。无效元素的权重严格为零。

可靠性进入聚合计算前执行 `detach`。因此，语义检索梯度会训练 relevance 分支，但无法通过 reliability 路径把 uncertainty 分支训练成第二套 attention。uncertainty 分支由专用的不确定性目标训练。

### 概率聚合

均值嵌入是使用聚合权重对原始输入特征求加权和的结果。其对角方差由两部分组成：

1. 按 `sum(weights**2 * token_variance)` 传播得到的 token 方差；
2. 根据样本级聚合表征预测的 global 对角方差。

最终方差被约束在配置的有限区间内。`global_uncertainty` 是 global variance 在特征维上的均值；概率嵌入的 `uncertainty_score` 是最终方差在特征维上的均值。

### 训练损失

独立包提供三个可选损失函数：

- `uncertainty_ranking_loss(clean, degraded, margin)`：当退化表征的不确定性没有高于干净表征时产生惩罚。
- `semantic_consistency_loss(clean_mean, degraded_mean)`：约束受控退化前后的语义表征保持一致。
- `variance_prior_loss(variance, target_log_variance)`：对 log variance 施加弱先验，避免方差长期贴住下界或上界。

方差下限负责数值安全；排序监督与弱先验负责从行为上抑制塌缩。若没有合适的训练目标，本模块不宣称仅凭结构就能阻止方差塌缩。

## Epistemic 模块

`EpistemicUncertaintyModule` 不依赖 encoder，并支持以下三种不确定性来源。

### Ensemble Disagreement

`from_ensemble(samples: [M, B, D])` 要求至少两个模型样本。该方法沿 `M` 维计算总体方差（`unbiased=False`），保留形状为 `[B, D]` 的对角方差，并沿 `D` 维取均值得到形状为 `[B]` 的样本分数。

### MC Dropout Disagreement

`from_mc_dropout(samples: [M, B, D])` 使用相同的稳定总体方差计算和输入校验。保留独立方法可以明确估计值的语义来源，也允许后续分别扩展，而无需让模块负责执行模型的 dropout 前向过程。

### Prototype Distance

`from_prototypes(features: [B, D], prototypes: [K, D])` 要求至少一个 prototype，且特征维度必须一致。默认指标是在数值安全归一化后计算最近余弦距离，同时支持 squared Euclidean distance。结果包含每个样本的最小距离和最近 prototype 索引。

### 组合与 OOD 判定

`combine` 接收可用的命名分量分数、显式非负权重，以及从验证集得到的校准统计量。每个分数先使用对应的均值和标准差做 z-score 标准化，再进行加权聚合。模块不会静默学习或硬编码任何权重、尺度或阈值。

只有调用方提供已校准阈值时，才返回 OOD 布尔判定；判定规则为 `combined_score > threshold`。非法权重、缺失统计量、标准差为零以及 batch 形状不一致都会触发明确异常。

## 数值与错误处理

- masked softmax 不允许无效位置参与归一化。
- 正方差先使用稳定、平滑的变换，再进行有限范围约束。
- 余弦距离使用带 epsilon 的安全归一化。
- Ensemble 和 MC 方差使用总体方差，避免小样本下出现 NaN。
- Aleatoric `forward` 信任固定训练链路的输入契约，不校验张量秩、batch、特征维度或显式 mask；Epistemic 公共方法仍校验采样次数、张量形状和 prototype 数量。
- 输出保持输入的 device 和浮点 dtype。

## 测试策略

实现遵循 red-green-refactor，并按以下顺序推进：

1. 通用结果容器和 Aleatoric 前向接口；
2. Aleatoric relevance、uncertainty、mask、聚合与梯度行为；
3. Aleatoric 损失；
4. Ensemble 和 MC Dropout 的 Epistemic 估计；
5. Prototype 估计；
6. 校准后的组合分数与 OOD 判定；
7. 使用文档和完整的包级验证。

Aleatoric 测试覆盖：输出形状、有限且为正的有界方差、默认全有效 mask、显式 mask 位置权重为零、有效位置权重归一化、单 token 和变长序列、dtype/device 保持以及 autograd。测试不再要求非法输入提前报错。

分支隔离测试验证：当损失只作用于 mean 时，relevance 参数能够得到梯度，而 uncertainty 参数不会通过 reliability pooling 路径接收梯度。

损失测试验证：退化样本不确定性越高，ranking loss 越低；相同输入的 semantic loss 为零；variance prior 在目标位置取得最小值。

Epistemic 测试验证：相同样本的 disagreement 为零，样本分歧增加时分数增大；两种距离指标都能得到正确的最近 prototype；校准后的加权组合、显式阈值 OOD 判定以及所有已记录的输入校验行为均正确。

## 验收标准

- 导入新包时不会导入项目现有检索代码。
- Aleatoric 和 Epistemic API 返回约定的形状和有限数值。
- relevance 与 uncertainty 参数在结构上相互独立，语义 mean 的梯度不会通过 reliability pooling 训练 uncertainty 分支。
- 视频和文本特征可通过同一个 Aleatoric 类的不同实例处理。
- Ensemble、MC Dropout 和 Prototype 三种 Epistemic 估计均已实现。
- 所有包内测试与 autograd smoke test 通过。
- 不修改任何现有项目代码，也不接入主训练或检索流程。

## 延后工作

以下内容延后到独立模块经过审阅和确认之后再考虑：与 encoder 接线、受控数据退化、损失权重选择、校准数据集选择、OOD 阈值拟合、检索重排序、选择性检索以及检索指标评估。
