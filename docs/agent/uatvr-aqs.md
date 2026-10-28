# UATVR + AQS 软件接口

实现包：`src/semantic_retrieval/aqs/`。设计与逐项实施计划位于 `docs/superpowers/specs/2026-10-29-uatvr-aqs-design.md` 和 `docs/superpowers/plans/2026-10-29-uatvr-aqs-implementation.md`。

本包提供训练模型、单步优化及严格 checkpoint API。没有科学训练 CLI、manifest/controller 扩展或新 study；CPU 合成验证不构成 CUDA 资格或检索收益证据。

## 数据和模型接入

输入沿用 BaselineModel 的五项：`input_ids`、`text_mask`、`video`、`frame_mask`、`video_ids`。视频是 `[B,F,3,H,W]`，ID 为 long `[B]`。训练 batch 在所有 rank 合并后必须视频身份唯一。沿用 `unique_video_batches`，每 step 为每段视频取一条描述，每 epoch 仍遍历全部描述；不兼容 caption-shuffle 因子臂。

`video_id_mapping(sorted(dataset.videos))` 返回整数映射与 SHA-256。必须核对返回映射等于 `CaptionDataset.video_ids`，再将摘要传给模型；不能仅凭整数相同就假定两个数据集的 ID 相同。辅助分支不把 ID 当语义输入。

```python
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.model import AqsBaselineModel
from semantic_retrieval.aqs.training import video_id_mapping

# 数据集/权重/超参数由调用方的认证配置提供；没有科学默认 M 或权重。
mapping, mapping_sha = video_id_mapping(sorted(dataset.videos))
if dataset.video_ids != mapping:
    raise ValueError('dataset canonical ID mapping differs')
model = AqsBaselineModel(
    clip_config, clip_state,
    aqs=AqsConfig(enabled=enabled, negatives=negative_count, weight=aux_weight),
    id_mapping_sha256=mapping_sha,
    temporal_layers=temporal_layers,
)
```

`AqsConfig.from_recipe` 严格读取完整 recipe，拒绝未知字段和固定算法常量漂移。`enabled=False` 或 `weight=0` 时跳过池化、选择和辅助损失；原有前向输出键、随机数消耗、损失及梯度保持原样。配置对象不可变，训练中不要替换模型配置或 ID 摘要。

模型构造时若分布式已初始化，会同步校验 rank 配置及 ID 映射摘要。若先构造模型再建 process group，所有 rank 必须在 DDP 包装前调用 `model.validate_rank_context()`。使用既有等长 rank-order 数据分片；模型内已完成全局特征/ID gather，不能再次 gather 或手工乘除 world size。

## 训练与诊断

```python
from semantic_retrieval.full_data.training import make_optimizer
from semantic_retrieval.aqs.training import aqs_step

optimizer = make_optimizer(model, learning_rate=learning_rate, total_steps=total_steps)
# 单进程 network=model；多进程使用 DDP(model, find_unused_parameters=True)。
metrics = aqs_step(model, network, optimizer, batch)
```

保留原有完整 batch T2V/V2T 检索损失、已加权 MIL 和 KL，新增 `weight * aqs`。不额外添加温度；使用原有缩放 logits。选择仅对 T2V 生效，从全局 batch 其他视频中贪心选择难度×多样性最大的候选，不跨 batch 挖掘。

AQS 难度按每条文本的有效负视频分数归一化；范围小于等于 `1e-8` 时难度置 1。多样性使用时序处理后真实帧 token 的池化方向，排除 padding 和聚合 token。并列按 canonical ID，禁止重复选取；B=1 时辅助损失为零。语义相关的不同视频仍可能是假负例，AQS 不能证明负例可靠。

返回标量：`loss`、`contrastive`、`mil`、`kl`、`aqs`、`aqs_weighted`、`aqs_active`、`mean_negatives`、`mean_difficulty`、`mean_uniqueness`、`empty_rows`。关闭时辅助项与计数记零，active 标志明确表示未执行。选择索引不嵌入 JSON 日志；需要持久化时由归档层外置数组并校验哈希。

`aqs_step` 保留原有冻结策略、梯度检查、裁剪和 logit scale clamp，并验证 optimizer 参数正好属于当前可训练模型，防止使用其他模型的 optimizer 形成无效训练。

## WTI 扩展与 CPU Gloo

`AqsCore.training_components(text, video)` 返回原有 logits/已加权 MIL/KL，以及停止梯度的原始 WTI 分数和帧池化表示。只编码一次，不增加特征 gather。vendor 文件和其 PROVENANCE 均未修改；`BaselineModel` 只提取一个默认不变的 core 类型工厂。

新 core 保留原 `_loose_similarity` 的算法，显式增加观察接口和 active-only 的反向内存布局节点。PyTorch 2.1.2 的 CPU Gloo all-gather 反向路径可能在接收非连续梯度时错误解释接收缓冲区布局；节点仅在 backward 调用 `.contiguous()`，不改变值、缩放或通信数量。辅助损失单独反传和完整双进程训练均有回归测试。AST 测试移除观察/布局扩展后与 vendor 方法完全一致；关闭 AQS 时不运行布局节点。

## Checkpoint 与恢复

`model_recipe(model)` 描述真实容量、聚合 token、采样数、batching 与可训练参数。调用方的 `base_recipe` 必须包含这个 `model` 字段，并包含采样器 seed、优化器等完整冻结配置；其 canonical JSON SHA-256 必须与认证 `base_identity.recipe_sha256` 一致。

```python
from semantic_retrieval.aqs.training import (
    model_recipe, CheckpointBinding, save_aqs_state, restore_aqs_state,
)

binding = CheckpointBinding(base_identity, base_recipe, mapping_sha)
save_aqs_state(checkpoint_path, model, optimizer, binding, progress, array_store)
# 在新建模型、同配方 optimizer 及 DDP 包装之后恢复。
progress = restore_aqs_state(checkpoint_path, fresh_model, fresh_optimizer, binding)
```

`CheckpointBinding` 内部保存 canonical 字节，读取字典属性返回副本。派生身份绑定 AQS 配置、实际模型 recipe、ID 映射及扩展源文件哈希；同一组 rank 必须使用相同身份。恢复拒绝 M、权重、映射、source SHA、world size、容量、冻结策略和 recipe 漂移，不使用 `strict=False`。

保留现有 full-data checkpoint schema、optimizer、各 rank RNG 和 epoch/batch cursor；另写不可覆盖的 `.aqs.json` 配方文件，明确记录 AQS 配置及代码身份。恢复先验证配方再修改模型/optimizer/RNG。保存中断可能留下未完成的出版事务，不能覆盖同名文件；正式运行须由归档层先 verify-run 后处理可证明一致的 transition。

数据面目录、manifest、source/input 身份、checkpoint 出版及大数组哈希由现有归档合同约束。这里的 API 不生成科学授权，也不将历史 baseline 权重加载等同于可恢复的 AQS 训练状态。

## Laya 隔离与推理

包不 import Laya 或 QueryMatch，不读其权重、processor、缓存、评分或逐查询输出，不借用其训练标签、伪监督和重排序。Laya 的 study、registry 和历史产物保留。隔离测试在新的 Python 进程中拦截这些模块和路径，再验证真实 tiny UATVR 的训练/推理。

推理使用继承的 UATVR `encode_text` / `encode_video` / `score`。没有 AQS 后处理，也没有 Laya Top-10 重排；DSL 按独立评估配置决定。固定同一权重、输入、精度和批处理路径时推理等价；新训练权重的排序当然可以改变。原有 singleton 文本池化 mask 别名限制没有在本任务中修复，不从 Laya 引入其 monkey-patch，也不声称跨任意 batch 形状逐位等价。

## 软件验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/aqs -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. conda run --no-capture-output -n tvr python -m pytest tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py tests/semantic_retrieval/full_data/test_factors.py tests/semantic_retrieval/full_data/test_audit.py -q
```

仅使用随机初始化的 tiny 模型和合成输入。`M=2, weight=0.2` 是测试常量，不是科学推荐。测试覆盖选择/梯度、关闭等价、双 rank Gloo、严格恢复、身份篡改与 Laya 隔离；不代表正式容量 CUDA 数值、性能或检索精度已验证。
