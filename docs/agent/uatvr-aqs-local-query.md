# UATVR + AQS 局部 Query 软件接口

本模块实现 `uatvr-aqs-local-query-v1`；依据 [设计](../superpowers/specs/2026-10-29-uatvr-aqs-local-query-design.md) 与 [实施计划](../superpowers/plans/2026-10-29-uatvr-aqs-local-query-implementation.md)。这是软件能力，尚无新结构的 CUDA 资格或检索效果结论。

## 构造和配置

```python
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.local_model import build_local_model
from semantic_retrieval.aqs.fusion_config import FusionConfig

model = build_local_model(
    clip_config, clip_state,
    local=LocalQueryConfig(enabled=True, beta=beta, tau=tau),
    fusion=FusionConfig(enabled=False, eta=0),
    aqs=aqs_config, id_mapping_sha256=id_mapping_sha256,
    query_chunk=8, video_chunk=8,
    temporal_layers=4,
)
```

β≥0、τ≥1e-38且有限，必须显式提供；下界用于确保FP32温度运算可表示。没有科学推荐值。chunk默认8只控制工程内存，参与恢复身份绑定。生产CLIP沿用224/32、512维；CPU测试使用28/4、64维fixture。关闭局部开关或β=0，工厂直接返回旧 `AqsBaselineModel`，不构造query或改变旧随机消耗；此时继续使用旧AQS训练/恢复API。

融合 `enabled=true` 或 η非零均拒绝。新增模块仅引用 `fusion_config` 的配置数据，不导入Gaussian/KL评分实现；没有Laya、QueryMatch运行依赖。

## token与评分

视觉编码只执行一次，输出原CLS和49个projected patch。9个固定窗口中心为 `{1,3,5}²`，每窗3×3、步长2，水平/垂直相邻共享3个patch，对角相邻共享1个。query与8-head attention跨帧共享，没有局部时间交互或邻窗一致性损失。

`encode_text(ids,mask)` 额外返回 `local_tokens` 和去除BOS/EOS/padding的 `local_mask`；`encode_video(frames,frame_mask)` 额外返回 `[B,F,9,d]` 的 `local_tokens`。这些字典只含文本或视频自身的信息，可分别交给原 `score(text,video)` 入口；token缓存如需持久化，调用方必须绑定checkpoint、源码、预处理和窗口规范，当前模块不提供新的磁盘缓存管理器。

局部分数逐词对所有有效frame/window做余弦相似度softmax加权，再对有效词均匀平均。评分以FP32分块计算；反向时采用non-reentrant checkpoint重计算块内相似度，避免保留整个四维相似度图。最终 `[Q,V]` 分数矩阵仍需要内存。

旧全局logits已经乘过scale，新模型计算 `global_logits + exp(logit_scale)*beta*local_raw`。同一个局部分数算子参与双向CE、AQS和评估；选择用detached融合原始分数，选中分数保留梯度。负样本单位为视频，排除正视频身份；原MIL/legacy VIB原样保留。

## 软件训练与恢复

`local_training.local_step(model,network,optimizer,batch)` 使用原 `make_optimizer`，限定原有BertAdam实现，返回损失、AQS、有效token计数和query梯度范数等标量。network可以是model本身或直接包装它的DDP；rank配置和形状不一致、非有限输入/损失/梯度、错误优化器绑定会拒绝。BertAdam非空状态须为整数step、next_m、非负next_v；其他优化器及其moment格式不能混用。

`LocalBinding(base_identity,base_recipe,id_mapping_sha256)` 的base_recipe须包含 `local_recipe(model)` 所有字段，可追加sampler等输入协议。base_identity沿用底层完整的source/input/world/recipe哈希合同。绑定后的记录包含实际源码字节哈希、窗口与评分配置、父初始化来源以及优化器组内参数名称和顺序。

- `save_local_state(path,model,optimizer,binding,progress,store)`：独占发布 `.pt`、底层 `.json` 和 `.local.json`；checkpoint和ArtifactArrayStore必须在同一run root下，数值大数组复用现有外置规则。
- `restore_local_state(...)`：所有rank先只读校验完整payload、sidecar、数组引用、模型/优化器和全部rank RNG，再比较认证后的checkpoint内容哈希，全部一致才修改状态。身份不符、源码变化、损坏数组、单rank读失败或各rank读取不同合法快照时，拒绝前不改变模型、优化器或RNG。
- `initialize_local_parent(path,model,expected_sha256=...)`：只接受有合法 `.aqs.json` 和底层元数据的旧AQS checkpoint，严格加载全部公共参数，保留本次新增query初始化，不加载旧优化器、RNG或进度。返回并记录parent来源。此操作是初始化，不是续训；请在创建optimizer/LocalBinding之前调用。

旧KL checkpoint不能用本接口恢复或充当旧AQS parent。精确恢复时需在新模型上重现保存时的parent来源字段，再构造对应binding；该字段也在recipe中，不能省略或改写来源。

## 离线配方

`local_runtime.local_spec_from_recipe(recipe)` 只解析配置，返回不可变 `LocalRuntimeSpec`；不读取checkpoint、不创建目录、不初始化CUDA、不创建模型或worker。严格schema为 `uatvr-aqs-local-query-software-recipe-v1`，字段为：

```text
schema, local, fusion, aqs, model, chunks, sampler, identity,
id_mapping_sha256, initialization, dsl, test_authorized
```

`model` 为现有model_recipe格式，必须生产CLIP且fixture=false；额外文本token须为完整32-token文本保留位置，额外视频token须为12帧保留位置。模型编码接口还会按实际序列长度检查适配位置容量。`chunks={query,video}`；`initialization={parent,resume}`，最多一项非空，每项为 `{architecture,sha256}`。sampler包含policy/global_batch/seed/epochs；identity包含source_sha/input_sha256/world_size。DSL和test_authorized必须为false。inactive配方不得声明local_queries参数，返回旧架构语义；旧/新checkpoint绑定不能混用。

软件解析通过不表示实验被批准。正式执行前仍须读研究约束和对应study/registry，完成源码、预留、manifest、预算与CUDA资格核验；不授权test。
