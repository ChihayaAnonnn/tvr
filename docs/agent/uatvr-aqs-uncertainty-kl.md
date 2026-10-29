# UATVR+AQS Gaussian KL 融合软件接口

评分身份：`uatvr-aqs-kl-v1`。使用现有 `tvr` 环境，无新增依赖。
本文说明软件接口；生产 GPU 资格、科学 eta 和研究运行尚未建立。

## 模型与训练

```python
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.aqs.fusion_model import FusionAqsModel
from semantic_retrieval.aqs.fusion_training import fusion_step
from semantic_retrieval.full_data.training import make_optimizer

# clip_config / clip_state 必须来自已校验的 CLIP 初始化；不是 checkpoint 的任意宽松加载。
# eta=0.1 仅作软件示例，不是经验证的科学默认值。
model = FusionAqsModel(clip_config, clip_state,
    aqs=AqsConfig(True,16,0.2), fusion=FusionConfig(True,0.1),
    id_mapping_sha256=id_mapping_sha256)
model = model.to(device)
optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=total_steps)
# 分布式时先初始化 process group；若模型构造更早，重新 validate_rank_context()。
# 每个 rank 使用相同配置，且 DDP 必须包装这个 model 实例。
metrics = fusion_step(model, network, optimizer, batch)
```

batch 为 `(input_ids,text_mask,video,frame_mask,video_ids)`，沿用原 AQS 输入。
完整全局 batch 的 video ID 必须唯一，sampler 不丢 caption，不构造语义伪负例标签。
每个 rank 返回自己的 scalar diagnostics（含 diagnostic_rank），调用者负责按 rank 记录/核对，不能混淆 rank0 与全局统计。

融合关闭或 eta0 时直接走旧 AQS 路径；AQS weight0 只关闭辅助选负，融合仍可影响主 CE 和推理。
训练保持原来已加权的 MIL 与 legacy VIB，不重复乘系数。
新功能没有修改 upstream UATVR、旧 AQS 实现、旧启动器或历史 checkpoint。

## 数值语义

旧头 `logsigma` 的采样意义为 log 标准差，新评分显式转换 `logvar=2*clip(logsigma,-4,2)`。
均值沿用原头的归一化输出。对称解析 KL 按维度归一化，以 `WTI-eta*D` 形成未缩放分数，再乘已有 logit scale。

旧 VIB 的参数解释与采样路径不一致，当前作为 legacy 训练项保留，并不声称已修复。
评分裁剪会使边界外梯度为0，训练日志记录两侧命中比例和 log 方差分位数。
新 KL 算子关闭 autocast 并用 FP32，未创建全量 pair×dimension 张量。
生产调用前仍须采用项目 FP32 配置（包括关闭 TF32）、经验证的 GPU 环境与资源合同；CPU 软件通过不代表 CUDA 已验证。

## 推理

`model.eval()` 后沿用 `encode_text/encode_video/score`。
新模式 `score(text,video)` 返回矩形融合 logits，不执行 AQS，也不进行 Gaussian 采样。
可按 query/video 分块；每块会重复准备其中的语义特征，本版没有新增跨块缓存。
评分函数保护 RNG，不原地写入调用者的 mask/token；singleton 行为通过独立隔离的 pooling 保持一致。
DSL 显式关闭。不得混用旧 WTI 缓存、阈值或评分身份。

## 配方与 checkpoint

```python
import hashlib
from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.aqs.fusion_training import (
    fusion_recipe, FusionBinding, save_fusion_state, restore_fusion_state,
)

recipe = dict(fusion_recipe(model), sampler=sampler_recipe)
identity = dict(source_sha=source_sha, input_sha256=input_sha256,
    recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest(),
    method='uatvr', world_size=world_size)
binding = FusionBinding(identity, recipe, id_mapping_sha256)
save_fusion_state(checkpoint, model, optimizer, binding,
    dict(epoch=epoch, batch_cursor=batch_cursor, update=update), store)
progress = restore_fusion_state(checkpoint, fresh_model, fresh_optimizer, binding)
```

源/输入身份必须由实际控制面证明，上述构造不自动证明研究授权。
`store` 为同一 run 的 `ArtifactArrayStore(run_dir/'arrays')`，checkpoint 放在 `run_dir/'checkpoints'`。
使用现有底层 checkpoint schema 和外置数组，新增专用 `.fusion.json`。
恢复核对融合参数、评分语义、ID、模型配方、源文件字节、输入和 world size。
缺失 sidecar、旧 `.aqs.json`、配置或 source 变化都拒绝，不能用 strict=False 绕过。
复用已有头使参数键可能相同，但不表示评分身份兼容。没有自动 warm-start 或恢复旧科学运行的功能。

## 离线配置与执行边界

`fusion_runtime.fusion_spec_from_recipe()` 只做严格、无 I/O 的软件 recipe 解析，返回不可变的 `FusionRuntimeSpec`。
schema 为 `uatvr-aqs-kl-software-recipe-v1`，字段固定为 schema/fusion/aqs/model/sampler/identity/dsl/test_authorized。
model 来自现有 model_recipe；sampler 为 policy/global_batch/seed/epochs；identity 为 source_sha/input_sha256/world_size。
必须显式 eta、DSL false、test_authorized false；旧 manifest 和未知字段拒绝。CLI/GPU controller 不在本版接口范围内。

没有引入 Laya 权重、processor、缓存、排名或运行时补丁。依赖隔离测试在禁止外部 import/路径、只允许 pytest 临时目录写入的新进程中运行真实 forward、eval、step 和 save/restore。

后续科学实验必须重新核对适用 AGENTS、研究文档、study/registry 和归档规则，按适用资格与研究 gate reserve-phase 并绑定精确源码/命令/输入/manifest。旧双卡运行不授予新评分链路资格，test 保持 sealed。Recall 与不确定性校准需独立论证。
