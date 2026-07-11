# A800 训练优化设计

## 1. 目标

在当前 5 张 NVIDIA A800 80GB PCIe 机器上迁移 A100 训练优化，同时保持
MSRVTT `trusted-v1` P0 科研口径不变。优化对象仅限数据供给、设备拓扑、显存工程
开关、启动前校验和运行记录，不改变模型、损失、采样语义、训练数据或评估协议。

## 2. 固定约束

- 唯一科研事实源仍是 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`。
- P0 使用 OpenAI CLIP ViT-B/16、WTI 和按精确 `video_id` 构造的双向多正例
  InfoNCE。
- P0 固定 4 GPU、每卡 micro-batch 64、global forward batch 256、gradient
  accumulation 1；第 5 张 GPU 不加入训练进程。
- 默认继续对 OpenAI CLIP 视觉 Transformer 前 4 层使用 activation
  checkpointing。A800 上关闭 checkpoint 只作为单独命名、单独输出目录的吞吐
  对照，不替代默认可信基线。
- 训练期只构造 internal-val dataloader，不读取 JSFusion test dataloader。
- 不启动或代跑长期训练；最终只向用户提供单行训练命令。
- 保持当前 `torch==2.1.2+cu121` 不变。只安装缺失的 `timm==0.9.16`，不降级
  环境中已安装的其他项目依赖。

## 3. 机器事实与资源选择

当前机器有 5 张 A800 80GB。GPU 0/1 位于 NUMA 0；GPU 2/3/4 位于 NUMA 1；
GPU 2 与 GPU 4 之间有 NV8，GPU 2 与 GPU 3 仅为 PHB。P0 默认 GPU 集合选择
`0,1,2,4`，相较 `0,1,2,3` 保留同样的跨 NUMA 边界，同时让 NUMA 1 内的一对
训练卡使用更好的互联。GPU 3 保留，不参与该训练的 DDP world。

项目与视频数据位于机械盘 `/data2`。完整 TQFS 缓存约 40.4 GiB，放到 NVMe：
`/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224`。系统盘实施前有约
125 GiB 可用空间；缓存构建前检查至少 50 GiB 可用空间，避免把系统盘逼近满盘。

## 4. 方案设计

### 4.1 启动配置与协议保护

`train_msrvtt.sh` 保持唯一训练脚本，并增加显式、可覆盖的工程参数：

- `CUDA_VISIBLE_DEVICES` 默认 `0,1,2,4`；
- `NPROC` 默认从可见 GPU 数量推导，但 P0 hygiene 启动必须验证为 4；
- `TRAIN_NUM_WORKERS` 控制每个 rank 的 DataLoader worker 数，默认 8；
- `TRAIN_PREFETCH_FACTOR` 控制每个 worker 的预取批次数，默认 2；
- `TQFS_CACHE_DIR` 默认指向上述 NVMe 路径；
- checkpoint 仍由现有 `CLIP_GRADIENT_CHECKPOINTING` 和
  `CLIP_VISUAL_CHECKPOINT_LAYERS` 控制。

脚本在启动 `torchrun` 前验证 hygiene P0 的 world size、batch、accumulation 和 GPU
列表无重复项。默认值直接满足科研口径；非法配置在创建训练进程前终止。关闭
checkpoint 时打印醒目的工程对照标记，并要求该运行使用独立 `RUN_ID` 和输出目录；
不尝试自动比较或合并 checkpoint。

### 4.2 DataLoader 配置传递

现有 `_video_loader_kwargs` 已实现 CUDA pin memory、persistent workers、worker
线程限制和固定 prefetch 2。本次把 prefetch 从硬编码值改为命令行参数，并保持
worker 为 0 时不传 `prefetch_factor`/`persistent_workers` 的合法行为。

`main_task_retrieval.py` 增加 `--prefetch_factor`，取值必须为正整数；
`--num_thread_reader` 继续表示每个 rank 的 worker 数。训练 shell 把
`TRAIN_NUM_WORKERS` 与 `TRAIN_PREFETCH_FACTOR` 映射到这两个参数。这样可在短 smoke
中比较 8/2、12/2 或 8/4，而无需修改代码，也不会改变模型数值语义。

实验 manifest 的运行配置记录 worker 数、prefetch factor、pin memory 是否启用、
persistent workers 是否启用、可见 GPU 列表与 checkpoint 设置，使吞吐结果能够
追溯。这里记录的是工程运行事实，不作为科研方法变量。

### 4.3 TQFS NVMe 缓存

保留现有 `TQFSFrameCache` 的配置契约、原子发布和严格校验。离线缓存脚本默认
cache 目录改为 NVMe 路径，同时保留 `--cache-dir` 覆盖入口。构建开始前检查：

1. cache 目录所在文件系统至少有 50 GiB 可用空间；
2. `--workers` 为正整数；
3. trusted-v1 train/val 视频 ID 无重复；
4. 现有 `cache_config.json` 与当前采样配置完全一致。

空间不足时在创建 worker pool 前失败。完整缓存不由代理自动长时间构建；只运行
少量视频的 smoke test，最终交付可恢复执行的完整构建命令。缓存再次运行应报告
全部已构建条目为 hit。

### 4.4 A800 checkpoint 对照

默认基线仍为 `CLIP_GRADIENT_CHECKPOINTING=1`、前 4 层。另提供明确的 A800
吞吐对照命令，唯一工程变量为 `CLIP_GRADIENT_CHECKPOINTING=0`。用户手动运行时先
观察至少 100 step 的秒/step、峰值显存、GPU 利用率和 loss 有限性。只有对照显示
稳定且更快，后续才可通过单独科研决策修改机器级推荐；本次不修改 SSOT 默认值。

## 5. 错误处理与安全边界

- 训练脚本拒绝 hygiene P0 使用非 4 卡、重复 GPU ID、非 256 batch 或非 1
  accumulation。
- CLI 拒绝负 worker 数和非正 prefetch factor。
- cache 空间检查使用 cache 目录所在文件系统的实际可用字节，不根据路径前缀猜测。
- cache 配置、dtype、shape 或帧数不匹配时继续立即失败，不静默重建或降级采样。
- 不修改或提交用户已删除的 `docs/reference/uatvr_backbone_upgrade_strategy.md`。
- 不安装 `xformers`、`mamba-ssm`、JAX 或其他 P0 不需要的依赖。

## 6. 测试与验收

实现遵循测试先行：先添加失败测试，再完成最小实现。

自动测试覆盖：

- DataLoader 正确接收自定义 prefetch factor，worker 为 0 时不传非法选项；
- CLI 拒绝非法 worker/prefetch 值；
- manifest 记录完整的数据流水线与 GPU 工程配置；
- shell 默认使用 `0,1,2,4`、NVMe cache、8 workers、prefetch 2；
- shell 在 hygiene P0 非 4 卡或 GPU ID 重复时于 `torchrun` 前失败；
- 缓存构建在空间不足时于 worker pool 创建前失败；
- 现有 checkpoint 输出/梯度一致性与 trusted-v1 隔离测试保持通过。

验收命令包括项目限定测试
`/home/xujie/miniconda3/envs/ret/bin/pytest -q tests`、修改文件的 Ruff 静态检查、
shell 语法检查、依赖导入检查、GPU 拓扑检查，以及 TQFS cache 的少量视频首次构建和
二次命中 smoke。不会用训练结果或长期训练来宣称本次工程修改完成。

## 7. 交付物

- A800 优化后的训练/DataLoader/cache 配置与回归测试；
- `ret` 环境中补装的 `timm==0.9.16`；
- NVMe cache smoke 结果及完整缓存构建单行命令；
- 默认可信基线单行训练命令；
- checkpoint-off A800 吞吐对照单行命令；
- 明确列出未处理的环境既有 `pip check` JAX 告警。
