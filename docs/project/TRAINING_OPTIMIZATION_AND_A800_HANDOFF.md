# MSRVTT 训练优化与 A800 迁移报告

更新日期：2026-07-12

## 1. 目标与固定实验口径

本轮优化针对 OpenAI CLIP ViT-B/16、WTI、MSRVTT trusted-v1 基线在训练启动阶段暴露的三个工程问题：40GB GPU 首次 forward 显存溢出、在线视频解码导致 GPU 间歇等待，以及退化视频触发大量 scikit-learn `ConvergenceWarning`。

这些改动只优化显存和数据供给，不改变当前科研协议：

- trusted-v1：seed 42，8500 train / 500 internal val；
- 最终相似度固定为 WTI；
- 双向多正例 InfoNCE 按精确 `video_id` 构造正例；
- global forward batch 固定为 256；
- 4 GPU 时每卡 micro-batch 64，gradient accumulation 为 1；
- 训练期只使用 internal val，不读取 JSFusion test。

完整科研决策仍以 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 为唯一事实源。

## 2. 环境与依赖

项目新增根目录 `requirements.txt`，目标环境为 Linux x86_64、Python 3.10/3.11 和 NVIDIA GPU。主要固定版本如下：

| 类别 | 库与版本 | 用途 |
|---|---|---|
| 训练 | `torch==2.1.2+cu121` | DDP、activation checkpointing、模型训练 |
| 视觉 | `torchvision==0.16.2+cu121` | CLIP 图像预处理 |
| 数值 | `numpy==1.26.0` | TQFS 特征、缓存张量 |
| 图像/视频 | `Pillow==10.0.1`、`opencv-python==4.8.1.78` | 视频解码与帧预处理 |
| TQFS | `scikit-learn==1.3.2` | KMeans 帧多样性选择 |
| CLIP/工具 | `ftfy==6.1.1`、`regex==2022.7.9`、`tqdm==4.65.0`、`requests==2.31.0`、`boto3==1.28.85`、`einops==0.7.0` | tokenizer、下载和模型工具 |
| 可选 backbone 支撑 | `timm==0.9.16` | 显式启用本地 EVA-CLIP 时使用 |
| 验证 | `pytest==7.1.2`、`ruff==0.6.9` | 单元测试和静态检查 |

PyTorch 与 torchvision 改为 CUDA 12.1 的 CPython 3.11 可用组合，解决旧配置 `torch==1.13.0+cu116` / `torchvision==0.14.0+cu116` 在 Python 3.11 下找不到匹配 wheel 的问题。`xformers` 和 `mamba-ssm` 不属于当前 P0 必需依赖，因此没有默认安装。

安装命令：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装后检查：

```bash
python -c "import torch, torchvision, sklearn, cv2; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3. 训练优化内容

### 3.1 CLIP 权重路径可迁移

原实现把 ViT-B/16 权重目录硬编码为另一个项目路径，导致启动时找不到模型。现在默认读取当前仓库的 `.cache/ViT-B-16.pt`，也可以通过 `CLIP_CACHE_DIR` 指定外部目录。

Git 不包含该权重文件。在新机器上必须完成下列任一操作：

- 把权重放到 `<repo>/.cache/ViT-B-16.pt`；
- 或设置 `CLIP_CACHE_DIR=/absolute/path/to/clip/cache`，该目录下包含 `ViT-B-16.pt`。

### 3.2 选择性 activation checkpointing

40GB GPU 在 4 卡、每卡 micro-batch 64 的首次 forward 中接近占满显存并 OOM。为保持 global forward batch 256，不使用降低 micro-batch 或梯度累积绕过问题，而是在 OpenAI CLIP 视觉 Transformer 内增加逐 block 的非重入式 activation checkpointing：

- 默认只 checkpoint 视觉 Transformer 前 4 层；
- 文本 Transformer 不启用；
- 使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`；
- eval 阶段自动关闭；
- checkpoint 开关和层数写入实验 manifest；
- 新增输出、输入梯度和参数梯度一致性测试。

默认环境变量：

```bash
CLIP_GRADIENT_CHECKPOINTING=1
CLIP_VISUAL_CHECKPOINT_LAYERS=4
```

优化思路是只重计算显存占用较高的部分视觉 block，在“无 checkpoint 显存不足”和“全部视觉层 checkpoint 重计算过多”之间取得平衡，同时保留完整的单次 forward 对比学习负例集合。

### 3.3 DataLoader 与设备传输

数据流水线增加了以下设置：

- CUDA 可用时启用 pinned memory；
- DataLoader worker 大于 0 时启用 `persistent_workers=True`；
- 每个 worker 使用 `prefetch_factor=2`；
- worker 内把 PyTorch 和 OpenCV 线程限制为 1；
- shell 默认把 OMP、MKL、OpenBLAS、NumExpr 线程数限制为 1；
- 单卡和 DDP 都显式使用 `tensor.to(device, non_blocking=True)`。

优化思路是避免“DataLoader 多进程 × OpenCV/BLAS 多线程”的线程爆炸，并让页锁定内存、预取和异步 H2D 传输形成稳定流水线。

### 3.4 TQFS 单次解码与退化视频处理

旧的 `slice_framepos=3` 路径会重复解码同一视频，并可能先预处理全部帧，再对选中帧重复预处理。新路径调整为：

1. 以指定帧率解码一次原始帧；
2. 在低分辨率灰度特征上执行 TQFS；
3. 只对最终选中的帧执行 CLIP 预处理。

旧实现始终请求 8 个 KMeans cluster，但部分短视频只有 3 到 7 个不同帧特征，因此持续产生 `ConvergenceWarning`。新实现先计算实际不同特征数，令：

```text
cluster_count = min(requested_frames, distinct_feature_count)
```

每个实际 cluster 选质量最高帧，不足部分再用确定性均匀索引补齐。缺少 scikit-learn 时直接报错，不再静默退化成另一种采样算法。

### 3.5 共享、原子、可校验的 TQFS 缓存

新增 `TQFSFrameCache` 和离线构建脚本 `scripts/build_msrvtt_tqfs_cache.py`。缓存按 `video_id` 保存经过 OpenAI CLIP 归一化的 `float32 [T, 3, 224, 224]` `.npy` 张量。

缓存设计包括：

- train、val、不同 DDP rank 和 DataLoader worker 共享；
- 命中时完全跳过视频解码、TQFS 和图像预处理；
- 临时文件写完后通过原子替换发布，允许并发构建；
- `cache_config.json` 固定视频目录、采样帧率、最大帧数、分辨率、张量布局、归一化和算法版本；
- 配置、dtype、shape 或帧数不匹配时立即失败，避免误用旧缓存；
- 在线 miss 可以补写，离线脚本可恢复执行，已有条目记为 hit。

trusted-v1 train + val 共 9000 个视频。按每个视频 8 帧估算，完整缓存约占 40.4 GiB，需预留额外文件系统空间。

离线生成：

```bash
python scripts/build_msrvtt_tqfs_cache.py --workers 16
```

worker 数应根据存储调整：单机械盘建议 8 到 16，NVMe/高速阵列可从 16 测到 32；不要仅按 CPU 线程数设置为 64/128。使用 `iostat -xz 2` 观察输入/输出盘，磁盘 `%util` 接近 100% 且 await 持续上升时应减少 worker。

如需自定义缓存位置：

```bash
python scripts/build_msrvtt_tqfs_cache.py \
  --cache-dir /fast_storage/tvr_cache/msrvtt_trusted_v1_f1_m8_r224 \
  --workers 16
```

训练时设置同一个目录：

```bash
TQFS_CACHE_DIR=/fast_storage/tvr_cache/msrvtt_trusted_v1_f1_m8_r224
```

## 4. 现场工程观察

以下数据用于选择工程配置，不是受控科研结果，也不代表检索精度提升：

| 配置 | 40GB GPU 观察 | step 20 / 40 / 60 |
|---|---|---|
| 不启用 checkpoint | 首次 forward 接近 39.4 GiB 后 OOM | 未进入稳定训练 |
| 视觉 12 层全部 checkpoint | 约 9.2 GiB，重计算开销明显 | 8.36 / 5.51 / 6.89 秒每 step |
| 视觉前 4 层 checkpoint | 约 30.6 GiB | 4.15 / 2.80 / 3.36 秒每 step |

前 4 层方案在现场观察中相对全 12 层 checkpoint 约有 2 倍吞吐，并给 40GB GPU 留出约 9GiB 余量。计算阶段各 GPU 利用率可达到约 93% 到 100%；后续低利用阶段主要来自视频解码和 TQFS，因此引入离线缓存。

缓存 smoke test 对 2 个视频首次得到 `built=2, hit=0`，再次执行得到 `built=0, hit=2`，验证了恢复与命中路径。缓存优化后的完整训练吞吐需要在缓存生成完毕后重新测量。

## 5. A800 80GB 机器迁移步骤

### 5.1 拉取与准备

```bash
git pull origin main
python -m pip install -r requirements.txt
```

确认 NVIDIA 驱动能够运行 CUDA 12.1 PyTorch wheel，并检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

准备不随 Git 分发的内容：

- MSRVTT 数据集；
- OpenAI CLIP `ViT-B-16.pt`；
- 至少约 45 GiB 的 TQFS 缓存空间；
- 日志和 checkpoint 空间。

### 5.2 修改路径但不改实验语义

假设数据集位于 `/datasets/MSRVTT`、缓存位于 `/fast_storage/tvr_cache`：

```bash
export DATA_PATH=/datasets/MSRVTT
export TQFS_CACHE_DIR=/fast_storage/tvr_cache/msrvtt_trusted_v1_f1_m8_r224
export CLIP_CACHE_DIR=/models/openai_clip
```

`DATA_PATH` 应具有下列结构：

```text
MSRVTT/
├── annotation/MSRVTT_v2.json
├── csv/MSRVTT_train.9k.csv
├── csv/MSRVTT_JSFUSION_test.csv
└── videos/compressed_videos/msrvtt_224_12fps/
```

先生成缓存：

```bash
python scripts/build_msrvtt_tqfs_cache.py \
  --features-path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps" \
  --cache-dir "${TQFS_CACHE_DIR}" \
  --workers 16
```

### 5.3 可信基线启动方式

即使机器有超过 4 张 A800，当前 P0 也使用 4 张卡，保持每卡 micro-batch 64 和 global forward batch 256：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
DATA_PATH=/datasets/MSRVTT \
TQFS_CACHE_DIR=/fast_storage/tvr_cache/msrvtt_trusted_v1_f1_m8_r224 \
CLIP_CACHE_DIR=/models/openai_clip \
bash run_train_msrvtt_bg.sh
```

启动后日志中必须核对：

```text
OpenAI CLIP gradient checkpointing: True (visual_layers=4, ...)
Samples: 170000 | Batch: 64 x 4 GPUs x 1 accum = 256 eff
eval_split=val
```

不要通过增加 GPU 数、降低每卡 micro-batch 或增加 accumulation 改变当前基线口径。

### 5.4 A800 可选吞吐对照

A800 80GB 理论上可容纳不启用 checkpoint 的每卡 micro-batch 64。完成同口径基线后，可以单独做工程吞吐对照：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
CLIP_GRADIENT_CHECKPOINTING=0 \
DATA_PATH=/datasets/MSRVTT \
TQFS_CACHE_DIR=/fast_storage/tvr_cache/msrvtt_trusted_v1_f1_m8_r224 \
CLIP_CACHE_DIR=/models/openai_clip \
bash run_train_msrvtt_bg.sh
```

该运行必须使用新的 `RUN_ID`/输出目录，并在结果中明确标注 checkpoint 关闭。先观察至少 100 个 step 的 `s/step`、显存、GPU 利用率和 loss 有限性，再决定是否把它作为 A800 的固定工程配置；不要把两种配置的中断 checkpoint 或吞吐数据混合记录。

## 6. 验证与验收标准

本轮代码增加或扩展了以下测试契约：

- checkpoint 前后输出与梯度一致；
- eval 不调用 checkpoint；
- 只 checkpoint 请求的前 N 层；
- CLI 参数和实验 manifest 记录完整；
- TQFS 缓存 round-trip、配置冲突、非法 dtype/shape/帧数拒绝；
- 缓存命中不触发解码；
- 退化视频不产生 KMeans convergence warning；
- MSRVTT train/val/test dataloader 正确透传缓存配置。

迁移机器的最低验收标准：

1. `pip install -r requirements.txt` 成功；
2. PyTorch 能识别预期数量的 A800；
3. 完整 TQFS 缓存构建完成，再次运行全部显示 hit；
4. 日志确认 4 GPU、每卡 64、accumulation 1、global forward batch 256；
5. 前 100 step 无 OOM、NaN/Inf、数据解码异常和重复 sklearn warning；
6. 训练期只评估 internal val。
