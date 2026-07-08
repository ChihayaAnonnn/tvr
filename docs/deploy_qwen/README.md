# deploy_qwen - 视频属性生成服务

> 本模块使用 Qwen3-VL（视觉语言模型）为视频生成结构化语义属性，作为 UATVR 模型 Query 分支的辅助输入。

---

## 模块概述

### 为什么需要这个模块？

纯视觉特征难以捕捉视频中的**高层语义信息**（如人物身份、动作描述、场景类型、OCR 文字等）。通过使用 VLM（Qwen3-VL-30B）预先分析视频帧，生成结构化的属性描述，可以为检索模型提供更丰富的语义线索。

### 生成的属性格式

```
【ENTITIES】
- main subject(s): a young man in casual clothes
- important objects: microphone, laptop

【ACTIONS】
- primary action(s): speaking, gesturing
- interactions: presenting to camera

【APPEARANCE & DETAILS】
- colors/materials/textures: blue shirt, black background
- shapes/parts/markings: round glasses

【SCENE】
- environment: indoor studio
- lighting/weather: artificial lighting
- camera/view: medium shot, frontal view

【TEXT/OCR】
- visible text/logos/brand: "LIVE" watermark
```

---

## 目录结构

```
deploy_qwen/
├── ../docs/deploy_qwen/README.md  # 📖 本文档（集中存放）
├── prompt_utils.py           # 🔧 Prompt 加载工具
│
├── prompts/                  # 📝 Prompt 模板
│   └── attributes_prompt.json    # 属性生成的系统/用户提示词
│
├── scripts/                  # 🚀 脚本集合
│   ├── start_server.sh           # 启动单个 vLLM 服务
│   ├── start_server_multi.sh     # 启动多 GPU 并行服务
│   ├── start_server_bg.sh        # 后台启动服务（被其他脚本调用）
│   ├── stop_server.sh            # 停止单个服务
│   ├── stop_server_multi.sh      # 停止所有服务
│   │
│   ├── generate_msrvtt_attributes.py   # MSRVTT 属性生成（核心逻辑）
│   ├── generate_msvd_attributes.py     # MSVD 属性生成
│   ├── run_msrvtt_multi_generate.sh    # MSRVTT 多进程并行生成
│   ├── run_msvd_multi_generate.sh      # MSVD 多进程并行生成
│   │
│   ├── merge_msrvtt_shards.py    # 合并 MSRVTT 分片结果
│   ├── merge_msvd_shards.py      # 合并 MSVD 分片结果
│   ├── monitor_*.sh              # 监控生成进度
│   │
│   ├── download_model.py         # 下载 Qwen 模型（huggingface）
│   └── test_client.py            # 测试 vLLM 服务连通性
│
├── env/                      # 🐍 环境配置
│   ├── install_vllm_env.sh       # 安装 vLLM 环境
│   ├── requirements_vllm.txt     # Python 依赖
│   └── proxy_env.sh              # 代理设置（可选）
│
├── models/                   # 🤖 模型权重
│   └── Qwen3-VL-30B-A3B-Instruct/    # Qwen3-VL MoE 模型
│
├── attributes/               # 📦 生成的属性文件
│   ├── msrvtt/
│   │   ├── final/                # 合并后的最终文件
│   │   │   ├── msrvtt_train9k_attributes.json
│   │   │   └── msrvtt_jsfusion_test_attributes.json
│   │   ├── shards/               # 分片文件（中间产物）
│   │   └── logs/                 # 生成日志
│   └── msvd/
│       └── ...
│
├── logs/                     # 📋 vLLM 服务日志
│   └── HHMMSS_vllm_Qwen3-VL-30B-A3B-Instruct_pXXXX.log
│
└── torch_wheels/             # 📦 离线 PyTorch 安装包
    ├── torch-2.1.2+cu121-*.whl
    ├── torchvision-0.16.2+cu121-*.whl
    └── triton-2.1.0-*.whl
```

---

## 快速开始

### 1. 环境安装

```bash
# 创建独立的 vLLM 环境（推荐）
conda create -n vllm python=3.10
conda activate vllm

# 安装依赖
cd deploy_qwen
bash env/install_vllm_env.sh
```

**依赖说明**：
- `vllm>=0.4.0`：高性能 LLM 推理引擎
- `torch==2.1.2+cu121`：CUDA 12.1 版本的 PyTorch
- `opencv-python`：视频帧提取
- `httpx`, `openai`：API 客户端

### 2. 模型准备

模型已预置于 `models/Qwen3-VL-30B-A3B-Instruct/`。如需重新下载：

```bash
python scripts/download_model.py --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

### 3. 启动 vLLM 服务

```bash
# 单 GPU 启动
bash scripts/start_server.sh

# 多 GPU 并行启动（推荐，加速生成）
bash scripts/start_server_multi.sh --gpus 0,1,2,3 --base_port 8000
```

**服务参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpus` | `0,1,2,3,4` | 使用的 GPU 列表 |
| `--base_port` | `8000` | 起始端口（每个 GPU +1） |
| `--max_model_len` | `32768` | 最大上下文（OpenClaw 等仅系统+工具即可能 >8k；OOM 可降到 16384） |
| `--gpu_mem_util` | `0.95` | GPU 显存利用率 |

### 4. 生成属性

```bash
# MSRVTT 数据集（多进程并行）
bash scripts/run_msrvtt_multi_generate.sh --num 4 --base_port 8000

# MSVD 数据集
bash scripts/run_msvd_multi_generate.sh --num 4 --base_port 8000

# 自定义参数
bash scripts/run_msrvtt_multi_generate.sh \
  --num 5 \
  --base_port 8000 \
  --num_frames 16 \
  --input_csv /path/to/your.csv \
  --output_dir /path/to/output
```

**生成参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num` | `4` | 并行进程数（应 ≤ GPU 数） |
| `--num_frames` | `16` | 每个视频采样帧数 |
| `--limit` | `-1` | 限制处理数量（-1=全部） |
| `--no_resume` | - | 禁用断点续传 |

### 5. 停止服务

```bash
bash scripts/stop_server_multi.sh
# 或单个停止
bash scripts/stop_server.sh
```

---

## 输出文件格式

### JSONL 格式（增量写入，支持断点续传）

```jsonl
{"video_id": "video7015", "video_path": "/data/.../video7015.mp4", "caption": "a man is talking", "num_frames": 16, "attributes": "【ENTITIES】\n- main subject(s): ..."}
{"video_id": "video7016", ...}
```

### JSON 格式（video_id -> attributes 映射）

```json
{
  "video7015": "【ENTITIES】\n- main subject(s): a young man\n...",
  "video7016": "..."
}
```

---

## 常见问题

### Q: 服务启动失败 / OOM

- 减小 `--max_model_len`（如 32768 → 16384）
- 减小 `--gpu_mem_util`（如 0.95 → 0.85）
- 使用更少的 GPU 或单 GPU 启动

### Q: 生成中断，如何续传？

脚本默认启用 `--resume`，会自动跳过已生成的 video_id。直接重新运行即可。

### Q: 如何修改 Prompt？

编辑 `prompts/attributes_prompt.json`，修改 `system` 或 `user` 字段。Prompt 中的 `{caption}` 会被替换为视频的原始描述。

### Q: 如何添加新数据集？

1. 参考 `scripts/generate_msrvtt_attributes.py` 创建新脚本
2. 实现 `load_video_ids_from_xxx()` 和 `load_captions_from_xxx()` 函数
3. 创建对应的 `run_xxx_multi_generate.sh` 和 `merge_xxx_shards.py`

---

## 与主模型的集成

生成的属性文件通过以下参数传递给训练/评估脚本：

```bash
# 训练时
--use_attributes \
--msrvtt_attributes_path /path/to/msrvtt_attributes.json

# 评估时
USE_ATTRIBUTES=1 \
ATTR_PATH=/path/to/msrvtt_jsfusion_test_attributes.json \
bash eval.sh
```

属性会被 `dataloaders/dataloader_msrvtt_retrieval.py` 加载，分割为多个语义块（ENTITIES、ACTIONS、SCENE、TEXT），作为 Query 分支的输入。
