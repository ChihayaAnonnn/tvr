# UATVR - Uncertainty-Aware Text-Video Retrieval

基于不确定性学习与语义锚点概率嵌入的视频-文本跨模态检索研究项目。

---

## 项目概述

当前代码主线以**课题工作一**为核心，围绕 Semantic Anchor Probabilistic Embedding（SAP）构建视觉检索主干；**课题工作二**的结构化属性增强与双视图融合管线已完成数据与接口准备，但主模型中的部分路径目前仍保留为兼容或预研代码。

UATVR 的当前核心创新：

1. **语义锚点概率嵌入（工作一）**：以可学习语义锚点将视频帧特征分解为多个语义表示，并为每个锚点预测语义相关性与锚点级不确定性。
2. **不确定性感知概率检索（工作一）**：通过不确定性调制门控组合视频均值与方差，结合 `PIENet`、可选 `UncertaintyAdaNorm`、高斯采样、MIL/KL/多样性正则进行联合优化。
3. **结构化属性增强（工作二预研）**：利用 Qwen-VL 生成视频属性，数据加载与实验管线已支持属性输入，为后续双视图融合实验提供基础。

**技术栈**：Python 3.10+ / PyTorch 2.x / CLIP ViT-B/16 / Transformer / Mamba（文本不确定性头可选） / Qwen3-VL（属性生成）

**数据集**：MSRVTT (`/data2/hxj/data/MSRVTT`) / MSVD (`/data2/hxj/data/MSVD`)

---

## 目录结构

| 目录/文件 | 职责 |
|-----------|------|
| `main_task_retrieval.py` | 训练/评估入口，负责 SAP、概率嵌入、属性开关等参数配置 |
| `modules/modeling_mulit.py` | 当前主模型；工作一视觉概率检索主干已落地，保留部分工作二兼容接口 |
| `query_models/module_sap.py` | 工作一核心 SAP 模块：语义锚点探测、门控打分、锚点级方差估计 |
| `modules/` | 核心组件：CLIP、Cross-Attention、Mamba、SpatialEnhancer、AdaNorm 等 |
| `prob_models/` | 概率建模与损失：PIENet、高斯采样、KL/MIL 等 |
| `query_models/` | 查询相关模块；`module_sap.py` 为工作一主路径，`module_query.py` 等为工作二预研组件 |
| `dataloaders/` | 数据加载器（MSRVTT / MSVD），支持可选属性输入 |
| `deploy_qwen/` | Qwen VLM 视频属性生成管线 → 见 [`deploy_qwen/README.md`](deploy_qwen/README.md) |
| `logs/` | 日志目录 → 见 [`logs/README.md`](logs/README.md) |
| `ckpts/` | 检查点存储 |

---

## 设计决策

### 工作一：语义锚点概率嵌入

| 模块 | 代码落点 | 作用 |
|------|----------|------|
| **语义锚点探测（SAP）** | `query_models/module_sap.py` | 以可学习锚点对帧级特征执行自注意力与跨注意力探测，输出锚点表示 `q_n`、语义相关性 `g_n` 和锚点级 `logσ_n` |
| **不确定性调制聚合** | `query_models/module_sap.py` / `modules/modeling_mulit.py` | 按 `softmax(log g_n - β * mean(logσ_n))` 聚合锚点，得到视频级 `mu_raw` 与 `logsigma_video` |
| **概率表示增强** | `modules/modeling_mulit.py` / `prob_models/` | 视频侧经 `PIENet` 增强，并可通过 `UncertaintyAdaNorm` 让方差反馈到均值表征，再进行高斯采样 |
| **检索与正则目标** | `modules/modeling_mulit.py` / `prob_models/` | 联合优化锚点级 WTI 检索损失、MIL 概率对比损失、KL 正则与锚点多样性约束 |

工作一的核心动机是将视频不确定性从单一样本级方差细化到语义锚点级方差，让模型能够区分哪些局部语义可靠、哪些局部语义模糊，并直接将该不确定性用于锚点聚合与表示增强。

### 工作二：结构化属性增强的当前状态

`deploy_qwen/` 已支持离线生成结构化属性（如实体、动作、场景、文字），`dataloaders/` 也支持按 `--use_attributes` 加载属性文本。  
但从当前 `modules/modeling_mulit.py` 实现看，属性编码、Query 分支和双视图融合主路径中仍有一部分代码处于注释保留状态，因此**当前稳定主线仍是工作一视觉概率检索框架**，工作二更适合视为已完成数据准备与接口预留的扩展方向。

### 历史债务

- `modeling.py` 是早期版本，当前主模型是 `modeling_mulit.py`
- `modeling_mulit.py` 文件名有拼写遗留，但仍是当前集成入口
- 工作一核心逻辑分散在 `modules/modeling_mulit.py`、`query_models/module_sap.py`、`prob_models/` 三处
- `tmp/` 目录为临时实验代码，可忽略

---

## 操作指南

| 任务 | 入口 | 说明 |
|------|------|------|
| **训练** | `train_msrvtt.sh` / `train_msvd.sh` | 默认应优先视为工作一训练脚本；查看脚本内 SAP / 概率建模参数 |
| **评估** | `eval.sh` | 通过环境变量配置：`INIT_MODEL`, `EVAL_BRANCH_MODE`, `USE_ATTRIBUTES` |
| **生成属性** | `deploy_qwen/` | 详见 [`deploy_qwen/README.md`](deploy_qwen/README.md) |
| **查看日志** | `logs/` | 详见 [`logs/README.md`](logs/README.md) |

**常见问题**：
- OOM → 减小 `--batch_size` / `--max_frames`，或启用 `--fp16`
- 若要复现实验主线，优先确认 `query_models/module_sap.py` 与 `modules/modeling_mulit.py` 的工作一路径，而不是注释中的工作二分支
- 添加新数据集 → 参考 `dataloaders/dataloader_msrvtt_retrieval.py` 实现

---

## 相关文档

| 文档 | 路径 | 内容 |
|------|------|------|
| Qwen 属性生成 | [`deploy_qwen/README.md`](deploy_qwen/README.md) | 工作二属性数据准备、环境配置、服务启停、批量生成 |
| 日志说明 | [`logs/README.md`](logs/README.md) | 各类日志格式与分析方法 |
| 视频检索技能 | `.cursor/skills/video-text-retrieval/SKILL.md` | 注意力机制、不确定性学习指南 |
