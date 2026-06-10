# UATVR - Uncertainty-Aware Text-Video Retrieval

基于不确定性学习与语义锚点概率嵌入的视频-文本跨模态检索研究项目。

---

## 项目概述

当前代码主线以**课题工作一**为核心，围绕 Semantic Anchor Probabilistic Embedding（SAP）构建视觉检索主干；**课题工作二**的结构化属性增强与双视图融合管线已完成数据与接口准备，但主模型中的部分路径目前仍保留为兼容或预研代码。

UATVR 的当前核心创新：

1. **语义锚点概率嵌入（工作一）**：以可学习语义锚点通过 TransformerDecoder 探测视频帧特征，为每个锚点预测语义相关性与锚点级不确定性。
2. **不确定性感知概率检索（工作一）**：通过 Dirichlet 模态概率聚合锚点（softplus → alpha → modal_probs），以混合方差公式得到视频级均值与方差，结合文本侧 `PIENet`（视频侧已移除）、可选 `UncertaintyAdaNorm`、MIL/Evidential NLL/正交性正则进行联合优化。
3. **结构化属性增强（工作二预研）**：利用 Qwen3-VL 生成视频属性，数据加载与实验管线已支持属性输入，为后续双视图融合实验提供基础。QueryFormer 存在但未接入主模型。

**技术栈**：Python 3.10+ / PyTorch 2.x / CLIP ViT-B/16 / Transformer / Mamba（文本不确定性头可选） / RALA（空间增强） / Qwen3-VL-30B（属性生成）

**数据集**：MSRVTT (`/data2/hxj/data/MSRVTT`) / MSVD (`/data2/hxj/data/MSVD`)

---

## 目录结构

| 目录/文件 | 职责 |
|-----------|------|
| `main_task_retrieval.py` | 训练/评估入口，负责 SAP、概率嵌入、属性开关等参数配置 |
| `modules/modeling_mulit.py` | 当前主模型；工作一视觉概率检索主干已落地，保留部分工作二兼容接口 |
| `query_models/module_sap.py` | 工作一核心 SAP 模块：语义锚点探测、sigmoid 门控打分、混合方差聚合 |
| `modules/` | 核心组件：CLIP、Cross-Attention、Mamba、SpatialEnhancer、RALA、AdaNorm 等 |
| `modules/mus_util.py` | MUS（映射不确定性分数）批量计算，移植自 UMIVR |
| `prob_models/` | 概率建模与损失：PIENet、UncertaintyModule（attention/GRU/Mamba 三种头）、UncertaintyAdaNorm；`probemb.py`（PCME 风格损失，当前未被主模型使用）、`screening_utils.py`（工作二预研） |
| `query_models/` | 查询相关模块；`module_sap.py` 为工作一主路径，`module_query.py` / `head.py` 为工作二预研组件 |
| `tests/` | 单元测试：`test_modeling_mulit_losses.py`（evidential loss、WTI、TAS、VIB 测试） |
| `dataloaders/tqfs_util.py` | TQFS（基于时序质量的帧采样器），移植自 UMIVR，`slice_framepos=3` 时启用 |
| `dataloaders/` | 数据加载器（MSRVTT / MSVD），支持可选属性输入 |
| `deploy_qwen/` | Qwen VLM 视频属性生成管线 → 见 [`deploy_qwen/README.md`](deploy_qwen/README.md) |
| `logs/` | 日志目录 → 见 [`logs/README.md`](logs/README.md) |
| `ckpts/` | 检查点存储 |

---

## 设计决策

### 工作一：语义锚点概率嵌入

| 模块 | 代码落点 | 作用 |
|------|----------|------|
| **语义锚点探测（SAP）** | `query_models/module_sap.py` | 以可学习锚点通过 TransformerDecoder 探测帧级特征，输出锚点表示 `anchors`、视频级 `mu_raw`、`logsigma` 和逐锚点 `gate_scores` |
| **不确定性调制聚合** | `query_models/module_sap.py` | 逐锚点 Dirichlet 模态概率（softplus → alpha_dir → modal_probs）+ L2 归一化；方差按混合方差公式 `log(Σ α_i · σ_i²)` 聚合到视频级 |
| **概率表示增强** | `modules/modeling_mulit.py` / `prob_models/` | 视频侧直接使用 SAP 输出的 mu/logsigma（PIENet 已移除）；文本侧经 `PIENet` 增强，可选 `UncertaintyAdaNorm` 让方差反馈到均值表征 |
| **检索与正则目标** | `modules/modeling_mulit.py` / `prob_models/` | 联合优化锚点级 WTI 检索损失（CrossEn）、MIL 概率对比损失、Evidential NLL、负证据正则、不确定性矩阵正则、锚点正交性约束 |

工作一的核心动机是将视频不确定性从单一样本级方差细化到语义锚点级方差，让模型能够区分哪些局部语义可靠、哪些局部语义模糊，并直接将该不确定性用于锚点聚合与表示增强。

### 工作二：结构化属性增强的当前状态

`deploy_qwen/` 已支持离线生成结构化属性（实体、动作、场景、文字/OCR），`dataloaders/` 支持按 `--use_attributes` 加载属性文本（兼容 v1 中文括号格式与 v2 `SUBJECTS:/ACTIONS:` 格式）。  
但从当前 `modules/modeling_mulit.py` 实现看，属性编码、Query 分支和双视图融合主路径中仍有一部分代码处于注释保留状态，因此**当前稳定主线仍是工作一视觉概率检索框架**，工作二更适合视为已完成数据准备与接口预留的扩展方向。

### 历史债务

- `modeling.py` 是早期版本，当前主模型是 `modeling_mulit.py`
- `modeling_mulit.py` 文件名有拼写遗留，但仍是当前集成入口
- 工作一核心逻辑分散在 `modules/modeling_mulit.py`、`query_models/module_sap.py`、`prob_models/` 三处
- `prob_models/probemb.py` 中的 PCME 风格损失（`MCSoftContrastiveLoss` 等）当前未被主模型使用；主模型使用 `modules/until_module.py` 中的 `MILNCELoss_BoF`，evidential/neg_reg/uncertainty_reg/orth loss 在 `modeling_mulit.py` 内联计算
- `modeling_mulit.py` 中 `self.uncertain_net_video`（视频侧独立不确定性头）已注释掉；视频侧不确定性完全由 SAP 的锚点级方差聚合提供
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
| MSRVTT 训练汇报 | [`docs/report_msrvtt_training.md`](docs/report_msrvtt_training.md) | 面向导师的技术说明（工作一主线，`train_msrvtt.sh` 锚点） |
| 实验计划与消融 | [`plan.md`](plan.md) | 实验追踪表（13+ 组）、已验证结论、待实现方案 |
| 模型结构审计 | [`model_review.md`](model_review.md) | 2026-05-24 审计报告：5 个已修复问题 + 设计确认点 |
| SAP 分析 | [`query_models/analysis.md`](query_models/analysis.md) | SAP 模块详细分析笔记 |

---

## Learned User Preferences

- 与用户交流及撰写文档使用简体中文。
- 面向导师/课题组的技术说明采用学术汇报风格：结论先行、模块级说明，少工程细节。
- 从数据到训练再到模型的梳理，以 `train_msrvtt.sh` 为唯一叙事主线。
- 汇报文档少贴代码、不罗列目录树；代码仅用于示意关键公式或参数名。
- 文档与代码分析须严格依据仓库实际实现与配置，不编造未出现的机制；脚本中注释或未启用的路径须明确标注。
- 后台 MSRVTT 训练通常用 `run_train_msrvtt_bg.sh` 启动；GPU 通过环境变量 `CUDA_VISIBLE_DEVICES` 覆盖 `train_msrvtt.sh` 默认值。

## Learned Workspace Facts

- 本项目在 Fang et al. 2023 UATVR（ICCV）官方实现基础上深度分叉演进；上游参考为 `bofang98/UATVR`，原 `train.sh` 已拆为 `train_msrvtt.sh` / `train_msvd.sh`。
- MSRVTT 导师汇报主文档为 `docs/report_msrvtt_training.md`（课题工作一 SAP + 概率检索主线）。
- `train_msrvtt.sh` 默认 `batch_size=128`、`gradient_accumulation_steps=2`（2 卡 DDP 每卡 micro=64，有效 batch=128、in-batch 负样本 127）、`CUDA_VISIBLE_DEVICES=1,2`；`w_evidential=1e-2`、`w_neg_reg=5e-2`、`w_orth=0.1`、`w_uncertainty_reg=1e-3`、`w_query_sim=0.5`、`w_mil=1e-2`（argparse 默认）。`accum=1` 在 80GB 双卡下每卡 micro=128 易 OOM。
- 根目录 `.gitignore` 已忽略 `cache_dir/`（本地模型缓存，避免 Git/LFS 误纳入版本库）。
- Shell 中 `--batch_size` 表示目标有效 batch，不是 dataloader micro-batch；`main_task_retrieval.py` 解析时会先除以 `gradient_accumulation_steps`（L372）。全局 micro = shell batch_size/accum，每卡 micro = shell batch_size/(accum×GPU数)。
- In-batch 对比负样本由每次 forward 的 allgather 后全局 B 决定，accum 不合并对比矩阵；但本仓库改 accum 会同步改变 micro-batch，因此也会间接改变负样本数。显存主要由 micro-batch 决定，OOM 调优优先增大 micro-batch 而非堆 accum。
- MSRVTT 超参搜索：`w_evidential=0.01, w_neg_reg=0.05` 为最优配置（R@1=47.7）；`w_uncertainty_reg` 小幅正向。evidential_loss 与检索路径脱钩（作用于 mu_video 而非 WTI），已在 `uncertainty_mode=nig_mil` 下通过 NIG-MIL 方案修复（R@1=49.0）。
- `ref/` 目录存放参考论文实现：UMIVR（ICCV 2025）、DUQ（IJCAI 2025）、GARE（NeurIPS 2025）、UCoFiA（ICCV 2023）、Video-ColBERT（CVPR 2025）、TF-CoVR、BSCE-GRA、2026-AAAI-SCAN。
- 根目录 `plan.md` 记录待验证方向（多粒度特征、文本锚点、课程学习等）；`model_review.md` 为 2026-05-24 模型结构审计报告。
- 当前 `modules/modeling_mulit.py` 的视频概率分支直接使用 SAP 输出的 `mu_raw` / `logsigma` 采样；视频侧 `PIENet` 已移除，旧的 `pie_net_video` 路径曾导致视频 batch 维度 mismatch。
- SAP EvidentialUncertaintyHead 中 `beta_nig` 有上界（sigmoid 有界化，`beta_max=5.0`），但 `epistemic_cont = beta / (v * (alpha_nig - 1))` 仍未 clamp；NIG 分母过小时仍可能导致认知不确定性爆炸。
- `hyperparam_search.py` 已修复日志解析、失败 trial 隔离和成功结果去重；双卡搜索使用 `--batch_size=256 --gradient_accumulation_steps=2`，对应每卡 micro-batch 64、有效 batch 256。
