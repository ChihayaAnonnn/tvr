# UATVR - Uncertainty-Aware Text-Video Retrieval

基于不确定性学习与语义锚点概率嵌入的视频-文本跨模态检索研究项目。

---

## 项目概述

当前代码主线围绕 Semantic Anchor Probabilistic Embedding（SAP）构建视觉检索主干。

UATVR 的核心创新：

1. **语义锚点概率嵌入**：以可学习语义锚点通过 TransformerDecoder 探测视频帧特征，为每个锚点预测语义相关性与锚点级不确定性。
2. **不确定性感知概率检索**：通过 Dirichlet 模态概率聚合锚点（softplus → alpha → modal_probs），以混合方差公式得到视频级均值与方差，结合文本侧 `PIENet`（视频侧已移除）、可选 `UncertaintyAdaNorm`、MIL/Evidential NLL/正交性正则进行联合优化。

**技术栈**：Python 3.10+ / PyTorch 2.x / CLIP ViT-B/16 / Transformer / Mamba（文本不确定性头可选） / RALA（空间增强）

**数据集**：MSRVTT (`/data2/hxj/data/MSRVTT`) / MSVD (`/data2/hxj/data/MSVD`)

---

## 当前实验状态（2026-06-22）

完整实验档案见 [`STATUS.md`](STATUS.md)。新会话优先使用以下当前结论：

- 当前工作树为 **B1-only v2**：主检索分数退回 `weighted_logits = wti_logits`，per-pair `confidence_mlp` 已移除。
- 2026-06-19 三组复现实验均已结束，均未复现历史 50.0：

| 实验 | 最佳 checkpoint | Best T2V R@1 | 对应 V2T R@1 | 结论 |
|------|----------------|:---:|:---:|------|
| `B1only_v2_repeat1` | `pytorch_model.bin.4` | **49.3** | 47.6 | 当前三组最优，但未到 50 |
| `exp1_repro_repeat2` | `pytorch_model.bin.4` | **49.1** | 46.4 | 未复现历史 Exp1 50.0 |
| `baseline_pure_sim_repeat2` | `pytorch_model.bin.4` | **48.7** | 47.3 | 低于历史 baseline 49.4 |

- 历史最高仍是 Exp1 的 50.0，但 `21348d1` repeat2 只到 49.1，说明 50.0 至少不是稳定复现结果。
- 置信度加权路径（per-video / per-pair / entropy / MLP）已验证无稳定收益；后续不再沿该方向继续堆实验。
- 原始 hard-negative batch packing 已跑完：`logs/20260619/hn_pack_wmil0_repeat1_4gpu_b64_train_msrvtt.log`，Best T2V R@1 = **48.1**（低于 B1-only v2 49.3），说明 raw HN packing 对 T2V top-1 不成立。
- raw hard-negative 映射质量审计发现明显假负例/近重复：180000 条中 exact caption pairs 4059，高风险 17637，hard caption 最大复用 504。
- 已生成 clean hard-negative 映射：`cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json`，保留 163592/180000 条，exact caption pairs 清零，最大 hard caption/video 复用降至 39/93；审计报告见 `cache_dir/hard_negatives/msrvtt_train_hardneg_audit.md`。
- `main_task_retrieval.py` 的 `--hard_negative_path` 默认值已切到 clean 映射；原始未清洗映射仅作追溯：`cache_dir/hard_negatives/msrvtt_train_hardneg.json`。
- 当前下一步：GPU 空闲后只跑 **clean-HN packing + `w_mil=0`** 一组判定实验；由于 GPU 被占满，该实验截至 2026-06-22 尚未启动。batch packing 默认关闭，仍需显式传 `--use_hard_negative_packing`。

---

## 目录结构

| 目录/文件 | 职责 |
|-----------|------|
| `main_task_retrieval.py` | 训练/评估入口，负责 SAP、概率嵌入、属性开关等参数配置 |
| `modules/modeling_mulit.py` | 当前主模型；视觉概率检索主干已落地 |
| `query_models/module_sap.py` | 核心 SAP 模块：语义锚点探测、sigmoid 门控打分、混合方差聚合 |
| `modules/` | 核心组件：CLIP、Cross-Attention、Mamba、SpatialEnhancer、RALA、AdaNorm 等 |
| `modules/mus_util.py` | MUS（映射不确定性分数）批量计算，移植自 UMIVR |
| `prob_models/` | 概率建模与损失：PIENet、UncertaintyModule（attention/GRU/Mamba 三种头）、UncertaintyAdaNorm；`probemb.py`（PCME 风格损失，当前未被主模型使用） |
| `query_models/` | 查询相关模块；`module_sap.py` 为主路径 |
| `tests/` | 单元测试：`test_modeling_mulit_losses.py`（evidential loss、WTI、TAS、VIB 测试） |
| `dataloaders/tqfs_util.py` | TQFS（基于时序质量的帧采样器），移植自 UMIVR，`slice_framepos=3` 时启用 |
| `dataloaders/` | 数据加载器（MSRVTT / MSVD），支持可选属性输入 |
| `logs/` | 日志目录 → 见 [`logs/README.md`](logs/README.md) |
| `ckpts/` | 检查点存储 |

---

## 设计决策

### 语义锚点概率嵌入

| 模块 | 代码落点 | 作用 |
|------|----------|------|
| **语义锚点探测（SAP）** | `query_models/module_sap.py` | 以可学习锚点通过 TransformerDecoder 探测帧级特征，输出锚点表示 `anchors`、视频级 `mu_raw`、`logsigma` 和逐锚点 `gate_scores` |
| **不确定性调制聚合** | `query_models/module_sap.py` | 逐锚点 Dirichlet 模态概率（softplus → alpha_dir → modal_probs）+ L2 归一化；方差按混合方差公式 `log(Σ α_i · σ_i²)` 聚合到视频级 |
| **概率表示增强** | `modules/modeling_mulit.py` / `prob_models/` | 视频侧直接使用 SAP 输出的 mu/logsigma（PIENet 已移除）；文本侧经 `PIENet` 增强，可选 `UncertaintyAdaNorm` 让方差反馈到均值表征 |
| **检索与正则目标** | `modules/modeling_mulit.py` / `prob_models/` | 联合优化锚点级 WTI 检索损失（CrossEn）、MIL 概率对比损失、Evidential NLL、负证据正则、不确定性矩阵正则、锚点正交性约束 |

核心动机是将视频不确定性从单一样本级方差细化到语义锚点级方差，让模型能够区分哪些局部语义可靠、哪些局部语义模糊，并直接将该不确定性用于锚点聚合与表示增强。

### 历史债务

- `modeling.py` 是早期版本，当前主模型是 `modeling_mulit.py`
- `modeling_mulit.py` 文件名有拼写遗留，但仍是当前集成入口
- 核心逻辑分散在 `modules/modeling_mulit.py`、`query_models/module_sap.py`、`prob_models/` 三处
- `prob_models/probemb.py` 中的 PCME 风格损失（`MCSoftContrastiveLoss` 等）当前未被主模型使用；主模型使用 `modules/until_module.py` 中的 `MILNCELoss_BoF`，evidential/neg_reg/uncertainty_reg/orth loss 在 `modeling_mulit.py` 内联计算
- `modeling_mulit.py` 中 `self.uncertain_net_video`（视频侧独立不确定性头）已注释掉；视频侧不确定性完全由 SAP 的锚点级方差聚合提供
- `tmp/` 目录为临时实验代码，可忽略

---

## 操作指南

| 任务 | 入口 | 说明 |
|------|------|------|
| **训练** | `train_msrvtt.sh` / `train_msvd.sh` | 训练脚本；查看脚本内 SAP / 概率建模参数 |
| **评估** | `eval.sh` | 通过环境变量配置：`INIT_MODEL`, `EVAL_BRANCH_MODE`, `USE_ATTRIBUTES` |
| **测试** | `pytest` | 全部测试；`pytest tests/test_modeling_mulit_losses.py -k test_name` 运行单个 |
| **Lint** | `ruff check .` | 检查代码风格；`ruff check --fix .` 自动修复 |
| **查看日志** | `logs/` | 详见 [`logs/README.md`](logs/README.md) |

**常见问题**：
- OOM → 减小 `--batch_size` / `--max_frames`，或启用 `--fp16`
- 若要复现实验主线，优先确认 `query_models/module_sap.py` 与 `modules/modeling_mulit.py` 的主路径
- 添加新数据集 → 参考 `dataloaders/dataloader_msrvtt_retrieval.py` 实现

---

## 相关文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 日志说明 | [`logs/README.md`](logs/README.md) | 各类日志格式与分析方法 |
| 视频检索技能 | `.cursor/skills/video-text-retrieval/SKILL.md` | 注意力机制、不确定性学习指南 |
| MSRVTT 训练汇报 | [`docs/report_msrvtt_training.md`](docs/report_msrvtt_training.md) | 面向导师的技术说明（SAP + 概率检索主线） |
| 实验计划与消融 | [`plan.md`](plan.md) | 实验追踪表（13+ 组）、已验证结论、待实现方案 |
| 模型结构审计 | [`model_review.md`](model_review.md) | 2026-05-24 审计报告：5 个已修复问题 + 设计确认点 |
| SAP 分析 | [`query_models/analysis.md`](query_models/analysis.md) | SAP 模块详细分析笔记 |
| 当前状态 | [`STATUS.md`](STATUS.md) | 最新实验结果、当前结论、下一步路线 |
| UACL/Hard Negative 计划 | [`UACL_HARD_NEG_PLAN.md`](UACL_HARD_NEG_PLAN.md) | UACL 与 query hard negative 的迁移计划 |

---

## Learned User Preferences

- (06-14) 与用户交流及撰写文档使用简体中文。
- (06-14) 面向导师/课题组的技术说明采用学术汇报风格：结论先行、模块级说明，少工程细节。
- (06-14) 从数据到训练再到模型的梳理，以 `train_msrvtt.sh` 为唯一叙事主线。
- (06-14) 汇报文档少贴代码、不罗列目录树；代码仅用于示意关键公式或参数名。
- (06-14) 文档与代码分析须严格依据仓库实际实现与配置，不编造未出现的机制；脚本中注释或未启用的路径须明确标注。
- (06-14) 后台 MSRVTT 训练通常用 `run_train_msrvtt_bg.sh` 启动；GPU 通过环境变量 `CUDA_VISIBLE_DEVICES` 覆盖 `train_msrvtt.sh` 默认值。
- (06-19) 训练相关请求只给出命令即可，运行由用户手动启动；不要代为启动长期训练进程。训练命令默认写成单行，不使用反斜杠分行，且默认不加 `NO_TAIL=1`，方便用户启动后直接观察日志。

## Learned Workspace Facts

- (06-14) 本项目在 Fang et al. 2023 UATVR（ICCV）官方实现基础上深度分叉演进；上游参考为 `bofang98/UATVR`，原 `train.sh` 已拆为 `train_msrvtt.sh` / `train_msvd.sh`。
- (06-14) MSRVTT 导师汇报主文档为 `docs/report_msrvtt_training.md`（SAP + 概率检索主线）。
- (06-14) `train_msrvtt.sh` 默认 `batch_size=128`、`gradient_accumulation_steps=2`（2 卡 DDP 每卡 micro=64）、`CUDA_VISIBLE_DEVICES=1,2`；各 loss 权重默认值见脚本内 argparse。
- (06-14) Shell 中 `--batch_size` 表示目标有效 batch，不是 dataloader micro-batch；`main_task_retrieval.py` 解析时会先除以 `gradient_accumulation_steps`（L372）。全局 micro = shell batch_size/accum，每卡 micro = shell batch_size/(accum×GPU数)。
- (06-14) 根目录 `.gitignore` 已忽略 `cache_dir/`（本地模型缓存，避免 Git/LFS 误纳入版本库）。
- (06-14) In-batch 对比负样本由每次 forward 的 allgather 后全局 B 决定，accum 不合并对比矩阵；显存主要由 micro-batch 决定，OOM 调优优先增大 micro-batch 而非堆 accum。
- (06-14) MSRVTT 超参搜索：`w_evidential=0.01, w_neg_reg=0.05` 为最优配置（R@1=47.7）；`w_uncertainty_reg` 小幅正向。evidential_loss 与检索路径脱钩，已在 `uncertainty_mode=nig_mil` 下通过 NIG-MIL 方案修复（R@1=49.0）。
- (06-14) `ref/` 目录存放参考论文实现：UMIVR（ICCV 2025）、DUQ（IJCAI 2025）、GARE（NeurIPS 2025）、UCoFiA（ICCV 2023）、Video-ColBERT（CVPR 2025）、TF-CoVR、BSCE-GRA、2026-AAAI-SCAN。
- (06-14) 根目录 `plan.md` 记录待验证方向（多粒度特征、文本锚点、课程学习等）；`model_review.md` 为 2026-05-24 模型结构审计报告。
- (06-14) 当前 `modules/modeling_mulit.py` 的视频概率分支直接使用 SAP 输出的 `mu_raw` / `logsigma` 采样；视频侧 `PIENet` 已移除，旧的 `pie_net_video` 路径曾导致视频 batch 维度 mismatch。
- (06-14) SAP 中 `EvidentialUncertaintyHead` 当前为 Dirichlet 模态概率头（非学习不确定性版本），`beta_nig`/`alpha_nig` 等 NIG 参数已随方向切换移除。`epistemic_cont` 由 anchor 多样性 × 模态熵直接计算，无需 clamp。
- (06-14) `hyperparam_search.py` 已修复日志解析、失败 trial 隔离和成功结果去重；双卡搜索使用 `--batch_size=256 --gradient_accumulation_steps=2`，对应每卡 micro-batch 64、有效 batch 256。
- (06-19) 2026-06-17 三组复现实验最终结果：B1only_v2_repeat1=49.3，Exp1 repro repeat2=49.1，baseline pure sim repeat2=48.7；均未到 50。
- (06-22) hard-negative 后续训练命令应使用 clean 映射：`--use_hard_negative_packing --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json --w_mil 0 --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1`；原始未清洗路径仅作追溯：`cache_dir/hard_negatives/msrvtt_train_hardneg.json`。
- (06-22) raw HN packing 4GPU/b64/w_mil=0 已完成但失败：Best T2V R@1=48.1；clean HN map 已生成但因 GPU 被占满尚未启动训练。
