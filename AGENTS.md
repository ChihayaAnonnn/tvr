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

## 当前实验状态（2026-07-04）

完整实验档案见 [`docs/project/STATUS.md`](docs/project/STATUS.md)。新会话优先使用以下当前结论：

- 当前决策：**Hard negative 主线终止**。相关代码保留为消融/诊断工具，训练主线不再继续跑 `--use_hard_negative_packing`、`--use_explicit_hard_negative_loss`、`w_hard_negative` sweep 或同机制 repeat。
- 2026-06-30 已在 `feat/uacl-explicit-hn-intra` 分支接入论文参考路线的第一版代码，默认均关闭：
  - 显式 hard-negative InfoNCE loss：`--use_explicit_hard_negative_loss --w_hard_negative ...`；额外编码 clean-map hard-negative video，并把 hard-negative logits 作为附加列并入 query-to-video CE 分母。
  - UACL-style 模态内对齐：`--use_uacl_intra_alignment --w_uacl_intra ... --w_uacl_kl ... --uacl_temperature ...`
  - 训练 dataloader 会在显式 HN 开启时额外返回 `hard_video/hard_video_mask/hard_valid`；batch packing 仍保留为 legacy/diagnostic 开关。
  - 2026-06-27/29 的三组旧实验使用的是早期 softplus/弱权重实现；2026-06-30 已改为更忠实论文/原版代码的 InfoNCE 分母扩展，并完成后续训练与诊断。
- 当前工作树为 **B1-only v2**：主检索分数退回 `weighted_logits = wti_logits`，per-pair `confidence_mlp` 已移除。
- 2026-06-19 三组复现实验均已结束，均未复现历史 50.0：

| 实验 | 最佳 checkpoint | Best T2V R@1 | 对应 V2T R@1 | 结论 |
|------|----------------|:---:|:---:|------|
| `B1only_v2_repeat1` | `pytorch_model.bin.4` | **49.3** | 47.6 | 当前三组最优，但未到 50 |
| `exp1_repro_repeat2` | `pytorch_model.bin.4` | **49.1** | 46.4 | 未复现历史 Exp1 50.0 |
| `baseline_pure_sim_repeat2` | `pytorch_model.bin.4` | **48.7** | 47.3 | 低于历史 baseline 49.4 |

- 历史最高仍是 Exp1 的 50.0，但 `21348d1` repeat2 只到 49.1，说明 50.0 至少不是稳定复现结果。
- 置信度加权路径（per-video / per-pair / entropy / MLP）已验证无稳定收益；后续不再沿该方向继续堆实验。
- Hard negative 全链路结论：
  - raw HN packing：Best T2V R@1 = **48.1**，低于 B1-only v2 的 49.3。
  - clean HN packing：曾有单次 49.7 高点，但 repeat/诊断最高未超过 B1-only，后续 2GPU 诊断最高 48.6，稳定性不足。
  - 旧 clean-map explicit HN InfoNCE：Best T2V R@1 = **49.4**，fixed/regressed = 32/31，净收益几乎为 0。
  - model-mined explicit HN：Best T2V R@1 = **48.6**，fixed/regressed = 31/38，净变化 -7。
- Hard negative 不适合当前 MSRVTT 主线的原因：训练信号确实能扩大 `ret_gap`、改善部分 GT rank，但验证 Top-1 fixed/regressed 不占优；MSRVTT 存在大量语义近邻/多正例式歧义，训练 hard negatives 与验证 Top-1 错误不够对齐，容易把同主题近邻边界扰动成退化样本。
- 当前下一步：**只验证 UACL-style 模态内对齐的稳定性**；HN 只保留为论文消融负结果和诊断脚本，不再作为优化主线。

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
| `logs/` | 原始日志目录；Markdown 分析报告见 [`docs/logs/README.md`](docs/logs/README.md) |
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
| **查看日志** | `logs/` / `docs/logs/` | 原始日志在 `logs/`，Markdown 分析报告见 [`docs/logs/README.md`](docs/logs/README.md) |

**常见问题**：
- OOM → 减小 `--batch_size` / `--max_frames`，或启用 `--fp16`
- 若要复现实验主线，优先确认 `query_models/module_sap.py` 与 `modules/modeling_mulit.py` 的主路径
- 添加新数据集 → 参考 `dataloaders/dataloader_msrvtt_retrieval.py` 实现

---

## 相关文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 文档总入口 | [`docs/README.md`](docs/README.md) | 项目文档集中索引 |
| 日志说明 | [`docs/logs/README.md`](docs/logs/README.md) | 各类日志分析报告与存放规则 |
| 视频检索技能 | `.cursor/skills/video-text-retrieval/SKILL.md` | 注意力机制、不确定性学习指南 |
| 实验计划与消融 | [`docs/project/plan.md`](docs/project/plan.md) | 实验追踪表（13+ 组）、已验证结论、待实现方案 |
| Query 分支分析 | [`docs/analysis/query_branch_analysis.md`](docs/analysis/query_branch_analysis.md) | Query 分支详细分析笔记 |
| 当前状态 | [`docs/project/STATUS.md`](docs/project/STATUS.md) | 最新实验结果、当前结论、下一步路线 |
| UACL/Hard Negative 计划 | [`docs/project/UACL_HARD_NEG_PLAN.md`](docs/project/UACL_HARD_NEG_PLAN.md) | UACL 与 query hard negative 的迁移计划 |
| Qwen 属性生成说明 | [`docs/deploy_qwen/README.md`](docs/deploy_qwen/README.md) | Qwen3-VL 属性生成服务使用说明 |
| Backbone 升级策略 | [`docs/reference/uatvr_backbone_upgrade_strategy.md`](docs/reference/uatvr_backbone_upgrade_strategy.md) | CLIP-like / video foundation backbone 替换建议 |
| 开题报告大纲 | [`docs/paper/开题报告_大纲.md`](docs/paper/开题报告_大纲.md) | 论文开题报告材料 |

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
- (06-14) MSRVTT 导师汇报主文档当前工作树未找到；如恢复，应放入 `docs/`。
- (06-14) `train_msrvtt.sh` 默认 `batch_size=128`、`gradient_accumulation_steps=2`（2 卡 DDP 每卡 micro=64）、`CUDA_VISIBLE_DEVICES=1,2`；各 loss 权重默认值见脚本内 argparse。
- (06-14) Shell 中 `--batch_size` 表示目标有效 batch，不是 dataloader micro-batch；`main_task_retrieval.py` 解析时会先除以 `gradient_accumulation_steps`（L372）。全局 micro = shell batch_size/accum，每卡 micro = shell batch_size/(accum×GPU数)。
- (06-14) 根目录 `.gitignore` 已忽略 `cache_dir/`（本地模型缓存，避免 Git/LFS 误纳入版本库）。
- (06-14) In-batch 对比负样本由每次 forward 的 allgather 后全局 B 决定，accum 不合并对比矩阵；显存主要由 micro-batch 决定，OOM 调优优先增大 micro-batch 而非堆 accum。
- (06-14) MSRVTT 超参搜索：`w_evidential=0.01, w_neg_reg=0.05` 为最优配置（R@1=47.7）；`w_uncertainty_reg` 小幅正向。evidential_loss 与检索路径脱钩，已在 `uncertainty_mode=nig_mil` 下通过 NIG-MIL 方案修复（R@1=49.0）。
- (06-14) `ref/` 目录存放参考论文实现：UMIVR（ICCV 2025）、DUQ（IJCAI 2025）、GARE（NeurIPS 2025）、UCoFiA（ICCV 2023）、Video-ColBERT（CVPR 2025）、TF-CoVR、BSCE-GRA、2026-AAAI-SCAN。
- (06-14) `docs/project/plan.md` 记录待验证方向（多粒度特征、文本锚点、课程学习等）；模型结构审计文档当前工作树未找到，如恢复，应放入 `docs/`。
- (06-14) 当前 `modules/modeling_mulit.py` 的视频概率分支直接使用 SAP 输出的 `mu_raw` / `logsigma` 采样；视频侧 `PIENet` 已移除，旧的 `pie_net_video` 路径曾导致视频 batch 维度 mismatch。
- (06-14) SAP 中 `EvidentialUncertaintyHead` 当前为 Dirichlet 模态概率头（非学习不确定性版本），`beta_nig`/`alpha_nig` 等 NIG 参数已随方向切换移除。`epistemic_cont` 由 anchor 多样性 × 模态熵直接计算，无需 clamp。
- (06-14) `hyperparam_search.py` 已修复日志解析、失败 trial 隔离和成功结果去重；双卡搜索使用 `--batch_size=256 --gradient_accumulation_steps=2`，对应每卡 micro-batch 64、有效 batch 256。
- (06-19) 2026-06-17 三组复现实验最终结果：B1only_v2_repeat1=49.3，Exp1 repro repeat2=49.1，baseline pure sim repeat2=48.7；均未到 50。
- (07-04) hard-negative 主线已终止：raw/clean packing、old clean-map explicit HN、model-mined explicit HN 与 fixed/regressed 诊断均未证明稳定收益；相关命令和开关只作消融/诊断追溯，不再作为训练建议。
- (07-04) hard-negative 不适合当前 MSRVTT 主线的主要原因：语义近邻/多正例式歧义导致 hard negatives 不是干净负例；训练 `ret_gap`/中段 rank 可改善，但 fixed/regressed 不占优，无法稳定提升 T2V R@1。
