# UATVR Agent 入口

UATVR 是基于 PyTorch、OpenAI CLIP ViT-B/16 与语义锚点概率嵌入（SAP）的文本—视频检索研究项目。交流和项目文档统一使用简体中文。

## 单一事实源

- 科研问题、证据、停止条件和实验顺序只以 [`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`](docs/project/RESEARCH_ISSUES_AND_ROADMAP.md) 为准。
- 历史原始日志、诊断 TSV、旧实施计划和过期 checkpoint 不保留在当前工作树；需要追溯时使用 Git 历史。
- `logs/` 与 `ckpts/` 只保存当前运行产物，不作为长期科研口径来源。

## 当前决策（2026-07-11）

- Hard negative 与 UACL 主线均已终止；代码只保留为消融/诊断入口，不再 sweep 或 repeat。
- MSRVTT 唯一有效协议为 `trusted-v1`：seed 42，8500 train / 500 internal val；JSFusion 1K 仅作显式盲测；主损失按精确 `video_id` 使用双向多正例 InfoNCE。
- 当前 P0 是 OpenAI CLIP hygiene WTI-only 新基线；完成前不判断 EVA adapter、SAP 或 uncertainty 的增益。
- hygiene profile 真实绕过 SpatialEnhancer、SAP、概率分支、PIENet、不确定性头和辅助 loss，主分数固定为 `weighted_logits = wti_logits`。
- OpenAI CLIP 自定义 LayerNorm 默认使用 native FP16，保留 FP32 master affine 参数；通过 `CLIP_LAYER_NORM_PRECISION=fp32` 回退。该设置不是全模型 AMP，也不影响 EVA LayerNorm。
- FP16 LayerNorm 实现已合并至 `main`；四卡可信基线应使用全局 forward batch 256、每卡 micro-batch 64、accum=1。

## 关键训练语义

- 主入口：`run_train_msrvtt_bg.sh` → `train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_mulit.py`。
- Shell `--batch_size` 表示目标有效 batch；解析后全局 forward batch = `batch_size / gradient_accumulation_steps`，每卡 micro = global forward batch / GPU 数。
- 梯度累积不会合并不同 forward 的 in-batch negatives；比较实验必须同时核对 forward contrastive batch 和 optimizer effective batch。
- 4 卡、`batch_size=256`、accum=1 时每卡 micro=64；这是当前可信基线口径。
- `--fp16` 仍不是完整 AMP 数据流；不要把它与 `clip_layer_norm_precision` 混淆。
- 不启动长期训练进程；训练请求只给用户单行命令，由用户手动运行。默认不加 `NO_TAIL=1`。

## 代码与验证入口

| 任务 | 入口 |
|---|---|
| 训练 | `train_msrvtt.sh` / `run_train_msrvtt_bg.sh` |
| 评估 | `eval.sh`，必须显式设置 `EVAL_SPLIT=val|test` |
| 主模型 | `modules/modeling_mulit.py` |
| OpenAI CLIP | `modules/module_clip.py` |
| SAP | `query_models/module_sap.py` |
| 概率模块 | `prob_models/` |
| 数据协议 | `dataloaders/splits/msrvtt_trusted_v1_seed42.json` |
| 测试 | `/home/xujie/miniconda3/envs/ret/bin/pytest -q tests` |
| 静态检查 | `/home/xujie/miniconda3/envs/ret/bin/ruff check ...` |

不要运行根目录无范围的 `pytest -q`；`research_refs/` 含第三方可选依赖测试。工作树可能包含用户改动，始终保留无关变化。

## 文档入口

- [`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`](docs/project/RESEARCH_ISSUES_AND_ROADMAP.md)：科研决策唯一主文档。
- [`docs/analysis/query_branch_analysis.md`](docs/analysis/query_branch_analysis.md)：Query 分支分析。
- [`docs/reference/uatvr_backbone_upgrade_strategy.md`](docs/reference/uatvr_backbone_upgrade_strategy.md)：Backbone 升级策略。
- [`docs/deploy_qwen/README.md`](docs/deploy_qwen/README.md)：属性生成说明。

## 稳定实现事实

- 当前主模型文件名仍为历史拼写 `modeling_mulit.py`。
- 视频概率分支直接使用 SAP 的 `mu_raw/logsigma`；视频侧 PIENet 已移除，文本侧保留 PIENet 与 padding mask。
- `uncertainty_mode=none` 真实关闭 evidential/neg_reg；`nig_mil` 仅作 deprecated 兼容。
- `w_uncertainty_reg`、`w_query_sim`、`fusion_mode` 在当前 WTI 主排序下不是有效 causal knob。
- 科研参考项目与本地权重位于忽略目录 `research_refs/`；不要纳入 Git。
