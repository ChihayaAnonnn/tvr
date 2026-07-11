# UATVR Agent 入口

UATVR 是基于 PyTorch、OpenAI CLIP ViT-B/16、WTI 与 trusted-v1 协议的文本—视频检索研究项目。交流和项目文档统一使用简体中文。

## 单一事实源

- 科研问题、证据、停止条件和实验顺序只以 [`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`](docs/project/RESEARCH_ISSUES_AND_ROADMAP.md) 为准。
- [`docs/analysis/multimodal_retrieval_research_synthesis.md`](docs/analysis/multimodal_retrieval_research_synthesis.md) 只提供外部论文证据分析，不是项目决策 SSOT。
- [`docs/analysis/query_branch_analysis.md`](docs/analysis/query_branch_analysis.md) 是历史结构快照，不代表当前实现或路线。
- 历史日志、诊断 TSV、旧计划和过期 checkpoint 只从 Git 历史追溯；`logs/` 与 `ckpts/` 仅保存当前运行产物。

## 当前决策（2026-07-11）

- MSRVTT 唯一有效协议为 `trusted-v1`：seed 42，8500 train / 500 internal val；JSFusion 1K 仅在方法与超参数冻结后作一次显式盲测。
- 主损失按精确 `video_id` 使用双向多正例 InfoNCE，训练与评估的最终 logits 固定为 WTI。
- 当前 P0 是 OpenAI CLIP hygiene WTI-only 新基线；完成前不判断 EVA adapter、pair-level uncertainty 或 candidate-conditioned alignment 的增益。
- P0 固定 global forward batch 256、4 卡时每卡 micro-batch 64、accum=1。换 backbone 只可作为后续匹配控制变量，不作为创新点。
- OpenAI CLIP 自定义 LayerNorm 默认使用 native FP16，并保留 FP32 master affine 参数；通过 `CLIP_LAYER_NORM_PRECISION=fp32` 显式回退。该设置不是全模型 AMP，也不影响 EVA LayerNorm。
- Hard negative 主线已停止，不再 sweep 或 repeat；独立、默认关闭的诊断入口保留。UACL 活动接口已删除，不恢复。

## 关键训练与评估语义

- 主入口：`run_train_msrvtt_bg.sh` → `train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_mulit.py`。
- Shell `--batch_size` 表示目标有效 batch；解析后 global forward batch = `batch_size / gradient_accumulation_steps`，每卡 micro = global forward batch / GPU 数。
- 梯度累积不会合并不同 forward 的 in-batch negatives；比较实验必须同时核对 forward contrastive batch 和 optimizer effective batch。
- 4 卡、`batch_size=256`、accum=1 时每卡 micro-batch 64，这是当前可信基线口径。
- `--fp16` 不是完整 AMP 数据流，不得与 `clip_layer_norm_precision` 混淆。
- `eval.sh` 必须显式设置 `EVAL_SPLIT=val|test`；训练期只构造 internal-val dataloader，禁止构造或读取 test dataloader。
- 不启动或代跑长期训练进程。训练请求只向用户提供单行命令，由用户手动运行；默认不加 `NO_TAIL=1`。

## 代码与验证入口

| 任务 | 入口 |
|---|---|
| 训练 | `train_msrvtt.sh` / `run_train_msrvtt_bg.sh` |
| 评估 | `eval.sh`，必须显式设置 `EVAL_SPLIT=val|test` |
| 主模型 | `modules/modeling_mulit.py` |
| 数据协议 | `dataloaders/splits/msrvtt_trusted_v1_seed42.json` |
| 测试 | `/home/xujie/miniconda3/envs/ret/bin/pytest -q tests`；静态检查使用 `/home/xujie/miniconda3/envs/ret/bin/ruff check ...` |

不要运行根目录无范围的 `pytest -q`；`research_refs/` 含第三方可选依赖测试。工作树可能包含用户改动，始终保留无关变化。

## 文档入口

- [`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`](docs/project/RESEARCH_ISSUES_AND_ROADMAP.md)：科研决策唯一 SSOT。
- [`docs/analysis/multimodal_retrieval_research_synthesis.md`](docs/analysis/multimodal_retrieval_research_synthesis.md)：四篇外部论文的证据综合，非 SSOT。
- [`docs/analysis/query_branch_analysis.md`](docs/analysis/query_branch_analysis.md)：Query 分支历史快照。
- [`docs/deploy_qwen/README.md`](docs/deploy_qwen/README.md)：属性生成说明。

## 稳定实现事实

- 当前主模型文件名仍为历史拼写 `modeling_mulit.py`。
- 活动模型图只保留 deterministic WTI 与可选独立 hard-negative；hygiene profile 明确拒绝 hard-negative 诊断路径。
- 旧模型 checkpoint 中的已删除参数会在模型构造前明确拒绝；旧 optimizer 参数组不迁移，不兼容时由原生错误终止。
- 纯 WTI 的 MUS TSV 可用于只读风险诊断，但不改变标签、loss 或最终排序。
- 科研参考项目与本地权重位于忽略目录 `research_refs/`，不得纳入 Git。
