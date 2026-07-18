# UATVR Agent 入口

UATVR 是基于 PyTorch、OpenAI CLIP ViT-B/16、WTI 与 `trusted-v1` 协议的文本—视频检索研究项目。交流和项目文档统一使用简体中文。

## 阅读与维护边界

本文件是当前项目约束的唯一入口。`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 与整个 `experiments/` 已删除，不再维护独立 SSOT、活动规格或实验归档，也不要自动重建这些文件。

除非用户明确要求，默认不读取 `docs/deep-research-report.md`、`docs/analysis/`、`docs/superpowers/` 或 `research_refs/`。历史日志、旧计划和过期 checkpoint 只从 Git 历史定向追溯，不作为当前设计依据。

## 当前决策（2026-07-18）

- 当前可信参照是 OpenAI CLIP ViT-B/16、deterministic WTI-only、单 seed 42。
- MSR-VTT 唯一有效协议为 `trusted-v1`：8500 train / 500 internal val。JSFusion 1K 只允许在方法、超参数、seed 集、选择规则和报告模板冻结后作一次显式确认评估。
- 科研核心是设计新的文本—视频匹配或训练机制，并相对同协议基线稳定改善检索；校准、错误预测、abstention 或人工审核不能替代检索收益。
- 当前没有独立活动实验规格；新机制的实现、训练和评估范围以用户当次明确指令为准。
- 旧 pair evidence refiner、Hard negative 与 UACL 路线均已停止。活动代码不保留这些分支、参数、诊断统计或 checkpoint 兼容逻辑。
- 不为已删除模块迁移旧 checkpoint，也不在主模型中扫描、拒绝或解释旧模块参数；需要复现旧模型时使用对应 Git 版本。
- 换 backbone 只可作为后续匹配控制变量，不作为创新点。

## 关键训练与评估语义

- 主入口：`run_train_msrvtt_bg.sh` → `train_msrvtt.sh` → `main_task_retrieval.py` → `modules/modeling_retrieval.py`。
- 正例只按精确 `video_id` 定义；主损失为双向多正例 InfoNCE，可信基线的最终 logits 为 WTI。
- Shell `--batch_size` 表示目标有效 batch；global forward batch = `batch_size / gradient_accumulation_steps`。梯度累积不会合并不同 forward 的 in-batch negatives。
- 基线口径固定为 global forward batch 256；4 卡时每卡 micro-batch 64、accumulation 1。比较实验必须同时匹配 forward contrastive batch、optimizer effective batch 和 optimizer steps。
- `--fp16` 不是完整 AMP 数据流，也不等于 `clip_layer_norm_precision`。OpenAI CLIP LayerNorm 默认 native FP16，并保留 FP32 master affine；`CLIP_LAYER_NORM_PRECISION=fp32` 仅作显式回退。
- `eval.sh` 必须显式设置 `EVAL_SPLIT=val|test`；训练期只构造 internal-val dataloader，禁止构造或读取 test dataloader。
- 真实数据缓存、统计拟合、head fitting 和 internal-val/test 访问都属于实验执行，必须由用户明确授权。
- agent 不启动或代跑长期训练。

## 代码与验证入口

| 任务 | 入口 |
|---|---|
| 训练 | `train_msrvtt.sh` / `run_train_msrvtt_bg.sh` |
| 评估 | `eval.sh`，必须显式设置 `EVAL_SPLIT=val|test` |
| 只读诊断 | `scripts/diagnose_msrvtt_p1_baseline.py` / `scripts/audit_msrvtt_p1_errors.py` |
| 主模型 | `modules/modeling_retrieval.py` |
| 数据协议 | `dataloaders/splits/msrvtt_trusted_v1_seed42.json` |
| 测试 | `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests` |
| 静态检查 | `/home/xujie/.conda/envs/tvr/bin/ruff check ...` |

不要运行根目录无范围的 `pytest -q`；`research_refs/` 含第三方可选依赖测试。工作树可能包含用户改动，始终保留无关变化。

`tests/` 只保留数据读取、数据协议、缓存与 manifest 等跨模块基础测试。设计或实现深度学习模块后，不得为单个模块单独创建 `tests/test_<module>.py`；必要的公式、前向、梯度与合成张量核对使用最小合成验证或已有统一验收入口，除非用户明确要求新增专项测试文件。

## 稳定实现事实

- 当前主模型文件为 `modules/modeling_retrieval.py`，只保留 backbone、文本/视频编码、WTI、精确 `video_id` 多正例损失与必要的分布式 gather。
- 主模型不承担旧 checkpoint 兼容、参数迁移、停用实验分支、输入类型穷举校验或训练诊断统计；数据 shape、mask 和 batch 契约由数据层与调用入口保证。
- prepared-WTI state 是核心计算复用接口，不限制为 eval/no-grad，也不重复检查内部生成的 tensor 类型、shape、dtype 和 device。
- 不在模型热路径使用 `.item()`、`.tolist()` 或全量 finite/binary 检查，避免 GPU 同步。
- 纯 WTI 的 MUS TSV 只用于只读错配诊断，不改变标签、loss 或最终排序。
- 科研参考项目与本地权重位于忽略目录 `research_refs/`，不得纳入 Git。
