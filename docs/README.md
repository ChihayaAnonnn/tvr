# UATVR 文档索引

本目录集中存放项目自有文档。根目录仅保留 `AGENTS.md`、`CLAUDE.md` 等工具约定入口；原始训练日志仍放在 `logs/`，第三方参考代码和模型卡保留在各自目录。

## 项目状态与计划

- [`project/RESEARCH_ISSUES_AND_ROADMAP.md`](project/RESEARCH_ISSUES_AND_ROADMAP.md)：科研决策唯一主文档，包含问题、证据、停止条件与实验顺序。
- [`project/STATUS.md`](project/STATUS.md)：当前摘要。
- [`project/plan.md`](project/plan.md)：实验计划入口（不维护独立优先级）。
- 当前科研决策、实验优先级与停止条件只以 [`project/RESEARCH_ISSUES_AND_ROADMAP.md`](project/RESEARCH_ISSUES_AND_ROADMAP.md) 为准；UACL/Hard Negative 仅作为历史负结果与边界分析归档，不构成当前下一步或优先级。
- [`project/UACL_HARD_NEG_PLAN.md`](project/UACL_HARD_NEG_PLAN.md)：UACL / Hard Negative 历史负结果与边界分析归档（不构成当前优先级）。

## 分析与报告

- [`analysis/query_branch_analysis.md`](analysis/query_branch_analysis.md)：Query 分支结构与数据流分析。
- [`logs/README.md`](logs/README.md)：日志分析报告存放规则。

## 工程说明与参考

- [`deploy_qwen/README.md`](deploy_qwen/README.md)：Qwen3-VL 视频属性生成服务说明。
- [`reference/uatvr_backbone_upgrade_strategy.md`](reference/uatvr_backbone_upgrade_strategy.md)：Backbone 升级策略参考。
- [`superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md`](superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md)：可信实验基座设计规格（已实施）。
