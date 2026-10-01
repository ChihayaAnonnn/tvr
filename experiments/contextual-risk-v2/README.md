# Contextual Risk v2 研究

> **当前结论：Stage 1 gate FAIL，HALT。** `next_authorized_stage = null`，
> 见 [阶段报告](summaries/stage1-baselines.md)。以下是归档流程说明，不是续跑授权。

本研究的控制面位于此目录，运行输出只能位于
`artifacts/contextual-risk-v2/`。开始前阅读 `study.yaml` 与 `registry.jsonl`；不得将新实验直接写入 `logs/` 或 `ckpts/`。

## 阶段与门禁

- Stage 0：archive qualification。schema、lifecycle、launcher、文档、CPU fixture 与四卡 smoke 都通过且登记前，禁止 GPU 科学实验。
- Stage 1--6：仅使用 validation；任何未通过的预声明 gate 均停止，不调参重跑。
- Stage 7：仅在架构、loss、K、seed 与 calibration 全部冻结后，以明确的 test 授权访问 test。

`registry.jsonl` 先批量提交 `reserved`，运行时只更新 run-local lifecycle，全部 terminal 后才
批量提交终态记录与摘要。大于 1 MiB 的数组必须外置，失败证据不可删除。
`failed/cancelled` 不可原地重试或复用 run ID；人工授权的新尝试必须在后续 phase 使用新的唯一
run ID，并通过 manifest 的 `parent_run_ids` 和 `selection_lineage` 保留 predecessor 与审核理由。
程序化校验这些 schema 必须使用 `experiment_artifacts.artifact_schema_validator`；普通
`Draft202012Validator` 不执行 `x-canonical-content-address` 跨字段关键字，不能证明数组路径中的
完整文件名与 `sha256` 相等。该项目 validator 还会递归拒绝 `NaN`、`Infinity` 和
`-Infinity`；归档 JSON 的读写必须使用 strict helper，禁止 Python JSON 的非标准数值扩展。

早期 qualification successor 的 selected metric 保留了顶层 `world_size`，其不可变 SHA-256
为 `61f65c79af6c88280be5ca9862f2db97d9d3027b7696163fb9b13c14eb4e4831`。只有该 digest 通过
`validate_selected_metrics_document` 对原始 UTF-8 bytes 重算并匹配 registry reference 后，才分派到
`schemas/metrics-legacy-qualification-v0.schema.json`；其他文件以及所有新写入均使用
`metrics.schema.json`，不得把 `world_size` 当作当前 schema 的静默扩展。

## 生命周期命令

```bash
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py reserve-phase ...
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py verify-run ...
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py start-run ...
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py transition-run ...
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py finalize-phase ...
```

命令须在仓库根目录执行；`PYTHONPATH=.` 让管理脚本导入归档契约。参见
`docs/agent/experiment-artifacts.md` 的完整身份参数和恢复规则。
