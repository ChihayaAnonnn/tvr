# Contextual Risk v2 研究

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
