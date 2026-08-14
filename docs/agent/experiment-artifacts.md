# 可靠检索实验归档手册

## 不可变布局

控制面是 `experiments/contextual-risk-v2/`：`study.yaml`、
`experiments/contextual-risk-v2/registry.jsonl`、phase 规格、摘要和 schema 均须受
Git 跟踪。数据面是 `artifacts/contextual-risk-v2/`，其中每个 run 目录保存 manifest、
状态、命令、logs、checkpoints、metrics、arrays、plots 和 provenance。不得将新实验直接写入 `logs/` 或 `ckpts/`；遗留目录只读，禁止移动、覆盖或删除。

`registry.jsonl` 是 append-only。每一 phase 必须先预留全部 run，提交 `reserved` 记录，
然后从干净的精确 source SHA 启动。运行中的 worker 不修改控制面；所有 worker 到达
terminal 后才批量追加终态记录和阶段摘要。每个 launch 均绑定 manifest SHA-256、命令 SHA、
配置、checkpoint/split 哈希和 artifact 绝对路径。

## 标准生命周期

1. 阅读 `study.yaml` 和 registry；确认本阶段尚未被 gate 停止，且 test 没有授权。
2. 从已提交 phase JSON 执行 `reserve-phase`，检查生成的 run ID 和 manifest。
3. 提交 registry reservation；之后执行 `verify-run`，用其返回的精确 run ID、目录与 manifest SHA 启动 worker。
4. worker 仅将实时状态推进为 `reserved -> running -> validating -> completed`（或合法失败/取消），并把日志写入 run-local 路径。
5. 关闭所有 writer 后，以 `transition-run` 写入 terminal 状态；执行 `finalize-phase`，确认输出哈希并提交 terminal registry 与摘要。

所有模型选择、风险比较、K、阈值和校准均只读 validation。只有最终配置冻结、registry
存在显式 `test_authorized=true` 授权后才可访问 test；历史 test 数字不可用于新决策。

## 数据与故障处理

数值载荷超过 1 MiB 时使用内容寻址的 NPY/NPZ 文件保存于 `arrays/`，JSON 只保留引用（相对
路径、shape、dtype、字节数和 SHA-256）。候选、winner 与 final JSON 不得重复嵌入数组。

发生故障时保留 `logs/`、`status.json`、`command.sh`、provenance/audit 与输出哈希。先执行
`verify-run` 并检查源码、预留、manifest 和 audit 是否一致；仅当基础设施失败且 manifest
完全相同，才可在追加失败证据后恢复。配置、命令、输入或 source SHA 改变时必须新预留 run，
绝不静默重试或覆盖既有目录。路径/哈希/审计无法验证时停止并请求人工处理。

## 常用命令

```bash
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py reserve-phase \
  --repo-root "$PWD" --phase <committed-phase.json> \
  --registry experiments/contextual-risk-v2/registry.jsonl \
  --artifact-root "$PWD/artifacts/contextual-risk-v2"

PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py verify-run \
  --repo-root "$PWD" --registry experiments/contextual-risk-v2/registry.jsonl \
  --run-dir <absolute-run-dir> --run-id <run-id> --manifest-sha256 <sha256>
```

Stage 0 的 CPU E2E 与四卡 smoke 均通过且登记后，才可以申请后续 scientific phase。
