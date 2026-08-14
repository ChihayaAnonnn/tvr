# UATVR

UATVR 的可靠检索实验使用两层归档：受 Git 管理的控制面在
`experiments/contextual-risk-v2/`，大文件和运行时证据在被忽略的
`artifacts/contextual-risk-v2/`。任何人不得将新实验直接写入 `logs/` 或 `ckpts/`；
这两个目录仅保留为只读历史证据。

## 实验快速开始

在计划、预留或启动任何实验前，先阅读锁定的研究协议和已提交的登记表：

```bash
sed -n '1,240p' experiments/contextual-risk-v2/study.yaml
sed -n '1,240p' experiments/contextual-risk-v2/registry.jsonl
```

控制面命令由 `scripts/manage_experiment_run.py` 提供。以下命令只展示流程；
`<phase.json>` 必须已提交，且预留、启动和终结之间必须保持干净的受跟踪源码。
`--artifact-root` 的语义是父级数据面根目录；管理器会自行追加 manifest 中的
study 名，因此这里传入 `$PWD/artifacts`，最终路径才是
`artifacts/contextual-risk-v2/<dataset>/<stage>/<run-id>/`。

```bash
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py \
  reserve-phase --repo-root "$PWD" --phase <phase.json> \
  --registry experiments/contextual-risk-v2/registry.jsonl \
  --artifact-root "$PWD/artifacts"

PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py \
  verify-run --repo-root "$PWD" \
  --registry experiments/contextual-risk-v2/registry.jsonl \
  --run-dir <absolute-run-dir> --run-id <run-id> \
  --manifest-sha256 <manifest-sha256>
```

先提交这一批 `reserved` 登记记录，再用同一组精确身份参数执行 `start-run`。
受跟踪控制面不会在运行时写入；被忽略的 run 目录可写入
`<run-dir>/status.json`、`<run-dir>/provenance/audit.jsonl`、
`<run-dir>/metrics/`、`<run-dir>/checkpoints/`、`<run-dir>/logs/` 和
`<run-dir>/arrays/`。所有 worker 结束后，再通过 `finalize-phase` 追加终态登记记录和小型阶段摘要。

指标位于 `<run-dir>/metrics/`，检查点位于 `<run-dir>/checkpoints/`，日志位于
`<run-dir>/logs/`。大于 1 MiB 的数组必须保存在 `<run-dir>/arrays/` 的 NPY/NPZ
引用中，JSON 只存标量、路径、形状和 SHA-256。

选择模型、变体、K、阈值和校准仅可读取 validation 结果。MSRVTT/MSVD test 在
最终配置冻结并获得登记授权前始终封存。完整的运行与恢复规程见
[实验归档手册](docs/agent/experiment-artifacts.md)。
