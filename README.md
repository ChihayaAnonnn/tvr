# Semantic retrieval reconstruction

## 仓库导航（2026-09-30 整理）

- [研究状态与证据](research/README.md)：先读 00–02，再决定是否提出新实验。
- [实验总览](experiments/README.md)：5 个 study 的结论、证据位置和续跑限制。
- [论文索引](related_work/README.md)：已下载论文、版本、阅读笔记和旁支参考资料。
- [本次整理记录](docs/maintenance/2026-09-30-cleanup.md)：实际清理范围、逐 run 清单和验证结果。

**当前没有获准继续的实验阶段。** Contextual Risk v2 在 Stage 1 gate 失败后停止；
OQ1 v1 在 Stage 1 以 inconclusive 终止；SAP 路线已关闭。详见实验总览，
不要把历史计划、交接文档或冻结协议里的 `registered` 当作续跑授权。
MSRVTT/MSVD test 继续 sealed。

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

## 当前开发状态

A 阶段交付独立的 `src/semantic_retrieval/` 基础组件：输入合同、本地 CLIP/tokenizer、OpenCV 视频预处理、split 校验、masked WTI、可微 gather、检索/校准指标与归档生命周期。旧 HEAD/June UATVR、概率分支、训练入口和旧 launcher 已删除；历史复现需单独 checkout `ff8137c28039b2d8b61f7874538310825f3e3b21`，不得在当前代码上启动旧登记命令。

B 阶段实现新编码器、共享匹配/候选存在头、多正例监督、canonical 索引缓存、有限扩检、validation 校准，以及新训练/评测入口。接口与显式数据格式见 [系统合同](docs/agent/semantic-retrieval-system.md)，设计见 [研究模型计划](docs/superpowers/plans/2026-10-01-retrieval-rebuild-b-system.md)。验证仅合成 CPU；尚无真实数据效果或 GPU 资格结论。语义分组/OT 默认关闭，CPU qualification 不能替代四卡 Stage 0 gate。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  /home/xujie/.conda/envs/tvr/bin/python -m pytest -q -p no:cacheprovider tests/semantic_retrieval

/home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py --help
/home/xujie/.conda/envs/tvr/bin/python scripts/build_msrvtt_trusted_split.py --help
```

完整迁移、删除范围与验证证据见 [A 阶段审计](docs/maintenance/2026-10-01-code-rebuild-audit.md)。运行新实验前仍须先阅读 [研究索引](research/README.md) 及控制面协议。
