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
归档 JSON 必须由 strict loader/writer 处理，递归拒绝 `NaN`、`Infinity`、`-Infinity`；schema
校验必须使用 `artifact_schema_validator`。早期 qualification successor 的 selected metric
由 registry 中固定的 digest
`61f65c79af6c88280be5ca9862f2db97d9d3027b7696163fb9b13c14eb4e4831` 唯一分派至
`metrics-legacy-qualification-v0.schema.json`；分派前必须对原始 UTF-8 bytes 重算并匹配该
registry digest，只读兼容其 `world_size` 字段。新文件不得使用该
legacy contract，也不得在当前 metrics schema 中静默增加字段。

发生故障时保留 `logs/`、`status.json`、`command.sh`、provenance/audit 与输出哈希。先执行
`verify-run` 并检查源码、预留、manifest 和 audit 是否一致。恢复只适用于未终态且只有一个
待提交 transition 的崩溃窗口；`failed/cancelled` 是不可变终态，不可原地重试，也不可复用相同
run ID。

2026-10-01 用户删除第8条，不再要求逐次人工授权。后续尝试仍通过现有生命周期使用
新的 phase/run ID；`parent_run_ids` 链接 predecessor，`selection_lineage` 记录策略、自动或
人工审核报告和理由。自动恢复尚未实现；文档变更不代表当前代码支持自动重试。对旧登记行的事实更正只可追加 `correction`，并以 `supersedes_sha256` 指向同一 run
的既有记录；当前 schema 不支持用 correction 将旧 run 跨 run 链接到新尝试。配置、命令、输入或
source SHA 改变时也必须新预留 run。路径/哈希/审计无法验证时停止并请求人工处理，绝不静默重试
或覆盖既有目录。

2026-10-01 重建四卡首次验收的已提交 `validating/exit_code=1` 事故，经本会话人工授权，
采用同 run 的 `correction(state=failed)` 封存控制面；原无效 status/audit 保持原字节，
不是正常 `finalize-phase` 或自动恢复。人工封存报告见
[manual seal](../../experiments/contextual-risk-v2/incidents/2026-10-01-rebuilt-q0-manual-seal.json)。
新合成四卡验收仅允许将该失败运行作为第二个 parent；入口校验已提交的审批、更正、
forensic inventory 和全部原文件哈希，第一个 parent 仍必须是已完成的合成输入准备。
控制器产生的 `.lock`、`status.json.lock`、`provenance/audit.jsonl.lock` 不属于内容证据。
此例外不修复旧状态、不读取其 checkpoint，也不适用于科学训练或其他归档事故。

新 phase 必须用 `scheduled_timestamp_utc` 显式表示 run ID 中预定的 UTC 身份时间；
`timestamp_utc` 只为旧 phase 和已存证据保留的兼容别名。若没有预定启动时间，应在
预留时间生成当时 UTC 值，不得用计划文档的创建时间冒充实际执行时间。
`selection_lineage` 必须引用受 Git 跟踪的报告或 commit；不得引用 worktree Git
管理状态中的 `sdd/` 路径。

`--artifact-root` 接收父级数据面根目录，管理器会再追加 phase 的 study 名；不要把
`contextual-risk-v2` 重复写进该参数。

## 常用命令

```bash
PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py reserve-phase \
  --repo-root "$PWD" --phase <committed-phase.json> \
  --registry experiments/contextual-risk-v2/registry.jsonl \
  --artifact-root "$PWD/artifacts"

PYTHONPATH=. /home/xujie/.conda/envs/tvr/bin/python scripts/manage_experiment_run.py verify-run \
  --repo-root "$PWD" --registry experiments/contextual-risk-v2/registry.jsonl \
  --run-dir <absolute-run-dir> --run-id <run-id> --manifest-sha256 <sha256>
```

Stage 0 的 CPU E2E 与四卡 smoke 均通过且登记后，才可以申请后续 scientific phase。

## 重建后的执行面

2026-10-01 起旧 UATVR 与 launcher 已退役。历史 manifest/registry/schema 保持原样，只作证据或在记录的 source SHA 复现；管理器不再生成旧 A3 argv。新 worker 生命周期位于 `src/semantic_retrieval/runtime/artifacts.py`，CPU 保护 fixture 位于 `runtime/qualification.py`。A 阶段没有科学训练入口；不得把 CPU fixture 当成四卡资格通过，也不得因历史登记授权而重新启用旧入口。通用 reserve/verify/start/transition/finalize 命令和原封存规则继续生效。

## 独立官方标签 pilot

控制面 `experiments/semantic-answerability-pilot-v1/`，数据面
`artifacts/semantic-answerability-pilot-v1/<dataset>/<stage>/<run-id>/`。
管理器 reserve 与 worker prepare 双重检查新 study；当前仅允许 CPU
source_inventory / benchmark_compile，不授权科学 GPU 或 test。
先提交源码/协议，再预留并提交整批 reserved，最后启动；终态与失败规则完全继承 Scheme A。

真实库超过 manifest 容量时，`dataset_splits` 可以绑定以 `.input-index.json` 结尾的
输入索引。索引严格为 `{schema_version:1,kind:"file_sha256_index",files:[{path,sha256},...]}`，
每片小于 1 MiB；路径须为绝对规范路径，不允许嵌套索引或重复身份。worker 先核验索引
SHA，再逐一核验全部子文件 SHA，消费方复用同一个已认证的输入映射。索引不削弱逐文件
绑定，不能只校验索引而跳过原视频。`write_input_indices` 默认每片最多 900 个引用。
