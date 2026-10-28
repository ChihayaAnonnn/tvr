# 实验总览

核对日期：2026-09-30。本页是导航，冻结的 study、append-only registry 与最终报告保持原样。
**run 的 `completed` 只表示运行完成，不表示研究假设成立或允许进入下一阶段。**

| Study | 当前研究结论 | 优先阅读的证据 | 数据状态 |
|---|---|---|---|
| contextual-risk-v2 | Stage 1 gate FAIL，停止；A3 未超过强标量基线 | [阶段报告](contextual-risk-v2/summaries/stage1-baselines.md) | [registry](contextual-risk-v2/registry.jsonl)：11 个 run；3 completed、8 cancelled |
| oq1-kl-variance-hubness-v1 | Stage 1 regime gate 未通过，HALT / inconclusive；不执行 Stage 2–6 | [最终判定](oq1-kl-variance-hubness-v1/summaries/stage1-arm0-baseline.md) | [registry](oq1-kl-variance-hubness-v1/registry.jsonl)：21 个 run；7 completed、3 failed、11 cancelled |
| sap-failure-supervision-v1 | 历史 Fix 1 归因被推翻；不能把原 PASS 解释为 SAP 获得有效监督 | [ledger E-52–E-58](../research/01-evidence-ledger.md)、[原协议](sap-failure-supervision-v1/study.yaml) | 未生成独立 registry/data plane；只保留历史日志与配置 |
| sap-failure-supervision-v2 | 同上；多 seed 不修复 NO-OP 的因果归因问题 | [ledger E-52–E-58](../research/01-evidence-ledger.md)、[原协议](sap-failure-supervision-v2/study.yaml) | 未生成独立 registry/data plane；历史权重已在 2026-08-24 清理时删除 |
| sap-contextual-uncertainty-v1 | 路线关闭，`do_not_resume: true` | [协议末尾 conclusion / not_produced](sap-contextual-uncertainty-v1/study.yaml)、[DE-3](../research/02-dead-ends.md) | 未生成独立 registry/data plane；只保留日志与配置 |

前两项的运行数量包含归档资格测试、故障注入、取消和重试；不能用完成率评价模型。
SAP 的原始配置和命名保留供溯源，引用结论时以 ledger 的更正为准。

## 到哪里找文件

| 内容 | 位置与用途 |
|---|---|
| 注册 run 的实际路径 | [逐 run 清单](../docs/maintenance/2026-09-30-runs.tsv)，32 行，附 registry 行号、状态、manifest/status 核对结果 |
| Contextual Risk v2 输出 | `artifacts/contextual-risk-v2/`，按 registry/run ID 定位 |
| OQ1 输出 | `artifacts/oq1-kl-variance-hubness-v1/`；部分 run 实际位于其下同名嵌套目录，已绑定 manifest，不能直接改名 |
| 历史关键 run 的导入映射 | [critical-runs.json](contextual-risk-v2/legacy/critical-runs.json)；属于 legacy 映射，不应与 32 个注册 run 混算 |
| 旧 checkpoint、日志 | `ckpts/`、`logs/`：只读证据；[资产索引](../research/06-assets.md) |
| SAP 剩余证据 | `artifacts/dead-lines/sap-fix1-campaign-provenance/`；不能据此声称权重仍在或可重新评测 |
| 临时目录抢救与备份 | `artifacts/rescued_from_tmp/`、`artifacts/rescued_worktrees/`、`artifacts/backups/` |

清单发现两条早期 Contextual Risk 资格测试记录仍指向已删除的 worktree；
对应本地归档存在，使用项目的 canonical manifest digest 校验与登记一致。
这只是历史产物定位，不是可恢复运行的完整认证；没有改写旧 registry 路径。

## 使用边界

新研究先读 [research/README.md](../research/README.md) 和
[归档手册](../docs/agent/experiment-artifacts.md)。现有 study 的停止判定继续生效，
test 保持 sealed。不得就地重跑失败 run、篡改历史终态或因未登记而删除历史资料。
本次整理没有读取 test 指标、运行训练或修改研究 gate。
