# 实验归档强制规则

本仓库的可靠检索实验采用 Scheme A：`experiments/contextual-risk-v2/` 是受跟踪的
控制面，`artifacts/contextual-risk-v2/` 是被忽略的数据面。任何 Agent 在计划、预留或启动实验前，必须读取 `experiments/contextual-risk-v2/study.yaml` 和 `experiments/contextual-risk-v2/registry.jsonl`。

1. 通过 `reserve-phase` 预留唯一 run，并先提交整批 `reserved` registry 记录。
2. 新输出只可写在 `artifacts/contextual-risk-v2/<dataset>/<stage>/<run-id>/`；不得将新实验直接写入 `logs/` 或 `ckpts/`。
3. worker 运行期间不得修改任何受跟踪文件；实时状态仅写入 run-local `status.json` 和 append-only audit。
4. 仅从干净源码启动，并绑定精确的 source SHA、命令、配置、输入哈希和 manifest SHA-256；未提交或不完全一致的预留必须拒绝。
5. 超过 1 MiB 的数值数组必须外置为 NPY/NPZ 引用；JSON 不得嵌入张量、原始相似度矩阵、逐查询数组或 bootstrap 样本，并须校验每个引用哈希。
6. 所有选择、消融、阈值、K 和校准只使用 validation；在研究 gate 明确授权前，MSRVTT/MSVD test 保持 sealed。
7. 同一 phase 的所有 worker 终态后，追加 terminal registry 记录并生成简洁的受跟踪摘要，再提交该批记录。
8. 保留失败运行的日志、状态、命令和哈希证据；不得静默重试。`failed/cancelled` 是不可变终态，绝不可原地重试或复用相同 run ID。只有人工审查和授权后，才可在后续 phase 预留新的唯一 run ID；新 manifest 以既有的 `parent_run_ids` 指向 predecessor，并在 `selection_lineage` 记录已审核的重试理由。对旧记录的更正只可使用 append-only `correction` 与同一 run 的 `supersedes_sha256`；现有 schema 不允许 correction 跨 run 指向新尝试。

恢复仅适用于未终态且 audit 中只有可修复的单个待提交 transition：先运行 `verify-run`。
不能证明身份、源码或输出完整性时停止并人工处理，不得新建同名覆盖目录。Stage 0 的
四卡资格测试通过并记录前，不得启动任何 GPU 科学实验。

详细命令和事故恢复步骤见 [docs/agent/experiment-artifacts.md](docs/agent/experiment-artifacts.md)。
