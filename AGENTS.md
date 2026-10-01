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
8. 保留失败运行的日志、状态、命令和哈希证据；不得静默重试。`failed/cancelled` 是不可变终态，绝不可原地重试或复用相同 run ID。只有人工审查和授权后，才可在后续 phase 预留新的唯一 run ID；新 manifest 以既有的 `parent_run_ids` 指向 predecessor，并在 `selection_lineage` 记录已审核的重试理由。`selection_lineage` 必须引用受 Git 跟踪的报告或 commit，不得指向 worktree Git 管理状态中的 `sdd/`。对旧记录的更正只可使用 append-only `correction` 与同一 run 的 `supersedes_sha256`；现有 schema 不允许 correction 跨 run 指向新尝试。

恢复仅适用于未终态且 audit 中只有可修复的单个待提交 transition：先运行 `verify-run`。
不能证明身份、源码或输出完整性时停止并人工处理，不得新建同名覆盖目录。Stage 0 的
四卡资格测试通过并记录前，不得启动任何 GPU 科学实验。

详细命令和事故恢复步骤见 [docs/agent/experiment-artifacts.md](docs/agent/experiment-artifacts.md)。

任何 Agent 或新 session 在提出实验方案前，必须先读 [research/README.md](research/README.md)：
其中 `00-north-star.md`（当前研究命题与证伪判据）、`01-evidence-ledger.md`（可信数字及其
SOLID/CAVEATED/RETRACTED 标记）、`02-dead-ends.md`（已证伪方向，不得重提）为强制前置阅读，
`07-protocol-constraints.md` 汇总了本文件之外仍然生效的常设约束。

## 新 benchmark pilot 路径合同

`experiments/semantic-answerability-pilot-v1/` 为独立控制面，
`artifacts/semantic-answerability-pilot-v1/<dataset>/<stage>/<run-id>/` 为数据面。
计划、预留或启动本 pilot 前还须读取其 study.yaml 与 registry.jsonl。
上述规则 1–8 同样适用；新路径不复活旧 study，不授权 test。
C1 仅授权 CPU 来源/编译；科学 GPU 仍由新增链路四卡资格与明确 gate 拒绝守卫控制。

## 授权双卡软件预检路径（2026-10-01）

用户批准 GPU 3、4 的梯度通路与断点恢复预检。独立控制面为
`experiments/dual-gpu-preflight-v1/`，数据面为
`artifacts/dual-gpu-preflight-v1/<dataset>/<stage>/<run-id>/`。
执行前必读本 study 与 registry；规则 1–8 同样适用。
仅合成数据、正式容量随机权重、FP32、两个 CUDA rank；不得替代四卡 Q0、
不得科学训练或解封 test，不改变既有 Q0 预留。
