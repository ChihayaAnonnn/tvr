# research/legacy/ —— research/ 建立之前的历史工作文档

这五份文件是 2026-08-23 建立 `research/` 体系**之前**的记录，于 2026-08-24 从仓库根目录移入此处。
**它们不再是权威来源**：凡与 `research/01-evidence-ledger.md` 冲突之处，一律以 ledger 为准。

保留全文的原因：ledger 是它们的蒸馏版，但若干 artifact 路径、原始表述和推导过程只存在于这里。

| 文件 | 原位置 | 性质 | 已知问题 |
|---|---|---|---|
| `PAPER.md` | 根目录 | 论文草稿 v0.1（2026-08-13） | 已确认多处数字缺陷，见 ledger「已知缺陷」段：base ckpt SHA 写错（D-02）、ViT-B/32 应为 B/16（D-03）、V2T 分母 1000 vs 1012（D-01）、Table 2 为 bf16 路径值 |
| `findings.md` | 根目录 | 详细实验发现 | `findings.md:209` 称 `query_models/module_sap.py` 已删除，与 master 实际状态矛盾（D-06）——该删除只存在于分支 `feature/reliable-retrieval-experiment-campaign` |
| `todo.md` | 根目录 | Stage 1 后续与修复清单 | 其中 "Fix 1 让 V2T AUROC 0.5661→0.7244" 的**归因已被推翻**（`w_sap_failure_bce` 在 `contextual_risk_sidecar_only` 下是 NO-OP），ledger 标为 CAVEATED |
| `progress.md` | 根目录 | 论文初稿 milestone 追踪 | 仅覆盖到 2026-08-13，后续进展见 `research/05-session-log.md` |
| `task_plan.md` | 根目录 | 论文初稿任务计划 | 同上，已完成 |

**新 session 不要从这里开始读**，入口是 `research/README.md`。
