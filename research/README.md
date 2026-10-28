# research/ —— UATVR 研究路径的持久记录（anti-drift anchor）

> **2026-09-30 导航补充：** [实验总览](../experiments/README.md)汇总现有 study 的停止状态；
> [论文索引](../related_work/README.md)汇总已下载资料。
> OQ1 v1 已于 2026-08-30 在 Stage 1 终止（inconclusive），见其
> [最终判定](../experiments/oq1-kl-variance-hubness-v1/summaries/stage1-arm0-baseline.md)。
> `00-north-star.md` 中的下一步计划与冻结协议的 `registered` 是历史状态，不能作为续跑授权。

**这是什么。** 本目录是 UATVR 课题跨 session 的**唯一权威记忆**。它存在的理由：
context 一满、新 session 一开，助手就会重新提出已被证伪的方向、混用不可信的数字、
忘记协议约束。本目录把"哪些结论是硬的、哪些是带 caveat 的、哪些已经死了、哪些规则
不能违反"固化下来，目标是 **5 分钟读完即可让一个全新 session 立刻正确**。

---

## 新 session 必读顺序

> **强制规则：在提出任何新实验之前，必须读完 `00-north-star.md` + `01-evidence-ledger.md`
> + `02-dead-ends.md` 三份文件。** 不读就提方案 = 一定会重复已被证伪的路。

1. **`00-north-star.md`** — 当前研究命题（thesis）与它的证伪判据。先知道我们在赌什么。
2. **`01-evidence-ledger.md`** — 每一条已确立的实证事实 + 数字 + split/protocol + artifact 路径
   + 置信标记（SOLID / CAVEATED / RETRACTED）。**引用数字前必须先在这里查状态标记。**
3. **`02-dead-ends.md`** — 已证伪的方向及其证据指针。**不要重试这里面的任何一条。**
4. `03-decisions.md` — 带日期的决策记录（ADR）。理解"为什么现在长这样"。
5. `04-open-questions.md` — 未决问题，每条都带"什么结果能结案"的判据。**新实验从这里挑。**
6. `05-session-log.md` — 倒序 session 日志（append-only）。看最近发生了什么。
7. `06-assets.md` — ckpt / split / launcher / script / figure 的权威路径与 SHA。
8. `07-protocol-constraints.md` — 不可违反的实验协议红线。**动 GPU 之前读这条。**

> 另有 `CLEANUP-PROPOSAL.md`（2026-08-23 由另一个 agent 写入本目录的磁盘/git 体积审计，
> 只读审计、未删任何东西）。它不属于上述 anti-drift 体系的 00–07，按需查阅即可。

## 写入规则

- 每条事实必须带 `file:line` 或 artifact 路径。找不到来源就写 `[未溯源]`，**不要编造**。
- 两个来源冲突时，**两个都记下来并标注冲突**，不要私自选一个。
- `05-session-log.md` 只追加、不改写历史条目。
- `01-evidence-ledger.md` 的行只能改状态标记（升/降级）并附日期，不能删。

## 相关但不在本目录的东西

- `research/legacy/` — 2026-08-24 从根目录移入的五份历史工作文档：`PAPER.md`（论文草稿 v0.1）、
  `findings.md` / `todo.md` / `progress.md` / `task_plan.md`。本目录 00–07 是它们的蒸馏版，
  **冲突时以 ledger 为准**；这些文档已知的数字缺陷见 `research/legacy/README.md` 的表格。
- `AGENTS.md` — 实验归档强制规则（Scheme A 控制面 / 数据面）。
- `docs/superpowers/{plans,specs,reports,tables}` — 原始 spec 与 report。
