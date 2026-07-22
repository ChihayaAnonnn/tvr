# Agent Documentation Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a general agent-documentation routing structure whose first registered category is a durable, structured log of confirmed past errors.

**Architecture:** `AGENT.md` is the root, extensible agent-documentation entry point. It routes to `docs/agent/README.md`, which is the category index; the index currently routes only to `docs/agent/past-errors.md`. Future agent guidance is added as a separate topic document and linked from the category index, without turning `AGENT.md` into a project summary.

**Tech Stack:** GitHub-Flavored Markdown; repository-local relative links.

## Global Constraints

- Do not summarize the project, its architecture, active work, or roadmap.
- Do not restore, replace, or modify the user-deleted `AGENTS.md`.
- Record only confirmed historical errors; do not infer unverified incidents from Git history.
- Keep `AGENT.md` as a route to the category index, not a direct route to any individual topic.
- Use `$...$` and `$$...$$` for any future Markdown mathematics recorded in the error log.

---

### Task 1: Create the root router, category index, and error archive

**Files:**

- Create: `AGENT.md`
- Create: `docs/agent/README.md`
- Create: `docs/agent/past-errors.md`

**Interfaces:**

- `AGENT.md` consumes the category index at `docs/agent/README.md` through a relative Markdown link.
- `docs/agent/README.md` consumes the history archive at `past-errors.md` through a relative Markdown link.
- `docs/agent/past-errors.md` produces immutable error IDs in the form `ERR-YYYYMMDD-NNN` for references from the category index or future topic documents.

- [x] **Step 1: Define the documentation acceptance checks**

```bash
test -f AGENT.md
test -f docs/agent/README.md
test -f docs/agent/past-errors.md
rg -n '\]\(docs/agent/README\.md\)' AGENT.md
rg -n '\]\(past-errors\.md\)' docs/agent/README.md
! rg -n '^# .*项目|^# .*概述|路线图|当前决策' AGENT.md docs/agent/README.md docs/agent/past-errors.md
rg -n 'ERR-20260719-001' docs/agent/past-errors.md
rg -n '\$\.\.\.\$|\$\$\.\.\.\$\$' docs/agent/past-errors.md
```

Expected before implementation: the file-existence checks fail because none of the three target documents exists.

- [x] **Step 2: Create `AGENT.md` as the general root router**

```markdown
# Agent 文档入口

本文件只提供 Agent 文档路由，不记录项目概述或具体主题内容。

## 文档路由

- [Agent 文档索引](docs/agent/README.md)
```

- [x] **Step 3: Create `docs/agent/README.md` as the category router**

```markdown
# Agent 文档索引

本目录按主题维护 Agent 工作所需的长期记录。新增主题时，创建独立文档并在本索引中添加链接。

## 已注册主题

- [历史错误记录](past-errors.md)
```

- [x] **Step 4: Create `docs/agent/past-errors.md` as the append-only history source of truth**

```markdown
# 历史错误记录

本文件按错误编号追加，仅记录已经确认的问题及其预防措施。

## ERR-20260719-001 — 数学公式分隔符无法渲染

- 发生日期：2026-07-19
- 范围：`docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md`
- 现象：公式在目标 Markdown 渲染器中无法渲染。
- 根因：文档使用了渲染器未保证支持的 `\\(...\\)` 和 `\\[...\\]` 数学分隔符。
- 修复：行内公式改为 `$...$`，块级公式改为 `$$...$$`。
- 预防规则：新增或修改 Markdown 数学公式时，只使用 `$...$` 与 `$$...$$`；提交前确认分隔符成对闭合。
- 验证：旧分隔符数量为 0；37 组块级公式和 74 组行内公式均成对闭合；`git diff --check` 无空白错误。
```

- [x] **Step 5: Run the acceptance checks**

Run:

```bash
test -f AGENT.md
test -f docs/agent/README.md
test -f docs/agent/past-errors.md
rg -n '\]\(docs/agent/README\.md\)' AGENT.md
rg -n '\]\(past-errors\.md\)' docs/agent/README.md
! rg -n '^# .*项目|^# .*概述|路线图|当前决策' AGENT.md docs/agent/README.md docs/agent/past-errors.md
rg -n 'ERR-20260719-001' docs/agent/past-errors.md
rg -n '\$\.\.\.\$|\$\$\.\.\.\$\$' docs/agent/past-errors.md
for doc_path in AGENT.md docs/agent/README.md docs/agent/past-errors.md; do
  whitespace=$(git diff --check --no-index /dev/null "$doc_path" || true)
  test -z "$whitespace"
done
```

Expected: every command exits with status 0, the two links resolve to the intended next routing layer, and no unrelated files are changed by this task.

- [ ] **Step 6: Commit only after user authorization**

```bash
git add AGENT.md docs/agent/README.md docs/agent/past-errors.md docs/superpowers/plans/2026-07-19-error-history-documents.md
git commit -m "docs: add agent error history routing"
```

Do not run this step unless the user explicitly requests a commit.
