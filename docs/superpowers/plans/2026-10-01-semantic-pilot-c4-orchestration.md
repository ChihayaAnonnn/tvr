# Semantic Pilot C4 Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the approved benchmark pilot's archived phases and qualify the new C1/C2/C3 chain before scientific GPU work.

**Architecture:** Reuse the existing reserve/start/terminal/finalize manager. A pilot protocol module builds exact batches and authenticates committed predecessor summaries, registry records, lifecycle identities and output hashes; the existing pilot permission boundary invokes it on reserve and start. A separate synthetic v2 qualification entry consumes the new sampler, arm modes and calibration identities, and publishes run-local evidence for the new Q0 gate.

**Tech Stack:** Python 3.10, pytest, PyTorch FP32/DDP, existing Scheme A artifact manager, JSON/YAML and Git.

**Spec:** `docs/superpowers/specs/2026-10-01-semantic-retrieval-experiment-design.md`, sections 9–10 and their referenced model/data contracts.

## Global Constraints

- Budget is `144+8+24=176 GPU 小时`; topology is physical GPUs `[0,2,3,4]` and four unique ranks.
- `test_authorized=false`; only train, val_development, val_calibration and once-frozen val_locked are admissible.
- New control/data planes are `experiments/semantic-answerability-pilot-v1/` and `artifacts/semantic-answerability-pilot-v1/`.
- Entire reserved batch is committed before any worker starts; workers never edit tracked files.
- Exact source/config/input/manifest SHA binding and verified external numeric arrays remain mandatory.
- Failed/cancelled runs cannot be resumed or silently retried; a successor needs explicit tracked reviewed authorization and a new phase/run identity.
- Old Q0 evidence and YAML PASS booleans do not qualify the new C1/C2/C3 protocol.
- This implementer performs CPU tests only; root owns real inputs, reservations and GPU execution.

## Review Focus

- A forged or dirty committed-summary reference must reject before reservation/start (Task 1).
- A completed predecessor with missing/mismatched archived outputs or uncommitted terminal records must reject (Task 1).
- An incomplete arm/seed batch, wrong source, borrowed old-study gate or over-budget batch must reject (Tasks 1–2).
- A scientific worker must never start before new four-GPU Q0 evidence, nor access locked before C0 freeze (Tasks 1–2).
- A resumed sampler or detached A2/A4 gradient path must be observed using real C2/C3 components (Task 3).

### Task 1: Evidence-backed pilot permission and dependencies

**Files:** Create `src/semantic_retrieval/runtime/pilot_protocol.py`; modify `src/semantic_retrieval/runtime/pilot_policy.py`; create `tests/semantic_retrieval/test_pilot_protocol.py`.

**Interfaces:** `phase_contract(phase_id)`, `verify_phase_evidence(repo_root, evidence_ref, expected_phase)` and `assert_phase_authorized(repo_root, manifest, operation)` consume tracked summary refs `{path,sha256,commit}` and ordinary manifest selection lineage. They enforce D1 → Q0-prep → Q0 → E0 → P1a/P1b/P2a/P2b → C0 → V0, fixed training batches and explicit budget ceilings.

- [ ] Write tests for YAML toggles, foreign/malformed/dirty evidence, uncommitted terminal records, broken output hashes, phase dependencies, fixed batches and sealed locked/test.
- [ ] Run the new file and observe expected missing-feature failures.
- [ ] Implement authentication through Git bytes, existing lifecycle loader and exact registry terminal records; preserve CPU preparation variants.
- [ ] Run protocol and existing pilot-policy tests; commit only owned files.

### Task 2: Connected phase planning, provisioning and execution

**Files:** Extend `pilot_protocol.py`; create `scripts/orchestrate_semantic_pilot.py`; create `tests/semantic_retrieval/test_pilot_orchestration.py`.

**Interfaces:** `build_phase(phase_id, *, phase_path, scheduled_timestamp_utc, run_templates, evidence)` produces existing manager payload `{phase,study,runs}` with exact arm/seed batch. `provision_run_config(repo_root, run_dir, template)` injects reserved runtime identity into immutable run-local config. CLI plan/reserve/run/finalize delegates to existing manager, commits batch reservation/terminal summary, invokes sealed command once and stops on failures.

- [ ] Write tests demonstrating real batch reservation/provisioning, startup refusal before commit, no duplicate retry, command failure archival and finalized predecessor handoff.
- [ ] Run tests and observe expected failures.
- [ ] Implement the CLI using structured argv and exact immutable templates; verify dependency and budget checks before mutations.
- [ ] Run integration tests and commit only owned files.

### Task 3: New synthetic v2 qualification chain

**Files:** Create `src/semantic_retrieval/runtime/pilot_qualification.py`, `scripts/qualify_semantic_pilot.py`, and `tests/semantic_retrieval/test_pilot_qualification.py`.

**Interfaces:** Synthetic prep publishes hashed v2 label references, raw-video/CLIP fixtures and candidate orders. `qualify_pilot_chain(context,cfg,device)` exercises C2 `PilotSampler`, `controlled_panel`, `pilot_trainable_prefixes`/model arm modes and checkpoint sampler progress, plus C3 `fit_pilot_calibration`/`calibrated_probabilities`. Q0 report includes protocol version, four-rank FP32/hardware identity, gradient paths, sampler/RNG/optimizer restoration, tamper rejection and separate natural/controlled gallery calibration identities.

- [ ] Import committed C1/C2/C3 dependency changes into this worktree without editing their files.
- [ ] Write CPU tests for real v2 fixture/chain and report rejection when old, incomplete, one-rank or mismatched evidence is presented as Q0 PASS.
- [ ] Observe RED, implement shared CPU/GPU chain, and run GREEN including CPU four-rank coverage where required.
- [ ] Run relevant suite and full project suite, record actual results in this plan, and commit only owned files.

## Execution Record

- Baseline: `a096015`; isolated branch `codex/semantic-pilot-c4`.
- Approved design and mandatory research/study/registry constraints read. No experiment reservation or GPU execution by this agent.
- Ruling: add archived score-export batches C0S (dev/cal) and V0S (locked), with E0/C0S/V0S budgets 4/8/12 GPUh and CPU C0/V0 analysis. C3 confirmed inference needs separate workers; unregistered subsidiary GPU work would violate Scheme A. Q0P synthetic CPU prep is independent; Q0 itself requires ready D1.
- Task 1 RED: 11 missing-module failures and one existing fail-closed test passed. Baseline CPU policy: 30 passed.
- Task 1 first GREEN investigation: existing partial-manifest rejection fixtures omit selection_lineage; guard must raise ValueError rather than KeyError. Failed lifecycle fixture requires running → validating → failed. Corrected both causes and updated the test graph for the C0S/V0S ruling.
