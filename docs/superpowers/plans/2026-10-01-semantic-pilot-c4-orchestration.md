# Semantic Pilot C4 Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

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

- [x] Write tests for YAML toggles, foreign/malformed/dirty evidence, uncommitted terminal records, broken output hashes, phase dependencies, fixed batches and sealed locked/test.
- [x] Run the new file and observe expected missing-feature failures.
- [x] Implement authentication through Git bytes, existing lifecycle loader and exact registry terminal records; preserve CPU preparation variants.
- [x] Run protocol and existing pilot-policy tests; commit only owned files.

### Task 2: Connected phase planning, provisioning and execution

**Files:** Extend `pilot_protocol.py`; create `scripts/orchestrate_semantic_pilot.py`, `runtime/pilot_workflow.py`, `runtime/pilot_candidates.py` and `scripts/prepare_semantic_pilot_candidates.py`; add focused orchestration/workflow/candidate tests.

**Interfaces:** `build_phase(phase_id, *, phase_path, scheduled_timestamp_utc, run_templates, evidence)` produces existing manager payload `{phase,study,runs}` with exact arm/seed batch. `provision_run_config(repo_root, run_dir, template)` injects reserved runtime identity into immutable run-local config. CLI plan/reserve/run/finalize delegates to existing manager, commits batch reservation/terminal summary, invokes sealed command once and stops on failures.

- [x] Write tests demonstrating real batch reservation/provisioning, startup refusal before commit, no duplicate retry, command failure archival and finalized predecessor handoff.
- [x] Run tests and observe expected failures.
- [x] Implement the CLI using structured argv and exact immutable templates; verify dependency and budget checks before mutations.
- [x] Run integration tests and commit only owned files.

### Task 3: New synthetic v2 qualification chain

**Files:** Create `src/semantic_retrieval/runtime/pilot_qualification.py`, `scripts/qualify_semantic_pilot.py`, and `tests/semantic_retrieval/test_pilot_qualification.py`.

**Interfaces:** Synthetic prep publishes hashed v2 label references, raw-video/CLIP fixtures and candidate orders. `qualify_pilot_chain(context,cfg,device)` exercises C2 `PilotSampler`, `controlled_panel`, `pilot_trainable_prefixes`/model arm modes and checkpoint sampler progress, plus C3 `fit_pilot_calibration`/`calibrated_probabilities`. Q0 report includes protocol version, four-rank FP32/hardware identity, gradient paths, sampler/RNG/optimizer restoration, tamper rejection and separate natural/controlled gallery calibration identities.

- [x] Import committed C1/C2/C3 dependency changes into this worktree without editing their files.
- [x] Write CPU tests for real v2 fixture/chain and report rejection when old, incomplete, one-rank or mismatched evidence is presented as Q0 PASS.
- [x] Observe RED, implement shared CPU/GPU chain, and run GREEN including CPU four-rank coverage where required.
- [x] Run relevant suite and full project suite, record actual results in this plan, and commit only owned files.

### Task 4: Authenticated analysis handoff

Root requested a connected C0/V0 factory and removal of caller-supplied gate booleans. Extend the workflow with immutable bundle/config/qualification documents assembled from successful committed D1/Q0/training/score archives. Recompute gradient and stability gates from actual C2 logs, and invariance from hash-bound C3 probe arrays with required coverage; missing measurements remain unavailable and cannot pass. C3 consumes `validate_analysis_qualification(context, document)` before using any gates. No confirmatory power claim is inferred from pilot sample quantities.

- [x] Observe RED for changed executable dependencies, incomplete training evidence and forged qualification gates.
- [x] Implement archive-derived gate producer/validator and C0/V0 bundle/config templates with all inputs reserved.
- [x] Coordinate C3 frozen expansion/invariance schemas, verify CPU integration, and record any unavailable measurement honestly.

## Execution Record

- Baseline: `a096015`; isolated branch `codex/semantic-pilot-c4`.
- Approved design and mandatory research/study/registry constraints read. No experiment reservation or GPU execution by this agent.
- Ruling: add archived score-export batches C0S (dev/cal) and V0S (locked), with E0/C0S/V0S budgets 4/8/12 GPUh and CPU C0/V0 analysis. C3 confirmed inference needs separate workers; unregistered subsidiary GPU work would violate Scheme A. Q0P synthetic CPU prep is independent; Q0 itself requires ready D1.
- Task 1 RED: 11 missing-module failures and one existing fail-closed test passed. Baseline CPU policy: 30 passed.
- Task 1 first GREEN investigation: existing partial-manifest rejection fixtures omit selection_lineage; guard must raise ValueError rather than KeyError. Failed lifecycle fixture requires running → validating → failed. Corrected both causes and updated the test graph for the C0S/V0S ruling.
- Task 1 GREEN: 42 protocol/CPU-policy tests passed in 1.36s. Owned commit `a089479`; C1/C2 dependency cherry-picks are not C4 deliverables.
- Task 2 RED: three missing-orchestrator failures. Actual CPU worker success/failure paths now execute through committed reservation and terminal summary. Existing source verification rejects archived execution before checking state; controller now explicitly rejects terminal state first and preserves observed vs controller exit codes separately.
- Ruling confirmed by root: E0 never ranks locked queries. Add V0E after C0 freeze for locked CLIP candidate panels. Evaluation allocation is E0=4, C0S=8, V0E=2, V0S=10 GPUh; C0/V0 are CPU analysis. This only subdivides the approved 24 GPUh budget.
- Task 2 controller GREEN: 45 combined protocol/policy/controller tests passed in6.80s, then49 in7.20s after complete-batch/quota/freeze gates.
- E0/V0E RED: four missing-module failures; GREEN: five tests passed in4.55s, including actual raw-video decode/CLIP encoding → ranks → v2 inventories → C3 panel files. Source labels/components and canonical source IDs are retained. Locked rank creation remains behind actual C0 freeze.
- New Q0 RED: four missing-module failures. CPU binding fixtures initially used inconsistent WORLD_SIZE environment at prepare/start; corrected fixture identity rather than weakening runtime guards. GREEN: four tests passed in15.13s, including actual four-rank Gloo hook/probe/backward, dynamic raw panels, P1 sampler/RNG/Adam restoration and real C3 scoring/calibration. Added a later-calibration failure test retaining partial evidence; latest shared-core/controller/workflow subset11 passed in30.24s.
- C1 source-proof/input-index dependencies integrated locally (`e7bf143`, `b043d18`, `f33988c`, `5add32a`); C2 `420f7ef` and C3 `742fdb0` integrated. These are dependencies, not C4 deliverables. Large gallery video refs now use the shared authenticated index contract; scientific configs/checkpoints remain direct.
- Training/score workflow RED: two missing-module failures; GREEN: fixed arm config and9000-input refusal tests pass; added901-real-file index expansion. CLI templates feeds exact committed batch configs to plan/reserve/run. Actual worker topology rechecks qualified UUIDs after environment binding, including C2 training.
- Full CPU suite:478 passed, one TypedStorage deprecation warning,128.89s. No real inputs/reservations/GPU workers launched by this implementer. CPU/fake fixture metadata cannot grant new Q0 GPU PASS. Real four-GPU qualification remains root's authorized task after integration and resource availability.
- Task 4 RED: seven missing gate APIs plus the `experiment_tracking.py` source-change bypass; later missing analysis/producer/score-parent factory paths and incomplete invariance coverage reproduced before implementation. GREEN: archive-based evidence tests12 passed in18.91s. Full committed software fixtures exercise C0/V0 handoff, reject forged/unreserved gates, and verify Q0 consumes its complete own inventory and E0 binds separate scientific weights. No GPU worker is executed by these fixtures. C3 confirmed actual system trajectory and twenty invariance probes use reserved frozen policies, argsort orders and measured `[N,4,5]` arrays; its final integration remains separate.
- Final owned-source verification:72 focused tests passed in51.46s; complete CPU suite500 passed in162.13s, one existing PyTorch TypedStorage warning. `git diff --check` and CLI help passed. C3 review commits `2d6cc974`/`72695a6` became available during this full run and will be integrated for a separate compatibility check; this500 result predates those commits. Real four-GPU PASS remains pending root execution.
- C3 compatibility: integrated math/producer fixes (`2d6cc974`, `72695a6`) and shared subset71 passed in104.83s. Actual manager-based boundary RED exposed missing `run_dir` and relative-summary handling; own alias plus C3 `fa98d2e` fixed it. V0 mixed-observation RED reproduced45 vs15 fragments; V0 now contains only15 locked producers. Ruff formatting/import fixes preserved explicit public gate re-exports; owned-source Ruff and format checks pass. Fresh C4 subset53 passed in45.01s plus real boundary/V0 tests2 passed in11.03s.
- Prelocked gate RED: five authenticated archived freezes with missing/false/extra/unreleased development gates incorrectly unlocked V0. C4 now requires the exact four authenticated C0 gates and explicit release; invariance is not a prelocked requirement. GREEN: seven archive gate/boundary/V0 tests passed in12.10s. C3 persists the recomputed four gates before freeze release. All C4 software tasks are implemented and checked; actual new four-GPU qualification is explicitly pending root, with no real workers/reservations or historical mutations by this implementer. Root owns fresh final repository verification after integration.
- Final shared boundary verification after C3 `fa98d2e` and `7139ab0`:84 protocol/CPU-policy/workflow/evidence/archive tests passed in25.22s. Owned source Ruff check, format check and diff check pass. Public gate re-exports remain explicit. CPU policy semantics match root's stricter preparation-branch routing; only formatting changes there. The prior full500 and shared71 results are recorded above with their source scope; no new full-suite or actual GPU claim is made here.
