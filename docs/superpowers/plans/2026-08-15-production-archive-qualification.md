# Production Archive Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the real four-GPU production launcher, controller, artifact binding, rank-zero lifecycle, and process-group cleanup with durable evidence.

**Architecture:** Add a backward-compatible portable artifact locator and a manifest-bound, validation-only qualification body behind the existing production launchers. Register exactly four runs in one phase at the main canonical artifact root, execute them sequentially, and verify their evidence without altering legacy data or prior failed attempts.

**Tech Stack:** Python 3, Bash, pytest, PyTorch distributed/NCCL, JSON/JSONL, Git.

## Global Constraints

- Use exactly four new run identities: success, non-owner-rank failure, owner-collective failure, and SIGTERM/different-collectives.
- Every run uses `run_train_msrvtt_contextual_risk_bg.sh` -> `run_train_msrvtt_bg.sh` -> `main_task_retrieval.py`.
- Do not access final-test data, JSFUSION, or MSRVTT test splits.
- Root new evidence at `/data2/hxj/project/UATVR/artifacts/contextual-risk-v2`.
- Preserve previous qualification evidence and all legacy roots.
- Stop after an unplanned failure; do not create or launch an unapproved retry.

---

### Task 1: Portable canonical artifact locator

**Files:**
- Modify: `tools/experiment_artifacts.py`
- Modify: `schemas/experiment_manifest.schema.json`
- Modify: `schemas/experiment_registry_record.schema.json`
- Test: `tests/test_experiment_artifacts.py`

**Interfaces:**
- Consumes: existing `RunManifest`, manifest validation, reservation, and registry serialization.
- Produces: optional `artifact_locator: str`; a normalized repo-relative locator whose resolution must equal `artifact_path`.

- [ ] **Step 1: Write failing tests** for an accepted normalized locator, rejected absolute/traversing locator, legacy manifests without the field, schema parity, and registry propagation.
- [ ] **Step 2: Run `pytest -q tests/test_experiment_artifacts.py`** and confirm only the new expectations fail.
- [ ] **Step 3: Implement the optional field** without adding it to legacy canonical payloads; derive it from the Git common root when the artifact path is below that root.
- [ ] **Step 4: Run `pytest -q tests/test_experiment_artifacts.py`** and confirm it passes.
- [ ] **Step 5: Commit** the tests, implementation, and schema changes.

### Task 2: Manifest-bound production qualification path

**Files:**
- Modify: `main_task_retrieval.py`
- Modify: `run_train_msrvtt_contextual_risk_bg.sh`
- Modify: `run_train_msrvtt_bg.sh`
- Create: `tools/archive_qualification.py`
- Test: `tests/test_main_task_retrieval.py`
- Test: `tests/test_training_launcher_scripts.py`
- Test: `tests/test_experiment_archive_e2e.py`

**Interfaces:**
- Consumes: manifest `config.artifact_qualification_scenario` with values `success`, `non_owner_failure`, `owner_collective_failure`, or `sigterm_different_collectives`.
- Produces: `run_archive_qualification(args, artifact_context)`; rank markers, finite validation metric, and an externalized two-MiB NPY reference.

- [ ] **Step 1: Write failing binding tests** proving the exact production sealed command and locator are required, manifest scenario mismatch is rejected, and inputs/test authorization are forbidden.
- [ ] **Step 2: Write failing launcher tests** proving the wrapper derives the scenario from the verified manifest, rejects an inherited override, avoids split/model preflight, delegates through the existing controller, and captures the raw log.
- [ ] **Step 3: Write failing worker tests** for success and all three intentional failure state machines, including readiness markers before faults.
- [ ] **Step 4: Run the narrow tests** and retain the RED output.
- [ ] **Step 5: Implement the minimal qualification body and launcher branches** before dataset/model construction while leaving the surrounding artifact lifecycle unchanged.
- [ ] **Step 6: Run the narrow tests** and confirm they pass.
- [ ] **Step 7: Commit** production code and tests.

### Task 3: Durable replica and phase preparation

**Files:**
- Create: `experiments/contextual-risk-v2/phases/stage0-production-archive-qualification.json`
- Modify: `experiments/contextual-risk-v2/registry.jsonl`
- Modify: `experiments/contextual-risk-v2/summaries/archive-qualification.md`
- Modify: `/data2/hxj/project/UATVR/.git/worktrees/experiment-artifact-archive/sdd/task-7-report.md`

**Interfaces:**
- Consumes: the two immutable earlier run directories and their hashes.
- Produces: verified no-overwrite replicas and four `reserved` records sharing one qualification phase.

- [ ] **Step 1: Mark the current report Stage 0 FAIL** and record the independent-review reasons.
- [ ] **Step 2: Inventory and hash the old evidence, copy to absent canonical destinations, then compare path/size/SHA-256** without deleting or rewriting originals.
- [ ] **Step 3: Generate four unique manifests** at the canonical root with portable locators, validation-only scope, exact production commands, and explicit lineage to both old attempts and human authorization.
- [ ] **Step 4: Reserve all four runs and update one phase document** while no launcher is running.
- [ ] **Step 5: Run reservation/phase validation and focused tests**, then commit all tracked reservations and reports.
- [ ] **Step 6: Verify `git status --porcelain --untracked-files=no` is empty** before launch.

### Task 4: Execute the approved four-run qualification

**Files:**
- Runtime only: `/data2/hxj/project/UATVR/artifacts/contextual-risk-v2/msrvtt/baseline/<run-id>/`

**Interfaces:**
- Consumes: four committed reservations from Task 3.
- Produces: terminal `completed`, `cancelled`, `cancelled`, `cancelled` runs plus raw logs and process/drain evidence.

- [ ] **Step 1: Audit GPUs, processes, and ports** and record the preflight in phase provenance.
- [ ] **Step 2: Launch success with its exact sealed command**, wait for terminal completion, and verify rank-zero lifecycle, four ranks, finite loss, and the two-MiB external array.
- [ ] **Step 3: Launch non-owner failure**, require the planned nonzero rank error to propagate and the production wrapper to publish `cancelled`; verify PGID drain.
- [ ] **Step 4: Launch owner-collective failure**, require rank zero to fail while peers are in the collective and publish `cancelled`; verify PGID drain.
- [ ] **Step 5: Launch SIGTERM/different-collectives**, wait for committed readiness markers, signal the production launcher process group, require bounded drain and `cancelled`.
- [ ] **Step 6: Stop immediately on any unplanned result**; do not create a fifth run or reuse an identity.

### Task 5: Qualification verification and closeout

**Files:**
- Modify: `experiments/contextual-risk-v2/registry.jsonl`
- Modify: `experiments/contextual-risk-v2/phases/stage0-production-archive-qualification.json`
- Modify: `experiments/contextual-risk-v2/summaries/archive-qualification.md`
- Modify: `/data2/hxj/project/UATVR/.git/worktrees/experiment-artifact-archive/sdd/task-7-report.md`
- Test: `tests/`

**Interfaces:**
- Consumes: all four terminal run directories and pre-qualification legacy inventory.
- Produces: auditable Stage 0 PASS or FAIL decision.

- [ ] **Step 1: Verify each manifest digest, registry transition, terminal output hash, audit chain, locator, raw log, and lineage**; require all JSON files below 1 MiB and numeric arrays externalized.
- [ ] **Step 2: Verify no qualification/torchrun process remains and compare the legacy inventory fingerprint** with the Task 6 baseline.
- [ ] **Step 3: Run focused tests**, followed by the complete repository pytest suite.
- [ ] **Step 4: Update the phase, summary, and Task 7 report** with exact run IDs, hashes, commands, terminal states, tests, and concerns.
- [ ] **Step 5: Commit** tracked terminal records, documentation, and any tested minimal fixes.
- [ ] **Step 6: Run final verification from the committed tree** and report the commit and Stage 0 result.
