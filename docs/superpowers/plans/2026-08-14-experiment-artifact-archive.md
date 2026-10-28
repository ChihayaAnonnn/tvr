# Experiment Artifact Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ad-hoc `logs/` and `ckpts/` outputs with a tested run-centric archive, committed phase registry, externalized numeric arrays, agent documentation, and resumable legacy inventory.

**Architecture:** `experiments/contextual-risk-v2/` is the tracked control plane and `artifacts/contextual-risk-v2/` is the ignored data plane. Phase reservations and terminal summaries are committed in batches; workers mutate only run-local atomic status files. Existing A3 code receives an optional artifact context while legacy entry points remain readable but are forbidden for new campaign runs.

**Tech Stack:** Python 3.10, standard-library JSON/JSONL, NumPy NPZ/NPY, SHA-256, `fcntl`, Bash, pytest, Git.

## Global Constraints

- Implement in a new isolated worktree, never directly on `master`.
- Do not delete, move, or overwrite existing `logs/` or `ckpts/` artifacts.
- New campaign outputs must live below `artifacts/contextual-risk-v2/`.
- Numeric payloads larger than 1 MiB must be NPY/NPZ references, never embedded JSON.
- Registry records are append-only and phase-batched: committed reservations before launch, terminal records after all phase workers stop.
- Runtime workers must not modify tracked files.
- Launch requires clean tracked source and an exact committed reservation.
- Preserve failed-run evidence and reject silent retries with changed manifests.
- Update root `README.md`, root `AGENTS.md`, `docs/agent/README.md`, and the study README.

---

### Task 1: Define the archive domain model and schemas

**Files:**
- Create: `experiment_artifacts.py`
- Create: `tests/test_experiment_artifacts.py`
- Create: `experiments/contextual-risk-v2/schemas/manifest.schema.json`
- Create: `experiments/contextual-risk-v2/schemas/status.schema.json`
- Create: `experiments/contextual-risk-v2/schemas/metrics.schema.json`
- Create: `experiments/contextual-risk-v2/schemas/registry.schema.json`

**Interfaces:**
- Produces: `RunManifest`, `RunPaths`, `build_run_id()`, `validate_manifest()`, `make_run_paths()`, and canonical stage/state constants.
- Consumes: repository root, artifact root, study name, dataset, stage, variant, seed, timestamp, and git SHA.

- [ ] **Step 1: Write failing identity and validation tests**

Add tests covering legal stages, UTC IDs, path traversal, malformed hashes, duplicate separators, test authorization, and canonical path construction:

```python
def test_build_run_id_and_paths_are_canonical(tmp_path):
    run_id = build_run_id(
        dataset="msrvtt", stage="feature_ablation", variant="no_context",
        seed=0, timestamp_utc="20260814T020304Z",
        git_sha="0123456789abcdef",
    )
    assert run_id == (
        "msrvtt-feature_ablation-no_context-seed000-"
        "20260814T020304Z-0123456789ab"
    )
    paths = make_run_paths(tmp_path, "contextual-risk-v2", run_id)
    assert paths.manifest == paths.root / "manifest.json"
    assert paths.checkpoints == paths.root / "checkpoints"

@pytest.mark.parametrize("variant", ["../escape", "bad/name", "", "a b"])
def test_run_id_rejects_unsafe_variant(variant):
    with pytest.raises(ValueError):
        build_run_id("msrvtt", "baseline", variant, 0,
                     "20260814T020304Z", "0" * 40)
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_artifacts.py`

Expected: collection fails because `experiment_artifacts` does not exist.

- [ ] **Step 3: Implement immutable dataclasses and strict validators**

Implement these public signatures:

```python
ALLOWED_STAGES: frozenset[str]
ALLOWED_STATES: frozenset[str]

@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    status: Path
    command: Path
    logs: Path
    checkpoints: Path
    metrics: Path
    arrays: Path
    plots: Path
    provenance: Path

# Public call signatures implemented in this task:
# build_run_id(dataset: str, stage: str, variant: str, seed: int,
#              timestamp_utc: str, git_sha: str) -> str
# make_run_paths(artifact_root: PathLike, study: str, run_id: str) -> RunPaths
# validate_manifest(payload: Mapping[str, object]) -> dict[str, object]
```

Validation must reject unknown top-level fields, unsafe names, non-absolute artifact paths, non-64-character SHA-256 strings, `test_authorized=true` outside stage `final`, and missing parent lineage for derived runs.

- [ ] **Step 4: Add matching JSON Schemas and assert parity**

The four schema files must declare `additionalProperties: false`, required fields, enums, and SHA patterns. Add a parity test comparing Python stage/state constants with schema enums.

- [ ] **Step 5: Run tests and commit**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_artifacts.py`

Expected: PASS.

```bash
git add experiment_artifacts.py tests/test_experiment_artifacts.py \
  experiments/contextual-risk-v2/schemas
git commit -m "feat: define experiment artifact contracts"
```

---

### Task 2: Implement phase-batched registry and atomic runtime lifecycle

**Files:**
- Modify: `experiment_artifacts.py`
- Modify: `tests/test_experiment_artifacts.py`
- Create: `experiments/contextual-risk-v2/registry.jsonl`

**Interfaces:**
- Produces: `append_registry_records()`, `load_registry()`, `verify_committed_reservation()`, `reserve_run_directory()`, and `transition_run_status()`.
- Consumes: validated manifest, committed registry path, `RunPaths`, expected state, next state, PID, host, and exit code.

- [ ] **Step 1: Write failing lifecycle and concurrency tests**

Test legal transitions, illegal backwards transitions, exclusive reservation, manifest mismatch, missing committed reservation, atomic replacement, terminal immutability, and two processes appending under `fcntl.flock`:

```python
def test_worker_requires_exact_committed_reservation(git_repo, manifest):
    append_registry_records(git_repo / REGISTRY, [reservation_record(manifest)])
    git_repo.commit_all("reserve phase")
    verify_committed_reservation(git_repo, REGISTRY, manifest)
    changed = {**manifest, "seed": 1}
    with pytest.raises(ValueError, match="committed reservation"):
        verify_committed_reservation(git_repo, REGISTRY, changed)

def test_terminal_run_is_immutable(run_paths, running_status):
    transition_run_status(run_paths, "running", "completed", exit_code=0)
    with pytest.raises(FileExistsError, match="terminal"):
        transition_run_status(run_paths, "completed", "running")
```

- [ ] **Step 2: Capture RED**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_artifacts.py -k 'registry or reservation or transition or terminal'`

Expected: failures for missing lifecycle functions.

- [ ] **Step 3: Implement append-only registry publication**

Use a sibling `.lock` file and `fcntl.flock(LOCK_EX)`. Validate every existing and new JSONL line before writing. Write combined content to a same-directory temporary file, `fsync`, then `os.replace`. Reject duplicate `(run_id, event)` pairs and require correction records to name `supersedes_sha256`.

- [ ] **Step 4: Implement committed-reservation verification**

Read the committed registry with:

```python
subprocess.run(
    ["git", "show", f"HEAD:{registry_relative_path.as_posix()}"],
    cwd=repo_root, check=True, capture_output=True, text=True,
)
```

Require one exact `reserved` record whose manifest SHA equals the run-local manifest SHA. Also require `git diff --quiet` and `git diff --cached --quiet` before worker launch.

- [ ] **Step 5: Implement atomic runtime transitions**

Allowed transitions are exactly those in the design. Write `status.json` through the existing `atomic_write_json` pattern, append `provenance/audit.jsonl`, and store UTC ISO-8601 timestamps. A terminal transition records exit code and output hashes.

- [ ] **Step 6: Run tests and commit**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_artifacts.py`

Expected: PASS, including the multiprocess append test.

```bash
git add experiment_artifacts.py tests/test_experiment_artifacts.py \
  experiments/contextual-risk-v2/registry.jsonl
git commit -m "feat: add committed experiment registry lifecycle"
```

---

### Task 3: Externalize large arrays and keep metric JSON small

**Files:**
- Modify: `experiment_artifacts.py`
- Modify: `main_task_retrieval.py`
- Modify: `tests/test_experiment_artifacts.py`
- Modify: `tests/test_contextual_retrieval_risk_metrics.py`
- Modify: `tests/test_contextual_retrieval_risk_training.py`

**Interfaces:**
- Produces: `ArrayReference`, `ArtifactArrayStore.write()`, `externalize_numeric_payload()`, and `verify_array_reference()`.
- Consumes: NumPy-compatible arrays, run array directory, semantic key, and JSON payload.

- [ ] **Step 1: Write failing storage and regression tests**

```python
def test_large_arrays_are_externalized_and_deduplicated(tmp_path):
    store = ArtifactArrayStore(tmp_path / "arrays")
    values = np.arange(2_000_000, dtype=np.float32)
    first = store.write("raw_wti", values)
    second = store.write("raw_wti_copy", values)
    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert len(json.dumps(first.as_dict()).encode()) < 1024

def test_a3_candidate_json_never_embeds_query_arrays(tmp_path):
    payload = build_fixture_a3_candidate(artifact_run_dir=tmp_path)
    assert len(json.dumps(payload).encode()) < 1_048_576
    assert "array_metadata" in payload["evidence"]
    assert "failure_risk_arrays" not in payload["evidence"]
```

- [ ] **Step 2: Capture RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_experiment_artifacts.py \
  tests/test_contextual_retrieval_risk_metrics.py \
  tests/test_contextual_retrieval_risk_training.py \
  -k 'external or array or candidate_json'
```

Expected: new tests fail because arrays are still embedded.

- [ ] **Step 3: Implement content-addressed array storage**

Store arrays at `arrays/sha256/${FIRST_TWO_HEX}/${SHA256}.npy`. Hash dtype, shape, and contiguous bytes using the existing `array_sha256` convention. Write through a temporary file and atomic rename. Reference documents contain only relative path, key, shape, dtype, byte size, format, and SHA-256.

- [ ] **Step 4: Integrate externalization into A1/A3 evidence**

Add optional `artifact_array_store` parameters to evidence builders. In artifact mode, externalize raw WTI, means, masks, A1 risk/vacuity, A3 risk, ranks, and bootstrap arrays before candidate/winner/final JSON publication. Legacy callers without a store retain current behavior.

- [ ] **Step 5: Enforce the 1 MiB JSON boundary**

Before artifact-mode JSON writes, recursively reject NumPy arrays, tensors, byte strings over 1 MiB, or serialized documents over 1 MiB. Error messages name the offending JSON path.

- [ ] **Step 6: Run compatibility tests and commit**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_experiment_artifacts.py \
  tests/test_contextual_retrieval_risk_metrics.py \
  tests/test_contextual_retrieval_risk_training.py \
  tests/test_uncertainty_protocol.py
git add experiment_artifacts.py main_task_retrieval.py \
  tests/test_experiment_artifacts.py \
  tests/test_contextual_retrieval_risk_metrics.py \
  tests/test_contextual_retrieval_risk_training.py
git commit -m "feat: externalize experiment array evidence"
```

Expected: tests pass; fixture candidate and final JSON remain below 1 MiB.

---

### Task 4: Add artifact-aware CLI and training integration

**Files:**
- Create: `scripts/manage_experiment_run.py`
- Create: `tests/test_manage_experiment_run.py`
- Modify: `main_task_retrieval.py`
- Modify: `run_train_msrvtt_contextual_risk_bg.sh`
- Modify: `tests/test_contextual_retrieval_risk_launcher.py`

**Interfaces:**
- Produces CLI commands `reserve-phase`, `verify-run`, `start-run`, `transition-run`, and `finalize-phase`.
- Adds `--artifact_run_dir`, `--artifact_run_id`, and `--artifact_manifest_sha256` to `main_task_retrieval.py`.

- [ ] **Step 1: Write failing CLI composition tests**

```python
def test_campaign_launcher_routes_every_output_to_artifact_run(repo_root):
    captured = capture_contextual_launcher(
        repo_root, run_id="msrvtt-baseline-smoke-seed000-20260814T020304Z-0123456789ab"
    )
    assert captured.output_dir == captured.run_dir / "checkpoints"
    assert captured.log_file == captured.run_dir / "logs" / "train.log"
    assert "--artifact_run_dir" in captured.worker_argv
    assert not captured.output_dir.is_relative_to(repo_root / "ckpts")
    assert not captured.log_file.is_relative_to(repo_root / "logs")
```

- [ ] **Step 2: Capture RED**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_manage_experiment_run.py \
  tests/test_contextual_retrieval_risk_launcher.py
```

Expected: missing CLI and legacy launcher path failures.

- [ ] **Step 3: Implement the administrative CLI**

`reserve-phase` consumes a committed phase JSON, builds manifests, creates run directories in `reserved` state, and appends reservation records. It never launches. `finalize-phase` requires all declared runs terminal, verifies hashes, appends terminal records, and writes a small validation summary.

- [ ] **Step 4: Add artifact context to the Python entry point**

When `--artifact_run_dir` is present:

```python
paths = make_run_paths(artifact_root, study, args.artifact_run_id)
verify_committed_reservation(repo_root, registry_path, manifest)
assert Path(args.output_dir).resolve() == paths.checkpoints.resolve()
array_store = ArtifactArrayStore(paths.arrays)
```

Rank zero owns lifecycle transitions; failures propagate across ranks and produce one terminal status. Eval metrics write below `metrics/`; checkpoints remain below `checkpoints/`.

- [ ] **Step 5: Convert the registered A3 launcher**

Require `ARTIFACT_RUN_ID` and derive all paths from the committed manifest. Preserve the exact CUDA/NCCL/model contract. Remove timestamp-based root `logs/` and `ckpts/` creation for artifact-mode runs. A test-only `LEGACY_A3_LAYOUT=1` seam is rejected outside launcher test mode.

- [ ] **Step 6: Run integration and root tests**

```bash
bash -n run_train_msrvtt_contextual_risk_bg.sh
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_manage_experiment_run.py \
  tests/test_contextual_retrieval_risk_launcher.py \
  tests/test_contextual_retrieval_risk_integration.py
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests
```

Expected: all project tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/manage_experiment_run.py tests/test_manage_experiment_run.py \
  main_task_retrieval.py run_train_msrvtt_contextual_risk_bg.sh \
  tests/test_contextual_retrieval_risk_launcher.py
git commit -m "feat: route registered runs through artifact archive"
```

---

### Task 5: Make archive rules unavoidable in documentation

**Files:**
- Create: `README.md`
- Create: `AGENTS.md`
- Create: `docs/agent/experiment-artifacts.md`
- Create: `experiments/contextual-risk-v2/README.md`
- Create: `experiments/contextual-risk-v2/study.yaml`
- Modify: `docs/agent/README.md`
- Modify: `.gitignore`
- Create: `tests/test_experiment_artifact_docs.py`

**Interfaces:**
- Produces human and agent guidance plus machine-enforced canonical path/rule strings.
- Consumes exact archive paths, lifecycle, test hygiene, and phase gates.

- [ ] **Step 1: Write failing documentation contract tests**

```python
@pytest.mark.parametrize("path", [
    "README.md", "AGENTS.md", "docs/agent/experiment-artifacts.md",
])
def test_docs_name_archive_and_forbid_legacy_outputs(repo_root, path):
    text = (repo_root / path).read_text()
    assert "artifacts/contextual-risk-v2" in text
    assert "experiments/contextual-risk-v2/registry.jsonl" in text
    assert "不得将新实验直接写入 `logs/` 或 `ckpts/`" in text
    assert "validation" in text.lower()
```

Also assert Git ignores `artifacts/example/file.bin` but not `experiments/contextual-risk-v2/study.yaml`.

- [ ] **Step 2: Capture RED**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_artifact_docs.py`

Expected: missing root docs and ignore rule failures.

- [ ] **Step 3: Write documentation and locked study config**

`AGENTS.md` includes all eight mandatory design rules. Root `README.md` gives commands to inspect study, list registry, reserve phase, verify run, and locate metrics. `study.yaml` encodes stages, variants, seeds `[0, 1, 2]`, bootstrap count `10000`, bootstrap seed `0`, stop thresholds, test-sealed policy, canonical A1 SHA, and hardware policy.

- [ ] **Step 4: Run tests and commit**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_experiment_artifact_docs.py tests/test_manage_experiment_run.py
git add README.md AGENTS.md .gitignore docs/agent \
  experiments/contextual-risk-v2 tests/test_experiment_artifact_docs.py
git commit -m "docs: enforce experiment archive workflow"
```

Expected: tests pass and canonical guidance is discoverable from the root.

---

### Task 6: Inventory legacy artifacts and migrate critical evidence

**Files:**
- Create: `scripts/inventory_legacy_artifacts.py`
- Create: `scripts/migrate_critical_artifacts.py`
- Create: `tests/test_legacy_artifact_migration.py`
- Create: `experiments/contextual-risk-v2/legacy/critical-runs.json`
- Create after execution: `experiments/contextual-risk-v2/legacy/inventory-summary.json`

**Interfaces:**
- Produces a resumable ignored JSONL inventory, tracked summary, and verified copied run directories.
- Consumes roots `logs/` and `ckpts/`, classification mapping, and destination artifact root.

- [ ] **Step 1: Write failing inventory and migration tests**

Test interrupted hash resume, symlink refusal, source mutation detection, destination collision, copy-plus-hash verification, dry-run, and no-delete behavior:

```python
def test_migration_copies_and_verifies_without_deleting_source(tmp_path):
    source = make_legacy_run(tmp_path / "ckpts")
    result = migrate_critical_run(source, tmp_path / "artifacts", move=False)
    assert source.exists()
    assert result.destination.exists()
    assert result.source_sha256 == result.destination_sha256
```

- [ ] **Step 2: Capture RED**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_legacy_artifact_migration.py`

Expected: missing script/module failures.

- [ ] **Step 3: Implement resumable inventory and safe migration**

Inventory every regular file with relative path, size, nanosecond mtime, SHA-256, classification, and completion state. Refuse symlinks and files changing during hashing. Resume from the last verified JSONL record. Migration defaults to copy-plus-verify and has no delete operation.

- [ ] **Step 4: Define and migrate critical runs**

The mapping includes canonical MSRVTT A1, registered A2-CG, selected A3, A3 DSL T=2, MSVD June baseline, and any canonical MSVD A1 discovered before migration. Unresolved items are recorded rather than guessed.

- [ ] **Step 5: Execute inventory and migration**

```bash
/home/xujie/.conda/envs/tvr/bin/python scripts/inventory_legacy_artifacts.py \
  --roots logs ckpts \
  --output artifacts/contextual-risk-v2/legacy/inventory.jsonl \
  --summary experiments/contextual-risk-v2/legacy/inventory-summary.json
/home/xujie/.conda/envs/tvr/bin/python scripts/migrate_critical_artifacts.py \
  --mapping experiments/contextual-risk-v2/legacy/critical-runs.json \
  --artifact-root artifacts/contextual-risk-v2 --copy --verify
```

Expected: originals remain and every migrated destination hash matches.

- [ ] **Step 6: Verify and commit tracked files only**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_legacy_artifact_migration.py
git diff --check
git add scripts/inventory_legacy_artifacts.py scripts/migrate_critical_artifacts.py \
  tests/test_legacy_artifact_migration.py \
  experiments/contextual-risk-v2/legacy
git commit -m "feat: inventory and migrate critical experiment artifacts"
```

---

### Task 7: Qualify the archive end to end

**Files:**
- Create: `tests/test_experiment_archive_e2e.py`
- Create after execution: `experiments/contextual-risk-v2/summaries/archive-qualification.md`

**Interfaces:**
- Consumes all archive APIs, CLI, docs, registry, launcher composition, and a CPU fixture model.
- Produces the Stage 0 decision required before GPU science runs.

- [ ] **Step 1: Write the end-to-end fixture test**

Reserve and commit one fixture run, verify authorization, transition through running/validating/completed, externalize a 2 MiB array, publish scalar metrics, append a terminal record, and prove no file appeared below root `logs/` or `ckpts/`.

- [ ] **Step 2: Run end-to-end and full suites**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_archive_e2e.py
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests
```

Expected: all tests pass; worker execution leaves tracked files unchanged.

- [ ] **Step 3: Run a four-rank artifact smoke without test access**

Reserve a committed `baseline` smoke run, launch one fixture training step, and verify four rank markers, one lifecycle owner, finite loss, externalized arrays, no test log, and terminal status. It is not a scientific candidate.

- [ ] **Step 4: Write qualification report and commit**

```bash
git add tests/test_experiment_archive_e2e.py \
  experiments/contextual-risk-v2/registry.jsonl \
  experiments/contextual-risk-v2/summaries/archive-qualification.md
git commit -m "test: qualify experiment artifact archive"
```

Expected: Stage 0 is passed only with fresh commands, test counts, smoke run ID, manifest/registry hashes, and output-tree evidence.
