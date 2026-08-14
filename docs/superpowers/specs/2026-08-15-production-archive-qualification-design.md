# Production Archive Qualification Design

## Status and decision

The first Stage 0 qualification remains immutable evidence, but it does not
qualify the production protocol because it invoked a test-only torchrun worker.
Stage 0 therefore remains **FAIL** until this design's four new registered runs
finish and their evidence is reviewed.

The corrected qualification is one phase containing exactly four unique runs:

1. success -> `completed`;
2. non-owner rank failure -> `cancelled`;
3. lifecycle-owner failure while peers occupy a collective -> `cancelled`;
4. SIGTERM while ranks occupy different collective states -> `cancelled`.

No run ID is reused. A failure outside the deliberately injected outcome stops
the qualification; it does not authorize an implicit retry.

## Production boundary

Every sealed command is exactly:

```text
env ARTIFACT_RUN_ID=<run-id> bash run_train_msrvtt_contextual_risk_bg.sh
```

The wrapper verifies the committed reservation, derives the qualification
scenario from that committed manifest, and delegates to
`run_train_msrvtt_bg.sh`. The base launcher retains its existing controller,
whole-process-group ownership, TERM/KILL drain, and raw combined stdout/stderr
capture. The worker remains `main_task_retrieval.py`; its normal
`_prepare_artifact_run` binding and rank-zero start/validate/complete-or-fail
lifecycle execute unchanged around a deliberately small qualification body.

The qualification body initializes the real four-rank NCCL process group but
does not instantiate the production model or open any MSRVTT split. Success
writes one finite scalar metric and a two-MiB content-addressed NPY array.
Failure scenarios produce rank readiness markers before the planned fault so
the retained evidence proves which rank/collective state was exercised.

Qualification behavior is not an environment-controlled test hook. Its
scenario is part of the committed manifest's effective configuration. The
wrapper rejects an inherited scenario override, and worker binding checks that
the scenario, locator, validation-only scope, and no-input contract match the
manifest.

## Durable artifact identity

New evidence is rooted at the long-lived canonical directory:

```text
/data2/hxj/project/UATVR/artifacts/contextual-risk-v2
```

New manifests and registry records include an optional, validated
`artifact_locator`, relative to the Git common repository root. Legacy
manifests and records remain valid without it so their canonical JSON and
digests do not change. A locator must be normalized, traversal-free, and
resolve to the manifest's absolute `artifact_path`.

The two earlier worktree-rooted qualification directories are copied without
overwrite to the durable tree and verified by path, size, and SHA-256. The
originals stay untouched; the copies are retention replicas, not rewritten
authorities.

## Evidence and safety constraints

- Capture raw launcher stdout/stderr in each run's `logs/train.log` from process
  start; store diagnosis separately.
- Use validation metrics only. Do not access final-test data, JSFUSION, test
  CSVs, or canonical legacy experiment outputs.
- Keep every JSON file below 1 MiB and externalize numeric arrays.
- Preserve the Task 6 legacy-root fingerprint.
- Before launch, check GPU processes, conflicting ports, and tracked-source
  cleanliness. After every terminal run, prove the controller PGID drained.
- Commit all four reservations in one phase before any launch. Execute them
  sequentially from that clean committed source.
- Preserve the complete append-only lineage from both earlier Stage 0 attempts
  and state that this phase is the human-authorized correction for the review.

## Acceptance

Stage 0 passes only if all four planned terminal states, manifests, audit
chains, raw logs, rank markers, process-group drain evidence, portable
locators, JSON limits, external arrays, and legacy fingerprint verify; focused
and full repository tests must also pass. Otherwise Stage 0 remains FAIL.
