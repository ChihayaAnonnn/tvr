# Experiment artifact archive qualification

## Decision

**Stage 0: PASS.** The corrected qualification exercised the real production
path end to end:

```text
run_train_msrvtt_contextual_risk_bg.sh
  -> run_train_msrvtt_bg.sh controller and whole-PGID drain
  -> main_task_retrieval.py committed artifact binding
  -> rank-zero lifecycle ownership
```

This authorizes validation-only campaign work. It does not authorize a final
test, JSFUSION access, or treat qualification fixtures as scientific results.

## Corrective lineage

Independent review rejected the earlier test-only torchrun qualification. Both
old registered runs and all three old probes remain unchanged. They were
copied without overwrite to the long-lived canonical root and verified by
relative path, size, and SHA-256; their originals remain in the worktree.

The corrected design/plan is commit `f68bf78`; portable locator support is
`f9c7505`; production qualification code is `fd0f721`; the phase definition is
`5d9f2ba`; the four reservations were committed together in `ce11ffd`; terminal
records and phase summary are `4e5bf38`. A post-run parser completeness fix is
`e11bd33` and does not alter the executed qualification state machine.

All new manifests name both earlier registered attempts in `parent_run_ids`
and record the human authorization and independent review in
`selection_lineage`. Every artifact uses a normalized repo-relative
`artifact_locator` resolving under `/data2/hxj/project/UATVR/artifacts`.

## Registered production runs

| Scenario | Run ID | Manifest digest | Terminal |
|---|---|---|---|
| success | `msrvtt-baseline-archive_prod_success-seed000-20260815T070000Z-5d9f2ba35ffa` | `d9151491979e19f4e5d6ef7805a75e12bbe7eeff335f74f2dbcfb7216a3bd933` | `completed`, exit 0 |
| non-owner rank failure | `msrvtt-baseline-archive_prod_nonowner_fail-seed000-20260815T070100Z-5d9f2ba35ffa` | `6e91a3f7a0530ae3ddab442b88e348e46d9219744e6a7f86b42f9a81e4d605fd` | `cancelled`, exit 1 |
| owner collective failure | `msrvtt-baseline-archive_prod_owner_collective_fail-seed000-20260815T070200Z-5d9f2ba35ffa` | `91be56d1394b2dbfe15ce85d3343dfff956050fd0e64201804e5d6bb56a8bc0c` | `cancelled`, exit 1 |
| SIGTERM, different collectives | `msrvtt-baseline-archive_prod_sigterm_mismatch-seed000-20260815T070300Z-5d9f2ba35ffa` | `b050fa1d6caf4443600bda3cd46c23515bdb95732e392372b90634fd0b9cf6f1` | `cancelled`, exit 143 |

Each run used the exact sealed command
`env ARTIFACT_RUN_ID=<id> bash run_train_msrvtt_contextual_risk_bg.sh`.
Each retained raw combined worker stdout/stderr in `logs/train.log` from
process start.

The success run performed a real four-rank CUDA/NCCL optimization step. Its
finite `qualification_loss` is `7.5`; `selected_validation.json` is 468 bytes.
The numeric payload is a `(524288,)` float32 NPY containing 2,097,152 bytes,
with protocol digest
`908251af60385de2b211f1fd4281d4e69cc3ece66815e1ccd2e019d24cace430`
and file digest
`647092735d0f83096419a07a203198e8caba708bd1effbd1b0ff903c75fb910a`.

The non-owner run preserves rank 2's planned traceback and torchrun's sibling
termination. The owner run preserves rank zero's planned failure while ranks
1-3 had marked entry to all-reduce, plus `worker_failure.json`. For SIGTERM,
ranks 0/2 marked all-reduce and ranks 1/3 marked broadcast before PID 3027873,
the outer production wrapper, received SIGTERM. The controller recorded TERM,
the bounded TERM timeout, KILL escalation, and complete PGID drain; no
`UNRESOLVED:live-process-group` record exists.

## Safety and integrity

- All qualification JSON files are below 1 MiB; the large numeric payload is
  only in content-addressed NPY storage.
- Every manifest is validation-authorized and test-unauthorized, has no input
  checkpoint or dataset split, and records `run_final_test=false`.
- No new run opened MSRVTT data, test CSVs, final-test artifacts, or JSFUSION.
- All terminal output hashes, manifest digests, audit chains, four-rank marker
  sets, registry transitions, and locators recomputed successfully.
- Task 6's 340 legacy files compare with the canonical inventory at
  missing=0, changed size/mtime=0, extra=0. The established fingerprint
  remains `441ae072e07c4a4683342ae2249c511143e5d2f5cf43f560c6f9ca0f964d7866`.
- No qualification process, worker PGID, GPU process, or listener on port
  29547 remained after the runs.

Old-evidence replica inventory digests were:

- first cancelled run: `6cc536925ff11bd8a50f572926398f5593272481a5d02e97c5e242ecf02e8a88`;
- old successor: `386ebbf213ad942e893db8ef72844c5a8c8836cd5afd09c7b35f23d8e52f5b42`;
- non-owner probe: `ae53ac7d07b93d90d9696024ff78b382b623ba9ab24dbc3339fcee39086bf58d`;
- collective probe: `8943c1dfbebd67ddcbb8d7c7b9af3c92690945d199ad2852a82ae110769fc8b8`;
- SIGTERM probe: `cfbb70f409dde4a01246e9d0c13de8ecc5eb41f3ed9ef97732859a9a0b097245`.

## Verification

Focused archive/launcher/legacy suite passed. The first full-suite run exposed
one parser-schema completeness regression introduced by the new option; a
focused RED/GREEN fix added the inert normal-A3 field to the dynamic contract
and required qualification to carry a complete artifact identity. Fresh full
verification then completed:

```text
611 passed, 1 skipped, 1 warning in 78.64s
```

The warning is the existing PyTorch `TypedStorage` deprecation warning.

Registry SHA-256 is
`cd13b6cde1e81acf9000b7a164aeb7a050f00b96097c2185e84993f09041b55b`;
phase SHA-256 is
`2aad8a7cfe4bf4acb63611db0849a803ecec384aeee9a51bdf399dc60065cc3c`;
terminal summary SHA-256 is
`c5b1297caad55456681b3bcb48a5b3f733afb5bb096fa9781e59b8d7a0eebadd`.
