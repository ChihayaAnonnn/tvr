# Stage 1 baselines — retry authorization

**Cancelled parents (in order):**
1. `msrvtt-baseline-validation_baselines-seed000-20260815T120000Z-70622947894c`
   - Manifest SHA: `ccd672f4bbdb18da49098d0b9d180f6620942c5fa2180defff92e8816de057ba`
   - Cancellation commit: `640197b`
   - Cause: `features_path` trailing-slash mismatch (see below).
2. `msrvtt-baseline-validation_baselines-seed000-20260818T010000Z-329ec6ae2070`
   - Manifest SHA: `768cefc630eee475ea489269885673a84bfc58f4dad24cf947cb63ec9e67c63d`
   - Cancellation commit: `5846b0d`
   - Cause: reservation was bound to the pre-pivot launcher's git SHA;
     after the human-authorized topology pivot from GPUs 0,2,3,4 to
     0,1,2,3 (commit 45bd2dd, reverted at 05e007e, cherry-picked back as
     05f57fa) the launcher's git SHA changed and the archive manager
     refused launch. Corrective order: modify launcher first (commit)
     → reserve then. Attempt 3 follows that order.

## Cause

Infrastructure defect, not a scientific outcome. The launcher
`run_train_msrvtt_bg.sh:365` hardcodes
`--features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/"`
with a trailing slash. The Stage 1 phase spec stored the same directory
without a trailing slash. `main_task_retrieval.py`'s
`_verify_artifact_runtime_bindings` (line 179) uses strict string equality
between the committed manifest's `evaluation.features_path` and the
runtime `args.features_path`, and correctly refuses the run before any
GPU work.

The launcher contract is the authoritative side of this asymmetry —
`tests/test_contextual_retrieval_risk_launcher.py:135-138` explicitly
constructs the reference config with `+ "/"` for `features_path`. Task 3
missed the convention when authoring the phase specs.

## Fix

All seven Stage 1–7 phase specs (`stage1-baselines.json` … `stage7-final.json`,
25 `features_path` occurrences total) now record the path with the trailing
slash to match the launcher contract. No other spec fields were touched by
this repair; retrieval semantics are unchanged because the underlying
training code joins path components regardless of trailing slash — the
mismatch was cosmetic at the code level and enforced only by the runtime
binding check.

## Retry scope

Single-run retry of Stage 1 `validation_baselines` under the same seed,
same init_model, same trusted-v1 split, same 4-GPU A800 fp32 torchrun-4
topology, and the same hyperparameters. Deviations from the frozen
launcher/spec are limited to:
1. `features_path` gains the trailing slash the launcher requires
   (`msrvtt_224_12fps/`).
2. `CUDA_VISIBLE_DEVICES` visible-device set is `0,1,2,3` instead of the
   originally frozen `0,2,3,4`, because GPU 4 on node199 hosts an
   unrelated persistent VLLM service. All other launcher content is
   unchanged. The frozen environment invariants
   (`gpu_count=4`, `gpu_model=A800`, `precision=fp32`,
   `distributed_topology=torchrun-4`) remain enforced at launch and are
   satisfied by either topology.

The retry declares:
- `parent_run_ids`: both cancelled attempts.
- `selection_lineage[0].continuation_rule`: `fresh_from_canonical_a1`
  (same rule as the parent; no scientific pivot).
- `selection_lineage[0].authorization`: `human approved`.
- `selection_lineage[0].report`: this file.

## Authorization

Authorized to retry Stage 1 baselines under the amended phase spec and
the {0,1,2,3} launcher topology. Infrastructure-only causes; no
scientific choice at stake. When the {0,2,3,4} topology becomes stable
again the launcher pivot commit (05f57fa) should be reverted, but this
is a follow-up and does not affect the scientific validity of any
Stage 1 result produced during the pivot window.

Any further failure of the same run must produce its own authorized
retry lineage.
