# Stage 1 baselines — retry authorization

**Cancelled parent:** `msrvtt-baseline-validation_baselines-seed000-20260815T120000Z-70622947894c`
**Manifest SHA of parent:** `ccd672f4bbdb18da49098d0b9d180f6620942c5fa2180defff92e8816de057ba`
**Cancellation commit:** `640197b`

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
same init_model, same trusted-v1 split, same 4-GPU A800 topology, and the
same hyperparameters. Only the phase spec's `features_path` string
changes; the retry's manifest SHA will therefore differ from the parent's.

The retry declares:
- `parent_run_ids`: the cancelled attempt.
- `selection_lineage[0].continuation_rule`: `fresh_from_canonical_a1`
  (same rule as the parent; no scientific pivot).
- `selection_lineage[0].authorization`: `human approved`.
- `selection_lineage[0].report`: this file.

## Authorization

Authorized to retry Stage 1 baselines under the amended phase spec.
Infrastructure-only cause; no scientific choice at stake. Any further
failure of the same run must produce its own authorized retry lineage.
