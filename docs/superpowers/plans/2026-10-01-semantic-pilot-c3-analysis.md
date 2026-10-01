# C3 benchmark pilot analysis

Spec: docs/superpowers/specs/2026-10-01-semantic-retrieval-experiment-design.md sections 6–9.
Scope: CPU analysis only; no experiment reservation, asset reads, GPU or old study changes.

## Task 1: Metrics and policy (RED → GREEN)
Implement tied-group RC, official alias-expanded ranking, explicit denominators,
binary AP and frozen development operating grid/strongest scalar selection.
Test ties, official alias identity, abstention and absent denominators, infeasibility.
Expected: pytest tests/semantic_retrieval/test_pilot_analysis.py passes.

## Task 2: Calibration and statistics (RED → GREEN)
Separate natural/controlled weighted six-K calibration with distinct fit/target
manifests, benchmark protocol identity, unique pair relationships, single-class
unavailable states and raw-to-calibrated action-equivalent threshold mapping.
Paired source-component bootstrap jointly resamples arms and recomputes seed
metrics before averaging; keep seed scatter separate from query CI.
Expected: synthetic calibration identity/missing class/bootstrap tests pass.

## Task 3: Connected artifact pipeline (RED → GREEN)
Export inference score arrays to hash-bound NPZ, read a declared bundle once,
freeze development selection before consuming locked rows, publish run-local
numeric references, summary, plots, sensitivity/invariance/strata/case consumers.
Provide reserved CPU worker script using shared lifecycle and immutable inputs.
Expected: synthetic end-to-end bundle tests, hash tamper, frozen-target guard pass.

## Review focus
Protocol/track identity leaks, fit-target gallery separation, unweighted K mixture,
tie order dependence, cluster duplication as independent observations, post-hoc
locked selection, fake unavailable probability reporting, plots without consumers.

## Implementation record

Completed the connected score producer, bundle assembly, separate calibration,
frozen selection, locked analysis, official ranking, tied RC, source-component
statistics, per-group outputs and plots. See docs/agent/semantic-pilot-analysis.md
for exact integration contracts. All work stayed CPU/synthetic; no study control
files, real assets or GPU scientific experiments were changed/run.

Validation: C3 plus existing metrics: 22 passed; full semantic suite with
CUDA_VISIBLE_DEVICES='': 171 passed before the final additional paired-effect
unit case. Ruff passed. Initial full semantic run with visible CUDA failed four
runtime/qualification tests plus one end-to-end fixture setup because ambient
TF32 was enabled; the CPU run resolved every one. A final CPU suite is recorded
in the commit handoff. Existing PyTorch TypedStorage deprecation warning remains.

Ruling: analysis worker config derives runtime identity from its committed run
manifest, because reserving a config containing manifest SHA creates a circular
hash dependency. Immutable analysis options/bundle references remain reserved.
Ruling: inference exports full-gallery presence independently; finite-panel
calibration is not transferred silently to the full-gallery RC comparison.
Ruling: unavailable measured diagnostic/probability inputs are explicit and never
filled with synthetic scientific results. Actual Q0/science evidence is external
to the authorized CPU implementation scope.

Final verification (2026-10-01):
`CUDA_VISIBLE_DEVICES='' /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval`
→ **172 passed**, one existing TypedStorage warning, 61.28s.
C3 + ranking/binary regression scope → **22 passed**.
`ruff check` on the four Python deliverables → **All checks passed**.
`git diff --check` → no whitespace errors. Complete and finite RC separation and
full-gallery trained presence were included in this final suite.
