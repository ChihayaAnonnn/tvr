# Semantic benchmark pilot C3 pipeline

Scope is `benchmark_annotation`, on validation only. This implementation does not
create semantic no-answer labels or authorize MSRVTT/MSVD test access.

## Registered pipeline

1. E0 supplies fixed CLIP candidate panels in a hash-bound JSON manifest. Required
   identity fields are `schema_version=1`, `arm`, `seed`, `gallery_manifest_sha256`,
   `split_sha256`, `canonical_ids`, and `groups`. There is one group for each
   natural/controlled × K=4/8/16/32/64/128 combination. Each group contains `track`,
   `K`, and `arrays={path,sha256,format:'npz'}`. NPZ contains `query_ids` and
   `candidate_mask`; optional `initial_candidate_mask` records expansion origin.
   The mask columns are the sorted full-gallery canonical IDs. Controlled panels
   must have equal candidate-present and candidate-absent counts.
2. C0S/V0S reserved score workers run
   `scripts/export_semantic_pilot_scores.py --config SYSTEM_CONFIG --panels PANELS
   --panels-sha256 SHA`. They use the existing SystemConfig/committed reservation
   lifecycle and require the new study's `validate_pilot_gpu_qualification` gate.
   This is one process on CUDA ordinal 0. `DeploymentConfig.mode='calibrate'`
   denotes raw export without a prior probability calibrator; this worker itself
   performs no calibration. The checkpoint, episode/full-gallery manifest and
   panel references must be reserved inputs. Official GT comes from the v2
   hash-bound relation table, preserving official video IDs rather than marking
   every canonical alias positive. Full raw frames are decoded in batches of 8;
   encoded features are cached on CPU and pair batches moved to the device.
3. `build_pilot_bundle(fragment_references, locked_gallery=..., frozen=...,
   qualification=...)` assembles hash-bound score fragments. All global label,
   preprocessing and policy identities must agree. Each group binds arm, seed,
   split, track, mechanism, K, model SHA, split SHA, gallery and NPZ hash.
4. C0 runs `scripts/analyze_semantic_pilot.py --run-dir RUN --config CONFIG`.
   CONFIG is immutable JSON with exactly `bundle={path,sha256}`, `mode='freeze'`,
   `bootstrap_samples=10000`; CONFIG and all bundle/NPZ references are reserved
   dataset inputs. Its manifest evaluation config is exactly
   `{'eval_split':'val_development','qualification_scenario':'pilot_analysis'}`.
   Runtime identity is derived from the reserved run manifest, avoiding a circular
   config/manifest SHA dependency. It writes `provenance/pilot_frozen.json`.
5. V0 uses `mode='locked'` and `eval_split='val_locked'`, with frozen path/SHA in
   its bundle. It verifies analysis module bytes, model/label/policy/preprocessing
   identities, target gallery and source-component partition disjointness before
   reading locked observations. Frozen operating points and distinct B-H1/B-H2
   strongest-scalar identities cannot change.

A score fragment uses `galleries={gallery_version: {version,manifest_sha256,
canonical_ids,aliases}}`; groups refer to this version. Bundle additionally names
its deployment target as `galleries.locked`. The official relevance NPZ columns
are the flattened official IDs in canonical/alias order. Numeric observations
include full pair logits, canonical/official relevance, candidate masks, trained
presence logits, full-gallery presence logits and measured finite-panel latency.
A0/P1/A1 presence is never calibrated or interpreted as trained probability.

## Statistical and probability contract

Pair calibration uses each unique query–canonical relation once and rejects
conflicting duplicates. Presence calibration uses all six K equally, separately
for natural and balanced controlled mechanisms. Fit and target gallery versions
and manifest hashes must differ. States record model, preprocessing, policy,
label protocol, split, class counts/prevalence and candidate mechanism. Missing
classes produce unavailable states, never pseudo observations.

Raw development thresholds use the 21-point probability grid; scalar acceptance
uses 21 development quantiles. The present-coverage, absent false-accept and micro
precision constraints are checked before recall maximization. No feasible point
is reported as infeasible. Locked grids are descriptive and do not select policy.
Monotonic temperature+bias mapping preserves frozen actions by comparing mapped
logit boundaries, including numerical sigmoid saturation.

Official expanded rankings are independent of abstention. Finite-budget RC
counts coarse GT misses as top-1 errors. Complete-gallery RC uses actual full
matching/presence scores and has a separate plot panel; finite-panel presence
calibration is not silently applied to a full gallery. Exact confidence ties are
added together and AURC uses coverage increments times right-end risk. Each arm
also reports its own pair-only curve.

Primary comparisons use 97.5% paired source-component bootstrap intervals;
auxiliary comparisons use 95%. The same original components resample all arms,
panel copies/K variants stay with their source, each training seed's metric is
recomputed, and then seed effects are averaged. Three-seed scatter is reported
separately. Default is 10,000 replicates with seed 20261001. Fewer replicates are
fixture-only and force an inconclusive verdict. Natural expansion comparisons
with fewer than 30 independent miss components are descriptive.

## Outputs and limits

NPZ input hashes are checked; output numeric arrays use the run-local
content-addressed ArtifactArrayStore. No per-query numeric tables or bootstrap
samples are embedded in JSON. Detailed group summaries and array-reference
manifests are split into small hashed documents to remain below 1 MiB.

Nine plot families consume RC, frozen grid curves, set precision/recall,
reliability, effects/CI, K/cost, measured alias/permutation probes, measured
loss/gradient probes and deterministic case chains. Missing training or invariance
probe inputs remain visibly unavailable; no synthetic measurement is substituted.
Latency covers finite-panel matching/presence, while offline full-bank/encoding
work contributes separately to evaluation GPU hours. The full science asset,
throughput, new four-card qualification and actual seed training were not run as
part of the CPU implementation tests. These external evidence gates are required
for a scientific PASS, alongside both hypotheses and their two frozen comparators.
