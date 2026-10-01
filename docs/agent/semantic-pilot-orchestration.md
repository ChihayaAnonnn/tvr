# Semantic pilot C4 workflow

The controller links successful committed archives in this order:
`D1 → Q0P/Q0 → E0 → P1a → P1b → P2a → P2b → C0S → C0 → V0E → V0S → V0`.
Q0P can prepare synthetic inputs independently; Q0 requires the real D1 quantity/source gate.
D1S source export can precede D1 and remains its declared parent. D1 evidence is the
single completed `benchmark_compile` seed0 batch, not the source-export summary.

Budgets are training144, qualification8 and evaluation24 GPUh. Mechanical evaluation
subphases allocate E0=4, C0S=8, V0E=2 and V0S=10. Training is exactly18 runs,
8 GPUh each, across the fixed four training batches. All scientific workers use FP32
and the Q0-authenticated UUID order for physical GPUs0,2,3,4. No YAML flag supplies PASS.

## Prepare and qualify the new chain

Reserve `Q0P` with variant `pilot-q0-prep`, dataset `synthetic`, GPU count0 and empty
scientific categories. Evaluation is exactly `eval_split=val` and
`qualification_scenario=pilot_q0_prep_v2`. Its sealed command runs
`scripts/qualify_semantic_pilot.py --mode prepare-production --run-dir {artifact_run_dir}`
with CUDA visibility disabled and `PYTHONPATH=src:.`.

The CPU worker publishes `provenance/pilot-input-inventory.json` and
`provenance/pilot-system-template.json`, plus129 distinct synthetic raw videos,
hash-bound v2 annotation/relations/ID maps, candidate orders and random formal
ViT-B/32 weights. These are explicitly software fixtures, not benchmark evidence.

Q0 uses variant `pilot-q0-v2`, the template's exact effective SystemConfig and its
entire input inventory. Runtime identity is injected into
`provenance/system-config.json` after reservation. The sealed four-rank command is
`python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/qualify_semantic_pilot.py
--mode four-gpu --run-dir {artifact_run_dir} --config {artifact_run_dir}/provenance/system-config.json`.
`CUDA_VISIBLE_DEVICES` must be the four physical UUIDs in the frozen order.

The shared chain exercises actual C2 dynamic panels, known global counts, P1
checkpoint/sampler/rank-RNG/Adam restore, an unknown batch, pre-backward DDP gradient
probes and updates for A2/A3/A4, checkpoint tamper and rank consistency. A planned
in-worker SIGTERM tests interruption and restoration without creating or reusing a
terminal cancelled run. Actual C3 full-gallery scoring and separate natural/controlled
fits check cross-gallery and cross-track identity. Diagnostic galleries intentionally
share synthetic content; they make no held-out scientific claim. Training and gradient
reports remain archived if calibration fails. Only a committed successful four-CUDA-rank
`pilot-qualification.json` authorizes subsequent GPU workers; CPU output cannot qualify.

## Fixed candidate production

E0's immutable body has exactly:

```json
{"schema_version":1,"clip":"ClipConfig object","weights":{"path":"ABS","sha256":"SHA"},
 "splits":{"train":{"episodes":"file ref","gallery":"file ref"},
           "val_development":{"episodes":"file ref","gallery":"file ref"},
           "val_calibration":{"episodes":"file ref","gallery":"file ref"}}}
```

Use `candidate_manifest_config(body, phase_id='E0')`; reserve the body, weights,
episode/gallery/label files and every raw-video identity. The sealed worker is
`scripts/prepare_semantic_pilot_candidates.py --run-dir {artifact_run_dir} --config ABS_BODY`.
It uses frozen CLIP pooled text and mean eight-frame features, cosine ranking and
canonical tie breaking, retaining original source-component/label provenance. It writes
`provenance/pilot-candidates.json`: ranked129-candidate inventories/orders, full rank
NPZs, six natural and balanced controlled K panels, and15 arm/seed panel manifests per
evaluation split. These refs directly feed C2 training and the C3 score CLI.

V0E has only `val_locked` in its body and uses `phase_id='V0E'`. It cannot rank or encode
locked queries before an authenticated C0 freeze. V0E/V0S/V0 must reserve the exact
`provenance/pilot_frozen.json` produced by C0, including model, target gallery, policy,
label and analysis code identities.

## Whole-batch controller

`templates` supports every C4 phase: Q0P, Q0, E0, P1a/P1b/P2a/P2b, C0S,
C0, V0E, V0S and V0. Supply the complete tracked summary map as `--evidence`
and the actual Python/PyTorch/CUDA environment as `--environment`. Q0 additionally
requires `--gpu-uuids UUID0,UUID2,UUID3,UUID4`; its worker independently checks the
physical indices and actual ordinal UUIDs. E0 requires `--clip-weights` pointing to
a JSON file reference for the scientific pretrained weights. V0E reuses E0's exact
weight/config identity. Q0P's random software weights never become scientific weights.

The factory commits immutable metadata before `plan`/`reserve`. Only raw videos
enter authenticated sharded input indices. Configs, labels, episodes, checkpoints,
panel metadata and analysis arrays stay directly bound. Score panel metadata is
created after training and binds the completed arm/seed's exact parent summary,
run ID, config, checkpoint and metadata. V0S additionally binds C0's freeze and
full argsort candidate orders for actual frozen expansion.

C0 templates assemble development/calibration observations from C0S; V0 bundles
contain only V0S locked observations and retain C0S summary references as gate
evidence. Both bind NPZ outputs and D1's target-gallery descriptor. C0 reads target-gallery metadata but
does not rank or score locked queries. Each analysis config fixes 10,000 bootstrap
replicates and binds all measured inputs. Qualification gates are recomputed from
successful committed D1/Q0 archives, all 18 complete C2 optimizer logs and gradient
probes, and C3's measured invariance arrays. Caller booleans cannot supply gates.
Invariance requires every arm/seed/track and the actual four alias multiplicities
times five permutations; missing coverage remains unavailable and prevents PASS.
These pilot gates make no confirmatory statistical-power claim.

Locked access additionally requires C0's archived `qualification_gates` to contain
exactly data, four_gpu_qualification, gradients and stability, all true, plus
`development_gate_passed=true`. C3 recomputes these from C0's authenticated inputs
before releasing the freeze. Missing/failed gates stop V0E/V0S/V0; invariance is
measured during V0S and is not a prerequisite for opening locked observations.

Executable source changes after Q0 reject scientific starts, including top-level
Python helpers such as `experiment_tracking.py`, runtime/model code, scripts and
archive schemas. Controller/worker commands remain exact hash-bound argv.

`scripts/orchestrate_semantic_pilot.py` exposes `templates`, `plan`, `reserve`, `run`
and `finalize`. Run with `PYTHONPATH=src:.`. Evidence is a JSON object whose phase keys
refer to tracked summaries via exact `{path,sha256,commit}`. Environment JSON is the
actual Python/PyTorch/CUDA/A800 FP32 environment; no simulated GPU identity is accepted.

For P1/P2 and C0S/V0S, `templates --phase-id PHASE --evidence EVIDENCE --environment ENV
--python-executable PYTHON --output experiments/semantic-answerability-pilot-v1/configs/NAME.json`
constructs the complete fixed batch from authenticated archive outputs. It binds the
real E0 data and P1 final checkpoint/config/metadata, derives trained model and sampler
identities, selects each exact C3 panel manifest, seals the UUID/import environment,
and commits templates and any raw-video index shards. Supply all four training summary
refs for either score-export batch. Small configs/episode/checkpoint refs remain direct;
large raw-video inventories use the shared authenticated `.input-index.json` contract.

`plan --phase-id PHASE --phase-path CONTROL_PHASE --templates TEMPLATES --evidence EVIDENCE`
commits the exact phase. `reserve --phase CONTROL_PHASE` reserves every member and
commits the whole reserved registry batch. `run --phase CONTROL_PHASE` verifies every
member before the first launch, injects exact immutable runtime config, executes the
sealed argv once, enforces each GPU-hour ceiling, closes all workers and commits the
whole terminal batch plus tracked summary. Failure cancels untouched siblings and stops
with a manual-review requirement. Terminal state and bytes cannot be reused or rewritten.

C0/V0 immutable bundle/config assembly and their CPU analysis commands use
[the C3 handoff](semantic-pilot-analysis.md). The controller does not select settings,
omit arms, retry failures, recover historical runs or authorize test. Existing old studies
and archived failure evidence remain unchanged.
