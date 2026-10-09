# Evidence-conditioned retrieval — implementation report

2026-10-10. Branch: `codex/evidence-conditioned-retrieval`; retained worktree: `/home/xujie/.codex/worktrees/ad3a/UATVR`. Approved baseline: `a9a2f76`. Original master and its untracked user files were preserved. All 11 implementation tasks and the consolidated independent-review repair pass are implemented. Production four-GPU software qualification passed. Scientific effectiveness remains untested because source-definition/audit prerequisites have not passed; MSRVTT/MSVD test stays sealed.

## Implemented code

All product modules are under `src/semantic_retrieval/evidence/`.

| Function | Files |
|---|---|
| Query-specific observation state, nested views, attempted/successful frame accounting, cache identity | `types.py`, `views.py`, `frame_cache.py`, `encoding.py`, `config.py` |
| Frozen CLIP, stable temporal positions, query-conditioned Gaussian scores and reproducible MC contrastive objectives | `model.py:20`, `matcher.py`, `sampling.py`, `objectives.py`, `training.py`; compatible optional positions in `src/semantic_retrieval/models/temporal.py` |
| Frozen rich-score teacher and source-isolated transition supervision | `execution.py:509`, `data.py`, `transitions.py`; reviewed-semantic episode schema v3 in `src/semantic_retrieval/data/episodes.py` |
| Distribution/geometry/mean value heads and validation-only temperature calibration | `value.py`, `calibration.py` |
| Fixed-view/random/margin/entropy/mean/distribution policy baselines under one budget/journal | `baselines.py`, `policy.py`, `journal.py:11`, `runtime.py` |
| Exact training checkpoint, inference-only model loading, hash-bound successor observation/action recovery | `checkpoint.py`, `recovery.py:43` |
| Proper outcome/rich-score metrics, external arrays, paired source-component bootstrap, synchronized operation latency | `evaluation.py:102`, `timing.py`, `execution.py` |
| Protected reserve/prepare/start/lifecycle, new-study gate, full-capacity software qualification | `authorization.py:31`, `qualification.py:94`, `schemas.py`, `cli.py`; four evidence CLI scripts under `scripts/` |

No placeholder scientific path or bypass gate was introduced. Unknown relevance is not converted into a negative target. Candidate selection never inserts ground-truth positives. Offline diagnostic rich frames use separate storage and do not enter online evidence. Ambiguous unfinished work fails closed; successful imports validate the whole snapshot before copying and refuse conflicting output.

## Commits

| Task / change | Commit |
|---|---|
| 1 strict types/config | `a804583` |
| 2 observation isolation/time positions | `13754f4` |
| 3 probabilistic scoring/objectives | `8f1f48d` |
| 4 frozen reference/matcher training | `72d672d` |
| 5 transition compiler/source contracts | `9108068` |
| 6 value models/calibration | `c33f017` |
| 7 budgeted recoverable actions | `3c1f17a` |
| 8 evaluation and matched baselines | `3774bac` |
| 9 DDP/checkpoint | `4e8e233` |
| 10 archive/real runtime/CLI integration | `f5d2fad` |
| 11 protected CPU qualification/new independent study | `df5850b` |
| Review: semantic lineage and objective guards | `1db61d2` |
| Review: compatibility/recovery/metrics/distributed repairs | `00f3656` |
| Review report and preregistered Q0 | `b9d5d2f` |
| Complete reserved Q0 batch | `ee8f3c1` |
| Complete terminal Q0 batch and summary | `a0bb35a` |
| Registered qualification; science gate status | `33db0c0` |

## Actual verification

CPU command prefix: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider`.

- Affected regression: `tests/semantic_retrieval tests/test_experiment_artifacts.py tests/test_manage_experiment_run.py tests/test_experiment_artifact_docs.py -q` — **614 passed**, one upstream TypedStorage deprecation warning, **218.10 s**.
- Final committed evidence suite: `tests/semantic_retrieval/evidence -q` — **92 passed**, one upstream warning, **42.36 s**. It includes actual four-Gloo-rank protected CLI training/checkpoint restore, actual two-Gloo-rank BPE/CLIP/index writes, real partial video decode, and successor import where repeated frame decode raises immediately.
- Targeted RED→GREEN tests and independent findings are recorded in `docs/superpowers/reviews/2026-10-10-evidence-conditioned-retrieval-review.md`.
- The committed four-GPU qualification binding was checked through `validate_qualification`; a scientific reference reservation remains rejected with `scientific stage is not authorized by its study gate`.

## GPU qualification and archive

Study: `experiments/evidence-conditioned-retrieval-v1/`; Q0 control spec `phases/Q0.json`; terminal summary `summaries/Q0.json`; registry `registry.jsonl` contains the committed reserved/completed batch. Unique run:

`synthetic-baseline-evidence-q0-four-gpu-seed713-20261009T175208Z-b9d5d2fb3adb`

Archive:

`/data2/hxj/project/UATVR/artifacts/evidence-conditioned-retrieval-v1/synthetic/baseline/synthetic-baseline-evidence-q0-four-gpu-seed713-20261009T175208Z-b9d5d2fb3adb/`

Exact source SHA: `b9d5d2fb3adb5a510ed2236c5a1481e773a1486c`.
Manifest SHA-256: `f481e259a2d2240b2322948a47088cf29af676bdf8780e539090703fdb509c80`.
Qualification report SHA-256: `8ed5b1ea14c51ccb67bec6268f65bc9a20feec27a73afc8fd799623b7e7cb17f`.

Command, with `CUDA_VISIBLE_DEVICES=0,2,3,4` and `OMP_NUM_THREADS=1`:

```text
/home/xujie/.conda/envs/tvr/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/qualify_evidence_retrieval.py --config <archive>/provenance/worker-config.json --run-id synthetic-baseline-evidence-q0-four-gpu-seed713-20261009T175208Z-b9d5d2fb3adb
```

The complete actual argv is immutable in archive `command.sh` / `manifest.json`. Four distinct A800 cards, full ViT-B/32 random weights, FP32, synthetic videos only. **13/13 checks passed**: real CLIP, hidden-evidence isolation, loc/scale/value gradients, scale influences prediction, frozen teacher, three-class labels, unknown rank, equal rank parameters, exact checkpoint resume, action recovery, finite FP32. Five matcher variants and three value head kinds passed. Continuous and restored results are identical.

Worker exercise **57.96 s**, complete controller wall **71.945 s**, allocated GPU use **0.07994 GPU-hours**. Peak allocated CUDA memory **2,011,554,304 bytes per rank**; driver-reported usage approximately 2.9 GiB. Run limit 1,800 s, cumulative software limit 8 GPU-hours, maximum two diagnosed successors; no failure or retry occurred. All cards returned to idle after exit. **578 array references** in **348 JSON files** and external checkpoint payloads were verified before terminal completion; terminal inventory **780 files**. No terminal outputs were overwritten.

`provenance/evidence-qualification.json`, `controller-result.json`, `gpu-schedule.json`, `array-integrity.json`, `logs/controller-worker.log`, checkpoint and referenced arrays contain the evidence. Reserve → committed reservation → verify → clean bound worker → verify → terminal transition → finalize/commit was followed.

## Independent review and limits

The independent final reviewer returned 0 Critical, 11 Important and 1 Minor finding. The fixed-frame baseline finding was regraded Important because it affects a mandatory resource comparison. The author repaired all 12 in one TDD pass and reran affected tests; no second independent approval is claimed. No minor remains deferred. The full finding/resolution table is in the tracked review report.

Scientific runs: **none**; scientific GPU-hours: **0**; hypothesis: **not tested**. The missing audited near-duplicate relation/scientific definition prevents proving five-role isolation. A clarification was requested; no substitute definition was assumed. Pretrained B/32 input preparation, the full source audit and a frozen small falsification pilot remain prerequisites. `experiments/evidence-conditioned-retrieval-v1/reports/science-gate-status.md` records them. The registered qualification opens no science or test gate by itself.

OS/process interruption coverage remains narrower than every possible crash window; unprovable partial execution/prediction is rejected rather than re-executed. Synthetic successor module integration is not presented as a real-data archived scientific successor run. Successor latency explicitly covers the current worker segment; predecessor latency is not invented. Missing bound population/rich diagnostic inputs produce explicit undefined reasons. Positive software results do not establish innovation, superiority, or calibration quality.

## Native execution rulings (chronological, exhaustive)

- Ruling: Native tool initially selected historical HEAD e372c1c; unused new branch moved to approved a9a2f76 — no user changes — cost if wrong: wrong baseline, verified exact SHA.

- Ruling: User execution instruction supersedes plan CPU-only/no-study scope for new qualification and gated scientific work; scientific definitions and old gates unchanged — cost if wrong: authorization scope, retained fail-closed gates.

- Task 4: Ruling: add explicit pair_coverage Tensor[P,4] to ObservedFeatures — approved scale head requires real frame-count/coverage metadata unavailable from packed features alone; compute only from observed state — cost if wrong: interface migration, covered by packing and model tests.

- Task 5: Ruling: explicit FrozenTransitionModel adapter binds frozen model plus prediction/acquisition/observable-feature functions; sample_states requires observe callback — reachable states must reflect actual decode outcomes, never fabricated successful frames — cost if wrong: integration callback mismatch, covered by real-model compiler and later protected E2E.

- Task 7: Ruling: gain/cost test uses different real frame-block counts with validated profile costs instead of arbitrary overwritten costs 1/.1 — altered action estimates must be rejected — cost if wrong: changed illustrative fixture only. Budget methods and explicit state serialization added to Task 1 types for shared checkpoint/journal use.

- Task 8: Ruling: add point_temperature to predictive scores for deterministic/learned-temperature objectives and integrate model/training variant dispatch now — baseline temperature must receive the same training loss gradient used at inference — cost if wrong: baseline fairness, verified explicit temperature gradient test.

- Task 9: Ruling: mark evidence tests as a package — default pytest collection collides with existing test_distributed basename — cost if wrong: test import resolution, narrow and shared regressions verify. EvidenceStepper accepts precomputed LossTerm mapping or explicit loss_builder(model,batch); no implicit teacher/MC configuration guessed from batch.

- Task 10: Ruling: journal verified prediction events and externally stored scores alongside actions — temporal/MC costs must be reserved before execution and survive recovery without duplicate fees; query-specific chains share the run-local journal namespace safely — cost if wrong: acquisition budget/replay correctness, targeted failure/recovery tests.

- Task 10: Ruling: add execution.py for actual video/stage orchestration and cli.py for shared thin identity handling — runtime entry remains small and no duplicate archive manager — cost if wrong: more module boundaries, covered by integration. Optional CLI identity overrides must equal mandatory config runtime; requiring manifest SHA in sealed argv would create a self-referential command hash. Production-capacity synthetic GPU configuration is explicitly allowed only for gpu_qualification under the new user authorization.

- Task 10: Ruling: fixed initial-index specification declares label-independent role galleries; attach relevance only after sparse CLIP selection and bootstrap by query connected component — prevents GT insertion and candidate-union pseudo-replication — cost if wrong: candidate coverage/statistical scope.

- Task 10: Ruling: add explicit value.kind and policy.baseline fields and wire all acquisition baselines through the same journal/budget runtime — module-only baselines are insufficient; random choices derive from query/observation identity so recovery and permutation preserve choices — cost if wrong: config migration before first production run, validated strict serializers.

- Task 11: Ruling: actual CLI qualification checks continuous and restored executions within each protected worker, including checkpoint replay and execution-before-commit recovery; failure scenario separately enters controller terminal lifecycle — avoids exposing interruption/bypass CLI switches — cost if wrong: process-boundary coverage remains narrower than all possible crash points.

- Task 11: Ruling: per-rank qualification arrays/caches/journals live in separate run-local provenance namespaces — prevent parallel writers sharing immutable cache files while checkpoint publication stays rank 0 — cost if wrong: reference-root integrity, checked by real replay.

- Task 11: Ruling: establish independent study with scientific gates initially closed; reuse only verified archived random CLIP NPZ bytes, never old study permissions; source audit and new four-GPU report remain mandatory — latest explicit user authorization permits this extension — cost if wrong: protocol scope, enforce asset SHA at both reserve/start.

- Task 11: Ruling: qualification covers all five actual matcher forward/gradient paths at the same capacity; scientific qualification compares architecture and scale floor rather than random/pretrained weight file paths — otherwise valid pretrained science would be rejected — cost if wrong: qualification equivalence, report binds supported variants and unchanged executable source.

- Task 11: Ruling: freeze-producer transition readers compare loaded matcher weight hash and label role in addition to declared producer identity — a self-consistent transition config alone cannot prove correct parent — cost if wrong: stale supervised features, targeted mismatch test RED→GREEN.

- Final: Ruling: fixed_frames greedy canonical-first allocation regraded Minor → Important — required equal-budget baseline must advance candidates through view levels in rounds — cost if wrong: unfair baseline, real state advancement regression added.

- Final: Ruling: actual CUDA resource/qualification judgment deferred by reviewer will be resolved by a new archived production four-rank run after final executable commit — CPU/Gloo tests cannot establish it — cost if wrong: scientific gate remains closed until committed terminal evidence.

- Final: Ruling: real source near-duplicate relation cannot be inferred from a shared canonical video ID — approved design leaves its scientific definition unspecified; async user question pending while independent implementation and synthetic qualification continue — cost if wrong: role leakage; no scientific run may launch before audit.

- Final: Ruling: scientific superiority, calibration quality and uncertainty contribution remain unproved until gated matched-budget validation experiments — software checks and successful training cannot establish research validity — cost if wrong: overstated claim; preserve sealed test and report unmet gates.

Deferred minors: none.

Historical verification note: after terminal/control-plane commits, `verify-run` from the new HEAD correctly refuses to treat changed study/report files as the old run's bound source. The completed Q0 was reverified successfully from a clean detached worktree at its committed reservation `ee8f3c1`; that temporary verification worktree was removed. No validation check was bypassed, no historical artifact changed, and no worker was restarted. The current branch's `validate_qualification` accepts the committed Q0 because executable source is unchanged, while scientific authorization remains denied.

## Final whole-repository CPU result

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider -q` — **643 passed**, **1 upstream TypedStorage deprecation warning**, **261.83 seconds**. No executable changes followed Q0. The final branch retains the development worktree for review; no integration decision is needed to preserve this completed work.
