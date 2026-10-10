# Evidence-conditioned retrieval — final implementation and research report

2026-10-10. Branch `codex/evidence-conditioned-retrieval`; retained worktree `/home/xujie/.codex/worktrees/ad3a/UATVR`. Tasks1–11 implemented; final product1d16e57; 690 related regression tests passed; final Q0-R5 four-CUDA-rank qualification passed. W0/D0/S1/S2/B1 and six S3 science producers completed. B2 variation gate failed; B3/B4/scaling correctly not run. C1/C2 scientifically unproved.

Authoritative current evidence and complete requested report: [research-evidence-2026-10-10.md](../../../experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md). Final independent [experiment audit](../../../experiments/evidence-conditioned-retrieval-v1/reports/EXPERIMENT_AUDIT.md) is WARN; independent [claim verdict](../../../experiments/evidence-conditioned-retrieval-v1/reports/CLAIM_VERDICT.md) is no for C1/C2 efficacy claims. Both semantic reviews are same-family/provisional; complete raw prompts/responses are retained locally under ignored .aris/traces according to skill privacy rules. All initial task commit IDs remain in the historical git version of this report; final resource JSON binds every actual run source SHA. The original worktree and user untracked files are preserved.

## Historical Native decisions and prior verification

The chronology below is the earlier development snapshot, retained for traceability. Statements about pending source approval, zero science, old test counts or old Q0 apply only to that earlier time, not the current report.

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

## Final independent review and completion

Independent source review found no remaining blocking correctness issue and reran 9 narrow tests. The experiment auditor independently verified all 9,156 transition classes, 384 component rows (192 distinct source components), archived lifecycles and resource totals. Audit verdict WARN retains historical optimizer-step/supervision and locked-input wording discrepancies, missing historical outer-controller launch-time source binding, and limitations of its generic concurrency/budget/successor enforcement. No observed campaign budget, duration or retry breach was found. These limitations are recorded, not retrospectively repaired or presented as universal enforcement.

The fresh result-to-claim reviewer judged both intended scientific efficacy claims unsupported by the available experiments (claim_supported=no). This means unproved/untested in the frozen pilot, not a universal refutation. Software integration and qualified GPU paths are complete; the full learned acquisition policy has no scientific value-head or calibration checkpoint, because the B2 prerequisite failed. B3/B4, locked-policy comparisons, scaling and multiseed runs remain stopped. No paper-level effectiveness or novelty claim is established.

Actual source-bound outputs include the reference and five matcher checkpoints and all preregistered transition artifacts. Final all-history allocated wall-time consumption is12.108477896449 scientific GPUh plus0.469454636127 software GPUh, including failures;35 runs are terminal (25 completed/6 failed/4 cancelled before spawn). Test remains sealed and science authorization remains closed. Product code is unchanged from1d16e57 and Q0-R5's qualified source f32bc82; tests are unchanged from the recorded690-pass regression. Final review/document changes do not require rerunning scientific experiments or GPU qualification.

Primary implementation commits and all per-run source identities are in the immutable evidence report and registry. Final evidence/reporting commits include19acabf (negative B2 and science closure),5fec22d (all-history accounting),0d8cdf6 (full evidence and resources),541dd67 (factual SIGTERM wording correction),eac1ef1 (audit/precheck); this final review is recorded in the subsequent git commit. The development branch/worktree is retained, with original user files preserved.

Final documentary verification command: `/home/xujie/.conda/envs/tvr/bin/python experiments/evidence-conditioned-retrieval-v1/analysis/verify_final_reviews.py` — exit0,180 checks passed. It verifies review/evidence hashes, typed cited values, all35 terminal lifecycles, closed science/test gates and unchanged product/tests; it adds no efficacy evidence. Result: `experiments/evidence-conditioned-retrieval-v1/reports/final-review-verification.json`. Final `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader` showed all five A800 GPUs at14MiB/0% utilization after all task workers ended.
