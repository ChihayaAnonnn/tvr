# Reliable Retrieval Scientific Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a validation-gated MSRVTT/MSVD experiment campaign that establishes retrieval quality, isolates A3's contextual-risk contributions, confirms multi-seed reliability, and produces sealed final-test paper evidence.

**Architecture:** All runs use the qualified run-centric archive from `2026-08-14-experiment-artifact-archive.md`. A campaign controller reads committed phase specifications, launches only committed reservations, computes gates from validation artifacts, and never authorizes test access before the final configuration is frozen. A3 feature masks keep network capacity fixed during feature ablations; loss and K sweeps are separate phases.

**Tech Stack:** Python 3.10, PyTorch 2.1.2, NumPy, SciPy, four A800 GPUs, MSRVTT trusted-v1, MSVD, pytest, SHA-256.

## Global Constraints

- Do not begin until archive Stage 0 is recorded as passed.
- Use validation metrics only for baseline comparison, ablation, K, loss, checkpoint, calibration, or stopping decisions.
- Existing A3/DSL test numbers are historical exploratory evidence and cannot select new configurations.
- Use seeds `0`, `1`, and `2`; bootstrap uses 10,000 samples and seed 0.
- Keep the retriever, A1, and every non-A3 tensor frozen in A3 runs.
- A3 never modifies retrieval scores; raw and DSL retrieval results are reported separately.
- Each phase is reserved and committed before launch, and finalized in one terminal registry commit.
- Stop a failed phase according to its frozen gate; do not tune after failure.
- Write no new output below legacy `logs/` or `ckpts/`.

---

### Task 1: Add deterministic risk baselines and paired statistics

**Files:**
- Create: `prob_models/retrieval_risk_baselines.py`
- Create: `tests/test_retrieval_risk_baselines.py`
- Modify: `prob_models/retrieval_risk_metrics.py`
- Modify: `tests/test_retrieval_risk_metrics.py`
- Modify: `main_task_retrieval.py`

**Interfaces:**
- Produces `score_risk_baselines(scores, direction) -> dict[str, np.ndarray]`.
- Produces `paired_bootstrap_delta(labels, first, second, metric, samples, seed) -> dict`.
- Adds baseline metrics to validation artifact JSON without changing A1/A3 arrays.

- [ ] **Step 1: Write failing baseline tests**

Test `one_minus_top1`, `negative_margin`, `entropy`, `negative_energy`, and `score_variance` under score shifts, row permutations, direction transpose, ties, and non-finite inputs:

```python
def test_margin_risk_orders_ambiguous_query_above_clear_query():
    scores = np.array([[8.0, 1.0, 0.0], [3.0, 2.9, 0.0]], dtype=np.float32)
    risk = score_risk_baselines(scores, "t2v")["negative_margin"]
    assert risk[1] > risk[0]

def test_paired_bootstrap_is_deterministic():
    first = paired_bootstrap_delta(LABELS, A3, A1, "auroc", 10_000, 0)
    second = paired_bootstrap_delta(LABELS, A3, A1, "auroc", 10_000, 0)
    assert first == second
```

- [ ] **Step 2: Capture RED**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_retrieval_risk_baselines.py tests/test_retrieval_risk_metrics.py
```

Expected: import failures for the new baseline/statistics APIs.

- [ ] **Step 3: Implement baseline scores and paired bootstrap**

Use float64 CPU arrays, stable sorting, explicit direction orientation, and stratified resampling by failure label. Return point delta, 2.5/97.5 percentiles, sample count, seed, and valid replicate count. Reject constant labels and fewer than 95% valid replicates.

- [ ] **Step 4: Integrate validation-only reporting**

Add `result["risk_baselines"][method][direction]` with rho, AUROC, AURC, coverage, ECE, Brier, and NLL. Artifact mode externalizes arrays. Existing retrieval and A1/A3 result keys remain byte-compatible.

- [ ] **Step 5: Run tests and commit**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_retrieval_risk_baselines.py tests/test_retrieval_risk_metrics.py \
  tests/test_directional_eval.py
git add prob_models/retrieval_risk_baselines.py \
  prob_models/retrieval_risk_metrics.py main_task_retrieval.py \
  tests/test_retrieval_risk_baselines.py tests/test_retrieval_risk_metrics.py
git commit -m "feat: add retrieval risk baselines and paired statistics"
```

---

### Task 2: Make A3 feature and objective ablations explicit

**Files:**
- Modify: `prob_models/contextual_retrieval_risk.py`
- Modify: `modules/modeling_mulit.py`
- Modify: `main_task_retrieval.py`
- Modify: `experiment_tracking.py`
- Modify: `tests/test_contextual_retrieval_risk.py`
- Modify: `tests/test_contextual_retrieval_risk_loss.py`
- Modify: `tests/test_contextual_retrieval_risk_training.py`

**Interfaces:**
- Adds `--contextual_risk_feature_set` with exact choices `full`, `score_only`, `semantic_only`, `no_context`, `no_score_geometry`.
- Reuses `--w_contextual_risk_fit` and `--w_contextual_risk_rank` for objective ablations.
- Produces manifest fields `feature_set`, `active_feature_blocks`, `fit_weight`, `rank_weight`, and `topk`.
- Modifies `run_train_msrvtt_contextual_risk_bg.sh`: its sealed command remains
  `env ARTIFACT_RUN_ID=<run-id> bash run_train_msrvtt_contextual_risk_bg.sh`.
  The verified manifest is the sole source for a manifest-derived allowlist:
  `config.model.contextual_risk_feature_set`,
  `config.model.contextual_risk_topk`,
  `config.loss.w_contextual_risk_fit`,
  `config.loss.w_contextual_risk_rank`, `config.optimizer.seed`, and
  `config.scheduler.epochs`. The wrapper 不得接受任意 trailing argv or
  environment override for these values.

- [ ] **Step 1: Write failing feature-mask tests**

Use identical initialized heads and verify masked blocks are exactly zero while the concatenated width and trainable parameter count remain 186,113:

```python
@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_ablation_preserves_capacity(feature_set):
    head = ContextualRetrievalRiskSidecar(
        embed_dim=512, projection_dim=128, hidden_dim=128,
        topk=16, feature_set=feature_set,
    )
    output = head(LOGITS, TEXT, VIDEO, "t2v")
    assert output["features"].shape[1] == 426
    assert sum(p.numel() for p in head.parameters()) == 186_113
    assert_mask_matches_declared_blocks(output, feature_set)
```

Also test that BCE-only is `fit_weight=rank_weight=0`, BCE+fit is `(0.25, 0)`, BCE+rank is `(0, 0.25)`, and full is `(0.25, 0.25)`.

- [ ] **Step 2: Capture RED**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py \
  tests/test_contextual_retrieval_risk_training.py \
  -k 'feature_set or ablation or preserves_capacity'
```

Expected: parser/constructor failures for `feature_set`.

- [ ] **Step 3: Implement fixed-width feature masking**

Construct the full 426-dimensional tensor, then multiply named slices by constant zero/one masks. Do not create variant-specific layers. Record block offsets in one module constant and expose them in diagnostics.

- [ ] **Step 4: Validate configuration and manifests**

Reject unsupported combinations, require full A3 for final stage, and include feature/loss/K values in immutable resume and archive manifests. Checkpoint replay must reconstruct the exact feature set.

The launcher accepts only the values frozen in `study.yaml`: five feature
sets, four loss pairs, K in `{1,4,8,16,32}`, seeds `{0,1,2}`, and epochs
`{3,5}`. Tests must cover committed manifest -> wrapper argv ->
`_verify_artifact_runtime_bindings()` and reject category duplication or an
out-of-allowlist value before model/data setup.

- [ ] **Step 5: Run compatibility tests and commit**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py \
  tests/test_contextual_retrieval_risk_training.py \
  tests/test_experiment_tracking.py
git add prob_models/contextual_retrieval_risk.py modules/modeling_mulit.py \
  main_task_retrieval.py experiment_tracking.py \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py \
  tests/test_contextual_retrieval_risk_training.py
git commit -m "feat: parameterize contextual risk ablations"
```

---

### Task 3: Add campaign controller, phase specs, and gate engine

**Files:**
- Create: `scripts/run_reliability_campaign.py`
- Create: `prob_models/reliability_campaign.py`
- Create: `tests/test_reliability_campaign.py`
- Create: `experiments/contextual-risk-v2/phases/stage1-baselines.json`
- Create: `experiments/contextual-risk-v2/phases/stage2-features.json`
- Create: `experiments/contextual-risk-v2/phases/stage3-loss-topk.json`
- Create: `experiments/contextual-risk-v2/phases/stage4-multiseed.json`
- Create: `experiments/contextual-risk-v2/phases/stage5-msvd.json`
- Create: `experiments/contextual-risk-v2/phases/stage6-analysis.json`
- Create: `experiments/contextual-risk-v2/phases/stage7-final.json`

**Interfaces:**
- Produces CLI `prepare`, `launch`, `monitor`, `evaluate-gate`, `finalize`, and `authorize-test`.
- Produces pure gate functions `evaluate_stage1_gate()` through `evaluate_stage5_gate()`.
- Consumes committed study config, phase spec, registry, and run-local scalar metrics.

- [ ] **Step 1: Write failing gate and controller tests**

Cover every threshold boundary, missing/duplicate run, failed worker, test access before authorization, retry with changed manifest, and fake-launch command composition:

```python
def test_stage1_gate_uses_validation_only_and_exact_threshold():
    result = evaluate_stage1_gate(
        a3={"mean_auroc": 0.82, "mean_aurc": 0.25,
            "t2v": {"auroc": 0.81, "rho": 0.3},
            "v2t": {"auroc": 0.83, "rho": 0.3}},
        strongest={"mean_auroc": 0.81, "mean_aurc": 0.24},
        split="validation",
    )
    assert result.passed
    with pytest.raises(ValueError, match="validation"):
        evaluate_stage1_gate(TEST_PAYLOAD, STRONGEST, split="test")
```

- [ ] **Step 2: Capture RED**

Run: `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_reliability_campaign.py`

Expected: missing module and CLI failures.

- [ ] **Step 3: Implement pure gate engine**

Encode the exact approved thresholds. Return a serializable decision containing pass/fail, inputs, comparisons, failed checks, study SHA, phase SHA, and next authorized stage. No gate reads arrays or test files.

- [ ] **Step 4: Implement fail-closed orchestration**

`prepare` delegates committed reservations to the archive CLI. `launch` runs at most one four-GPU scientific training at a time on devices `0,2,3,4`; it may run CPU metric jobs concurrently but never another training. `monitor` records health without mutation. `authorize-test` requires completed Stage 5, frozen configuration hash, and an explicit Stage 7 reservation.

- [ ] **Step 5: Write concrete phase specifications**

Stage 2 variants are exactly `score_only`, `semantic_only`, `no_context`, `no_score_geometry`, `full`. Stage 3 losses are `bce`, `bce_fit`, `bce_rank`, `full`; K values are `1,4,8,16,32`. Stage 4 seeds are `0,1,2`. Every run states dataset, parent checkpoint, seed, epochs, continuation rule, feature set, losses, K, precision, hardware, validation-only flag, and an explicit `scheduled_timestamp_utc` used only as the run-identity schedule. If no start is scheduled, generate it at reservation time.

- [ ] **Step 6: Run tests and commit**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q \
  tests/test_reliability_campaign.py tests/test_manage_experiment_run.py
git add scripts/run_reliability_campaign.py \
  prob_models/reliability_campaign.py tests/test_reliability_campaign.py \
  experiments/contextual-risk-v2/phases
git commit -m "feat: add validation-gated reliability campaign"
```

---

### Task 4: Execute Stage 1 validation baselines

**Files:**
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage1-baselines.md`
- Create after execution: `experiments/contextual-risk-v2/summaries/validation.csv`

**Interfaces:**
- Consumes frozen MSRVTT validation evidence from canonical A1/A3 and intrinsic SAP/A2 artifacts.
- Produces one aligned baseline table and Stage 1 gate decision without accessing test.

- [ ] **Step 1: Verify Stage 0 and reserve Stage 1**

```bash
/home/xujie/.conda/envs/tvr/bin/python scripts/run_reliability_campaign.py \
  prepare --phase experiments/contextual-risk-v2/phases/stage1-baselines.json
git add experiments/contextual-risk-v2/registry.jsonl
git commit -m "exp: reserve contextual risk Stage 1 baselines"
```

Expected: every run is committed as `reserved`, `test_authorized=false`, and the tracked tree is clean.

- [ ] **Step 2: Evaluate aligned baselines**

```bash
/home/xujie/.conda/envs/tvr/bin/python scripts/run_reliability_campaign.py \
  launch --phase experiments/contextual-risk-v2/phases/stage1-baselines.json
/home/xujie/.conda/envs/tvr/bin/python scripts/run_reliability_campaign.py \
  evaluate-gate --stage 1
```

Expected: raw score, margin, entropy, energy, variance, SAP, A2, A1, and A3 use identical validation labels/order/digests.

- [ ] **Step 3: Finalize or stop**

If the exact gate fails, finalize Stage 1 as stopped and do not reserve Stage 2. If it passes, record authorization for Stage 2. Commit registry terminal records, validation CSV, and report.

---

### Task 5: Execute Stage 2 feature ablations

**Files:**
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage2-features.md`

**Interfaces:**
- Consumes Stage 1 pass and five seed-0 committed reservations.
- Produces aligned validation metrics, freeze audits, and Stage 2 gate.

- [ ] **Step 1: Reserve and commit all five runs**

Run the controller `prepare` command with `stage2-features.json`, inspect exact manifests, then commit reservations before launching.

- [ ] **Step 2: Launch sequentially and monitor to terminal**

Run controller `launch`; after each run verify finite loss, nonzero A3 gradients, no non-A3 mismatch, validation-only evidence, and artifact JSON below 1 MiB. Do not change epochs or hyperparameters between variants.

- [ ] **Step 3: Evaluate the frozen Stage 2 gate**

Require full A3 to beat both score-only and no-context by the approved mean AUROC/AURC threshold without directional degradation. On failure, stop the campaign and report which conceptual claim failed.

- [ ] **Step 4: Finalize and commit**

Append terminal records and commit `stage2-features.md`. Do not commit artifact tensors or logs.

---

### Task 6: Execute Stage 3 loss and Top-K pilots

**Files:**
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage3-loss-topk.md`
- Create after execution: `experiments/contextual-risk-v2/summaries/frozen-configuration.json`

**Interfaces:**
- Consumes Stage 2 pass, four loss variants, five K variants, and seed 0.
- Produces one frozen architecture/loss/K/configuration SHA for all later stages.

- [ ] **Step 1: Reserve and commit nine declared runs**

Confirm loss weights and K values match the phase specification exactly. Runs sharing the full `(0.25,0.25,K=16)` configuration reference one canonical result instead of retraining a duplicate.

- [ ] **Step 2: Launch and validate all unique runs**

Use only validation candidates. Preserve identical canonical A1, data digests, precision, effective batch, and continuation rule.

- [ ] **Step 3: Freeze configuration with deterministic tie-breaks**

Filter on `min rho >= 0.20`, retain candidates within 0.005 mean AUROC and 0.005 mean AURC of best, then choose smallest K, fewest active auxiliary losses, and earliest epoch. Write the complete configuration and SHA before any Stage 4 reservation.

- [ ] **Step 4: Finalize and commit**

Commit terminal registry records, report, and frozen configuration. Any later change requires a new study version and cannot use the current test authorization lineage.

---

### Task 7: Execute Stage 4 three-seed confirmation

**Files:**
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage4-multiseed.md`
- Create after execution: `experiments/contextual-risk-v2/summaries/multiseed-validation.json`

**Interfaces:**
- Consumes frozen configuration and seeds 0/1/2.
- Produces mean/std and paired 10,000-bootstrap A3-vs-A1 validation deltas.

- [ ] **Step 1: Reserve all seeds in one committed phase**

Seed 0 may reuse an exact Stage 3 run only when manifest and checkpoint hashes match; otherwise run all three. No seed-specific hyperparameter changes are allowed.

- [ ] **Step 2: Launch sequentially and collect validation arrays**

Verify each run's source, canonical A1, split, non-A3 state, output spread, and terminal hashes. Keep test unauthorized.

- [ ] **Step 3: Compute aggregate gate**

Require both directional point estimates to improve and either mean AUROC 95% delta interval strictly above zero or mean AURC interval strictly below zero. Report failures without selecting a favorable seed.

- [ ] **Step 4: Finalize and commit**

Commit terminal records and multi-seed summary. Reserve Stage 5 only on pass.

---

### Task 8: Establish a sealed MSVD protocol and run Stage 5

**Files:**
- Create: `dataloaders/msvd_protocol.py`
- Create: `tests/test_msvd_protocol.py`
- Create: `experiments/contextual-risk-v2/datasets/msvd-split-manifest.json`
- Create: `run_train_msvd_contextual_risk_bg.sh`
- Create: `tests/test_msvd_contextual_risk_launcher.py`
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage5-msvd.md`

**Interfaces:**
- Produces immutable MSVD train/validation/test hashes, artifact-aware A1/A3 launcher, and three-seed MSVD validation comparison.
- Consumes frozen Stage 3 configuration without test-specific tuning.

- [ ] **Step 1: Write failing split and launcher tests**

Assert disjoint train/val/test video IDs, caption counts, file hashes, no test loader during training/selection, canonical artifact paths, and exact A1/A3 freeze boundaries.

- [ ] **Step 2: Implement and commit the MSVD protocol before runs**

Hash `train_list.txt`, `val_list.txt`, `test_list.txt`, `raw-captions.pkl`, and video inventory. Validate disjoint IDs and expected counts. The launcher reads committed reservations and writes only to run directories.

- [ ] **Step 3: Qualify or train canonical MSVD A1 on validation only**

If no migrated canonical A1 satisfies score-risk gates and frozen-retriever audits, reserve a three-seed A1 phase first. Select its configuration on validation and freeze its checkpoint lineage before A3.

- [ ] **Step 4: Run frozen A3 seeds 0/1/2**

Do not alter feature set, loss weights, or K. Compute paired validation intervals against MSVD A1 and require exact raw retrieval equality.

- [ ] **Step 5: Apply gate and commit**

Continue only if both directional point estimates improve and either aggregate AUROC or AURC paired interval passes. Commit protocol, terminal registry records, and report.

---

### Task 9: Run Stage 6 calibration, robustness, and efficiency analyses

**Files:**
- Create: `scripts/evaluate_reliability_analysis.py`
- Create: `tests/test_reliability_analysis.py`
- Create after execution: `experiments/contextual-risk-v2/summaries/stage6-analysis.md`
- Create after execution: `experiments/contextual-risk-v2/summaries/figures/`

**Interfaces:**
- Produces ECE, Brier, NLL, reliability diagrams, full risk-coverage curves, gallery-size/hard-negative metrics, latency, memory, and parameter count.
- Consumes only frozen validation arrays and models; no architecture changes result from this stage.

- [ ] **Step 1: Write failing analysis tests**

Test fixed calibration bins, validation-fitted temperature scaling, deterministic gallery subsets, hard-negative construction, timing warmup exclusion, and plot data hashes.

- [ ] **Step 2: Implement calibration and risk-coverage analysis**

Fit calibration parameters on validation only. Report pre/post calibration metrics but preserve raw A3 scores as primary ranking evidence.

- [ ] **Step 3: Implement robustness protocols**

Use gallery sizes `100, 500, 1000, full` with deterministic nested subsets and a semantic hard-negative subset built without failure labels. Measure whether risk rises when stronger candidates are introduced.

- [ ] **Step 4: Implement efficiency measurement**

Report 100 warmup iterations and 500 timed iterations for the sidecar alone, full-matrix feature construction, peak CUDA memory, parameter count, and artifact storage size.

- [ ] **Step 5: Run frozen analyses and commit**

Run on MSRVTT and MSVD validation artifacts, generate SVG/PDF plot data plus Markdown report, verify hashes, and commit only summaries/figures suitable for the paper.

---

### Task 10: Authorize sealed final tests and generate paper tables

**Files:**
- Modify after execution: `experiments/contextual-risk-v2/registry.jsonl`
- Modify after execution: `experiments/contextual-risk-v2/summaries/final_test.csv`
- Modify after execution: `experiments/contextual-risk-v2/summaries/paper_tables.md`
- Create after execution: `docs/superpowers/reports/2026-08-14-reliable-retrieval-final-results.md`

**Interfaces:**
- Consumes passed Stages 4 and 5, frozen configuration SHA, declared final seed runs, and Stage 7 committed authorization.
- Produces the only confirmatory MSRVTT/MSVD test metrics for the new campaign.

- [ ] **Step 1: Freeze and commit final authorization**

The authorization lists dataset, seed, selected validation checkpoint hash, frozen configuration hash, calibration hash, raw/DSL evaluation settings, expected metric files, and `test_authorized=true`. Verify no earlier phase contains test metrics.

- [ ] **Step 2: Execute each declared final run exactly once**

Run raw retrieval and fixed validation-selected DSL settings, then A1/A3 risk evaluation. A terminal run is immutable and may never retry in place. After a runtime infrastructure failure, preserve and finalize the failed run first; only explicit human authorization may create a new unique successor run with the identical scientific configuration and recorded `parent_run_ids` plus `selection_lineage`. Scientific underperformance cannot trigger another run.

- [ ] **Step 3: Compute paper comparisons without selection**

Generate retrieval R@1/5/10, AUROC with paired CI, AURC, rho, coverage curves, calibration, robustness, efficiency, ablations, and qualitative case IDs. Attribute DSL ranking gains separately from A3 reliability gains.

- [ ] **Step 4: Verify complete lineage**

For every paper cell, verify run ID, committed reservation, manifest hash, checkpoint hash, metric path/hash, array references, split hash, seed, and terminal registry record. Fail report generation on an unresolved cell.

- [ ] **Step 5: Run full verification and commit final evidence**

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests
git diff --check
git add experiments/contextual-risk-v2/registry.jsonl \
  experiments/contextual-risk-v2/summaries/final_test.csv \
  experiments/contextual-risk-v2/summaries/paper_tables.md \
  docs/superpowers/reports/2026-08-14-reliable-retrieval-final-results.md
git commit -m "docs: report reliable retrieval experiment campaign"
```

Expected: all project tests pass, every launched run is terminal, no new legacy output exists, and the final report includes stopped stages and protocol deviations.
