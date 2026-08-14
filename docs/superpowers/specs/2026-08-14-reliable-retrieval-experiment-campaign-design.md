# Reliable Retrieval Experiment Campaign and Artifact Archive Design

## 1. Objective

Build and execute a staged, validation-only experiment campaign supporting the paper's dual claim:

1. the UATVR/WTI/DSL retrieval path preserves competitive text-video retrieval quality; and
2. the detached A3 contextual sidecar improves failure detection and selective retrieval over intrinsic-uncertainty and score-only baselines.

The campaign must stop weak variants early, keep MSRVTT/MSVD test sets sealed until final configurations are frozen, and archive every run as a self-contained, auditable unit.

## 2. Scope and sequencing

The work is split into two sequential components:

1. **Artifact control plane:** implement the run-centric archive, registry, schemas, launch reservation, metric/array separation, documentation, and legacy inventory.
2. **Scientific campaign:** execute validation baselines and pilot ablations, apply stop gates, expand successful variants to multi-seed and MSVD, then run frozen final tests and prepare paper tables.

No new scientific run may launch until the artifact control plane passes its tests and documentation gate.

## 3. Repository layout

### 3.1 Tracked control plane

```text
experiments/
└── contextual-risk-v2/
    ├── README.md
    ├── study.yaml
    ├── registry.jsonl
    ├── schemas/
    │   ├── manifest.schema.json
    │   ├── status.schema.json
    │   ├── metrics.schema.json
    │   └── registry.schema.json
    └── summaries/
        ├── validation.csv
        ├── final_test.csv
        └── paper_tables.md
```

Tracked files contain configuration, small scalar metrics, hashes, state, and paper summaries. They must not contain model tensors, optimizer states, per-query arrays, raw similarity matrices, or full environment dumps with secrets.

`registry.jsonl` is append-only. Corrections are new records referencing the superseded record; existing lines are never edited or removed.

### 3.2 Ignored artifact data plane

```text
artifacts/
└── contextual-risk-v2/
    └── <dataset>/
        └── <stage>/
            └── <run_id>/
                ├── manifest.json
                ├── status.json
                ├── command.sh
                ├── logs/
                │   ├── train.log
                │   ├── validation.log
                │   └── test.log
                ├── checkpoints/
                ├── metrics/
                │   ├── val_epoch_000.json
                │   ├── selected_validation.json
                │   └── final_test.json
                ├── arrays/
                ├── plots/
                └── provenance/
                    ├── hashes.json
                    ├── environment.txt
                    └── source.patch
```

`artifacts/` is ignored by Git. Each run directory is immutable after its state reaches `completed` or `failed`, except for an append-only administrative audit log.

## 4. Run identity and lifecycle

The run ID is deterministic in shape and unique in value:

```text
<dataset>-<stage>-<variant>-seed<NNN>-<UTC timestamp>-<git sha12>
```

Allowed stages are:

- `baseline`
- `feature_ablation`
- `loss_ablation`
- `topk_sweep`
- `multiseed`
- `calibration`
- `robustness`
- `efficiency`
- `final`

Lifecycle states are:

```text
reserved -> running -> validating -> completed
                              \-> failed
reserved/running -------------> cancelled
```

Transitions use atomic write-and-rename. A launcher must refuse an existing run ID or terminal run directory. Every status record includes timestamp, worker PID, host, exit code when terminal, and SHA-256 of the manifest.

## 5. Manifest and registry contract

Every manifest records at least:

- run ID, study name, dataset, stage, variant, and seed;
- absolute artifact path and repository-relative tracked registry path;
- git commit, tracked-clean state, and optional source patch hash;
- Python, PyTorch, CUDA, GPU model/count, precision, and distributed topology;
- input checkpoint paths and SHA-256 values;
- dataset split manifests and SHA-256 values;
- complete model, loss, optimizer, scheduler, and evaluation configuration;
- parent run IDs and selection lineage;
- whether validation or test access is authorized;
- expected scalar outputs and large-array file references.

Each registry record contains the immutable manifest digest, artifact path, lifecycle state, parent lineage, selected-validation tuple, test-access status, and small summary metrics. Registry records never embed arrays.

## 6. Metric and array storage

JSON is limited to configuration, scalar metrics, short lists, paths, shapes, dtypes, and hashes. Any numeric payload larger than 1 MiB is stored in `arrays/` as `.npy` or compressed `.npz` and referenced by:

```json
{
  "path": "arrays/t2v_failure_risk.npz",
  "key": "failure_risk",
  "shape": [1000],
  "dtype": "float32",
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

Raw WTI matrices, embeddings, risk arrays, bootstrap samples, and per-query ranks must never be duplicated into candidate, winner, and final-test JSON files. Shared arrays are content-addressed or referenced once from the run directory.

## 7. Documentation and agent guardrails

The artifact implementation must create or update all of the following:

- root `README.md`: human quick start, archive layout, how to locate and compare runs;
- root `AGENTS.md`: mandatory agent rules and prelaunch checklist;
- `docs/agent/README.md`: link to the experiment archive guide;
- `docs/agent/experiment-artifacts.md`: detailed agent workflow, forbidden legacy paths, recovery procedures, and examples;
- `experiments/contextual-risk-v2/README.md`: study-specific stages, gates, and commands;
- `.gitignore`: ignore `artifacts/` while keeping `experiments/` tracked.

`AGENTS.md` must require agents to:

1. read the study file and registry before planning or launching a run;
2. reserve a unique run through the artifact API;
3. write new experimental outputs only beneath the run directory;
4. never create new scientific outputs directly in root `logs/` or `ckpts/`;
5. keep arrays out of JSON and verify every referenced hash;
6. use validation only until a study gate explicitly authorizes test access;
7. append a terminal registry record and generate a concise tracked summary;
8. preserve failed-run evidence and never silently retry with changed settings.

Automated tests must parse these documents and assert that the mandatory rules and canonical paths remain present, preventing future agents from accidentally reverting to the legacy layout.

## 8. Legacy artifact policy

Existing `logs/` and `ckpts/` are read-only legacy roots after the new archive is activated.

Migration proceeds without deleting or overwriting originals:

1. inventory every legacy regular file with path, size, modification time, and SHA-256; hashing is resumable and records progress so the 88 GiB scan can restart safely;
2. classify runs as `critical`, `historical`, `failed`, or `unresolved`;
3. migrate only canonical A1, registered A2, selected A3, and A3 DSL T=2 artifacts into run-centric directories;
4. verify byte-identical hashes before registering migrated copies;
5. keep all other 88 GiB in place and expose them through the legacy inventory;
6. do not delete legacy artifacts during this campaign.

The migration tool must be resumable and default to copy-plus-verify. Moving or deleting an original requires a separate explicit user-authorized cleanup plan.

## 9. Scientific stages and stop gates

### Stage 0: Archive qualification

Gate: schemas, lifecycle tests, launcher integration, documentation tests, and a CPU fixture run all pass. No GPU science run before this gate.

### Stage 1: Validation-only non-learning baselines

Evaluate raw score, Top-1/Top-2 margin, entropy, energy, score variance, SAP signals, A2 logvar, A1, and the existing A3 checkpoint on the same frozen MSRVTT validation evidence.

Gate: continue if A3 beats the strongest baseline by at least `+0.01` mean directional AUROC or `-0.01` mean directional AURC, while each direction retains `AUROC > 0.5` and `rho >= 0.20`.

### Stage 2: Pilot feature ablations

Run seed 0 for score-only, semantics-only, no-context, no-score-geometry, and full A3. All selection uses validation.

Gate: relative to both score-only and no-context, full A3 must improve mean AUROC by at least `0.005` or reduce mean AURC by at least `0.005`; neither direction may lose more than `0.01` AUROC, and only the declared sidecar tensors may change.

### Stage 3: Pilot loss and Top-K ablations

Run BCE-only, BCE+fit, BCE+rank, full loss, and `K in {1, 4, 8, 16, 32}` on seed 0.

Gate: among candidates with `min directional rho >= 0.20`, retain every configuration within `0.005` mean AUROC and `0.005` mean AURC of the best candidate, then freeze the one with the smallest K, fewest active loss terms, and earliest epoch in that order. Do not inspect test.

### Stage 4: Multi-seed confirmation

Train the frozen configuration on seeds 0, 1, and 2. Compute mean, standard deviation, and paired bootstrap confidence intervals against A1.

Gate: proceed if the point estimate improves both directions and the paired 95% bootstrap confidence interval is strictly above zero for mean AUROC delta or strictly below zero for mean AURC delta. Bootstrap count is 10,000 with seed 0 and is frozen in `study.yaml` before launch.

### Stage 5: MSVD confirmation

Port the frozen architecture and loss without MSVD-specific test tuning. Use MSVD validation for checkpoint selection, then hold test until all seeds finish.

Gate: A3 must improve both directional point estimates and achieve either a strictly positive paired 95% bootstrap interval for mean AUROC delta or a strictly negative interval for mean AURC delta. Raw retrieval ranks and R@K must match the frozen MSVD retriever exactly.

### Stage 6: Calibration, robustness, and efficiency

Produce ECE, Brier score, NLL, reliability diagrams, full risk-coverage curves, gallery-size tests, hard-negative tests, parameter count, memory, and latency. Calibration parameters are fitted on validation only.

Gate: these analyses may qualify or limit the claim but do not trigger architecture retuning.

### Stage 7: Frozen final tests and paper tables

After all architecture, loss, K, seeds, and calibration choices are frozen, authorize test access once per declared final run. Generate raw/DSL retrieval tables, risk tables, ablations, statistical comparisons, efficiency tables, and qualitative cases from registered artifacts.

## 10. Test hygiene

- Existing MSRVTT A3 and DSL results are historical exploratory evidence only.
- No new ablation, seed choice, threshold, K, calibration, or stopping decision may use MSRVTT or MSVD test metrics.
- Test access requires a registry authorization record identifying the frozen configuration and expected output files.
- A failed test execution caused by infrastructure may be retried only with identical configuration and an appended failure record.
- Scientific underperformance never authorizes a test rerun or retuning.

## 11. Completion criteria

The campaign is complete when:

- all launched runs are registered and terminal;
- every paper number resolves to a run ID, manifest, metric file, and hash;
- required MSRVTT and MSVD stages either pass or are reported as stopped by their predeclared gate;
- no new outputs were written to legacy `logs/` or `ckpts/`;
- the final paper summaries distinguish retrieval improvements due to DSL from reliability improvements due to A3;
- protocol deviations and stopped stages are reported rather than hidden.
