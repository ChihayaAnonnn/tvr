# Native full-data campaign

Use the linked worktree `/home/xujie/.codex/worktrees/ad3a/UATVR` and existing `/home/xujie/.conda/envs/tvr/bin/python`. Read study.yaml and registry.jsonl first. The environment contract is environment.yaml (canonical parsed hash78c6fc62); no environment rebuild is needed. All source must be committed and tracked-clean before launch.

Independent documentation/CPU validation (safe without free GPUs):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data/test_runtime.py tests/semantic_retrieval/full_data/test_controller.py tests/semantic_retrieval/full_data/test_worker.py
```

Authorized Native campaign invocation, once independent review is clean:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/xujie/.conda/envs/tvr/bin/python scripts/run_full_data_campaign.py
```

It waits up to6h for four cards with no compute process, at most256MiB used and zero utilization. It never stops external jobs. It commits phase definitions and whole reserved batches, runs exact sealed commands under a locked GPU-count-aware controller, closes all owned writers before terminal hashes, finalizes and commits terminal batches. Each selected four-card phase is serial within the campaign. A changed waiting source, pre-existing unclosed phase, software failure, unproven identity or insufficient conservative budget stops execution. Do not blindly rerun after failure.

Q0 uses only authenticated formal-capacity random CLIP weights and both true model adapters. All nine checks must pass on four CUDA ranks before the campaign records qualification and opens the already defined baseline gate. Scientific phases use official pretrained CLIP, all8246 training videos/all164920 captions, all500 validation videos/all10000 captions; five epochs, raw bidirectional metrics, earliest best validation raw T2V R@1 checkpoint. Seeds713/714/715 are frozen; seed713 pair runs first. No mixed train/test annotation is loaded by training.

The run-local `provenance/controller-result.json` records charged wall time/card count. Scores, ranks, loss traces and large checkpoint tensors are external NPY references with hashes. Per-epoch provenance binds metrics/checkpoint/sidecar. Exact epoch-boundary resume requires a new phase/run with explicit failed-parent checkpoint+sidecar refs, identical recipe/input identity and proven executable compatibility. The successor carries authenticated historical states into its own data plane; no predecessor output is changed. At most two diagnosed successors follow a launched failure. Executable source fixes require fresh qualification before subsequent science.

Results: per-run `metrics/selected_validation.json`, `provenance/baseline-result.json`; phase summaries in `summaries/`; aggregate `reports/full-baselines.json` only after all frozen phases complete. Paper test1k scores are external context, not directly comparable with source-isolated train8246/val500. Test remains sealed. Baseline establishment alone does not support the new uncertainty/evidence hypothesis; prior failed pilot stays closed.
