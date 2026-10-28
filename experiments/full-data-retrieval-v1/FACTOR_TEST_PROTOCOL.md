# Completed-factor JSFUSION test authorization — 2026-10-21

User explicitly requests testing the completed sampling and batch contrasts on 1k test against original baselines. This amendment authorizes exactly the two completed UATVR seed713 treatments frozen in study.yaml final_test_factors. temporal_offset uses earliest-best validation epoch5; caption_shuffle uses earliest-best validation epoch3. Neither failed CLIP4Clip temporal_offset nor unstarted caption_shuffle is eligible. The earlier fixed-epoch5 validation endpoint remains separately reported; this test comparison follows the existing validation-selected final-test policy, not fixed-epoch5 selection.

Compare against the already archived original UATVR seed713 test, not the three-seed average. No baseline re-evaluation, test-based checkpoint selection, tuning, training, or new seed is authorized by this amendment. Historical records and old six-checkpoint permission stay unchanged. Future arms and other studies remain sealed. This is descriptive single-seed follow-up after previous test exposure, not independent confirmation.

Reuse the existing strict inference-only evaluator: full JSFUSION1000x1000, 12 frames, 32 text tokens, ViT-B/32, FP32, DSL=false, both T2V and V2T R@1. All inputs and selected parent/result/checkpoint identities remain bound. Each factor variant can complete only once. Reserve both final runs together after a fresh matching-executable two-rank production Q0 with factor paths; commit reservations, run without tracked edits, finalize and commit terminal inventories. Existing per-run/cumulative software and scientific budgets, sharing/headroom and failure limits apply unchanged. No new software environment or dependencies.

Implementation validation: added explicit permission-scoping and factor duplicate-rejection behavioral tests; first observed 2 failures, then 5 dedicated tests passed. Existing baseline duplicate guard is preserved. Full affected regression and real GPU Q0 precede final evaluation.

Final prelaunch verification (2026-10-22): 243 affected tests passed in 475.61s,
one upstream PyTorch TypedStorage deprecation warning. Command:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -c 'from semantic_retrieval.runtime.distributed import configure_fp32; configure_fp32(); import pytest; raise SystemExit(pytest.main(["-q", "tests/semantic_retrieval/full_data", "tests/semantic_retrieval/test_runtime_artifacts.py", "tests/test_manage_experiment_run.py", "tests/test_experiment_artifacts.py", "tests/semantic_retrieval/test_pilot_policy.py"]))'`.
Independent read-only review found no correctness blockers. Original and factor-control
UATVR seed713 selected model SHA-256 both equal
`f12bd3aa6dc18e03badc85ba48defd786fe5667c33464aa597188e3a12e982a4`.
Archived original test R@1 is T2V42.9/V2T42.2. Prelaunch authenticated cumulative
usage is software0.612415GPUh/scientific128.834041GPUh; Q0 plus two worst-case
24h two-GPU final reservations fit existing 8/512GPUh caps.
