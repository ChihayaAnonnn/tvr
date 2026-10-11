# Independent final software review

Reviewed source6370ce21937a43b9309d78a1ea1a49ba8b363fe5, fresh independent agent `full_data_final_review`, read-only. Native implementation remained sequential. No review GPU job, source/index/HEAD mutation or extra reviewer delegation occurred.

Confirmed strengths: vendor original/adapted hashes match; initialization/freezing/optimizer/meanP follow disclosed recipe; complete-caption sampling, full-gallery metrics/ties, input authentication, sealed-test gates, executable-source qualification guards, count-aware quota and self-contained checkpoint recovery are present. No Critical findings or confirmed Minor findings.

Two Important findings were corrected in one pass:

1. SIGTERM/SIGHUP previously bypassed controller cleanup, leaving the separate owned worker group alive. Scoped handlers now enter the existing exception cleanup and restore previous handlers afterwards. A real protected CPU controller SIGTERM test first failed because its worker survived, then passed: worker stopped, non-success terminal state/receipt archived and final log hash matched. A real two-rank CPU torchrun test verifies launcher/ranks share the owned group in the pinned PyTorch2.1 environment, and all writers stop. SIGKILL remains necessarily fail-closed manual integrity recovery; no history is overwritten.
2. Torchrun previously defaulted elastic metadata to system temporary directories. The sealed command now binds `--log-dir {artifact_run_dir}/logs/elastic` before the worker script. The command assertion first failed, then passed. The actual two-rank launcher creates its attempt directory under the specified run-local logs root.

Correction validation:11 controller/campaign/worker tests passed in26.43s. The independent reviewer executed the RUNBOOK CPU command verbatim:10 passed in19.66s, no documentation/environment divergence; model/data tests20 passed in9.88s; all provenance hashes matched. An upstream TypedStorage deprecation was the only warning.

Final post-correction regression: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data tests/semantic_retrieval/test_no_legacy_dependencies.py` →53 passed in37.82s, one upstream deprecation warning, no failures.

Declined to judge, with primary rulings: actual four-CUDA-rank numerical reproducibility/memory/throughput require new Q0; scientific schedule completion and retrieval quality require authenticated scientific runs. Both remain explicit outstanding gates, never inferred from CPU tests. Prior negative pilot remains closed; test remains sealed. Four-card resource shortage is external, not evidence for or against the scientific hypothesis.


## Approved two-card amendment review (2026-10-11)

Independent read-only reviewer `two_card_final_review`: no Critical, Important or Minor findings in the bounded amendment. Verified two-rank launch, fixed global batches64/128 with one update per batch, sealed topology/memory identity, mapped-device60GiB allocator cap, >=68GiB prelaunch free-memory rule, dynamic GPU budgeting and resume identity. Science requires authenticated matching-topology Q0; other studies/test remain unchanged. Independent CPU tests18 passed in8.42s; diff whitespace check passed. Live GPU memory/overhead/contention/throughput were explicitly not judged.

Actual affected regression: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -c 'from semantic_retrieval.runtime.distributed import configure_fp32; configure_fp32(); import pytest; raise SystemExit(pytest.main(["-q", "tests/semantic_retrieval/full_data", "tests/semantic_retrieval/test_runtime_artifacts.py", "tests/semantic_retrieval/evidence/test_authorization.py", "tests/semantic_retrieval/evidence/test_source_preparation.py", "tests/test_manage_experiment_run.py", "tests/test_experiment_artifacts.py", "tests/semantic_retrieval/test_pilot_policy.py", "tests/semantic_retrieval/test_video_and_splits.py", "tests/semantic_retrieval/test_no_legacy_dependencies.py"]))'` ->248 passed in161.38s; only upstream TypedStorage deprecation warning.


## Live NCCL discovery and correction

First production two-rank Q0 `synthetic-baseline-full-data-q0-seed713-20261010T174621Z-3bc4abe03b8c` failed after real UATVR backward at evaluation metadata gathering: NCCL requires contiguous buffers, while encoder outputs can be transposed views. CPU Gloo accepted these views, hiding the backend-specific contract. Terminal/archive commit a56a6cb retains all artifacts. Controller elapsed83.415507s, charged0.04634195GPUh; science stayed closed.

Shared `gather_metadata` now detaches, validates equal shape/dtype/device, then makes contiguous send/receive buffers. Caller tensor/gradient semantics unchanged. Delegated real two-Gloo-rank transpose test first failed on the contiguous-input assertion; after correction it checks rank-order values, detached metadata and differentiable gradients. Independent reviewer found no Critical/Important/Minor issues, independently3 tests passed in5.11s.

Affected regression command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -c 'from semantic_retrieval.runtime.distributed import configure_fp32; configure_fp32(); import pytest; raise SystemExit(pytest.main(["-q", "tests/semantic_retrieval/test_distributed.py", "tests/semantic_retrieval/full_data", "tests/semantic_retrieval/test_trainer.py", "tests/semantic_retrieval/test_training_resume.py", "tests/semantic_retrieval/test_gpu_qualification.py", "tests/semantic_retrieval/test_dual_gpu_preflight.py", "tests/semantic_retrieval/test_pilot_qualification.py"]))'` ->91 passed in134.95s, one upstream TypedStorage warning. Renewed live NCCL Q0 is still required and will use a new identity; failed history is not overwritten.


## Parallel supply final review (2026-10-11)

Independent reviewer `two_card_final_review` found no Critical/Important/Minor issues. Verified ordered batches, parent RNG preservation, deterministic decoding, bounded spawn prefetch, explicit worker cleanup, authenticated loader policy, reclosed science gate and strict old-source checkpoint rejection. Independently8 tests passed in28.82s, including exact serial/parallel training losses/model tensors and corrupt-video cleanup. Actual CUDA capacity/transfer/throughput remain pending live verification. Main affected regression250 passed in189.57s; details/actual command in TRAINING_OPTIMIZATION.md.

2026-10-11 final supply tuning: first four-worker100-update window measured17.29samples/s, data wait2.1499s/update and update1.5511s. Final bounded policy is8 spawned workers/rank, prefetch2; normal controller shutdown archived the four-worker attempt before edits. Final full-data regression46 passed in103.07s; independent review clear with13 tests passed in49.01s. Science reclosed for fresh eight-worker two-rank Q0; strict resume compatibility, batch/optimizer/data/test protocol unchanged. See TRAINING_OPTIMIZATION.md for accounting and limits.
