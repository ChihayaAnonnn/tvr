# Full-data implementation evidence

Task 1 implementation: complete allowlist-first caption/source compilation, transitive source quarantine, complete bidirectional retrieval metrics with canonical tie breaking, and CPU preparation authorization checked at reservation and worker start. Full exports decode every eligible video, check container vs decoded frame count and authenticate every source video. Sharded metadata keeps JSON below 1 MiB. No pilot filtering, raw test decoding or legacy launcher reuse.

Validation (2026-10-10):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -c 'from semantic_retrieval.runtime.distributed import configure_fp32; configure_fp32(); import pytest; raise SystemExit(pytest.main(["-q", "tests/semantic_retrieval/full_data", "tests/semantic_retrieval/test_runtime_artifacts.py", "tests/semantic_retrieval/evidence/test_authorization.py", "tests/semantic_retrieval/evidence/test_source_preparation.py", "tests/test_manage_experiment_run.py", "tests/test_experiment_artifacts.py", "tests/semantic_retrieval/test_pilot_policy.py"]))'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/test_no_legacy_dependencies.py
```

Results: 197 passed in 90.09 seconds; 12 passed in 2.93 seconds. New behavior tests first failed on absent modules (4 compiler/metric, 4 authorization, 1 actual video export), then passed after implementation. Initial regression invocation omitted the CPU/FP32 environment contract and failed existing guards; rerunning with CUDA hidden and TF32 disabled passed without weakening guards or changing those tests.

P0 is an independent CPU certification phase. GPU science remains unauthorized pending a new complete production-capacity four-CUDA-rank qualification. At the pre-launch device observation, GPUs 1/2/4 were occupied by external jobs; only 0/3 were idle. No external process was modified.

Task 2: pinned official UATVR namespace with original/adapted file hashes; trainable CLIP4Clip meanP; exact official bottom freezing and BertAdam schedule; complete unique-video caption sampler; source-certified dataset reader with actual1FPS decoding; strict external-array checkpoint and optimizer/RNG recovery. Both actual adapters pass CPU backward; two Gloo ranks match serial global gradients for each method. Evaluation preserves RNG despite upstream unused inference Gaussian draws. Full recipe and differences are in `BASELINE_RECIPE.md`.

Validation: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data tests/semantic_retrieval/test_video_and_splits.py --tb=short` →28 passed in10.77s. Added optimizer nonfinite/shape/order and inference-RNG behavior tests were observed failing, then passing after fixes. TypedStorage deprecation is an upstream PyTorch warning; no numerical failure was hidden.

P0 `verify-run` passed in a clean detached worktree at reservation commit `f614e56` after archival. Running its launch verifier from the later terminal-summary commit correctly refuses changed tracked control files; the verifier was not weakened to accept a later source tree. P0 outputs and history were unchanged.
