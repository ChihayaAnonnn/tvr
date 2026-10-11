# Parallel supply and durable supervision (2026-10-11)

User authorized completing training optimization after the read-only performance audit. Purpose: overlap unchanged CPU decoding with GPU training, preserve full-caption batches/inputs/RNG/FP32/selection, and keep campaign supervision alive independently of desktop tool-session lifetime. No new scientific method, batch tuning, test unsealing or gate relaxation.

## Contract and implementation

`batch_loader(dataset, batches, workers, pin_memory)` yields ordered batches from the supplied sampler. Formal runs seal four spawned workers per rank, prefetch two batches per worker, one Torch/OpenCV CPU thread per worker, and90s worker timeout. Zero workers is retained for CPU fixtures/reference verification. Decode/hash errors propagate to the existing rank-wide error barrier. Context exit closes workers even after decode failure or early termination; pinned PyTorch2.1 requires its private iterator shutdown API because no public close API exists. No cache, frame decoder or model definition changed.

The manifest, study and typed runtime bind worker count/backend/prefetch. Scientific checkpoint/source compatibility guards are unchanged: optimized source cannot resume an old executable identity. Existing checkpoint remains preserved; complete baseline restarts from the official pretrained initialization after fresh two-rank Q0. Q0 exercises the spawned/pinned synthetic supply at real capacity before the existing forward/backward/exact-resume checks. Old Q0 remains archived; the current science gate is closed until new Q0 passes.

Every100 updates logs mean main-thread data wait, update duration including H2D/compute/loss synchronization/communication, and global samples/s. These wall-time counters do not separately measure CUDA kernels or NCCL and must not be described as kernel timings.

## Verified orphan shutdown

Before tracked edits, original run verified with `manage_experiment_run.py verify-run`. Launch PID3662871/start ticks matched stored launch identity; expected controller PID3655709 was absent. Epoch1 checkpoint/sidecar matched their published hashes. Only the verified owned process group was stopped; group quiescence was checked before terminal inventory. Original UATVR run sealed failed and unlaunched CLIP4Clip cancelled in73178fe. Recovery receipt explicitly states controller-exit cause unknown and observed launcher exit unavailable. Budget is conservatively reconstructed from kernel process start tick through proven group stop:36130 seconds x2 =20.072222GPUh. This is a recovery charge, not a fabricated controller exit receipt. Original files/checkpoint and negative/failure history remain preserved.

## Supervision

Use an independent tmux server/session for the Native campaign, with the approved Python environment and clean linked worktree. Keep completed pane visible (`remain-on-exit`) so a controller exception remains inspectable. Run-local worker logs and archive receipts remain authoritative; tmux output is orchestration state, not a substitute for archived outputs. Do not use a foreground desktop tool session as the lifetime owner of an hours-long campaign. Process/lock/source bindings, existing time caps, writer cleanup and unique-run reservation remain enforced by the unchanged controller.

## Validation

Observed RED for missing batch_loader, then12 targeted loader/runtime/data tests passed in23.24s. New authorization rejection first failed for loader mismatch, then12 authorization/data integration tests passed in8.53s. Real video spawned-worker tests cover ordered pixels/masks, unchanged parent RNG, deterministic replay, corrupt-video error propagation and worker cleanup. Actual tiny CLIP4Clip training compares full loss arrays and final model tensors bit-for-bit between serial and spawned loading. Full affected regression and independent review follow before committing/launching.

CPU audit32-video serial/four-thread probe showed8.541s/2.322s (3.68x); this was a descriptive CPU-only thread probe, not the production spawned DataLoader or an end-to-end speedup claim. Actual optimized training throughput is judged from100-update logs and comparison of protocol-matched wall time, never from allocated memory or SM utilization alone.

Final prelaunch validation:250 affected tests passed in189.57s, one upstream TypedStorage deprecation warning. Independent review found no Critical/Important/Minor issues; independently8 prefetch/data integration tests passed in28.82s. GPU transfer/prefetch capacity and sustained throughput remain subject to new Q0 and actual training; no speedup inferred from CPU equality tests.

Actual regression command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -c 'from semantic_retrieval.runtime.distributed import configure_fp32; configure_fp32(); import pytest; raise SystemExit(pytest.main(["-q", "tests/semantic_retrieval/full_data", "tests/semantic_retrieval/test_runtime_artifacts.py", "tests/semantic_retrieval/evidence/test_authorization.py", "tests/semantic_retrieval/evidence/test_source_preparation.py", "tests/test_manage_experiment_run.py", "tests/test_experiment_artifacts.py", "tests/semantic_retrieval/test_pilot_policy.py", "tests/semantic_retrieval/test_video_and_splits.py", "tests/semantic_retrieval/test_no_legacy_dependencies.py"]))'`.


## Measured production supply tuning

Four-worker Q0 completed192.934s (e33853d), all9 checks passed. Formal restored model/optimizer hashes exactly match preceding Q0 for both methods. First scientific100-update window: data_wait2.1499s/update, update1.5511s, global17.29samples/s, loss2.656216 (same observed first100 loss as original serial run). Data wait remained58.1% of this measured window. Existing free CPU/memory permit16 total CPU loaders, so final policy increases4->8 workers per rank; prefetch factor remains2. This is an operational performance decision from wall-time counters, not validation-metric tuning; science recipe/batch/seed remain unchanged.

The live controller was identity-verified and sent SIGTERM; its normal owned-group cleanup and terminal phase commit completed3acbb13 before any tracked edit. Four-worker outputs remain archived, unlaunched paired arm cancelled. Eight-worker run uses new source/Q0/fresh run identities; strict old-source resume remains unchanged. No further worker-count sweep is planned in this optimization task.
