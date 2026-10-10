# Evidence-conditioned retrieval: independent review and fix pass

Reviewed range: `a9a2f76..df5850b`. Reviewer: independent `final_evidence_review` (GPT-6-Astra, xhigh), final code review authorized by the user. Verdict before repairs: not ready; 0 Critical, 11 Important, 1 Minor. This document records the author's single repair pass, not a second independent approval.

| Finding | Resolution and validation |
|---|---|
| I1 multi-rank scientific output races | Training index/cache stores are rank-local; transitions/calibration/evaluation have one publisher and broadcast results/errors. Real Gloo publisher test checks both success and failure. |
| I2 parent semantic compatibility | Check committed terminal producer stage/configuration, CLIP, views, split/index identities, variant/floor, and value/calibration parent chain; bind sidecar config/source/manifest to producer. Shape-compatible mismatch regression. |
| I3 successor action recovery disconnected | Immutable sharded recovery snapshots bind config/source, initial observations, all event files and numerical references. Import validates all hashes before copying to a unique successor, preserves cost states and fails closed on unprovable partial work. Real video module integration and execute-before-commit recovery tests. |
| I4 inference loader rejects a four-rank parent | Separate validated weight-only loader does not restore RNG/optimizer or require consumer world size; exact resume retains world-size checks. Real four-Gloo archived CLI checkpoint loaded by one process. |
| I5 initial sparse index aborts on partial decode | Index consumes only successful initial observations, preserves all attempted initial frames, rejects an empty candidate. Real video test with declared two frames and one decodable frame. |
| I6 semantic label format cannot carry lineage | Explicit reviewed-semantic episode schema v3 carries query source/component metadata while preserving unknown labels; v1/v2 compatibility retained. |
| I7 large transition JSON | Versioned v2 root and bounded metadata shards, each hash-verified; v1 reader retained. 3,000-record round-trip and corrupted-shard rejection. |
| I8 insufficient scientific evaluation evidence | External per-episode costs/labels/times, scoped scripted/selected action NLL/Brier/ECE and unknown coverage, optional bound rich-score NLL/95% intervals, paired component bootstrap. Unavailable diagnostic scopes have explicit undefined reasons. Calibration records before/after proper scores. |
| I9 online latency omits initial acquisition and CUDA completion | Start online timer before initial decode/CLIP; synchronize measured CUDA operations; report decode, CLIP, temporal, MC, pair/scale/value and query-cache timings separately. Shared index/offline diagnostics are separate. Successor timings explicitly cover the current worker segment, not unobserved predecessor latency. |
| I10 inactive loss configuration accepted | Stage-specific objective allowlist rejects zero-active or unused weighted objectives before launch. |
| I11 value capacity not qualified | Qualify distribution/geometry/mean heads with real DDP gradients and equal-rank parameters; scientific authorization binds hidden dimension/feature schema and tested head kind. |
| M12 fixed_frames allocates greedily to first ID | Regraded Important because it affects a mandatory resource comparison; fixed view levels advance candidates in rounds ordered by attempted count and canonical ID. |

No deferred minor remains. Review declined to establish actual four-CUDA qualification, real source/near-duplicate validity, or scientific superiority. These require separate empirical or source evidence; they are not granted by review repairs.

Validation uses `/home/xujie/.conda/envs/tvr/bin/python` with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1`, `-m pytest -p no:cacheprovider`, synthetic CPU fixtures and temporary archives only. Targeted RED failures were observed before implementing each new interface. The real-video evaluation test invokes the implementation module on synthetic inputs; it does not grant or forge scientific CLI authorization.

Targeted commands: evidence `test_contracts.py test_data.py` plus shared episode tests (25 passed); `test_end_to_end.py::test_protected_four_gloo_ranks_train_and_restore test_checkpoint.py` (3 passed before later additions); `test_baselines.py` (8); `test_transitions.py` (4); `test_distributed.py` (4); `test_authorization.py test_end_to_end.py` (9); `test_journal.py test_contracts.py test_runtime.py` (21 before new integration test); `test_evaluation.py` (4); `test_training.py test_matcher.py` (11); `test_runtime.py::test_real_video_evaluation_partial_initial_decode_and_successor` (1). Only observed passing results are listed.

Broader regression and actual GPU results will be appended after completion. Scientific training remains gated on audited five-role source isolation, including the design's unresolved near-duplicate relation; MSRVTT/MSVD test remains sealed.

Repairs committed as `1db61d2` (semantic/source/objective contracts) and `00f3656` (parent/recovery/evaluation/distributed fixes). On `00f3656`, the complete evidence suite passed: **92 passed, 1 upstream TypedStorage deprecation warning, 42.36 seconds** with `python -m pytest -p no:cacheprovider tests/semantic_retrieval/evidence -q` and the CPU environment above. This includes a real two-Gloo-rank BPE/CLIP sparse index, real partial-decode evaluation, and a successor run where any repeated video decode raises immediately. Gallery recovery metadata is stored per candidate with an aggregate digest; snapshot inventories are sharded, avoiding a full-gallery JSON payload.

The affected broad regression passed **614 tests**, with one upstream warning, in **218.10 seconds**. Command: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider tests/semantic_retrieval tests/test_experiment_artifacts.py tests/test_manage_experiment_run.py tests/test_experiment_artifact_docs.py -q`. The later complete evidence suite above covers the final gallery recovery changes.

Actual production four-GPU Q0 passed and was archived in `a0bb35a`. Source `b9d5d2fb3adb5a510ed2236c5a1481e773a1486c`; manifest `f481e259a2d2240b2322948a47088cf29af676bdf8780e539090703fdb509c80`; qualification report `8ed5b1ea14c51ccb67bec6268f65bc9a20feec27a73afc8fd799623b7e7cb17f`. Four distinct A800 cards (physical 0/2/3/4), full ViT-B/32 random weights, FP32: all 13 checks passed, all five matcher variants and three value heads qualified, continuous/restored payloads identical. Worker exercise 57.96 s; controller 71.945 s / 0.07994 allocated GPU-hours. Peak allocated CUDA memory 2,011,554,304 bytes per rank (driver-reported usage was approximately 2.9 GiB). All 578 array references in 348 JSON files plus checkpoint external payload were verified before terminal completion; terminal inventory has 780 files. There was no failed attempt or retry.

Remaining limits: a synthetic software qualification does not establish scientific effectiveness; no real-data scientific run or test evaluation was launched. Recovery rejects ambiguous execution/prediction windows; the qualified replay and synthetic successor module test do not claim coverage of every possible OS/process crash. Successor latency reports the current worker segment explicitly. Real near-duplicate source membership and the five-role audit remain unresolved; source-based science gates stay closed.

Final whole-repository CPU validation: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider -q` — **643 passed**, one upstream warning, **261.83 s**. All review fixes are committed; no executable changes followed the successful Q0 qualification.

## 后续最终复核（2026-10-10，早期状态以上为历史）

来源修订已获用户批准；真实科学pilot已运行。最终产品1d16e57：受影响回归690 passed（306.44s），独立最终窄测试9 passed（5.51s），无剩余阻塞。最终Q0-R5四CUDA rank、13/13检查、五matcher/三head及578数组引用全部通过。W0/D0/S1/S2/B1与六个S3生产者均终态归档；B2可检验性门槛不足，停止B3/B4/扩展。V_locked未用于模型评价或选择，test sealed。

最新事实、协议偏差、审查限制及全部资源见experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md；独立实验审计/claim verdict另存同目录。早期段落的无科学运行、未定义来源和旧测试计数不再表示当前状态。
