# Full-data Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans Native, sequential implementation; independent final review only.

**Goal:** Run authenticated strong baselines on all permitted MSRVTT data and report complete validation retrieval metrics.

**Architecture:** Independent full-data study and package reuse Scheme A, tokenizer, video/weight readers. Pinned official algorithms are separate from the prior evidence model. CPU compilation and software qualification precede scientific training.

**Tech Stack:** Existing PyTorch/OpenCV/NumPy environment, official UATVR/CLIP4Clip source.

**Spec:** docs/superpowers/specs/2026-10-10-full-data-baselines-design.md

## Global Constraints

8246training/500validation videos, every caption,12frames/32tokens,FP32. Test sealed. New chain four CUDA ranks required. New study independent; negative prior B2 unchanged. No tracked edits during workers; all runs reserved/committed/verified/archived. Software8GPUh/science512GPUh;1800/86400seconds per run; at most2 software successors. Active-phase count-aware atomic quota. Complete cohort and strict load; no prior trained retrieval checkpoint.

## Review Focus

Official model fidelity, all-caption evaluation denominator and tied scores; cross-source leakage and test parsing; duplicate-positive contrastive batches; multi-rank gradient scaling; checkpoint/active-quota crash windows. No result inferred from a tiny or partially completed cohort.

### Task 1: Complete input preparation and retrieval metrics

Files: src/semantic_retrieval/full_data/{data.py,metrics.py,authorization.py}; scripts/prepare_full_data_retrieval.py; tests/semantic_retrieval/full_data/test_data_metrics.py; reserve/start guard integration.
Interfaces: compile_sources(raw,allowlist,trusted,content)->full video/caption records and quarantine; retrieval_metrics(scores,query_video_ids,gallery_ids)->bidirectional ranks/recalls; assert_full_data_authorized(...) at reserve and start.
- [ ] Write tests for all captions, excluded-source train membership, no sealed caption decoding, deterministic ties/permutations and actual best-positive V2T ranks; observe failures.
- [ ] Implement strict compiler, metrics and CPU source worker using bound indices and existing parser/lifecycle.
- [ ] Run narrow tests and affected archive regressions; commit source.
- [ ] Reserve/commit P0 CPU phase, run complete hash/decode compilation, validate/finalize/commit.

### Task 2: Official model fidelity and training

Files: src/semantic_retrieval/full_data/{models.py,training.py,upstream_uatvr/}; tests/semantic_retrieval/full_data/test_models_training.py.
Interfaces: official model adapter exposes training loss and separable full-gallery features/similarity; all-gather positive-aware loss; strict optimizer/checkpoint/RNG state.
- [ ] Write meanP numerical equivalence, intended gradient and multi-positive behavior tests; observe failures.
- [ ] Namespace pinned source, record original/adapted hashes and patches; implement trainable baseline adapters and exact schedules.
- [ ] Run real CPU forward/backward and exact-resume tests; commit.

### Task 3: Protected worker, controller and four-card qualification

Files: src/semantic_retrieval/full_data/{runtime.py,controller.py,qualification.py}; scripts/{train_full_data_retrieval.py,qualify_full_data_retrieval.py}; qualification/controller tests.
Interfaces: typed sealed worker config; new study gates at reserve/start; atomic GPU-count-aware active quota and chain successor cap; lifecycle outputs only in run root.
- [ ] Test unqualified/test/config mismatch rejection, concurrent quota reservations, retry cap and interrupted recovery; observe failures.
- [ ] Implement actual end-to-end worker and synthetic full-capacity four-rank qualification.
- [ ] Run CPU integration/regressions, commit; reserve/commit Q0 and launch only when four GPUs idle.
- [ ] Authenticate and archive Q0; open science only with committed P0/Q0 evidence.

### Task 4: Full-data scientific runs

- [ ] Freeze exact upstream optimizer, decoder and schedule differences before reservation; read current study/registry.
- [ ] Reserve/commit complete baseline phase; run seed713 complete training/validation with all input hashes. No tracked edits while workers live.
- [ ] Archive every arm/failure; run remaining frozen seeds where valid and within budget. No scientific hyperparameter rescue.
- [ ] Compare same-protocol baselines and external paper values with explicit split/candidate/training differences. Do not claim C1/C2 from baseline establishment.

### Task 5: Independent review and final report

- [ ] Independent final review of source and archives, fix correctness findings with behavioral tests and requalify changed GPU paths as required.
- [ ] Record exact commits/tests/resources/all results, unresolved scientific definitions and external GPU blockage. Source stays clean and committed.
