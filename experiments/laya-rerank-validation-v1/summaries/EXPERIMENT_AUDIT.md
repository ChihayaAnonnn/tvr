# Experiment Audit — Laya validation reranking

Date: 2026-10-23. Fresh read-only Codex reviewer and independently delegated GT/weights subreviewer. **Review independence: same-family; acceptance status: provisional.** No GPU replay or repository edits were performed. The only written deliverables are this report and its JSON counterpart in `/tmp`.

## Overall verdict: WARN — scope qualifiers required, no remaining integrity blocker

The completed accuracy result and final performance result are supported by authenticated files and independent CPU recomputation. The fixed method worsens local-validation retrieval and raises measured online cost. WARN records the limited scope; it does not indicate fabricated or mismatched final numbers. Failed performance predecessors are excluded from the final comparison.

All abbreviated code filenames below resolve under `/data2/hxj/project/UATVR/src/semantic_retrieval/laya_rerank/`; paths beginning `src/`, `experiments/`, or `dataloaders/` resolve under `/data2/hxj/project/UATVR/`.

## A. Ground-truth provenance: PASS

GT is official MSRVTT caption/video annotation on the repository's **local seed-0 holdout from the official 9k training membership**, not model-generated reference and not an official test result. The subreviewer reconstructed all 174,920 certified caption records with allowlist-first annotation extraction, all source unions, and all 500 validation-video file hashes. Exactly 10,000 unique queries, 20 captions/video, 500 videos and 493 source groups were reproduced; saved targets and group indices match exactly. No excluded/test caption values were decoded.

Evidence: `src/semantic_retrieval/data/msrvtt_snapshot.py:103`; `src/semantic_retrieval/full_data/loader.py:93`; `src/semantic_retrieval/evidence/source_preparation.py:27`; `dataloaders/splits/msrvtt_trusted_v1_seed0.json:2`; `src/semantic_retrieval/laya_rerank/science.py:78` and `:187`. Judge invocation receives query strings and visual evidence only (`evaluator.py:61`); targets are computed afterward (`evaluator.py:73`). RGB decoding verifies certified bytes and accepts only validation (`evidence.py:12`).

The parent epoch05 checkpoint/sidecar/result/metrics hashes match; all 384 hydrated UATVR tensors match the model digest, checkpoint progress is epoch 5/update 12885, and 377 optimizer states have step 12885. Accuracy deliberately uses the authenticated archived epoch 5 score matrix (`science.py:76–95`), rather than rerunning UATVR. The original training path evaluates immediately before checkpoint saving (`src/semantic_retrieval/full_data/baseline.py:158–169`). All pinned Laya inventory files match; a fresh CPU load reproduced all 416 checkpoint tensors exactly. Loader evidence: `judge.py:20–25`, `worker.py:40`, `/data2/hxj/open-models/laya-vision/code/laya/vlm.py:959`. Upstream training corpus/revision provenance is not independently established (`/data2/hxj/open-models/laya-vision/models/vlm_agent_config.json:20`); do not claim contamination-free generalization.

## B. Score normalization and computation: PASS

All six primary accuracy NPY contracts verified. Independent recomputation gives:

| Metric | Archived UATVR | UATVR + Laya |
|---|---:|---:|
| R@1 |53.19%|16.05%|
| R@5 |79.70%|55.04%|
| R@10 |86.83%|86.83%|
| MRR |0.6489606049330214|0.33876127953619595|

Corrected 518; worsened 4232. Delta R@1 =−37.14pp. All 10,000 source-group bootstrap draws were reproduced exactly, seed 713, 493 groups,95% CI **[−39.929725903614454,−34.258489751780786]pp**. Full 500 permutations, stable original sorting, unchanged Top10 membership/tail, stable probability-only reranking and R@10 invariance were verified in every row. Both 5000-query rank outputs match the merged arrays exactly. Raw logits transformed with checkpoint temperature 3.8573756217956543 reproduce every P(match) bit-for-bit. Subtracting the maximum in softmax is numerical stabilization; recall/MRR denominators are dataset counts, not prediction statistics. No rounding or fusion was found.

Evidence: `core.py:21–45`, `judge.py:68–72`, `evaluator.py:25–37`, `science.py:185–202`, and `/data2/hxj/project/UATVR/artifacts/laya-rerank-validation-v1/msrvtt/baseline/msrvtt-baseline-laya-accuracy-seed713-20261023T085620Z-34d5b5f84908/provenance/laya-validation.json:1`.

All six 100-element timing arrays and selected-index arrays verified. Independent pooled and per-round recalculation exactly matches the reports:

| Online statistic | Baseline | Enhanced |
|---|---:|---:|
| Median ms |181.58377055078745|300.76409655157477|
| P95 ms |185.16071517951787|307.42126336554065|
| Serial queries/s |5.508699665579134|3.3252321573885895|
| Peak PyTorch allocated bytes |746690560|1751102976|
| Loaded model/index resident allocated bytes |682641920|1520204288|

Median ratio 1.6563379846077906; P95 ratio 1.6602942101810754. Peak allocated increment 1,004,412,416bytes; resident allocated increment 837,562,368bytes. Memory differences and round maxima were independently recalculated from hash-authenticated runtime receipts; GPU allocation counters themselves were not replayed. Evidence: `measurement.py:83`, `:95`, `:126–175`; `/data2/hxj/project/UATVR/artifacts/laya-rerank-validation-v1/msrvtt/efficiency/msrvtt-efficiency-laya-performance-retry2-seed713-20261023T094942Z-06464a9df9f4/provenance/laya-performance.json:1`.

## C. Result existence and identity: PASS

All **22,032 accuracy output files**, **35 final-performance output files**, and **215 final-qualification output files** match terminal inventory hashes. Manifest canonical digests, completed statuses, committed registry/summary anchors and source-qualified execution identities match. Accuracy completion: `experiments/laya-rerank-validation-v1/registry.jsonl:8`, `/data2/hxj/project/UATVR/artifacts/laya-rerank-validation-v1/msrvtt/baseline/msrvtt-baseline-laya-accuracy-seed713-20261023T085620Z-34d5b5f84908/status.json:22040`; final performance completion: registry`:20`, `/data2/hxj/project/UATVR/artifacts/laya-rerank-validation-v1/msrvtt/efficiency/msrvtt-efficiency-laya-performance-retry2-seed713-20261023T094942Z-06464a9df9f4/status.json:43`. Complete file SHA-256 audit inputs are in the companion JSON; array-contract hashes are distinguished from whole-NPY-file hashes.

The scalar comparison equals the raw reports. The human draft correctly rounds the metrics, describes the local holdout, repeats100 queries rather than claiming300 unique queries, identifies PyTorch rather than whole-card memory, and excludes failed predecessors. One initial budget rounding typo was corrected before this final audit: software0.222402GPUh, science including failures1.092670GPUh. Independent summation of all 10 run receipts gives 0.22240235773911182 and1.09267021555861GPUh respectively.

## D. Executed metric paths and engineering checks: PASS

Accuracy worker calls evaluation, merges both ranks and publishes computed metrics (`worker.py:54`, `science.py:174–202`). Final controller runs preparation plus **B0,E0,E1,B1,B2,E2**, with seven distinct PID/start-tick/nonce receipts in increasing order (`science_controller.py:59`). The same independently hash-selected100 query IDs appear in all six rounds; executed code performs five warmups before 100 synchronized timed queries (`measurement.py:124–131`). Completed per-arm reports prove the strict archived baseline/enhanced Top10 guards were reached without failure (`measurement.py:118–122`). Maximum reported baseline score difference is2.47955322265625e−05; do not call replay bit-exact. The guards check exact Top10 ordering for this measured sample, not all10,000 fresh online outputs.

Independent source/CPU inspection found the inherited singleton mask alias: upstream pooling clears BOS in-place (`src/semantic_retrieval/full_data/upstream_uatvr/modeling.py:412–413`); a32-of34 mask slice aliases at batch 1 but copies at larger batches (`:483`,`:508`), changing the later WTI mask (`:512`). The measurement-only adapter clones this temporary input (`measurement.py:18–39`), preserving original archived batch semantics without changing weights, frame preprocessing, batch 1 serving or upstream baseline source. The actual-helper regression independently passed. The executor's trained-checkpoint CPU diagnostic is recorded in `experiments/laya-rerank-validation-v1/incidents/singleton-pooling-mask-alias.md:7`; this audit independently establishes the alias mechanism and confirms the new GPU-qualified/final-run guards.

The final qualification explicitly compares original multi-query scores against protected multi-query/single-query scores and exact rankings (`qualification.py:84–92`); its source matches final performance. All qualification checks passed, cached/raw-pixel error 0 and batch 1/8 max error 4.470348358154297e−07. Qualification resume and distributed-vs-single arrays were independently compared in the accuracy qualification and match bitwise. Native cached/raw-pixel qualification uses random weights and a limited synthetic input (`qualification.py:143–150`), so it is not exhaustive trained-validation equivalence.

The failed ownership-monitor predecessor remains failed and unused. New monitoring retains only the authenticated launch PID/start ticks, handles disappearance/zombies and rejects PID reuse (`science_controller.py:25–39`). Its historical generic failure log cannot distinguish external interference from the demonstrated race; the final incident text states this limit. No partial predecessor timing was imported.

## E. Scope assessment: WARN

Supported scope is one frozen configuration and one validation-selected checkpoint, full local validation accuracy, and the same 100-query workload repeated three times on one A800 80GB GPU with query batch 1 and candidate microbatch 8. This is 300 timing observations per arm, not 300 distinct queries. There is no independent test, across-seed robustness, general efficiency benefit or verified absence of upstream training overlap. The final draft does not overstate these claims.

The original-cache payload/disk sizes are 12,294,000/12,294,256bytes; Laya visual-cache payload/disk sizes are 295,996,000/296,378,015bytes, independently reproduced. Frame metadata confirms 500 videos each with four distinct frames and zero deduplication removals. Cache construction times 134.04404825996608s and334.04066618182696s are authenticated descriptive measurements from different stages/runs, not a controlled cache-speed ratio. CPU payload excludes Python-container overhead. Enhanced cache RSS change 4096bytes is allocator-dependent and cannot be presented as total cache footprint. GPU memory values are process-level PyTorch counters, not whole-device telemetry. Occupancy guards reported no detected unrelated process; this is monitored execution evidence, not continuous independent hardware tracing.

## F. Evaluation classification: PASS

Accuracy: **real_gt**. Software qualifications: synthetic-only engineering checks with random initialization. Performance: systems measurements on real validation inputs. No proxy accuracy is presented as a ground-truth result.

## Claim impact and disposition

- Fixed validation negative result: supported with the local-holdout/single-checkpoint qualification.
- Higher online latency and allocated-memory cost on the stated workload: supported; all statistics recompute exactly.
- Complete candidates, no GT labels to judge, no fusion and no test unsealing: supported by inspected code and authenticated data.
- Native cache equivalence: supported only for the recorded synthetic random-weight qualification; exhaustive trained-model equivalence is not established.
- General efficiency advantage, official-test/generalization or contamination-free claims: unsupported and absent from the audited final draft.

No further change is required to the scoped scientific/performance claims. Preserve these qualifiers and the failed-run history. The durable executor receipt `experiments/laya-rerank-validation-v1/summaries/cpu-tests.json` records the exact command/environment/source and raw pytest output: 152 passed, 2 skipped, exit 0. This is verified executor-run evidence; the auditor independently ran the targeted mask regression, not the broad suite. Audit JSON records the exact audited file hashes and verification limits. Full reviewer/tool trace should be archived by the executor separately from these two summary deliverables.
