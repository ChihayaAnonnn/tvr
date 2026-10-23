# Laya validation integration — CPU delivery

2026-10-23. Branch: `codex/laya-validation-v1`, based on `codex/full-data-baselines` (`03a3fe7657852c3ea9be542f637e19cb6c6a6934`). Approved design saved in `docs/superpowers/specs/2026-10-23-laya-vision-validation-reranking-design.md`. This report supersedes the earlier partial implementation note: the current requirement is **four GPUs**, and no historical two-GPU authorization applies.

Implemented: authenticated original-resolution RGB evidence before CLIP transforms; native local frozen Laya FP32 choice adapter using checkpoint temperature and block option attention; query-independent CPU visual-token cache; immutable identity/hash-bound per-query blocks; complete stable Top10 reranking and paired T2V metrics; source-group paired bootstrap seed713/10000; manifest-only evaluator entry; independent reserve/start gate; fixed original seed713 epoch05 identity, no factor checkpoint substitutions; external runtime/weight inventory and dependency snapshots. Existing UATVR scoring code is unchanged. Labels, IDs, filenames, ranks and retriever scores do not enter the judge.

CPU tests cover short sampling, duplicate-frame removal, tampered/undecodable video rejection, invalid/nonfinite probabilities, full permutation/tail invariants, R@10 invariance, source-group bootstrap, per-query identity isolation, partial-block recomputation and idempotence, microbatch1/8, independent shards including two spawned CPU processes, wrong split/coverage rejection, missing input bindings and closed science gates.

Fresh verification:

- Legacy-env affected regression: **73 passed, 1 explicitly skipped**, 78.06s. Paths: `tests/semantic_retrieval/laya_rerank`, `tests/semantic_retrieval/full_data/test_baseline.py`, `test_factors.py`, `tests/test_manage_experiment_run.py`, `tests/semantic_retrieval/test_runtime_artifacts.py`. The skip requires explicit real local weights/environment.
- Isolated overlay with real local weights and synthetic RGB inputs: **19 passed**, 47.19s, two upstream SWIG deprecation warnings. Tests retain absolute logits tolerance `1e-5`, matching argmax and final ranking for batch1/8; raw-pixel vs cached-feature forward; query permutation/independence; truncation rejection; eval/frozen parameters.
- All **39** external runtime/processor/config/weight content hashes match the recorded inventory. `git diff --check` clean. CLI help works.
- Independent read-only reviewer found no remaining direct correctness blocker for this CPU-only delivery. The reviewer checked native API calls, parent checkpoint world-size identity reconstruction, checkpoint/publication hashes, query/cache isolation, manifest input closure and scientific gate refusal.

Dependency diagnosis: legacy Torch2.1.2/Transformers4.52.4 cannot load the downloaded processor/config; Transformers4.57.6 still misinterprets its `extra_special_tokens`. The local backbone records Transformers5.17.0, which works with the isolated Torch2.5.1 overlay. Existing base packages are inherited read-only and snapshotted; two pinned existing CUDA shared-library directories are needed for loading the CUDA-enabled Torch binary even when tests hide CUDA. Native CPU commands are in the study README. No external source/weight content was replaced, and offline flags prohibit remote model fallback.

## Scope and remaining gates

No experiment reservations, four-GPU qualification, scientific GPU evaluation, test access, performance comparison or accuracy-gain conclusion were produced. The independent registry is empty, `gpu_qualification=null`, and `science_authorized=false`.

Query-block recovery is implemented/tested; complete worker lifecycle recovery is not qualified. The existing runtime rejects restarting a `running` worker as a fresh reservation; recovery must first follow verify-run and the existing archive protocol. Failed/cancelled runs cannot be overwritten or reused.

Before any future scientific execution, implement/review the production four-GPU random-weight synthetic inference/cache qualification plus budget/retry/deadline controller and worker-level recovery; reserve/commit/verify/run/finalize and archive its actual checks. Formal single-query timing/resource worker and its controlled-idle-GPU measurements remain future work. CPU success does not prove CUDA equivalence or scientific benefit. Do not enable science by simply changing a Boolean.
