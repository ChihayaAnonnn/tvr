# Laya implementation progress — incomplete

Branch: codex/laya-rerank-validation; parent 03a3fe7.

User-approved specification remains the conversation's fixed Top-10/four-frame validation design.
Latest explicit authorization: this Laya study may use two GPUs initially; must qualify the new path,
not reuse full-data study qualification. No test access.

Implemented pure frame selection, stable reranking and paired metrics; implemented real Laya adapter
using native input construction, cached image features and unrounded temperature-scaled logits.
Adapter is NOT yet validated against real weights. No scientific runs started.

Verification: initial core test collection failed (missing implementation); then 3 core tests passed.
`/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/laya_rerank tests/semantic_retrieval/full_data/test_factors.py tests/semantic_retrieval/full_data/test_baseline.py`: 13 passed, 1.98 s.

Actual CPU loading attempted offline: transformers 4.52.4 expects preprocessor_config.json,
while downloaded processor uses the newer format. Do not mutate downloaded files or old tvr environment.
System python venv creation failed because ensurepip is unavailable. Next: use conda Python's venv
or an isolated conda environment, install pinned model-compatible dependencies, validate real CPU
forward/cache equivalence before claiming integration works.

Outstanding: environment; real CPU tests; raw RGB evidence decoder; authenticated persistence/resume;
source-group bootstrap; independent study and lifecycle integration; timing worker; qualification;
full validation; independent review. Existing functions are not an executable experiment pipeline.
