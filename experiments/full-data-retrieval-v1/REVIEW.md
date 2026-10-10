# Independent final software review

Reviewed source6370ce21937a43b9309d78a1ea1a49ba8b363fe5, fresh independent agent `full_data_final_review`, read-only. Native implementation remained sequential. No review GPU job, source/index/HEAD mutation or extra reviewer delegation occurred.

Confirmed strengths: vendor original/adapted hashes match; initialization/freezing/optimizer/meanP follow disclosed recipe; complete-caption sampling, full-gallery metrics/ties, input authentication, sealed-test gates, executable-source qualification guards, count-aware quota and self-contained checkpoint recovery are present. No Critical findings or confirmed Minor findings.

Two Important findings were corrected in one pass:

1. SIGTERM/SIGHUP previously bypassed controller cleanup, leaving the separate owned worker group alive. Scoped handlers now enter the existing exception cleanup and restore previous handlers afterwards. A real protected CPU controller SIGTERM test first failed because its worker survived, then passed: worker stopped, non-success terminal state/receipt archived and final log hash matched. A real two-rank CPU torchrun test verifies launcher/ranks share the owned group in the pinned PyTorch2.1 environment, and all writers stop. SIGKILL remains necessarily fail-closed manual integrity recovery; no history is overwritten.
2. Torchrun previously defaulted elastic metadata to system temporary directories. The sealed command now binds `--log-dir {artifact_run_dir}/logs/elastic` before the worker script. The command assertion first failed, then passed. The actual two-rank launcher creates its attempt directory under the specified run-local logs root.

Correction validation:11 controller/campaign/worker tests passed in26.43s. The independent reviewer executed the RUNBOOK CPU command verbatim:10 passed in19.66s, no documentation/environment divergence; model/data tests20 passed in9.88s; all provenance hashes matched. An upstream TypedStorage deprecation was the only warning.

Final post-correction regression: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data tests/semantic_retrieval/test_no_legacy_dependencies.py` →53 passed in37.82s, one upstream deprecation warning, no failures.

Declined to judge, with primary rulings: actual four-CUDA-rank numerical reproducibility/memory/throughput require new Q0; scientific schedule completion and retrieval quality require authenticated scientific runs. Both remain explicit outstanding gates, never inferred from CPU tests. Prior negative pilot remains closed; test remains sealed. Four-card resource shortage is external, not evidence for or against the scientific hypothesis.
