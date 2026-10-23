# Frozen Laya validation integration

The approved design is `docs/superpowers/specs/2026-10-23-laya-vision-validation-reranking-design.md`.
Branch `codex/laya-validation-v1` descends from `codex/full-data-baselines` at `03a3fe7657852c3ea9be542f637e19cb6c6a6934`.

`fixed.json` freezes the original UATVR seed713 epoch05 checkpoint, 500-video/10000-query validation, FP32, DSL off, Top10, four frames, and measurement protocol. It binds only the original checkpoint, never the temporal-offset or caption-shuffle treatments. `external.input-index.json` hashes the local Laya Python runtime, processor, backbone config and weights. No substitute weights are fetched. `requirements.txt` pins the active integration dependencies; `environment.lock.txt` snapshots the complete read-only inherited base plus isolated overlay. The base environment and downloaded source/weight bytes are not edited.

Components are in `src/semantic_retrieval/laya_rerank/`: RGB evidence before transforms; native frozen judge; visual token cache; stable full-gallery evaluation; source-group bootstrap; immutable per-query completion blocks. The manifest-only future worker is `scripts/evaluate_laya_rerank.py --run-dir <reserved-run-dir>`.

Both reserve and prepare_run enforce the independent Laya policy. No scientific reservations/runs are enabled: `gpu_qualification=null`, `science_authorized=false`. The registry is deliberately empty. Do not toggle science on merely because CPU tests pass. A separately implemented/reserved two-GPU synthetic production-capacity inference/cache qualification, GPU budget/retry controller, worker lifecycle recovery and their review/archive are prerequisites. Historical dual-GPU qualifications do not count. Formal timing and scientific/resource reports remain future work.

Per-query block recovery is tested, including recomputation of partial blocks. A process interrupted after the lifecycle becomes running cannot simply invoke the worker again: use existing verify-run recovery protocol; failed/cancelled runs are immutable and require new reservations. No lifecycle recovery qualification is claimed by these unit tests.

## CPU checks

Run the narrow pure/integration suite with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. \
 /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/laya_rerank
```

The native test is opt-in because it loads the actual local model and needs the isolated overlay. It uses synthetic RGB frames, never validation/test data. Example for this host:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 LD_LIBRARY_PATH=/home/xujie/miniconda3/envs/ret/lib/python3.10/site-packages/nvidia/nccl/lib:/home/xujie/miniconda3/envs/ret/lib/python3.10/site-packages/nvidia/cudnn/lib \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
 LAYA_CPU_WEIGHTS=/data2/hxj/open-models/laya-vision/models \
 PYTHONPATH=src:.:/data2/hxj/open-models/laya-vision/code \
 .venvs/laya-validation-v1/bin/python -m pytest -q tests/semantic_retrieval/laya_rerank
```

The shared-library paths refer to pinned read-only dependencies already installed on this host; the new Torch/Transformers packages live in the overlay. GPU scientific reproducibility must authenticate the actual qualified runtime/environment before launch. The native test checks raw-pixel versus cached-feature logits at absolute tolerance 1e-5, microbatch1/8, argmax, final ordering, query independence/permutation, strict truncation, and model freeze.

2026-10-23 scoped amendment: user explicitly permits two GPUs for this study. New `qualify_laya_rerank.py` has a two-rank random production-capacity UATVR/Laya controller, 30-minute wall cap, cumulative software budget, owned-session cleanup and actual killed-process/resume/continuous checks. Scientific execution stays disabled pending actual qualification and its archival.
