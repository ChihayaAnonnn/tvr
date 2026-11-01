# UATVR + AQS Gaussian KL: four-GPU training

The config-driven entry follows the repository-wide simplified rules in
AGENTS.md. It needs a config and input files, with no study, registry,
reservation, manifest or formal qualification gate. Historical sealed worker
code is retained for old runs; the shell wrapper now invokes `fusion_train`.

Remote checkout: `172.31.74.95:/data2/hxj/project/tvr`.
Python: `/data2/hxj/env/uatvr-kl-check/bin/python` (override `UATVR_KL_PYTHON`).
Config: `configs/uatvr_gaussian_kl_4gpu.json`.

The config uses original CLIP ViT-B/32, FP32, four ranks, global batch64
(16 per rank), seed713, 5 epochs, BertAdam base LR5e-5 / CLIP multiplier0.001,
M16, AQS weight0.2 and fixed eta0.1. Eta0.1 is an exploratory starting value,
not a validated optimum. Local spatial queries and Laya are absent.

`artifacts/inputs/msrvtt-complete.json` contains the previous certified
8246-video/164920-caption training and 500-video/10000-caption validation cohort,
including the original compressed-224/12fps video paths. Test is not loaded.

```bash
cd /data2/hxj/project/tvr
# Inspect the command without CUDA initialization.
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --dry-run
# Actual full-capacity, real-data smoke in a separate output directory.
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --smoke --output /absolute/new-smoke-directory
# Full training, using config.output as a new directory.
bash scripts/run_uatvr_gaussian_kl_4gpu.sh
# Resume in that same output directory.
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --resume /absolute/training-directory/latest.pt
```

Check GPU availability before launching. Default devices are0,1,2,3; override
CUDA_VISIBLE_DEVICES with four distinct indices or UUIDs. Allocator cap is80%
per card. The smoke checks actual gradients/updates, fresh-model exact next-step
recovery, repeatable rectangular inference and model agreement across ranks.

The output contains the effective config, simple status, per-rank diagnostic logs,
epoch validation metrics/scores, interval/epoch checkpoints and latest.pt/best.pt
links. Redirect console output to one persistent log when detaching the job.
Selection is earliest epoch with highest raw validation T2V R@1; DSL is off.
Checkpoints include model, optimizer and its warmup/cosine state, epoch/batch
cursor, history and each rank's RNG. Resume validates scoring/config compatibility
without binding to a source commit or external registry.
