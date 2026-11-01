# UATVR + AQS Gaussian KL: four-GPU entry

This entry uses your UATVR Gaussian KL matching implementation, not HVP or the
spatial-local-query branch. It constructs `FusionAqsModel`, requires active
symmetric Gaussian KL score fusion, and uses `fusion_step` and versioned fusion
checkpoint save/restore. Historical two-rank launchers are unchanged.

Remote checkout: `/data2/hxj/project/tvr` on `172.31.74.95`.
Default Python: `/data2/hxj/env/uatvr-kl-check/bin/python`; override with
`UATVR_KL_PYTHON`. Defaults: `CUDA_VISIBLE_DEVICES=0,1,2,3`, four torchrun ranks,
one CPU compute thread per rank, deterministic cuBLAS, FP32. The allocator limit
is 80% of each card (32 GiB on A100 40GB). No process is stopped by this script.

The global batch belongs to the frozen recipe. Keep the existing global batch64
for 16 examples per rank, rather than multiplying it to256. Smaller final batches
must still be divisible by4. The wrapper does not select or change eta.

Inspect the exact command without initializing CUDA:

```bash
cd /data2/hxj/project/tvr
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --mode qualify --run-dir /absolute/reserved/qualification-run --dry-run
```

For execution, use distinct, already reserved Scheme A run directories:

```bash
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --mode qualify --run-dir /absolute/reserved/qualification-run
bash scripts/run_uatvr_gaussian_kl_4gpu.sh --mode train --run-dir /absolute/reserved/training-run
```

These commands are launch interfaces, not permission to create arbitrary run
folders or launch an experiment. No study/run/eta is automatically created or
chosen by the wrapper. Before execution, freeze the dedicated study and reserve
runs with the standard artifact manager. This change alone does not reserve or
start a qualification or scientific training run.

## Sealed manifest contract

- Study/registry: `uatvr-aqs-kl-training-v1` /
  `experiments/uatvr-aqs-kl-training-v1/registry.jsonl`.
- `environment`: gpu_count4, precision fp32, four registered GPU UUIDs in runtime
  order. Runtime visible devices must match these UUIDs exactly.
- `config.model`: architecture `uatvr-aqs-kl-v1`, formal CLIP B/32 recipe,
  explicit active `FusionConfig.recipe()` with positive frozen eta.
- `config.loss.aqs`: explicit active `AqsConfig.recipe()` matching the training
  recipe's negatives and weight. Optimizer kind `pinned_bert_adam`, scheduler
  kind `warmup_cosine`.
- `config.evaluation`: mode, dsl=false, four-rank validation-only training recipe,
  complete `fusion_launch.source_inventory(repo)` as source_files, a hashed
  environment_spec JSON with pinned `packages`, and train-only source_report
  and qualification references. The frozen study contains matching recipe,
  fusion settings and resource_policy.gpu_uuids.
- Dataset synthetic for qualification, msrvtt for training; test_authorized=false.
  Both manifest and reservation must use the exact output directory. Bind the
  certified train/validation input indices and pretrained checkpoint references
  for training; no test execution entry is included.

Qualification creates full-capacity random model weights without consuming the
training checkpoint or real dataset. It checks intended gradients and parameter
updates, exact next-step continuation after fresh-model checkpoint restore,
finite repeatable inference and four-rank model agreement. The training run
requires the completed qualification's committed registry record, manifest,
terminal output hash, source inventory, environment and identical fusion recipe.
Training starts from the pinned CLIP input, preserves the complete unique-video
caption batches and selects checkpoints using validation only.

CPU tests cover four Gloo ranks, real fused forward/backward, state agreement,
exact checkpoint continuation, sharding, device identity and reservation guards.
They do not establish CUDA numerical correctness, memory fit or throughput.
