# Independent UATVR+AQS training

User-approved initial recipe: MSRVTT complete certified training cohort, seed713,
5 epochs, global batch64 (32 per GPU), learning rate5e-5, M16, auxiliary weight0.2.
This is a new AQS run, independent of previous baseline and Laya campaign guards.
No Laya model, overlay, reranker, teacher, or cached Laya features are used.
Only the certified train and validation cohorts are consumed; no test entry exists.

Use the existing `tvr` environment recorded in `.aris/compute/local.md`.
From a committed, unchanged checkout, prepare an exclusive artifact directory:

```bash
PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python scripts/prepare_aqs_training.py --run-dir /absolute/artifact/run --negatives 16 --weight 0.2
```

Launch with CUDA_VISIBLE_DEVICES specifying two free GPUs, OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONPATH=src:.
Use `/home/xujie/.conda/envs/tvr/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 --module semantic_retrieval.aqs.launch --run-dir /absolute/artifact/run --mode qualify`,
then the same command with `--mode train` only after qualification succeeds.

Qualification uses production ViT-B/32, 12 frames224, text32, batch64 and the same
optimizer schedule. It checks finite losses/gradients/parameters, intended gradients,
parameter update, and exact continuation of loss/model/optimizer after checkpoint
restore on two CUDA ranks. Its synthetic results are software checks, not retrieval results.

The run freezes source commit and Python source hashes, input index hashes, CLIP
SHA256, sampler, ID mapping and AQS recipe. Each raw video is hash-checked on decode.
Source files must remain unchanged during execution.
Training preserves all captions in deterministic unique-video batches, including
the smaller final batch. Every 500 updates and each epoch publishes identity-bound
checkpoints. These support the existing restore API; automatic restart orchestration
is not included in this entry. Failures stop the run and publish failure.json.
Validation uses the inherited full 10000-query/500-video gallery, raw T2V and V2T
metrics, and chooses the earliest epoch with maximum validation T2V R1.
Logs include full original losses, AQS loss, weighted AQS and selection diagnostics.
The warmup schedule uses a zero learning rate for the first update.
