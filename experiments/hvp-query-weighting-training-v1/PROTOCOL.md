# Query-owned HVP restart

User explicitly requested old HVP checkpoint deletion, stopping the previous training,
query-owned word weighting in training, and a fresh dual-card launch.

Train and inference both use `abw,aw->ab`. No paired candidate caption weights.
Keep original seed42, B32, layers0/5/11, 5 epochs, global batch32, lr1e-4,
CLIP coefficient1e-3, weight_decay0.2, warmup0.1, words24, frames12, fp16.
Start from original CLIP, with a new optimizer and no resume.
Preserve the user's existing 9k train/JSFUSION1k per-epoch evaluation protocol,
no added validation or single-process evaluation. Test-selected reproduction
results must be labelled accordingly. Ties count each query once.

Use only idle registered GPU0/3, reserve60GiB/rank; do not stop other jobs.
Fresh full-capacity random-weight FP32 and FP16 two-rank gradient, optimizer,
checkpoint roundtrip and full encoder/distributed inference qualification is
required before scientific training. No reuse of inference-only qualification.

Old trained HVP checkpoints are deleted only under an explicit hashed cleanup
record; retain logs, matrices, reports, original CLIP and qualification artifacts.
