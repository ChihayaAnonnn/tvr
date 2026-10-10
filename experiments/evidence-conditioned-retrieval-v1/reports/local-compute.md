# Local compute ledger

### env: tvr@fbf85621
how: reuse /home/xujie/.conda/envs/tvr/bin/python; no installation or environment mutation.
tier: 8 CPU threads, 64 GiB host RAM, up to 4 allocated A800 80GB cards.
weights: completed W0 public CLIP bytes, SHA recorded in compute-env-spec.json.
validated: imports and genuine W0 CPU B/32 forward passed; fresh documentation witness and new Q0 pending.
gotcha: generic `python` is absent; always use the absolute interpreter. GPU probes use the archived study qualification, never an unarchived scientific invocation.

Documented CPU invocation (run verbatim from /home/xujie/.codex/worktrees/ad3a/UATVR):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/xujie/.conda/envs/tvr/bin/python -c 'import torch,numpy,cv2,yaml; torch.manual_seed(713); x=torch.randn(8,8); y=x@x; assert y.shape==(8,8) and torch.isfinite(y).all(); print("ENV_WITNESS",torch.__version__,numpy.__version__,cv2.__version__,yaml.__version__,tuple(y.shape),y.device)'
```

Final environment readiness also requires the archived new-chain four-GPU qualification.
