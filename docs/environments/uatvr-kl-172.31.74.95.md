# Gaussian KL CPU compatibility on 172.31.74.95

The environment specification is `uatvr-kl-172.31.74.95.json` beside this file.
The remote existing `tvr` environment is Python3.11; local reference is3.10.
Do not replace or upgrade the existing environment. The dedicated venv inherits
its existing packages and adds only the pinned missing schema-validation stack.
These checks use synthetic CPU inputs and provide no scientific GPU qualification.

Initial setup, once (refuse to overwrite an existing dedicated environment):

```bash
test ! -e /data2/hxj/env/uatvr-kl-check && /home/xujie/.conda/envs/tvr/bin/python -m venv --system-site-packages /data2/hxj/env/uatvr-kl-check
/data2/hxj/env/uatvr-kl-check/bin/python -m pip install jsonschema==4.23.0 jsonschema-specifications==2025.4.1 referencing==0.36.2 rpds-py==0.25.1 attrs==25.3.0
```

Run the CPU seeded-kernel witness and regression suite verbatim:

```bash
cd /data2/hxj/project/UATVR-kl-check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /data2/hxj/env/uatvr-kl-check/bin/python -c 'import torch; torch.set_num_threads(1); torch.manual_seed(713); x=torch.randn(32,32); torch.manual_seed(713); y=torch.randn(32,32); assert torch.equal(x,y); assert torch.equal(x@x.T,y@y.T); print("CPU seeded kernel PASS",torch.__version__)'
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /data2/hxj/env/uatvr-kl-check/bin/python -m pytest tests/semantic_retrieval/aqs/uncertainty -q --tb=short
```

Checked assets: MSRVTT 10000 videos present, train9k/test1k IDs disjoint;
annotation CSVs and MSRVTT_v2.json match local hashes. MSVD YouTubeClips has1970
entries; its annotation/split completeness has not yet been audited.
CLIP B/32 was not located in the checked remote model/cache paths.
No videos or test labels are consumed by this software suite.
