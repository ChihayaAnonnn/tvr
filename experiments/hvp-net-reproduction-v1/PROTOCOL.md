# HVP-Net corrected reproduction

The user explicitly requested two idle GPUs and retained the author's train/test selection protocol, declining separate validation or single-process evaluation. This independent study authorizes 9k training and per-epoch JSFUSION 1k test checkpoint selection only for this HVP-Net reproduction. It does not authorize other studies or unseal their test splits. Results must be labeled test-selected reproduction, not a matched validation-selected comparison.

Both ranks extract features and score separate query rows; CPU communication uses Gloo alongside NCCL. Fixes remove candidate-paired-caption weighting and strictly load checkpoints. A fresh two-rank production-capacity random-weight gradient, numerical, inference and checkpoint qualification precedes science. GPU 0/3 UUIDs are pinned; no other processes are modified. This scoped two-card authorization replaces the four-card topology for this study only.

All new output remains run-local under artifacts/hvp-net-reproduction-v1; source SHAs, tracked source file hashes, environment/dependency hashes, input indices and CLIP SHA-256 are frozen in the manifest. One five-epoch training attempt with global batch 32; no automatic retry. Qualification budget one hour, training budget 36 hours. No exact stochastic training-resume claim is made for upstream-style checkpoints.
