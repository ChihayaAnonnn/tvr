# Diagnosed Q0 prelaunch cancellation

Run: synthetic-baseline-laya-q0-dual-seed713-20261023T082006Z-19b6f564873d.
The single-process controller called the worker-only prepare_run environment validator. All Python/Torch/CUDA/model/precision values matched, but the controller is single-process while the manifest correctly records the two-rank worker topology. Guard rejected before start_run and before launching any GPU worker. verify-run succeeded; status was still reserved. Terminal cancellation preserves the immutable run and a zero-spawn/zero-GPU-hour receipt; no checkpoint/data were read and no scientific run started.

Fix: controller verifies the sealed reservation and scoped policy, then uses manager.start_run for the lifecycle; actual child ranks perform full runtime environment/precision equality checks with real WORLD_SIZE=2. Keep the manifest worker topology torchrun-2; do not spoof controller WORLD_SIZE or relax the worker environment guard. Complete metrics/lifecycle publication remains run-local and occurs only after every owned worker exits.

One diagnosed successor is authorized by the campaign's existing two-successor policy, under a new source/run identity linked to this cancelled run. Budgets and test sealing are unchanged.
