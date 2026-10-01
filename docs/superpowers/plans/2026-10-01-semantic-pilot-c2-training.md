# C2 semantic benchmark pilot training implementation

Spec: `docs/superpowers/specs/2026-10-01-semantic-retrieval-experiment-design.md`, sections 4, 5, 9.

Implement in isolated `codex/semantic-pilot-c2`; no assets, GPU runs, reservations or old studies are touched. Use test-driven development with CPU synthetic tensors.

1. Add a strict optional pilot configuration binding arm, independent sampler seed, hashed fixed candidate ordering and hashed P1 parent config/checkpoint/metadata. Default model/train settings remain compatible. Enforce four ranks, global 16, local 4, fixed 120/240 updates and registered optimizer/loss/model capacities in production.
2. Add a global query shuffle sampler using a private Python RNG, rejecting undersized inventories. Take 16 unique queries without padding across cycle boundaries, shard four per rank; identical branch seeds produce identical tables. Shuffle a balanced eight-cell K/panel schedule per 120 updates (15 each). Store versioned identity/RNG/order/cursor/iteration state in checkpoint progress, validate upon restore.
3. Construct panels from SHA-bound CLIP ranking and full canonical episode inventory, retaining exactly K IDs sorted for input, one positive for present, all negatives for absent. Refuse unknown or multiple-positive inventories. Decode/collate the selected episodes inside each real training iteration.
4. Add A0 WTI-only forward without attention/MLP work; detach A2 representation/logits; add A4 detached eight score statistics and two 2880 hidden layers. Freeze unused parameters. A2/A3 head initialization is shared and module-construction seeded; branches import identical matcher/coarse tensors from strict final 120-update P1 checkpoint while initializing fresh heads.
5. Integrate protected training CLI, distributed global query uniqueness and official annotation cross-query labels. Preserve optimizer/scheduler/RNG/loop checkpoint restore and sampler position together. Refuse pilot steps without a valid optimizer update.
6. Verify meaningful sampler uniqueness/balance/restore/RNG isolation, panel labels/order, parameter counts, gradients, parent strictness and real dynamic CLI consumption. Run targeted tests, the semantic suite, and Ruff; report any wider suite environment limits to coordinator.

Integration contract: C1 supplies benchmark v2 Episodes with full candidate inventory and `label_protocol` mapping; an additional bound JSON mapping query ID to fixed CLIP canonical ranking is consumed by C2. C4 validates parent reservation and terminal evidence and freezes all input hashes. Public model field `pilot_arm` and optional train field `pilot` carry these identities into manifest config.
