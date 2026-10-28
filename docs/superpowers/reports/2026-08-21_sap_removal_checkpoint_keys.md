# SAP removal — pre-removal checkpoint key record

Date: 2026-08-21
Plan: `docs/superpowers/plans/2026-08-21-sap-removal.md` (phase 0)

## Why this file exists

SAP shipped in the init commit (`204a1db`), so **every checkpoint produced before the
removal contains `sap.*` weights**. After the removal those keys can no longer be
constructed by the model. This file records exactly what they were, so a future reader
can tell a pre-removal checkpoint from a post-removal one by inspection:

```python
import torch
sd = torch.load(path, map_location="cpu")
pre_removal = any(k.startswith("sap.") for k in sd)
```

## Reference checkpoint

Canonical A1 checkpoint, the one every A3/CSD study initialises from:

`ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`

- total state-dict keys: **482**
- `sap.*` keys: **47**
- `sap.*` parameters: **8,943,361**

## The 47 keys

| Key | Shape | Params |
|---|---|---|
| `sap.anchor_tokens` | (16, 512) | 8192 |
| `sap.decoder.layers.0.self_attn.in_proj_weight` | (1536, 512) | 786432 |
| `sap.decoder.layers.0.self_attn.in_proj_bias` | (1536,) | 1536 |
| `sap.decoder.layers.0.self_attn.out_proj.weight` | (512, 512) | 262144 |
| `sap.decoder.layers.0.self_attn.out_proj.bias` | (512,) | 512 |
| `sap.decoder.layers.0.multihead_attn.in_proj_weight` | (1536, 512) | 786432 |
| `sap.decoder.layers.0.multihead_attn.in_proj_bias` | (1536,) | 1536 |
| `sap.decoder.layers.0.multihead_attn.out_proj.weight` | (512, 512) | 262144 |
| `sap.decoder.layers.0.multihead_attn.out_proj.bias` | (512,) | 512 |
| `sap.decoder.layers.0.linear1.weight` | (2048, 512) | 1048576 |
| `sap.decoder.layers.0.linear1.bias` | (2048,) | 2048 |
| `sap.decoder.layers.0.linear2.weight` | (512, 2048) | 1048576 |
| `sap.decoder.layers.0.linear2.bias` | (512,) | 512 |
| `sap.decoder.layers.0.norm1.weight` | (512,) | 512 |
| `sap.decoder.layers.0.norm1.bias` | (512,) | 512 |
| `sap.decoder.layers.0.norm2.weight` | (512,) | 512 |
| `sap.decoder.layers.0.norm2.bias` | (512,) | 512 |
| `sap.decoder.layers.0.norm3.weight` | (512,) | 512 |
| `sap.decoder.layers.0.norm3.bias` | (512,) | 512 |
| `sap.decoder.layers.1.self_attn.in_proj_weight` | (1536, 512) | 786432 |
| `sap.decoder.layers.1.self_attn.in_proj_bias` | (1536,) | 1536 |
| `sap.decoder.layers.1.self_attn.out_proj.weight` | (512, 512) | 262144 |
| `sap.decoder.layers.1.self_attn.out_proj.bias` | (512,) | 512 |
| `sap.decoder.layers.1.multihead_attn.in_proj_weight` | (1536, 512) | 786432 |
| `sap.decoder.layers.1.multihead_attn.in_proj_bias` | (1536,) | 1536 |
| `sap.decoder.layers.1.multihead_attn.out_proj.weight` | (512, 512) | 262144 |
| `sap.decoder.layers.1.multihead_attn.out_proj.bias` | (512,) | 512 |
| `sap.decoder.layers.1.linear1.weight` | (2048, 512) | 1048576 |
| `sap.decoder.layers.1.linear1.bias` | (2048,) | 2048 |
| `sap.decoder.layers.1.linear2.weight` | (512, 2048) | 1048576 |
| `sap.decoder.layers.1.linear2.bias` | (512,) | 512 |
| `sap.decoder.layers.1.norm1.weight` | (512,) | 512 |
| `sap.decoder.layers.1.norm1.bias` | (512,) | 512 |
| `sap.decoder.layers.1.norm2.weight` | (512,) | 512 |
| `sap.decoder.layers.1.norm2.bias` | (512,) | 512 |
| `sap.decoder.layers.1.norm3.weight` | (512,) | 512 |
| `sap.decoder.layers.1.norm3.bias` | (512,) | 512 |
| `sap.norm.weight` | (512,) | 512 |
| `sap.norm.bias` | (512,) | 512 |
| `sap.evidential_head.dirichlet_layer.weight` | (1, 512) | 512 |
| `sap.evidential_head.dirichlet_layer.bias` | (1,) | 1 |
| `sap.anchor_proj.weight` | (512, 512) | 262144 |
| `sap.anchor_proj.bias` | (512,) | 512 |
| `sap.logsigma_head.0.weight` | (256, 512) | 131072 |
| `sap.logsigma_head.0.bias` | (256,) | 256 |
| `sap.logsigma_head.2.weight` | (512, 256) | 131072 |
| `sap.logsigma_head.2.bias` | (512,) | 512 |

## Which load paths see these keys (phase 0 step 2)

Two paths load a full state dict into the model:

1. **`--init_model` → `modules/until_module.py:init_preweight`** — tolerant. Logs
   missing/unexpected keys and continues. This is the path every pre-removal checkpoint
   takes. After the removal the 47 `sap.*` keys become *unexpected* and are ignored.

2. **`main_task_retrieval.py:6326` and `:6357` → `model.load_state_dict(checkpoint)`** —
   strict (default `strict=True`). Would raise on stale `sap.*` keys.

**Finding: no pre-removal checkpoint can reach the strict path.** Both strict call sites
load only a checkpoint that the *current process* selected and wrote during this same
run:

- `:6326` goes through `prob_models/uncertainty_protocol.py:validate_selected_checkpoint`,
  which rejects any candidate whose `recorded_path.parent` is not
  `Path(output_dir).resolve()` ("selected {method} model is outside the output
  directory"), and additionally re-hashes the file and requires the SHA-256 to match the
  value recorded at save time.
- `:6357` iterates `selection_payload[direction]["checkpoint"]`, which is likewise
  populated from this run's per-epoch validation candidates.

So the strict path only ever sees checkpoints written by the *same* code revision that is
loading them. Risk 1 in the plan's risk table is closed structurally; no change to
`:6326` is needed, and none was made.

## Baseline suite

`823 passed, 26 skipped` at commit `67fba49` (pre-removal baseline).

Note: the plan and the delegation brief both quote `745 passed, 26 skipped`. That figure
is stale — it predates the SAP contextual-uncertainty study's own test additions. `823`
is the correct pre-removal baseline, measured immediately before phase 1.
