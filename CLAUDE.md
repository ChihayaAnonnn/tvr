# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UATVR** (Uncertainty-Aware Text-Video Retrieval) is a deep learning research project for cross-modal video-text retrieval built on CLIP ViT-B/16. The current mainline is **Work 1**: Semantic Anchor Probabilistic Embedding (SAP), which decomposes video frame features into learnable semantic anchors with per-anchor uncertainty estimates. **Work 2** (structured attribute enhancement via Qwen3-VL) has data pipelines ready but the dual-view fusion in the main model is partially integrated.

Datasets: MSRVTT (`/data2/hxj/data/MSRVTT`), MSVD (`/data2/hxj/data/MSVD`).

## Commands

### Training
```bash
# MSRVTT (single GPU)
bash train_msrvtt.sh

# MSVD (4-GPU DDP)
bash train_msvd.sh
```

### Evaluation
```bash
# Basic eval (requires INIT_MODEL env var)
INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> bash eval.sh

# NIG-MIL mode
INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> UNCERTAINTY_MODE=nig_mil bash eval.sh

# With attributes and branch mode
INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
  USE_ATTRIBUTES=1 \
  EVAL_BRANCH_MODE=query_only \
  bash eval.sh
```

Eval env vars: `INIT_MODEL` (required), `DATATYPE` (msrvtt|msvd), `EVAL_BRANCH_MODE` (default|base_only|query_only), `USE_ATTRIBUTES` (0|1), `ATTR_PATH`, `FUSION_MODE` (prob_mos|logits_linear), `ROPE_MODE` (none|2d|3d), `USE_ADA_NORM` (0|1), `UNCERTAINTY_MODE` (none|nig_mil).

### Tests
```bash
pytest                                    # all tests
pytest tests/test_modeling_mulit_losses.py -k test_name  # single test
```

### Linting
```bash
ruff check .                              # check
ruff check --fix .                        # auto-fix
```

## Architecture

### Active Pipeline (Work 1: SAP)

```
Video Frames → CLIP Visual → SpatialEnhancer(RALA) → SAP(16 anchors)
                                                          ├─ Dirichlet modal_probs → WTI aggregation → retrieval score
                                                          ├─ gamma, logsigma → MIL sampling (saturated)
                                                          └─ NIG(ν,α,β) → epistemic uncertainty (collapsed)

Text         → CLIP Text → mean_pooling → PIENet(text) → WTI
                                                          └─→ MoE Fusion → Final Score
```

Note: Video-side PIENet was removed; SAP's Dirichlet aggregation replaces it. The uncertainty branch (NIG/MIL) is structurally broken — `logsig_v` collapses to floor within 200 steps, MIL_loss saturates at 0.16 after Epoch 1. Current best config (`w_evidential=0, w_neg_reg=0`) disables the harmful losses. The model effectively runs as pure contrastive learning (sim_loss only). See STATUS.md for root cause analysis.

### Key Integration Points

- **`main_task_retrieval.py`** — Training/eval entry point. Invoked via `torchrun`. Contains argparse, distributed init, training loop, and eval loop.
- **`modules/modeling_mulit.py` (UATVR class)** — Central model class (~1018 lines). Inherits `CLIP4ClipPreTrainedModel`. Contains CLIP encoder setup, SAP/text-side PIENet/uncertainty head instantiation, MoE fusion MLP, `get_similarity_logits()` (scoring), and `forward()` (loss computation). Note the filename typo "mulit" — this is historical debt, not a bug.
- **`query_models/module_sap.py` (SemanticAnchorProbing + EvidentialUncertaintyHead)** — Core Work 1 module. Learnable anchor tokens + TransformerDecoder probe frame features. Dirichlet modal probability aggregation. Note: EvidentialUncertaintyHead (NIG dual-layer uncertainty) is structurally broken — see STATUS.md. The active retrieval path only uses Dirichlet modal_probs for aggregation.
- **`prob_models/uncertainty_module.py`** — Text-side uncertainty heads (attention/GRU/Mamba) and `UncertaintyAdaNorm`. Note: `EvidentialUncertaintyHead` in this file is dead code; the active one lives in `module_sap.py` (also effectively dead).
- **`prob_models/probemb.py`** — PCME-style probabilistic losses (`MCSoftContrastiveLoss` etc.). Currently NOT used by the active model.
- **`modules/spatial_enhancer.py`** — RALA (Rotary Linear Attention) for 2D/3D spatial enhancement.
- **`dataloaders/data_dataloaders.py`** — Registry pattern (`DATALOADER_DICT`) for MSRVTT/MSVD. Supports optional attribute loading via `--use_attributes`.

### Loss Terms (combined in `forward()`)

| Loss | Flag | Current | Status |
|------|------|---------|--------|
| sim_loss | — | 1.0 | ✅ 唯一驱动 R@1 的 loss |
| MIL_loss | `--w_mil` | 1e-2 | ⚠️ Epoch 1 后饱和，退化为 vanilla InfoNCE |
| orth_loss | `--w_orth` | 0.1 | ✅ Epoch 1 自行消解 |
| uncertainty_reg_loss | `--w_uncertainty_reg` | 1e-3 | ❌ detached，无梯度，纯监控 |
| evidential_loss | `--w_evidential` | **0** | ❌ 已关闭（有害，干扰语义表征） |
| neg_reg_loss | `--w_neg_reg` | **0** | ❌ 已关闭（有害） |

Other weight flags: `--w_query_sim` (default 0.5, query-side similarity).

### Work 2 (Attribute Pipeline)

`deploy_qwen/` uses Qwen3-VL-30B via vLLM to generate structured attributes (entities, actions, scene, text/OCR) offline. Results in `deploy_qwen/attributes/{msrvtt,msvd}/`. See `deploy_qwen/README.md` for details. Data loading supports `--use_attributes`, but the dual-view fusion path in the model is not active.

## Code Conventions

- **Comments**: All code comments must be in Chinese. Technical terms/API names may stay in English.
- **Style**: Google Python Style Guide. 4-space indent, 120-char line length, type annotations on public functions, Google-style docstrings.
- **Naming**: `snake_case` for functions/modules, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants.
- **Editing**: Minimal, patch-style edits. Never rewrite full files unless explicitly requested. Do not refactor or rename unnecessarily.
- **Imports**: One module per line, stdlib → third-party → local ordering.
- **Functions**: Max ~50 lines per function. Complex logic should be split.

## Experiment Management

Shell scripts use env vars for configuration: `CUDA_VISIBLE_DEVICES`, `MASTER_PORT`, `RUN_ID`, `OUTPUT_DIR`. Checkpoints go to `ckpts/`, logs to `logs/`. Diagnostic TSV logs (gate scores, MoE weights, causal chain) are written during training.

## Known Debt

- `modeling.py` is the legacy model — ignore it. `modeling_mulit.py` is the active one.
- Work 2 fusion code in `modeling_mulit.py` is partially commented out. The stable path is Work 1 (SAP + probabilistic embedding).
- `tmp/` contains throwaway experiment code.
