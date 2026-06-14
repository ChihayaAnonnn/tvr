# CLAUDE.md

This file provides quick command reference. For project architecture, design decisions, and workspace facts, see [AGENTS.md](AGENTS.md).

## Commands

### Training
```bash
bash train_msrvtt.sh       # MSRVTT
bash train_msvd.sh         # MSVD

# Background (nohup, logs to logs/)
bash run_train_msrvtt_bg.sh
bash run_train_msvd_bg.sh

# With uncertainty mode
UNCERTAINTY_MODE=nig_mil bash train_msrvtt.sh
```

### Evaluation
```bash
# Basic eval
INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> bash eval.sh

# With options
INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> \
  UNCERTAINTY_MODE=nig_mil \
  USE_ADA_NORM=1 \
  bash eval.sh
```

Key eval env vars: `INIT_MODEL` (required), `DATATYPE` (msrvtt|msvd), `UNCERTAINTY_MODE` (none|nig_mil), `USE_ADA_NORM` (0|1), `USE_ATTRIBUTES` (0|1).

### Tests
```bash
pytest                                              # all tests
pytest tests/test_modeling_mulit_losses.py -k test_name  # single test
```

### Linting
```bash
ruff check .              # check
ruff check --fix .        # auto-fix
```

### Formatting
```bash
ruff format .
```

### Hyperparameter Search
```bash
# Bayesian optimization, 2 parallel trials
python hyperparam_search.py --n_trials 30 --gpus 1,2 --parallel 2

# Grid search
python hyperparam_search.py --mode grid --grid_params w_evidential,w_neg_reg --gpus 1,2 --parallel 2

# View existing results
python hyperparam_search.py --report_only
```

## Architecture

**Data flow:** Video frames → CLIP ViT-B/16 → SpatialEnhancer → SemanticAnchorProbing (SAP)
→ Dirichlet modal probability aggregation → video probabilistic embedding (μ, σ).
Text → CLIP → PIENet → text probabilistic embedding. Similarity computed in
probabilistic space; trained with CrossEn + MIL + Evidential NLL + orthogonality reg.

**Key files:** `main_task_retrieval.py` (entry), `modules/modeling_mulit.py` (model),
`query_models/module_sap.py` (SAP core), `prob_models/` (uncertainty heads & losses).
See [AGENTS.md](AGENTS.md) for detailed design decisions.

## Project Rules

Code rules in `.claude/rules/` (auto-loaded by Claude Code):
- `code-style.md` — Python naming, formatting; **all comments must be in Chinese**
- `engineering.md` — patch-first editing, no deletions, no renames
- `persona.md` — concise answers, senior engineer tone
- `architecture.md` — follow existing structure, no new layers

## Known Debt (quick reference)

- `modeling.py` is legacy; `modeling_mulit.py` is the active model (note typo "mulit").
- `tmp/` contains throwaway experiment code — ignore it.
- `prob_models/probemb.py` PCME losses are not used by the active model.

