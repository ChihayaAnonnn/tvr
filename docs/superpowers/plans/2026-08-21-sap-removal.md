# Plan: remove SAP from the codebase

Status: **approved 2026-08-21, execution delegated**
Falsification record: `experiments/sap-contextual-uncertainty-v1/study.yaml` → `outcome:`
Date: 2026-08-21

## Why

SAP is falsified (see `outcome.findings` F1–F4). It is now dead weight that runs on
every forward pass, widens the A3 feature contract, and occupies ~500 lines plus a
CLI surface that future work has to keep reasoning about.

## The one thing that makes this non-trivial

SAP is in the **init commit** (`204a1db`). It is not something this campaign added —
it is part of the inherited "June UATVR arch" that `--use_uatvr_full` restores, and
`modules/modeling_mulit.UATVR` is the model class every A3/CSD experiment runs on.

Concretely that means:

1. **Every existing checkpoint contains `sap.*` keys.** Including the canonical A1
   checkpoint that all A3 studies initialise from.
2. Two different load paths, with different tolerance:
   - `--init_model` → `modules/until_module.py:100 init_preweight` → **tolerant**,
     logs missing/unexpected keys, does not raise. Safe.
   - `main_task_retrieval.py:6329 model.load_state_dict(checkpoint)` (final test from
     a selected checkpoint) → **strict by default**. Would raise on stale `sap.*`.
3. `prob_models/contextual_retrieval_risk.py` has a hard consistency check
   (line ~278): `use_sap_uncertainty` must be true *exactly* for the feature sets in
   `SAP_CONTEXTUAL_FEATURE_SETS`. Removing one side without the other raises at
   model construction.

So the ordering below is chosen to keep the tree green at every step and to never
leave a state where an old checkpoint silently loads wrong.

## Footprint

Verified by `grep -rlniE "\bsap\b|sap_|_sap|semantic_anchor"`. `dataloaders/VOCAB.json`
is a false positive (the English word "sap") — leave it alone.

### Delete outright
| Path | Lines | Note |
|---|---|---|
| `query_models/module_sap.py` | 167 | `SemanticAnchorProbing`, `EvidentialUncertaintyHead`. Only importer is `modeling_mulit.py:21` |
| `tests/test_sap_failure_bce_loss.py` | 56 | tests a confirmed no-op |
| `run_train_msrvtt_sap_features_bg.sh` | — | launcher for this study only |
| `run_train_msrvtt_fix1_2gpu_bg.sh` | — | "Fix 1" = `w_sap_failure_bce`, a no-op |
| `run_train_msrvtt_fix1_4gpu_bg.sh` | — | same |

### Edit
| Path | Hits | What |
|---|---|---|
| `modules/modeling_mulit.py` | 47 | the real work — see phase 2 |
| `main_task_retrieval.py` | 50 | CLI flag, feature-set choices, `_sap_uncertainty_direct_auroc` + its evidence plumbing |
| `prob_models/contextual_retrieval_risk.py` | 47 | `SAP_UNCERTAINTY_BLOCK`, `sap_uncertainty_descriptor`, `use_sap_uncertainty`, the two SAP feature sets |
| `experiment_tracking.py` | 5 | the duplicate registry entries added this session |
| `tests/test_contextual_retrieval_risk.py` | 36 | |
| `tests/test_contextual_retrieval_risk_final_fixes.py` | 10 | |
| `tests/test_contextual_retrieval_risk_integration.py` | 1 | |

### Keep, do not touch
- `experiments/sap-*/study.yaml` (×3) — these are the **experimental record**, including
  the falsification. Deleting them destroys the evidence. They are control-plane YAML,
  not code; nothing imports them at runtime.
- `docs/superpowers/plans/*`, `docs/superpowers/reports/2026-08-11_sap_collapse_*`,
  `docs/superpowers/specs/*` — historical design record.
- `experiments/contextual-risk-v2/study.yaml` (1 incidental mention).

### Checkpoints
- `ckpts/ckpt_msrvtt_20260821_164109_sapfeat_full_sap_s0` — 4.8 GB, from the aborted arm.
- `ckpts/ckpt_msrvtt_20260821_163757_sapfeat_full_sap_s0` — 8 KB, from the fail-closed launch.

**DECIDED 2026-08-21: KEEP BOTH. Do not delete either directory.** The user chose to
retain them. Deleting anything under `ckpts/` is out of scope for this plan.

Note that the only unique information in the 4.8 GB directory has already been
extracted: the per-block weight-norm analysis of
`contextual_risk_sidecar.network.0.weight`, recorded as finding **F5** in
`study.yaml`. Nothing in the phases below needs to read these directories.

## Phases

Each phase ends with `pytest -q` green. Commit per phase so any step is revertible.

### Phase 0 — safety net (no code change)
1. Record the exact `sap.*` key list from the canonical A1 checkpoint into
   `docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`. This is the
   record of what the weights used to contain; without it a future reader cannot tell
   whether an old checkpoint is pre- or post-removal.
2. Confirm which checkpoints are reachable via the strict `load_state_dict` path at
   `main_task_retrieval.py:6329`.
3. Snapshot the current full-suite baseline (`745 passed, 26 skipped` last clean run)
   so phase-by-phase deltas are unambiguous.

### Phase 1 — retire the no-op loss
Smallest independent change, zero coupling.
- `main_task_retrieval.py`: drop the `--w_sap_failure_bce` argument (line ~1007) and
  its entry in the manifest key list (line ~1644).
- `modules/modeling_mulit.py`: drop `self.w_sap_failure_bce` (354), `_sap_failure_bce_loss`
  (1349), and the five `sap_failure_bce_loss` plumbing sites (755, 768–769, 1110–1116,
  1141/1163/1178, 1305).
- Delete `tests/test_sap_failure_bce_loss.py`.

Nothing else depends on this. Suite must stay green.

### Phase 2 — unwire SAP from the A3 sidecar
This is the coupled part; do the two sides in one commit or construction will raise.
- `prob_models/contextual_retrieval_risk.py`: remove `SAP_UNCERTAINTY_BLOCK`,
  `SAP_UNCERTAINTY_DIM/_SLOTS/_FIELDS`, `sap_uncertainty_descriptor`, the
  `use_sap_uncertainty` constructor arg and its consistency assertion, `_sap_uncertainty_block`,
  and the `full_sap` / `sap_only` entries in `CONTEXTUAL_FEATURE_SETS`.
- `experiment_tracking.py`: remove the mirrored `full_sap` / `sap_only` entries. The
  drift-guard test added this session
  (`test_manifest_feature_set_registry_matches_the_sidecar`) will catch it if only one
  side is edited — **keep that test, it is the reason this phase is safe**.
- `main_task_retrieval.py`: remove `"full_sap", "sap_only"` from the feature-set choices
  (1065), `_sap_uncertainty_direct_auroc` (3908) and its call sites (4755–4763), and the
  `uses_sap` branches in the evidence writer (3967–4102, 4654–4656, 4955, 5035).
- `modules/modeling_mulit.py`: remove the `use_sap_uncertainty=` wiring at 464 and the
  `sap_video_uncertainty` construction/passing at 1087, 1125–1126, 1325.
- Update the three `tests/test_contextual_retrieval_risk*.py` files.

After this the sidecar is back to the five pre-SAP feature sets, which is exactly what
`scripts/manage_experiment_run.py:REGISTERED_A3_FEATURE_SETS` already expects — so the
archive-plane gap recorded in `known_limits` closes for free.

### Phase 3 — remove the module
- `modules/modeling_mulit.py`: remove the import (21), `self.sap = SemanticAnchorProbing(...)`
  (325), `num_anchors` (366), the `sap_out` block (1044–1054), and the `anchor_ent`
  normalisation at 1231 (**check what consumes `anchor_ent` before deleting — it may feed
  a CRF/diagnostic path that is not SAP-specific**).
- Delete `query_models/module_sap.py`.
- Update the help text at `main_task_retrieval.py:903` — it currently advertises
  "SAP + spatial_enhancer + ada_norm text".

### Phase 4 — launchers and loose ends
- Delete the three launcher scripts listed above.
- Grep again; the only remaining hits should be `docs/`, `experiments/*/study.yaml`,
  and `VOCAB.json`.
- Full suite; then one **eval-only smoke run** from the canonical A1 checkpoint to prove
  a pre-removal checkpoint still loads through `init_preweight` and reproduces
  T2V R@1 75.9 / V2T R@1 93.0 and A1 V2T failure AUROC 0.7294. This is the real
  acceptance test — the unit suite cannot catch a checkpoint-compat regression.

### Phase 5 — record
- Append to `findings.md`: SAP removed, pointer to the `outcome:` block.
- Update `PAPER.md` §2.3 if it describes SAP as live (it is on the known-open list).
- Update the memory file `project_direction_shift.md` to note SAP is now *removed*,
  not merely *abandoned*.

## Risks

| Risk | Mitigation |
|---|---|
| Old checkpoint fails strict `load_state_dict` at 6329 | Phase 0 step 2 enumerates which checkpoints reach that path; phase 4 smoke run proves the `init_preweight` path |
| `anchor_ent` (1231) is used by something non-SAP | Explicitly flagged in phase 3; trace consumers before deleting |
| One-sided edit of the duplicated feature-set registry | The drift-guard test added this session catches it |
| Deleting the study YAMLs destroys the falsification evidence | Explicitly out of scope — keep all three |
| Removing SAP changes A3 numbers | It must not: F1 shows the block contributes nothing. If phase-4 numbers move, that itself is a finding and the phase should be halted, not patched |

## Estimate

Phases 1–4 are mechanical: roughly 200 edited lines across 7 files plus ~230 lines of
deletions. The gating cost is the phase-4 eval smoke run (~5 min on one GPU), not the
editing.

## Decisions already made — do not revisit

1. **Checkpoints: keep both.** See the Checkpoints section above.
2. **Study YAMLs: keep all three.** They are the experimental record.
3. **`docs/superpowers/**`: keep everything.** Historical design record.
4. The weight-norm diagnostic that motivated keeping the 4.8 GB checkpoint has already
   been run; its result is finding F5 in
   `experiments/sap-contextual-uncertainty-v1/study.yaml`.

## Stop conditions

Halt and report rather than working around, if any of these occur:

- Phase-4 eval numbers differ from T2V R@1 75.9 / V2T R@1 93.0 / A1 V2T failure AUROC
  0.7294. Finding F1 predicts they must not move. If they move, that is a new finding
  and the removal is not neutral — stop.
- `anchor_ent` (`modules/modeling_mulit.py:1231`) turns out to have a non-SAP consumer
  that cannot be cleanly separated.
- Any test failure that is not obviously caused by the edit just made.
- Any change that would require touching `main_task_retrieval.py:6329`'s strict
  `load_state_dict` call to make old checkpoints load. Loosening strictness there to
  paper over the removal is not acceptable.
