# Cleanup Proposal — /data2/hxj/project/UATVR

Audit date: 2026-08-23. **Read-only audit — nothing was deleted, moved, or modified.**
All sizes are `du -sb` actual bytes converted to GB (1 GB = 2^30), not rounded `du -sh` output.

---

## Headline finding

**No git history rewrite is required.** The entire reachable git repository — every object referenced
by every branch, tag, remote-tracking ref, reflog entry, and all five worktree indexes — totals
**40.0 MB across 2,897 objects**. The other **155.3 GB** of `.git` is unreachable garbage that a
routine `git gc` + `git lfs prune` reclaims with zero history impact.

The bloat is *not* "something large was committed". It is checkpoints that were **staged / pushed but
never landed in a surviving commit**, plus an **orphaned 76.5 GB Git-LFS cache** that nobody
mentioned and that no `.gitattributes` in any ref refers to.

---

## Summary table

| Tier | What | Reclaimable | Risk |
|---|---|---|---|
| **Tier 0** — zero-risk | garbage `tmp_pack_*`, `__pycache__`, `.pytest_cache`, `.ruff_cache` | **1.86 GB** | none |
| **Tier 1** — safe, trivial recovery | `git gc` unreachable objects, orphaned LFS cache, stale merged worktrees | **153.4 GB** | none/negligible |
| **Tier 2** — needs user confirmation | unreferenced ckpts, `deploy_qwen` weights, `cache_dir`, `research_refs`, `data/generated`, detached-worktree ckpts | **~144 GB** (+ up to 33.9 GB `artifacts/` after per-run triage) | real data loss |
| **Tier 3** — dangerous / history rewrite | — | **0 GB — NOT NEEDED, DO NOT DO** | catastrophic |
| | **Tier 0 + Tier 1 (recommended now)** | **155.3 GB** | **essentially zero** |

After Tier 0 + Tier 1, `.git` shrinks from **155.32 GB → ~0.05 GB**.

Disk context: `/data2` is 7.3T, 73% used, 1.9T free. **`/` is 916G, 98% used, only 25G free** — and
`/tmp` lives on `/`. See the `/tmp` warning in Tier 2.

---

## Evidence base

```
$ git count-objects -vH
count: 4930          size: 30.69 GiB      (loose, unreachable)
in-pack: 728         size-pack: 46.28 GiB (7 packs)
garbage: 8           size-garbage: 1.85 GiB

$ du -sh .git/*
79G  .git/objects
77G  .git/lfs        <-- not in the original problem statement
```

Reachability proof (all read-only):

```
$ git rev-list --objects --all --reflog --indexed-objects \
    | cut -d' ' -f1 | git cat-file --batch-check='%(objectsize)' \
    | awk '{s+=$1} END {print s/1048576 " MB across " NR}'
40.0 MB across 2897
```

Largest objects in the packs, and whether anything references them:

| Object | Size | Content (first bytes) | Reachable? |
|---|---|---|---|
| `d9dec28e` | 5.04 GB | JSON (`{"selection": {"candidates": [`) | **no** |
| `6c2a442e` | 4.59 GB | — | **no** |
| `4bfdafc5` | 3.61 GB | — | **no** |
| `7a43478d` | 2.82 GB | zip, `mp_rank_00_model_states` (DeepSpeed ckpt) | **no** |
| `c66b4813` | 2.20 GB | zip, `functorch/` (torch wheel/env) | **no** |
| `b8777825` | 0.86 GB | zip, `archive/data.pkl` (`torch.save`) | **no** |
| `0fcedd7b` | 0.86 GB | `torch.save` archive | **no** |
| 15× loose | 419.3 MB ea. | zip, `pytorch_model.bin/` | **no** |

Verified unreachable from refs, reflogs (including all four per-worktree reflogs), and every worktree
index. The largest object in *any* worktree index is **1,356,917 bytes (1.36 MB)**.

`.gitignore` timing: `/ckpts/` and `/deploy_qwen/` are ignored today, but the file's history
(`204a1db init commit` → … → `9aae71e`) shows those rules were tightened *after* the offending
`git add` operations — which is exactly why the blobs exist as loose objects with no commit.

---

## Tier 0 — Zero risk

**Subtotal: 1.86 GB**

### 0.1 — Eight garbage `tmp_pack_*` files — 1.85 GB

`.git/objects/pack/tmp_pack_{mK530K,xMCafs,ACD7h9,CErIlw,iBC6QP,crpTHO,IREMKv,ZqHagS}`

Dated 2026-07-10 → 2026-08-15. Sizes: 1,325,924,364 / 289,144,844 / 224,657,420 / 78,512,140 /
47,579,148 / 22,151,180 / 12 / 12 bytes. Two are 12-byte stubs.

**Evidence they are unneeded:** git itself classifies them as garbage — `git count-objects -v`
prints `warning: garbage found:` for each and reports `garbage: 8 / size-garbage: 1.85 GiB`. They
are leftovers from interrupted `gc`/`fetch`/`push` runs, have no `.idx`, and are unreadable by git.

```bash
rm -f /data2/hxj/project/UATVR/.git/objects/pack/tmp_pack_*
```
(Or simply run the Tier 1.1 `git gc`, which removes them automatically.)

### 0.2 — Python / tool caches — ~6 MB

31 `__pycache__` dirs in project source (93 including `research_refs/` and `deploy_qwen/`), plus
`.pytest_cache` (204 KB, last touched 2026-04-26) and `.ruff_cache` (84 KB, 2026-05-13).
All matched by `.gitignore:__pycache__/` and `.gitignore:.*`. Regenerated automatically.

```bash
find /data2/hxj/project/UATVR -name __pycache__ -type d -not -path '*/.git/*' -prune -exec rm -rf {} +
rm -rf /data2/hxj/project/UATVR/.pytest_cache /data2/hxj/project/UATVR/.ruff_cache
```

---

## Tier 1 — Safe, trivial or no recovery needed

**Subtotal: 153.4 GB**

### 1.1 — Unreachable git objects — 76.93 GB

`.git/objects` is 78.82 GB (30.68 GB loose across 4,930 objects + 46.29 GB in real packs). Reachable
content is 40.0 MB. Excluding the 1.85 GB already counted in Tier 0, `git gc` reclaims **76.93 GB**.

**Evidence:** the reachability proof above. Nothing is lost — every commit on `master`,
`feature/reliable-retrieval-experiment-campaign`, all `origin/*` refs, all reflogs and all worktree
HEADs is preserved by definition, since `gc` only drops what is unreachable.

```bash
cd /data2/hxj/project/UATVR && git gc --prune=now
```

Notes:
- Do **not** add `git reflog expire --expire-unreachable=now --all`. It is unnecessary (the big blobs
  are not reflog-reachable — reflogs point at commits, and no commit tree contains them) and it
  would needlessly discard your recovery history.
- This `gc` will take a while and is what previously got interrupted and produced the `tmp_pack_*`
  files. Run it in a detached session (`tmux`/`nohup`) so it cannot be killed mid-write.
- Recovery if you change your mind: nothing to recover — these objects are already invisible to git.

### 1.2 — Orphaned Git-LFS object cache — 76.50 GB

`.git/lfs/objects` — 20 objects, largest 9.29 GB, then 11× 4.64 GB.

**Evidence they are orphaned:**
- No `.gitattributes` exists at the repo root, is tracked by `git ls-files`, or appears in
  `git ls-tree -r` for `master`, `feature/reliable-retrieval-experiment-campaign`, or `origin/main`.
- `git grep -l 'version https://git-lfs.github.com'` returns **0 files** in every ref checked:
  `master`, `feature/…campaign`, `origin/main`, `origin/199`, `origin/worktree-exp-hooks`,
  `169ba95`, `21348d1`, `3559e67`.
- `git lfs ls-files` outputs nothing.

LFS filters are configured in `.git/config` (`filter.lfs.*`, and an
`lfs.https://github.com/ChihayaAnonnn/tvr.git/info/lfs.access` entry), so someone attempted to push
checkpoints through LFS to GitHub. That attempt left the local object cache behind; no tracked file
in any branch points at it.

```bash
cd /data2/hxj/project/UATVR && git lfs prune --verify-remote --verbose   # dry-run first:
cd /data2/hxj/project/UATVR && git lfs prune --dry-run
```
If `git lfs prune` refuses or is unavailable, and after confirming the checks above still hold:
```bash
rm -rf /data2/hxj/project/UATVR/.git/lfs/objects
```
**Resolving check before you run it:** `git lfs ls-files --all` should print nothing. If it prints
anything, stop and re-triage.

### 1.3 — `/tmp/uatvr-june` worktree — 0.01 GB (6.7 MB)

Detached at `3559e67`, **0 commits not in master** (`git merge-base --is-ancestor` → YES).
11 dirty entries, but `diff -rq` against the repo root shows the only differences are `.pyc` files —
every source file (`crf_ladder/`, `uncertainty_modules/`, `scripts/*.py`, `modules/crf_head.py`) is
byte-identical to what already exists at the root and in git.

Referenced once, historically, in `docs/superpowers/tables/2026-08-11_r1_and_rho.md:108`
("`老 code (/tmp/uatvr-june/)`") — a provenance note about a past measurement, not a live dependency.
It is also on the 98%-full `/` filesystem and will vanish on reboot regardless.

```bash
cd /data2/hxj/project/UATVR && git worktree remove --force /tmp/uatvr-june
```

### 1.4 — Deregister the two detached worktrees, keep their checkpoints

`/data2/hxj/project/UATVR_baseline_169ba95` (5.45 GB) and `/data2/hxj/project/UATVR_exp1_21348d1`
(5.50 GB). Both are **fully merged** into master (0 commits not in master). Their working trees are
~5 MB of code that is identical to git history; **99.9% of their size is one checkpoint each**
(Tier 2.6 below).

The *git registration* and the *code tree* are Tier 1. The checkpoints are not. So the safe Tier 1
action is to move the ckpts out first, then remove the worktrees:

```bash
# inspect first
git -C /data2/hxj/project/UATVR_baseline_169ba95 status --porcelain
# then, preserving the checkpoints:
mv /data2/hxj/project/UATVR_baseline_169ba95/ckpts/ckpt_msrvtt_20260617_baseline_pure_sim_repeat2 \
   /data2/hxj/project/UATVR/ckpts/
mv /data2/hxj/project/UATVR_exp1_21348d1/ckpts/ckpt_msrvtt_20260617_exp1_repro_repeat2 \
   /data2/hxj/project/UATVR/ckpts/
cd /data2/hxj/project/UATVR
git worktree remove --force /data2/hxj/project/UATVR_baseline_169ba95
git worktree remove --force /data2/hxj/project/UATVR_exp1_21348d1
```
Reclaims ~10 MB by itself; the 10.95 GB is Tier 2.6. Each also carries a few MB of untracked
`logs/20260617/`, `logs/causal_chain/2026061{8,9}/` and `search_results/` (664 KB) — **check those
are not unique experiment records before removing** (`ls -R` them; they are not referenced by any of
the five root markdown files).

---

## Tier 2 — Needs user confirmation

**Subtotal: ~144 GB**, plus up to 33.9 GB from `artifacts/` after per-run triage.

### 2.1 — Unreferenced checkpoints in `ckpts/` — 39.28 GB

`ckpts/` totals **87.22 GB** across 41 dirs (23 of them are <1 MB stubs, 452 KB combined).

**Load-bearing — DO NOT DELETE (47.94 GB).** Cited by PAPER.md Appendix A, findings.md, todo.md, or
`experiments/contextual-risk-v2/legacy/critical-runs.json`:

| Ckpt | Size | Cited by |
|---|---|---|
| `ckpt_msrvtt_20260813_031640_contextual_risk_a3` | 16.27 GB | `critical-runs.json`, 4 `docs/superpowers/*` files; underpins findings.md §2.1 bitwise-equivalence claim |
| `ckpt_msrvtt_20260811_194405` | 9.90 GB | `docs/superpowers/plans/2026-08-11-sap-retrieval-grounded-uncertainty.md` |
| `ckpt_msvd_20260813_034603_msvd_june_baseline` | 8.93 GB | `critical-runs.json` |
| `ckpt_msrvtt_crf_20260810_025752_crf_ladder_full` | 5.78 GB | **PAPER.md:206** "CRF donor ckpt", findings.md:71 |
| `ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg` | 3.32 GB | **PAPER.md:204** "A2-CG ckpt", findings.md:55 |
| `ckpt_msrvtt_20260811_231959_risk_sidecar_a1` | 3.31 GB | **PAPER.md:202** "A1 sidecar ckpt", findings.md:25 |
| `ckpt_msrvtt_20260617_b1only_v2_repeat1` | 0.43 GB | **PAPER.md:201** "Base retriever ckpt" — the frozen base for everything |
| `ckpt_msrvtt_20260813_214730_a3_dsl_t2_eval` | 8 KB | grep hit in docs |

**Removal candidates (39.28 GB)** — no reference in PAPER.md, findings.md, todo.md, progress.md,
task_plan.md, `research/`, `critical-runs.json`, any `docs/` file, or any `.py`/`.sh`:

| Ckpt | Size | Date |
|---|---|---|
| `ckpt_msrvtt_20260811_124611` | 6.42 GB | 2026-08-11 |
| `ckpt_msrvtt_20260810_173323` | 6.42 GB | 2026-08-10 |
| `ckpt_msrvtt_20260810_145818` | 6.42 GB | 2026-08-10 |
| `ckpt_msrvtt_20260808_170200` | 5.75 GB | 2026-08-11 |
| `ckpt_msrvtt_20260715_013131` | 4.68 GB | 2026-07-15 |
| `ckpt_msrvtt_20260813_000924_contextual_risk_a3` | 2.68 GB | 2026-08-13 (superseded by the 031640 A3 run) |
| `ckpt_msrvtt_20260808_152839` | 2.30 GB | 2026-08-08 |
| `ckpt_msrvtt_20260811_144002` | 1.98 GB | 2026-08-11 |
| `ckpt_msrvtt_20260811_115556` | 1.28 GB | 2026-08-11 |
| `ckpt_msrvtt_20260812_113951_csd_logvar_a2_cg_smoke` | 0.66 GB | smoke test |
| `ckpt_msrvtt_20260812_113344_csd_logvar_a2_cg_smoke` | 0.66 GB | smoke test |

The two `_smoke` dirs (1.32 GB) are the highest-confidence deletes — smoke tests by name.

**Check that resolves the rest:** for each candidate, confirm no `experiments/*/registry.jsonl` row
and no `artifacts/contextual-risk-v2/**/training_args.json` records it as `init_model` or output:
```bash
grep -rn "ckpt_msrvtt_20260810_173323" /data2/hxj/project/UATVR/experiments /data2/hxj/project/UATVR/artifacts
```
Only delete if that returns nothing. Command shape:
```bash
rm -rf /data2/hxj/project/UATVR/ckpts/ckpt_msrvtt_20260812_113344_csd_logvar_a2_cg_smoke \
       /data2/hxj/project/UATVR/ckpts/ckpt_msrvtt_20260812_113951_csd_logvar_a2_cg_smoke
```

### 2.2 — `deploy_qwen/` model weights and wheels — 60.05 GB of 60.10 GB

**This is a separate concern living inside the repo**: a Qwen3-VL vLLM deployment used once to
generate per-video attribute annotations.

- `models/Qwen3-VL-30B-A3B-Instruct/` — **58 GB**, 13× 4.7 GB `.safetensors`. Re-downloadable via the
  folder's own `scripts/download_model.py`.
- `torch_wheels/` — 2.2 GB, 3 `.whl` files. Re-downloadable.
- `logs/` — 29 MB of vLLM server logs.
- **`attributes/` — 55 MB — LOAD-BEARING. KEEP.** Read by `eval.sh:98-99`
  (`deploy_qwen/attributes/msrvtt/final/msrvtt_jsfusion_test_attributes.json{,l}`) and
  `scripts/build_idf_weights.py:8-10`.

Gitignored (`.gitignore:11`), 0 tracked files, real content untouched since 2026-04-01. Nothing
Python-imports `deploy_qwen`; the only two references are file-path reads of `attributes/`.

```bash
# keep attributes/ and scripts/; drop the weights and wheels
rm -rf /data2/hxj/project/UATVR/deploy_qwen/models /data2/hxj/project/UATVR/deploy_qwen/torch_wheels
```
**Gate:** only if you will not re-run attribute generation. If you might, the weights cost a
multi-hour re-download but no data loss.

### 2.3 — `cache_dir/` — 17.90 GB

HF model download cache: `LanguageBind/Video-LLaVA-7B/` (17 GB) + `hf/xet/` (1.9 GB). Gitignored
(`.gitignore:4`), 0 tracked, untouched since 2026-07-05. Fully regenerable by re-download.

The `cache_dir` string in `main_task_retrieval.py:692` is the `--cache_dir` CLI argument, not a
hardcoded dependency on this directory. The one path reference,
`scripts/build_msrvtt_hard_negatives.py:88` → `cache_dir/hard_negatives/...`, points at a subpath
that **does not exist**.

```bash
rm -rf /data2/hxj/project/UATVR/cache_dir
```
**Gate:** confirm no active/planned run uses Video-LLaVA-7B. Nothing in `research/`, PAPER.md,
findings.md or todo.md mentions it.

### 2.4 — `research_refs/` — 13.42 GB

12 cloned baseline repos (UATVR_official, InternVideo, EVA, big_vision, UCoFiA, DUQ, EagleNet,
TempMe, Video-ColBERT, GARE, umivr) + `model_weights/` **9.3 GB** (siglip2-large 3.4 GB,
internvideo2 2.7 GB, eva_clip 1.9 GB, siglip2-base 1.5 GB) + `umivr/cache_dir` 3.9 GB +
`papers/multimodal-retrieval` 25 MB of PDFs.

Gitignored (`.gitignore:19`), 0 tracked, untouched since 2026-07-12. Referenced only from five
markdown design reports under `docs/reference/` — and **those links are already broken**: they point
at `/data2/hxj/project/tvr/research_refs/...` (note `tvr`, not `UATVR`). No `.py` or `.sh` imports it.

Recommended split: drop the 13.2 GB of weights and clones, **keep `papers/` (25 MB)** — PDFs are the
part a paper draft actually needs.

```bash
rm -rf /data2/hxj/project/UATVR/research_refs/model_weights \
       /data2/hxj/project/UATVR/research_refs/umivr/cache_dir
```
**Gate:** confirm you will not re-run any baseline comparison from these repos before the deadline.

### 2.5 — `artifacts/` — 33.92 GB — TRIAGE REQUIRED, DO NOT BULK DELETE

The experiment **data plane**, deliberately paired with the tracked control plane
`experiments/contextual-risk-v2/` (documented in `AGENTS.md:4,7`). Gitignored (`.gitignore:8`).
`contextual-risk-v2/msrvtt` 25 GB, `msvd` 9.0 GB, `legacy` 108 KB.

Note: several multi-GB files carry a `.json` extension but are checkpoint blobs
(`feature_ablation/.../checkpoints/final_test.json` = 5.0 GB,
`best_contextual_risk_checkpoint.json` = 4.6 GB).

**At least one run directory is directly load-bearing** — findings.md:40 cites
`artifacts/contextual-risk-v2/msrvtt/feature_ablation/msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/checkpoints/{training_args,final_test}.json`
as the source of the PAPER.md Table 1 / fig4 numbers.

**Check that resolves it:** cross-reference every run dir against
`experiments/contextual-risk-v2/registry.jsonl` and the `archive_rules` in `study.yaml`; any run not
in the registry and not cited in findings.md is a candidate. The large `*.json` checkpoint blobs
inside *retained* runs can likely be dropped while keeping the metrics `.jsonl`/`.npy` — that alone
probably recovers most of the 33.9 GB without losing any cited number. No blanket command given
deliberately.

### 2.6 — Checkpoints inside the two detached worktrees — 10.95 GB

- `/data2/hxj/project/UATVR_baseline_169ba95/ckpts/ckpt_msrvtt_20260617_baseline_pure_sim_repeat2` — 5.45 GB
- `/data2/hxj/project/UATVR_exp1_21348d1/ckpts/ckpt_msrvtt_20260617_exp1_repro_repeat2` — 5.50 GB

Both names are **unique** — no counterpart exists in the root `ckpts/`. Neither is referenced in any
of the five root markdown files, `research/`, or `critical-runs.json`. They are June-2026
baseline/exp1 reproduction runs from an era the project has moved past.

**Gate:** these are the only copies. If the June baseline reproduction may need to be re-cited, keep
them (move to `ckpts/` per Tier 1.4). Otherwise:
```bash
rm -rf /data2/hxj/project/UATVR_baseline_169ba95 /data2/hxj/project/UATVR_exp1_21348d1
```

### 2.7 — `data/generated/` — 2.31 GB

**Not** the raw dataset. 28,600 `.mp4` CRF-ladder re-encodes (`crf30`…`crf50`) produced by the x264
recipe in `crf_ladder/`: `msrvtt_crf` 2.3 GB, `msvd_crf` 131 MB, `msrvtt_trusted_v1` 700 KB.
Gitignored via `.gitignore:13 /data/generated/`, 0 tracked.

The real datasets live **outside this repo** at `/data2/hxj/data/MSRVTT` and `/data2/hxj/data/MSVD`
(hardcoded in the run scripts) — those are untouched by any of this.

Referenced by `main_task_retrieval.py:849,1451,1563`, `eval.sh:64`,
`run_train_msrvtt_crf_bg.sh:80-81`, `scripts/evaluate_crf_ladder.py:53-54`. Regenerating costs
re-encode CPU time, not data loss — but PAPER.md Figure 3 (`fig3_crf_ladder.pdf`) depends on this
ladder. **Recommendation: keep.** Listed only for completeness; 2.31 GB is not worth the risk.

### 2.8 — `/tmp` artifacts — VOLATILE, ACTION NEEDED (this *adds* safety, does not reclaim)

`/` is **98% full with 25 GB free**, and `/tmp` is on it. Three `/tmp` paths are cited as evidence:

| Path | Size | Cited by |
|---|---|---|
| `/tmp/hybrid_ckpt/pytorch_model.bin.hybrid` | **443 MB** | **PAPER.md:207 Appendix A**, findings.md:71 |
| `/tmp/hybrid_audit/crf_audit_report.json` | small | findings.md:72 |
| `/tmp/frozen_equiv/frozen_retrieval_equivalence.json` | small | PAPER.md:53, findings.md:35 |

`/tmp/hybrid_ckpt/pytorch_model.bin.hybrid` is the **only** artifact behind PAPER.md Table 4's
headline transplant row and it has no backup in `ckpts/`. A reboot or `/tmp` sweep destroys it.

```bash
mkdir -p /data2/hxj/project/UATVR/ckpts/hybrid_ckpt_20260811
cp -a /tmp/hybrid_ckpt/. /data2/hxj/project/UATVR/ckpts/hybrid_ckpt_20260811/
cp -a /tmp/hybrid_audit /tmp/frozen_equiv /data2/hxj/project/UATVR/artifacts/  # then update PAPER.md:207
```
Do this **before** anything else in this document.

### 2.9 — Root-level clutter

**The five root markdown files.** `research/` now exists (created 2026-08-23 01:48) and its
`README.md:36` explicitly declares itself the distillation of four of them. Its README promises
`01-evidence-ledger.md` … `07-protocol-constraints.md`, but **only `00-north-star.md` and
`README.md` exist so far** — `00-north-star.md` already cites ledger entries E-08/E-11/E-12 that
cannot be resolved.

| File | Size | Modified | Recommendation |
|---|---|---|---|
| `PAPER.md` | 27 KB | 2026-08-22 | **Keep at root.** It is the manuscript, not a working note, and it is the canonical Appendix A path list. Not claimed by `research/README.md`. |
| `findings.md` | 25 KB | 2026-08-22 | **Do not move or delete until `research/01-evidence-ledger.md` exists and is verified complete.** It is the live evidence ledger and holds paths found nowhere else (`/tmp/hybrid_audit`, the cited `artifacts/` run dir, the fix1_v2 seed ckpts). Then archive to `research/archive/`. |
| `todo.md` | 6 KB | 2026-08-20 | Same gate — its §2 protocol constraints must land in `research/07-protocol-constraints.md` first. Then archive. |
| `progress.md` | 1.3 KB | 2026-08-13 | Stale single-session scratchpad, all items ticked. → `research/archive/`. |
| `task_plan.md` | 3.3 KB | 2026-08-13 | Stale plan for that same session, inconsistent with `progress.md`. → `research/archive/`. |

Because another agent is actively building `research/`, **do not move these files now** — coordinate
first. Archive rather than delete: they are a few KB and are cited by git history.

**Nine `run_train_*.sh` launchers at root.** Only one is cited in the docs
(`run_train_msrvtt_risk_sidecar_bg.sh`, PAPER.md:209,219). But launchers are cheap (1–20 KB total)
and encode exact hyperparameters for reproducibility. **Recommendation: keep all, delete none.**
Optionally move to `scripts/launchers/` — but note `PAPER.md:219` hardcodes
`bash run_train_msrvtt_risk_sidecar_bg.sh`, so any move requires editing PAPER.md.
Also note `findings.md:156` cites `run_train_msrvtt_fix1_4gpu_bg.sh`, which **does not exist at root**
— it lives only inside the campaign worktree. Another reason not to remove that worktree.

Two untracked launchers (`run_train_msvd_june_bg.sh`, `run_train_msvd_risk_sidecar_bg.sh`) and two
untracked scripts (`scripts/evaluate_sap_uncertainty.py`,
`scripts/verify_frozen_retrieval_equivalence.py`) should be **committed, not deleted** — PAPER.md:53
cites `verify_frozen_retrieval_equivalence.py` as the proof script for a headline claim.

The modified `.gitignore` (uncommitted, flips `# logs/` → `logs/`) should also be committed;
`logs/` still has 4 tracked files from before the rule, which will stay tracked.

---

## Tier 3 — Dangerous / history rewrite

**Reclaimable: 0 GB. Recommendation: DO NOT DO THIS. It is not necessary.**

The premise that motivates history surgery — "something large was committed" — is **false here**.
Total reachable history is 40.0 MB. There is nothing large *in* the history to excise. Every giant
object is already unreachable and is removed by the Tier 1 `gc`, which touches no commit, no tree,
and no SHA.

For the record, had a rewrite been needed, the blast radius would have been:

- **The repo has a live remote**: `origin git@github.com:ChihayaAnonnn/tvr.git`, with
  `origin/main`, `origin/199`, and `origin/worktree-exp-hooks` tracked locally. A rewrite changes
  every commit SHA and requires `push --force`, breaking every other clone.
- **`feature/reliable-retrieval-experiment-campaign` has 37 commits that exist nowhere else** —
  `git branch -r --contains d3a3cd9` returns empty, so the branch is **not pushed to any remote**.
  A botched rewrite loses the entire active campaign permanently.
- **findings.md:238-252 and todo.md:26-27 cite specific commit SHAs as evidence** (`eb2ee0b`,
  `c448240`, `4020fc1`, `d3a3cd9`, `266a2b6`). A rewrite invalidates every one of those citations in
  a paper draft.
- Tools would be `git filter-repo` or `bfg`; both rewrite all four worktrees' HEADs and require
  re-cloning.

If someone later insists on a rewrite anyway, the only defensible sequence is: full `cp -a` backup of
`.git` to a different filesystem → push all local branches to the remote → rewrite → verify → force
push. Given that Tier 1 already reclaims 155.3 GB, **there is no scenario in this repo where the
tradeoff makes sense.**

---

## Recommended execution order

1. **Rescue the volatile `/tmp` evidence first** (Tier 2.8). `/` is 98% full; this is the only real
   data-loss risk in the whole audit. ~443 MB copy.
2. **Commit the untracked work** — `.gitignore`, the two `scripts/*.py` (one is cited by PAPER.md:53),
   the two `run_train_msvd_*.sh`. Costs nothing, removes ambiguity from every later step.
3. **Tier 0** — caches and `tmp_pack_*` (1.86 GB). Instant, zero risk.
4. **Tier 1.2** — `git lfs prune --dry-run`, then prune (76.50 GB). Biggest single win.
5. **Tier 1.1** — `git gc --prune=now` (76.93 GB). **Run under `tmux`/`nohup`** — interruption is what
   created the `tmp_pack` mess in the first place. Verify with `git count-objects -vH` and
   `git fsck --connectivity-only` afterwards.
6. **Tier 1.3** — remove `/tmp/uatvr-june`.
7. *Stop and reassess.* At this point ~155.3 GB is back and `.git` is ~50 MB. Everything past here
   destroys real data — decide whether you still need the space.
8. **Tier 2**, in ascending risk: 2.2 `deploy_qwen/models` (60 GB, re-downloadable) → 2.3 `cache_dir`
   (17.9 GB, re-downloadable) → 2.4 `research_refs/model_weights` (13.2 GB) → 2.1 the two `_smoke`
   ckpts, then the rest after the registry grep → 2.6 detached-worktree ckpts → 2.5 `artifacts/`
   per-run triage.
9. **Tier 2.9** — root markdown reorganisation, only after `research/01`–`07` exist and are verified.

---

## Do not touch under any circumstances

- **`.worktrees/reliable-retrieval-experiment-campaign/`** (36.02 GB) — **37 commits not in master and
  not pushed to any remote.** Clean tree, HEAD `d3a3cd9`, active as of 2026-08-22. Its
  `main_task_retrieval.py`, `prob_models/contextual_retrieval_risk.py`, `modules/modeling_mulit.py`,
  `experiments/sap-failure-supervision-v{1,2}/`, and `experiments/contextual-risk-v2/summaries/`
  are the direct evidence targets of `todo.md:6,7`, `findings.md:120-156`, and
  `research/00-north-star.md:41`. Its `ckpts/` (36 GB, the `fix1_v2_s{0,1,2}`, `fix1_poc_w0p1`, and
  `sapfeat_full_sap_s0` runs) are cited in `todo.md:51` and `findings.md:134,162-164` and exist
  nowhere else. It also holds the only copy of `run_train_msrvtt_fix1_4gpu_bg.sh`.
- **The 8 load-bearing checkpoints in Tier 2.1's keep-list** (47.94 GB) — PAPER.md Appendix A.
- **`/tmp/hybrid_ckpt/pytorch_model.bin.hybrid`** — until it has been copied to `/data2`.
- **`deploy_qwen/attributes/`** (55 MB) — read by `eval.sh` and `scripts/build_idf_weights.py`.
- **`experiments/`** (116 KB) — the *tracked* control plane: `registry.jsonl`, `study.yaml`, schemas.
- **`artifacts/contextual-risk-v2/msrvtt/feature_ablation/msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/`** — source of PAPER.md Table 1 / fig4.
- **`dataloaders/splits/msrvtt_trusted_v1_seed0.json`** — the split manifest, PAPER.md:93,208.
- **`paper/figs/`** — the manuscript's five figures.
- **`logs/`** (2.8 MB) — gitignored but cited by PAPER.md:203 and findings.md:162-164,366-367. Tiny; keep.
- **`crf_ladder/`, `uncertainty_modules/`, `query_models/`, `prob_models/`** — live tracked source,
  imported by `experiment_tracking.py:660` and `modules/modeling_mulit.py:21`. (Note: findings.md:209
  claims `query_models/module_sap.py` was deleted, but it still exists on disk and is imported by
  `scripts/evaluate_sap_uncertainty.py:183` — resolve that inconsistency before touching it.)
- **`related_work/dab_eccv2026/`** (2.9 MB) — cited by findings.md:221 and
  `research/00-north-star.md:6,33,35,38`.
- **`/data2/hxj/data/MSRVTT`, `/data2/hxj/data/MSVD`** — the real datasets, outside this repo.
- **`git reflog`** — do not expire it. It is your undo history and it costs nothing.
