# Claude Code Handoff

## Goal

Complete the validation-gated reliability campaign proving that A3 predicts retrieval failure risk without changing retrieval ranks, then freeze the configuration and run the sealed MSRVTT/MSVD final tests once.

## Workspace

- Worktree: `/data2/hxj/project/UATVR/.worktrees/reliable-retrieval-experiment-campaign`
- Branch: `feature/reliable-retrieval-experiment-campaign`
- Current HEAD: `3135a5a8c31dfb020ed18b7e06b844fb8124479a`
- Main canonical artifacts: `/data2/hxj/project/UATVR/artifacts/contextual-risk-v2`
- Do not work in the main checkout and do not touch its user changes/untracked files.

## Source of truth

- Plan: `docs/superpowers/plans/2026-08-14-reliable-retrieval-experiment-campaign.md`
- Design: `docs/superpowers/specs/2026-08-14-reliable-retrieval-experiment-campaign-design.md`
- Progress ledger: `/data2/hxj/project/UATVR/.git/worktrees/reliable-retrieval-experiment-campaign/sdd/progress.md`
- Archive rules: `AGENTS.md` and `docs/agent/experiment-artifacts.md`

## Completed

- Archive/Stage 0 is integrated on `master` and independently qualified.
- Task 1: risk baselines + paired bootstrap, commit `3ace19e`, independent review CLEAN.
- Task 2: fixed-capacity feature/loss/K ablations and real committed-manifest runtime binding, commits `be4b2cf` + `27d4016`, independent review CLEAN.
- Task 3 implementation: commit `3135a5a`; full suite `833 passed, 1 skipped`; **not independently reviewed yet**.
- No scientific GPU run or JSFUSION/final-test access has occurred on this branch.

## Immediate next step

Review Task 3 before any experiment:

- Brief: `/data2/hxj/project/UATVR/.git/worktrees/reliable-retrieval-experiment-campaign/sdd/task-3-brief.md`
- Report: `/data2/hxj/project/UATVR/.git/worktrees/reliable-retrieval-experiment-campaign/sdd/task-3-report.md`
- Diff package: `/data2/hxj/project/UATVR/.git/worktrees/reliable-retrieval-experiment-campaign/sdd/review-27d4016..3135a5a.diff`

Pay special attention to absolute worktree paths in phase specs, canonical artifact-root resolution, phase-batched reservations, validation-only gates, test authorization, and whether the Stage 1 baseline run unnecessarily trains A3 instead of consuming frozen evidence.

## Unresolved decision

`study.yaml` uses `bce_only/full_loss`, while the approved plan, launcher, and phase specs use `bce/full`. Recommended resolution before reservation: change the study names to `bce`, `bce_fit`, `bce_rank`, `full`, preserve historical Stage 0 SHA records, then regenerate/revalidate affected phase hashes. The user was asked but had not answered before handoff.

## Hard rules

- Validation only for all selection/gates; historical test numbers cannot select configurations.
- Terminal runs are immutable. Infrastructure retry requires explicit human authorization and a new unique successor with `parent_run_ids` and `selection_lineage`.
- Reserve and commit an entire phase before launch; finalize the entire phase in one terminal commit.
- One four-GPU training at a time on devices `0,2,3,4`.
- Never write new outputs under legacy `logs/` or `ckpts/`.
- Do not authorize test until Stages 4 and 5 pass and the final configuration is frozen.
- Stage 5/7 MSVD launch must remain fail-closed until Task 8 provides and qualifies the artifact-aware MSVD launcher.

## Useful commands

```bash
cd /data2/hxj/project/UATVR/.worktrees/reliable-retrieval-experiment-campaign
git status --short --branch
git log --oneline -6
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_reliability_campaign.py tests/test_manage_experiment_run.py
```

After Task 3 review is CLEAN, update the SDD ledger and continue Task 4 (Stage 1 baselines). Stop immediately if its frozen validation gate fails.
