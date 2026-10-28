# MSRVTT JSFUSION 1k final evaluation — 2026-10-12

All six frozen evaluations completed successfully. The gallery contains all 1,000 official JSFUSION videos and 1,000 paired queries. Both methods use CLIP ViT-B/32, FP32 and raw retrieval scores (DSL=false). Checkpoints were selected using validation only, before test authorization; no test-based tuning or reselection occurred.

## Results

R@1 is reported in percent; ± denotes sample standard deviation across seeds 713, 714 and 715, not a confidence interval. Paper values are the ViT-B/32 rows without ‡ in UATVR Table 6 (`related_work/uatvr_iccv2023/paper.txt`, lines 406–414).

| Method | Our T→V R@1 | Paper T→V R@1 | Difference, pp | Our V→T R@1 | Paper V→T R@1 | Difference, pp |
|---|---:|---:|---:|---:|---:|---:|
| UATVR | 42.40 ± 0.44 | 47.5 | -5.10 | 42.73 ± 0.50 | 46.9 | -4.17 |
| CLIP4Clip | 39.60 ± 0.10 | 44.5 | -4.90 | 41.33 ± 0.12 | 42.7 | -1.37 |

| Method | Seed | Selected epoch | T→V R@1 | V→T R@1 |
|---|---:|---:|---:|---:|
| UATVR | 713 | 5 | 42.9 | 42.2 |
| UATVR | 714 | 2 | 42.2 | 42.8 |
| UATVR | 715 | 2 | 42.1 | 43.2 |
| CLIP4Clip | 713 | 5 | 39.6 | 41.2 |
| CLIP4Clip | 714 | 5 | 39.5 | 41.4 |
| CLIP4Clip | 715 | 5 | 39.7 | 41.4 |

The local UATVR baseline exceeds the local CLIP4Clip baseline by 2.80 pp T→V and 1.40 pp V→T. Neither reproduces its paper R@1. Training used the approved 8,246-video subset, retaining the 500-video validation set and removing its training source relatives; the paper training protocol differs. These differences do not establish the cause of the performance gap. This evaluation concerns baseline models and does not support the new evidence-conditioned method's scientific hypothesis.

## Confirmed source limitation

The read-only audit in `jsfusion-source-audit-2026-10-12.json`, independently reproduced by the final reviewer, confirms zero train/test video-ID overlap but 321 shared original YouTube source groups: 390 training videos and 350 test videos (35% of test). There are 54 cross-partition temporally overlapping pairs. Zero exact file-SHA duplicate groups does not demonstrate absence of visual near duplicates.

These are complete official 1k benchmark results with confirmed source and partial temporal dependence; they must not be interpreted as source-isolated generalization. No sample filtering, retraining, checkpoint changes or repeated test selection was performed after this discovery. Other studies and MSVD test remain sealed; this exposed test cannot become a hyperparameter/model-selection set.

## Verification and archive

- Targeted behavior tests passed; affected regression suite: **227 passed in 167.53 s** (one upstream TypedStorage warning). Independent targeted review verification: **17 passed in 8.71 s**. Exact commands and implementation review are recorded in `../FINAL_TEST_PROTOCOL.md`.
- Fresh matching-source, formal-capacity two-rank Q0 passed all nine checks, including gradients, numerical behavior and exact model/optimizer recovery. Run: `synthetic-baseline-full-data-q0-seed713-20261012T142922Z-878dfad2c1f6`.
- Final review found no remaining implementation blockers; all six run authorizations, frozen epochs and DSL=false were checked independently. Source counts were independently reproduced.
- All six metric file hashes and external array reference hashes were verified. R@1 was independently recomputed from each 1,000-element rank array and matches the archived metrics.
- `scripts/manage_experiment_run.py::verify_run` passed for all six identities in an isolated, clean checkout of the exact reservation commit `96bf024`; current committed terminal records and report aggregates were checked separately. The tool intentionally rejects current HEAD after report/summary additions because only registry changes are permitted relative to the frozen source. No guard was relaxed and no archived worker was resumed.
- Final evaluation cost: **1.464920 GPU-hours**, charged as elapsed controller time × two GPUs. Fresh Q0: **0.101456 GPU-hours**; combined this test campaign: **1.566376 GPU-hours**. These charges include initialization and decoding, not solely GPU kernel time; prior training costs are excluded.
- Data: `/data2/hxj/project/UATVR/artifacts/full-data-retrieval-v1/msrvtt/final/<run-id>/`; Q0 under the same study's `synthetic/baseline/` tree. `jsfusion-final-test.json` contains all run IDs, complete retrieval metrics and aggregates. Large scores/ranks are external, hashed run-local arrays.
- Lifecycle commits: `c0c4526` implementation/protocol; `ca9fc55` final phase; `96bf024` six reservations; `485c06b` six terminal records and phase summary; `fbc87c3` result summary. All workers terminated before this supplementary audit/comparison was written.
