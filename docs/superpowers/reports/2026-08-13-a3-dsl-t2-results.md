# A3 winner with fixed DSL T=2: post-hoc results (2026-08-13)

## Outcome

The fixed, no-search DSL evaluation completed successfully and reported **T2V R@1 = 52.9**, exceeding the historical fixed-T=2 result of 52.6 by **0.3 percentage points**. V2T R@1 was 52.2.

This is an exploratory retrieval result, not a second registered A3 final-test claim. DSL, not A3, changes the retrieval ranking. The A3 head remains a detached failure-risk estimator.

## Immutable inputs and execution

- Evaluation source HEAD: `c03c37c` on `feature/contextual-intrinsic-uncertainty`; tracked source/index were clean at launch.
- Selected checkpoint: `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3`.
- Checkpoint SHA-256: `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c`.
- Dataset: MSRVTT JSFUSION test, 1000 texts × 1000 videos.
- Evaluation: `do_train=false`, `do_eval=true`, `eval_split=test`, FP32.
- Retrieval post-processing: `DSL=true`, `dsl_temperature=2.0`; no temperature search.
- A3 configuration: top-K 16, projection 128, hidden 128, temperature 1.0.
- Output: `ckpts/ckpt_msrvtt_20260813_214730_a3_dsl_t2_eval`.
- External log: `logs/20260813/214730_a3_dsl_t2_eval_msrvtt.log`, SHA-256 `749e3abb736e427dbcb77495dd1f82bf5c595d080a1fbd4e327983ddfe5d032f`.
- Output `log.txt` SHA-256: `0fe397f6ae531c012d5108fcc1fe37f915697612d99853bf388262edb5d03892`.

No optimizer was loaded, no checkpoint was selected from test results, and neither the original A3 final-test artifact nor its one-shot claim was modified.

## Retrieval metrics

| Direction | R@1 | R@5 | R@10 | MdR | MnR |
|---|---:|---:|---:|---:|---:|
| T2V | **52.9** | 77.3 | 85.3 | 1.0 | 10.5 |
| V2T | 52.2 | 77.9 | 85.6 | 1.0 | 7.9 |

| T2V condition | R@1 | Delta from raw | Delta from historical DSL |
|---|---:|---:|---:|
| Frozen retriever, raw | 50.0 | — | -2.6 |
| Historical DSL T=2 | 52.6 | +2.6 | — |
| Current selected checkpoint, DSL T=2 | **52.9** | **+2.9** | **+0.3** |

Thus the predeclared comparison is `52.9 > 52.6`. On a 1000-query test set, the displayed 0.3-point margin corresponds to roughly three additional Top-1 queries and should not be overinterpreted as a large effect.

## Risk metrics logged during the same pass

| Head | Direction | Spearman rho | AUROC | AURC |
|---|---|---:|---:|---:|
| A1 direct | T2V | 0.5623 | 0.7895 | 0.2458 |
| A1 direct | V2T | 0.4913 | 0.7641 | 0.2580 |
| A3 contextual | T2V | 0.5590 | 0.8162 | 0.2499 |
| A3 contextual | V2T | 0.5431 | 0.8290 | 0.2711 |

A3 continues to score detached raw-WTI context, so these contextual-risk values are effectively unchanged from the registered no-DSL pass. A1 evaluation receives the directional DSL retrieval matrices when computing rank/failure metrics, which accounts for its metric movement.

## Interpretation and limitations

- The result answers the narrow question positively: under the fixed historical DSL T=2 configuration, the current evaluation reports 52.9 and exceeds 52.6.
- It does **not** show that A3 improves R@1. A3 cannot change the ranking in the current architecture.
- Because the frozen non-A3 retriever is the same lineage, the 0.3-point difference from the older DSL result is more plausibly associated with evaluator evolution, directional DSL handling, FP32 numerical details, or ordering differences than with learned A3 parameters.
- This post-hoc run used exactly one fixed temperature and did not tune against JSFUSION.
- Both logs were free of exceptions, CUDA memory exhaustion, and non-finite diagnostics. The eval-only path persisted logs but no separate metric JSON.
