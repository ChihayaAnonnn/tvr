**Final judgment: `claim_supported: no` for the intended efficacy claims overall, and separately for C1 and C2.** Both remain scientifically unproved. B1 and B2 supply descriptive pilot evidence and a valid stopping decision; neither completed the controlled efficacy comparisons required by the claims. The B2 failure establishes insufficient transition variation under this frozen pilot and its preregistered criterion. It does not establish that uncertainty descriptors are universally useless or that evidence-conditioned retrieval is universally ineffective.

**C1 — incremental uncertainty information beyond geometry: unsupported; the decisive comparison was not executed.** The relevant observations are MSRVTT trusted-v1 **V_development**, T2V, DSL=false, seed713, 64 source components with one query each, a 64-video gallery and fixed top-8 candidates. B1 final correct counts were deterministic 14/64, fixed-scale 18/64, learned-temperature 14/64, standard-probabilistic 15/64 and Gaussian 17/64. Gaussian minus deterministic was +0.046875 accuracy, with a paired source-component bootstrap 95% interval of [−0.015625, 0.125] over 2,000 draws. This is compatible with no improvement; it is neither affirmative efficacy evidence nor an equivalence result. Fixed-scale's higher point estimate also prevents presenting Gaussian as the clear winner, without establishing fixed-scale superiority. These are fixed-frame matcher comparisons, not a test of learned acquisition information conditional on geometry. [B1 results](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/B1-development-comparison.json)

C1 attribution is additionally limited by **121 versus 110 optimizer updates** for Gaussian/fixed-scale versus the other matchers, despite 128 visits each, and by their different rich-score supervision. In B1, all arms recorded 4,096 frame requests, 4,096 CLIP frames and 27,648 temporal tokens, but Gaussian/fixed-scale/standard-probabilistic recorded 36,864 MC elements versus 4,608 for deterministic/learned-temperature. Thus these comparisons do not satisfy an equal-actual-compute interpretation across all four counters. Initial-index work is separately accounted for. Shared CPU/I/O during concurrent workers also prevents an isolated method-latency ranking. These limitations are disclosed; disclosure does not remove their implications for causal attribution. Teacher rich-score NLL measures fit to frozen-model outputs (`synthetic_proxy`), not human semantic uncertainty or incremental prediction of real corrections. [Training comparison](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/S2-matcher-completion.json), [research evidence](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md)

The missing C1 evidence is precisely the preregistered DistributionValue-versus-GeometryValue prediction comparison on the same Gaussian matcher, the deterministic-matcher control with the same acquisition-learning process, and a locked policy comparison with actual resource accounting. GeometryValue removes explicit log-scale while retaining the current Gaussian-derived score geometry; even a positive result would support incremental information under that specific feature partition, not removal of all probability-derived information. B3/B4 did not run, so there is no measured transition NLL/Brier improvement, calibrated acquisition advantage or locked accuracy improvement meeting C1's criteria. [Frozen claim and gates](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/PREREGISTRATION.md), [feature implementation](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/value.py)

**C2 — independent value or necessity of the full gain distribution versus a same-feature MeanRegressor: unsupported; no scientific comparison was executed.** There is no scientific value-head/calibration checkpoint, same-feature mean-versus-distribution prediction result or locked acquisition comparison. The implemented policy selects using `expected_gain = r_corrected − r_worsened`, its cost ratio and a stopping threshold. Conditional on identical predicted means, costs, available actions, budgets and thresholds, different three-class distributions produce the same decisions under this rule. Consequently, even a future categorical head outperforming a fitted MeanRegressor could establish an empirical mean-estimation/training advantage without establishing that the decision requires the full distribution. Proper scoring of class probabilities would address a separate forecasting claim; a scalar mean must not be assigned a fabricated categorical NLL. Neither forecasting nor decision superiority has been measured here. [Approved design](/home/xujie/.codex/worktrees/ad3a/UATVR/docs/superpowers/specs/2026-10-10-evidence-conditioned-retrieval-design.md), [policy](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/policy.py), [experiment tracker](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/EXPERIMENT_TRACKER.md)

**The supported scientific observation is the frozen pilot's insufficient transition variation.** All six Gaussian/deterministic × T_value/V_development/V_calibration producers completed, each covering 64 components and 1,526 transitions. Gaussian V_development contains 9 corrected and 12 worsened transition rows, occurring in only **4 corrected and 4 worsened components**. The registered requirement is at least **8 corrected and 4 worsened components**, so `passed=false` is correct. Counting the 9 corrected transition rows instead of 4 components would incorrectly pass the correction threshold. All six groups retain candidate misses and have zero unknown transitions and zero components without legal actions. The 384 component-table rows are repeated observations of 192 distinct sources across two matchers and three roles; the gate population is the 64 development components. Neither action count nor MC count creates additional independent source evidence. [B2 gate](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/B2-transition-gate.json), [component coverage](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/B2-component-coverage.md)

Software evidence is stronger and has a different scope: the archived affected regression suite reports **690 passed**, and final formal-capacity, FP32, four-GPU Q0-R5 reports **13 successful checks**, including actual CLIP paths, gradient/DDP behavior and exact resume within the tested checkpoint/action cases. This supports implementation and tested execution behavior. Synthetic inputs and random formal-capacity weights do not establish scientific learning, efficacy, generalization or recovery from every possible crash. [Q0-R5 qualification](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/Q0-R5-final-source-qualification.json), [integration review](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/gain-mse-integration-review.json)

**Integrity remains `warn`; this review is `same-family` and `provisional`.** The final audit records A/B/C/D/F passes with their stated scopes and E WARN. Preserve the frozen optimizer-skip wording discrepancy, the difference between V_locked metadata/labels being integrity-checked and being “unread,” the missing historical outer-controller launch-time hash, and incomplete general enforcement of active/concurrent budget reservations, GPU counts and successor limits. The reviewed campaign has no observed budget, duration or retry-count breach. Its 35 terminal lifecycles comprise 25 completed, 6 failed and 4 authenticated pre-spawn cancellations; allocated GPU wall time is 12.108477896449138 scientific GPUh plus 0.46945463612675664 software GPUh, including failures. These are campaign accounting results, not GPU-active utilization or matched method cost. [Final audit](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/EXPERIMENT_AUDIT.md), [audit JSON](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/EXPERIMENT_AUDIT.json), [resource accounting](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/final-resource-accounting.json)

Source isolation is supported only under the approved YouTube-ID-or-identical-SHA256 connected-component definition: validation500 is retained, 254 shared-source training videos are excluded and 8,246 remain eligible. This does not establish absence of unknown cross-URL visual duplicates. Official single-positive annotations define the observed correctness outcomes; they do not establish complete semantic relevance. V_locked was not used for model evaluation or selection, and MSRVTT/MSVD test remains sealed. [Study contract and closed authorization](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/study.yaml)

For this review I independently checked all 15 audited input hashes, all 7 precheck source hashes and their committed bytes at `541dd67f39044d4c9de73a660426283901c343fd`, and all 19 typed cited values. I also recomputed B2 aggregates and the gate from all 384 component rows, reconciled the 35 registry lifecycles and resource-table arithmetic, and inspected the value-feature and policy definitions. The precheck establishes existence, not semantic acceptance. I did not rerun inference, GPU work or the regression suite, reconstruct raw source metadata, parse raw mixed annotations or inspect test labels; underlying archive/inference integrity beyond these checks relies on the separately recorded audit. [Evidence precheck](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/evidence-precheck.json), [registry](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/registry.jsonl)

A conservative revised claim is: **“We implemented and software-verified an evidence-conditioned retrieval and acquisition pipeline. In one preregistered, source-separated MSRVTT validation pilot, matcher comparisons were descriptive and the Gaussian transition producer failed the frozen component-variation gate. Downstream value learning, calibration, locked policy comparisons and expansion were therefore not run. The study establishes neither incremental uncertainty information beyond geometry nor an independent decision benefit or necessity of a full gain distribution.”**

Confidence is **high in this support judgment and in the necessity of preserving the stop**, not high that C1/C2 are false in general. The evidence gaps below are conditional requirements for any future, separately authorized and preregistered investigation. They authorize no experiment, supplementary analysis campaign, pivot, ablation, altered gate or reopening of this study.

```json
{
  "claim_supported": "no",
  "per_claim": {
    "C1": {
      "claim_supported": "no",
      "tested": false,
      "tested_status": "Descriptive matcher comparisons and transition feasibility were tested; the decisive incremental prediction and matched-budget acquisition comparisons were not executed.",
      "support_status": "unproved_not_universally_refuted",
      "reason": "B1 is not a conditional-information test, its Gaussian-minus-deterministic interval includes zero, supervision and optimizer updates differ, and actual MC counts differ. DistributionValue versus GeometryValue, the deterministic acquisition-learning control and locked policy comparisons were not run."
    },
    "C2": {
      "claim_supported": "no",
      "tested": false,
      "tested_status": "No scientific same-feature DistributionValue versus MeanRegressor comparison, calibration or locked acquisition evaluation was executed.",
      "support_status": "unproved_not_universally_refuted",
      "reason": "Software implements the heads, but the policy consumes only expected gain. Equal means imply equal decisions under identical other inputs; mean-estimation improvement or class calibration alone would not establish full-distribution decision necessity."
    }
  },
  "what_results_support": [
    "Implemented software behavior within the 690-test regression and 13-check four-GPU Q0-R5 scopes, including tested gradients, observation isolation and checkpoint/action resume.",
    "Descriptive B1 results for MSRVTT trusted-v1 V_development, T2V, DSL=false, seed713, 64 source components, 64-video gallery and top8 candidates.",
    "Complete six-producer B2 coverage and the correctly failed frozen gate: Gaussian development has 4 corrected and 4 worsened components against thresholds 8 and 4.",
    "Stopping B3, B4 and scaling without changing the frozen pilot to pursue a pass.",
    "Source separation under the approved YouTube-ID-or-identical-SHA256 definition and reported all-history allocated GPU wall-hour accounting, with their stated limitations."
  ],
  "what_results_dont_support": [
    "Incremental uncertainty information beyond score geometry or an isolated causal effect of learned sigma.",
    "Matched-actual-budget acquisition superiority, calibrated transition prediction, locked-policy gains or calibrated semantic uncertainty.",
    "Independent scientific benefit or decision necessity of the full three-class gain distribution over the same-feature MeanRegressor.",
    "Robust multiseed effectiveness, larger-gallery or cross-dataset generalization, official test benchmark performance or novelty.",
    "Universal refutation of evidence-conditioned retrieval, absence of all visual duplicates, completely unread V_locked data, arbitrary-crash recovery or complete concurrent-campaign enforcement."
  ],
  "missing_evidence": [
    "Adequate independent-source correction and worsening coverage under a separately justified frozen protocol; this pilot failed its existing requirement.",
    "C1: same-matcher DistributionValue versus GeometryValue proper-score and gain-prediction comparisons, with geometry understood as the current retained feature partition.",
    "C1: deterministic matcher with the same value-learning process and controlled supervision/training exposure, plus locked acquisition comparisons accounting for actual frame requests, CLIP frames, temporal pair-frames and MC pair-samples.",
    "C2: same-feature MeanRegressor versus categorical head prediction and policy results, separating scalar gain error, proper class-probability scores and decision utility.",
    "Validation-only calibration and threshold selection followed by an untouched-for-model-selection locked evaluation; independent-source uncertainty estimates and replication appropriate to any asserted scope."
  ],
  "suggested_claim_revision": "An evidence-conditioned retrieval and acquisition pipeline was implemented and software-verified. A single preregistered source-separated MSRVTT validation pilot yielded descriptive matcher observations but insufficient component-level transition variation to proceed. B3/B4/scaling were stopped. Neither uncertainty information beyond geometry nor independent decision benefit or necessity of a full gain distribution has been established.",
  "next_experiments_needed": {
    "authorized": false,
    "executed": false,
    "current_study_action": "Remain closed; preserve all negative results and unexecuted stages. No automatic pivot, ablation, supplementary science, scaling or test access.",
    "scope": "Future evidence gaps only, requiring a separate explicit authorization and preregistered protocol; this review does not reopen the present campaign.",
    "evidence_requirements": [
      "An independently justified transition population with sufficient source-level outcome variation, without relaxing or retuning this study's frozen gate after seeing its result.",
      "The missing C1 controlled feature, matcher and actual-budget comparisons, with scientific calibration and locked evaluation.",
      "The missing C2 same-feature output-head comparison, with any forecast and decision claims evaluated separately; the current expected-mean policy cannot by itself demonstrate structural necessity of a full distribution.",
      "Independent-source precision and replication commensurate with any intended generalization claim."
    ]
  },
  "confidence": "high",
  "confidence_scope": "High confidence that the present evidence does not support the intended efficacy claims and that B2 requires stopping. This is not confidence that the underlying research hypotheses are universally false.",
  "integrity_status": "warn",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "evidence_precheck": {
    "verified_values": 19,
    "path_missing": 0,
    "value_not_found": 0,
    "meaning": "Cited values exist in verified committed bytes; this does not confer semantic claim acceptance."
  },
  "science_reopening_authorized": false,
  "test_authorized": false
}
```
