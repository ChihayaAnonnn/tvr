**Final whole-study verdict: WARN.**  
**integrity_status: warn · review_independence: same-family · acceptance_status: provisional**

Audited final snapshot: `541dd67f39044d4c9de73a660426283901c343fd`.

The archived results support the reported **insufficient B2 variation and consequent stop**. I found no fabricated results, prediction-normalized performance, unlabelled model-derived ground truth, or inflated effectiveness claim in the final evidence report. WARN remains appropriate because historical protocol discrepancies and controller provenance/enforcement limitations remain material, although now disclosed.

| Check | Status | Finding and exact evidence |
|---|---|---|
| **A. Ground-truth provenance** | **PASS, scoped** | Official query→video relations are exported from allowlisted annotations at [source_preparation.py:125](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/source_preparation.py:125) and `:139`. Candidate retrieval precedes label binding at [execution.py:410](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/execution.py:410). Transition targets use dataset relevance of predicted endpoints at [transitions.py:75](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/transitions.py:75). Rich-score references are frozen-model outputs at [training.py:54](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/training.py:54); the final report explicitly labels their NLL as a proxy and rejects a semantic-ground-truth interpretation at [research-evidence-2026-10-10.md:19](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md:19). This is a custom validation pilot, not an official benchmark evaluation. |
| **B. Score normalization** | **PASS** | Accuracy uses episode/known-label denominators at [evaluation.py:54](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/evaluation.py:54); proper scores use valid targets at `:39`; bootstrap units are source components at `:82`. MC softmax at [matcher.py:26](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/matcher.py:26) defines model probabilities, rather than normalizing measured performance by prediction statistics. Negative Gaussian NLL and fixed-scale coverage of 1.0 are legitimate proxy-density diagnostics. |
| **C. Results, numbers and archives** | **PASS** | Completed phase certification checks committed control files, terminal cohorts, source/command/config/input identities and complete output inventories at [archive_reader.py:52](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/analysis/archive_reader.py:52). I independently verified B1 metrics and all six B2 archives, classes, cohorts and gate calculations. Final B2 values match [B2-transition-gate.json:13](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/B2-transition-gate.json:13). All 35 registered runs have terminal records; resource totals reconcile. The archived regression log reports 690 passed at [final-root-regression.log:17](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/validation/final-root-regression.log:17), and its hash matches the integration report. |
| **D. Invoked versus unexecuted metrics** | **PASS** | B1 episode and rich-score metrics are called at [execution.py:736](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/execution.py:736), `:778` and `:808`; transition compilation executes at `:638`. Value prediction, scientific calibration, threshold selection and locked-policy comparisons were not run. This is correctly stated at [research-evidence-2026-10-10.md:36](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md:36) and [EXPERIMENT_TRACKER.md:13](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/EXPERIMENT_TRACKER.md:13). Implemented software paths are not presented as executed scientific comparisons. |
| **E. Scope and protocol fidelity** | **WARN** | The final report correctly restricts evidence to MSRVTT trusted-v1 validation, T2V, DSL=false, seed713, 64 components per role and a 64-video/top-8 pilot at [research-evidence-2026-10-10.md:7](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/research-evidence-2026-10-10.md:7). It acknowledges low power, source-definition limits and unproved generalization. Historical optimizer-step and “V_locked unread” discrepancies remain, accurately disclosed at `:67` and `:68`; disclosure does not erase the frozen wording mismatch. |
| **F. Evaluation type** | **PASS** | Retrieval correctness, candidate presence and transition outcomes are **real_gt**, limited to official `benchmark_annotation` relations. Rich-score NLL/coverage/width are **synthetic_proxy**, using frozen-model references. Q0 and synthetic fixture checks are **simulation_only**, as shown by generated inputs and labels at [qualification.py:42](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/qualification.py:42), `:127` and `:179`. No human evaluation or executed self-supervised evaluation was identified. |

**Independent result verification.** The first pass certified D0, S1, S2, B1-R2 and Q0-R5, covering respectively 35, 2,175, 11,200, 141,162 and 787 terminal output files. B1 counts, corrections/worsenings, costs, component intervals and rich NLL were recomputed from verified arrays. All 320 B1 episode labels and costs also matched archived prediction journals and compiled official relations.

The final pass certified S3-G and S3-D, **5,655 terminal output files each**. It independently:

- Verified sealed configurations, parent lineage, producer identities, transition shards and array hashes.
- Reconstructed the frozen random trajectories and every legal state/action identity from bound metadata, without model inference.
- Checked all **9,156 transition classes** against dataset relations for their archived before/after identities.
- Recomputed every aggregate and matched all **384 component-table rows**. These rows represent two matchers over the same three 64-component roles, not 384 independent sources.
- Confirmed matching candidate sets across the two matchers.

| Matcher | Role | Transitions | Corrected / unchanged / worsened | Corrected / worsened components | Candidate misses |
|---|---|---:|---:|---:|---:|
| Gaussian | T_value | 1,526 | 8 / 1,512 / 6 | 4 / 1 | 6 |
| Gaussian | V_development | 1,526 | 9 / 1,505 / 12 | 4 / 4 | 4 |
| Gaussian | V_calibration | 1,526 | 4 / 1,522 / 0 | 1 / 0 | 5 |
| Deterministic | T_value | 1,526 | 4 / 1,513 / 9 | 2 / 2 | 6 |
| Deterministic | V_development | 1,526 | 6 / 1,513 / 7 | 2 / 3 | 4 |
| Deterministic | V_calibration | 1,526 | 0 / 1,521 / 5 | 0 / 2 | 5 |

Every arm includes all 64 source components; unknown transitions, unknown candidate-presence labels and components without legal transitions are all zero. Candidate misses remain represented. Gaussian development has **4 corrected and 4 worsened components**, below the preregistered **8/4** requirement at [PREREGISTRATION.md:41](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/PREREGISTRATION.md:41); therefore `passed=false` is correct.

**Safeguards and accounting.** Current [study.yaml:54](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/study.yaml:54) disables science and clears authorized stages. Read-only authorization probes explicitly rejected both a scientific start and a test split. Test rejection is independent at [authorization.py:98](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/authorization.py:98); scientific authorization is checked at `:143`. No B3/B4 runs appear in the completed registry.

Reviewed numerical/runtime safeguards include separate global DDP denominators and finite-gradient rejection at [training.py:112](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/training.py:112), strict checkpoint validation/loading at [checkpoint.py:123](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/checkpoint.py:123), and refusal to recover unverifiable unfinished work at [journal.py:204](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/journal.py:204). Q0-R5’s archived report has 13 successful checks on four distinct GPUs and matching continuous/resumed episode payloads. These establish tested software behavior, not scientific efficacy or universal crash recovery.

Independent all-history accounting verified 35 lifecycles and 193 metadata-file hashes:

| Category | Allocated GPU wall hours |
|---|---:|
| Scientific, including failures | 12.108477896449 |
| Software qualification | 0.469454636127 |
| CPU preparation | 0 |
| **Total** | **12.577932532576** |

Failures contribute 2.012990906974 hours within those totals. Four cancellations have authenticated pre-spawn, zero-work receipts. No observed budget, per-run duration or retry-count breach was found. The 29 GPU-worker running intervals show no cross-phase overlap. These observations support this campaign’s accounting, not arbitrary concurrent-campaign enforcement.

**Unresolved qualifications and action items.**

1. Preserve the optimizer-budget discrepancy in every reuse of B1. Gaussian/fixed-scale performed **121 updates**, other arms **110**, despite 128 visits each. Rich-score-only updates explain the difference; the literal skip wording at [PREREGISTRATION.md:27](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/PREREGISTRATION.md:27) is inaccurate. Do not claim matched optimizer steps, identical supervision, or an isolated causal effect of sigma.
2. Retain “V_locked was not used for model evaluation/selection.” Its manifests and labels were read for integrity validation at [runtime.py:35](/home/xujie/.codex/worktrees/ad3a/UATVR/src/semantic_retrieval/evidence/runtime.py:35). Literal “unread” is unsupported.
3. Preserve historical outer-controller provenance limitations. [administrative-controller-snapshot.json:6](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/reports/administrative-controller-snapshot.json:6) accurately declares no historical launch-time binding. Later snapshots cannot establish earlier executed client bytes.
4. Before any future separately authorized use requiring general campaign enforcement, add atomic accounting for active phases, GPU-count-aware reservations and an executable successor cap. The current controller counts terminal runs and assumes two GPU hours per current run at [native_phase_controller.py:23](/home/xujie/.codex/worktrees/ad3a/UATVR/experiments/evidence-conditioned-retrieval-v1/analysis/native_phase_controller.py:23) and `:30`. These gaps remain; they did not cause an observed breach here.
5. Keep the present science stop and test seal. This audit supplies no authorization to reopen the campaign.

**Claim impact.**

- **Supported:** the implemented, tested software paths; the descriptive B1 observations; the complete B2 counts; the failed variation gate; and stopping downstream work.
- **Needs the stated qualifications:** teacher-score fit, source isolation under the operational definition, resource accounting, tested recovery behavior and B1 method comparisons.
- **Not established:** C1’s independent uncertainty contribution, C2’s necessity of a full gain distribution, locked-policy superiority, generalization or robust multi-seed effectiveness.
- **Unsupported:** interpreting this pilot as a universal refutation of evidence-conditioned retrieval.

**Snapshot and hash record.** Product executable source is unchanged from `1d16e57fabfce3a5af1eec3d1bca5ffe42c29a5d`. B1 used `e79c2a24afdd79b0cfa7c10eeb2d951a4b349027`; S3-G used `8c05fa644df91df9868aefda883acb879ad2174b`; S3-D used `ef2fa0a580adbe767e65c4424a915a79471abcb3`.

The following paths are relative to `experiments/evidence-conditioned-retrieval-v1/`; values are SHA-256:

```text
study.yaml
a999381789a7b7e9668abe0a046aaab59ba737399186cda8ba4282a5128f9755
registry.jsonl
0ec91c9f37e1a626be143e42e7855859428b897f0fb2057bab8be0b745c38719
PREREGISTRATION.md
30b31722dcd4eebdad3f5bbf73f75c5e701e214ba367e5d64fa3c9f0a908f639
EXPERIMENT_TRACKER.md
3eb74390ca0c4b6e9da917c8984a9a82d3a3beb9a02f6a25fea57dfbe6f2798b
summaries/S3-G.json
98ba5a94ed82b922a7170e08d9302bdb39f967380c46aebd44af604393aac877
summaries/S3-D.json
d04ce62d5574e8011c011ff4c4965333f094da5bee4f84850b051488e853cb51
reports/B1-development-comparison.json
50896dc40fd854a96d116b9e2f3efeee6c6f865eb93c5c838de5f65da37d13ea
reports/B2-transition-gate.json
3b047ce8fa2709a239b1a8520525ab29c8c9b2119482e4b457103a79dce3029a
reports/B2-component-coverage.md
7748954b07840bfbbe4b59d56536b3f0572bed4e0b24cd7992959782d1a24c22
reports/final-resource-accounting.json
4f8b3b56b7b7897a9b43d108a2c749715c0821624f3e102ce0f7bdf097ccf961
reports/research-evidence-2026-10-10.md
dd9b995ef7e3446d9bf8d992e86530960717bc03701bc7a222bb82e063cad049
reports/administrative-controller-snapshot.json
9664881f6d57f47caff61f03183b85c898993a34057073b3c8190f7c3a9cca9b
reports/validation/final-root-regression.log
1db5818f4407dc884887b98eca38c867574aa4f748c6673d27cc639dccb7e47d
analysis/native_phase_controller.py
4006595271046eac96df9a8f1c9311ca60f095c8025adb047524832da2060e7b
analysis/account_resources.py
db81423ef02472dd84d5bf2fc48e7f78f63f21f26f5551c21869d585177f7158
```

During review, the author changed only the evidence narrative between `0d8cdf6` and `541dd67`: its former “timeout failures” description was corrected to diagnosed SIGTERM/143 termination around 1,282 seconds. I checked the one-line diff against the failure receipts/review. Its previous SHA-256 was `c6a6a0b23e52f6a60fb0dd7e53d8fc924cbff81aa932a4e1de1571461e36d0d2`; the corrected hash is recorded above. This reporting issue is resolved. No numerical results, gate criteria or executable files changed in that revision.

**Limits:** no GPU work, scientific inference rerun, raw mixed-annotation parsing, test-label inspection or file modification was performed. Transition endpoint predictions were verified through archived identities, classes and producing code, not recomputed with the model. Source grouping was checked against hash-bound D0 exports rather than independently reconstructing original YouTube metadata; unknown cross-URL visual duplicates remain outside the definition. The regression log was authenticated but the full suite was not rerun. The audit cannot certify absence of every transient historical file modification, every possible crash failure, or unrecorded actions outside the reviewed artifacts.