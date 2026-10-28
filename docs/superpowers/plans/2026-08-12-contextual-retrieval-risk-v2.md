# Contextual Retrieval Risk v2 (A3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen-retriever, candidate-conditioned sidecar that learns bidirectional top-1 retrieval failure risk and is promoted to one JSFUSION test only after passing the registered validation gate.

**Architecture:** A shared directional head consumes detached query/candidate means, a softmax top-K candidate context, normalized top-K scores/gaps, and the existing A1 score features. Training is A3-only with weighted BCE plus hardness fitting and pairwise ranking. A generic append-only uncertainty protocol preserves exact A1/retrieval outputs, validates checkpoint lineage, selects on validation only, and blocks test evaluation on failure.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy, pytest, SHA-256, existing UATVR/MSRVTT DDP and trusted-v1 evaluation.

## Global Constraints

- Initialize from `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`.
- Train only `contextual_risk_sidecar.*`; all existing model tensors, including A1 and CSD, stay frozen and in evaluation mode.
- Build both directions from the globally gathered WTI matrix, means, and group IDs; preserve multi-positive targets.
- Detach logits and embeddings before feature construction; no A3 gradient may reach WTI, CLIP, SAP, A1, CSD, or I1.
- Use `K=16`, projection width `128`, and `L = BCE + 0.25 * SmoothL1 + 0.25 * rank`.
- Keep behavior opt-in behind `--use_contextual_risk_sidecar`; legacy checkpoints and commands remain valid.
- Use trusted-v1 deterministic FP32 validation and exact paired hashes for raw WTI, retrieval, A1 failure risk, and A1 vacuity.
- Select by `(min(rho_t2v, rho_v2t), mean(AUROC), -mean(AURC), -epoch)` among valid candidates.
- Train three epochs; continue to five total only if epoch index 2 strictly improves over epoch index 1 under the registered tuple.
- Run JSFUSION exactly once only when the validation-selected checkpoint passes every A3 gate.
- Never commit logs, checkpoints, generated evidence JSON, or experiment clones.

## File Map

- `prob_models/contextual_retrieval_risk.py`: A3 features, network, labels, and objective.
- `prob_models/uncertainty_protocol.py`: method-neutral hashes, state audit, artifacts, selection, and acceptance helpers.
- `modules/modeling_mulit.py`: construction, detached forward path, and evaluation scoring.
- `main_task_retrieval.py`: CLI, freezing, diagnostics, evaluation, selection, continuation, and final-test gate.
- `experiment_tracking.py`: A3 manifest contract.
- `run_train_msrvtt_contextual_risk_bg.sh`: registered launcher.
- `tests/test_contextual_retrieval_risk*.py`, `tests/test_uncertainty_protocol.py`: unit, integration, and protocol coverage.

---

### Task 1: Candidate-conditioned features, shared head, and loss

**Files:**
- Create: `prob_models/contextual_retrieval_risk.py`
- Create: `tests/test_contextual_retrieval_risk.py`
- Create: `tests/test_contextual_retrieval_risk_loss.py`

**Public interfaces:**

- `contextual_topk_features(logits: Tensor, query_means: Tensor, candidate_means: Tensor, direction: str, topk: int = 16, temperature: float = 1.0) -> dict[str, Tensor]`
- `ContextualRetrievalRiskSidecar.forward(logits: Tensor, query_means: Tensor, candidate_means: Tensor, direction: str) -> dict[str, Tensor]`
- `contextual_retrieval_risk_objective(sidecar: ContextualRetrievalRiskSidecar, logits: Tensor, text_means: Tensor, video_means: Tensor, positive_mask: Tensor, hardness_temperature: float = 1.0, fit_weight: float = 0.25, rank_weight: float = 0.25) -> tuple[Tensor, dict[str, object]]`

- [ ] **Step 1: Write RED tests for direction, top-K padding, and detach semantics**

```python
def test_topk_features_follow_direction_and_pad_to_registered_width():
    logits = torch.tensor([[4., 3., 1.], [0., 2., 5.]])
    text = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    video = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    t2v = contextual_topk_features(logits, text, video, "t2v", topk=4)
    v2t = contextual_topk_features(logits, video, text, "v2t", topk=4)
    assert t2v["top_indices"].shape == (2, 4)
    assert v2t["top_indices"].shape == (3, 4)
    assert t2v["top_valid"].sum(1).tolist() == [3, 3]
    assert v2t["top_valid"].sum(1).tolist() == [2, 2, 2]
    torch.testing.assert_close(t2v["top1_mean"], video[t2v["top_indices"][:, 0]])
    assert t2v["normalized_scores"].shape == (2, 4)
    assert t2v["score_gaps"].shape == (2, 4)

def test_contextual_head_detaches_retrieval_inputs():
    logits = torch.randn(3, 4, requires_grad=True)
    text = torch.randn(3, 8, requires_grad=True)
    video = torch.randn(4, 8, requires_grad=True)
    head = ContextualRetrievalRiskSidecar(embed_dim=8, projection_dim=5, topk=2)
    head(logits, text, video, "t2v")["failure_logit"].sum().backward()
    assert logits.grad is None and text.grad is None and video.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py
```

Expected: collection fails because `prob_models.contextual_retrieval_risk` does not exist.

- [ ] **Step 3: Implement deterministic contextual feature construction**

Use `_oriented_scores` and `score_distribution_features` from `prob_models.retrieval_risk`. Standardize each row, choose `effective_k=min(K,num_candidates)`, pad scores/gaps with zero and indices by repeating the last real index, and expose a validity mask so padding receives zero attention:

```python
scores = _oriented_scores(logits.detach(), direction).float()
query = query_means.detach().float()
candidate = candidate_means.detach().float()
effective_k = min(topk, scores.size(1))
mean = scores.mean(1, keepdim=True)
std = scores.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
normalized = (scores - mean) / std
top_values, top_indices = normalized.topk(effective_k, dim=1)
top_valid = torch.ones_like(top_values, dtype=torch.bool)
if effective_k < topk:
    width = topk - effective_k
    top_indices = torch.cat((top_indices, top_indices[:, -1:].expand(-1, width)), 1)
    top_values = torch.cat((top_values, top_values.new_zeros(top_values.size(0), width)), 1)
    top_valid = torch.cat((top_valid, top_valid.new_zeros(top_valid.size(0), width)), 1)
selected = candidate[top_indices]
attention_logits = top_values.masked_fill(~top_valid, -torch.inf) / temperature
attention = torch.softmax(attention_logits, dim=1)
context = torch.sum(attention.unsqueeze(-1) * selected, dim=1)
gaps = (top_values[:, :1] - top_values).masked_fill(~top_valid, 0.0)
```

Validate finite 2-D tensors, matching embedding dimensions, row counts, `topk > 0`, and `temperature > 0`. Return query, top-1, context, existing 8 features, normalized scores, gaps, indices, mask, attention, and normalized entropy.

- [ ] **Step 4: Implement modality-aware projections and one shared failure logit**

```python
self.text_projection = nn.Linear(embed_dim, projection_dim)
self.video_projection = nn.Linear(embed_dim, projection_dim)
self.network = nn.Sequential(
    nn.Linear(3 * projection_dim + 8 + 2 * topk + 2, hidden_dim),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

For T2V, project the query with `text_projection` and candidates/context with `video_projection`; reverse those roles for V2T. Concatenate a two-element direction code after the feature values, return `failure_logit`, `failure_risk=sigmoid(logit)`, `topk_entropy`, and all audit features.

- [ ] **Step 5: Write and implement the registered objective**

Tests must prove multi-positive failure labels, bounded adaptive class weights, the ranking direction, both directional gradients, and rejection of all-positive/all-negative rows.

```python
top_index = scores.detach().argmax(1, keepdim=True)
failure = (~mask.gather(1, top_index).squeeze(1)).float()
num_positive = failure.sum()
num_negative = failure.numel() - num_positive
positive_weight = (num_negative / num_positive.clamp_min(1.0)).clamp(0.25, 4.0)
bce = F.binary_cross_entropy_with_logits(
    output["failure_logit"], failure,
    pos_weight=positive_weight.detach(),
)
hardness = retrieval_hardness(scores, mask, hardness_temperature)
fit = F.smooth_l1_loss(output["failure_risk"], _robust_rank_normalize(hardness))
rank = _pairwise_uncertainty_ranking_loss(output["failure_risk"], hardness)
direction_loss = bce + fit_weight * fit + rank_weight * rank
```

Average T2V and V2T losses and return per-direction failure rate, weight, prediction/target mean and std, valid ranking fraction, entropy, and saturation fraction.

- [ ] **Step 6: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py
git add prob_models/contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py
git commit -m "feat: add contextual retrieval risk head"
```

---

### Task 2: Model integration and exact A3-only freezing

**Files:**
- Modify: `modules/modeling_mulit.py`
- Modify: `main_task_retrieval.py`
- Create: `tests/test_contextual_retrieval_risk_integration.py`
- Modify: `tests/test_csd_logvar_config.py`

- [ ] **Step 1: Write RED integration tests**

Cover these invariants:

```python
assert sorted(trainable_names) == sorted(
    name for name, _ in model.named_parameters()
    if name.startswith("contextual_risk_sidecar.")
)
assert model.contextual_risk_sidecar.training
assert not model.clip.training
assert not model.retrieval_risk_sidecar.training
assert model.contextual_risk_sidecar_only
```

Also assert: flags require `--use_uatvr_full`; sidecar-only requires the use flag and A1; A3 and CSD-only/I1-only are mutually exclusive; legacy state dicts load with only A3 keys missing; DDP gathered `[N_text,N_video]`, `[N_text,D]`, and `[N_video,D]` shapes reach the objective.

- [ ] **Step 2: Add opt-in CLI arguments and validation**

Add next to existing sidecar arguments:

```python
parser.add_argument("--use_contextual_risk_sidecar", action="store_true")
parser.add_argument("--contextual_risk_sidecar_only", action="store_true")
parser.add_argument("--contextual_risk_topk", type=int, default=16)
parser.add_argument("--contextual_risk_projection_dim", type=int, default=128)
parser.add_argument("--contextual_risk_hidden_dim", type=int, default=128)
parser.add_argument("--contextual_risk_temperature", type=float, default=1.0)
parser.add_argument("--w_contextual_risk_fit", type=float, default=0.25)
parser.add_argument("--w_contextual_risk_rank", type=float, default=0.25)
```

Reject nonpositive dimensions/temperature, negative loss weights, invalid dependency combinations, and more than one `*_sidecar_only` mode.

- [ ] **Step 3: Construct and initialize A3 in `modules/modeling_mulit.py`**

Import the new module, store both flags, and construct the head only when enabled:

```python
self.contextual_risk_sidecar = ContextualRetrievalRiskSidecar(
    embed_dim=self.embed_dim,
    projection_dim=self.task_config.contextual_risk_projection_dim,
    hidden_dim=self.task_config.contextual_risk_hidden_dim,
    topk=self.task_config.contextual_risk_topk,
    temperature=self.task_config.contextual_risk_temperature,
)
```

Preserve legacy load behavior by following the current A1/CSD conditional construction and initialization pattern.

- [ ] **Step 4: Add detached A3 training and scoring paths**

In the WTI path, after global gathering and normalization, call the A3 objective with `text_pooled`, `mu_video`, WTI logits, and group-derived positive mask. Return early in A3-only mode with a loss dictionary and diagnostics, matching the existing sidecar-only contract. Add:

```python
def score_contextual_retrieval_risk(
    self, logits, text_means, video_means, direction
):
    if not self.use_contextual_risk_sidecar:
        raise RuntimeError("contextual retrieval risk sidecar is disabled")
    return self.contextual_risk_sidecar(
        logits.detach(), text_means.detach(), video_means.detach(), direction
    )
```

- [ ] **Step 5: Implement freeze and mode control**

Add `configure_contextual_risk_sidecar_training` and `_set_contextual_risk_training_mode`. Freeze all names first, enable only `contextual_risk_sidecar.*`, call `model.eval()`, and put only the A3 head in training mode at each epoch. Abort if any unexpected trainable name exists or the trainable count is zero.

- [ ] **Step 6: Run focused and compatibility tests, then commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk_integration.py \
  tests/test_retrieval_risk_integration.py \
  tests/test_csd_logvar_config.py
git add modules/modeling_mulit.py main_task_retrieval.py \
  tests/test_contextual_retrieval_risk_integration.py \
  tests/test_csd_logvar_config.py
git commit -m "feat: integrate A3 sidecar-only training"
```

---

### Task 3: Full-matrix directional evaluation and diagnostics

**Files:**
- Modify: `main_task_retrieval.py`
- Modify: `prob_models/retrieval_risk_metrics.py`
- Create: `tests/test_contextual_retrieval_risk_metrics.py`

- [ ] **Step 1: Write RED tests for metrics and complete candidate context**

Tests must detect the common error of scoring each mini-batch independently: split-batch scoring must differ from the required full-matrix scoring on a crafted example, and `eval_epoch` must call A3 only after all WTI rows/columns and unique modality means are assembled.

Assert the result contains:

```python
{
    "spearman_risk_vs_rank": float,
    "failure_auroc": {"value": float, "ci_low": float, "ci_high": float},
    "aurc": float,
    "risk_coverage": dict,
    "risk_stats": {"mean": float, "std": float, "min": float, "max": float},
    "saturation_fraction": float,
}
```

- [ ] **Step 2: Add AURC and saturation without changing A1 outputs**

Extend `evaluate_retrieval_risk` with `aurc = mean(1 - accuracy_at_prefix)` over the stable ascending-risk order and saturation at risk `<1e-4` or `>1-1e-4`. Retain all old keys so existing A1 tests remain backward compatible.

- [ ] **Step 3: Collect one canonical mean per query/candidate and score full matrices**

Reuse the existing CSD auxiliary collection/deduplication logic. For T2V call A3 with raw WTI, ordered text means, and unique ordered video means. For V2T use the same matrices and reverse orientation through the sidecar API. Store arrays and SHA-256 digests in the validation evidence; do not feed DSL scores to A3.

- [ ] **Step 4: Add 20-step training diagnostics and runtime aborts**

Log total/BCE/fit/rank, failure balance, adaptive weight, risk/hardness mean/std, valid-pair fraction, gradient norm, saturation, and top-K entropy for both directions. Raise on non-finite values, missing classes, or zero A3 gradient for two consecutive windows.

- [ ] **Step 5: Run tests and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk_metrics.py \
  tests/test_retrieval_risk_metrics.py \
  tests/test_directional_eval.py
git add main_task_retrieval.py prob_models/retrieval_risk_metrics.py \
  tests/test_contextual_retrieval_risk_metrics.py
git commit -m "feat: evaluate A3 on full candidate matrices"
```

---

### Task 4: Generic append-only uncertainty protocol and A3 gate

**Files:**
- Create: `prob_models/uncertainty_protocol.py`
- Modify: `prob_models/csd_logvar_protocol.py`
- Create: `tests/test_uncertainty_protocol.py`
- Create: `tests/test_contextual_retrieval_risk_protocol.py`

- [ ] **Step 1: Write RED protocol tests**

Cover artifact path/hash mismatch, duplicate epochs, overwrite attempts, resume from a non-highest epoch, non-sidecar state changes, paired digest mismatch, invalid candidates being excluded, lexicographic selection, earlier-epoch tie break, every A3 acceptance threshold, and `allow_final_test=False` on failure.

- [ ] **Step 2: Extract method-neutral artifact and state helpers**

Implement these explicit APIs:

- `validate_candidate_artifacts(candidates: Sequence[dict], *, output_dir: PathLike, method_name: str) -> list[dict]`
- `validate_resume_lineage(candidates: Sequence[dict], *, output_dir: PathLike, model_checkpoint: PathLike, optimizer_checkpoint: PathLike, optimizer_payload: Mapping, sidecar_prefix: str, method_name: str) -> dict`
- `non_sidecar_state_dict_mismatches(base_checkpoint: PathLike, candidate_checkpoint: PathLike, allowed_prefixes: tuple[str, ...]) -> list[str]`
- `compare_paired_invariance(baseline: Mapping, candidate: Mapping, required_digest_keys: tuple[str, ...]) -> dict[str, object]`
- `append_candidate_atomic(path: PathLike, candidate: Mapping) -> list[dict]`

Make the existing CSD public functions thin wrappers around these helpers and keep all CSD tests green. `append_candidate_atomic` must refuse an existing epoch/path and use temporary-file plus `os.replace` for the JSON payload.

- [ ] **Step 3: Implement registered A3 selection and acceptance**

```python
def contextual_risk_selection_tuple(candidate):
    return (
        min(candidate["rho"]),
        sum(candidate["auroc"]) / 2.0,
        -sum(candidate["aurc"]) / 2.0,
        -int(candidate["epoch"]),
    )

def contextual_risk_acceptance(metrics):
    directions = (metrics["t2v"], metrics["v2t"])
    checks = {
        "rho": all(x["spearman_risk_vs_rank"] >= 0.20 for x in directions),
        "auroc_ci": all(x["failure_auroc"]["ci_low"] > 0.50 for x in directions),
        "selective": all(monotone_selective_accuracy(x) for x in directions),
        "spread": all(x["risk_stats"]["std"] >= 0.02 for x in directions),
        "invariance": bool(metrics["paired_invariance"]["passed"]),
        "state": not metrics["non_a3_mismatches"],
    }
    return {"passed": all(checks.values()), "checks": checks}
```

Candidate `valid` additionally requires finite arrays/evidence, complete paired hashes, intact artifacts, and no runtime abort condition.

- [ ] **Step 4: Run protocol regressions and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_uncertainty_protocol.py \
  tests/test_contextual_retrieval_risk_protocol.py \
  tests/test_csd_logvar_protocol.py
git add prob_models/uncertainty_protocol.py prob_models/csd_logvar_protocol.py \
  tests/test_uncertainty_protocol.py \
  tests/test_contextual_retrieval_risk_protocol.py
git commit -m "refactor: generalize uncertainty experiment protocol"
```

---

### Task 5: Training lifecycle, manifest, continuation, and no-test gate

**Files:**
- Modify: `main_task_retrieval.py`
- Modify: `experiment_tracking.py`
- Create: `tests/test_contextual_retrieval_risk_training.py`
- Modify: `tests/test_experiment_tracking.py`

- [ ] **Step 1: Write RED lifecycle tests**

Use fake validation candidates to prove: epochs 0–2 are append-only; continuation occurs only when tuple(epoch 2) is strictly greater than tuple(epoch 1); resumption uses exact epoch-2 model plus optimizer and starts human epoch 4; epoch 3/4 candidates may replace the winner only if valid; final test is called exactly once on pass and never on failure; no test metric is passed to selection.

- [ ] **Step 2: Add manifest fields**

Record method name/version, canonical A1 absolute path and SHA-256, all architecture/loss settings, trusted-v1 split identity, seed/world-size/precision/effective batch, registered thresholds, continuation rule, candidate JSON path, and final-test eligibility. Reject resume when immutable manifest fields differ.

- [ ] **Step 3: Wire append-only selection and gated final test**

Create a separate A3 candidate/selection artifact name. After each validation: hash saved model/optimizer, audit non-A3 state against canonical A1, compare paired outputs, calculate metrics and gate, then atomically append. Resolve the winner from validation only. Replace unconditional `args.run_final_test` behavior in A3-only mode with:

```python
acceptance = contextual_risk_acceptance(winner["metrics"])
if args.run_final_test and acceptance["passed"]:
    selected = validate_selected_checkpoint(selection, output_dir=args.output_dir)
    run_selected_jsfusion_once(selected)
elif args.run_final_test:
    logger.warning("A3 validation gate failed; JSFUSION test is intentionally skipped")
```

- [ ] **Step 4: Run lifecycle and regression tests, then commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk_training.py \
  tests/test_experiment_tracking.py \
  tests/test_csd_logvar_protocol.py
git add main_task_retrieval.py experiment_tracking.py \
  tests/test_contextual_retrieval_risk_training.py \
  tests/test_experiment_tracking.py
git commit -m "feat: register A3 validation-only selection"
```

---

### Task 6: Registered launcher and preflight verification

**Files:**
- Create: `run_train_msrvtt_contextual_risk_bg.sh`
- Create: `tests/test_contextual_retrieval_risk_launcher.py`

- [ ] **Step 1: Write launcher contract test**

Parse the shell file and assert canonical A1 path, `--use_contextual_risk_sidecar`, `--contextual_risk_sidecar_only`, `K=16`, FP32, seed 0, four devices, effective batch 256, three epochs, trusted-v1 validation, `--run_final_test`, unique timestamped log/output paths, and absence of CSD/I1-only flags. Assert the script refuses an existing output directory.

- [ ] **Step 2: Implement the launcher**

Follow the existing background-launcher PID/log contract. Use `torchrun --nproc_per_node=4`, per-device batch 64, gradient accumulation 1, canonical A1 `--init_model`, `--epochs 3`, and the registered A3 arguments. The Python final-test gate, not shell branching, decides whether JSFUSION is eligible.

- [ ] **Step 3: Run narrow and full regression suites**

```bash
bash -n run_train_msrvtt_contextual_risk_bg.sh
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_contextual_retrieval_risk_launcher.py \
  tests/test_contextual_retrieval_risk.py \
  tests/test_contextual_retrieval_risk_loss.py \
  tests/test_contextual_retrieval_risk_integration.py \
  tests/test_contextual_retrieval_risk_metrics.py \
  tests/test_contextual_retrieval_risk_protocol.py \
  tests/test_contextual_retrieval_risk_training.py
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests
```

- [ ] **Step 4: Verify a one-process synthetic smoke run and commit**

Run the repository's smallest fixture configuration through construction, one A3 train step, checkpoint round-trip, full-matrix evaluation, and failed-gate test suppression. Confirm raw WTI/A1 hashes are identical before and after.

```bash
git add run_train_msrvtt_contextual_risk_bg.sh \
  tests/test_contextual_retrieval_risk_launcher.py
git commit -m "feat: add registered A3 launcher"
```

---

### Task 7: Registered A3 experiment and scientific report

**Files:**
- Create after completion: `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md`

- [ ] **Step 1: Confirm clean tracked state and launch exactly once**

```bash
git status --short --branch
bash run_train_msrvtt_contextual_risk_bg.sh
```

Record worker PID, output directory, log path, command, manifest hash, and canonical A1 hash. Monitor finite losses, class balance, gradients, output spread, entropy, checkpoints, and tracebacks without altering hyperparameters.

- [ ] **Step 2: Apply the registered continuation rule**

After epoch-index 2 validation, compare only epoch indices 2 and 1 with the registered tuple. If strictly improved, resume the exact epoch-2 model and optimizer in the same output directory to five total epochs; otherwise stop. Do not inspect or run JSFUSION to make this decision.

- [ ] **Step 3: Validate artifacts and report**

Re-hash every candidate, rerun validation-only only if evidence recovery is necessary, and report directional rho/AUROC CI/AURC/coverage, output statistics, acceptance checks, paired digests, non-A3 audit, selected epoch, and any protocol deviation. If the gate fails, explicitly report that no JSFUSION result exists. If it passes, report the one test invocation separately from validation selection.

- [ ] **Step 4: Commit only the tracked report**

```bash
git add docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md
git commit -m "docs: report registered A3 results"
git status --short --branch
```
