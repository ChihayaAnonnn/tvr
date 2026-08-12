# Intrinsic Spherical Concentration (I1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a candidate-independent scalar concentration signal that learns sample-local ambiguity from reproducible token/frame perturbation disagreement while preserving retrieval and all existing uncertainty outputs exactly.

**Architecture:** Independent text/video heads consume detached normalized pooled means plus local dispersion summaries and emit positive spherical concentration `kappa`; uncertainty is `1/(1+kappa)`. Two seeded dropout views of already-computed token/frame features provide detached training targets. I1 has its own intrinsic validation gate, selection tuple, artifacts, launcher, and report and is never described as retrieval failure probability unless it separately clears the A3 gate.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy, pytest, SHA-256, existing UATVR/MSRVTT DDP and trusted-v1 evaluation.

## Global Constraints

- Initialize independently from `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`, never from A3.
- Train only `intrinsic_concentration_sidecar.*`; freeze WTI, CLIP, SAP, A1, CSD, and A3 in evaluation mode.
- Never consume candidate sets, WTI/DSL logits, ranks, A1, A3, or pairwise match labels in the I1 head or loss.
- Use only detached pooled means and sample-local token/frame summaries from the existing encoder pass.
- Parameterize `kappa = softplus(raw_kappa) + 1e-4` and `u = 1/(1+kappa)` with scalar output per sample.
- Generate two reproducible dropout views without a second encoder forward and retain at least one valid element per sample/view.
- Use `L = SmoothL1(u,target) + pairwise_rank(u,d_view) + 0.01 * concentration_prior`.
- Primary validation is held-out perturbation disagreement and registered severity response; retrieval metrics are descriptive only.
- Train three epochs and continue to five only under the registered intrinsic validation tuple.
- JSFUSION/corruption evaluation is permitted once only after the selected validation checkpoint passes the intrinsic gate.
- Never commit logs, checkpoints, generated evidence JSON, or experiment clones.

## File Map

- `prob_models/intrinsic_concentration.py`: local summaries, seeded views, scalar heads, and objective.
- `prob_models/uncertainty_protocol.py`: shared artifacts, hashes, state audit, and I1 protocol helpers.
- `modules/modeling_mulit.py`: I1 construction, detached feature path, and evaluation auxiliary outputs.
- `main_task_retrieval.py`: CLI, freeze rules, training diagnostics, severity evaluation, selection, and test gate.
- `experiment_tracking.py`: I1 manifest fields.
- `run_train_msrvtt_intrinsic_concentration_bg.sh`: registered launcher.
- `tests/test_intrinsic_concentration*.py`: mathematical, integration, evaluation, protocol, and launcher tests.

---

### Task 1: Local summaries, seeded views, concentration head, and objective

**Files:**
- Create: `prob_models/intrinsic_concentration.py`
- Create: `tests/test_intrinsic_concentration.py`
- Create: `tests/test_intrinsic_concentration_loss.py`

**Public interfaces:**

- `local_dispersion_summary(features: Tensor, mask: Tensor, pooled: Tensor) -> Tensor`
- `seeded_dropout_pool(features: Tensor, mask: Tensor, drop_probability: float, seed: int) -> Tensor`
- `stochastic_view_disagreement(features: Tensor, mask: Tensor, drop_probability: float, seed: int) -> Tensor`
- `IntrinsicConcentrationSidecar.forward(pooled: Tensor, summary: Tensor, modality: str) -> dict[str, Tensor]`
- `intrinsic_concentration_objective(sidecar: IntrinsicConcentrationSidecar, text_features: Tensor, text_mask: Tensor, text_pooled: Tensor, video_features: Tensor, video_mask: Tensor, video_pooled: Tensor, drop_probability: float = 0.2, seed: int = 0) -> tuple[Tensor, dict[str, object]]`

- [ ] **Step 1: Write RED mathematical tests**

```python
def test_concentration_is_scalar_positive_and_bounded_uncertainty():
    sidecar = IntrinsicConcentrationSidecar(embed_dim=8, hidden_dim=5)
    pooled = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    summary = torch.randn(4, 5)
    out = sidecar(pooled, summary, "text")
    assert out["raw_kappa"].shape == (4,)
    assert torch.all(out["kappa"] > 0)
    assert torch.all((out["uncertainty"] > 0) & (out["uncertainty"] < 1))
    assert torch.std(out["uncertainty"]) > 0

def test_seeded_views_repeat_and_keep_one_valid_element():
    features = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    left = seeded_dropout_pool(features, mask, 0.9, seed=17)
    right = seeded_dropout_pool(features, mask, 0.9, seed=17)
    torch.testing.assert_close(left, right)
    assert torch.isfinite(left).all()

def test_local_summary_is_sample_local():
    changed = features.clone()
    changed[1] *= 100
    before = local_dispersion_summary(features, mask, pooled)
    after = local_dispersion_summary(changed, mask, pooled)
    torch.testing.assert_close(before[0], after[0])
```

Also test invalid masks, probabilities outside `[0,1)`, all-masked samples, non-finite values, deterministic rank/step seed composition, and non-identical Xavier initialization of the two modality heads.

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration.py \
  tests/test_intrinsic_concentration_loss.py
```

Expected: collection fails because `prob_models.intrinsic_concentration` does not exist.

- [ ] **Step 3: Implement five sample-local dispersion features**

Normalize valid tokens/frames and pooled means, then return:

```python
cosine_disagreement = 1.0 - torch.sum(normalized_features * normalized_pooled[:, None], -1)
summary = torch.stack((
    masked_mean(cosine_disagreement),
    masked_std(cosine_disagreement),
    mask.float().mean(1),
    masked_mean(features.norm(dim=-1)),
    masked_std(features.norm(dim=-1)),
), dim=1)
```

Mask every statistic explicitly and use population standard deviation so a single valid element is finite.

- [ ] **Step 4: Implement reproducible dropout pooling and disagreement**

Use a device-local `torch.Generator`, draw only over valid entries, and repair an empty keep mask by selecting one valid index from the same generator. Normalize the masked mean. Build views with seeds `2*seed` and `2*seed+1`, then return detached `1-cosine_similarity(view1,view2)`.

```python
generator = torch.Generator(device=features.device)
generator.manual_seed(int(seed))
keep = (torch.rand(mask.shape, generator=generator, device=mask.device) >= drop_probability) & mask
empty = ~keep.any(1)
for row in torch.nonzero(empty, as_tuple=False).flatten().tolist():
    valid = torch.nonzero(mask[row], as_tuple=False).flatten()
    choice = valid[torch.randint(valid.numel(), (1,), generator=generator, device=mask.device)]
    keep[row, choice] = True
pooled = (features * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True)
return F.normalize(pooled, dim=-1)
```

- [ ] **Step 5: Implement independent scalar heads and loss**

```python
self.text_head = nn.Sequential(
    nn.Linear(embed_dim + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
)
self.video_head = nn.Sequential(
    nn.Linear(embed_dim + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
)
raw = head(torch.cat((pooled.detach().float(), summary.detach().float()), 1)).squeeze(1)
kappa = F.softplus(raw) + 1e-4
uncertainty = 1.0 / (1.0 + kappa)
```

Initialize every linear weight with Xavier uniform and nonzero bias conventionally. For each modality compute rank-normalized detached disagreement target, Smooth-L1, `_pairwise_uncertainty_ranking_loss`, and `prior = log(kappa).square().mean()`. Average modality losses. Return disagreement, targets, prediction statistics, valid-pair fraction, saturation, and each component.

- [ ] **Step 6: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration.py \
  tests/test_intrinsic_concentration_loss.py
git add prob_models/intrinsic_concentration.py \
  tests/test_intrinsic_concentration.py \
  tests/test_intrinsic_concentration_loss.py
git commit -m "feat: add intrinsic concentration head"
```

---

### Task 2: Model integration and isolated I1 training

**Files:**
- Modify: `modules/modeling_mulit.py`
- Modify: `main_task_retrieval.py`
- Create: `tests/test_intrinsic_concentration_integration.py`
- Modify: `tests/test_csd_logvar_config.py`

- [ ] **Step 1: Write RED integration tests**

Assert exact trainable names are `intrinsic_concentration_sidecar.*`; all other modules are in eval mode; gradients stop at detached pooled/token/frame features; both modality heads receive gradients; a legacy A1 checkpoint loads with only I1 keys missing; enabling I1 does not instantiate or activate A3; and DDP seed composition differs by rank/step while replaying exactly for the same tuple.

- [ ] **Step 2: Add explicit CLI and validation**

```python
parser.add_argument("--use_intrinsic_concentration", action="store_true")
parser.add_argument("--intrinsic_concentration_only", action="store_true")
parser.add_argument("--intrinsic_concentration_hidden_dim", type=int, default=128)
parser.add_argument("--intrinsic_view_drop_probability", type=float, default=0.2)
parser.add_argument("--w_intrinsic_rank", type=float, default=1.0)
parser.add_argument("--w_intrinsic_prior", type=float, default=0.01)
```

Require UATVR-full, A1 loaded, and the use flag for only-mode; validate dimension, probability, and nonnegative weights; reject simultaneous A3/CSD/I1-only modes.

- [ ] **Step 3: Construct I1 and expose sample-local auxiliary inputs**

Create `IntrinsicConcentrationSidecar(embed_dim=self.embed_dim, hidden_dim=self.task_config.intrinsic_concentration_hidden_dim)` only when enabled. Reuse already-computed normalized `text_token`/`visual_output`, their masks, `text_pooled`, and `mu_video`; do not run either encoder twice. Use an explicit `intrinsic_seed = base_seed + global_step * world_size + rank` passed from the training loop.

- [ ] **Step 4: Add I1-only early return and eval outputs**

In training, call the objective after global gathering needed for stable effective batch statistics, but keep all views sample-local. Return only the I1 loss in I1-only mode. During evaluation, return canonical per-sample means, summaries, uncertainty, and held-out disagreement without changing logits.

- [ ] **Step 5: Freeze and control modes exactly**

Add `configure_intrinsic_concentration_training` and `_set_intrinsic_concentration_training_mode`, mirroring the audited A3 behavior. Assert no trainable name lies outside the I1 prefix and abort on zero trainable parameters.

- [ ] **Step 6: Run tests and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration_integration.py \
  tests/test_contextual_retrieval_risk_integration.py \
  tests/test_csd_logvar_config.py
git add modules/modeling_mulit.py main_task_retrieval.py \
  tests/test_intrinsic_concentration_integration.py \
  tests/test_csd_logvar_config.py
git commit -m "feat: integrate I1 sidecar-only training"
```

---

### Task 3: Held-out disagreement and severity evaluation

**Files:**
- Modify: `prob_models/retrieval_risk_metrics.py`
- Modify: `main_task_retrieval.py`
- Create: `tests/test_intrinsic_concentration_metrics.py`

- [ ] **Step 1: Write RED metric tests**

Use synthetic uncertainty/disagreement arrays to assert Spearman correlation, distribution and saturation, deterministic severity means, endpoint delta, monotonicity, and failure on constant/non-finite inputs. Assert retrieval rank rho/AUROC live under a `descriptive_retrieval` key and cannot satisfy the intrinsic gate.

- [ ] **Step 2: Implement intrinsic metric aggregation**

Add:

```python
def evaluate_intrinsic_concentration(
    uncertainty, heldout_disagreement, severity_uncertainty,
    descriptive_retrieval=None,
):
    rho = spearmanr(uncertainty, heldout_disagreement).statistic
    severity = {str(level): summary(values) for level, values in severity_uncertainty.items()}
    means = [severity[str(level)]["mean"] for level in (0.0, 0.1, 0.2, 0.3)]
    return {
        "spearman_uncertainty_vs_disagreement": float(rho),
        "uncertainty_stats": summary(uncertainty),
        "saturation_fraction": float(np.mean((uncertainty < 1e-4) | (uncertainty > 1-1e-4))),
        "severity": severity,
        "severity_monotonic": all(a <= b for a, b in zip(means, means[1:])),
        "severity_endpoint_delta": float(means[-1] - means[0]),
        "descriptive_retrieval": descriptive_retrieval,
    }
```

- [ ] **Step 3: Register evaluation seeds and severities**

For validation use held-out seed `1_000_003 + sample_id`, distinct from all training seeds. Evaluate severities exactly `(0.0, 0.1, 0.2, 0.3)` separately for text-token dropout and video-frame dropout. At severity zero, compute the intact sample output; at other severities recompute only pooling/summaries/head output from the saved features. Do not alter WTI matrices or encode inputs again.

- [ ] **Step 4: Add runtime diagnostics**

Every 20 steps log total/fit/rank/prior, uncertainty/disagreement mean/std, valid-pair fraction, modality gradient norms, saturation, and kappa min/max. Abort on non-finite values, all-masked views, or zero I1 gradient for two consecutive windows; treat low spread as validation failure only.

- [ ] **Step 5: Run tests and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration_metrics.py \
  tests/test_retrieval_risk_metrics.py \
  tests/test_directional_eval.py
git add prob_models/retrieval_risk_metrics.py main_task_retrieval.py \
  tests/test_intrinsic_concentration_metrics.py
git commit -m "feat: evaluate intrinsic concentration response"
```

---

### Task 4: I1 protocol, selection, and final-test eligibility

**Files:**
- Modify: `prob_models/uncertainty_protocol.py`
- Create: `tests/test_intrinsic_concentration_protocol.py`
- Modify: `main_task_retrieval.py`
- Modify: `experiment_tracking.py`
- Modify: `tests/test_experiment_tracking.py`

- [ ] **Step 1: Write RED protocol tests**

Cover both modality thresholds, nonmonotonic severity, insufficient endpoint delta/spread, saturation, paired invariance, non-I1 state changes, invalid artifact exclusion, deterministic selection, continuation, and final-test suppression. Prove descriptive retrieval metrics cannot change pass/fail or selection.

- [ ] **Step 2: Implement the intrinsic acceptance gate**

```python
def intrinsic_concentration_acceptance(metrics):
    modalities = (metrics["text"], metrics["video"])
    checks = {
        "disagreement_rho": all(
            x["spearman_uncertainty_vs_disagreement"] >= 0.40 for x in modalities
        ),
        "severity_monotonic": all(x["severity_monotonic"] for x in modalities),
        "endpoint_delta": all(x["severity_endpoint_delta"] >= 0.05 for x in modalities),
        "spread": all(x["uncertainty_stats"]["std"] >= 0.02 for x in modalities),
        "saturation": all(x["saturation_fraction"] < 0.05 for x in modalities),
        "invariance": bool(metrics["paired_invariance"]["passed"]),
        "state": not metrics["non_i1_mismatches"],
    }
    return {"passed": all(checks.values()), "checks": checks}
```

- [ ] **Step 3: Register intrinsic selection and continuation tuple**

Use validation only:

```python
def intrinsic_concentration_selection_tuple(candidate):
    text, video = candidate["metrics"]["text"], candidate["metrics"]["video"]
    return (
        min(text["spearman_uncertainty_vs_disagreement"],
            video["spearman_uncertainty_vs_disagreement"]),
        min(text["severity_endpoint_delta"], video["severity_endpoint_delta"]),
        -max(text["saturation_fraction"], video["saturation_fraction"]),
        -int(candidate["epoch"]),
    )
```

Continue from three to five total epochs only when epoch index 2 strictly exceeds epoch index 1 under this tuple.

- [ ] **Step 4: Add manifest, append-only candidates, and test gate**

Record canonical A1/hash, method/version, hidden width, view probability, loss weights, training/held-out seed formulas, severities, precision/world-size/batch, thresholds, continuation rule, and candidate artifact. Audit all non-I1 state, required raw WTI/A1 hashes, and exact retrieval equality. Run one descriptive corruption/JSFUSION evaluation only if selected validation acceptance passes; otherwise log an intentional skip.

- [ ] **Step 5: Run tests and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration_protocol.py \
  tests/test_uncertainty_protocol.py \
  tests/test_experiment_tracking.py
git add prob_models/uncertainty_protocol.py main_task_retrieval.py \
  experiment_tracking.py tests/test_intrinsic_concentration_protocol.py \
  tests/test_experiment_tracking.py
git commit -m "feat: register I1 validation protocol"
```

---

### Task 5: Registered launcher and verification

**Files:**
- Create: `run_train_msrvtt_intrinsic_concentration_bg.sh`
- Create: `tests/test_intrinsic_concentration_launcher.py`

- [ ] **Step 1: Write launcher contract test**

Assert canonical A1 path, I1 use/only flags, hidden width 128, view drop 0.2, rank weight 1.0, prior 0.01, FP32, seed 0, four GPUs, effective batch 256, three epochs, trusted-v1, final-test request subject to the Python gate, unique paths, and absence of A3/CSD-only flags.

- [ ] **Step 2: Implement launcher and syntax check**

Follow the existing registered background PID/log conventions and refuse an existing output directory. Use `torchrun --nproc_per_node=4`, per-device batch 64, gradient accumulation 1, and the exact registered arguments.

```bash
bash -n run_train_msrvtt_intrinsic_concentration_bg.sh
```

- [ ] **Step 3: Run focused and full verification**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_intrinsic_concentration_launcher.py \
  tests/test_intrinsic_concentration.py \
  tests/test_intrinsic_concentration_loss.py \
  tests/test_intrinsic_concentration_integration.py \
  tests/test_intrinsic_concentration_metrics.py \
  tests/test_intrinsic_concentration_protocol.py
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests
```

- [ ] **Step 4: Run one synthetic lifecycle smoke test and commit**

Exercise one I1 train step, seed replay, checkpoint round-trip, four severities, paired hashes, and failed-gate test suppression. Confirm no existing parameter changes.

```bash
git add run_train_msrvtt_intrinsic_concentration_bg.sh \
  tests/test_intrinsic_concentration_launcher.py
git commit -m "feat: add registered I1 launcher"
```

---

### Task 6: Registered I1 experiment and scientific report

**Files:**
- Create after completion: `docs/superpowers/reports/2026-08-12-intrinsic-concentration-results.md`

- [ ] **Step 1: Launch independently from canonical A1**

```bash
git status --short --branch
bash run_train_msrvtt_intrinsic_concentration_bg.sh
```

Record PID, command, paths, manifest and canonical hashes. Monitor both modality losses, disagreement, kappa/uncertainty spread, gradients, saturation, artifact creation, and tracebacks without changing registered settings.

- [ ] **Step 2: Apply validation-only continuation**

After three epochs, compare epoch indices 2 and 1 using only the intrinsic selection tuple. Resume the exact epoch-2 model and optimizer in the same output directory to five total epochs only on strict improvement.

- [ ] **Step 3: Validate and report the scientific claim**

Report text/video held-out disagreement rho, severity means and endpoint deltas, output distributions/saturation, intrinsic acceptance checks, descriptive retrieval rho/AUROC, paired digests, non-I1 audit, selected epoch, continuation decision, and protocol deviations. State clearly that I1 is candidate-independent ambiguity, not retrieval-failure probability. A failure must have no JSFUSION result; a pass receives exactly one descriptive test invocation.

- [ ] **Step 4: Commit only the tracked report**

```bash
git add docs/superpowers/reports/2026-08-12-intrinsic-concentration-results.md
git commit -m "docs: report registered I1 results"
git status --short --branch
```
