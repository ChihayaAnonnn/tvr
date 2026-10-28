# Retrieval-Risk Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen-backbone binary evidential sidecar that predicts T2V/V2T retrieval failure without changing WTI similarities or rankings.

**Architecture:** Extract permutation-invariant features from detached rows and columns of the complete WTI matrix. A shared MLP produces binary Dirichlet evidence, supervised by detached multi-positive retrieval hardness. The original retrieval score path remains untouched.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy, pytest, existing UATVR/MSRVTT DDP pipeline.

## Global Constraints

- Initialize A1 from `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4`, never the failed route 1+3 checkpoint.
- Never multiply retrieval scores by SAP uncertainty, confidence, sidecar risk, or vacuity.
- Detach WTI logits before feature extraction; risk loss may update only `retrieval_risk_sidecar.*`.
- Construct training positives from DDP-gathered `group_ids`, not an identity matrix.
- Compute evaluation risk from the complete candidate axis, not candidate chunks.
- Keep all new CLI defaults disabled/zero for old-checkpoint compatibility.
- Accept A1 only if retrieval metrics are unchanged, matching-direction Spearman `rho >= 0.20`, AUROC is greater than `0.5`, and selective accuracy improves as high-risk queries are removed.
- Do not commit logs, checkpoints, generated data, or research clones.

---

### Task 1: Score features and binary Dirichlet head

**Files:**
- Create: `prob_models/retrieval_risk.py`
- Create: `tests/test_retrieval_risk_features.py`

**Interfaces:**
- Produces `score_distribution_features(logits, direction, temperature=1.0) -> Tensor[num_queries, 8]`.
- Produces `RetrievalRiskSidecar(feature_dim=8, hidden_dim=32, temperature=1.0)`.
- `forward()` returns `alpha`, `failure_risk`, `vacuity`, and `features`.

- [ ] **Step 1: Write failing feature tests**

```python
def test_features_are_candidate_permutation_invariant():
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.5, 0.4, 0.3]])
    actual = score_distribution_features(logits, "t2v")
    expected = score_distribution_features(logits[:, [2, 0, 1]], "t2v")
    torch.testing.assert_close(actual, expected)


def test_direction_selects_query_axis():
    logits = torch.randn(2, 3)
    assert score_distribution_features(logits, "t2v").shape == (2, 8)
    assert score_distribution_features(logits, "v2t").shape == (3, 8)


@pytest.mark.parametrize("bad", [torch.ones(3), torch.ones(2, 1)])
def test_features_reject_invalid_candidate_matrices(bad):
    with pytest.raises(ValueError):
        score_distribution_features(bad, "t2v")
```

- [ ] **Step 2: Run RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_risk_features.py
```

Expected: missing `prob_models.retrieval_risk`.

- [ ] **Step 3: Implement feature extraction**

Orient rows for T2V and `logits.T` for V2T. Validate a finite 2-D matrix, at least two candidates, a supported direction, and positive temperature. Produce these six statistics plus a two-value direction code:

```python
mean = scores.mean(1)
std = scores.std(1, unbiased=False)
z = (scores - mean[:, None]) / std.clamp_min(1e-6)[:, None]
top2 = z.topk(2, dim=1).values
prob = F.softmax(z / temperature, dim=1)
entropy = -(prob * prob.clamp_min(1e-12).log()).sum(1) / math.log(scores.size(1))
concentration = torch.logsumexp(z / temperature, 1) - math.log(scores.size(1))
features = [scores.max(1).values, mean, std, top2[:, 0] - top2[:, 1], entropy, concentration]
```

- [ ] **Step 4: Write failing Dirichlet-output test**

```python
def test_sidecar_returns_binary_dirichlet_outputs():
    output = RetrievalRiskSidecar(hidden_dim=16)(torch.randn(4, 5), "t2v")
    assert output["alpha"].shape == (4, 2)
    assert torch.all(output["alpha"] > 1)
    torch.testing.assert_close(
        output["alpha"].sum(1) * output["vacuity"],
        torch.full((4,), 2.0),
    )
```

- [ ] **Step 5: Implement the head with the detach boundary**

```python
class RetrievalRiskSidecar(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=32, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, logits, direction):
        features = score_distribution_features(logits.detach(), direction, self.temperature)
        alpha = F.softplus(self.network(features)) + 1.0
        strength = alpha.sum(1)
        return {"alpha": alpha, "failure_risk": alpha[:, 1] / strength,
                "vacuity": 2.0 / strength, "features": features}
```

- [ ] **Step 6: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_risk_features.py
git add prob_models/retrieval_risk.py tests/test_retrieval_risk_features.py
git commit -m "feat: add retrieval risk sidecar features"
```

---

### Task 2: Multi-positive evidential objective

**Files:**
- Modify: `prob_models/retrieval_risk.py`
- Create: `tests/test_retrieval_risk_loss.py`

**Interfaces:**
- Produces `positive_mask_from_group_ids(query_ids, candidate_ids=None)`.
- Produces `retrieval_hardness(logits, positive_mask, temperature=1.0)`.
- Produces `retrieval_risk_objective(sidecar, logits, positive_mask, temperature=1.0)`.

- [ ] **Step 1: Write failing multi-positive and detach tests**

```python
def test_group_ids_create_multi_positive_mask():
    mask = positive_mask_from_group_ids(torch.tensor([7, 7, 9]))
    assert mask.tolist() == [[True, True, False], [True, True, False], [False, False, True]]


def test_hardness_is_detached_and_increases_for_overtaking_negatives():
    mask = torch.tensor([[True, True, False], [False, False, True]])
    easy = torch.tensor([[5., 4., 0.], [0., 1., 5.]])
    hard = torch.tensor([[1., 0., 2.], [3., 2., 1.]], requires_grad=True)
    assert torch.all(retrieval_hardness(hard, mask) > retrieval_hardness(easy, mask))
    assert not retrieval_hardness(hard, mask).requires_grad
```

- [ ] **Step 2: Run RED, then implement mask validation and hardness**

```python
positive = torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
negative = torch.logsumexp(scores.masked_fill(mask, -torch.inf), dim=1)
hardness = F.softplus((negative - positive) / temperature).detach()
```

Reject shape mismatches, rows without positives, rows without negatives, non-finite logits, and non-positive temperatures.

- [ ] **Step 3: Write failing objective isolation test**

```python
def test_risk_objective_updates_only_sidecar():
    logits = torch.randn(4, 4, requires_grad=True)
    sidecar = RetrievalRiskSidecar(hidden_dim=8)
    loss, output = retrieval_risk_objective(sidecar, logits, torch.eye(4, dtype=torch.bool))
    loss.backward()
    assert logits.grad is None
    assert sum(p.grad.abs().sum() for p in sidecar.parameters()) > 0
    assert {"edl", "fit", "rank", "t2v", "v2t"} <= output.keys()
```

- [ ] **Step 4: Implement both directional objectives**

For each direction, compute `failure` from whether the top-ranked candidate belongs to the positive mask. Combine:

```python
target = torch.stack((1.0 - failure, failure), dim=1)
strength = alpha.sum(1, keepdim=True)
probability = alpha / strength
edl = (((target - probability) ** 2).sum(1) +
       (alpha * (strength - alpha) / (strength.square() * (strength + 1))).sum(1)).mean()
fit = F.smooth_l1_loss(failure_risk, robust_rank_normalize(hardness))
rank = pairwise_uncertainty_ranking_loss(failure_risk, hardness, margin=0.05)
```

Average T2V and V2T totals. Return detached diagnostics but keep the aggregate loss differentiable.

- [ ] **Step 5: Add a deterministic tiny-training behavior test**

Train an 8-hidden-unit sidecar for 120 Adam steps on two clear and two ambiguous rows with `torch.manual_seed(0)`. Assert mean ambiguous risk exceeds mean clear risk.

- [ ] **Step 6: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_risk_features.py tests/test_retrieval_risk_loss.py
git add prob_models/retrieval_risk.py tests/test_retrieval_risk_loss.py
git commit -m "feat: train evidential retrieval failure risk"
```

---

### Task 3: June-model integration and frozen-sidecar mode

**Files:**
- Modify: `modules/modeling_mulit.py:164-439,650-861`
- Modify: `main_task_retrieval.py:240-490,657-735,1010-1080,1635-1680`
- Create: `tests/test_retrieval_risk_integration.py`

**Interfaces:**
- Adds CLI flags `--use_retrieval_risk_sidecar`, `--retrieval_risk_sidecar_only`, `--w_retrieval_risk`, `--retrieval_risk_temperature`, and `--retrieval_risk_hidden_dim`.
- Adds `configure_retrieval_risk_sidecar_training(model, args)`.
- Adds `UATVR.score_retrieval_risk(logits, direction)`.
- Adds loss keys `retrieval_risk_loss`, `retrieval_risk_edl`, `retrieval_risk_fit`, and `retrieval_risk_rank`.

- [ ] **Step 1: Write failing CLI and freeze tests**

```python
def test_old_command_line_keeps_sidecar_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_task_retrieval.py", "--do_eval"])
    args = get_args()
    assert not args.use_retrieval_risk_sidecar
    assert not args.retrieval_risk_sidecar_only
    assert args.w_retrieval_risk == 0.0


def test_sidecar_only_freezes_every_other_parameter():
    model = TinyModelWithBackboneAndSidecar()
    configure_retrieval_risk_sidecar_training(
        model, SimpleNamespace(use_retrieval_risk_sidecar=True,
                               retrieval_risk_sidecar_only=True)
    )
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "retrieval_risk_sidecar.weight", "retrieval_risk_sidecar.bias"
    }
```

- [ ] **Step 2: Run RED and add backward-compatible arguments**

```python
parser.add_argument("--use_retrieval_risk_sidecar", action="store_true")
parser.add_argument("--retrieval_risk_sidecar_only", action="store_true")
parser.add_argument("--w_retrieval_risk", type=float, default=0.0)
parser.add_argument("--retrieval_risk_temperature", type=float, default=1.0)
parser.add_argument("--retrieval_risk_hidden_dim", type=int, default=32)
```

The freeze helper raises when sidecar-only is requested without sidecar mode, freezes every parameter, then re-enables only the sidecar parameters. Invoke it after the existing CLIP freeze block and before optimizer creation.

- [ ] **Step 3: Instantiate and wire the sidecar**

In `UATVR.__init__`, create `RetrievalRiskSidecar` only when enabled. Extend `forward(..., group_ids=None)` and `get_similarity_logits(..., group_ids=None)`. Import and use `allgather_no_grad` for integer group IDs, preserving the same rank order as the gathered text/video features, then construct the global positive mask. Add a two-rank test around the existing distributed gather harness to verify local IDs `[3, 7]` and `[7, 9]` become `[3, 7, 7, 9]` on both ranks.

When sidecar mode is enabled:

```python
risk_loss, risk_output = retrieval_risk_objective(
    self.retrieval_risk_sidecar, wti_logits, positive_mask,
    self.retrieval_risk_temperature,
)
loss["retrieve_logits"] = wti_logits
loss["retrieval_risk_loss"] = self.w_retrieval_risk * risk_loss
```

In sidecar-only mode, skip Gaussian sampling and set legacy MIL/evidential/negative/orthogonal losses to scalar zero. Never construct or return `weighted_logits` on this path.

- [ ] **Step 4: Pass group IDs and expose diagnostics**

Retain `_group_ids` in `train_epoch` and pass `group_ids=_group_ids`. Add risk loss to `total`. After `model.train()`, call `model.eval()` in sidecar-only mode and restore only `retrieval_risk_sidecar.train()` so frozen dropout state cannot change feature targets.

- [ ] **Step 5: Add exact score-path and checkpoint-default tests**

Test that sidecar-enabled retrieval logits equal raw WTI logits bit-for-bit and that risk backward leaves the raw logits without a gradient. Instantiate with an old argument namespace and assert no sidecar parameters are required. Load a state dictionary that contains no `retrieval_risk_sidecar.*` keys while enabling a newly initialized sidecar and assert only those keys are reported missing; inject an unrelated unexpected key and assert it is rejected. Save/reload a trained sidecar state dictionary and assert identical outputs.

- [ ] **Step 6: Run focused verification and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_risk_features.py tests/test_retrieval_risk_loss.py tests/test_retrieval_risk_integration.py
/home/xujie/miniconda3/envs/ret/bin/python -m py_compile prob_models/retrieval_risk.py modules/modeling_mulit.py main_task_retrieval.py
git diff --check
git add modules/modeling_mulit.py main_task_retrieval.py tests/test_retrieval_risk_integration.py
git commit -m "feat: integrate frozen retrieval risk sidecar"
```

---

### Task 4: Risk metrics and complete-matrix evaluation

**Files:**
- Create: `prob_models/retrieval_risk_metrics.py`
- Create: `tests/test_retrieval_risk_metrics.py`
- Modify: `main_task_retrieval.py:1320-1620`
- Modify: `tests/test_directional_eval.py`

**Interfaces:**
- Produces `build_eval_positive_mask(num_text, num_video, cut_off_points=None)`.
- Produces `positive_ranks(scores, positive_mask)`.
- Produces `evaluate_retrieval_risk(scores, positive_mask, risk, vacuity, bootstrap_samples=1000, seed=0)`.
- Extends `eval_epoch()` with top-level `retrieval_risk` while leaving `t2v` and `v2t` dictionaries unchanged.

- [ ] **Step 1: Write failing multi-sentence mask and rank tests**

```python
def test_multi_sentence_mask_maps_caption_ranges_to_videos():
    mask = build_eval_positive_mask(5, 2, [2, 5])
    assert mask.tolist() == [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]


def test_positive_ranks_support_multiple_positives():
    scores = np.array([[.8, .9, .1], [.7, .6, .5]])
    mask = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    assert positive_ranks(scores, mask).tolist() == [0, 2]
```

- [ ] **Step 2: Run RED and implement masks/ranks**

```python
best_positive = np.where(mask, scores, -np.inf).max(axis=1)
ranks = (scores > best_positive[:, None]).sum(axis=1).astype(np.int64)
```

- [ ] **Step 3: Write failing metric-quality test**

```python
def test_perfect_risk_has_perfect_auroc_and_selective_accuracy():
    result = evaluate_retrieval_risk(
        scores, np.eye(4, dtype=bool),
        np.array([.1, .2, .8, .9]), np.full(4, .3),
        bootstrap_samples=100, seed=3,
    )
    assert result["failure_auroc"]["value"] == 1.0
    assert result["risk_coverage"]["accuracy_at_0.50"] == 1.0
    assert result["spearman_risk_vs_rank"] > 0
```

- [ ] **Step 4: Implement JSON-safe metrics**

Use `scipy.stats.spearmanr`, average-rank Mann-Whitney AUROC, deterministic stratified bootstrap 95% CI, and ascending-risk prefix accuracy. Report AURC plus coverages `0.25`, `0.50`, `0.75`, `1.00`, and min/max/mean/std for risk and vacuity. Raise clear errors for constant failure labels rather than emitting silent NaNs.

- [ ] **Step 5: Integrate after complete similarity assembly**

Preserve the complete raw similarity matrix after `_run_on_single_gpu()`. Build a multi-sentence positive mask from `cut_off_points_` or identity for one-to-one evaluation. Score risk in query chunks if needed, retaining the full candidate dimension. Evaluate T2V with rows/mask and V2T with transposed rows/mask. Do not alter `metric_t2v_matrix` or `metric_v2t_matrix`.

- [ ] **Step 6: Add exact retrieval-regression test**

Run `_compute_directional_metrics` on a fixed matrix with risk reporting disabled and enabled. Require exact equality of the existing `t2v`/`v2t` dictionaries and independently assert the `retrieval_risk` payload.

- [ ] **Step 7: Run GREEN and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_risk_metrics.py tests/test_directional_eval.py tests/test_retrieval_risk_integration.py
/home/xujie/miniconda3/envs/ret/bin/python -m py_compile prob_models/retrieval_risk_metrics.py main_task_retrieval.py
git diff --check
git add prob_models/retrieval_risk_metrics.py main_task_retrieval.py tests/test_retrieval_risk_metrics.py tests/test_directional_eval.py
git commit -m "feat: evaluate retrieval failure risk"
```

---

### Task 5: A1 launcher and final verification

**Files:**
- Create: `run_train_msrvtt_risk_sidecar_bg.sh`
- Create: `tests/test_retrieval_risk_launcher.py`
- Modify: `docs/superpowers/tables/2026-08-11_r1_and_rho.md`

**Interfaces:**
- Produces a narrow wrapper around `run_train_msrvtt_bg.sh` using June `bin.4` and the trusted-v1 split.

- [ ] **Step 1: Write failing launcher-contract test**

```python
def test_launcher_uses_frozen_june_sidecar_contract():
    text = Path("run_train_msrvtt_risk_sidecar_bg.sh").read_text()
    assert "ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4" in text
    for flag in ("--use_uatvr_full", "--use_retrieval_risk_sidecar",
                 "--retrieval_risk_sidecar_only", "--w_retrieval_risk"):
        assert flag in text
```

- [ ] **Step 2: Run RED and create the wrapper**

The script validates `BASE_CKPT`, sets a `risk_sidecar_a1` run tag, then calls the existing background launcher with:

```bash
--use_uatvr_full
--init_model "${BASE_CKPT}"
--use_retrieval_risk_sidecar
--retrieval_risk_sidecar_only
--w_retrieval_risk 0.05
--retrieval_risk_temperature 1.0
--retrieval_risk_hidden_dim 32
```

- [ ] **Step 3: Record the immutable experiment contract**

Append the initialization, trainable parameter prefix, raw-WTI score path, loss weight, temperature, and acceptance gates to `docs/superpowers/tables/2026-08-11_r1_and_rho.md`.

- [ ] **Step 4: Run the complete relevant suite**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q \
  tests/test_retrieval_risk_features.py \
  tests/test_retrieval_risk_loss.py \
  tests/test_retrieval_risk_integration.py \
  tests/test_retrieval_risk_metrics.py \
  tests/test_retrieval_risk_launcher.py \
  tests/test_directional_eval.py \
  tests/test_multi_positive_loss.py \
  tests/test_experiment_tracking.py
/home/xujie/miniconda3/envs/ret/bin/python -m py_compile \
  prob_models/retrieval_risk.py prob_models/retrieval_risk_metrics.py \
  modules/modeling_mulit.py main_task_retrieval.py
bash -n run_train_msrvtt_bg.sh run_train_msrvtt_risk_sidecar_bg.sh
git diff --check
```

- [ ] **Step 5: Inspect scope and commit**

```bash
git status --short
git diff --stat HEAD~4
git add run_train_msrvtt_risk_sidecar_bg.sh tests/test_retrieval_risk_launcher.py docs/superpowers/tables/2026-08-11_r1_and_rho.md
git commit -m "chore: add frozen risk sidecar experiment"
```

Do not launch the multi-GPU experiment during code development unless the user explicitly requests it after reviewing verification results.
