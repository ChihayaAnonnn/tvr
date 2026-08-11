# SAP Retrieval-Grounded Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SAP uncertainty measurable from the correct model outputs and align it with text-video retrieval failures while preserving MSRVTT retrieval quality.

**Architecture:** Keep semantic anchors as the video representation mechanism, but separate anchor allocation, Gaussian embedding variance, and retrieval risk. Measure the existing route 1+3 checkpoint first; if it misses the registered gate, add candidate-space evidential learning and a detached retrieval-hardness ranking head. Add closed-form probabilistic distance only as a fallback.

**Tech Stack:** Python 3.10, PyTorch, SciPy, pytest, existing UATVR/MSRVTT evaluation pipeline.

## Global Constraints

- Do not modify or restart the active route 1+3 run `ckpts/ckpt_msrvtt_20260811_194405/`.
- Preserve backward loading of checkpoints without new uncertainty-head parameters.
- Keep JSFUSION T2V R@1 without DSL at or above 48.0.
- The registered uncertainty target is matching-direction Spearman `rho >= 0.20`.
- Never use CRF `u_video` as a proxy for SAP output in the SAP evaluator.
- Use detached retrieval hardness so the uncertainty loss cannot worsen retrieval to make its target easier.
- Do not call anchor entropy retrieval uncertainty unless it passes the retrieval-grounding evaluation.
- Do not commit logs, checkpoints, generated videos, or temporary research repositories.

---

### Task 1: Expose and evaluate the existing SAP signals

**Files:**
- Modify: `modules/modeling_mulit.py`
- Replace: `scripts/diagnose_uncertainty_retrieval_correlation.py`
- Create: `tests/test_sap_uncertainty_evaluator.py`

**Interfaces:**
- Produces: `UATVR.score_sap_uncertainty(video, video_mask) -> dict[str, Tensor]`.
- Produces evaluator keys `logsigma_video`, `u_mode`, and `epistemic_cont` with distribution statistics and three correlations.
- Consumes the completed route 1+3 checkpoints without retraining.

- [ ] **Step 1: Write failing provenance and shape tests**

```python
class FakeModel:
    def score_video_risk(self, *_):
        raise AssertionError("CRF path used")

    def score_sap_uncertainty(self, video, video_mask):
        batch = video.shape[0]
        return {
            "logsigma_video": torch.arange(batch, dtype=torch.float32),
            "u_mode": torch.arange(batch, dtype=torch.float32) + 1,
            "epistemic_cont": torch.arange(batch, dtype=torch.float32) + 2,
        }


def fake_batch(batch=3):
    ids = torch.zeros(batch, 1, dtype=torch.long)
    mask = torch.ones_like(ids)
    segment = torch.zeros_like(ids)
    video = torch.zeros(batch, 1, 1, 2, 3, 8, 8)
    video_mask = torch.ones(batch, 1, 2)
    return ids, mask, segment, video, video_mask


def test_sap_evaluator_does_not_call_crf(monkeypatch):
    result = collect_sap_uncertainty(
        FakeModel(), [fake_batch(batch=3)], torch.device("cpu")
    )
    assert set(result) == {"logsigma_video", "u_mode", "epistemic_cont"}


def test_sap_collection_has_one_value_per_video():
    result = collect_sap_uncertainty(
        FakeModel(), [fake_batch(batch=3)], torch.device("cpu")
    )
    assert all(value.shape == (3,) for value in result.values())
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_sap_uncertainty_evaluator.py
```

Expected: failure because the SAP scoring API and collection helper do not exist.

- [ ] **Step 3: Add the SAP-only scoring API**

Reuse `get_visual_output`; flatten `visual_hidden` exactly as `_loose_similarity` does and never invoke `crf_head`:

```python
@torch.no_grad()
def score_sap_uncertainty(self, video, video_mask):
    video = torch.as_tensor(video).float()
    batch = video.shape[0]
    video_flat = video.reshape(-1, *video.shape[-3:])
    frame_num = video_flat.shape[0] // batch
    mask_flat = video_mask.view(-1, video_mask.shape[-1])
    _, hidden = self.get_visual_output(
        video_flat, mask_flat, shaped=True, video_frame=frame_num
    )
    time, spatial, dim = hidden.shape[1:]
    memory = hidden.reshape(batch, time * spatial, dim)
    memory_mask = mask_flat[:, :time].unsqueeze(-1).expand(-1, -1, spatial)
    sap = self.sap(memory, padding_mask=(memory_mask.reshape(batch, -1) == 0))
    return {
        "logsigma_video": sap["logsigma"].exp().mean(dim=-1),
        "u_mode": sap["u_mode"],
        "epistemic_cont": sap["epistemic_cont"].flatten(1).mean(dim=1),
    }
```

- [ ] **Step 4: Replace the evaluator's CRF pass**

Keep the existing `eval_epoch` similarity capture. Collect the SAP dictionary and compute each signal against `t2v_rank`, `v2t_rank`, and `maximum(t2v_rank, v2t_rank)`. Write `"source": "sap"` to JSON and assert signal length equals the similarity diagonal length.

- [ ] **Step 5: Run focused and regression tests**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_sap_uncertainty_evaluator.py tests/test_evaluate_crf_ladder.py
```

- [ ] **Step 6: Evaluate every route 1+3 checkpoint**

Run `scripts/diagnose_uncertainty_retrieval_correlation.py` once per `pytorch_model.bin.*`, storing `*.sap_uncertainty.json` next to each checkpoint.

Decision: stop this plan if any checkpoint has T2V R@1 `>= 48.0` and a genuine SAP signal has matching-direction `rho >= 0.20`; otherwise continue.

- [ ] **Step 7: Commit**

```bash
git add modules/modeling_mulit.py scripts/diagnose_uncertainty_retrieval_correlation.py tests/test_sap_uncertainty_evaluator.py
git commit -m "fix: evaluate SAP uncertainty independently of CRF"
```

---

### Task 2: Add candidate-space evidential retrieval utilities

**Files:**
- Create: `prob_models/retrieval_uncertainty.py`
- Create: `tests/test_retrieval_uncertainty.py`

**Interfaces:**
- Produces: `retrieval_evidential_loss(logits, positive_mask, temperature) -> tuple[Tensor, dict[str, Tensor]]`.
- Produces `u_t2v` over rows and `u_v2t` over columns.

- [ ] **Step 1: Write failing numerical tests**

```python
def test_clear_match_has_lower_candidate_uncertainty():
    clear = torch.tensor([[8.0, -4.0], [-4.0, 8.0]])
    ambiguous = torch.zeros(2, 2)
    mask = torch.eye(2, dtype=torch.bool)
    _, clear_out = retrieval_evidential_loss(clear, mask, 1.0)
    _, ambiguous_out = retrieval_evidential_loss(ambiguous, mask, 1.0)
    assert torch.all(clear_out["u_t2v"] < ambiguous_out["u_t2v"])
    assert torch.all(clear_out["u_v2t"] < ambiguous_out["u_v2t"])


def test_loss_reaches_all_similarity_entries():
    logits = torch.randn(4, 4, requires_grad=True)
    loss, _ = retrieval_evidential_loss(logits, torch.eye(4, dtype=torch.bool), 1.0)
    loss.backward()
    assert torch.count_nonzero(logits.grad) == logits.numel()
```

- [ ] **Step 2: Run tests and verify missing-module failure**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_uncertainty.py
```

- [ ] **Step 3: Implement stable Dirichlet evidence and EDL MSE**

```python
def retrieval_evidential_loss(logits, positive_mask, temperature=1.0):
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must be equal 2-D matrices")
    alpha = F.softplus(logits / temperature) + 1.0

    def direction(cur_alpha, cur_target):
        strength = cur_alpha.sum(dim=1, keepdim=True)
        prob = cur_alpha / strength
        target = cur_target.float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1)
        fit = ((target - prob) ** 2).sum(dim=1)
        variance = (
            cur_alpha * (strength - cur_alpha)
            / (strength.square() * (strength + 1.0))
        ).sum(dim=1)
        return (fit + variance).mean(), cur_alpha.shape[1] / strength.squeeze(1)

    loss_t2v, u_t2v = direction(alpha, positive_mask)
    loss_v2t, u_v2t = direction(alpha.T, positive_mask.T)
    return 0.5 * (loss_t2v + loss_v2t), {
        "alpha": alpha, "u_t2v": u_t2v, "u_v2t": u_v2t
    }
```

- [ ] **Step 4: Run tests and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_uncertainty.py
git add prob_models/retrieval_uncertainty.py tests/test_retrieval_uncertainty.py
git commit -m "feat: add candidate-space evidential retrieval loss"
```

---

### Task 3: Integrate retrieval evidence without reusing anchor evidence

**Files:**
- Modify: `main_task_retrieval.py`
- Modify: `modules/modeling_mulit.py`
- Modify: `run_train_msrvtt_bg.sh`
- Create: `tests/test_retrieval_edl_integration.py`

**Interfaces:**
- Adds CLI options `--w_retrieval_edl` and `--retrieval_edl_temperature`.
- Produces diagnostics for the mean and cross-batch standard deviation of `u_t2v` and `u_v2t`.

- [ ] **Step 1: Write failing default and gradient tests**

```python
def test_retrieval_edl_is_off_for_old_checkpoints(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_task_retrieval.py", "--do_eval"])
    args = get_args()
    assert args.w_retrieval_edl == 0.0
    assert args.retrieval_edl_temperature == 1.0


def test_retrieval_edl_backpropagates():
    model = object.__new__(UATVR)
    nn.Module.__init__(model)
    model.retrieval_edl_temperature = 1.0
    logits = torch.randn(8, 8, requires_grad=True)
    model._retrieval_edl(logits).backward()
    assert logits.grad.abs().sum() > 0
```

- [ ] **Step 2: Add backward-compatible CLI configuration**

```python
parser.add_argument("--w_retrieval_edl", type=float, default=0.0)
parser.add_argument("--retrieval_edl_temperature", type=float, default=1.0)
```

Copy both values into `UATVR.__init__` as `self.w_retrieval_edl` and `self.retrieval_edl_temperature`.

- [ ] **Step 3: Compute EDL from raw WTI logits**

Use `wti_logits` before any confidence multiplication:

```python
def _retrieval_edl(self, wti_logits):
    positive_mask = torch.eye(
        wti_logits.size(0), dtype=torch.bool, device=wti_logits.device
    )
    edl, _ = retrieval_evidential_loss(
        wti_logits, positive_mask, self.retrieval_edl_temperature
    )
    return edl


edl, retrieval_u = retrieval_evidential_loss(
    wti_logits,
    torch.eye(wti_logits.size(0), dtype=torch.bool, device=wti_logits.device),
    self.retrieval_edl_temperature,
)
loss["retrieval_edl_loss"] = self.w_retrieval_edl * edl
```

Rename `_evidential_nll_loss` to `_legacy_similarity_regularizer`; it does not use `alpha_dir` and must not claim to supervise anchor evidence. New experiments use `--w_retrieval_edl 0.05`; old checkpoints retain zero.

- [ ] **Step 4: Run verification and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_uncertainty.py tests/test_retrieval_edl_integration.py
/home/xujie/miniconda3/envs/ret/bin/python -m py_compile main_task_retrieval.py modules/modeling_mulit.py
bash -n run_train_msrvtt_bg.sh
git add main_task_retrieval.py modules/modeling_mulit.py run_train_msrvtt_bg.sh tests/test_retrieval_edl_integration.py
git commit -m "feat: train retrieval uncertainty from candidate evidence"
```

---

### Task 4: Ground a dedicated video-risk head in retrieval hardness

**Files:**
- Modify: `query_models/module_sap.py`
- Modify: `prob_models/retrieval_uncertainty.py`
- Modify: `modules/modeling_mulit.py`
- Modify: `main_task_retrieval.py`
- Create: `tests/test_retrieval_hardness.py`

**Interfaces:**
- Produces: `retrieval_hardness(logits, temperature) -> tuple[Tensor, Tensor]`.
- Produces: `robust_rank_normalize(values) -> Tensor` and `pairwise_uncertainty_ranking_loss(predicted, hardness, margin) -> Tensor`.
- Adds SAP output `risk_video: Tensor[B]` without changing `u_mode`.

- [ ] **Step 1: Write failing monotonicity and detach tests**

```python
def test_hardness_increases_when_negative_overtakes_positive():
    easy = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    hard = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    easy_t2v, easy_v2t = retrieval_hardness(easy, 1.0)
    hard_t2v, hard_v2t = retrieval_hardness(hard, 1.0)
    assert torch.all(hard_t2v > easy_t2v)
    assert torch.all(hard_v2t > easy_v2t)


def test_hardness_target_is_detached():
    logits = torch.randn(4, 4, requires_grad=True)
    target, _ = retrieval_hardness(logits, 1.0)
    assert not target.requires_grad
```

- [ ] **Step 2: Implement detached soft-rank hardness**

```python
def retrieval_hardness(logits, temperature=1.0):
    size = logits.size(0)
    eye = torch.eye(size, dtype=torch.bool, device=logits.device)
    positive = logits.diagonal()
    row_neg = logits.masked_fill(eye, -torch.inf)
    col_neg = logits.T.masked_fill(eye, -torch.inf)
    h_t2v = F.softplus(torch.logsumexp(row_neg / temperature, 1) - positive / temperature)
    h_v2t = F.softplus(torch.logsumexp(col_neg / temperature, 1) - positive / temperature)
    return h_t2v.detach(), h_v2t.detach()


def robust_rank_normalize(values):
    order = torch.argsort(torch.argsort(values))
    return order.to(values.dtype) / max(values.numel() - 1, 1)


def pairwise_uncertainty_ranking_loss(predicted, hardness, margin=0.05):
    hardness_delta = hardness[:, None] - hardness[None, :]
    valid = hardness_delta.abs() > margin
    if not valid.any():
        return predicted.sum() * 0.0
    predicted_delta = predicted[:, None] - predicted[None, :]
    direction = hardness_delta.sign()
    return F.softplus(-direction[valid] * predicted_delta[valid]).mean()
```

- [ ] **Step 3: Add a scalar risk head to SAP**

```python
self.risk_head = nn.Sequential(
    nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1)
)
nn.init.zeros_(self.risk_head[-1].weight)
nn.init.zeros_(self.risk_head[-1].bias)
risk_video = torch.sigmoid(self.risk_head(mu_agg)).squeeze(-1)
```

Return `risk_video` from SAP, gather it with the other compact SAP tensors during DDP training, and add it to `score_sap_uncertainty`. This keeps its length aligned with global `wti_logits` and makes the corrected evaluator report `risk_video` after Task 4.

- [ ] **Step 4: Train intrinsic video risk against V2T hardness**

```python
_, hardness_v2t = retrieval_hardness(wti_logits, self.hardness_temperature)
target = robust_rank_normalize(hardness_v2t)
fit = F.smooth_l1_loss(risk_video, target)
rank = pairwise_uncertainty_ranking_loss(risk_video, hardness_v2t, margin=0.05)
loss["risk_grounding_loss"] = self.w_risk_grounding * (fit + rank)
```

Use defaults `w_risk_grounding=0.0` and `hardness_temperature=1.0`; the first new run passes `--w_risk_grounding 0.05`. T2V uncertainty remains query-conditioned and comes from Task 3.

- [ ] **Step 5: Verify checkpoint compatibility and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_retrieval_hardness.py tests/test_retrieval_edl_integration.py tests/test_sap_uncertainty_evaluator.py
git add query_models/module_sap.py prob_models/retrieval_uncertainty.py modules/modeling_mulit.py main_task_retrieval.py tests/test_retrieval_hardness.py
git commit -m "feat: ground SAP video risk in retrieval hardness"
```

---

### Task 5: Add consistent Gaussian semantics and closed-form distance if needed

Execute only if Task 4 still gives `rho < 0.20`.

**Files:**
- Modify: `prob_models/tensor_utils.py`
- Modify: `modules/until_module.py`
- Modify: `modules/modeling_mulit.py`
- Modify: `main_task_retrieval.py`
- Create: `tests/test_gaussian_parameterization.py`
- Create: `tests/test_closed_form_probabilistic_distance.py`

**Interfaces:**
- Adds `--gaussian_parameterization {legacy_logstd,logvar}`.
- Produces closed-form text-video Gaussian logits.

- [ ] **Step 1: Write tests that distinguish log-standard-deviation from log-variance**

```python
def test_logvar_sampling_uses_half_exponent(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda *shape, **kw: torch.ones(*shape, **kw))
    sample = sample_gaussian_tensors(
        torch.zeros(1, 2), torch.full((1, 2), math.log(4.0)), 1, "logvar"
    )
    assert torch.allclose(sample, torch.full_like(sample, 2.0))
```

- [ ] **Step 2: Implement explicit conversion**

```python
def gaussian_std(logscale, parameterization):
    if parameterization == "logvar":
        return torch.exp(0.5 * logscale)
    if parameterization == "legacy_logstd":
        return torch.exp(logscale)
    raise ValueError(f"unknown Gaussian parameterization: {parameterization}")
```

Checkpoints without the new argument restore `legacy_logstd`; new runs explicitly use `logvar`.

- [ ] **Step 3: Add a PCME++-style closed-form distance**

```python
mu_distance = ((video_mu[:, None] - text_mu[None, :]) ** 2).mean(dim=-1)
variance_distance = (video_logvar.exp()[:, None] + text_logvar.exp()[None, :]).mean(dim=-1)
distance = mu_distance + variance_distance
logits = shift - F.softplus(scale) * distance
```

Use match-matrix BCE, initialize `scale=5`, `shift=5`, and add VIB with weight `1e-4`.

- [ ] **Step 4: Verify and commit**

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pytest -q tests/test_gaussian_parameterization.py tests/test_closed_form_probabilistic_distance.py
git add prob_models/tensor_utils.py modules/until_module.py modules/modeling_mulit.py main_task_retrieval.py tests/test_gaussian_parameterization.py tests/test_closed_form_probabilistic_distance.py
git commit -m "feat: add consistent Gaussian distance training"
```

---

### Task 6: Run the registered ablation and write the decision report

**Files:**
- Create: `docs/superpowers/reports/2026-08-12-sap-retrieval-grounded-uncertainty-results.md`
- Modify: `docs/superpowers/tables/2026-08-11_r1_and_rho.md`

- [ ] **Step 1: Run the smallest isolating matrix**

1. E0: route 1+3 checkpoint with corrected evaluator.
2. E1: E0 plus candidate EDL, `w_retrieval_edl=0.05`.
3. E2: E1 plus video-risk grounding, `w_risk_grounding=0.05`.
4. E3: E2 plus closed-form Gaussian distance only if E2 misses the gate.

All new training runs use the same June `bin.4`, trusted-v1 split, five epochs, and effective batch 256.

- [ ] **Step 2: Record frozen metrics**

Record T2V/V2T R@1, R@5, R@10; uncertainty distribution statistics; matching-direction Spearman correlations; failure AUROC for rank greater than zero; and risk-coverage AURC.

- [ ] **Step 3: Apply the decision rule**

Accept the smallest configuration satisfying T2V R@1 `>= 48.0` and matching-direction `rho >= 0.20`. If none passes, retain SAP only as multi-anchor pooling and move retrieval uncertainty to limitations.

- [ ] **Step 4: Check report completeness and commit**

```bash
rg -n "TBD|TODO|PLACEHOLDER" docs/superpowers/reports/2026-08-12-sap-retrieval-grounded-uncertainty-results.md
git add docs/superpowers/reports/2026-08-12-sap-retrieval-grounded-uncertainty-results.md docs/superpowers/tables/2026-08-11_r1_and_rho.md
git commit -m "docs: report retrieval-grounded SAP ablation"
```
