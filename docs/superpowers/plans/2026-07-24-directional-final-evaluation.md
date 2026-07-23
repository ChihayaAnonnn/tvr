# Directional Final Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make final test evaluation compute, log, return, and persist only the retrieval direction that selected each checkpoint.

**Architecture:** Add a normalized `directions` keyword argument to metric aggregation, retaining default bidirectional validation. `eval_epoch` will use this to avoid constructing or logging unrequested direction metrics; the final-test loop passes its selection key as the sole requested direction.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest.

## Global Constraints

- `eval_epoch` default remains `("t2v", "v2t")` for all existing validation callers.
- Accepted directions are exactly `t2v` and `v2t`; empty or invalid input raises `ValueError`.
- Single-direction final testing must retain shared feature extraction/similarity computation.
- No test split data may influence checkpoint selection.

---

### Task 1: Direction-aware metric helper

**Files:**
- Modify: `main_task_retrieval.py:1138-1167`
- Test: `tests/test_rspr_rerank.py`

**Interfaces:**
- Produces: `_compute_directional_metrics(..., directions=("t2v", "v2t")) -> dict[str, dict]`.
- Consumes: T2V/V2T score matrices and optional multi-caption cut-off points.

- [ ] **Step 1: Write the failing tests**

```python
def test_directional_metrics_only_evaluates_requested_t2v(monkeypatch):
    monkeypatch.setattr(main_task_retrieval, "compute_metrics", lambda matrix: {"rows": len(matrix)})
    metrics = main_task_retrieval._compute_directional_metrics(
        np.eye(2), np.eye(2), directions=("t2v",)
    )
    assert metrics == {"t2v": {"rows": 2}}


def test_directional_metrics_rejects_empty_and_unknown_direction():
    with pytest.raises(ValueError, match="directions"):
        main_task_retrieval._compute_directional_metrics(np.eye(2), np.eye(2), directions=())
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_rspr_rerank.py -k 'directional_metrics_only or directional_metrics_rejects' -v`
Expected: FAIL because `_compute_directional_metrics` has no `directions` parameter.

- [ ] **Step 3: Implement the minimal helper change**

```python
def _compute_directional_metrics(..., directions=("t2v", "v2t")):
    directions = _normalize_eval_directions(directions)
    metrics = {}
    if "t2v" in directions:
        metrics["t2v"] = ...
    if "v2t" in directions:
        metrics["v2t"] = ...
    return metrics
```

For multi-caption evaluation, only reshape/invoke the requested direction's matrix.

- [ ] **Step 4: Run the focused tests to verify success**

Run: `pytest tests/test_rspr_rerank.py -k 'directional_metrics_only or directional_metrics_rejects' -v`
Expected: PASS.

### Task 2: Eval logging, return payload, and final-test caller

**Files:**
- Modify: `main_task_retrieval.py:1177-1416,1659-1678`
- Test: `tests/test_rspr_rerank.py`

**Interfaces:**
- Consumes: `eval_epoch(..., directions=(direction,))`.
- Produces: result dictionary containing only requested keys and `final_test.json` selections whose `test_metrics` has the matching key.

- [ ] **Step 1: Write the failing test**

```python
def test_eval_epoch_single_direction_returns_only_that_direction(monkeypatch):
    monkeypatch.setattr(main_task_retrieval, "_run_on_single_gpu", lambda *_args: [np.eye(2)])
    metrics = main_task_retrieval.eval_epoch(
        args, model, empty_loader, torch.device("cpu"), n_gpu=1, directions=("v2t",)
    )
    assert set(metrics) == {"v2t"}
```

- [ ] **Step 2: Run the test to verify failure**

Run: `pytest tests/test_rspr_rerank.py::test_eval_epoch_single_direction_returns_only_that_direction -v`
Expected: FAIL because `eval_epoch` has no `directions` parameter.

- [ ] **Step 3: Implement the minimal call-site changes**

```python
def eval_epoch(..., directions=("t2v", "v2t")):
    directions = _normalize_eval_directions(directions)
    metrics = _compute_directional_metrics(..., directions=directions)
    for direction in directions:
        logger.info(..., metrics[direction]["R1"], ...)
    return {direction: _serialize_retrieval_metrics(metrics[direction]) for direction in directions}

# In the final-test loop:
metrics = eval_epoch(..., directions=(direction,))
```

- [ ] **Step 4: Run focused and regression tests**

Run: `pytest tests/test_rspr_rerank.py -v`
Expected: PASS, including the pre-existing bidirectional evaluation test.

### Task 3: Static and behavior verification

**Files:**
- Modify: none

- [ ] **Step 1: Verify syntax and test suite**

Run: `python -m py_compile main_task_retrieval.py && pytest tests/test_rspr_rerank.py -v`
Expected: exit code 0.

- [ ] **Step 2: Inspect final diff**

Run: `git diff --check && git diff -- main_task_retrieval.py tests/test_rspr_rerank.py`
Expected: no whitespace errors; default two-direction behavior and single-direction final calls are visible.
