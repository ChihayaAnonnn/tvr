import numpy as np
import torch

from main_task_retrieval import (
    _apply_dual_softmax,
    _compute_directional_metrics,
    _compute_retrieval_risk_metrics,
)


class _DeterministicRiskModel:
    use_retrieval_risk_sidecar = True

    def __init__(self):
        self.seen_shapes = []

    def score_retrieval_risk(self, logits, direction):
        self.seen_shapes.append((direction, tuple(logits.shape)))
        size = logits.size(0) if direction == "t2v" else logits.size(1)
        risk = torch.linspace(0.1, 0.9, size, device=logits.device)
        return {
            "failure_risk": risk,
            "vacuity": torch.full_like(risk, 0.4),
        }


def _matrix_with_successes_and_failures():
    return np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 5.0, 3.0, 0.0],
            [5.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )


def test_risk_reporting_preserves_directional_metrics_exactly():
    matrix = _matrix_with_successes_and_failures()
    before = _compute_directional_metrics(matrix, matrix)
    model = _DeterministicRiskModel()

    risk = _compute_retrieval_risk_metrics(
        model=model,
        risk_similarity_matrix=matrix,
        metric_t2v_matrix=matrix,
        metric_v2t_matrix=matrix,
        positive_mask=np.eye(4, dtype=bool),
        directions=("t2v", "v2t"),
        device=torch.device("cpu"),
        bootstrap_samples=20,
    )
    after = _compute_directional_metrics(matrix, matrix)

    assert after == before
    assert set(risk) == {"t2v", "v2t"}
    assert model.seen_shapes == [
        ("t2v", (4, 4)),
        ("v2t", (4, 4)),
    ]


def test_risk_reporting_uses_transposed_v2t_positive_mask():
    matrix = _matrix_with_successes_and_failures()
    model = _DeterministicRiskModel()

    result = _compute_retrieval_risk_metrics(
        model=model,
        risk_similarity_matrix=matrix,
        metric_t2v_matrix=matrix,
        metric_v2t_matrix=matrix,
        positive_mask=np.eye(4, dtype=bool),
        directions=("v2t",),
        device=torch.device("cpu"),
        bootstrap_samples=20,
    )

    assert result["v2t"]["num_queries"] == 4


def test_dual_softmax_uses_registered_temperature_without_mutating_input():
    matrix = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    before = matrix.copy()

    t2v, v2t_directional = _apply_dual_softmax(matrix, temperature=2.0)

    tensor = torch.from_numpy(before)
    expected_t2v = tensor * torch.softmax(tensor / 2.0, dim=0) * len(tensor)
    transposed = tensor.T
    expected_v2t = (
        transposed
        * torch.softmax(transposed / 2.0, dim=0)
        * len(transposed)
    ).T
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_allclose(t2v, expected_t2v.numpy())
    np.testing.assert_allclose(v2t_directional, expected_v2t.numpy())
