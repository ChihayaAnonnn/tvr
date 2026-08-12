from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main_task_retrieval
from main_task_retrieval import (
    _apply_dual_softmax,
    _build_paired_digest,
    _compute_csd_logvar_metrics,
    _compute_directional_metrics,
    _compute_retrieval_risk_metrics,
    _eval_autocast_context,
    _merge_csd_eval_outputs,
    _run_on_single_gpu,
    _select_eval_device_ids,
)
from modules.modeling_mulit import UATVR as FullUATVR
from prob_models.csd_logvar_protocol import array_sha256


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


@pytest.mark.parametrize(
    ("a2_only", "enabled"),
    [(True, False), (False, True)],
)
def test_eval_autocast_is_disabled_only_for_a2_only_mode(
    monkeypatch, a2_only, enabled
):
    calls = []

    def fake_autocast(**kwargs):
        calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(main_task_retrieval.torch, "autocast", fake_autocast)

    with _eval_autocast_context(
        SimpleNamespace(csd_logvar_sidecar_only=a2_only)
    ):
        pass

    assert calls == [
        {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": enabled,
        }
    ]


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


def test_risk_reporting_retains_exact_output_array_digests():
    matrix = _matrix_with_successes_and_failures()
    model = _DeterministicRiskModel()

    risk, digests = _compute_retrieval_risk_metrics(
        model=model,
        risk_similarity_matrix=matrix,
        metric_t2v_matrix=matrix,
        metric_v2t_matrix=matrix,
        positive_mask=np.eye(4, dtype=bool),
        directions=("t2v", "v2t"),
        device=torch.device("cpu"),
        bootstrap_samples=20,
        return_array_digests=True,
    )

    expected_risk = np.array(
        [0.1, 0.36666664, 0.6333333, 0.9], dtype=np.float32
    )
    expected_vacuity = np.full(4, 0.4, dtype=np.float32)
    assert set(risk) == {"t2v", "v2t"}
    assert digests == {
        "t2v": {
            "failure_risk": array_sha256(expected_risk),
            "vacuity": array_sha256(expected_vacuity),
        },
        "v2t": {
            "failure_risk": array_sha256(expected_risk),
            "vacuity": array_sha256(expected_vacuity),
        },
    }


def test_paired_digest_captures_raw_wti_a1_outputs_and_both_retrievals():
    matrix = _matrix_with_successes_and_failures()
    result = {
        "t2v": {"R1": 50.0},
        "v2t": {"R1": 25.0},
        "retrieval_risk": {"t2v": {}, "v2t": {}},
    }
    risk_digests = {
        "t2v": {"failure_risk": "tf", "vacuity": "tv"},
        "v2t": {"failure_risk": "vf", "vacuity": "vv"},
    }

    paired = _build_paired_digest(
        result,
        matrix,
        risk_digests,
        ("t2v", "v2t"),
    )

    assert paired == {
        "raw_wti_sha256": array_sha256(matrix),
        "a1_failure_risk_sha256": {"t2v": "tf", "v2t": "vf"},
        "a1_vacuity_sha256": {"t2v": "tv", "v2t": "vv"},
        "retrieval": {
            "t2v": {"R1": 50.0},
            "v2t": {"R1": 25.0},
        },
    }


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


def test_csd_worker_merge_keeps_one_text_per_caption_and_one_unique_video():
    outputs = [
        {
            "similarity_rows": [np.array([[1.0, 2.0], [3.0, 4.0]])],
            "text_mu": torch.tensor([[10.0], [11.0]]),
            "text_logvar": torch.tensor([[-10.0], [-11.0]]),
            "video_mu": torch.tensor([[20.0], [21.0]]),
            "video_logvar": torch.tensor([[-8.0], [-9.0]]),
        },
        {
            "similarity_rows": [np.array([[5.0, 6.0], [7.0, 8.0]])],
            "text_mu": torch.tensor([[12.0], [13.0]]),
            "text_logvar": torch.tensor([[-12.0], [-7.0]]),
            # Every parallel worker sees all videos; only worker zero is used.
            "video_mu": torch.tensor([[120.0], [121.0]]),
            "video_logvar": torch.tensor([[-6.0], [-5.0]]),
        },
    ]

    matrix, intrinsic = _merge_csd_eval_outputs(outputs)

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        ),
    )
    assert intrinsic["text_mu"].squeeze(-1).tolist() == [10.0, 11.0, 12.0, 13.0]
    assert intrinsic["video_mu"].squeeze(-1).tolist() == [20.0, 21.0]


def test_csd_worker_merge_rejects_misaligned_retrieval_axes():
    outputs = [
        {
            "similarity_rows": [np.ones((2, 3), dtype=np.float32)],
            "text_mu": torch.ones(1, 2),
            "text_logvar": torch.ones(1, 2),
            "video_mu": torch.ones(3, 2),
            "video_logvar": torch.ones(3, 2),
        }
    ]

    with np.testing.assert_raises_regex(
        ValueError, "intrinsic uncertainty axes do not align"
    ):
        _merge_csd_eval_outputs(outputs)


class _CSDCollectionModel(torch.nn.Module):
    loose_type = True

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_similarity_logits(
        self,
        sequence_output,
        text_token,
        visual_output,
        visual_hidden,
        input_mask,
        video_mask,
        loose_type,
        return_csd_aux=False,
    ):
        del text_token, visual_hidden, input_mask, video_mask, loose_type
        text_mu = sequence_output.squeeze(1)
        video_mu = visual_output.mean(1)
        logits = text_mu @ video_mu.T
        if not return_csd_aux:
            return logits, {}
        return logits, {
            "text_mu": text_mu,
            "text_logvar": -text_mu,
            "video_mu": video_mu,
            "video_logvar": -video_mu,
        }


def test_single_gpu_csd_collection_deduplicates_chunked_modalities():
    model = _CSDCollectionModel()
    args = SimpleNamespace(
        eval_vid_chunk_size=2,
        use_csd_logvar_sidecar=True,
    )
    batch_list_t = [
        (torch.ones(2, 1), torch.zeros(2, 1)),
        (torch.ones(2, 1), torch.zeros(2, 1)),
    ]
    batch_list_v = [
        (torch.ones(2, 1),),
        (torch.ones(1, 1),),
    ]
    text_outputs = [
        (torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]]), torch.ones(2, 1, 2)),
        (torch.tensor([[[3.0, 0.0]], [[4.0, 0.0]]]), torch.ones(2, 1, 2)),
    ]
    visual_outputs = [
        (torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), torch.ones(2, 1, 1, 2)),
        (torch.tensor([[[1.0, 1.0]]]), torch.ones(1, 1, 1, 2)),
    ]

    output = _run_on_single_gpu(
        model,
        args,
        batch_list_t,
        batch_list_v,
        text_outputs,
        visual_outputs,
    )

    assert len(output["similarity_rows"]) == 2
    assert output["text_mu"].shape == (4, 2)
    assert output["text_logvar"].shape == (4, 2)
    assert output["video_mu"].shape == (3, 2)
    assert output["video_logvar"].shape == (3, 2)


def test_full_model_similarity_aux_is_opt_in_without_changing_default_return():
    class _ModelShell:
        training = False
        sim_header = "seqTransf"
        use_csd_logvar_sidecar = True

        def _loose_similarity(self, *args, return_csd_aux=False, **kwargs):
            logits = torch.tensor([[2.0]])
            if return_csd_aux:
                return logits, {"text_mu": torch.tensor([[1.0]])}
            return logits

    shell = _ModelShell()
    call_args = (None, None, None, None, torch.ones(1, 1), torch.ones(1, 1))

    default = FullUATVR.get_similarity_logits(shell, *call_args)
    auxiliary = FullUATVR.get_similarity_logits(
        shell, *call_args, return_csd_aux=True
    )

    assert default[0].item() == 2.0
    assert default[1] == {}
    assert auxiliary[0].item() == 2.0
    assert auxiliary[1]["text_mu"].item() == 1.0


def test_csd_reporting_uses_caption_uncertainty_for_t2v_and_video_for_v2t():
    scores = np.array(
        [[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [5.0, 3.0]],
        dtype=np.float32,
    )
    positive_mask = np.array(
        [[True, False], [True, False], [False, True], [False, True]]
    )
    intrinsic = {
        "text_mu": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "video_mu": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "text_logvar": torch.log(
            torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.9, 0.9]])
        ),
        "video_logvar": torch.log(torch.tensor([[0.1, 0.1], [0.9, 0.9]])),
    }

    result = _compute_csd_logvar_metrics(
        metric_t2v_matrix=scores,
        metric_v2t_matrix=scores,
        positive_mask=positive_mask,
        intrinsic=intrinsic,
        bootstrap_samples=20,
        logvar_min=-12.0,
        logvar_max=4.0,
    )

    assert result["t2v"]["num_queries"] == 4
    assert result["v2t"]["num_queries"] == 2
    assert set(result) == {
        "t2v",
        "v2t",
        "text_distribution",
        "video_distribution",
        "distance_components",
    }


def test_parallel_eval_uses_only_workers_with_text_batches():
    assert _select_eval_device_ids(n_gpu=4, num_text_batches=2) == [0, 1]
    assert _select_eval_device_ids(n_gpu=4, num_text_batches=5) == [0, 1, 2]
