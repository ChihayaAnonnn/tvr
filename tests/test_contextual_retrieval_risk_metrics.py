from contextlib import nullcontext
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import main_task_retrieval as retrieval
from modules.modeling_mulit import UATVR as FullUATVR
from prob_models.contextual_retrieval_risk import (
    ContextualRetrievalRiskSidecar,
)
from prob_models.csd_logvar_protocol import array_sha256
from prob_models.retrieval_risk_metrics import evaluate_retrieval_risk


def _scores_with_two_failures():
    return np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 5.0, 3.0, 0.0],
            [5.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )


def test_risk_metrics_add_prefix_mean_aurc_and_strict_saturation():
    result = evaluate_retrieval_risk(
        _scores_with_two_failures(),
        np.eye(4, dtype=bool),
        np.array([0.0, 1e-4, 0.8, 1.0]),
        np.full(4, 0.3),
        bootstrap_samples=0,
    )

    assert result["aurc"] == pytest.approx(
        np.mean([0.0, 0.0, 1.0 / 3.0, 0.5])
    )
    assert result["saturation_fraction"] == 0.5
    assert {
        "spearman_risk_vs_rank",
        "failure_auroc",
        "aurc",
        "risk_coverage",
        "risk_stats",
        "saturation_fraction",
    } <= result.keys()
    assert set(result["failure_auroc"]) == {"value", "ci_low", "ci_high"}
    assert set(result["risk_stats"]) == {"mean", "std", "min", "max"}


def test_full_candidate_context_differs_from_split_minibatch_context():
    torch.manual_seed(5)
    sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=3,
        projection_dim=4,
        hidden_dim=6,
        topk=2,
        dropout=0.0,
    ).eval()
    logits = torch.tensor(
        [
            [5.0, 4.0, 0.0, -1.0],
            [3.0, 2.0, 8.0, 1.0],
            [-2.0, 7.0, 6.0, 0.0],
            [1.0, -1.0, 2.0, 9.0],
        ]
    )
    text = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    video = torch.tensor(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0]]
    )

    full = sidecar(logits, text, video, "t2v")["failure_risk"]
    split = torch.cat(
        [
            sidecar(logits[:2, :2], text[:2], video[:2], "t2v")[
                "failure_risk"
            ],
            sidecar(logits[2:, 2:], text[2:], video[2:], "t2v")[
                "failure_risk"
            ],
        ]
    )

    assert not torch.allclose(full, split)


def test_model_scoring_api_aligns_rectangular_v2t_modality_axes():
    model = object.__new__(FullUATVR)
    nn.Module.__init__(model)
    model.use_contextual_risk_sidecar = True
    model.contextual_risk_sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=3,
        projection_dim=2,
        hidden_dim=4,
        topk=2,
        dropout=0.0,
    )
    logits = torch.randn(4, 3)
    text_means = torch.randn(4, 3)
    video_means = torch.randn(3, 3)

    output = model.score_contextual_retrieval_risk(
        logits, text_means, video_means, "v2t"
    )

    assert output["failure_risk"].shape == (3,)
    torch.testing.assert_close(output["query_mean"], video_means)


class _RecordingContextualModel(nn.Module):
    use_retrieval_risk_sidecar = True
    use_contextual_risk_sidecar = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_sequence_output(self, input_ids, segment_ids, input_mask):
        del segment_ids, input_mask
        batch = input_ids.size(0)
        return torch.ones(batch, 1, 2), torch.ones(batch, 2, 2)

    def get_visual_output(self, video, video_mask):
        del video_mask
        batch = video.size(0)
        return torch.ones(batch, 1, 2), torch.ones(batch, 1, 1, 2)

    def score_contextual_retrieval_risk(
        self, logits, text_means, video_means, direction
    ):
        self.calls.append(
            (
                direction,
                logits.detach().cpu().clone(),
                text_means.detach().cpu().clone(),
                video_means.detach().cpu().clone(),
            )
        )
        if direction == "t2v":
            risk = logits.new_tensor([0.1, 0.8, 0.2, 0.9])
        else:
            risk = logits.new_tensor([0.9, 0.2, 0.3])
        return {"failure_risk": risk}

    def score_retrieval_risk(self, logits, direction):
        size = logits.shape[0] if direction == "t2v" else logits.shape[1]
        return {
            "failure_risk": torch.linspace(0.1, 0.9, size, device=logits.device),
            "vacuity": torch.linspace(0.2, 1.0, size, device=logits.device),
        }


class _MultiSentenceDataset:
    multi_sentence_per_video = True
    cut_off_points = [1, 2, 4]
    sentence_num = 4
    video_num = 3


class _OneBatchLoader(list):
    dataset = _MultiSentenceDataset()


def test_eval_epoch_scores_a3_once_after_full_raw_matrix_and_means(monkeypatch):
    raw_wti = np.array(
        [[5.0, 0.0, 0.0], [5.0, 4.0, 0.0], [0.0, 0.0, 5.0], [6.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    text_means = torch.tensor(
        [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]]
    )
    video_means = torch.tensor([[20.0, 0.0], [21.0, 0.0], [22.0, 0.0]])

    def fake_worker(*_args, **_kwargs):
        return {
            "similarity_rows": [raw_wti[:2], raw_wti[2:]],
            "text_mu": text_means,
            "video_mu": video_means,
        }

    monkeypatch.setattr(retrieval, "_run_on_single_gpu", fake_worker)
    monkeypatch.setattr(
        retrieval, "logger", logging.getLogger("test.contextual.eval"), raising=False
    )
    monkeypatch.setattr(retrieval, "_eval_autocast_context", lambda _args: nullcontext())
    monkeypatch.setattr(retrieval.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        retrieval,
        "_apply_dual_softmax",
        lambda matrix, _temperature: (matrix, matrix),
    )
    batch = (
        torch.ones(4, 2, dtype=torch.long),
        torch.ones(4, 2, dtype=torch.long),
        torch.zeros(4, 2, dtype=torch.long),
        torch.ones(4, 1, 1, 1, 1, 1),
        torch.ones(4, 1, dtype=torch.long),
    )
    model = _RecordingContextualModel()
    args = SimpleNamespace(
        local_rank=0,
        eval_vid_chunk_size=2,
        use_csd_logvar_sidecar=False,
        use_contextual_risk_sidecar=True,
        DSL=True,
        dsl_temperature=1.0,
        log_mus_scores=False,
        contextual_risk_bootstrap_samples=0,
    )

    result = retrieval.eval_epoch(
        args,
        model,
        _OneBatchLoader([batch]),
        torch.device("cpu"),
        n_gpu=1,
    )

    assert [call[0] for call in model.calls] == ["t2v", "v2t"]
    for _, seen_logits, seen_text, seen_video in model.calls:
        np.testing.assert_array_equal(seen_logits.numpy(), raw_wti)
        torch.testing.assert_close(seen_text, text_means)
        torch.testing.assert_close(seen_video, video_means)
    evidence = result["contextual_risk_evidence"]
    assert evidence["raw_wti"] == raw_wti.tolist()
    assert evidence["text_means"] == text_means.tolist()
    assert evidence["video_means"] == video_means.tolist()
    assert evidence["sha256"]["raw_wti"] == array_sha256(raw_wti)
    assert evidence["sha256"]["text_means"] == array_sha256(text_means.numpy())
    assert evidence["sha256"]["video_means"] == array_sha256(video_means.numpy())
    assert evidence["positive_mask"]
    assert evidence["array_metadata"]["raw_wti"] == {
        "dtype": "float32",
        "shape": list(raw_wti.shape),
        "sha256": array_sha256(raw_wti),
    }
    assert evidence["array_metadata"]["positive_mask"]["dtype"] == "bool"
    assert evidence["metric_config"] == {
        "bootstrap_samples": 0,
        "seed_by_direction": {"t2v": 0, "v2t": 1},
    }
    assert evidence["retrieval_config"] == {
        "directions": ["t2v", "v2t"],
        "cut_off_points": [1, 2, 4],
        "independent_directions": False,
        "dsl": True,
        "dsl_temperature": 1.0,
        "score_path": "post_dsl_directional_scores",
    }
    for direction in ("t2v", "v2t"):
        assert evidence["array_metadata"]["retrieval_scores"][direction][
            "shape"
        ] == list(raw_wti.shape)
    assert set(result["contextual_risk"]) == {"t2v", "v2t"}
    assert result["contextual_risk"]["t2v"]["num_failures"] == 2
    assert result["contextual_risk"]["v2t"]["num_failures"] == 1


def test_full_model_contextual_aux_is_independent_of_csd_and_legacy_is_unchanged():
    class _ModelShell:
        training = False
        sim_header = "seqTransf"
        use_csd_logvar_sidecar = False
        use_contextual_risk_sidecar = True

        def __init__(self):
            self.calls = []

        def _loose_similarity(self, *args, **kwargs):
            del args
            self.calls.append(kwargs)
            if kwargs.get("return_contextual_aux"):
                return torch.tensor([[2.0]]), {
                    "text_mu": torch.tensor([[1.0]]),
                    "video_mu": torch.tensor([[3.0]]),
                }
            return torch.tensor([[2.0]])

    shell = _ModelShell()
    call_args = (None, None, None, None, torch.ones(1, 1), torch.ones(1, 1))

    legacy = FullUATVR.get_similarity_logits(shell, *call_args)
    contextual = FullUATVR.get_similarity_logits(
        shell, *call_args, return_contextual_aux=True
    )

    assert legacy == (torch.tensor([[2.0]]), {})
    assert contextual[1] == {
        "text_mu": torch.tensor([[1.0]]),
        "video_mu": torch.tensor([[3.0]]),
    }
    assert shell.calls[0]["return_csd_aux"] is False
    assert shell.calls[0]["return_contextual_aux"] is False
    assert shell.calls[1]["return_csd_aux"] is False
    assert shell.calls[1]["return_contextual_aux"] is True


class _ContextualCollectionModel(nn.Module):
    loose_type = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

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
        return_contextual_aux=False,
    ):
        del text_token, visual_hidden, input_mask, video_mask, loose_type
        assert return_csd_aux is False
        assert return_contextual_aux is True
        text_mu = sequence_output.squeeze(1)
        video_mu = visual_output.mean(1)
        return text_mu @ video_mu.T, {
            "text_mu": text_mu,
            "video_mu": video_mu,
        }


def test_a3_only_worker_collects_ordered_means_without_csd_logvars():
    output = retrieval._run_on_single_gpu(
        _ContextualCollectionModel(),
        SimpleNamespace(
            eval_vid_chunk_size=2,
            use_csd_logvar_sidecar=False,
            use_contextual_risk_sidecar=True,
        ),
        [
            (torch.ones(2, 1), torch.zeros(2, 1)),
            (torch.ones(2, 1), torch.zeros(2, 1)),
        ],
        [(torch.ones(2, 1),), (torch.ones(1, 1),)],
        [
            (torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]]), torch.ones(2, 1, 2)),
            (torch.tensor([[[3.0, 0.0]], [[4.0, 0.0]]]), torch.ones(2, 1, 2)),
        ],
        [
            (torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), torch.ones(2, 1, 1, 2)),
            (torch.tensor([[[1.0, 1.0]]]), torch.ones(1, 1, 1, 2)),
        ],
    )

    assert set(output) == {"similarity_rows", "text_mu", "video_mu"}
    assert output["text_mu"].shape == (4, 2)
    assert output["video_mu"].shape == (3, 2)


def _valid_diagnostic_terms():
    terms = {
        "contextual_risk_loss": torch.tensor(1.0),
        "contextual_risk_bce": torch.tensor(0.7),
        "contextual_risk_fit": torch.tensor(0.2),
        "contextual_risk_rank": torch.tensor(0.1),
    }
    for direction in ("t2v", "v2t"):
        for name, value in {
            "failure_rate": 0.5,
            "positive_weight": 1.0,
            "prediction_mean": 0.4,
            "prediction_std": 0.1,
            "hardness_mean": 0.3,
            "hardness_std": 0.2,
            "valid_ranking_fraction": 0.75,
            "saturation_fraction": 0.0,
            "entropy": 0.8,
        }.items():
            terms[f"contextual_risk_{direction}_{name}"] = torch.tensor(value)
    return terms


def test_model_exposes_all_directional_training_diagnostics():
    model = object.__new__(FullUATVR)
    nn.Module.__init__(model)
    model.contextual_risk_temperature = 1.0
    model.w_contextual_risk_fit = 0.25
    model.w_contextual_risk_rank = 0.25
    model.contextual_risk_sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=3,
        projection_dim=2,
        hidden_dim=4,
        topk=2,
        dropout=0.0,
    )
    logits = torch.tensor(
        [[0.0, 4.0, 1.0], [3.0, 0.0, 5.0], [2.0, 1.0, 4.0]]
    )
    terms = model._contextual_risk_terms(
        logits,
        torch.randn(3, 3),
        torch.randn(3, 3),
        torch.tensor([7, 7, 9]),
    )

    expected = set(_valid_diagnostic_terms())
    assert expected <= terms.keys()
    retrieval._validate_contextual_risk_diagnostics(terms)


def test_contextual_diagnostic_validation_aborts_on_nonfinite_or_missing_class():
    retrieval._validate_contextual_risk_diagnostics(_valid_diagnostic_terms())

    nonfinite = _valid_diagnostic_terms()
    nonfinite["contextual_risk_fit"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        retrieval._validate_contextual_risk_diagnostics(nonfinite)

    missing = _valid_diagnostic_terms()
    missing["contextual_risk_v2t_failure_rate"] = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="both success and failure"):
        retrieval._validate_contextual_risk_diagnostics(missing)


def test_contextual_zero_gradient_aborts_on_second_consecutive_window():
    assert retrieval._advance_contextual_zero_gradient_windows(0, 0.0) == 1
    with pytest.raises(RuntimeError, match="two consecutive windows"):
        retrieval._advance_contextual_zero_gradient_windows(1, 0.0)
    assert retrieval._advance_contextual_zero_gradient_windows(1, 0.2) == 0


def test_contextual_diagnostic_window_spans_epochs_and_emits_exactly_at_20_steps():
    model = nn.Module()
    emitted = []

    for step in range(12):
        terms = _valid_diagnostic_terms()
        terms["contextual_risk_loss"] = torch.tensor(float(step))
        retrieval._accumulate_contextual_risk_window(model, terms)
        emitted.append(
            retrieval._finish_contextual_risk_window_step(
                model, float(step + 1)
            )
        )

    state_after_epoch_a = retrieval._contextual_risk_window_state(model)
    assert state_after_epoch_a["optimizer_steps"] == 12
    assert state_after_epoch_a["term_count"] == 12
    assert all(payload is None for payload in emitted)

    for step in range(12, 20):
        terms = _valid_diagnostic_terms()
        terms["contextual_risk_loss"] = torch.tensor(float(step))
        retrieval._accumulate_contextual_risk_window(model, terms)
        emitted.append(
            retrieval._finish_contextual_risk_window_step(
                model, float(step + 1)
            )
        )

    assert all(payload is None for payload in emitted[:19])
    assert emitted[19]["contextual_risk_loss"] == pytest.approx(9.5)
    assert emitted[19]["contextual_risk_gradient_norm"] == pytest.approx(10.5)
    reset = retrieval._contextual_risk_window_state(model)
    assert reset == {
        "term_sums": {},
        "term_count": 0,
        "gradient_total": 0.0,
        "optimizer_steps": 0,
        "zero_gradient_windows": 0,
    }


def test_contextual_gradient_norm_uses_only_a3_parameters():
    model = nn.Module()
    model.contextual_risk_sidecar = nn.Linear(2, 1, bias=False)
    model.backbone = nn.Linear(2, 1, bias=False)
    model.contextual_risk_sidecar.weight.grad = torch.tensor([[3.0, 4.0]])
    model.backbone.weight.grad = torch.tensor([[100.0, 100.0]])

    assert retrieval._contextual_risk_gradient_norm(model) == 5.0


def test_a3_diagnostics_use_complete_20_step_windows_even_in_smoke_mode():
    args = SimpleNamespace(
        use_csd_logvar_sidecar=False,
        use_contextual_risk_sidecar=True,
    )
    assert retrieval._contextual_risk_log_interval(args) == 20
    assert not retrieval._should_log_training_step(1, 20, smoke_cap=1, args=args)
    assert retrieval._should_log_training_step(20, 20, smoke_cap=None, args=args)


def test_a3_only_eval_is_fp32_and_deterministic(monkeypatch):
    calls = []

    def fake_autocast(**kwargs):
        calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(retrieval.torch, "autocast", fake_autocast)
    monkeypatch.setattr(retrieval.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(retrieval.torch.backends.cudnn, "deterministic", False)

    with retrieval._eval_autocast_context(
        SimpleNamespace(
            csd_logvar_sidecar_only=False,
            contextual_risk_sidecar_only=True,
        )
    ):
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True

    assert calls[0]["enabled"] is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False


def test_contextual_window_log_contains_registered_diagnostics(monkeypatch, caplog):
    test_logger = logging.getLogger("test.contextual.window")
    monkeypatch.setattr(retrieval, "logger", test_logger, raising=False)
    caplog.set_level(logging.INFO, logger=test_logger.name)
    means = {
        key: float(value)
        for key, value in _valid_diagnostic_terms().items()
    }
    means["contextual_risk_gradient_norm"] = 0.25

    retrieval._log_contextual_risk_window(means)

    message = "\n".join(record.getMessage() for record in caplog.records)
    for token in (
        "total=",
        "BCE=",
        "fit=",
        "rank=",
        "failure=",
        "weight=",
        "risk=",
        "hardness=",
        "valid-pair=",
        "grad=",
        "saturation=",
        "top-K entropy=",
        "T2V",
        "V2T",
    ):
        assert token in message
