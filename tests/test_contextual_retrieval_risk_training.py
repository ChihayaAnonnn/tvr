import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import numpy as np

import main_task_retrieval
from experiment_artifacts import ArtifactArrayStore
from main_task_retrieval import (
    _a3_exact_fp32_backend,
    _a3_training_autocast_enabled,
    _compute_directional_metrics,
    _contextual_risk_continuation_eligible,
    _contextual_risk_lifecycle_owner,
    _contextual_risk_should_continue,
    _initialize_contextual_risk_protocol,
    _maybe_run_contextual_risk_final_test,
    _publish_experiment_manifest,
    _record_contextual_risk_selection,
    _restore_training_checkpoint,
    _serialize_retrieval_metrics,
    save_model,
)
from prob_models.retrieval_risk_metrics import (
    evaluate_retrieval_risk,
    positive_ranks,
)
from prob_models.contextual_retrieval_risk import ContextualRetrievalRiskSidecar
from prob_models.retrieval_risk import RetrievalRiskSidecar
from prob_models.uncertainty_protocol import array_sha256, file_sha256


def _array_metadata(value):
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": array_sha256(array),
    }


def _complete_evidence(rho=0.30, raw="raw"):
    size = 8
    raw_wti = np.full((size, size), -2.0, dtype=np.float32)
    for index in range(size):
        raw_wti[index, index] = 0.0
        non_positive = [value for value in range(size) if value != index]
        raw_wti[index, non_positive[:index]] = 1.0
    if raw != "raw":
        raw_wti[0, 1] += 0.125
    positive_mask = np.eye(size, dtype=bool)
    text_means = np.arange(size * 2, dtype=np.float32).reshape(size, 2)
    video_means = text_means[::-1].copy()
    noise = np.asarray([3, 0, 6, 1, 7, 2, 5, 4], dtype=np.float32)
    quality = 0.2 if rho <= 0.20 else 0.5 if rho <= 0.21 else 0.8 if rho <= 0.30 else 1.0
    failure_risk = {}
    metrics = {}
    a1_failure_risk = {}
    a1_vacuity = {}
    for offset, direction in enumerate(("t2v", "v2t")):
        scores = raw_wti if direction == "t2v" else raw_wti.T
        mask = positive_mask if direction == "t2v" else positive_mask.T
        ranks = positive_ranks(scores, mask).astype(np.float32)
        risk = (quality * ranks + (1.0 - quality) * noise).astype(np.float32)
        failure_risk[direction] = risk
        metrics[direction] = evaluate_retrieval_risk(
            scores, mask, risk, bootstrap_samples=0, seed=offset
        )
        a1_failure_risk[direction] = (ranks / size).astype(np.float32)
        a1_vacuity[direction] = np.linspace(
            0.1, 0.8, size, dtype=np.float32
        )
    arrays = {
        "raw_wti": raw_wti,
        "text_means": text_means,
        "video_means": video_means,
        "positive_mask": positive_mask,
    }
    evidence = {
        **{name: value.tolist() for name, value in arrays.items()},
        "failure_risk": {
            direction: value.tolist() for direction, value in failure_risk.items()
        },
        "a1_failure_risk": {
            direction: value.tolist() for direction, value in a1_failure_risk.items()
        },
        "a1_vacuity": {
            direction: value.tolist() for direction, value in a1_vacuity.items()
        },
        "sha256": {
            "raw_wti": array_sha256(raw_wti),
            "text_means": array_sha256(text_means),
            "video_means": array_sha256(video_means),
            "positive_mask": array_sha256(positive_mask),
            "failure_risk": {
                direction: array_sha256(value)
                for direction, value in failure_risk.items()
            },
        },
        "array_metadata": {
            **{name: _array_metadata(value) for name, value in arrays.items()},
            "failure_risk": {
                direction: _array_metadata(value)
                for direction, value in failure_risk.items()
            },
            "a1_failure_risk": {
                direction: _array_metadata(value)
                for direction, value in a1_failure_risk.items()
            },
            "a1_vacuity": {
                direction: _array_metadata(value)
                for direction, value in a1_vacuity.items()
            },
        },
        "metric_config": {
            "bootstrap_samples": 0,
            "seed_by_direction": {"t2v": 0, "v2t": 1},
        },
        "retrieval_scores": {
            "t2v": raw_wti.tolist(),
            "v2t": raw_wti.tolist(),
        },
        "retrieval_config": {
            "directions": ["t2v", "v2t"],
            "cut_off_points": "one_to_one",
            "independent_directions": True,
            "dsl": False,
            "dsl_temperature": 1.0,
            "score_path": "post_dsl_directional_scores",
        },
        "a1_head_config": {
            "feature_dim": 2,
            "hidden_dim": 32,
            "temperature": 1.0,
        },
    }
    evidence["array_metadata"]["retrieval_scores"] = {
        direction: _array_metadata(raw_wti)
        for direction in ("t2v", "v2t")
    }
    return evidence, metrics


def test_externalized_a3_evidence_recomputes_from_hash_verified_references(tmp_path):
    evidence, expected = _complete_evidence()
    store = ArtifactArrayStore(tmp_path / "arrays")
    externalized = copy.deepcopy(evidence)
    externalized["artifact_array_root"] = str(tmp_path)
    for name in ("raw_wti", "text_means", "video_means", "positive_mask"):
        reference = store.write(
            name,
            np.asarray(evidence[name], dtype=evidence["array_metadata"][name]["dtype"]),
        ).as_dict()
        externalized[name] = reference
        externalized["array_metadata"][name] = reference
    for name in ("failure_risk", "a1_failure_risk", "a1_vacuity", "retrieval_scores"):
        for direction in ("t2v", "v2t"):
            reference = store.write(
                f"{name}.{direction}",
                np.asarray(
                    evidence[name][direction],
                    dtype=evidence["array_metadata"][name][direction]["dtype"],
                ),
            ).as_dict()
            externalized[name][direction] = reference
            externalized["array_metadata"][name][direction] = reference

    actual, _ = main_task_retrieval._recompute_contextual_risk_metrics(
        externalized,
        bootstrap_samples=0,
        expected_retrieval_config=externalized["retrieval_config"],
    )

    assert actual == expected


def _paired_digest(raw="raw"):
    evidence, _ = _complete_evidence(raw=raw)
    directional = _compute_directional_metrics(
        np.asarray(evidence["retrieval_scores"]["t2v"], dtype=np.float32),
        np.asarray(evidence["retrieval_scores"]["v2t"], dtype=np.float32),
    )
    return {
        "raw_wti_sha256": evidence["array_metadata"]["raw_wti"]["sha256"],
        "text_means_sha256": evidence["array_metadata"]["text_means"]["sha256"],
        "video_means_sha256": evidence["array_metadata"]["video_means"]["sha256"],
        "a1_failure_risk_sha256": {
            direction: evidence["array_metadata"]["a1_failure_risk"][direction]["sha256"]
            for direction in ("t2v", "v2t")
        },
        "a1_vacuity_sha256": {
            direction: evidence["array_metadata"]["a1_vacuity"][direction]["sha256"]
            for direction in ("t2v", "v2t")
        },
        "retrieval": {
            direction: _serialize_retrieval_metrics(directional[direction])
            for direction in ("t2v", "v2t")
        },
    }


def _direction(rho=0.30, auroc=0.70, aurc=0.25):
    return {
        "spearman_risk_vs_rank": rho,
        "failure_auroc": {"value": auroc, "ci_low": 0.51, "ci_high": 0.80},
        "risk_coverage": {
            "accuracy_at_0.25": 0.90,
            "accuracy_at_0.50": 0.80,
            "accuracy_at_0.75": 0.70,
            "accuracy_at_1.00": 0.60,
        },
        "risk_stats": {"mean": 0.40, "std": 0.02},
        "aurc": aurc,
    }


def _validation(rho=0.30, auroc=0.70, aurc=0.25, *, raw="raw"):
    evidence, metrics = _complete_evidence(rho=rho, raw=raw)
    return {
        "t2v": _paired_digest(raw)["retrieval"]["t2v"],
        "v2t": _paired_digest(raw)["retrieval"]["v2t"],
        "paired_digest": _paired_digest(raw),
        "contextual_risk": metrics,
        "contextual_risk_evidence": evidence,
    }


def _real_checkpoint_evidence(tmp_path):
    evidence, _ = _complete_evidence()
    raw = torch.tensor(evidence["raw_wti"], dtype=torch.float32)
    text = torch.tensor(evidence["text_means"], dtype=torch.float32)
    video = torch.tensor(evidence["video_means"], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        contextual = ContextualRetrievalRiskSidecar(
            embed_dim=2,
            projection_dim=3,
            hidden_dim=4,
            topk=3,
            temperature=1.0,
            dropout=0.1,
        ).eval()
        retrieval = RetrievalRiskSidecar(
            hidden_dim=4, temperature=1.0
        ).eval()
    with torch.no_grad():
        for direction in ("t2v", "v2t"):
            contextual_risk = contextual(
                raw,
                text if direction == "t2v" else video,
                video if direction == "t2v" else text,
                direction,
            )["failure_risk"].numpy()
            a1 = retrieval(raw, direction)
            for name, value in (
                ("failure_risk", contextual_risk),
                ("a1_failure_risk", a1["failure_risk"].numpy()),
                ("a1_vacuity", a1["vacuity"].numpy()),
            ):
                evidence[name][direction] = value.tolist()
                evidence["array_metadata"][name][direction] = _array_metadata(value)
                if name in evidence.get("sha256", {}):
                    evidence["sha256"][name][direction] = array_sha256(value)
    evidence["contextual_head_config"] = {
        "embed_dim": 2,
        "projection_dim": 3,
        "hidden_dim": 4,
        "topk": 3,
        "temperature": 1.0,
        "dropout": 0.1,
    }
    evidence["a1_head_config"] = {
        "feature_dim": 8,
        "hidden_dim": 4,
        "temperature": 1.0,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    state = {
        **{
            f"contextual_risk_sidecar.{name}": value
            for name, value in contextual.state_dict().items()
        },
        **{
            f"retrieval_risk_sidecar.{name}": value
            for name, value in retrieval.state_dict().items()
        },
    }
    torch.save(state, checkpoint)
    return checkpoint, evidence


def test_checkpoint_state_anchors_contextual_and_a1_output_evidence(tmp_path):
    checkpoint, evidence = _real_checkpoint_evidence(tmp_path)

    main_task_retrieval._verify_contextual_risk_checkpoint_outputs(
        checkpoint,
        evidence,
        expected_head_configs={
            "contextual": evidence["contextual_head_config"],
            "a1": evidence["a1_head_config"],
        },
    )

    contextual_tamper = copy.deepcopy(evidence)
    changed = np.asarray(
        contextual_tamper["failure_risk"]["t2v"], dtype=np.float32
    ) + 0.125
    contextual_tamper["failure_risk"]["t2v"] = changed.tolist()
    contextual_tamper["array_metadata"]["failure_risk"]["t2v"] = (
        _array_metadata(changed)
    )
    contextual_tamper["sha256"]["failure_risk"]["t2v"] = array_sha256(changed)
    with pytest.raises(ValueError, match="checkpoint-derived contextual"):
        main_task_retrieval._verify_contextual_risk_checkpoint_outputs(
            checkpoint,
            contextual_tamper,
            expected_head_configs={
                "contextual": evidence["contextual_head_config"],
                "a1": evidence["a1_head_config"],
            },
        )

    a1_tamper = copy.deepcopy(evidence)
    changed = np.asarray(
        a1_tamper["a1_failure_risk"]["v2t"], dtype=np.float32
    ) + 0.125
    a1_tamper["a1_failure_risk"]["v2t"] = changed.tolist()
    a1_tamper["array_metadata"]["a1_failure_risk"]["v2t"] = (
        _array_metadata(changed)
    )
    with pytest.raises(ValueError, match="checkpoint-derived A1"):
        main_task_retrieval._verify_contextual_risk_checkpoint_outputs(
            checkpoint,
            a1_tamper,
            expected_head_configs={
                "contextual": evidence["contextual_head_config"],
                "a1": evidence["a1_head_config"],
            },
        )

    config_tamper = copy.deepcopy(evidence)
    config_tamper["contextual_head_config"]["temperature"] = 2.0
    with pytest.raises(ValueError, match="configuration differs from manifest"):
        main_task_retrieval._verify_contextual_risk_checkpoint_outputs(
            checkpoint,
            config_tamper,
            expected_head_configs={
                "contextual": evidence["contextual_head_config"],
                "a1": evidence["a1_head_config"],
            },
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_model_registers_canonical_cpu_checkpoint_outputs(tmp_path):
    checkpoint, expected = _real_checkpoint_evidence(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contextual = ContextualRetrievalRiskSidecar(
        **expected["contextual_head_config"]
    ).cuda().eval()
    contextual.load_state_dict({
        name.removeprefix("contextual_risk_sidecar."): value
        for name, value in state.items()
        if name.startswith("contextual_risk_sidecar.")
    })
    retrieval = RetrievalRiskSidecar(
        **expected["a1_head_config"]
    ).cuda().eval()
    retrieval.load_state_dict({
        name.removeprefix("retrieval_risk_sidecar."): value
        for name, value in state.items()
        if name.startswith("retrieval_risk_sidecar.")
    })
    shell = SimpleNamespace(
        contextual_risk_sidecar=contextual,
        retrieval_risk_sidecar=retrieval,
    )
    raw = np.asarray(expected["raw_wti"], dtype=np.float32)
    mask = np.asarray(expected["positive_mask"], dtype=bool)
    text = torch.tensor(expected["text_means"], device="cuda")
    video = torch.tensor(expected["video_means"], device="cuda")
    _, evidence = main_task_retrieval._compute_contextual_retrieval_risk_metrics(
        model=shell,
        risk_similarity_matrix=raw,
        positive_mask=mask,
        text_means=text,
        video_means=video,
        directions=("t2v", "v2t"),
        device=torch.device("cuda"),
        bootstrap_samples=0,
        return_evidence=True,
    )
    _, a1 = main_task_retrieval._compute_retrieval_risk_metrics(
        model=shell,
        risk_similarity_matrix=raw,
        metric_t2v_matrix=raw,
        metric_v2t_matrix=raw,
        positive_mask=mask,
        directions=("t2v", "v2t"),
        device=torch.device("cuda"),
        bootstrap_samples=0,
        return_array_digests=True,
        return_array_evidence=True,
        canonical_cpu_replay=True,
    )
    evidence.update({
        "contextual_head_config": expected["contextual_head_config"],
        "a1_head_config": expected["a1_head_config"],
        "a1_failure_risk": {
            direction: a1[direction]["failure_risk_values"]
            for direction in ("t2v", "v2t")
        },
        "a1_vacuity": {
            direction: a1[direction]["vacuity_values"]
            for direction in ("t2v", "v2t")
        },
    })
    for name, stem in (
        ("a1_failure_risk", "failure_risk"),
        ("a1_vacuity", "vacuity"),
    ):
        evidence["array_metadata"][name] = {
            direction: {
                "dtype": a1[direction][f"{stem}_dtype"],
                "shape": a1[direction][f"{stem}_shape"],
                "sha256": a1[direction][stem],
            }
            for direction in ("t2v", "v2t")
        }

    main_task_retrieval._verify_contextual_risk_checkpoint_outputs(
        checkpoint,
        evidence,
        expected_head_configs={
            "contextual": expected["contextual_head_config"],
            "a1": expected["a1_head_config"],
        },
    )


def test_retrieval_evidence_config_cannot_override_registered_args():
    evidence, _ = _complete_evidence()
    with pytest.raises(ValueError, match="retrieval configuration mismatch"):
        main_task_retrieval._recompute_contextual_risk_metrics(
            evidence,
            bootstrap_samples=0,
            expected_retrieval_config={
                **evidence["retrieval_config"],
                "dsl": True,
            },
        )


def test_regenerated_canonical_baseline_comparison_is_exact():
    registration = {
        "paired_digest": _paired_digest(),
        "evidence": _complete_evidence()[0],
    }
    main_task_retrieval._compare_regenerated_contextual_baseline(
        registration, copy.deepcopy(registration)
    )

    changed = copy.deepcopy(registration)
    changed["evidence"]["a1_vacuity"]["t2v"][0] += 0.01
    with pytest.raises(ValueError, match="regenerated canonical baseline"):
        main_task_retrieval._compare_regenerated_contextual_baseline(
            registration, changed
        )


def test_registered_training_and_baseline_reload_share_exact_fp32_model_boundary(
    tmp_path, monkeypatch
):
    class NestedHalfModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.ModuleDict({
                "projection": torch.nn.Linear(2, 2, bias=False).half()
            })
            self.encoder.register_buffer(
                "running_scale", torch.ones(2, dtype=torch.bfloat16)
            )

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    checkpoint = tmp_path / "canonical-a1.pt"
    torch.save({}, checkpoint)
    args = SimpleNamespace(
        init_model=str(checkpoint),
        cache_dir=None,
        use_uatvr_full=False,
        cross_model="cross-base",
        contextual_risk_sidecar_only=True,
        local_rank=1,
    )
    monkeypatch.setattr(
        main_task_retrieval, "_import_uatvr", lambda _full: NestedHalfModel
    )
    training_model = main_task_retrieval.init_model(
        args, torch.device("cpu"), 1, 1
    )

    observed = []

    def fake_eval(_args, model, *_rest, **_kwargs):
        parameter_dtypes = {value.dtype for value in model.parameters()}
        buffer_dtypes = {value.dtype for value in model.buffers()}
        observed.append((parameter_dtypes, buffer_dtypes))
        output = model.encoder["projection"](
            torch.ones(1, 2, dtype=torch.float32)
        )
        output = output * model.encoder.running_scale
        assert output.dtype == torch.float32
        return {"paired_digest": {}, "contextual_risk_evidence": {}}

    monkeypatch.setattr(main_task_retrieval, "eval_epoch", fake_eval)
    main_task_retrieval._regenerate_contextual_risk_baseline(
        args, object(), torch.device("cpu"), 1, str(checkpoint)
    )

    assert {value.dtype for value in training_model.parameters()} == {
        torch.float32
    }
    assert {value.dtype for value in training_model.buffers()} == {
        torch.float32
    }
    assert observed == [({torch.float32}, {torch.float32})]


def test_typed_evidence_rejects_dtype_metadata_conversion_attack():
    evidence, _ = _complete_evidence()
    attacked = copy.deepcopy(evidence)
    values = np.asarray(attacked["text_means"], dtype=np.float64)
    attacked["array_metadata"]["text_means"] = _array_metadata(values)
    attacked["sha256"]["text_means"] = array_sha256(values)

    with pytest.raises(ValueError, match="text_means dtype"):
        main_task_retrieval._contextual_evidence_array(
            attacked, "text_means"
        )


def _write_pair(run, reference, epoch):
    model_path = run / f"pytorch_model.bin.{epoch}"
    optimizer_path = run / f"pytorch_opt.bin.{epoch}"
    torch.save(
        {
            "backbone": torch.tensor([1.0]),
            "contextual_risk_sidecar.weight": torch.tensor([float(epoch)]),
        },
        model_path,
    )
    payload = {
        "epoch": epoch,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": epoch + 1},
        "global_step": (epoch + 1) * 5,
    }
    torch.save(payload, optimizer_path)
    marker_path = run / f"checkpoint_epoch.{epoch}.published.json"
    main_task_retrieval.atomic_write_json(
        marker_path,
        {
            "epoch": epoch,
            "model_path": str(model_path.resolve()),
            "model_sha256": file_sha256(model_path),
            "optimizer_path": str(optimizer_path.resolve()),
            "optimizer_sha256": file_sha256(optimizer_path),
        },
    )
    return model_path, optimizer_path, payload


def _record(run, reference, candidates, epoch, metrics):
    model, optimizer, payload = _write_pair(run, reference, epoch)
    with mock.patch.object(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
    ):
        winner = _record_contextual_risk_selection(
            epoch=epoch,
            checkpoint_path=model,
            optimizer_checkpoint_path=optimizer,
            validation_metrics=metrics,
            paired_baseline=_paired_digest(),
            reference_checkpoint=reference,
            candidate_path=run / "contextual_risk_candidates.json",
            selection_path=run / "best_contextual_risk_checkpoint.json",
            bootstrap_samples=0,
            expected_retrieval_config=metrics[
                "contextual_risk_evidence"
            ]["retrieval_config"],
            expected_head_configs={},
        )
    candidates[:] = json.loads(
        (run / "contextual_risk_candidates.json").read_text()
    )
    return winner, model, optimizer, payload


def test_initial_three_candidates_are_append_only_and_drive_strict_continuation(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
        lambda *_args, **_kwargs: None,
    )
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    candidates = []

    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        _record(run, reference, candidates, epoch, _validation(rho=rho))

    assert [item["epoch"] for item in candidates] == [0, 1, 2]
    for candidate in candidates:
        assert candidate["model_sha256"] == file_sha256(candidate["path"])
        assert candidate["optimizer_sha256"] == file_sha256(
            candidate["optimizer_path"]
        )
        assert candidate["non_a3_mismatches"] == []
        assert candidate["paired_invariance"]["passed"] is True
        assert candidate["finite"] is True
        assert candidate["evidence"]
        assert "metrics" in candidate and "acceptance" in candidate
    assert _contextual_risk_should_continue(candidates) is True

    not_improved = copy.deepcopy(candidates)
    not_improved[2]["rho"] = list(not_improved[1]["rho"])
    not_improved[2]["auroc"] = list(not_improved[1]["auroc"])
    not_improved[2]["aurc"] = list(not_improved[1]["aurc"])
    assert _contextual_risk_should_continue(not_improved) is False

    with pytest.raises(ValueError, match="append-only"):
        _record_contextual_risk_selection(
            epoch=2,
            checkpoint_path=candidates[2]["path"],
            optimizer_checkpoint_path=candidates[2]["optimizer_path"],
            validation_metrics=_validation(),
            paired_baseline=_paired_digest(),
            reference_checkpoint=reference,
            candidate_path=run / "contextual_risk_candidates.json",
            selection_path=run / "best_contextual_risk_checkpoint.json",
            bootstrap_samples=0,
            expected_retrieval_config=_validation()[
                "contextual_risk_evidence"
            ]["retrieval_config"],
            expected_head_configs={},
        )


def test_continuation_requires_exact_epoch_two_pair_and_restores_human_epoch_four(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
        lambda *_args, **_kwargs: None,
    )
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    reference_sha = file_sha256(reference)
    monkeypatch.setattr(
        main_task_retrieval, "CONTEXTUAL_RISK_CANONICAL_A1_SHA256", reference_sha
    )
    candidates = []
    pairs = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        _, model, optimizer, payload = _record(
            run, reference, candidates, epoch, _validation(rho=rho)
        )
        pairs.append((model, optimizer, payload))
    baseline_registration = {
            "reference_checkpoint": str(reference.resolve()),
            "reference_checkpoint_sha256": reference_sha,
            "paired_digest": _paired_digest(),
            "evidence": _complete_evidence()[0],
        }
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json",
        baseline_registration,
    )
    regenerated = []

    def regenerate(*_args, **_kwargs):
        regenerated.append(True)
        return copy.deepcopy(baseline_registration)

    monkeypatch.setattr(
        main_task_retrieval,
        "_regenerate_contextual_risk_baseline",
        regenerate,
    )
    model, optimizer_path, optimizer_payload = pairs[2]
    protocol = _initialize_contextual_risk_protocol(
        SimpleNamespace(
            output_dir=str(run),
            init_model=str(model),
            resume_model=str(optimizer_path),
            epochs=5,
            contextual_risk_bootstrap_samples=0,
        ),
        object(),
        object(),
        torch.device("cpu"),
        1,
        resume_payload=optimizer_payload,
    )

    assert protocol["resume_candidate"]["epoch"] == 2
    assert regenerated == [True]
    opt = SimpleNamespace(load_state_dict=lambda state: None)
    scheduler = SimpleNamespace(load_state_dict=lambda state: None)
    resumed_epoch, global_step = _restore_training_checkpoint(
        optimizer_payload, opt, scheduler, require_complete=True
    )
    assert resumed_epoch + 1 == 4
    assert global_step == 15

    with pytest.raises(ValueError, match="exactly one recorded candidate"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run),
                init_model=str(tmp_path / "copy.pt"),
                resume_model=str(optimizer_path),
                epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    wrong_optimizer = tmp_path / "wrong-optimizer.pt"
    torch.save(optimizer_payload, wrong_optimizer)
    with pytest.raises(ValueError, match="optimizer checkpoint path mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run),
                init_model=str(model),
                resume_model=str(wrong_optimizer),
                epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    candidate_path = run / "contextual_risk_candidates.json"
    selection_path = run / "best_contextual_risk_checkpoint.json"
    original_candidates = json.loads(candidate_path.read_text())
    original_selection = json.loads(selection_path.read_text())
    coherent_baseline = copy.deepcopy(baseline_registration)
    coherent_candidates = copy.deepcopy(original_candidates)

    def coherently_rewrite_means(payload):
        changed = np.asarray(
            payload["evidence"]["text_means"], dtype=np.float32
        ) + 0.5
        changed_sha = array_sha256(changed)
        payload["evidence"]["text_means"] = changed.tolist()
        payload["evidence"]["array_metadata"]["text_means"] = (
            _array_metadata(changed)
        )
        payload["evidence"]["sha256"]["text_means"] = changed_sha
        payload["paired_digest"]["text_means_sha256"] = changed_sha

    coherently_rewrite_means(coherent_baseline)
    for candidate in coherent_candidates:
        coherently_rewrite_means(candidate)
    coherent_selection = copy.deepcopy(original_selection)
    coherent_selection["candidates"] = coherent_candidates
    coherent_selection["winner"] = coherent_candidates[2]
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json", coherent_baseline
    )
    main_task_retrieval.atomic_write_json(candidate_path, coherent_candidates)
    main_task_retrieval.atomic_write_json(selection_path, coherent_selection)
    with pytest.raises(ValueError, match="regenerated canonical baseline"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json", baseline_registration
    )
    main_task_retrieval.atomic_write_json(candidate_path, original_candidates)
    main_task_retrieval.atomic_write_json(selection_path, original_selection)
    monkeypatch.setattr(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpoint replay must follow canonical paired-input comparison"
        ),
    )
    mismatched_candidate = copy.deepcopy(original_candidates[2])
    mismatched_means = np.asarray(
        mismatched_candidate["evidence"]["text_means"], dtype=np.float32
    ) + 0.125
    mismatched_candidate["evidence"]["text_means"] = mismatched_means.tolist()
    mismatched_candidate["evidence"]["array_metadata"]["text_means"] = (
        _array_metadata(mismatched_means)
    )
    mismatched_candidate["evidence"]["sha256"]["text_means"] = array_sha256(
        mismatched_means
    )
    mismatched_candidate["paired_digest"]["text_means_sha256"] = array_sha256(
        mismatched_means
    )
    with pytest.raises(ValueError, match="immutable candidate audit mismatch"):
        main_task_retrieval._reaudit_contextual_risk_candidate(
            mismatched_candidate,
            paired_baseline=baseline_registration["paired_digest"],
            reference_checkpoint=reference,
            bootstrap_samples=0,
            expected_retrieval_config=mismatched_candidate["evidence"][
                "retrieval_config"
            ],
            expected_head_configs={},
        )
    monkeypatch.setattr(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
        lambda *_args, **_kwargs: None,
    )
    mean_tamper = copy.deepcopy(original_candidates)
    changed_means = np.asarray(
        mean_tamper[2]["evidence"]["text_means"], dtype=np.float32
    ) + 0.25
    mean_tamper[2]["evidence"]["text_means"] = changed_means.tolist()
    mean_tamper[2]["evidence"]["array_metadata"]["text_means"] = (
        _array_metadata(changed_means)
    )
    mean_tamper[2]["evidence"]["sha256"]["text_means"] = array_sha256(
        changed_means
    )
    mean_tamper[2]["paired_digest"]["text_means_sha256"] = array_sha256(
        changed_means
    )
    mean_selection = copy.deepcopy(original_selection)
    mean_selection["candidates"] = mean_tamper
    mean_selection["winner"] = mean_tamper[2]
    main_task_retrieval.atomic_write_json(candidate_path, mean_tamper)
    main_task_retrieval.atomic_write_json(selection_path, mean_selection)
    with pytest.raises(ValueError, match="immutable candidate audit mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    tampered = copy.deepcopy(original_candidates)
    tampered[2]["paired_digest"] = _paired_digest("forged")
    tampered[2]["paired_invariance"] = {
        "passed": True,
        "checks": original_candidates[2]["paired_invariance"]["checks"],
    }
    forged_selection = copy.deepcopy(original_selection)
    forged_selection["candidates"] = tampered
    forged_selection["winner"] = tampered[2]
    main_task_retrieval.atomic_write_json(candidate_path, tampered)
    main_task_retrieval.atomic_write_json(selection_path, forged_selection)
    with pytest.raises(ValueError, match="immutable candidate .* evidence mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    finite_metric_tamper = copy.deepcopy(original_candidates)
    finite_metric_tamper[2]["metrics"]["t2v"][
        "spearman_risk_vs_rank"
    ] += 0.01
    finite_metric_tamper[2]["rho"][0] += 0.01
    metric_selection = copy.deepcopy(original_selection)
    metric_selection["candidates"] = finite_metric_tamper
    metric_selection["winner"] = finite_metric_tamper[2]
    main_task_retrieval.atomic_write_json(candidate_path, finite_metric_tamper)
    main_task_retrieval.atomic_write_json(selection_path, metric_selection)
    with pytest.raises(ValueError, match="metric summary mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    digest_tamper = copy.deepcopy(original_candidates)
    digest_tamper[2]["evidence"]["array_metadata"]["text_means"][
        "sha256"
    ] = "0" * 64
    digest_selection = copy.deepcopy(original_selection)
    digest_selection["candidates"] = digest_tamper
    digest_selection["winner"] = digest_tamper[2]
    main_task_retrieval.atomic_write_json(candidate_path, digest_tamper)
    main_task_retrieval.atomic_write_json(selection_path, digest_selection)
    with pytest.raises(ValueError, match="text_means SHA-256 mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    coherent_retrieval_tamper = copy.deepcopy(original_candidates)
    forged_scores = -np.asarray(
        original_candidates[0]["evidence"]["raw_wti"], dtype=np.float32
    )
    forged_directional = _compute_directional_metrics(
        forged_scores, forged_scores
    )
    forged_retrieval = {
        direction: _serialize_retrieval_metrics(
            forged_directional[direction]
        )
        for direction in ("t2v", "v2t")
    }

    def forge_retrieval_evidence(evidence):
        for direction in ("t2v", "v2t"):
            evidence["retrieval_scores"][direction] = forged_scores.tolist()
            evidence["array_metadata"]["retrieval_scores"][
                direction
            ] = _array_metadata(forged_scores)

    for candidate in coherent_retrieval_tamper:
        candidate["paired_digest"]["retrieval"] = forged_retrieval
        forge_retrieval_evidence(candidate["evidence"])
    retrieval_selection = copy.deepcopy(original_selection)
    retrieval_selection["candidates"] = coherent_retrieval_tamper
    retrieval_selection["winner"] = coherent_retrieval_tamper[2]
    attacked_baseline_registration = json.loads(
        (run / "a3_paired_baseline.json").read_text()
    )
    original_baseline_registration = copy.deepcopy(attacked_baseline_registration)
    attacked_baseline_registration["paired_digest"]["retrieval"] = forged_retrieval
    forge_retrieval_evidence(attacked_baseline_registration["evidence"])
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json", attacked_baseline_registration
    )
    main_task_retrieval.atomic_write_json(
        candidate_path, coherent_retrieval_tamper
    )
    main_task_retrieval.atomic_write_json(selection_path, retrieval_selection)
    with pytest.raises(ValueError, match="regenerated canonical baseline"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json", original_baseline_registration
    )

    array_tamper = copy.deepcopy(original_candidates)
    array_tamper[2]["evidence"]["failure_risk"]["t2v"][0] += 0.125
    changed_risk = np.asarray(
        array_tamper[2]["evidence"]["failure_risk"]["t2v"],
        dtype=np.float32,
    )
    changed_sha = array_sha256(changed_risk)
    array_tamper[2]["evidence"]["array_metadata"]["failure_risk"][
        "t2v"
    ]["sha256"] = changed_sha
    array_tamper[2]["evidence"]["sha256"]["failure_risk"][
        "t2v"
    ] = changed_sha
    array_selection = copy.deepcopy(original_selection)
    array_selection["candidates"] = array_tamper
    array_selection["winner"] = array_tamper[2]
    main_task_retrieval.atomic_write_json(candidate_path, array_tamper)
    main_task_retrieval.atomic_write_json(selection_path, array_selection)
    with pytest.raises(ValueError, match="metric summary mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
                contextual_risk_bootstrap_samples=0,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )


def test_later_valid_candidates_can_win_but_leaked_candidate_cannot(tmp_path):
    """Weight-level leakage (``non_a3_mismatches``) is the only remaining
    fail-closed condition after the paired_invariance gate was removed."""

    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    candidates = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22, 0.40)):
        winner, *_ = _record(
            run, reference, candidates, epoch, _validation(rho=rho)
        )
    assert winner["epoch"] == 3

    # Simulate a training bug that mutates a frozen non-sidecar weight so the
    # candidate's ``backbone`` tensor differs from the A1 reference.
    leaky_model = run / "pytorch_model.bin.4"
    torch.save(
        {
            "backbone": torch.tensor([2.0]),
            "contextual_risk_sidecar.weight": torch.tensor([4.0]),
        },
        leaky_model,
    )
    leaky_optimizer = run / "pytorch_opt.bin.4"
    torch.save(
        {
            "epoch": 4,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "scheduler_state_dict": {"last_epoch": 5},
            "global_step": 25,
        },
        leaky_optimizer,
    )
    main_task_retrieval.atomic_write_json(
        run / "checkpoint_epoch.4.published.json",
        {
            "epoch": 4,
            "model_path": str(leaky_model.resolve()),
            "model_sha256": file_sha256(leaky_model),
            "optimizer_path": str(leaky_optimizer.resolve()),
            "optimizer_sha256": file_sha256(leaky_optimizer),
        },
    )

    with pytest.raises(RuntimeError, match="terminal A3 invariance violation"):
        with mock.patch.object(
            main_task_retrieval, "_verify_contextual_risk_checkpoint_outputs"
        ):
            _record_contextual_risk_selection(
                epoch=4,
                checkpoint_path=leaky_model,
                optimizer_checkpoint_path=leaky_optimizer,
                validation_metrics=_validation(rho=0.99),
                paired_baseline=_paired_digest(),
                reference_checkpoint=reference,
                candidate_path=run / "contextual_risk_candidates.json",
                selection_path=run / "best_contextual_risk_checkpoint.json",
                bootstrap_samples=0,
                expected_retrieval_config=_validation()[
                    "contextual_risk_evidence"
                ]["retrieval_config"],
                expected_head_configs={},
            )
    persisted = json.loads(
        (run / "contextual_risk_candidates.json").read_text()
    )
    assert persisted[4]["valid"] is False
    assert persisted[4]["non_a3_mismatches"]


def test_paired_invariance_failure_is_audit_only_not_terminal(tmp_path):
    """After the paired_invariance fail-fast is removed, a candidate whose
    baseline↔candidate digests drift (e.g. from CUDA/DataLoader nondeterminism)
    must be recorded with the failing checks preserved for audit, and MUST NOT
    abort training or block the winner selection.  Only weight-level leakage
    (``non_a3_mismatches``) may fail the run closed.
    """

    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)

    candidates = []
    # raw="changed" perturbs the WTI/embedding digests so paired_invariance
    # will report passed=False and every array check False.  Weights that
    # matter to the freeze contract remain identical to the reference.
    drifted = _validation(rho=0.99, raw="changed")

    winner, *_ = _record(run, reference, candidates, 0, drifted)

    persisted = json.loads((run / "contextual_risk_candidates.json").read_text())
    assert len(persisted) == 1
    candidate = persisted[0]
    checks = candidate["paired_invariance"]["checks"]
    assert candidate["paired_invariance"]["passed"] is False
    assert checks["raw_wti"] is False
    assert not all(checks.values())
    assert candidate["non_a3_mismatches"] == []
    assert candidate["valid"] is True
    assert winner is not None and winner["epoch"] == 0


def test_selection_receives_validation_candidate_only(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    model, optimizer, _ = _write_pair(run, reference, 0)
    observed = []

    def select(candidates, *, output_dir):
        observed.extend(copy.deepcopy(candidates))
        return candidates[0]

    monkeypatch.setattr(main_task_retrieval, "choose_contextual_risk_checkpoint", select)
    monkeypatch.setattr(
        main_task_retrieval,
        "_verify_contextual_risk_checkpoint_outputs",
        lambda *_args, **_kwargs: None,
    )
    metrics = _validation()
    metrics["test_metrics"] = {"t2v": {"R1": 99.0}}
    _record_contextual_risk_selection(
        epoch=0,
        checkpoint_path=model,
        optimizer_checkpoint_path=optimizer,
        validation_metrics=metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference,
        candidate_path=run / "contextual_risk_candidates.json",
        selection_path=run / "best_contextual_risk_checkpoint.json",
        bootstrap_samples=0,
        expected_retrieval_config=metrics[
            "contextual_risk_evidence"
        ]["retrieval_config"],
        expected_head_configs={},
    )

    assert len(observed) == 1
    assert "test_metrics" not in observed[0]
    assert set(observed[0]["metrics"]) == {
        "t2v", "v2t", "paired_invariance", "non_a3_mismatches"
    }


def test_final_test_gate_calls_jsfusion_once_only_on_validation_pass(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    selection = {"winner": winner, "candidates": [winner]}
    calls = []
    args = SimpleNamespace(run_final_test=True, output_dir=str(tmp_path))

    result = _maybe_run_contextual_risk_final_test(
        args, selection, lambda selected: calls.append(selected) or {"test": 1}
    )

    assert result == {"test": 1}
    assert calls == [winner]
    claim = json.loads((tmp_path / "a3_final_test_claim.json").read_text())
    assert claim["status"] == "completed"
    result_path = tmp_path / "final_test.json"
    assert json.loads(result_path.read_text()) == {"test": 1}
    assert claim["result_path"] == str(result_path.resolve())
    assert claim["result_sha256"] == file_sha256(result_path)

    with pytest.raises(RuntimeError, match="cannot be retried"):
        _maybe_run_contextual_risk_final_test(
            args,
            selection,
            lambda selected: calls.append(selected),
        )
    assert calls == [winner]

    failed = copy.deepcopy(selection)
    failed["winner"]["metrics"]["v2t"]["risk_stats"]["std"] = 0.0
    calls.clear()
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(warning=lambda *args: None),
        raising=False,
    )
    assert _maybe_run_contextual_risk_final_test(
        args, failed, lambda selected: calls.append(selected)
    ) is None
    assert calls == []

    calls.clear()
    assert _maybe_run_contextual_risk_final_test(
        args, {"winner": None, "candidates": []},
        lambda selected: calls.append(selected),
    ) is None
    assert calls == []


def test_failed_final_test_claim_cannot_be_retried(tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    selection = {"winner": winner, "candidates": [winner]}
    args = SimpleNamespace(run_final_test=True, output_dir=str(tmp_path))

    with pytest.raises(OSError, match="test failed"):
        _maybe_run_contextual_risk_final_test(
            args, selection, lambda selected: (_ for _ in ()).throw(
                OSError("test failed")
            )
        )
    claim = json.loads((tmp_path / "a3_final_test_claim.json").read_text())
    assert claim["status"] == "failed"
    assert claim["error_type"] == "OSError"

    with pytest.raises(RuntimeError, match="cannot be retried"):
        _maybe_run_contextual_risk_final_test(args, selection, lambda selected: None)


@pytest.mark.parametrize("status", ["started", "completed"])
def test_final_test_claim_crash_windows_are_diagnosable(tmp_path, status):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    main_task_retrieval.atomic_write_json(
        tmp_path / "a3_final_test_claim.json",
        {"status": status},
    )
    error = RuntimeError if status == "started" else ValueError
    message = "started and non-retryable" if status == "started" else "corrupt result"

    with pytest.raises(error, match=message):
        _maybe_run_contextual_risk_final_test(
            SimpleNamespace(run_final_test=True, output_dir=str(tmp_path)),
            {"winner": winner, "candidates": [winner]},
            lambda selected: pytest.fail("crash-window claim must not rerun test"),
        )


def test_completed_final_test_claim_rejects_result_hash_tampering(tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    args = SimpleNamespace(run_final_test=True, output_dir=str(tmp_path))
    selection = {"winner": winner, "candidates": [winner]}
    _maybe_run_contextual_risk_final_test(
        args, selection, lambda selected: {"test": 1}
    )
    (tmp_path / "final_test.json").write_text('{"test": 2}\n')

    with pytest.raises(ValueError, match="corrupt result"):
        _maybe_run_contextual_risk_final_test(args, selection, lambda selected: None)


def test_final_test_is_deferred_when_three_epoch_continuation_is_eligible(
    tmp_path, monkeypatch
):
    candidates = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        checkpoint = tmp_path / f"pytorch_model.bin.{epoch}"
        optimizer = tmp_path / f"pytorch_opt.bin.{epoch}"
        checkpoint.write_bytes(f"model-{epoch}".encode())
        optimizer.write_bytes(f"optimizer-{epoch}".encode())
        evidence, _ = _complete_evidence(rho=rho)
        candidates.append({
            "epoch": epoch,
            "path": str(checkpoint.resolve()),
            "model_sha256": file_sha256(checkpoint),
            "optimizer_path": str(optimizer.resolve()),
            "optimizer_sha256": file_sha256(optimizer),
            "valid": True,
            "finite": True,
            "runtime_abort": False,
            "rho": [rho, rho],
            "auroc": [0.70, 0.70],
            "aurc": [0.25, 0.25],
            "evidence": evidence,
            "paired_digest": _paired_digest(),
            "paired_invariance": {
                "passed": True,
                "checks": {
                    "raw_wti": True,
                    "text_means": True,
                    "video_means": True,
                    "a1_failure_risk": True,
                    "a1_vacuity": True,
                    "retrieval": True,
                },
            },
            "non_a3_mismatches": [],
            "metrics": {
                **_validation(rho=rho)["contextual_risk"],
                "paired_invariance": {"passed": True},
                "non_a3_mismatches": [],
            },
        })
    calls = []
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        raising=False,
    )

    result = _maybe_run_contextual_risk_final_test(
        SimpleNamespace(run_final_test=True, output_dir=str(tmp_path)),
        {"winner": candidates[2], "candidates": candidates},
        lambda selected: calls.append(selected),
    )

    assert result is None
    assert calls == []


def test_invalid_epoch_two_does_not_defer_terminal_winner(tmp_path, monkeypatch):
    candidates = []
    for epoch, rho in enumerate((0.20, 0.21, 0.99)):
        checkpoint = tmp_path / f"pytorch_model.bin.{epoch}"
        optimizer = tmp_path / f"pytorch_opt.bin.{epoch}"
        checkpoint.write_bytes(f"model-{epoch}".encode())
        optimizer.write_bytes(f"optimizer-{epoch}".encode())
        candidate = {
            "epoch": epoch,
            "path": str(checkpoint.resolve()),
            "model_sha256": file_sha256(checkpoint),
            "optimizer_path": str(optimizer.resolve()),
            "optimizer_sha256": file_sha256(optimizer),
            "valid": epoch != 2,
            "finite": True,
            "runtime_abort": False,
            "rho": [rho, rho],
            "auroc": [0.70, 0.70],
            "aurc": [0.25, 0.25],
            "evidence": {"risk": [0.1, 0.2]},
            "paired_digest": _paired_digest(),
            "paired_invariance": {
                "passed": True,
                "checks": {
                    "raw_wti": True, "a1_failure_risk": True,
                    "a1_vacuity": True, "retrieval": True,
                },
            },
            "non_a3_mismatches": [],
            "metrics": {
                **_validation(rho=rho)["contextual_risk"],
                "paired_invariance": {"passed": True},
                "non_a3_mismatches": [],
            },
        }
        candidates.append(candidate)
    calls = []
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        raising=False,
    )

    assert _contextual_risk_continuation_eligible(
        candidates, output_dir=tmp_path
    ) is False
    result = _maybe_run_contextual_risk_final_test(
        SimpleNamespace(run_final_test=True, output_dir=str(tmp_path)),
        {"winner": candidates[1], "candidates": candidates},
        lambda selected: calls.append(selected) or {"test": 1},
    )
    assert result == {"test": 1}
    assert calls == [candidates[1]]


def test_a3_lifecycle_owner_is_global_rank_zero_not_local_rank_zero():
    assert _contextual_risk_lifecycle_owner(
        SimpleNamespace(rank=0, local_rank=0)
    ) is True
    assert _contextual_risk_lifecycle_owner(
        SimpleNamespace(rank=4, local_rank=0)
    ) is False
    assert _contextual_risk_lifecycle_owner(
        SimpleNamespace(rank=1, local_rank=1)
    ) is False


def test_a3_autocast_is_disabled_without_changing_a2_or_legacy():
    assert _a3_training_autocast_enabled(
        SimpleNamespace(contextual_risk_sidecar_only=True)
    ) is False
    assert _a3_training_autocast_enabled(
        SimpleNamespace(
            contextual_risk_sidecar_only=False,
            csd_logvar_sidecar_only=True,
        )
    ) is True
    assert _a3_training_autocast_enabled(SimpleNamespace()) is True


def test_a3_exact_fp32_disables_and_restores_tf32_without_changing_a2():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with _a3_exact_fp32_backend(
            SimpleNamespace(contextual_risk_sidecar_only=True)
        ):
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

        with _a3_exact_fp32_backend(
            SimpleNamespace(csd_logvar_sidecar_only=True)
        ):
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def _manifest_publish_args(*, rank, tmp_path, resume_model=""):
    """Namespace covering exactly the fields _publish_experiment_manifest reads."""
    return SimpleNamespace(
        rank=rank,
        local_rank=rank,
        contextual_risk_sidecar_only=True,
        resume_model=resume_model,
        output_dir=str(tmp_path),
    )


def _silence_manifest_logger(monkeypatch):
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args, **kwargs: None),
        raising=False,
    )


def test_publish_experiment_manifest_non_rank_zero_ignores_rank_zero_write(
    tmp_path, monkeypatch
):
    """Non-zero ranks must not race the fresh check against rank 0's write.

    Reproduces the 2026-08-12 third-lineage failure at main_task_retrieval.py
    where ranks 1--3 raised FileExistsError on the manifest rank 0 itself had
    just published (log
    6858631e06e34d85fd3ccd52f5b3110b4a565d8ce92a8c3893d6a801caf1cf7e).
    """
    _silence_manifest_logger(monkeypatch)
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps({"marker": "from-rank-0"}))
    args = _manifest_publish_args(rank=1, tmp_path=tmp_path)

    _publish_experiment_manifest(
        args,
        tracking_payload={"marker": "tracking"},
        manifest_path=manifest_path,
        batch_semantics={},
    )

    # rank != 0 must not touch either artifact.
    assert json.loads(manifest_path.read_text()) == {"marker": "from-rank-0"}
    assert not (tmp_path / "training_args.json").exists()


def test_publish_experiment_manifest_rank_zero_writes_when_absent(
    tmp_path, monkeypatch
):
    _silence_manifest_logger(monkeypatch)
    manifest_path = tmp_path / "experiment_manifest.json"
    args = _manifest_publish_args(rank=0, tmp_path=tmp_path)

    _publish_experiment_manifest(
        args,
        tracking_payload={"marker": "tracking"},
        manifest_path=manifest_path,
        batch_semantics={"world_size": 4},
    )

    assert json.loads(manifest_path.read_text()) == {"marker": "tracking"}
    persisted_args = json.loads((tmp_path / "training_args.json").read_text())
    assert persisted_args["rank"] == 0
    assert persisted_args["contextual_risk_sidecar_only"] is True


def test_publish_experiment_manifest_rank_zero_refuses_existing_fresh(
    tmp_path, monkeypatch
):
    _silence_manifest_logger(monkeypatch)
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps({"stale": True}))
    args = _manifest_publish_args(rank=0, tmp_path=tmp_path)

    with pytest.raises(FileExistsError, match="fresh A3 run refuses"):
        _publish_experiment_manifest(
            args,
            tracking_payload={"marker": "tracking"},
            manifest_path=manifest_path,
            batch_semantics={},
        )
    assert json.loads(manifest_path.read_text()) == {"stale": True}


def test_publish_experiment_manifest_resume_missing_raises_on_every_rank(
    tmp_path, monkeypatch
):
    _silence_manifest_logger(monkeypatch)
    manifest_path = tmp_path / "experiment_manifest.json"
    args = _manifest_publish_args(
        rank=1, tmp_path=tmp_path, resume_model="/tmp/prior/opt.bin"
    )

    with pytest.raises(FileNotFoundError, match="A3 continuation requires"):
        _publish_experiment_manifest(
            args,
            tracking_payload={"marker": "tracking"},
            manifest_path=manifest_path,
            batch_semantics={},
        )


def test_publish_experiment_manifest_resume_validates_on_every_rank(
    tmp_path, monkeypatch
):
    _silence_manifest_logger(monkeypatch)
    calls = []

    def _fake_validate(existing, requested):
        calls.append((existing, requested))

    monkeypatch.setattr(
        main_task_retrieval,
        "validate_contextual_risk_resume_manifest",
        _fake_validate,
    )
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps({"lineage": "prior"}))
    args = _manifest_publish_args(
        rank=1, tmp_path=tmp_path, resume_model="/tmp/prior/opt.bin"
    )

    _publish_experiment_manifest(
        args,
        tracking_payload={"marker": "tracking"},
        manifest_path=manifest_path,
        batch_semantics={},
    )

    assert calls == [({"lineage": "prior"}, {"marker": "tracking"})]
    # Non-rank-zero resume must not re-publish the manifest or write args.
    assert json.loads(manifest_path.read_text()) == {"lineage": "prior"}
    assert not (tmp_path / "training_args.json").exists()


def test_a3_save_refuses_checkpoint_overwrite(tmp_path, monkeypatch):
    existing = tmp_path / "pytorch_model.bin.0"
    existing.write_bytes(b"immutable")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None),
        raising=False,
    )

    with pytest.raises(FileExistsError, match="append-only"):
        save_model(
            0,
            SimpleNamespace(
                output_dir=str(tmp_path),
                csd_logvar_sidecar_only=False,
                contextual_risk_sidecar_only=True,
            ),
            model,
            optimizer,
            tr_loss=0.5,
        )
    assert existing.read_bytes() == b"immutable"
